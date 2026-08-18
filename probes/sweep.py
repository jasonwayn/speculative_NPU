"""NPU block-size sweep + power/energy.  B=1 = AR baseline."""
import sys, os, json, time, threading, subprocess, torch, torch.nn as nn, rebel
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
TGT="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden"
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
CTX_MAX=int(os.environ.get("CTX_MAX","2560")); NSAMP=int(os.environ.get("NSAMP","20"))
MAXNEW=int(os.environ.get("MAXNEW","2048")); DEV=int(os.environ.get("DEV","0"))
BLOCKS=[int(x) for x in os.environ.get("BLOCKS","1,4,8,16,32,64").split(",")]
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from optimum.rbln import RBLNQwen3ForCausalLM
from dflash.model import DFlashDraftModel, extract_context_feature
torch.set_num_threads(8)

class Power:
    def __init__(s,dev,hz=5): s.dev=dev; s.hz=hz; s.on=False; s.rows=[]
    def _loop(s):
        while s.on:
            try:
                j=json.loads(subprocess.run(["rbln-stat","--json"],capture_output=True,text=True,timeout=3).stdout)
                d=j["devices"][s.dev]
                s.rows.append((time.time(), float(str(d["card_power"]).replace("uW",""))/1e6,
                               float(str(d.get("temperature","0C")).replace("C",""))))
            except Exception: pass
            time.sleep(1.0/s.hz)
    def __enter__(s): s.on=True; s.th=threading.Thread(target=s._loop,daemon=True); s.th.start(); s.t0=time.time(); return s
    def __exit__(s,*a): s.on=False; s.th.join(timeout=3); s.t1=time.time()
    def stats(s,idle=0.0):
        w=round(s.t1-s.t0,1)
        if len(s.rows)<2: return dict(wall_s=w)
        E=Ed=0.0
        for i in range(1,len(s.rows)):
            dt=s.rows[i][0]-s.rows[i-1][0]; pm=0.5*(s.rows[i][1]+s.rows[i-1][1])
            E+=pm*dt; Ed+=max(pm-idle,0)*dt
        P=[r[1] for r in s.rows]
        return dict(wall_s=w,n_pwr=len(s.rows),P_mean=round(sum(P)/len(P),2),P_max=round(max(P),2),
                    E_J=round(E,1),E_dyn_J=round(Ed,1),T_max=round(max(r[2] for r in s.rows),1))

tok=AutoTokenizer.from_pretrained(SRC)
cpu=AutoModelForCausalLM.from_pretrained(SRC,dtype=torch.float32).eval()
embed=cpu.model.embed_tokens; lmw=cpu.lm_head
m=RBLNQwen3ForCausalLM.from_pretrained(TGT,export=False,rbln_device=DEV)
pdec=m.prefill_decoder; CHUNK=pdec.rbln_config.prefill_chunk_size
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=torch.float32).eval()
H=cfg.hidden_size; NT=len(draft.target_layer_ids); MASK=draft.mask_token_id
LID=draft.target_layer_ids; BT=torch.tensor([0],dtype=torch.int16)
class Wrap(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,n,t,p,mk): return s.d(position_ids=p,attention_mask=mk,noise_embedding=n,
                                        target_hidden=t,past_key_values=None,use_cache=False,is_causal=False)
WR=Wrap(draft).eval()
class LMH(nn.Module):
    def __init__(s,l): super().__init__(); s.l=l
    def forward(s,x): return s.l(x)
ds=[json.loads(l) for l in open("/home/work/npu_work/dflash_work/dflash/cache/gsm8k.jsonl")][:NSAMP]
STOP={tok.eos_token_id,151645}
with Power(DEV) as p0: time.sleep(6)
IDLE=p0.stats().get("P_mean",0.0); print("idle_W "+str(round(IDLE,2)),flush=True)

def build(B):
    rt=None
    if B>1:
        rt=rebel.compile_from_torch(WR,input_info=[("noise_emb",[1,B,H],"float32"),
            ("target_hidden",[1,CTX_MAX,NT*H],"float32"),("position_ids",[1,CTX_MAX+B],"int64"),
            ("attn_mask",[1,1,B,CTX_MAX+B],"float32")]).create_runtime(device=DEV)
    lrt=rebel.compile_from_torch(LMH(lmw).eval(),
        input_info=[("x",[1,max(B,1),H],"float32")]).create_runtime(device=DEV)
    return rt,lrt

def run(B,rt,lrt,warm=False):
    T={"draft":0.0,"verify":0.0,"lmh":0.0}
    NL=max(B,1)
    TH=torch.zeros(1,CTX_MAX,NT*H); PO=torch.zeros(1,CTX_MAX+B,dtype=torch.long)
    MK=torch.zeros(1,1,B,CTX_MAX+B) if B>1 else None
    def lmh(x):
        n=x.shape[1]
        if n<NL: x=torch.cat([x,torch.zeros(1,NL-n,H)],dim=1)
        s=time.time(); o=torch.as_tensor(lrt(x.contiguous().numpy())); T["lmh"]+=time.time()-s
        return torch.argmax(o[:,:n].float(),dim=-1)
    def tpre(seg,off):
        s=time.time(); L=seg.shape[1]; pad=1 if L%CHUNK==0 else 0
        if pad: seg=torch.cat([seg,seg[:,-1:]],dim=1)
        r=pdec.prefill_forward(seg,cache_position=torch.arange(off,off+seg.shape[1],dtype=torch.int32).unsqueeze(0),
            attention_mask=torch.ones(seg.shape[1],dtype=torch.int64),batch_idx=0,
            block_tables=BT,is_external_block_tables=False)
        T["verify"]+=time.time()-s; hs=r.hidden_states
        return tuple(h[:,:L] for h in hs) if pad else hs
    def twh(ids):
        s=time.time()
        if ids.shape[1]%CHUNK==0:
            i2=torch.cat([ids,ids[:,-1:]],dim=1); hs=m(input_ids=i2,attention_mask=torch.ones_like(i2)).hidden_states
            hs=tuple(h[:,:ids.shape[1]] for h in hs)
        else: hs=m(input_ids=ids,attention_mask=torch.ones_like(ids)).hidden_states
        T["verify"]+=time.time()-s; return hs
    acc=[]; ntok=0
    data=ds[:1] if warm else ds
    mx=64 if warm else MAXNEW
    for ex in data:
        ids=tok.apply_chat_template([{"role":"user","content":ex["turns"][0]}],add_generation_prompt=True,
                                    return_tensors="pt",enable_thinking=False)
        P=ids.shape[1]; hs=twh(ids); bonus=int(lmh(hs[-1][:,-1:].float())[0,0])
        th=extract_context_feature(hs,LID); ver=ids; cached=(P//CHUNK)*CHUNK
        if B>1:
            TH.zero_(); PO.zero_(); MK.zero_(); MK[:,:,:,:CTX_MAX]=float("-inf"); prev=CTX_MAX
        while ver.shape[1]<P+mx and th.shape[1]<=CTX_MAX:
            blk=torch.full((1,B),MASK,dtype=torch.long); blk[0,0]=bonus
            if B>1:
                L=th.shape[1]; pn=CTX_MAX-L
                TH[:,pn:]=th; PO[:,pn:]=torch.arange(L+B)
                if pn<prev: MK[:,:,:,pn:prev]=0.0
                prev=pn
                with torch.no_grad(): noi=embed(blk).detach()
                s=time.time(); h=torch.as_tensor(rt(noi.numpy(),TH.numpy(),PO.numpy(),MK.numpy())).float()
                T["draft"]+=time.time()-s
                blk[:,1:]=lmh(h[:,1-B:,:])
            vl=ver.shape[1]; cand=torch.cat([ver,blk],dim=1)
            hs2=tpre(cand[:,cached:],cached)
            post=lmh(hs2[-1][:,vl-cached:vl-cached+B].float())
            a=int((blk[:,1:]==post[:,:-1]).cumprod(dim=1).sum(dim=1)[0]) if B>1 else 0
            acc.append(a+1)
            th=torch.cat([th,extract_context_feature(hs2,LID)[:,vl-cached:vl-cached+a+1,:]],dim=1)
            ver=torch.cat([ver,blk[:,:a+1]],dim=1); bonus=int(post[0,a])
            cached=(ver.shape[1]//CHUNK)*CHUNK
            if any(int(t) in STOP for t in ver[0,P:]): break
        ntok+=ver.shape[1]-P
    return acc,ntok,T

res=[]
for B in BLOCKS:
    print("===== B="+str(B)+" =====",flush=True)
    t0=time.time(); rt,lrt=build(B); ct=round(time.time()-t0,1)
    run(B,rt,lrt,warm=True)                           # 워밍업 (버림)
    with Power(DEV) as pw: acc,ntok,T=run(B,rt,lrt)
    st=pw.stats(idle=IDLE); hist={}
    for a in acc: hist[a]=hist.get(a,0)+1
    r=dict(B=B,compile_s=ct,tau=round(sum(acc)/len(acc),3),cycles=len(acc),tokens=ntok,**st,
           draft_s=round(T["draft"],1),verify_s=round(T["verify"],1),lmh_s=round(T["lmh"],1))
    r["tok_s"]=round(ntok/st["wall_s"],2); r["J_per_tok"]=round(st["E_J"]/ntok,3)
    r["Jdyn_per_tok"]=round(st["E_dyn_J"]/ntok,3)
    r["hist_pct"]={k:round(v/len(acc)*100,1) for k,v in sorted(hist.items())}
    print("SWEEP "+json.dumps(r),flush=True); res.append(r)
print("SWEEP_ALL "+json.dumps(res),flush=True)

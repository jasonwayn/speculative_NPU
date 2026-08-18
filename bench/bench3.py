import sys, os, json, time, threading, subprocess, torch, torch.nn as nn, rebel, bisect
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
TGT=os.environ.get("TGTDIR","/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden")
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
B=int(os.environ.get("B","16")); NSAMP=int(os.environ.get("NSAMP","20"))
MAXNEW=int(os.environ.get("MAXNEW","2048")); DEV=int(os.environ.get("DEV","0"))
DSETS=os.environ.get("DSETS","gsm8k").split(",")
OFFSET=int(os.environ.get("OFFSET","0"))
BUCKETS=[int(x) for x in os.environ.get("BUCKETS","256,512,1024,2048,3072").split(",")]
PREC=os.environ.get("PREC","float32")
DT=getattr(torch,PREC); DTS=PREC
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from optimum.rbln import RBLNQwen3ForCausalLM
from dflash.model import DFlashDraftModel
def extract_context_feature(hs, ids, off=1):
    return torch.cat([hs[i+off] for i in ids], dim=-1)
torch.set_num_threads(int(os.environ.get("NTHREADS","8")))

class Power:
    def __init__(s,dev,hz=5): s.dev=dev; s.hz=hz; s.on=False; s.rows=[]
    def _loop(s):
        while s.on:
            try:
                j=json.loads(subprocess.run(["rbln-stat","--json"],capture_output=True,text=True,timeout=3).stdout)
                d=j["devices"][s.dev]
                s.rows.append((time.time(),float(str(d["card_power"]).replace("uW",""))/1e6,
                               float(str(d.get("temperature","0C")).replace("C","")),
                               str(d.get("pstate","?"))))
            except Exception: pass
            time.sleep(1.0/s.hz)
    def __enter__(s):
        s.on=True; s.th=threading.Thread(target=s._loop,daemon=True); s.th.start(); s.t0=time.time(); return s
    def __exit__(s,*a): s.on=False; s.th.join(timeout=3); s.t1=time.time()
    def stats(s,idle=0.0):
        w=round(s.t1-s.t0,1)
        if len(s.rows)<2: return dict(wall_s=w)
        E=0.0; Ed=0.0
        for i in range(1,len(s.rows)):
            dt=s.rows[i][0]-s.rows[i-1][0]; pm=0.5*(s.rows[i][1]+s.rows[i-1][1])
            E+=pm*dt; Ed+=max(pm-idle,0)*dt
        P=[r[1] for r in s.rows]; TT=[r[2] for r in s.rows]
        n=max(1,len(TT)//10)
        return dict(wall_s=w,P_mean=round(sum(P)/len(P),2),E_J=round(E,1),E_dyn_J=round(Ed,1),
                    T_start=round(sum(TT[:n])/n,1), T_end=round(sum(TT[-n:])/n,1),
                    T_max=round(max(TT),1), T_mean=round(sum(TT)/len(TT),1),
                    perf_states="/".join(sorted(set(r[3] for r in s.rows))))

# --- 로딩 직렬화 (동시 로드 시 호스트 OOM) ---
import fcntl
_LOCK=open("/home/work/npu_work/dflash_work/.load.lock","a+")
if os.environ.get("SERIAL_LOAD","1")=="1":
    fcntl.flock(_LOCK, fcntl.LOCK_EX)
    print("load-lock acquired",flush=True)

tok=AutoTokenizer.from_pretrained(SRC)
# 전체 모델 대신 임베딩 행렬만 로드 (embed_tokens 와 lm_head 는 tied)
from safetensors import safe_open
import glob as _glob
_W=None
for _f in sorted(_glob.glob(SRC+"/*.safetensors")):
    with safe_open(_f,framework="pt") as _h:
        for _k in _h.keys():
            if _k.endswith("model.embed_tokens.weight") or _k=="embed_tokens.weight":
                _W=_h.get_tensor(_k).to(DT); break      # 목표 dtype 으로 직접 (fp32 중간본 제거)
    if _W is not None: break
if _W is None: raise SystemExit("embed_tokens.weight not found")
embed=nn.Embedding.from_pretrained(_W,freeze=True)          # 가중치 공유 (복사 없음)
lmw=nn.Linear(_W.shape[1],_W.shape[0],bias=False).to(DT)
with torch.no_grad(): lmw.weight=nn.Parameter(_W,requires_grad=False)   # 동일 텐서 재사용
lmw=lmw.eval()
print("embed/lm_head loaded "+str(tuple(_W.shape)),flush=True)
m=RBLNQwen3ForCausalLM.from_pretrained(TGT,export=False,rbln_device=DEV)
pdec=m.prefill_decoder; CHUNK=pdec.rbln_config.prefill_chunk_size
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=DT).eval()
H=cfg.hidden_size; NT=len(draft.target_layer_ids); MASK=draft.mask_token_id
LID=draft.target_layer_ids; BT=torch.tensor([0],dtype=torch.int16)

class Wrap(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,n,t,p,mk):
        return s.d(position_ids=p,attention_mask=mk,noise_embedding=n,
                   target_hidden=t,past_key_values=None,use_cache=False,is_causal=False)
WR=Wrap(draft).eval()
class LMH(nn.Module):
    """argmax 까지 온디바이스 — 로짓(9.7MB) 대신 토큰 id(64B)만 반환"""
    def __init__(s,l): super().__init__(); s.l=l
    def forward(s,x): return s.l(x)

CACHE=os.environ.get("RBLN_CACHE","/home/work/npu_work/dflash_work/rbln_cache")
os.makedirs(CACHE,exist_ok=True)
def get_cm(name, build):
    """컴파일 결과를 디스크에 캐시 — 워커마다 재컴파일하지 않도록"""
    path=os.path.join(CACHE,name+".rbln")
    if os.path.exists(path):
        return rebel.RBLNCompiledModel(path)
    cm=build(); cm.save(path); return cm
print("preparing buckets "+str(BUCKETS),flush=True); t0=time.time()
BK={}
for C in BUCKETS:
    nm=f"draft_B{B}_C{C}_{DTS}"
    cm=get_cm(nm, lambda C=C: rebel.compile_from_torch(WR,input_info=[("noise_emb",[1,B,H],DTS),
        ("target_hidden",[1,C,NT*H],DTS),("position_ids",[1,C+B],"int64"),
        ("attn_mask",[1,1,B,C+B],DTS)]))
    BK[C]=dict(rt=cm.create_runtime(device=DEV), TH=torch.zeros(1,C,NT*H,dtype=DT),
               PO=torch.zeros(1,C+B,dtype=torch.long), MK=torch.zeros(1,1,B,C+B,dtype=DT), filled=0)
lcm=get_cm(f"lmhead_B{B}_{DTS}", lambda: rebel.compile_from_torch(LMH(lmw).eval(),
    input_info=[("x",[1,B,H],DTS)]))
lrt=lcm.create_runtime(device=DEV)
print("compiled "+str(round(time.time()-t0,1))+"s",flush=True)
if os.environ.get("SERIAL_LOAD","1")=="1":
    fcntl.flock(_LOCK, fcntl.LOCK_UN); print("load-lock released",flush=True)
BKEYS=sorted(BK); USE={c:0 for c in BKEYS}
STOP={tok.eos_token_id,151645}

def run(dsname,warm=False):
    T={"draft":0.0,"verify":0.0,"lmh":0.0}
    def lmh(x):
        n=x.shape[1]
        if n<B: x=torch.cat([x,torch.zeros(1,B-n,H,dtype=x.dtype)],dim=1)
        s=time.time(); o=torch.as_tensor(lrt(x.contiguous().numpy())); T["lmh"]+=time.time()-s
        return torch.argmax(o[:,:n].float(),dim=-1)
    def tpre(seg,off):
        s=time.time(); L=seg.shape[1]; pad=1 if L%CHUNK==0 else 0
        if pad: seg=torch.cat([seg,seg[:,-1:]],dim=1)
        r=pdec.prefill_forward(seg,
            cache_position=torch.arange(off,off+seg.shape[1],dtype=torch.int32).unsqueeze(0),
            attention_mask=torch.ones(seg.shape[1],dtype=torch.int64),batch_idx=0,
            block_tables=BT,is_external_block_tables=False)
        T["verify"]+=time.time()-s; hs=r.hidden_states
        return tuple(h[:,:L] for h in hs) if pad else hs
    def twh(ids):
        s=time.time()
        if ids.shape[1]%CHUNK==0:
            i2=torch.cat([ids,ids[:,-1:]],dim=1)
            hs=m(input_ids=i2,attention_mask=torch.ones_like(i2)).hidden_states
            hs=tuple(h[:,:ids.shape[1]] for h in hs)
        else: hs=m(input_ids=ids,attention_mask=torch.ones_like(ids)).hidden_states
        T["verify"]+=time.time()-s; return hs
    path="/home/work/npu_work/dflash_work/dflash/cache/"+dsname+".jsonl"
    ds=[json.loads(l) for l in open(path)]
    ds=ds[:1] if warm else ds[OFFSET:OFFSET+NSAMP]
    mx=64 if warm else MAXNEW
    acc=[]; ntok=0; skipped=0
    for ex in ds:
        ids=tok.apply_chat_template([{"role":"user","content":ex["turns"][0]}],
                                    add_generation_prompt=True,return_tensors="pt",enable_thinking=False)
        P=ids.shape[1]
        if P>BKEYS[-1]: skipped+=1; continue
        hs=twh(ids); bonus=int(lmh(hs[-1][:,-1:].to(DT))[0,0])
        CAP=BKEYS[-1]+B+8
        THB=torch.zeros(1,CAP,NT*H,dtype=DT)                 # 사전할당
        VRB=torch.zeros(1,CAP,dtype=torch.long)
        _t0=extract_context_feature(hs,LID).to(DT); TL=_t0.shape[1]
        THB[:,:TL]=_t0; VRB[:,:P]=ids; VL=P
        cached=(P//CHUNK)*CHUNK
        cur=None; prev=None
        while VL<P+mx:
            L=TL; th=THB[:,:TL]; ver=VRB[:,:VL]
            if L>BKEYS[-1]: break
            k=BKEYS[bisect.bisect_left(BKEYS,L)]
            b=BK[k]
            if k!=cur:
                b["TH"].zero_(); b["PO"].zero_(); b["MK"].zero_()
                b["MK"][:,:,:,:k]=torch.finfo(DT).min      # ctx 구간 전부 닫고 시작
                b["filled"]=0; cur=k
            # 우측 정렬: ctx 는 [0:L], 패딩은 [L:k], 블록은 항상 뒤 B칸
            # → TH 는 새로 늘어난 부분만 쓰면 됨 (append-only)
            if b["filled"]<L:
                b["TH"][:,b["filled"]:L]=th[:,b["filled"]:L]
                b["PO"][:,b["filled"]:L]=torch.arange(b["filled"],L)
                b["MK"][:,:,:,b["filled"]:L]=0.0            # 새로 유효해진 구간 열기
                b["filled"]=L
            b["PO"][:,k:]=torch.arange(L,L+B)               # 블록 위치는 매번 갱신
            if not warm: USE[k]+=1
            blk=torch.full((1,B),MASK,dtype=torch.long); blk[0,0]=bonus
            with torch.no_grad(): noi=embed(blk).detach()
            s=time.time()
            h=torch.as_tensor(b["rt"](noi.numpy(),b["TH"].numpy(),b["PO"].numpy(),b["MK"].numpy()))
            T["draft"]+=time.time()-s
            blk[:,1:]=lmh(h[:,1-B:,:].to(DT))
            vl=VL
            seg=torch.cat([VRB[:,cached:VL],blk],dim=1)      # 꼬리만 구성 (전체 cand 미생성)
            hs2=tpre(seg,cached)
            post=lmh(hs2[-1][:,vl-cached:vl-cached+B].to(DT))
            a=int((blk[:,1:]==post[:,:-1]).cumprod(dim=1).sum(dim=1)[0])
            acc.append(a+1)
            nf=extract_context_feature(hs2,LID)[:,vl-cached:vl-cached+a+1,:].to(DT)
            THB[:,TL:TL+nf.shape[1]]=nf; TL+=nf.shape[1]
            VRB[:,VL:VL+a+1]=blk[:,:a+1]; VL+=a+1
            bonus=int(post[0,a]); cached=(VL//CHUNK)*CHUNK
            if any(int(t) in STOP for t in VRB[0,P:VL]): break
            if TL>=CAP-B-2: break
        ntok+=VL-P
    return acc,ntok,T,skipped

with Power(DEV) as p0: time.sleep(6)
IDLE=p0.stats().get("P_mean",0.0); print("idle_W "+str(round(IDLE,2)),flush=True)
res=[]
for dn in DSETS:
    run(dn,warm=True)
    for kk in USE: USE[kk]=0
    with Power(DEV) as pw: acc,ntok,T,sk=run(dn)
    st=pw.stats(idle=IDLE)
    r=dict(dataset=dn,B=B,prec=PREC,chunk=CHUNK,tau=round(sum(acc)/len(acc),3),cycles=len(acc),tokens=ntok,skipped=sk,**st,
           draft_s=round(T["draft"],1),verify_s=round(T["verify"],1),lmh_s=round(T["lmh"],1),
           tok_s=round(ntok/st["wall_s"],2),J_per_tok=round(st["E_J"]/ntok,3),
           Jdyn_per_tok=round(st["E_dyn_J"]/ntok,3),
           draft_ms=round(T["draft"]/len(acc)*1000,1),
           verify_ms=round(T["verify"]/len(acc)*1000,1),
           bucket_use={k:v for k,v in USE.items() if v})
    print("BENCH2 "+json.dumps(r),flush=True); res.append(r)
print("BENCH2_ALL "+json.dumps(res),flush=True)

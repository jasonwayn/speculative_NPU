"""GPU 벤치 — NPU 와 동일 조건 (동일 프롬프트/샘플수/MAXNEW/B). 전력·에너지 측정."""
import sys, os, json, time, threading, subprocess, torch
sys.path.insert(0,os.path.expanduser("~/dflash_bench/dflash"))
TGT="Qwen/Qwen3-4B"; DRF="z-lab/Qwen3-4B-DFlash-b16"
NSAMP=int(os.environ.get("NSAMP","20")); MAXNEW=int(os.environ.get("MAXNEW","2048"))
DSETS=os.environ.get("DSETS","gsm8k").split(",")
MODE=os.environ.get("MODE","dflash")          # dflash | ar
DTYPE=os.environ.get("DTYPE","bfloat16")
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dflash.model import DFlashDraftModel, dflash_generate
import dflash.model as _dm
if not torch.cuda.is_available(): raise SystemExit("no cuda")

class Power:
    def __init__(s,hz=5): s.hz=hz; s.on=False; s.rows=[]
    def _loop(s):
        while s.on:
            try:
                o=subprocess.run(["nvidia-smi","--query-gpu=power.draw,utilization.gpu,temperature.gpu,clocks.sm",
                                  "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3).stdout.strip()
                p,u,t,c=[float(x) for x in o.split(",")]
                s.rows.append((time.time(),p,u,t,c))
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
        P=[r[1] for r in s.rows]; U=[r[2] for r in s.rows]
        TT=[r[3] for r in s.rows]; C=[r[4] for r in s.rows]
        n=max(1,len(TT)//10)
        return dict(wall_s=w,P_mean=round(sum(P)/len(P),2),P_max=round(max(P),2),
                    util_mean=round(sum(U)/len(U),1),E_J=round(E,1),E_dyn_J=round(Ed,1),
                    T_start=round(sum(TT[:n])/n,1),T_end=round(sum(TT[-n:])/n,1),
                    T_max=round(max(TT),1),T_mean=round(sum(TT)/len(TT),1),
                    clk_mean=round(sum(C)/len(C),0),clk_min=round(min(C),0))

DT=getattr(torch,DTYPE)
tok=AutoTokenizer.from_pretrained(TGT)
target=AutoModelForCausalLM.from_pretrained(TGT,dtype=DT).cuda().eval()
draft=None
if MODE=="dflash":
    cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True)
    draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=DT).cuda().eval()
STOP=[tok.eos_token_id,151645]

def load(dn):
    p=os.path.expanduser("~/dflash_bench/cache/"+dn+".jsonl")
    return [json.loads(l) for l in open(p)][:NSAMP]

def once(ids):
    with torch.inference_mode():
        if MODE=="dflash":
            r=dflash_generate(draft,target=target,input_ids=ids,max_new_tokens=MAXNEW,
                              stop_token_ids=STOP,temperature=0.0,return_stats=True)
            return r.acceptance_lengths, r.num_output_tokens
        o=target.generate(ids,max_new_tokens=MAXNEW,do_sample=False,
                          eos_token_id=STOP,pad_token_id=tok.eos_token_id)
        return [], o.shape[1]-ids.shape[1]

torch.cuda.synchronize()
with Power() as p0: time.sleep(8)
IDLE=p0.stats().get("P_mean",0.0); print("idle_W "+str(round(IDLE,2)),flush=True)

res=[]
for dn in DSETS:
    ds=load(dn)
    w=tok.apply_chat_template([{"role":"user","content":ds[0]["turns"][0]}],add_generation_prompt=True,
                              return_tensors="pt",enable_thinking=False).cuda()
    with torch.inference_mode():
        if MODE=="dflash": dflash_generate(draft,target=target,input_ids=w,max_new_tokens=32,
                                           stop_token_ids=STOP,temperature=0.0)
        else: target.generate(w,max_new_tokens=32,do_sample=False)
    torch.cuda.synchronize()
    allacc=[]; ntok=0
    with Power() as pw:
        for ex in ds:
            ids=tok.apply_chat_template([{"role":"user","content":ex["turns"][0]}],add_generation_prompt=True,
                                        return_tensors="pt",enable_thinking=False).cuda()
            a,n=once(ids); allacc+=a; ntok+=n
        torch.cuda.synchronize()
    st=pw.stats(idle=IDLE)
    r=dict(dataset=dn,mode=MODE,dtype=DTYPE,samples=len(ds),tau=round(sum(allacc)/len(allacc),3) if allacc else None,
           cycles=len(allacc),tokens=ntok,**st,
           tok_s=round(ntok/st["wall_s"],2),J_per_tok=round(st["E_J"]/ntok,3),
           Jdyn_per_tok=round(st["E_dyn_J"]/ntok,3))
    print("GPU2 "+json.dumps(r),flush=True); res.append(r)
print("GPU2_ALL "+json.dumps(res),flush=True)

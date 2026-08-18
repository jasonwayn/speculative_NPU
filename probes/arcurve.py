"""NPU AR 배치 곡선 — 실제 GSM8K 프롬프트, 전력 포함."""
import torch, time, json, threading, subprocess
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer
M="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-c64-bs8"
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
tok=AutoTokenizer.from_pretrained(SRC); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
ds=[json.loads(l) for l in open("/home/work/npu_work/dflash_work/dflash/cache/gsm8k.jsonl")][:16]
prompts=[tok.apply_chat_template([{"role":"user","content":e["turns"][0]}],
         add_generation_prompt=True,tokenize=False,enable_thinking=False) for e in ds]
class P:
    def __init__(s,d=0): s.d=d; s.on=False; s.r=[]
    def _l(s):
        while s.on:
            try:
                j=json.loads(subprocess.run(["rbln-stat","--json"],capture_output=True,text=True,timeout=3).stdout)
                s.r.append(float(str(j["devices"][s.d]["card_power"]).replace("uW",""))/1e6)
            except Exception: pass
            time.sleep(0.2)
    def __enter__(s): s.on=True; s.t=threading.Thread(target=s._l,daemon=True); s.t.start(); s.t0=time.time(); return s
    def __exit__(s,*a): s.on=False; s.t.join(timeout=2); s.t1=time.time()
    def st(s): return (round(sum(s.r)/max(len(s.r),1),1), round(s.t1-s.t0,2))
M1="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64"
M4="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-b4"
NEW=128
print(f"{'batch':>5} {'sec':>7} {'tok':>6} {'tok/s':>8} {'per-seq':>8} {'W':>6} {'J/tok':>7}",flush=True)
res=[]
for bs,path,dev in [(1,M1,0),(4,M4,1),(8,M,2),(2,M,3)]:
    try:
        m=RBLNQwen3ForCausalLM.from_pretrained(path,export=False,rbln_device=dev)
    except Exception as e:
        print(f"{bs:5d}  load FAIL {str(e)[:70]}",flush=True); continue
    enc=tok(prompts[:bs],return_tensors="pt",padding=True)
    try:
        m.generate(**enc,max_new_tokens=8,do_sample=False)
    except Exception as e:
        print(f"{bs:5d}  gen FAIL {str(e)[:70]}",flush=True); del m; continue
    with P(dev) as pw:
        o=m.generate(**enc,max_new_tokens=NEW,do_sample=False)
    w,dt=pw.st(); n=bs*NEW
    r=dict(batch=bs,sec=dt,tokens=n,tok_s=round(n/dt,2),per_seq=round(NEW/dt,2),
           W=w,J_per_tok=round(w*dt/n,4))
    print(f"{bs:5d} {dt:7.2f} {n:6d} {n/dt:8.2f} {NEW/dt:8.2f} {w:6.1f} {w*dt/n:7.4f}",flush=True)
    res.append(r); del m
print("ARCURVE "+json.dumps(res),flush=True)

"""디코드 경로 배치 amortize 측정: b1 모델(배치1) vs b4 모델(배치4)."""
import torch, time
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("/home/work/npu_work/eagle_test/Qwen3-4B")
tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
P="Explain in detail why the sky appears blue during the day."
NEW=64
def run(path,bs,dev):
    m=RBLNQwen3ForCausalLM.from_pretrained(path,export=False,rbln_device=dev)
    enc=tok([P]*bs,return_tensors="pt",padding=True)
    m.generate(**enc,max_new_tokens=8,do_sample=False)
    t0=time.perf_counter(); m.generate(**enc,max_new_tokens=NEW,do_sample=False)
    dt=time.perf_counter()-t0; n=bs*NEW
    print(f"  batch {bs}: {dt:6.2f}s  총 {n} tok  {n/dt:7.2f} tok/s  "
          f"시퀀스당 {NEW/dt:6.2f} tok/s",flush=True)
    del m; return n/dt
print("=== 디코드 배치 확장 ===",flush=True)
a=run("/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64",1,0)
b=run("/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-b4",4,1)
print(f"\n배치4 / 배치1 처리량 = {b/a:.2f}배  (amortize 되면 4에 가까움)",flush=True)
print("GENBATCH_DONE",flush=True)

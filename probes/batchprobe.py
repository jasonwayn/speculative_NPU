"""배치 4 모델이 가중치 읽기를 amortize 하는가?
   (4,N) 1회 시간  vs  (1,N) 4회 시간 을 비교."""
import torch, time, json, traceback, inspect
from optimum.rbln import RBLNQwen3ForCausalLM
B1="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64"
B4="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-b4"

def timeit(fn,n=12,warm=3):
    for _ in range(warm): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    ts.sort(); return ts[len(ts)//2]*1e3

print("=== batch=1 모델 ===",flush=True)
m1=RBLNQwen3ForCausalLM.from_pretrained(B1,export=False,rbln_device=0)
print("  cfg batch:",m1.rbln_config.batch_size,flush=True)
for N in [17,48,63]:
    x=torch.randint(1000,5000,(1,N)); am=torch.ones(1,N,dtype=torch.long)
    t=timeit(lambda: m1(input_ids=x,attention_mask=am))
    print(f"  (1,{N:2d}) 1회 = {t:7.2f} ms   →  4회분 {4*t:7.2f} ms",flush=True)
del m1

print("\n=== batch=4 모델 ===",flush=True)
m4=RBLNQwen3ForCausalLM.from_pretrained(B4,export=False,rbln_device=1)
print("  cfg batch:",m4.rbln_config.batch_size,flush=True)
for N in [17,48,63]:
    for bs in [1,4]:
        x=torch.randint(1000,5000,(bs,N)); am=torch.ones(bs,N,dtype=torch.long)
        try:
            t=timeit(lambda: m4(input_ids=x,attention_mask=am))
            print(f"  ({bs},{N:2d}) = {t:7.2f} ms",flush=True)
        except Exception as e:
            print(f"  ({bs},{N:2d}) FAIL {type(e).__name__}: {str(e)[:100]}",flush=True)
print("\nBATCHPROBE_DONE",flush=True)

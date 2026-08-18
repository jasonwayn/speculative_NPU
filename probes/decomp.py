"""고정비 분해: dispatch vs 입력전송 vs 가중치 스트리밍."""
import torch, torch.nn as nn, rebel, time, json, os
DEV=int(os.environ.get("DEV","0")); DT="float16"; TD=torch.float16
N=int(os.environ.get("ITERS","200"))

def bench(mod, info, args, tag, note=""):
    rt=rebel.compile_from_torch(mod.eval(), input_info=info).create_runtime(device=DEV)
    a=[x.numpy() for x in args]
    for _ in range(10): rt(*a)                      # 워밍업
    ts=[]
    for _ in range(N):
        t=time.perf_counter(); rt(*a); ts.append(time.perf_counter()-t)
    ts.sort(); med=ts[len(ts)//2]*1e3; p10=ts[len(ts)//10]*1e3
    print(f"  {tag:34s} median={med:7.3f} ms  p10={p10:7.3f} ms   {note}",flush=True)
    return med

class Scale(nn.Module):
    def forward(s,x): return x*2.0
class Lin(nn.Module):
    def __init__(s,i,o): super().__init__(); s.l=nn.Linear(i,o,bias=False)
    def forward(s,x): return s.l(x)

print("=== A. dispatch 바닥 (가중치·연산 없음, 입력 최소) ===",flush=True)
t_disp=bench(Scale(), [("x",[1,1,64],DT)], [torch.randn(1,1,64,dtype=TD)], "scale, 128 B in")

print("\n=== B. 입력 전송 (연산 없음, 입력 크기만 변화) ===",flush=True)
prev=None
for L in [128, 512, 2048]:
    nbytes=1*L*12800*2
    m=bench(Scale(), [("x",[1,L,12800],DT)], [torch.randn(1,L,12800,dtype=TD)],
            f"scale, {nbytes/1e6:.0f} MB in", f"({nbytes/1e6:.0f} MB)")
    if prev:
        dB=(nbytes-prev[1]); dt=(m-prev[0])/1e3
        print(f"      -> 증분 대역폭 {dB/1e9/dt:.1f} GB/s",flush=True)
    prev=(m,nbytes)

print("\n=== C. 가중치 스트리밍 (입력 최소, 가중치만 변화) ===",flush=True)
prev=None
for (i,o) in [(2560,4096),(2560,32768),(2560,151936)]:
    w=i*o*2
    m=bench(Lin(i,o), [("x",[1,1,i],DT)], [torch.randn(1,1,i,dtype=TD)],
            f"Linear({i},{o})", f"({w/1e6:.0f} MB weights)")
    if prev:
        dW=(w-prev[1]); dt=(m-prev[0])/1e3
        print(f"      -> 증분 대역폭 {dW/1e9/dt:.1f} GB/s",flush=True)
    prev=(m,w)
print("\nDECOMP_DONE",flush=True)

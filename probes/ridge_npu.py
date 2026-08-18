"""NPU: forward 1회 시간 vs 처리 토큰 수 → ridge point"""
import torch, time, json
from optimum.rbln import RBLNQwen3ForCausalLM
M="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64"
m=RBLNQwen3ForCausalLM.from_pretrained(M,export=False,rbln_device=0)
CH=m.prefill_decoder.rbln_config.prefill_chunk_size
print("chunk =",CH,flush=True)
for _ in range(3):
    m(input_ids=torch.randint(1000,5000,(1,63)),attention_mask=torch.ones(1,63,dtype=torch.long))
print(f"{'tokens':>7} {'ms':>9} {'ms/token':>10}",flush=True)
rows=[]
for N in [1,2,4,8,16,32,48,63,96,127,192,255,384,511]:
    if N % CH == 0: continue
    x=torch.randint(1000,5000,(1,N)); am=torch.ones(1,N,dtype=torch.long)
    for _ in range(3): m(input_ids=x,attention_mask=am)
    ts=[]
    for _ in range(15):
        t0=time.perf_counter(); m(input_ids=x,attention_mask=am); ts.append(time.perf_counter()-t0)
    ts.sort(); md=ts[len(ts)//2]*1e3
    rows.append((N,md)); print(f"{N:7d} {md:9.3f} {md/N:10.4f}",flush=True)
print("RIDGE_NPU "+json.dumps(rows),flush=True)

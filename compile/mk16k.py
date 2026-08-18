"""3단계 kill criterion: 긴 컨텍스트 타깃이 ATOM+ 에서 컴파일/적재되는가.
KV 추정: 36층 x 2 x 8헤드 x N x 128 x 2B  =>  16K 에서 2.42 GB (모델 ~10GB, 카드 15.7GB)"""
import os, time, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
for N in [16384, 8192]:
    out=f"/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-{N//1024}k"
    if os.path.exists(out):
        print(f"SKIP {N} exists",flush=True); continue
    kv=36*2*8*N*128*2/1e9
    print(f"=== try max_seq_len={N}  (KV est {kv:.2f} GB)",flush=True)
    t=time.time()
    try:
        m=RBLNQwen3ForCausalLM.from_pretrained(SRC,export=True,
            rbln_batch_size=1, rbln_max_seq_len=N, rbln_prefill_chunk_size=64,
            rbln_output_hidden_states=True)
        m.save_pretrained(out)
        print(f"OK {N} {time.time()-t:.1f}s -> {out}",flush=True)
        break
    except Exception as e:
        print(f"FAIL {N} {time.time()-t:.1f}s {type(e).__name__}: {str(e)[:300]}",flush=True)
        traceback.print_exc()
print("MK16K_DONE",flush=True)

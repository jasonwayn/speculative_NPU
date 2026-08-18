"""양자화 + hidden_states 동시 지원 컴파일. h-c64 기준선과 동일 조건(4096/chunk64/bs1)."""
import os, time, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
for W in ["int4","int8"]:
    out=f"/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-{W}"
    if os.path.exists(out): print(f"SKIP {W}",flush=True); continue
    t=time.time()
    try:
        m=RBLNQwen3ForCausalLM.from_pretrained(SRC,export=True,
            rbln_batch_size=1, rbln_max_seq_len=4096, rbln_prefill_chunk_size=64,
            rbln_output_hidden_states=True,
            rbln_quantization={"format":"rbln","weights":W,"activations":"fp16"})
        m.save_pretrained(out)
        print(f"OK {W} {time.time()-t:.1f}s",flush=True)
    except Exception as e:
        print(f"FAIL {W} {time.time()-t:.1f}s {type(e).__name__}: {str(e)[:300]}",flush=True)
        traceback.print_exc()
print("MKQ2_DONE",flush=True)

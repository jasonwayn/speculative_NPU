"""타깃 int4 양자화 컴파일 가능성 확인."""
import os, time, traceback, torch
from optimum.rbln import RBLNQwen3ForCausalLM
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
for W,A in [("int4","fp16"),("int8","fp16")]:
    out=f"/home/work/npu_work/dflash_work/rbln-Qwen3-4B-{W}"
    if os.path.exists(out): print(f"SKIP {W} exists",flush=True); continue
    t=time.time()
    try:
        m=RBLNQwen3ForCausalLM.from_pretrained(SRC,export=True,
            rbln_batch_size=1, rbln_max_seq_len=2048, rbln_prefill_chunk_size=64,
            rbln_quantization={"format":"rbln","weights":W,"activations":A})
        m.save_pretrained(out)
        print(f"OK {W}/{A} {time.time()-t:.1f}s -> {out}",flush=True)
    except Exception as e:
        print(f"FAIL {W}/{A} {time.time()-t:.1f}s {type(e).__name__}: {str(e)[:400]}",flush=True)
        traceback.print_exc()
print("MKQ_DONE",flush=True)

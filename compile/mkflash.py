"""플래시 어텐션 타깃 컴파일. eager 대비 verify 비용 비교용.

제약: max_seq_len 은 kvcache_partition_len 의 배수이자 2배 이상,
      kvcache_block_size == kvcache_partition_len.
"""
import os, time, traceback
from optimum.rbln import RBLNQwen3ForCausalLM

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
OUT = "/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-16k-flash"
if os.path.exists(OUT):
    print("SKIP exists", OUT, flush=True)
    raise SystemExit
t = time.time()
try:
    m = RBLNQwen3ForCausalLM.from_pretrained(
        SRC, export=True,
        rbln_batch_size=1,
        rbln_max_seq_len=16384,
        rbln_prefill_chunk_size=64,
        rbln_output_hidden_states=True,
        rbln_attn_impl="flash_attn",
        rbln_kvcache_partition_len=8192,
    )
    m.save_pretrained(OUT)
    print("OK %.1fs -> %s" % (time.time() - t, OUT), flush=True)
except Exception as e:
    print("FAIL %.1fs %s: %s" % (time.time() - t, type(e).__name__, str(e)[:400]), flush=True)
    traceback.print_exc()
print("MKFLASH_DONE", flush=True)

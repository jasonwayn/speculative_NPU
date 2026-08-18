import torch, time, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
DST="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-c64-bs8"
try:
    t0=time.time()
    m=RBLNQwen3ForCausalLM.from_pretrained(SRC,export=True,
        rbln_batch_size=8, rbln_max_seq_len=2048,
        rbln_prefill_chunk_size=64,
        rbln_decoder_batch_sizes=[8,4,2,1])
    print(f"COMPILED {time.time()-t0:.1f}s",flush=True)
    m.save_pretrained(DST); print("saved",flush=True)
    print("decoder_batch_sizes =",m.rbln_config.decoder_batch_sizes,flush=True)
    print("MK8_OK",flush=True)
except Exception as e:
    traceback.print_exc(); print("MK8_FAIL "+str(e)[:200],flush=True)

import torch, time, json, os, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
BS=int(os.environ.get("BS","4"))
DST=f"/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-b{BS}"
try:
    t0=time.time()
    m=RBLNQwen3ForCausalLM.from_pretrained(SRC,export=True,
        rbln_batch_size=BS, rbln_max_seq_len=4096,
        rbln_output_hidden_states=True, rbln_prefill_chunk_size=64)
    print(f"COMPILED bs={BS} {time.time()-t0:.1f}s",flush=True)
    m.save_pretrained(DST); print("saved",DST,flush=True)
    print("cfg batch_size =",m.rbln_config.batch_size,flush=True)
    print("MKB_OK",flush=True)
except Exception as e:
    traceback.print_exc(); print("MKB_FAIL "+str(e)[:200],flush=True)

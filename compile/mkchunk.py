import torch, time, json, os, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
CS=int(os.environ.get("CS","16"))
DST=f"/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c{CS}"
try:
    t0=time.time()
    m=RBLNQwen3ForCausalLM.from_pretrained(SRC,export=True,
        rbln_batch_size=1, rbln_max_seq_len=4096,
        rbln_output_hidden_states=True, rbln_prefill_chunk_size=CS)
    print(f"COMPILED cs={CS} in {time.time()-t0:.1f}s",flush=True)
    m.save_pretrained(DST); print("saved "+DST,flush=True)
    print("effective chunk =",m.prefill_decoder.rbln_config.prefill_chunk_size,flush=True)
    ids=torch.randint(1000,5000,(1,77))
    o=m(input_ids=ids,attention_mask=torch.ones_like(ids))
    print("smoke: logits",tuple(o.logits.shape),"hidden",len(o.hidden_states),flush=True)
    print("MKCHUNK_OK "+json.dumps({"cs":CS}),flush=True)
except Exception as e:
    traceback.print_exc(); print("MKCHUNK_FAIL "+json.dumps({"cs":CS,"err":str(e)[:200]}),flush=True)

import torch, time, json, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
DST="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-k16"
try:
    t0=time.time()
    m=RBLNQwen3ForCausalLM.from_pretrained(SRC,export=True,
        rbln_batch_size=1, rbln_max_seq_len=4096,
        rbln_output_hidden_states=True, rbln_logits_to_keep=16)
    print(f"COMPILED {time.time()-t0:.1f}s",flush=True)
    m.save_pretrained(DST); print("saved",DST,flush=True)
    ids=torch.randint(1000,5000,(1,68))
    o=m(input_ids=ids,attention_mask=torch.ones_like(ids))
    print("logits",tuple(o.logits.shape)," hidden",len(o.hidden_states),"x",tuple(o.hidden_states[0].shape),flush=True)
    print("TGT16 "+json.dumps({"ok":True,"logits":list(o.logits.shape)}),flush=True)
except Exception as e:
    traceback.print_exc(); print("TGT16 "+json.dumps({"ok":False,"err":str(e)[:200]}),flush=True)

"""§3.2 우회 판정: hidden_states 가 prefix 전체에 대해, 어느 레이어까지 나오나."""
import torch, time
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
D="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64"
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
cfg=AutoConfig.from_pretrained(SRC)
print("target num_hidden_layers =",cfg.num_hidden_layers,"hidden_size =",cfg.hidden_size,flush=True)
tok=AutoTokenizer.from_pretrained(SRC)
m=RBLNQwen3ForCausalLM.from_pretrained(D,export=False,rbln_device=0)
pdec=m.prefill_decoder; CHUNK=pdec.rbln_config.prefill_chunk_size
print("prefill_chunk_size =",CHUNK,"max_seq_len =",m.rbln_config.max_seq_len,flush=True)

for L in [513, 2001]:
    ids=torch.randint(1000,5000,(1,L),dtype=torch.long)
    t=time.time()
    out=m(input_ids=ids,attention_mask=torch.ones_like(ids))
    dt=time.time()-t
    hs=out.hidden_states
    print(f"\n--- prompt_len={L}  wall={dt:.2f}s",flush=True)
    print(f"  num hidden tensors = {len(hs)}",flush=True)
    print(f"  shapes[0,1,-2,-1] = {[tuple(hs[i].shape) for i in (0,1,-2,-1)]}",flush=True)
    print(f"  logits shape      = {tuple(out.logits.shape) if out.logits is not None else None}",flush=True)
    covers = all(h.shape[1]==L for h in hs)
    print(f"  covers_full_prefix = {covers}",flush=True)
print("\nHSPROBE_DONE",flush=True)

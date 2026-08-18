"""NPU 타깃이 어떤 입력 길이에서 깨지는지 스윕."""
import torch, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
m=RBLNQwen3ForCausalLM.from_pretrained("/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden",export=False)
print("prefill_chunk_size =",getattr(m.rbln_config,"prefill_chunk_size",None),flush=True)
bad=[]
for L in [1,15,16,17,63,64,65,100,127,128,129,130,140,150,160,200,255,256,257,300]:
    ids=torch.randint(1000,5000,(1,L))
    try:
        o=m(input_ids=ids,attention_mask=torch.ones_like(ids))
        ok=tuple(o.hidden_states[0].shape)
        print(f"  L={L:4d} OK {ok}",flush=True)
    except Exception as e:
        bad.append(L); print(f"  L={L:4d} FAIL {type(e).__name__}: {str(e)[:90]}",flush=True)
print("BAD:",bad,flush=True)
if bad:
    ids=torch.randint(1000,5000,(1,bad[0]))
    try: m(input_ids=ids,attention_mask=torch.ones_like(ids))
    except Exception: traceback.print_exc()

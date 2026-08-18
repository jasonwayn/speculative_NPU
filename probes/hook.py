"""정상 forward() 가 prefill_forward 를 어떻게 부르는지 인자 그대로 찍는다."""
import torch
from optimum.rbln import RBLNQwen3ForCausalLM
import optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils as ru

orig_pf = ru.RBLNRuntimeModel.prefill_forward
def spy_pf(self, inputs, cache_position=None, attention_mask=None, batch_idx=None,
           block_tables=None, is_external_block_tables=None, position_ids=None,
           position_embed=None, token_type_ids=None, local_block_tables=None, lora_int_ids=None):
    def d(x):
        if x is None: return None
        if torch.is_tensor(x): return f"T{tuple(x.shape)}:{x.dtype}={x.flatten()[:4].tolist()}"
        return repr(x)
    print("  prefill_forward(", flush=True)
    for k,v in [("inputs",inputs),("cache_position",cache_position),("attention_mask",attention_mask),
                ("batch_idx",batch_idx),("block_tables",block_tables),
                ("is_external_block_tables",is_external_block_tables),("position_ids",position_ids),
                ("local_block_tables",local_block_tables)]:
        print(f"      {k:26s} = {d(v)}",flush=True)
    print("  )",flush=True)
    return orig_pf(self, inputs, cache_position, attention_mask, batch_idx, block_tables,
                   is_external_block_tables, position_ids, position_embed, token_type_ids,
                   local_block_tables, lora_int_ids)
ru.RBLNRuntimeModel.prefill_forward = spy_pf

# 저수준 그래프 호출도 본다
orig_fwd = ru.RBLNRuntimeModel.forward
def spy_fwd(self, *a, **k):
    print(f"    -> graph.forward: {len(a)} positional, kwargs={list(k.keys())}",flush=True)
    for i,x in enumerate(a):
        print(f"         [{i}] {('T'+str(tuple(x.shape))+':'+str(x.dtype)) if torch.is_tensor(x) else repr(x)[:60]}",flush=True)
    return orig_fwd(self,*a,**k)
ru.RBLNRuntimeModel.forward = spy_fwd

m=RBLNQwen3ForCausalLM.from_pretrained("/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden",
                                       export=False, rbln_device=1)
ids=torch.randint(1000,5000,(1,40))
print("=== 정상 호출 ===",flush=True)
m(input_ids=ids, attention_mask=torch.ones_like(ids))
print("HOOK_DONE",flush=True)

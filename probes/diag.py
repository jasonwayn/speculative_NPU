import torch
from optimum.rbln import RBLNQwen3ForCausalLM
m=RBLNQwen3ForCausalLM.from_pretrained("/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden",
                                       export=False, rbln_device=1)
pd=m.prefill_decoder; rc=pd.rbln_config
for k in ["logits_to_keep","use_local_attention","use_attention_mask","use_position_ids",
          "prefill_chunk_size","cache_impl","phases"]:
    print(f"  {k} = {getattr(rc,k,'<none>')}",flush=True)
print("  pd.phase =",getattr(pd,"phase","<none>"),flush=True)
rt=getattr(pd,"runtime",None) or getattr(pd,"_runtime",None)
print("  runtime attrs:",[a for a in dir(pd) if not a.startswith('__')][:20],flush=True)
try:
    import rebel
    print("  expected inputs:",pd.get_input_names() if hasattr(pd,"get_input_names") else "n/a",flush=True)
except Exception as e: print("  ",e,flush=True)

"""NPU 타깃으로 (a) 임의 길이 prefill, (b) hidden states, (c) 여러 토큰 로짓 을 얻을 수 있는지."""
import torch, json, traceback, inspect
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer
DST="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden"; SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
m=RBLNQwen3ForCausalLM.from_pretrained(DST,export=False)
rc=m.rbln_config
print("output_hidden_states =",rc.output_hidden_states," logits_to_keep =",getattr(rc,"logits_to_keep",None),flush=True)
print("max_seq_len =",rc.max_seq_len," batch_size =",rc.batch_size," phases =",getattr(rc,"phases",None),flush=True)
tok=AutoTokenizer.from_pretrained(SRC)
for L in [17, 68, 130]:
    ids=torch.randint(1000,5000,(1,L))
    try:
        o=m(input_ids=ids, attention_mask=torch.ones_like(ids))
        hs=o.hidden_states
        print(f"  len={L:4d} OK  logits={tuple(o.logits.shape)}  hidden={len(hs)}x{tuple(hs[0].shape)}",flush=True)
    except Exception as e:
        print(f"  len={L:4d} FAIL {type(e).__name__}: {str(e)[:120]}",flush=True)
print("\n=== forward 시그니처 / 런타임 ===",flush=True)
try:
    import optimum.rbln.transformers.models.decoderonly.modeling_decoderonly as md
    print(inspect.signature(md.RBLNDecoderOnlyModelForCausalLM.forward),flush=True)
except Exception as e: print("sig ERR",e,flush=True)
print("PROBE_OK",flush=True)

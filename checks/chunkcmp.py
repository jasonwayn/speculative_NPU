"""chunk=128 vs chunk=64 타깃이 같은 값을 내는가."""
import torch, json
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer
A="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden"     # chunk 128
Bp="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64"     # chunk 64
tok=AutoTokenizer.from_pretrained("/home/work/npu_work/eagle_test/Qwen3-4B")
ids=tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
        return_tensors="pt").input_ids
print("len",ids.shape[1],flush=True)
outs={}
for tag,path,dev in [("c128",A,0),("c64",Bp,1)]:
    m=RBLNQwen3ForCausalLM.from_pretrained(path,export=False,rbln_device=dev)
    o=m(input_ids=ids,attention_mask=torch.ones_like(ids))
    outs[tag]=(o.logits.float().clone(), o.hidden_states[-1].float().clone())
    print(f"  {tag}: chunk={m.prefill_decoder.rbln_config.prefill_chunk_size} "
          f"argmax={int(torch.argmax(o.logits[0,-1]))} {repr(tok.decode([int(torch.argmax(o.logits[0,-1]))]))}",flush=True)
    del m
la,ha=outs["c128"]; lb,hb=outs["c64"]
print(f"\nlogits  max|d|={(la-lb).abs().max():.4e}  cos={torch.nn.functional.cosine_similarity(la.flatten(),lb.flatten(),dim=0):.6f}",flush=True)
print(f"hidden  max|d|={(ha-hb).abs().max():.4e}  cos={torch.nn.functional.cosine_similarity(ha.flatten(),hb.flatten(),dim=0):.6f}",flush=True)
print("argmax 동일:", int(torch.argmax(la[0,-1]))==int(torch.argmax(lb[0,-1])),flush=True)

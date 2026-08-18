"""NPU 타깃의 hidden_states[-1] 에 lm_head 를 곱하면 반환 logits 와 같은가?"""
import torch, json
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
DST="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden"; SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
torch.set_num_threads(8)
m=RBLNQwen3ForCausalLM.from_pretrained(DST,export=False)
tok=AutoTokenizer.from_pretrained(SRC)
cpu=AutoModelForCausalLM.from_pretrained(SRC,dtype=torch.float32).eval()
lm=cpu.lm_head; nrm=cpu.model.norm
ids=tok("The capital of France is Paris, and the capital of Germany is",return_tensors="pt").input_ids
o=m(input_ids=ids,attention_mask=torch.ones_like(ids))
hs=o.hidden_states
ref=int(torch.argmax(o.logits[0,-1]))
print("NPU logits shape",tuple(o.logits.shape)," hidden",len(hs),"x",tuple(hs[0].shape),flush=True)
print("NPU 반환 logits argmax =",ref, repr(tok.decode([ref])),flush=True)
with torch.no_grad():
    for tag,h in [("hs[-1] 그대로",hs[-1]), ("norm(hs[-1])",nrm(hs[-1]))]:
        lg=lm(h.float())
        a=int(torch.argmax(lg[0,-1]))
        cs=float(torch.nn.functional.cosine_similarity(lg[0,-1],o.logits[0,-1].float(),dim=0))
        print(f"  {tag:16s} argmax={a} {repr(tok.decode([a])):12s} match={a==ref}  cos={cs:.6f}",flush=True)
    # 전 위치 로짓을 만들 수 있는지 (검증에 필요)
    best = nrm(hs[-1]) if True else hs[-1]
    lg=lm(best.float())
    print("  전 위치 로짓 shape =",tuple(lg.shape),flush=True)
print("LMHEAD_DONE",flush=True)

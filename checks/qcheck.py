"""양자화 모델이 정상 텍스트를 내는지 직접 확인."""
import torch, json
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer
D="/home/work/npu_work/dflash_work/"
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
tok=AutoTokenizer.from_pretrained(SRC)
q="What is 17 times 24? Answer briefly."
ids=tok.apply_chat_template([{"role":"user","content":q}],add_generation_prompt=True,
                            return_tensors="pt",enable_thinking=False)
for name in ["h-c64","h-c64-int8","h-c64-int4"]:
    try:
        m=RBLNQwen3ForCausalLM.from_pretrained(D+"rbln-Qwen3-4B-"+name,export=False,rbln_device=0)
        am=torch.ones_like(ids)
        o=m.generate(ids,attention_mask=am,max_new_tokens=48,do_sample=False,pad_token_id=tok.eos_token_id)
        txt=tok.decode(o[0][ids.shape[1]:],skip_special_tokens=True)
        print(f"--- {name}\n{repr(txt[:300])}\n",flush=True)
        del m
    except Exception as e:
        print(f"--- {name} FAIL {type(e).__name__}: {str(e)[:200]}\n",flush=True)
print("QCHECK_DONE",flush=True)

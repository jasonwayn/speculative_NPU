"""양자화 모델 원시 토큰 확인."""
import torch
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer
D="/home/work/npu_work/dflash_work/"
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
tok=AutoTokenizer.from_pretrained(SRC)
ids=tok.apply_chat_template([{"role":"user","content":"What is 17 times 24? Answer briefly."}],
    add_generation_prompt=True,return_tensors="pt",enable_thinking=False)
for name in ["h-c64","h-c64-int8","h-c64-int4"]:
    try:
        m=RBLNQwen3ForCausalLM.from_pretrained(D+"rbln-Qwen3-4B-"+name,export=False,rbln_device=0)
        o=m.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=24,
                     do_sample=False,pad_token_id=tok.eos_token_id)
        out=o[0][ids.shape[1]:].tolist()
        print("RES %-12s ids=%s"%(name,out[:12]),flush=True)
        print("RES %-12s txt=%r"%(name,tok.decode(out,skip_special_tokens=False)[:160]),flush=True)
        del m
    except Exception as e:
        print("RES %-12s FAIL %s: %s"%(name,type(e).__name__,str(e)[:160]),flush=True)
print("QC2_DONE",flush=True)

"""공식 dflash_generate 를 내 루프와 완전히 같은 조건으로 (3샘플, 96토큰)."""
import sys, torch, json
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
TGT="/home/work/npu_work/eagle_test/Qwen3-4B"; DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dflash.model import DFlashDraftModel, dflash_generate
torch.set_num_threads(8)
tok=AutoTokenizer.from_pretrained(TGT)
target=AutoModelForCausalLM.from_pretrained(TGT,dtype=torch.float32).eval()
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=torch.float32).eval()
ds=[json.loads(l) for l in open("/home/work/npu_work/dflash_work/dflash/cache/gsm8k.jsonl")][:3]
STOP=[tok.eos_token_id,151645]; allacc=[]
for i,ex in enumerate(ds):
    ids=tok.apply_chat_template([{"role":"user","content":ex["turns"][0]}],add_generation_prompt=True,
                                return_tensors="pt",enable_thinking=False)
    r=dflash_generate(draft,target=target,input_ids=ids,max_new_tokens=96,
                      stop_token_ids=STOP,temperature=0.0,return_stats=True)
    a=r.acceptance_lengths; allacc+=a
    print(f"[{i}] new={r.num_output_tokens} cycles={len(a)} tau={sum(a)/len(a):.3f} "
          f"running={sum(allacc)/len(allacc):.3f}",flush=True)
print("OFFICIAL3 "+json.dumps(dict(tau=round(sum(allacc)/len(allacc),3),cycles=len(allacc))),flush=True)

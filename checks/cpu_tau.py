"""CPU 레퍼런스 τ — 공식 dflash_generate 를 그대로 사용. 내일 NPU 포팅 검증의 정답값."""
import sys, json, time, torch, traceback
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
TGT="/home/work/npu_work/eagle_test/Qwen3-4B"
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
OUT="/home/work/npu_work/dflash_work/cpu_tau_results.jsonl"

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dflash.model import DFlashDraftModel, dflash_generate

torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(TGT)
print("loading target (fp32, CPU)...", flush=True); t0=time.time()
target = AutoModelForCausalLM.from_pretrained(TGT, dtype=torch.float32).eval()
print(f"  target loaded {time.time()-t0:.0f}s", flush=True)
cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation="eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=torch.float32).eval()
print("  draft loaded. block", draft.block_size, "tgt_ids", draft.target_layer_ids, flush=True)

# GSM8K 스타일 프롬프트 (논문 Table 1 의 GSM8K / T=0 대응)
PROMPTS = [
 "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step.",
 "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let's think step by step.",
 "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need? Let's think step by step.",
 "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year? Let's think step by step.",
 "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. How many flowers does Mark have in his garden? Let's think step by step.",
 "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. How many pieces does he eat that day? Let's think step by step.",
 "Ken created a care package to send to his brother. He placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. How much did the box weigh? Let's think step by step.",
 "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime. If she works 10 hours every day for 5 days, how much money does she make? Let's think step by step.",
]
STOP=[tok.eos_token_id, 151645]
f=open(OUT,"a")
allacc=[]
for i,p in enumerate(PROMPTS):
    msgs=[{"role":"user","content":p}]
    ids=tok.apply_chat_template(msgs, add_generation_prompt=True,
                                return_tensors="pt", enable_thinking=False)
    print(f"\n=== [{i}] prompt {ids.shape[1]} tok ===", flush=True)
    try:
        t0=time.time()
        r = dflash_generate(draft, target=target, input_ids=ids, max_new_tokens=128,
                            stop_token_ids=STOP, temperature=0.0, return_stats=True)
        el=time.time()-t0
        acc=r.acceptance_lengths
        tau=sum(acc)/len(acc)
        allacc += acc
        rec=dict(i=i, prompt_tokens=int(ids.shape[1]), new_tokens=int(r.num_output_tokens),
                 cycles=len(acc), tau=round(tau,3), sec=round(el,1),
                 running_tau=round(sum(allacc)/len(allacc),3))
        print("  "+json.dumps(rec), flush=True)
        f.write(json.dumps(rec)+"\n"); f.flush()
    except Exception as e:
        print("  FAILED", flush=True); traceback.print_exc()
        f.write(json.dumps(dict(i=i, error=str(e)[:300]))+"\n"); f.flush()
if allacc:
    print("\nFINAL_TAU "+json.dumps(dict(tau=round(sum(allacc)/len(allacc),3),
          cycles=len(allacc), paper_target=6.53)), flush=True)
f.close()

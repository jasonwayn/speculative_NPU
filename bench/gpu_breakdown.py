"""CUDA-event breakdown of the official GPU DFlash generation loop."""
import json
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.expanduser("~/dflash_bench/dflash"))
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from dflash.model import DFlashDraftModel, dflash_generate

TGT = "Qwen/Qwen3-4B"
DRF = "z-lab/Qwen3-4B-DFlash-b16"
NSAMP = int(os.environ.get("NSAMP", "3"))
MAXNEW = int(os.environ.get("MAXNEW", "512"))
DSET = os.environ.get("DSET", "gsm8k")
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "0"))
NATURAL_LONG = int(os.environ.get("NATURAL_LONG", "0"))
DTYPE = os.environ.get("DTYPE", "bfloat16")

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

dtype = getattr(torch, DTYPE)
tokenizer = AutoTokenizer.from_pretrained(TGT)
target = AutoModelForCausalLM.from_pretrained(TGT, dtype=dtype).cuda().eval()
config = AutoConfig.from_pretrained(DRF, trust_remote_code=True)
draft = DFlashDraftModel.from_pretrained(DRF, config=config, dtype=dtype).cuda().eval()
stop_ids = [tokenizer.eos_token_id, 151645]

class Profiler:
    def __init__(self):
        self.enabled = False
        self.first_target = True
        self.target_depth = 0
        self.events = {"prefill": [], "draft": [], "draft_lmhead": [], "verify": []}

    def begin_generation(self):
        self.first_target = True

    def record(self, name, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        self.events[name].append((start, end))
        return result

    def milliseconds(self):
        torch.cuda.synchronize()
        return {name: [start.elapsed_time(end) for start, end in pairs]
                for name, pairs in self.events.items()}

profiler = Profiler()
original_draft = draft.forward
original_target = target.forward
original_lmhead = target.lm_head.forward

def timed_draft(self, *args, **kwargs):
    return profiler.record("draft", original_draft, *args, **kwargs)

def timed_target(self, *args, **kwargs):
    if not profiler.enabled:
        return original_target(*args, **kwargs)
    name = "prefill" if profiler.first_target else "verify"
    profiler.first_target = False
    profiler.target_depth += 1
    try:
        return profiler.record(name, original_target, *args, **kwargs)
    finally:
        profiler.target_depth -= 1

def timed_lmhead(self, *args, **kwargs):
    if profiler.target_depth:
        return original_lmhead(*args, **kwargs)
    return profiler.record("draft_lmhead", original_lmhead, *args, **kwargs)

draft.forward = types.MethodType(timed_draft, draft)
target.forward = types.MethodType(timed_target, target)
target.lm_head.forward = types.MethodType(timed_lmhead, target.lm_head)

dataset_path = os.path.expanduser(f"~/dflash_bench/cache/{DSET}.jsonl")
all_dataset = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
dataset = all_dataset[:NSAMP]

def ids_for(example, index=0):
    if NATURAL_LONG:
        ordered = [all_dataset[(index + offset) % len(all_dataset)]["turns"][0]
                   for offset in range(len(all_dataset))]
        document = "Mathematics problem collection:\n\n" + "\n\n".join(
            f"{number + 1}. {text}" for number, text in enumerate(ordered))
        document += ("\n\nSummarize the recurring problem types and the main reasoning "
                     "strategies needed to solve this collection.")
        low, high, best = 1, len(document), None
        while low <= high:
            middle = (low + high) // 2
            candidate = tokenizer.apply_chat_template(
                [{"role": "user", "content": document[:middle]}],
                add_generation_prompt=True, return_tensors="pt", enable_thinking=False)
            if candidate.shape[1] <= PROMPT_LEN:
                best = candidate; low = middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ValueError("PROMPT_LEN is too small")
        return best.cuda()
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": example["turns"][0]}],
        add_generation_prompt=True, return_tensors="pt", enable_thinking=False).cuda()

# Warm up without recording.
with torch.inference_mode():
    dflash_generate(draft, target=target, input_ids=ids_for(dataset[0], 0), max_new_tokens=32,
                    stop_token_ids=stop_ids, temperature=0.0)
torch.cuda.synchronize()

profiler.enabled = True
all_acceptance = []
tokens = 0
wall_start = time.perf_counter()
prompt_lengths = []
for index, example in enumerate(dataset):
    profiler.begin_generation()
    input_ids = ids_for(example, index)
    prompt_lengths.append(input_ids.shape[1])
    result = dflash_generate(
        draft, target=target, input_ids=input_ids, max_new_tokens=MAXNEW,
        stop_token_ids=stop_ids, temperature=0.0, return_stats=True)
    all_acceptance.extend(result.acceptance_lengths)
    tokens += result.num_output_tokens
torch.cuda.synchronize()
wall_seconds = time.perf_counter() - wall_start

measurements = profiler.milliseconds()
def stats(values):
    values = sorted(values)
    n = len(values)
    return {
        "calls": n,
        "mean_ms": round(sum(values) / max(n, 1), 3),
        "median_ms": round(values[n // 2], 3) if n else None,
        "p10_ms": round(values[round((n - 1) * 0.1)], 3) if n else None,
        "p90_ms": round(values[round((n - 1) * 0.9)], 3) if n else None,
        "total_ms": round(sum(values), 3),
    }

result = {
    "dataset": DSET,
    "samples": len(dataset),
    "maxnew": MAXNEW,
    "tokens": tokens,
    "rounds": len(all_acceptance),
    "tau": round(sum(all_acceptance) / len(all_acceptance), 3),
    "wall_s": round(wall_seconds, 3),
    "tok_s": round(tokens / wall_seconds, 2),
    "prompt_lengths": prompt_lengths,
    "natural_long": NATURAL_LONG,
    "sections": {name: stats(values) for name, values in measurements.items()},
}
print("GPU_BREAKDOWN " + json.dumps(result), flush=True)

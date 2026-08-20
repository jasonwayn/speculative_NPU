"""Compare the pre-compilation RBLN Qwen3 wrapper with Hugging Face on CPU."""
import json

import torch
from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.configuration_utils import RBLNCompileConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
ROOT = "/home/work/npu_work/dflash_work"
CHUNK = 256

tokenizer = AutoTokenizer.from_pretrained(SRC)
dataset = [json.loads(line) for line in open(ROOT + "/dflash/cache/gsm8k.jsonl")]
document = "Mathematics problem collection:\n\n" + "\n\n".join(
    "%d. %s" % (number + 1, row["turns"][0])
    for number, row in enumerate(dataset)
)
document += (
    "\n\nSummarize the recurring problem types and the main reasoning "
    "strategies needed to solve this collection."
)
low, high, ids = 1, len(document), None
while low <= high:
    middle = (low + high) // 2
    candidate = tokenizer.apply_chat_template(
        [{"role": "user", "content": document[:middle]}],
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    if candidate.shape[1] <= 1024:
        ids = candidate
        low = middle + 1
    else:
        high = middle - 1
assert ids is not None and ids.shape[1] == 1024
ids = ids[:, :CHUNK]

model_config = AutoConfig.from_pretrained(SRC)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32).eval()
with torch.inference_mode():
    reference = hf(ids, use_cache=False, output_hidden_states=True)

cls = RBLNQwen3ForCausalLM
rc, _ = cls.prepare_rbln_config(rbln_config={
    "batch_size": 1,
    "max_seq_len": 4096,
    "prefill_chunk_size": CHUNK,
    "output_hidden_states": True,
    "create_runtimes": False,
})
rc.max_seq_len = 4096
rc = cls._update_rbln_config(
    preprocessors=None, model=hf, model_config=model_config, rbln_config=rc
)
wrapped = cls._wrap_model_if_needed(hf, rc).eval()
info = cls.get_input_info(
    batch_size=1, query_length=CHUNK, rbln_config=rc, model_config=model_config
)
cc = RBLNCompileConfig(compiled_model_name="prefill_256", input_info=info)
args = list(cc.get_dummy_inputs(fill=0))
for index, (name, _, _) in enumerate(cc.input_info):
    if name == "input_ids":
        args[index] = ids
    elif name == "cache_position":
        args[index] = torch.arange(CHUNK, dtype=torch.int32).unsqueeze(0)
    elif name == "block_tables":
        args[index] = torch.tensor([0], dtype=torch.int16)
    elif name == "query_position":
        args[index] = torch.tensor(CHUNK - 1, dtype=torch.int16)

print("WRAPPED_CONFIG dtype=%s attn_impl=%s" % (rc.dtype, rc.attn_impl), flush=True)
with torch.inference_mode():
    output = wrapped(*args)
print("WRAPPED_OUTPUT type=%s len=%d" % (type(output).__name__, len(output)), flush=True)
logits, hidden_states = output
print(
    "WRAPPED_CONTRACT logits=%s hidden_count=%d first_shape=%s first_dtype=%s"
    % (tuple(logits.shape), len(hidden_states), tuple(hidden_states[0].shape), hidden_states[0].dtype),
    flush=True,
)

# The wrapper hidden-state tuple follows the HF convention: embedding at 0,
# then one output per decoder layer.
for layer in range(17):
    got = hidden_states[layer + 1][0].float()
    ref = reference.hidden_states[layer + 1][0].float()
    cosine = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    relative = (got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)
    print(
        "WRAPPED_LAYER layer=%d cos_min=%.8f cos_mean=%.8f rel_max=%.6g"
        % (layer, float(cosine.min()), float(cosine.mean()), float(relative.max())),
        flush=True,
    )
print(
    "WRAPPED_TOKEN hf=%d wrapped=%d"
    % (int(torch.argmax(reference.logits[0, -1])), int(torch.argmax(logits[0, -1]))),
    flush=True,
)

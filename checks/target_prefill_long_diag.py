"""Validate long target prefill and its KV continuation against PyTorch."""
import copy
import inspect
import json
import os
import textwrap

import rebel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as CF
import optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils as RU
from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNPageTableManager

ROOT = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
GRAPH = os.environ.get("GRAPH", ROOT + "/fused_17_256")
DEV = int(os.environ.get("DEV", "0"))
PROMPT_LEN = 1024
KEEP = tuple(int(value) for value in os.environ.get("KEEP", "2,10,18,26,34").split(","))
TARGET_LAYERS = tuple(value - 1 for value in KEEP)
BT = torch.tensor([0], dtype=torch.int16)

# Permit continuation at a non-multiple prefix length.
method = RU.RBLNRuntimeModel
lines = inspect.getsource(method.prefill_forward).split("\n")
out, index, removed = [], 0, False
while index < len(lines):
    line = lines[index]
    if "prefix_cached_len % self.rbln_config.prefill_chunk_size != 0" in line:
        indent = len(line) - len(line.lstrip())
        index += 1
        while index < len(lines) and (
            not lines[index].strip()
            or len(lines[index]) - len(lines[index].lstrip()) > indent
        ):
            index += 1
        removed = True
        continue
    out.append(line)
    index += 1
assert removed
body = textwrap.dedent("\n".join(out)).replace("super()", "super(_Base, self)")
namespace = dict(RU.__dict__)
namespace["_Base"] = method
exec(compile(body, "<prefill_diag_patch>", "exec"), namespace)
method.prefill_forward = namespace["prefill_forward"]

original_init = CF.RBLNDecoderOnlyModelForCausalLMConfig.__init__


def config_init(self, *args, **kwargs):
    requested = kwargs.get("prefill_chunk_size")
    if requested is not None and requested % 64:
        kwargs = dict(kwargs)
        kwargs["prefill_chunk_size"] = 64
        original_init(self, *args, **kwargs)
        self.prefill_chunk_size = requested
    else:
        original_init(self, *args, **kwargs)


CF.RBLNDecoderOnlyModelForCausalLMConfig.__init__ = config_init

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
    if candidate.shape[1] <= PROMPT_LEN:
        ids = candidate
        low = middle + 1
    else:
        high = middle - 1
assert ids is not None and ids.shape[1] == PROMPT_LEN
print("INPUT tokens=%d chars=%d" % (ids.shape[1], high), flush=True)

model_config = AutoConfig.from_pretrained(SRC)
reference_dtype = torch.bfloat16 if os.environ.get("REF_DTYPE", "bf16") == "bf16" else torch.float32
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=reference_dtype).eval()
with torch.inference_mode():
    cpu = hf(ids, use_cache=False, output_hidden_states=True)
cpu_hidden = tuple(cpu.hidden_states[layer] for layer in KEEP)
cpu_logits = cpu.logits
print("CPU_READY dtype=%s" % reference_dtype, flush=True)

rbln_config, _ = RBLNQwen3ForCausalLM.prepare_rbln_config(rbln_config={
    "batch_size": 1,
    "max_seq_len": 4096,
    "prefill_chunk_size": 17,
    "output_hidden_states": True,
    "create_runtimes": False,
})
rbln_config.max_seq_len = 4096
rbln_config = RBLNQwen3ForCausalLM._update_rbln_config(
    preprocessors=None,
    model=hf,
    model_config=model_config,
    rbln_config=rbln_config,
)
if os.environ.get("GRAPH_DTYPE") == "bf16":
    rbln_config.dtype = torch.bfloat16
page_table = RBLNPageTableManager(rbln_config)
mask = torch.zeros(1, 1, 1, 4096, dtype=torch.float32)
common = dict(
    main_input_name="input_ids",
    embed_tokens=hf.model.embed_tokens,
    dec_attn_mask=mask,
    page_table_manager=page_table,
)
runtimes = {}
for chunk in (17, 256):
    compiled = rebel.RBLNCompiledModel(GRAPH + "/prefill_%d.rbln" % chunk)
    config = copy.deepcopy(rbln_config)
    config.prefill_chunk_size = chunk
    if chunk == 17:
        config.logits_to_keep = 17
    visible_config = copy.deepcopy(model_config)
    visible_config.num_hidden_layers = len(KEEP) if chunk == 17 else len(KEEP) - 1
    runtimes[chunk] = RU.RBLNRuntimeModel(
        runtime=rebel.Runtime(compiled, tensor_type="pt", device=DEV),
        phase="prefill",
        batch_size=1,
        rbln_config=config,
        config=visible_config,
        **common,
    )
print("NPU_READY", flush=True)


def prefill(chunk, tokens, offset=0):
    length = tokens.shape[1]
    return runtimes[chunk].prefill_forward(
        tokens,
        cache_position=torch.arange(offset, offset + length, dtype=torch.int32).unsqueeze(0),
        attention_mask=torch.ones(length, dtype=torch.int64),
        batch_idx=0,
        block_tables=BT,
        is_external_block_tables=False,
    )


def compare_hidden(tag, npu_states, cpu_states, npu_position, cpu_position=None):
    cpu_position = npu_position if cpu_position is None else cpu_position
    for output_index, target_layer in enumerate(TARGET_LAYERS):
        got = npu_states[output_index][0, npu_position].float()
        ref = cpu_states[output_index][0, cpu_position].float()
        cosine = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
        maximum = (got - ref).abs().max().item()
        mean = (got - ref).abs().mean().item()
        print(
            "%s pos=%d target_layer=%d cos=%.8f max=%.6g mean=%.6g"
            % (tag, cpu_position, target_layer, cosine, maximum, mean),
            flush=True,
        )


# Inspect every chunk boundary without using an exact multiple, which avoids
# the SDK's zero-padding slice corner case.
full_prefix = None
for length in (255, 511, 767, 1023):
    result = prefill(256, ids[:, :length])
    if length == 1023:
        full_prefix = result
    compare_hidden("PREFIX", result.hidden_states, cpu_hidden, length - 1)
    cpu_boundary_token = int(torch.argmax(cpu_logits[0, length - 1]))
    npu_boundary_token = int(torch.argmax(result.logits[0, -1].float()))
    print(
        "PREFIX_LOGIT pos=%d cpu=%d npu=%d same=%s"
        % (length - 1, cpu_boundary_token, npu_boundary_token,
           cpu_boundary_token == npu_boundary_token),
        flush=True,
    )

for output_index, target_layer in enumerate(TARGET_LAYERS):
    got = full_prefix.hidden_states[output_index][0].float()
    ref = cpu_hidden[output_index][0, :1023].float()
    per_token = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    sorted_cos = torch.sort(per_token).values
    worst_positions = torch.topk(per_token, k=5, largest=False).indices
    print(
        "FULL_HIDDEN target_layer=%d min=%.8f p01=%.8f median=%.8f mean=%.8f below_099=%d"
        % (
            target_layer,
            float(sorted_cos[0]),
            float(sorted_cos[max(0, int(0.01 * len(sorted_cos)) - 1)]),
            float(sorted_cos[len(sorted_cos) // 2]),
            float(per_token.mean()),
            int((per_token < 0.99).sum()),
        ),
        flush=True,
    )
    for position in worst_positions:
        position = int(position)
        got_vector = got[position]
        ref_vector = ref[position]
        error = (got_vector - ref_vector).norm()
        print(
            "FULL_DETAIL target_layer=%d pos=%d token=%d text=%r ref_norm=%.6g npu_norm=%.6g rel_l2=%.6g"
            % (
                target_layer,
                position,
                int(ids[0, position]),
                tokenizer.decode([int(ids[0, position])]),
                float(ref_vector.norm()),
                float(got_vector.norm()),
                float(error / ref_vector.norm().clamp_min(1e-12)),
            ),
        flush=True,
    )

save_path = os.environ.get("SAVE_NPU")
if save_path:
    torch.save(
        {
            "ids": ids[:, :1023].cpu(),
            "keep": KEEP,
            "hidden": tuple(
                state[:, :1023].detach().cpu()
                for state in full_prefix.hidden_states[:len(KEEP)]
            ),
        },
        save_path,
    )
    print("SAVED_NPU %s" % save_path, flush=True)
    if os.environ.get("SAVE_ONLY") == "1":
        raise SystemExit(0)

# Cross-check the independently compiled verify-width graph. If width 17 and
# width 256 disagree, the corruption is in chunk-specific compilation/runtime.
width17 = prefill(17, ids[:, :1023])
for output_index, target_layer in enumerate(TARGET_LAYERS):
    from256 = full_prefix.hidden_states[output_index][0].float()
    from17 = width17.hidden_states[output_index][0].float()
    per_token = torch.nn.functional.cosine_similarity(from256, from17, dim=-1)
    worst = torch.topk(per_token, k=5, largest=False)
    print(
        "WIDTH_256_VS_17 target_layer=%d min=%.8f mean=%.8f below_099=%d worst=%s"
        % (
            target_layer,
            float(per_token.min()),
            float(per_token.mean()),
            int((per_token < 0.99).sum()),
            " ".join(
                "%d:%.6f" % (int(position), float(value))
                for position, value in zip(worst.indices, worst.values)
            ),
        ),
        flush=True,
    )
    worst = torch.topk(per_token, k=min(12, len(per_token)), largest=False)
    print(
        "FULL_WORST target_layer=%d %s"
        % (
            target_layer,
            " ".join(
                "%d:%.6f" % (int(position), float(value))
                for position, value in zip(worst.indices, worst.values)
            ),
        ),
        flush=True,
    )

# Recreate the production 1024 path: 1023 through prefill, final real token
# through the verify graph. This also leaves the target KV cache at 1024.
prefix = prefill(256, ids[:, :1023])
last = prefill(17, ids[:, 1023:1024], offset=1023)
compare_hidden("FINAL", last.hidden_states, cpu_hidden, 0, 1023)
cpu_token = int(torch.argmax(cpu_logits[0, 1023]))
npu_token = int(torch.argmax(last.logits[0, -1].float()))
print("FINAL_LOGIT cpu=%d npu=%d same=%s" % (cpu_token, npu_token, cpu_token == npu_token), flush=True)

# KV continuation check. Feed the CPU greedy next token at position 1024 and
# compare against a full PyTorch 1025-token forward.
next_id = torch.tensor([[cpu_token]], dtype=ids.dtype)
with torch.inference_mode():
    cpu_next = hf(
        torch.cat([ids, next_id], dim=1),
        use_cache=False,
        output_hidden_states=True,
    )
continued = prefill(17, next_id, offset=1024)
next_hidden = tuple(cpu_next.hidden_states[layer] for layer in KEEP)
compare_hidden("CONTINUE", continued.hidden_states, next_hidden, 0, 1024)
cpu_next_token = int(torch.argmax(cpu_next.logits[0, 1024]))
npu_next_token = int(torch.argmax(continued.logits[0, -1].float()))
print(
    "CONTINUE_LOGIT cpu=%d npu=%d same=%s"
    % (cpu_next_token, npu_next_token, cpu_next_token == npu_next_token),
    flush=True,
)
print("TARGET_PREFILL_DIAG_DONE", flush=True)

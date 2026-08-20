import os
import torch
from transformers import AutoModelForCausalLM

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
PAYLOAD = os.environ.get("PAYLOAD", "/tmp/npu_layers9_17.pt")

payload = torch.load(PAYLOAD, map_location="cpu")
ids = payload["ids"]
layers = tuple(value - 1 for value in payload["keep"])
reference_dtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}[os.environ.get("REF_DTYPE", "float32")]
npu_hidden = tuple(value.to(reference_dtype) for value in payload["hidden"])
model = AutoModelForCausalLM.from_pretrained(
    SRC, dtype=reference_dtype, attn_implementation="sdpa"
).eval()
print("STEP_REFERENCE dtype=%s payload=%s" % (reference_dtype, PAYLOAD), flush=True)

for input_index, input_layer in enumerate(layers[:-1]):
    next_layer = input_layer + 1
    assert layers[input_index + 1] == next_layer
    injected = npu_hidden[input_index]

    def hook(_module, args, kwargs, value=injected):
        if args:
            args = (value,) + tuple(args[1:])
        else:
            kwargs["hidden_states"] = value
        return args, kwargs

    handle = model.model.layers[next_layer].register_forward_pre_hook(hook, with_kwargs=True)
    with torch.inference_mode():
        output = model(ids, use_cache=False, output_hidden_states=True)
    handle.remove()
    cpu_step = output.hidden_states[next_layer + 1][0].float()
    npu_next = npu_hidden[input_index + 1][0].float()
    cosine = torch.nn.functional.cosine_similarity(cpu_step, npu_next, dim=-1)
    rel = (cpu_step - npu_next).norm(dim=-1) / cpu_step.norm(dim=-1).clamp_min(1e-12)
    print(
        "STEP %d_TO_%d cos_min=%.8f cos_mean=%.8f rel_max=%.6g "
        "p608_cpu=%.6g p608_npu=%.6g p813_cpu=%.6g p813_npu=%.6g"
        % (
            input_layer,
            next_layer,
            float(cosine.min()),
            float(cosine.mean()),
            float(rel.max()),
            float(cpu_step[608].norm()),
            float(npu_next[608].norm()),
            float(cpu_step[813].norm()),
            float(npu_next[813].norm()),
        ),
        flush=True,
    )

import torch
from transformers import AutoModelForCausalLM

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
PAYLOAD = "/tmp/npu_layers15_16.pt"

payload = torch.load(PAYLOAD, map_location="cpu")
ids = payload["ids"]
npu_layer15 = payload["hidden"][0].to(dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    SRC, dtype=torch.bfloat16, attn_implementation="sdpa"
).eval()


def inject(_module, args, kwargs):
    if args:
        args = (npu_layer15,) + tuple(args[1:])
    else:
        kwargs["hidden_states"] = npu_layer15
    return args, kwargs


handle = model.model.layers[16].register_forward_pre_hook(inject, with_kwargs=True)
with torch.inference_mode():
    output = model(ids, use_cache=False, output_hidden_states=True)
handle.remove()
hidden = output.hidden_states[17][0].float()
top = torch.topk(hidden.norm(dim=-1), k=12)
print("CPU_WITH_NPU_LAYER15")
print("pos608_norm=%.6g pos813_norm=%.6g" % (
    float(hidden[608].norm()), float(hidden[813].norm())
))
print("top=" + " ".join(
    "%d:%.6g" % (int(position), float(value))
    for position, value in zip(top.indices, top.values)
))
print("final_token", int(torch.argmax(output.logits[0, -1])))

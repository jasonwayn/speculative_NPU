"""Run the target layer 1 probe and compare NPU stages with CPU stages."""
import rebel
import torch
from transformers import AutoModelForCausalLM


SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
GRAPH = "/home/work/npu_work/dflash_work/diag_target_layer1_probe_v3.rbln"
PAYLOAD = "/tmp/npu_layers0_9.pt"
CHUNK = 256
MAXC = 4096

payload = torch.load(PAYLOAD, map_location="cpu")
ids = payload["ids"][:, :CHUNK]
npu_layer0 = payload["hidden"][0][:, :CHUNK].float().contiguous()
npu_layer1_full = payload["hidden"][1][:, :CHUNK].float()
runtime = rebel.Runtime(rebel.RBLNCompiledModel(GRAPH), tensor_type="pt", device=0)
npu_stages = runtime(
    npu_layer0,
    torch.arange(CHUNK, dtype=torch.int32).unsqueeze(0),
    torch.tensor([0], dtype=torch.int16),
)
stage_names = (
    "pre_norm",
    "attention",
    "post_attention",
    "post_norm",
    "gate",
    "up",
    "activated",
    "fused",
    "mlp",
    "output",
)

model = AutoModelForCausalLM.from_pretrained(
    SRC, dtype=torch.float32, attn_implementation="sdpa"
).eval()
layer = model.model.layers[1]
captured = {}


def save(name):
    def hook(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured[name] = value.detach()
    return hook


def inject(_module, args, kwargs):
    if args:
        args = (npu_layer0,) + tuple(args[1:])
    else:
        kwargs["hidden_states"] = npu_layer0
    return args, kwargs


handles = [
    layer.register_forward_pre_hook(inject, with_kwargs=True),
    layer.input_layernorm.register_forward_hook(save("pre_norm")),
    layer.self_attn.register_forward_hook(save("attention")),
    layer.post_attention_layernorm.register_forward_hook(save("post_norm")),
    layer.mlp.register_forward_hook(save("mlp")),
    layer.register_forward_hook(save("output")),
]
with torch.inference_mode():
    model(ids, use_cache=False, output_hidden_states=False)
for handle in handles:
    handle.remove()
captured["post_attention"] = npu_layer0 + captured["attention"]
mlp_module = layer.mlp
with torch.inference_mode():
    captured["gate"] = mlp_module.gate_proj(captured["post_norm"])
    captured["up"] = mlp_module.up_proj(captured["post_norm"])
    captured["activated"] = mlp_module.act_fn(captured["gate"])
    captured["fused"] = captured["activated"] * captured["up"]


def compare(tag, got, ref):
    got = got[0].float()
    ref = ref[0].float()
    cosine = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    relative = (got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)
    worst = int(torch.argmax(relative))
    print(
        "%s cos_min=%.8f cos_mean=%.8f rel_max=%.6g worst=%d "
        "p128_rel=%.6g p128_got_norm=%.6g p128_ref_norm=%.6g"
        % (
            tag,
            float(cosine.min()),
            float(cosine.mean()),
            float(relative.max()),
            worst,
            float(relative[128]),
            float(got[128].norm()),
            float(ref[128].norm()),
        ),
        flush=True,
    )


for name, npu in zip(stage_names, npu_stages):
    compare("STAGE_" + name, npu, captured[name])

# Compare each MLP operation with exactly the NPU tensor that entered it. This
# separates compiler error in an operation from propagation of prior error.
npu_post_norm, npu_gate, npu_up, npu_activated, npu_fused, npu_mlp = npu_stages[3:9]
torch.save(npu_gate.cpu(), "/tmp/npu_layer1_gate.pt")
print(
    "GATE_STATS shape=%s min=%.6g max=%.6g mean=%.6g"
    % (
        tuple(npu_gate.shape),
        float(npu_gate.min()),
        float(npu_gate.max()),
        float(npu_gate.mean()),
    ),
    flush=True,
)
with torch.inference_mode():
    compare("SAME_INPUT_gate", npu_gate, mlp_module.gate_proj(npu_post_norm))
    compare("SAME_INPUT_up", npu_up, mlp_module.up_proj(npu_post_norm))
    compare("SAME_INPUT_activated", npu_activated, mlp_module.act_fn(npu_gate))
    compare("SAME_INPUT_fused", npu_fused, npu_activated * npu_up)
    compare("SAME_INPUT_down", npu_mlp, mlp_module.down_proj(npu_fused))
compare("PROBE_VS_FULL_GRAPH", npu_stages[-1], npu_layer1_full)
print("LAYER1_PROBE_DONE", flush=True)

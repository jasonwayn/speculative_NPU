"""Compare NPU SiLU formula variants against PyTorch float32 SiLU."""
import rebel
import torch


GRAPH = "/home/work/npu_work/dflash_work/diag_silu_accuracy_probe.rbln"
x = torch.load("/tmp/npu_layer1_gate.pt", map_location="cpu").float().contiguous()
reference = torch.nn.functional.silu(x)
names = ("native", "sigmoid_form", "exp_clamped", "stable", "tanh_form", "piecewise")
runtime = rebel.Runtime(rebel.RBLNCompiledModel(GRAPH), tensor_type="pt", device=0)
outputs = runtime(x)
for name, output in zip(names, outputs):
    output = output.float()
    error = output - reference
    per_token = error.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)
    print(
        "SILU_VARIANT name=%s rel_max=%.8g rel_mean=%.8g abs_max=%.8g abs_mean=%.8g"
        % (
            name,
            float(per_token.max()),
            float(per_token.mean()),
            float(error.abs().max()),
            float(error.abs().mean()),
        ),
        flush=True,
    )
print("SILU_ACCURACY_DONE", flush=True)

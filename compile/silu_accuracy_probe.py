"""Compile mathematically equivalent SiLU formulas for NPU accuracy testing."""
import rebel
import torch
import torch.nn as nn


OUT = "/home/work/npu_work/dflash_work/diag_silu_accuracy_probe.rbln"
SHAPE = [1, 256, 9728]


class SiluProbe(nn.Module):
    def forward(self, x):
        native = torch.nn.functional.silu(x)
        sigmoid_form = x * torch.sigmoid(x)
        exp_clamped = x / (1.0 + torch.exp(-torch.clamp(x, -20.0, 20.0)))
        exp_positive = torch.exp(torch.clamp(x, max=0.0))
        stable = torch.where(
            x >= 0.0,
            x / (1.0 + torch.exp(-torch.clamp(x, max=20.0))),
            x * exp_positive / (1.0 + exp_positive),
        )
        tanh_form = 0.5 * x * (1.0 + torch.tanh(0.5 * x))
        piecewise = torch.where(
            x < -10.0,
            torch.zeros_like(x),
            torch.where(x > 10.0, x, stable),
        )
        return native, sigmoid_form, exp_clamped, stable, tanh_form, piecewise


input_info = [("x", SHAPE, "float32")]
compiled = rebel.compile_from_torch(SiluProbe().eval(), input_info=input_info)
compiled.save(OUT)
print("SILU_PROBE_COMPILE_OK", OUT, flush=True)

"""같은 서버에서 sin/cos 오차: NPU(fp32 입력) vs CPU(fp32) vs CPU(fp16).

기준은 CPU float64. fp32 가 제대로 구현돼 있으면 |x| 와 무관하게 약 1e-7 이어야 한다
(출력이 [-1,1] 이므로 fp32 ulp ≈ 6e-8).
"""
import os, math, torch, rebel
import torch.nn as nn
DEV = int(os.environ.get("DEV", "0")); N = 4096
torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))


class SinCos(nn.Module):
    def forward(self, x):
        return torch.sin(x), torch.cos(x)


cm = rebel.compile_from_torch(SinCos().eval(), input_info=[("x", [1, N], "float32")],
                              example_inputs=[torch.zeros(1, N, dtype=torch.float32)])
rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
torch.manual_seed(0)

print("%-14s %-13s %-13s %-13s %s"
      % ("입력 범위", "NPU fp32", "CPU fp32", "CPU fp16", "NPU/CPU 배수"), flush=True)
for hi, tag in [(2 * math.pi, "[0, 2pi)"), (100, "[0, 100)"), (1000, "[0, 1000)"),
                (4096, "[0, 4096)"), (1e5, "[0, 1e5)"), (1e8, "[0, 1e8)")]:
    x = (torch.rand(1, N).double() * hi).float()
    ref_s, ref_c = x.double().sin(), x.double().cos()

    s, c = [torch.as_tensor(t).double() for t in rt(x.contiguous())]
    e_npu = max((s - ref_s).abs().max().item(), (c - ref_c).abs().max().item())

    e_cpu = max((x.sin().double() - ref_s).abs().max().item(),
                (x.cos().double() - ref_c).abs().max().item())

    xh = x.half()
    e_f16 = max((xh.sin().double() - ref_s).abs().max().item(),
                (xh.cos().double() - ref_c).abs().max().item())

    print("%-14s %-13.3e %-13.3e %-13.3e %.3g"
          % (tag, e_npu, e_cpu, e_f16, e_npu / max(e_cpu, 1e-20)), flush=True)
print("CPUNPU_DONE", flush=True)

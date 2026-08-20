"""가설: NPU 가 sin/cos 인자를 fp16 으로 잘라서 쓴다.

fp16 의 ulp(옆 표현가능 값과의 간격)와 관측 오차가 자릿수까지 맞는다:
   x~6.28  ulp 0.0039   관측 5.3e-3
   x~100   ulp 0.0625   관측 6.1e-2
   x~1000  ulp 0.5      관측 4.9e-1
   x~4096  ulp 4.0      관측 1.68 (포화)
sin' <= 1 이므로 인자 오차가 그대로 출력 오차 상한이 된다.

판별: NPU 결과가 sin(fp64(x)) 보다 sin(fp16(x)) 에 훨씬 가까우면 가설 성립.
"""
import os, math, torch, rebel
import torch.nn as nn
DEV = int(os.environ.get("DEV", "0")); N = 4096


class SinCos(nn.Module):
    def forward(self, x):
        return torch.sin(x), torch.cos(x)


cm = rebel.compile_from_torch(SinCos().eval(), input_info=[("x", [1, N], "float32")],
                              example_inputs=[torch.zeros(1, N, dtype=torch.float32)])
rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
torch.manual_seed(0)

print("%-14s %-14s %-14s %-14s %s"
      % ("입력 범위", "vs fp64", "vs fp16가설", "fp16 ulp", "판정"), flush=True)
for hi, tag in [(2 * math.pi, "[0, 2pi)"), (100, "[0, 100)"), (1000, "[0, 1000)"),
                (4096, "[0, 4096)")]:
    x = (torch.rand(1, N) * hi).float()
    s, c = [torch.as_tensor(t).double() for t in rt(x.contiguous())]
    xd = x.double()
    x16 = x.half().double()                       # fp16 으로 잘린 인자
    e64 = max((s - xd.sin()).abs().max().item(), (c - xd.cos()).abs().max().item())
    e16 = max((s - x16.sin()).abs().max().item(), (c - x16.cos()).abs().max().item())
    ulp = float(torch.tensor(hi, dtype=torch.float16).float()
                - torch.nextafter(torch.tensor(hi, dtype=torch.float16),
                                  torch.tensor(0.0, dtype=torch.float16)).float())
    verdict = "fp16 가설 성립" if e16 < e64 / 5 else ("부분적" if e16 < e64 / 1.5 else "가설 기각")
    print("%-14s %-14.3e %-14.3e %-14.4f %s" % (tag, e64, e16, abs(ulp), verdict), flush=True)

print("\n--- fp16 로 정확히 표현되는 값만 넣어보기 ---", flush=True)
for hi, tag in [(2 * math.pi, "[0, 2pi)"), (1000, "[0, 1000)")]:
    x = (torch.rand(1, N) * hi).half().float()    # 입력 자체가 이미 fp16 격자 위
    s, c = [torch.as_tensor(t).double() for t in rt(x.contiguous())]
    xd = x.double()
    e = max((s - xd.sin()).abs().max().item(), (c - xd.cos()).abs().max().item())
    print("%-14s 오차 %.3e  %s" % (tag, e, "-> 인자 절단이 원인" if e < 1e-2 else ""), flush=True)
print("FP16HYPO_DONE", flush=True)

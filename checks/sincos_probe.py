"""RBLN ATOM+ 의 sin/cos 정확도 — 모델 없이 단독 측정.

그래프는 sin(x), cos(x) 두 줄이 전부다. 입력 크기만 바꿔가며 CPU(fp64) 기준과 비교한다.
"""
import os, math, torch, rebel
import torch.nn as nn

DEV = int(os.environ.get("DEV", "0"))
N = 4096
TAU = 2.0 * math.pi


class SinCos(nn.Module):
    def forward(self, x):
        return torch.sin(x), torch.cos(x)


cm = rebel.compile_from_torch(SinCos().eval(), input_info=[("x", [1, N], "float32")],
                              example_inputs=[torch.zeros(1, N, dtype=torch.float32)])
rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
print("compiled: sin(x), cos(x)  input [1,%d] float32\n" % N, flush=True)


def run(x):
    s, c = [torch.as_tensor(t).double() for t in rt(x.contiguous())]
    xd = x.double()
    return (s - xd.sin()).abs(), (c - xd.cos()).abs()


torch.manual_seed(0)

print("--- 1. 입력 크기별 최대 오차 (균등 난수) ---", flush=True)
print("%-22s %-12s %-12s %s" % ("입력 범위", "sin 최대오차", "cos 최대오차", "판정"), flush=True)
for hi, tag in [(TAU, "[0, 2pi)"), (10, "[0, 10)"), (100, "[0, 100)"), (1000, "[0, 1000)"),
                (4096, "[0, 4096)"), (10000, "[0, 10000)"), (100000, "[0, 100000)")]:
    x = (torch.rand(1, N) * hi).float()
    es, ec = run(x)
    m = max(es.max().item(), ec.max().item())
    verdict = "정상" if m < 1e-2 else ("열화" if m < 0.5 else "사용 불가")
    print("%-22s %-12.3e %-12.3e %s" % (tag, es.max(), ec.max(), verdict), flush=True)

print("\n--- 2. 같은 각도인데 크기만 다르게 (범위 축소 문제인지) ---", flush=True)
print("%-22s %-12s %-12s" % ("x", "sin 오차", "cos 오차"), flush=True)
base = 1.234567
for k in (0, 1, 10, 100, 1000):
    x = torch.full((1, N), base + k * TAU, dtype=torch.float32)
    es, ec = run(x)
    print("%-22s %-12.3e %-12.3e" % ("1.2346 + %d x 2pi" % k, es.max(), ec.max()), flush=True)

print("\n--- 3. 오차가 1e-3 을 넘기 시작하는 지점 ---", flush=True)
lo, hi = 1.0, 100000.0
for _ in range(24):
    mid = (lo + hi) / 2
    x = (torch.rand(1, N) * mid).float()
    es, ec = run(x)
    if max(es.max().item(), ec.max().item()) > 1e-3: hi = mid
    else: lo = mid
print("약 |x| = %.0f 부터 오차 > 1e-3" % lo, flush=True)
print("(참고: Qwen3 는 rope_theta=1e6 이라 최고주파 차원의 각도 = position 값)", flush=True)
print("SINCOS_DONE", flush=True)

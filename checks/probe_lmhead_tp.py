"""lm_head 에 tensor_parallel_size 를 걸어본다.

앞선 수동 4 등분은 설계가 틀렸다 — 호스트가 네 카드를 순차로 부르면 각자 1/4 만
읽어도 네 번 기다리므로 총 시간이 안 줄어든다. 드래프터가 3.06 배를 낸 이유는
tensor_parallel_size 를 주면 런타임이 네 카드에 동시 디스패치하기 때문이다.

두 변형을 본다.
  plain  Linear 만. 출력은 로짓 전체 -> 호스트 argmax (현행과 같은 전송량)
  amax   Linear + amax/argmax 를 그래프 안에서. 전송량이 거의 0 이 되지만
         수동 샤딩에서 DEVICE_GRAPH_CONVERSION 이 났으므로 될지 불확실
"""
import glob as _glob
import os
import time

import rebel
import torch
from safetensors import safe_open
from torch import nn

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
DT, DTS = torch.float16, "float16"
B = 16
TPS = [int(v) for v in os.environ.get("TPLIST", "1,2,4").split(",")]
torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))

W = None
for f in sorted(_glob.glob(SRC + "/*.safetensors")):
    with safe_open(f, framework="pt") as h:
        for k in h.keys():
            if k.endswith("model.embed_tokens.weight"):
                W = h.get_tensor(k).to(DT)
                break
    if W is not None:
        break
V, H = W.shape
print("vocab=%d hidden=%d  weight=%.0f MB" % (V, H, V * H * 2 / 1e6), flush=True)


class Plain(nn.Module):
    def __init__(s):
        super().__init__()
        s.lin = nn.Linear(H, V, bias=False)
        with torch.no_grad():
            s.lin.weight = nn.Parameter(W, requires_grad=False)

    def forward(s, x):
        return s.lin(x)


class WithArgmax(nn.Module):
    def __init__(s):
        super().__init__()
        s.lin = nn.Linear(H, V, bias=False)
        with torch.no_grad():
            s.lin.weight = nn.Parameter(W, requires_grad=False)

    def forward(s, x):
        logits = s.lin(x)
        return torch.argmax(logits, dim=-1)


torch.manual_seed(0)
x = torch.randn(1, B, H, dtype=DT)
xn = x.contiguous().numpy()
with torch.no_grad():
    ref = torch.argmax(Plain().eval()(x).float(), dim=-1)
print("reference ready", flush=True)


def bench(tag, cls, tp, host_argmax):
    kw = {"input_info": [("x", [1, B, H], DTS)]}
    if tp > 1:
        kw["tensor_parallel_size"] = tp
    t0 = time.time()
    try:
        cm = rebel.compile_from_torch(cls().eval(), **kw)
        rt = rebel.Runtime(cm, tensor_type="pt", device=(list(range(tp)) if tp > 1 else 0))
    except Exception as exc:  # noqa: BLE001
        print("  %-8s TP%d  COMPILE/RT FAIL  %s: %s"
              % (tag, tp, type(exc).__name__, str(exc)[:180]), flush=True)
        return
    ct = time.time() - t0

    def once():
        o = rt(x.contiguous())
        a = torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o)
        return torch.argmax(a.float(), dim=-1) if host_argmax else a.long()

    for _ in range(3):
        once()
    ts = []
    for _ in range(20):
        s = time.time()
        got = once()
        ts.append((time.time() - s) * 1000)
    ts.sort()
    ok = bool((got.reshape(-1) == ref.reshape(-1)).all())
    print("  %-8s TP%d  compile %3.0fs   %6.2f ms   match=%s"
          % (tag, tp, ct, ts[10], ok), flush=True)
    del rt, cm


print("=== plain (로짓 반환 + 호스트 argmax)", flush=True)
for tp in TPS:
    bench("plain", Plain, tp, True)

print("=== argmax in-graph (전송 거의 0)", flush=True)
for tp in TPS:
    bench("argmax", WithArgmax, tp, False)

print("LMH2_DONE", flush=True)

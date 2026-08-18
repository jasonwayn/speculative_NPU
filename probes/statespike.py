"""상태 유지 그래프 최소 검증: mark_static_address 로 표시한 버퍼가 호출 간 유지되는가."""
import torch, rebel, numpy as np
from torch import nn
from rebel import CompileContext

class Acc(nn.Module):
    """buf 에 x 를 누적. 상태가 유지되면 반환값이 매 호출 증가해야 한다."""
    def forward(s, buf, x):
        buf[:, 0:1, :] = buf[:, 0:1, :] + x
        return buf[:, 0:1, :] + 0.0

N, D = 4, 8
info = [("buf", [1, N, D], "float32"), ("x", [1, 1, D], "float32")]
ex = [torch.zeros(1, N, D), torch.ones(1, 1, D)]
ctx = CompileContext(use_weight_sharing=True)
ctx.mark_static_address(ex[0], "buf")
print("compiling...", flush=True)
cm = rebel.compile_from_torch(Acc().eval(), input_info=info, example_inputs=ex, compile_context=ctx)
rt = cm.create_runtime(device=0)
print("runtime ok", flush=True)

x = np.ones((1, 1, D), dtype=np.float32)
buf = np.zeros((1, N, D), dtype=np.float32)
for form, args in [("rt(x)", (x,)), ("rt(buf,x)", (buf, x))]:
    try:
        vals = [float(np.asarray(rt(*args)).ravel()[0]) for _ in range(4)]
        print("%-12s -> %s   %s" % (form, vals,
              "STATE_PERSISTS" if vals[-1] > vals[0] else "no_persistence"))
        break
    except Exception as e:
        print("%-12s FAIL %s: %s" % (form, type(e).__name__, str(e)[:150]))
print("SPIKE_DONE", flush=True)

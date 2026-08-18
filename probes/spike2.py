"""세그폴트 원인 분리: (a) in-place 쓰기  (b) mark_static_address  (c) 둘 다"""
import sys, torch, rebel, numpy as np
from torch import nn
from rebel import CompileContext
N, D = 4, 8
info = [("buf", [1, N, D], "float32"), ("x", [1, 1, D], "float32")]

class InPlace(nn.Module):
    def forward(s, buf, x):
        buf[:, 0:1, :] = buf[:, 0:1, :] + x
        return buf[:, 0:1, :] + 0.0

class Pure(nn.Module):
    def forward(s, buf, x):
        return buf[:, 0:1, :] + x

case = sys.argv[1]
mod = InPlace() if "inplace" in case else Pure()
ex = [torch.zeros(1, N, D), torch.ones(1, 1, D)]
ctx = None
if "static" in case:
    ctx = CompileContext(use_weight_sharing=True)
    ctx.mark_static_address(ex[0], "buf")
print("case=%s  compiling" % case, flush=True)
cm = rebel.compile_from_torch(mod.eval(), input_info=info, example_inputs=ex, compile_context=ctx)
print("COMPILED", flush=True)
rt = cm.create_runtime(device=0)
print("RUNTIME_OK", flush=True)
x = np.ones((1, 1, D), dtype=np.float32); buf = np.zeros((1, N, D), dtype=np.float32)
for form, args in [("rt(x)", (x,)), ("rt(buf,x)", (buf, x))]:
    try:
        vals = [float(np.asarray(rt(*args)).ravel()[0]) for _ in range(4)]
        print("%s -> %s  %s" % (form, vals, "PERSISTS" if vals[-1] > vals[0] else "no_persist"), flush=True)
        break
    except Exception as e:
        print("%s FAIL %s: %s" % (form, type(e).__name__, str(e)[:120]), flush=True)
print("CASE_DONE", flush=True)

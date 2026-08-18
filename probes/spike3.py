"""compile_from_torch 인자 조합 매트릭스. 어떤 조합이 살아남는지."""
import sys, torch, rebel, numpy as np
from torch import nn
from rebel import CompileContext
N, D = 4, 8
class Acc(nn.Module):
    def forward(s, buf, x):
        buf[:, 0:1, :] = buf[:, 0:1, :] + x
        return buf[:, 0:1, :] + 0.0
class Pure(nn.Module):
    def forward(s, buf, x):
        return buf[:, 0:1, :] + x

mode = sys.argv[1]
mod = (Acc() if "acc" in mode else Pure()).eval()
info = [("buf", [1, N, D], "float32"), ("x", [1, 1, D], "float32")]
ex = [torch.zeros(1, N, D), torch.ones(1, 1, D)]
kw = {}
if "info" in mode: kw["input_info"] = info
if "ex" in mode: kw["example_inputs"] = ex
if "static" in mode:
    c = CompileContext(use_weight_sharing=True)
    c.mark_static_address(ex[0], "buf")
    kw["compile_context"] = c
print("mode=%s kwargs=%s" % (mode, sorted(kw)), flush=True)
cm = rebel.compile_from_torch(mod, **kw)
print("COMPILED", flush=True)
rt = cm.create_runtime(device=0)
print("RUNTIME_OK", flush=True)
x = np.ones((1, 1, D), dtype=np.float32); buf = np.zeros((1, N, D), dtype=np.float32)
for form, args in [("rt(x)", (x,)), ("rt(buf,x)", (buf, x))]:
    try:
        v = [float(np.asarray(rt(*args)).ravel()[0]) for _ in range(4)]
        print("%s -> %s  %s" % (form, v, "PERSISTS" if v[-1] > v[0] else "no_persist"), flush=True)
        break
    except Exception as e:
        print("%s FAIL %s" % (form, str(e)[:110]), flush=True)
print("DONE", flush=True)

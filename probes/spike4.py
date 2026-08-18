"""rbln_cache_update 로 디바이스 상주 캐시가 호출 간 유지되는지 검증.

호출 1: position 0 에 7 기록
호출 2: position 1 에 9 기록하고 캐시 전체 반환
  -> 반환된 캐시의 position 0 이 7 이면 상태 유지, 0 이면 미유지.
"""
import sys, torch, rebel, numpy as np
from torch import nn
from rebel import CompileContext
import optimum.rbln  # 커스텀 op 등록

N, D = 4, 8

class W(nn.Module):
    def forward(s, past_key_values, state, position):
        axis = torch.tensor(1, dtype=torch.int16)
        torch.ops.rbln_custom_ops.rbln_cache_update(past_key_values, state, position, axis)
        return past_key_values + 0.0

info = [("past_key_values", [1, N, D], "float32"),
        ("state", [1, 1, D], "float32"),
        ("position", [1], "int16")]
ex = [torch.zeros(1, N, D), torch.zeros(1, 1, D), torch.zeros(1, dtype=torch.int16)]
ctx = CompileContext(use_weight_sharing=True)
ctx.mark_static_address(ex[0], "past_key_values")
print("compiling", flush=True)
cm = rebel.compile_from_torch(W().eval(), input_info=info, compile_context=ctx)
print("COMPILED", flush=True)
rt = cm.create_runtime(device=0)
print("RUNTIME_OK", flush=True)

def call(*a):
    return np.asarray(rt(*a))

s7 = (np.ones((1, 1, D), np.float32) * 7)
s9 = (np.ones((1, 1, D), np.float32) * 9)
cache0 = np.zeros((1, N, D), np.float32)
print("expects:", getattr(rt, "get_input_info", lambda: "?")() if hasattr(rt,"get_input_info") else "?", flush=True)
for form in ["no_cache_arg", "with_cache_arg"]:
    try:
        if form == "no_cache_arg":
            call(s7, np.array([0],np.int16)); out = call(s9, np.array([1],np.int16))
        else:
            call(cache0, s7, np.array([0],np.int16)); out = call(cache0, s9, np.array([1],np.int16))
        p0 = float(out[0, 0, 0]); p1 = float(out[0, 1, 0])
        print("%-15s pos0=%.1f pos1=%.1f  %s" % (form, p0, p1,
              "STATE_PERSISTS" if p0 == 7.0 else "no_persistence"), flush=True)
        break
    except Exception as e:
        print("%-15s FAIL %s: %s" % (form, type(e).__name__, str(e)[:130]), flush=True)
print("SPIKE4_DONE", flush=True)

"""사용자 모듈에서 paged_causal_attn_prefill + 디바이스 상주 KV 캐시가 되는지 검증.

지속성 판정:
  호출1  seq=0   v=1  -> 캐시 위치 0..15 에 v=1 기록
  호출2  seq=16  v=0  -> 위치 16..31 기록 후 0..31 참조
  캐시가 유지되면 출력 > 0 (호출1의 v=1 이 섞임), 유지 안 되면 출력 ~ 0
"""
import math
import numpy as np
import torch
from torch import nn
import rebel
from rebel import CompileContext
import optimum.rbln  # 커스텀 op 등록

NKV, NH, HD = 8, 32, 128
BLK = 4096          # kvcache_block_size
S = 16              # 한 번에 넣는 토큰 수 (DFlash 블록)
REP = NH // NKV


class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer("scale", torch.tensor(1.0 / math.sqrt(HD)))

    def forward(s, q, k, v, past_key_values_0, past_key_values_1, seq, block_tables):
        q5 = q.view(1, NKV, REP, -1, HD)
        k5 = k.unsqueeze(2)
        v5 = v.unsqueeze(2)
        out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
            q=q5, k=k5, v=v5,
            kcache=past_key_values_0.unsqueeze(2),
            vcache=past_key_values_1.unsqueeze(2),
            seq=seq, scale=s.scale,
            block_table=block_tables, block_size=BLK,
            is_bidirectional=True,
        )
        return out.view(1, NH, -1, HD)


info = [
    ("q", [1, NH, S, HD], "float32"),
    ("k", [1, NKV, S, HD], "float32"),
    ("v", [1, NKV, S, HD], "float32"),
    ("past_key_values_0", [1, NKV, BLK, HD], "float32"),
    ("past_key_values_1", [1, NKV, BLK, HD], "float32"),
    ("seq", [1, 1], "int32"),
    ("block_tables", [1], "int16"),
]
ex = [
    torch.zeros(1, NH, S, HD), torch.zeros(1, NKV, S, HD), torch.zeros(1, NKV, S, HD),
    torch.zeros(1, NKV, BLK, HD), torch.zeros(1, NKV, BLK, HD),
    torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1, dtype=torch.int16),
]
ctx = CompileContext(use_weight_sharing=True)
for (name, _, _), t in zip(info, ex):
    if "past_key_values" in name:
        ctx.mark_static_address(t, name)

print("compiling", flush=True)
cm = rebel.compile_from_torch(Attn().eval(), input_info=info, compile_context=ctx)
print("COMPILED", flush=True)
rt = cm.create_runtime(device=0)
print("RUNTIME_OK", flush=True)

qn = np.ones((1, NH, S, HD), np.float32)
kn = np.ones((1, NKV, S, HD), np.float32)
v1 = np.ones((1, NKV, S, HD), np.float32)
v0 = np.zeros((1, NKV, S, HD), np.float32)
bt = np.zeros(1, np.int16)


kc = np.zeros((1, NKV, BLK, HD), np.float32)
vc = np.zeros((1, NKV, BLK, HD), np.float32)


def run(form, v, pos):
    if form == "short":
        return np.asarray(rt(qn, kn, v, np.array([[pos]], np.int32), bt))
    return np.asarray(rt(qn, kn, v, kc, vc, np.array([[pos]], np.int32), bt))


for form in ["short", "full"]:
    try:
        run(form, v1, 0)
        out = run(form, v0, S)
        mx = float(np.abs(out).max())
        print("%-6s out_absmax=%.6f  ->  %s" % (form, mx,
              "CACHE_PERSISTS" if mx > 1e-3 else "no_persistence"), flush=True)
        break
    except Exception as e:
        print("%-6s FAIL %s: %s" % (form, type(e).__name__, str(e)[:140]), flush=True)
print("SPIKE5_DONE", flush=True)

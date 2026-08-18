"""rebel.Runtime (optimum-rbln 이 쓰는 생성자) 로 static KV 캐시가 유지되는지 검증.

지속성 판정:
  호출1  seq=0   v=1  -> 캐시 0..15 에 v=1
  호출2  seq=16  v=0  -> 16..31 기록 후 0..31 참조
  유지되면 출력 > 0, 아니면 ~0
"""
import math, inspect
import torch
from torch import nn
import rebel
from rebel import CompileContext
import optimum.rbln

NKV, NH, HD = 8, 32, 128
BLK, S = 4096, 16
REP = NH // NKV


class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.register_buffer("scale", torch.tensor(1.0 / math.sqrt(HD)))

    def forward(s, q, k, v, past_key_values_0, past_key_values_1, seq, block_tables):
        out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
            q=q.view(1, NKV, REP, -1, HD), k=k.unsqueeze(2), v=v.unsqueeze(2),
            kcache=past_key_values_0.unsqueeze(2), vcache=past_key_values_1.unsqueeze(2),
            seq=seq, scale=s.scale, block_table=block_tables, block_size=BLK,
            is_bidirectional=True)
        return out.view(1, NH, -1, HD)


info = [("q", [1, NH, S, HD], "float32"),
        ("k", [1, NKV, S, HD], "float32"),
        ("v", [1, NKV, S, HD], "float32"),
        ("past_key_values_0", [1, NKV, BLK, HD], "float32"),
        ("past_key_values_1", [1, NKV, BLK, HD], "float32"),
        ("seq", [1, 1], "int32"),
        ("block_tables", [1], "int16")]
ex = [torch.zeros(*sh) if dt == "float32" else torch.zeros(*sh, dtype=getattr(torch, dt))
      for _, sh, dt in info]
ctx = CompileContext(use_weight_sharing=True)
static = {}
for (name, _, _), t in zip(info, ex):
    if "past_key_values" in name:
        static[name] = t
        ctx.mark_static_address(t, name)

print("Runtime sig:", inspect.signature(rebel.Runtime.__init__), flush=True)
print("compiling", flush=True)
# optimum-rbln 과 동일하게 input_info + example_inputs + compile_context 를 모두 전달
cm = rebel.compile_from_torch(Attn().eval(), input_info=info,
                              example_inputs=ex, compile_context=ctx)
print("COMPILED", flush=True)
rt = rebel.Runtime(cm, tensor_type="pt", device=0)
print("RUNTIME_OK", flush=True)

qn = torch.ones(1, NH, S, HD)
kn = torch.ones(1, NKV, S, HD)
v1 = torch.ones(1, NKV, S, HD)
v0 = torch.zeros(1, NKV, S, HD)
bt = torch.zeros(1, dtype=torch.int16)
kc, vc = static["past_key_values_0"], static["past_key_values_1"]


def go(form, v, pos):
    p = torch.tensor([[pos]], dtype=torch.int32)
    if form == "short":
        return rt(qn, kn, v, p, bt)
    return rt(qn, kn, v, kc, vc, p, bt)


for form in ["short", "full"]:
    try:
        go(form, v1, 0)
        out = go(form, v0, S)
        mx = float(torch.as_tensor(out).abs().max())
        kmax = float(kc.abs().max())
        print("%-6s out_absmax=%.6f  host_kcache_absmax=%.6f  -> %s"
              % (form, mx, kmax, "CACHE_PERSISTS" if mx > 1e-3 else "no_persistence"), flush=True)
        break
    except Exception as e:
        print("%-6s FAIL %s: %s" % (form, type(e).__name__, str(e)[:150]), flush=True)
print("SPIKE6_DONE", flush=True)

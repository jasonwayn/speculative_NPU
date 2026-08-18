"""torch.export.export 가 왜 실패하는지 실제 예외를 본다.

rebel 이 export 실패 시 jit.trace 로 폴백하고, 그러면 mutation 의미가 사라져
static KV 캐시가 유지되지 않는다. 그래서 export 통과 여부가 관건.
"""
import math, traceback
import torch
from torch import nn
import optimum.rbln  # 커스텀 op 등록

NKV, NH, HD = 8, 32, 128
BLK, S = 4096, 16
REP = NH // NKV


class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer("scale", torch.tensor(1.0 / math.sqrt(HD)))

    def forward(s, q, k, v, past_key_values_0, past_key_values_1, seq, block_tables):
        q5 = q.view(1, NKV, REP, -1, HD)
        out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
            q=q5, k=k.unsqueeze(2), v=v.unsqueeze(2),
            kcache=past_key_values_0.unsqueeze(2),
            vcache=past_key_values_1.unsqueeze(2),
            seq=seq, scale=s.scale,
            block_table=block_tables, block_size=BLK,
            is_bidirectional=True,
        )
        return out.view(1, NH, -1, HD)


args = (
    torch.zeros(1, NH, S, HD), torch.zeros(1, NKV, S, HD), torch.zeros(1, NKV, S, HD),
    torch.zeros(1, NKV, BLK, HD), torch.zeros(1, NKV, BLK, HD),
    torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1, dtype=torch.int16),
)

print("torch:", torch.__version__, flush=True)
print("=== 1) torch.export.export on my module ===", flush=True)
try:
    ep = torch.export.export(Attn().eval(), args)
    print("EXPORT_OK", flush=True)
    print(str(ep.graph)[:600], flush=True)
except Exception as e:
    print("EXPORT_FAIL", type(e).__name__, flush=True)
    traceback.print_exc()

print("\n=== 2) 같은 시도, strict=False ===", flush=True)
try:
    ep = torch.export.export(Attn().eval(), args, strict=False)
    print("EXPORT_OK_NONSTRICT", flush=True)
except Exception as e:
    print("NONSTRICT_FAIL", type(e).__name__, str(e)[:300], flush=True)

print("\n=== 3) 참조: optimum-rbln 래퍼는 export 되는가 ===", flush=True)
try:
    from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import AttentionOp
    print("AttentionOp import OK", flush=True)
except Exception as e:
    print("import fail:", str(e)[:200], flush=True)
print("DIAG_DONE", flush=True)

"""export 그래프가 kcache 변형을 '입력 변형'으로 기록하는지 확인.

기록된다면 rebel 이 mark_static_address 와 짝지어 디바이스 버퍼로 되돌릴 수 있다.
기록되지 않으면(순수 함수로 접힘) 캐시는 절대 유지되지 않는다.
"""
import math
import torch
from torch import nn
import optimum.rbln

NKV, NH, HD = 8, 32, 128
BLK, S = 4096, 16
REP = NH // NKV


class ViaView(nn.Module):
    """참조 구현과 동일하게 unsqueeze(2) 뷰를 넘긴다."""
    def __init__(s):
        super().__init__(); s.register_buffer("scale", torch.tensor(1.0 / math.sqrt(HD)))

    def forward(s, q, k, v, past_key_values_0, past_key_values_1, seq, block_tables):
        out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
            q=q.view(1, NKV, REP, -1, HD), k=k.unsqueeze(2), v=v.unsqueeze(2),
            kcache=past_key_values_0.unsqueeze(2), vcache=past_key_values_1.unsqueeze(2),
            seq=seq, scale=s.scale, block_table=block_tables, block_size=BLK,
            is_bidirectional=True)
        return out.view(1, NH, -1, HD)


class Direct(nn.Module):
    """뷰 없이 5D 캐시를 그대로 입력으로 받는다."""
    def __init__(s):
        super().__init__(); s.register_buffer("scale", torch.tensor(1.0 / math.sqrt(HD)))

    def forward(s, q, k, v, past_key_values_0, past_key_values_1, seq, block_tables):
        out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
            q=q.view(1, NKV, REP, -1, HD), k=k.unsqueeze(2), v=v.unsqueeze(2),
            kcache=past_key_values_0, vcache=past_key_values_1,
            seq=seq, scale=s.scale, block_table=block_tables, block_size=BLK,
            is_bidirectional=True)
        return out.view(1, NH, -1, HD)


base = (torch.zeros(1, NH, S, HD), torch.zeros(1, NKV, S, HD), torch.zeros(1, NKV, S, HD))
tail = (torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1, dtype=torch.int16))
cases = [
    ("via_view(4D cache)", ViaView(), base + (torch.zeros(1, NKV, BLK, HD),
                                             torch.zeros(1, NKV, BLK, HD)) + tail),
    ("direct(5D cache)", Direct(), base + (torch.zeros(1, NKV, 1, BLK, HD),
                                           torch.zeros(1, NKV, 1, BLK, HD)) + tail),
]
for name, mod, args in cases:
    print("=== %s ===" % name, flush=True)
    try:
        ep = torch.export.export(mod.eval(), args)
        gs = ep.graph_signature
        mut = getattr(gs, "user_inputs_to_mutate", None)
        buf = getattr(gs, "buffers_to_mutate", None)
        print("  user_inputs_to_mutate =", mut, flush=True)
        print("  buffers_to_mutate     =", buf, flush=True)
        print("  n_outputs             =", len(gs.output_specs), flush=True)
        for o in gs.output_specs:
            print("    out:", o.kind, getattr(o, "target", None), flush=True)
    except Exception as e:
        print("  EXPORT_FAIL %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
print("SIG_DONE", flush=True)

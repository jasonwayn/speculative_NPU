"""draft_two 의 RoPE 수정판 B — 호스트에서 범위 축소만 하고 cos/sin 은 그래프에 남긴다.

수정판 A(draft_two_rope)는 cos/sin 을 통째로 입력으로 받아 정확하지만, Append 그래프가
0.79 -> 6.00 ms 로 느려졌다 (텐서 2개 추가). Block 은 안 변했으므로 고정 비용이 아니라
가벼운 그래프에 특유한 문제다.

여기서는 입력 개수를 그대로 둔다: pos [1,N] int32 -> angle [1,N,hd/2] float32.
NPU sin/cos 는 큰 인자에서만 무너진다 (실측: [0,4096) 오차 1.67, [0,2pi) 오차 5.4e-3).
호스트가 2pi 나머지만 구해 넘기면 오차가 position 과 무관하게 평평해진다.
"""
import math
import torch
from draft_two import _Base, PAGED, rope

TWO_PI = 2.0 * math.pi


def host_angle(pos, inv, dtype):
    """pos [1,N] -> angle [1,N,hd/2], 각 성분이 [0, 2pi) 로 축소됨. fp64 로 계산."""
    f = pos.double().unsqueeze(-1) * inv.double().view(1, 1, -1)
    return torch.remainder(f, TWO_PI).to(dtype)


def _cs(angle):
    e = torch.cat([angle, angle], dim=-1)
    return e.cos(), e.sin()


class AppendRR(_Base):
    def forward(s, th_new, angle_ctx, seq_ctx, block_tables, *caches):
        A = th_new.shape[1]
        th = s.d.hidden_norm(s.d.fc(th_new))
        cc, sc = _cs(angle_ctx)
        q0 = torch.zeros(1, s.nkv, s.rep, A, s.hd, dtype=th.dtype)
        acc = None
        for i, layer in enumerate(s.d.layers):
            at = layer.self_attn
            kc, vc = caches[2 * i], caches[2 * i + 1]
            k = at.k_norm(at.k_proj(th).view(1, A, -1, s.hd)).transpose(1, 2)
            k = rope(k, cc, sc).unsqueeze(2)
            v = at.v_proj(th).view(1, A, -1, s.hd).transpose(1, 2).unsqueeze(2)
            o = PAGED(q=q0, k=k, v=v, kcache=kc.unsqueeze(2), vcache=vc.unsqueeze(2),
                      seq=seq_ctx, scale=s.scale, block_table=block_tables,
                      block_size=s.max_ctx, is_bidirectional=True)
            acc = o if acc is None else acc + o
        return acc


class BlockRR(_Base):
    def forward(s, noise_emb, angle_blk, seq_blk, block_tables, *caches):
        B = noise_emb.shape[1]
        cb, sb = _cs(angle_blk)
        hs = noise_emb
        for i, layer in enumerate(s.d.layers):
            at = layer.self_attn
            kc, vc = caches[2 * i], caches[2 * i + 1]
            resid = hs
            x = layer.input_layernorm(hs)
            q = at.q_norm(at.q_proj(x).view(1, B, -1, s.hd)).transpose(1, 2)
            q = rope(q, cb, sb).view(1, s.nkv, s.rep, B, s.hd)
            k = at.k_norm(at.k_proj(x).view(1, B, -1, s.hd)).transpose(1, 2)
            k = rope(k, cb, sb).unsqueeze(2)
            v = at.v_proj(x).view(1, B, -1, s.hd).transpose(1, 2).unsqueeze(2)
            o = PAGED(q=q, k=k, v=v, kcache=kc.unsqueeze(2), vcache=vc.unsqueeze(2),
                      seq=seq_blk, scale=s.scale, block_table=block_tables,
                      block_size=s.max_ctx, is_bidirectional=True)
            o = o.view(1, s.nh, B, s.hd).transpose(1, 2).reshape(1, B, s.nh * s.hd)
            hs = resid + at.o_proj(o)
            hs = hs + layer.mlp(layer.post_attention_layernorm(hs))
        return s.d.norm(hs)

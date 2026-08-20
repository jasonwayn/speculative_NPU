"""draft_two 의 RoPE 수정판 — cos/sin 을 그래프 안에서 계산하지 않고 입력으로 받는다.

원본 `_Base.cs()` 는 그래프 안에서 `e.cos()` / `e.sin()` 을 부른다. RBLN 은 큰 인자의
삼각함수 정확도가 무너진다 (position 3000 에서 절대오차 약 2.0 = 값이 무관해짐).
Qwen3 는 rope_theta=1e6 이라 최고주파 차원의 inv_freq 가 1 이고, 각도(라디안)가 곧
position 값이 되어 정면으로 걸린다.

optimum-rbln 의 타깃 경로는 rotary_emb 를 traced 영역 밖에서 만들어 넘기기 때문에
멀쩡하다. 드래프터도 같은 방식으로 바꾼다.
"""
import torch
from draft_two import _Base, PAGED, rope


def host_cs(pos, inv, dtype):
    """호스트에서 fp64 로 계산해 넘긴다. pos [1,N] -> cos/sin [1,N,hd]"""
    f = pos.double().unsqueeze(-1) * inv.double().view(1, 1, -1)
    e = torch.cat([f, f], dim=-1)
    return e.cos().to(dtype), e.sin().to(dtype)


class AppendR(_Base):
    """Append 와 동일하되 pos_ctx 대신 cos_ctx/sin_ctx 를 받는다."""

    def forward(s, th_new, cos_ctx, sin_ctx, seq_ctx, block_tables, *caches):
        A = th_new.shape[1]
        th = s.d.hidden_norm(s.d.fc(th_new))
        q0 = torch.zeros(1, s.nkv, s.rep, A, s.hd, dtype=th.dtype)
        acc = None
        for i, layer in enumerate(s.d.layers):
            at = layer.self_attn
            kc, vc = caches[2 * i], caches[2 * i + 1]
            k = at.k_norm(at.k_proj(th).view(1, A, -1, s.hd)).transpose(1, 2)
            k = rope(k, cos_ctx, sin_ctx).unsqueeze(2)
            v = at.v_proj(th).view(1, A, -1, s.hd).transpose(1, 2).unsqueeze(2)
            o = PAGED(q=q0, k=k, v=v, kcache=kc.unsqueeze(2), vcache=vc.unsqueeze(2),
                      seq=seq_ctx, scale=s.scale, block_table=block_tables,
                      block_size=s.max_ctx, is_bidirectional=True)
            acc = o if acc is None else acc + o
        return acc


class BlockR(_Base):
    """Block 과 동일하되 pos_blk 대신 cos_blk/sin_blk 를 받는다."""

    def forward(s, noise_emb, cos_blk, sin_blk, seq_blk, block_tables, *caches):
        B = noise_emb.shape[1]
        hs = noise_emb
        for i, layer in enumerate(s.d.layers):
            at = layer.self_attn
            kc, vc = caches[2 * i], caches[2 * i + 1]
            resid = hs
            x = layer.input_layernorm(hs)
            q = at.q_norm(at.q_proj(x).view(1, B, -1, s.hd)).transpose(1, 2)
            q = rope(q, cos_blk, sin_blk).view(1, s.nkv, s.rep, B, s.hd)
            k = at.k_norm(at.k_proj(x).view(1, B, -1, s.hd)).transpose(1, 2)
            k = rope(k, cos_blk, sin_blk).unsqueeze(2)
            v = at.v_proj(x).view(1, B, -1, s.hd).transpose(1, 2).unsqueeze(2)
            o = PAGED(q=q, k=k, v=v, kcache=kc.unsqueeze(2), vcache=vc.unsqueeze(2),
                      seq=seq_blk, scale=s.scale, block_table=block_tables,
                      block_size=s.max_ctx, is_bidirectional=True)
            o = o.view(1, s.nh, B, s.hd).transpose(1, 2).reshape(1, B, s.nh * s.hd)
            hs = resid + at.o_proj(o)
            hs = hs + layer.mlp(layer.post_attention_layernorm(hs))
        return s.d.norm(hs)

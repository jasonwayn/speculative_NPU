"""DFlash 드래프트 상태 유지 포팅 — 그래프 2개가 캐시를 공유하는 방식.

한 그래프 안에서 PAGED 를 두 번 부르면 export 가 캐시 변형을 기록하지 않아 순서가
보장되지 않는다(실측: cos 0.6). optimum-rbln 도 prefill/decoder 를 따로 컴파일하고
static_tensors 를 공유한다. 같은 방식으로 나눈다.

  Append 그래프 : 새로 채택된 위치의 컨텍스트 K/V 를 캐시에 기록
  Block  그래프 : 캐시를 읽어 블록 어텐션 수행
호스트가 Append -> Block 순으로 호출하므로 순서가 보장된다.
"""
import torch
from torch import nn
import optimum.rbln  # rbln_custom_ops 등록

PAGED = torch.ops.rbln_custom_ops.paged_causal_attn_prefill


def rope(x, cos, sin):
    """x: [1, heads, S, hd],  cos/sin: [1, S, hd]"""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)


class _Base(nn.Module):
    def __init__(s, draft, max_ctx):
        super().__init__()
        s.d = draft
        c = draft.config
        s.nh = c.num_attention_heads
        s.nkv = c.num_key_value_heads
        s.hd = getattr(c, "head_dim", c.hidden_size // s.nh)
        s.rep = s.nh // s.nkv
        s.max_ctx = max_ctx
        s.register_buffer("scale", torch.tensor(s.hd ** -0.5))
        s.register_buffer("inv", 1.0 / (c.rope_theta ** (torch.arange(0, s.hd, 2).float() / s.hd)))

    def cs(s, pos):
        f = pos.float().unsqueeze(-1) * s.inv.view(1, 1, -1)
        e = torch.cat([f, f], dim=-1)
        return e.cos(), e.sin()


class Append(_Base):
    """새 컨텍스트 위치의 K/V 를 각 레이어 캐시에 기록한다.

    PAGED 는 k/v 를 seq 위치에 써 넣는 부수효과가 있으므로 이를 쓰기 수단으로 쓴다.
    어텐션 출력은 쓰지 않지만 DCE 방지를 위해 반환한다(q 는 0).
    """

    def forward(s, th_new, pos_ctx, seq_ctx, block_tables, *caches):
        A = th_new.shape[1]
        th = s.d.hidden_norm(s.d.fc(th_new))
        cc, sc = s.cs(pos_ctx)
        # prefill op 는 q_len == k_len 을 가정하는 것으로 보인다 -> A 길이로 맞춘다
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


class Block(_Base):
    """캐시된 컨텍스트 + 블록으로 드래프트 forward 를 수행한다."""

    def forward(s, noise_emb, pos_blk, seq_blk, block_tables, *caches):
        B = noise_emb.shape[1]
        cb, sb = s.cs(pos_blk)
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

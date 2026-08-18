"""DFlash 드래프트를 디바이스 상주 KV 캐시로 포팅.

현재(무상태): 매 라운드 컨텍스트 전체(C개)를 호스트에서 넘기고 fc/k_proj 를 전부 재계산.
              C=16384 에서 draft 171ms.
이 버전    : 컨텍스트 K/V 를 5개 레이어의 디바이스 캐시에 상주. 라운드마다 새로 채택된
              위치(<=B개)만 넘겨 캐시에 추가. 드래프트 비용이 C 와 무관해진다.

컴파일 인자는 반드시 input_info + example_inputs + compile_context 세 개를 모두 전달해야
static 캐시가 런타임 입력에서 제외되고 호출 간 유지된다 (문서 없음, 실측으로 확인).
"""
import math
import torch
from torch import nn
import optimum.rbln  # rbln_custom_ops 등록

PAGED = torch.ops.rbln_custom_ops.paged_causal_attn_prefill


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, cos, sin):
    """x: [1, heads, S, hd],  cos/sin: [1, S, hd]"""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)


class StatefulDraft(nn.Module):
    """원본 DFlashDraftModel 의 가중치를 재사용하고 forward 만 재작성."""

    def __init__(s, draft, max_ctx: int):
        super().__init__()
        s.d = draft
        cfg = draft.config
        s.nh = cfg.num_attention_heads
        s.nkv = cfg.num_key_value_heads
        s.hd = getattr(cfg, "head_dim", cfg.hidden_size // s.nh)
        s.rep = s.nh // s.nkv
        s.eps = cfg.rms_norm_eps
        s.nl = cfg.num_hidden_layers
        s.max_ctx = max_ctx
        s.register_buffer("scale", torch.tensor(s.hd ** -0.5))
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, s.hd, 2).float() / s.hd))
        s.register_buffer("inv", inv)

    def _cs(s, pos):
        """pos: [1, S] -> cos, sin  [1, S, hd]"""
        f = pos.float().unsqueeze(-1) * s.inv.view(1, 1, -1)
        e = torch.cat([f, f], dim=-1)
        return e.cos(), e.sin()

    def forward(s, noise_emb, th_new, pos_ctx, pos_blk, seq_ctx, seq_blk,
                block_tables, *caches):
        """
        noise_emb  [1, B, H]        블록 마스크 토큰 임베딩
        th_new     [1, A, NT*H]     새로 채택된 위치의 target_hidden (뒤쪽은 무효, seq 로 잘림)
        pos_ctx    [1, A]           그 위치들의 절대 위치
        pos_blk    [1, B]           블록의 절대 위치
        seq_ctx    [1, 1] int32     캐시에 이미 있는 컨텍스트 길이 (쓰기 시작점)
        seq_blk    [1, 1] int32     컨텍스트 갱신 후 길이 (블록 쓰기 시작점)
        caches     2*nl 개 static 텐서 [1, nkv, max_ctx, hd]
        """
        B = noise_emb.shape[1]
        A = th_new.shape[1]
        # 모델 레벨 컨텍스트 투영 (새 위치에만)
        th = s.d.hidden_norm(s.d.fc(th_new))
        cc, sc = s._cs(pos_ctx)
        cb, sb = s._cs(pos_blk)

        hs = noise_emb
        sink = None
        for i, layer in enumerate(s.d.layers):
            at = layer.self_attn
            kc, vc = caches[2 * i], caches[2 * i + 1]
            resid = hs
            x = layer.input_layernorm(hs)

            q = at.q_norm(at.q_proj(x).view(1, B, -1, s.hd)).transpose(1, 2)
            q = rope(q, cb, sb).view(1, s.nkv, s.rep, B, s.hd)

            # (1) 컨텍스트 K/V 를 캐시에 기록. 출력은 쓰지 않지만 DCE 방지를 위해 반환.
            kx = at.k_norm(at.k_proj(th).view(1, A, -1, s.hd)).transpose(1, 2)
            kx = rope(kx, cc, sc).unsqueeze(2)
            vx = at.v_proj(th).view(1, A, -1, s.hd).transpose(1, 2).unsqueeze(2)
            o1 = PAGED(q=q, k=kx, v=vx, kcache=kc.unsqueeze(2), vcache=vc.unsqueeze(2),
                       seq=seq_ctx, scale=s.scale, block_table=block_tables,
                       block_size=s.max_ctx, is_bidirectional=True)
            sink = o1 if sink is None else sink + o1

            # (2) 블록 K/V 기록 + [컨텍스트 | 블록] 어텐션
            # export 가 캐시 변형을 기록하지 않아 두 호출의 순서가 보장되지 않는다.
            # 0 을 곱한 항으로 인위적 데이터 의존성을 만들어 (1) -> (2) 순서를 강제.
            q = q + (o1.sum() * 0.0)
            kb = at.k_norm(at.k_proj(x).view(1, B, -1, s.hd)).transpose(1, 2)
            kb = rope(kb, cb, sb).unsqueeze(2)
            vb = at.v_proj(x).view(1, B, -1, s.hd).transpose(1, 2).unsqueeze(2)
            o = PAGED(q=q, k=kb, v=vb, kcache=kc.unsqueeze(2), vcache=vc.unsqueeze(2),
                      seq=seq_blk, scale=s.scale, block_table=block_tables,
                      block_size=s.max_ctx, is_bidirectional=True)
            o = o.view(1, s.nh, B, s.hd).transpose(1, 2).reshape(1, B, s.nh * s.hd)
            hs = resid + at.o_proj(o)
            hs = hs + layer.mlp(layer.post_attention_layernorm(hs))
        return s.d.norm(hs), sink

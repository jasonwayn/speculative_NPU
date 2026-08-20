"""수정안 검증: RoPE cos/sin 을 호스트에서 계산해 그래프 입력으로 넣는다.

optimum-rbln 의 타깃 래퍼는 rotary_emb 를 traced 영역 밖에서 만들어 넘긴다
(prepare_forward_args 가 rotary_emb 를 반환). 그래서 타깃은 멀쩡하다.
드래프터는 stock HF 구현을 그대로 트레이스해서 NPU 가 sin/cos 를 직접 계산한다.
같은 방식으로 바꾸면 고쳐지는지 본다.
"""
import os, sys, torch, rebel
import torch.nn as nn
D = "/home/work/npu_work/dflash_work"; DRF = os.path.join(D, "Qwen3-4B-DFlash-b16")
DEV = int(os.environ.get("DEV", "0")); DTS = "float32"; DT = torch.float32; B = 16
sys.path.insert(0, os.path.join(D, "dflash"))
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size; NT = len(draft.target_layer_ids)
ROT = draft.rotary_emb
blob = torch.load(os.path.join(D, "diag_draft_input_1024.pt"))
TH = blob["target_hidden"].to(DT); NOI = blob["noise_embedding"].to(DT)


class _Box:
    cos = None
    sin = None


BOX = _Box()


class ConstRope(nn.Module):
    """position_ids 를 무시하고 호스트가 넣어준 값을 그대로 돌려준다."""

    def forward(self, x, position_ids):
        return BOX.cos, BOX.sin


class Fixed(nn.Module):
    """cos/sin 을 직접 받으므로 position_ids 는 그래프에서 안 쓰인다 (DCE 되므로 입력에서 뺀다)."""

    def __init__(s, m): super().__init__(); s.draft = m
    def forward(s, n, t, a, cos, sin):
        BOX.cos, BOX.sin = cos, sin
        p = torch.zeros(1, t.shape[1] + n.shape[1], dtype=torch.int64)
        return s.draft(position_ids=p, attention_mask=a, noise_embedding=n,
                       target_hidden=t, past_key_values=None, use_cache=False, is_causal=False)


class Plain(nn.Module):
    def __init__(s, m): super().__init__(); s.draft = m
    def forward(s, n, t, p, a):
        return s.draft(position_ids=p, attention_mask=a, noise_embedding=n,
                       target_hidden=t, past_key_values=None, use_cache=False, is_causal=False)


C = 256
N = C + B
x0 = torch.zeros(1, 1, 1, dtype=DT)
with torch.no_grad():
    probe_cos, probe_sin = ROT(x0, torch.zeros(1, N, dtype=torch.int64))
RD = probe_cos.shape[-1]
print("rotary output dim = %d" % RD, flush=True)

info_fix = [("noise_embedding", [1, B, H], DTS), ("target_hidden", [1, C, NT * H], DTS),
            ("attention_mask", [1, 1, B, N], DTS),
            ("cos", [1, N, RD], DTS), ("sin", [1, N, RD], DTS)]
ex_fix = [torch.zeros(*s_, dtype=getattr(torch, d)) for _, s_, d in info_fix]

info_plain = [("noise_embedding", [1, B, H], DTS), ("target_hidden", [1, C, NT * H], DTS),
              ("position_ids", [1, N], "int64"), ("attention_mask", [1, 1, B, N], DTS)]
ex_plain = [torch.zeros(*s_, dtype=getattr(torch, d)) for _, s_, d in info_plain]

draft.rotary_emb = ConstRope()
cm_fix = rebel.compile_from_torch(Fixed(draft).eval(), input_info=info_fix, example_inputs=ex_fix)
rt_fix = rebel.Runtime(cm_fix, tensor_type="pt", device=DEV)
draft.rotary_emb = ROT
cm_pl = rebel.compile_from_torch(Plain(draft).eval(), input_info=info_plain, example_inputs=ex_plain)
rt_pl = rebel.Runtime(cm_pl, tensor_type="pt", device=DEV)
print("compiled both", flush=True)


def go(off):
    th = TH[:, :C].contiguous()
    pos = torch.cat([torch.arange(off, off + C, dtype=torch.int64),
                     torch.arange(off + C, off + C + B, dtype=torch.int64)]).unsqueeze(0)
    mask = torch.zeros(1, 1, B, N, dtype=DT)
    with torch.no_grad():
        hc_cos, hc_sin = ROT(x0, pos)                       # 호스트(fp32) 계산
        draft.rotary_emb = ROT
        oc = draft(position_ids=pos, attention_mask=mask, noise_embedding=NOI,
                   target_hidden=th, past_key_values=None, use_cache=False, is_causal=False)
    hc = (oc[0] if isinstance(oc, (list, tuple)) else oc).float()

    o1 = rt_pl(NOI.contiguous(), th, pos.contiguous(), mask.contiguous())
    h1 = torch.as_tensor(o1[0] if isinstance(o1, (list, tuple)) else o1).float()
    o2 = rt_fix(NOI.contiguous(), th, mask.contiguous(),
                hc_cos.to(DT).contiguous(), hc_sin.to(DT).contiguous())
    h2 = torch.as_tensor(o2[0] if isinstance(o2, (list, tuple)) else o2).float()

    def cs(a, b):
        c = torch.nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]),
                                                  b.reshape(-1, b.shape[-1]), dim=-1)
        return float(c.min()), float(c.mean())
    print("pos %-5d..%-5d  기존 cos_min=%.6f mean=%.6f   |   수정 cos_min=%.6f mean=%.6f"
          % (off, off + N, *cs(hc, h1), *cs(hc, h2)), flush=True)


for off in (0, 384, 768, 1024, 3000):
    go(off)
print("ROPEFIX_DONE", flush=True)

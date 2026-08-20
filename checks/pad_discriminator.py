"""K=1040 정상 / K=1041 붕괴 주장의 원인 판별.

주장: RBLN attention 커널이 shape 에 따라 정확도가 무너진다 (K=1041 에서).
검증 대상: 그 결론이 성립하려면 "패딩 열이 실제로 마스킹되고 있다" 가 참이어야 한다.
원 진단은 마스크 값(-100/-1e4/-1e9/finfo.min)을 바꿔도 결과가 같다는 것으로
마스크 구성을 배제했는데, **네 값 모두 exp() 후 0 이라 마스크가 적용돼도 무시돼도
결과가 같다.** 그래서 그 실험은 판별력이 없다.

여기서는 마스크 값이 아니라 **마스킹된 위치의 내용**을 바꾼다.

  A  ctx1024 그래프, K=1040                          기준
  B  ctx1025 그래프, 패딩 hidden = 0,   마스킹        원 진단의 "붕괴" 케이스
  C  ctx1025 그래프, 패딩 hidden = 큰 난수, 마스킹    B 와 같아야 정상
  D  ctx1025 그래프, 패딩 hidden = 0,   마스킹,
     패딩 position_id 를 블록과 겹치지 않게 변경       위치 충돌 배제

판정
  C != B            -> 마스크가 새고 있다. 커널 정밀도 문제가 아니라 마스킹 문제.
  C == B, D ~= A    -> position_ids 충돌이 원인 (원 진단의 1025 케이스는 pad 의
                       position 1024 가 블록 첫 토큰과 겹친다).
  C == B, D != A    -> shape 의존 정밀도 문제. 원 진단이 맞다.
"""
import os, time, copy, torch, rebel
import torch.nn as nn

D = "/home/work/npu_work/dflash_work"
DRF = os.environ.get("DRF", os.path.join(D, "Qwen3-4B-DFlash-b16"))
DEV = int(os.environ.get("DEV", "0"))
DT = torch.float32
DTS = "float32"
B = 16

from transformers import AutoConfig
import sys
sys.path.insert(0, os.path.join(D, "dflash"))
from dflash.model import DFlashDraftModel

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True)
cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size
NT = len(draft.target_layer_ids)
print("draft loaded H=%d NT=%d" % (H, NT), flush=True)

blob = torch.load(os.path.join(D, "diag_draft_input_1024.pt"))
TH = blob["target_hidden"].to(DT)          # [1, 1024, NT*H]
NOI = blob["noise_embedding"].to(DT)       # [1, 16, H]
TL = TH.shape[1]
print("loaded target_hidden %s  noise %s" % (tuple(TH.shape), tuple(NOI.shape)), flush=True)


class Stateless(nn.Module):
    def __init__(self, m):
        super().__init__(); self.draft = m

    def forward(self, noise_embedding, target_hidden, position_ids, attention_mask):
        return self.draft(position_ids=position_ids, attention_mask=attention_mask,
                          noise_embedding=noise_embedding, target_hidden=target_hidden,
                          past_key_values=None, use_cache=False, is_causal=False)


def info(C):
    return [("noise_embedding", [1, B, H], DTS),
            ("target_hidden", [1, C, NT * H], DTS),
            ("position_ids", [1, C + B], "int64"),
            ("attention_mask", [1, 1, B, C + B], DTS)]


def ex(C):
    return [torch.zeros(1, B, H, dtype=DT), torch.zeros(1, C, NT * H, dtype=DT),
            torch.zeros(1, C + B, dtype=torch.int64),
            torch.zeros(1, 1, B, C + B, dtype=DT)]


rt = {}
for C in (1024, 1025):
    t = time.time()
    cm = rebel.compile_from_torch(Stateless(draft).eval(), input_info=info(C),
                                  example_inputs=ex(C))
    rt[C] = rebel.Runtime(cm, tensor_type="pt", device=DEV)
    print("compiled C=%d  %.0fs" % (C, time.time() - t), flush=True)


def build(C, pad_fill, pad_pos):
    """C=1024 면 패딩 없음. C=1025 면 마지막 열이 패딩(마스킹)."""
    th = torch.zeros(1, C, NT * H, dtype=DT)
    th[:, :TL] = TH
    if C > TL:
        th[:, TL:] = pad_fill
    pos = torch.cat([torch.arange(C, dtype=torch.int64),
                     torch.arange(TL, TL + B, dtype=torch.int64)]).unsqueeze(0)
    if C > TL and pad_pos is not None:
        pos[0, TL:C] = pad_pos
    mask = torch.zeros(1, 1, B, C + B, dtype=DT)
    if C > TL:
        mask[:, :, :, TL:C] = torch.finfo(DT).min
    return NOI.contiguous(), th, pos, mask


def npu(C, *a):
    o = rt[C](*[x.contiguous() for x in a])
    return torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o).float()


def cpu(*a):
    with torch.no_grad():
        o = draft(position_ids=a[2], attention_mask=a[3], noise_embedding=a[0],
                  target_hidden=a[1], past_key_values=None, use_cache=False,
                  is_causal=False)
    return (o[0] if isinstance(o, (list, tuple)) else o).float()


def cos(x, y):
    c = torch.nn.functional.cosine_similarity(x.reshape(-1, x.shape[-1]),
                                              y.reshape(-1, y.shape[-1]), dim=-1)
    return float(c.min()), float(c.mean())


G = torch.randn(1, 1, NT * H, dtype=DT) * 50.0    # 마스킹된 자리에 넣을 큰 난수

cases = {
    "A_1040":            (1024, None, None),
    "B_1041_zero":       (1025, torch.zeros(1, 1, NT * H, dtype=DT), None),
    "C_1041_garbage":    (1025, G, None),
    "D_1041_zero_pos99": (1025, torch.zeros(1, 1, NT * H, dtype=DT), 4000),
}

out_npu, out_cpu = {}, {}
for name, (C, fill, pp) in cases.items():
    a = build(C, fill, pp)
    out_npu[name] = npu(C, *a)
    out_cpu[name] = cpu(*a)
    mn, mean = cos(out_cpu[name], out_npu[name])
    print("%-20s NPU-vs-CPU cos_min=%.6f cos_mean=%.6f" % (name, mn, mean), flush=True)

print("", flush=True)
base = "A_1040"
for name in cases:
    if name == base:
        continue
    mn_n, _ = cos(out_npu[base], out_npu[name])
    mn_c, _ = cos(out_cpu[base], out_cpu[name])
    print("%-20s vs A : NPU cos_min=%.6f   CPU cos_min=%.6f" % (name, mn_n, mn_c), flush=True)

mn_bc, _ = cos(out_npu["B_1041_zero"], out_npu["C_1041_garbage"])
print("", flush=True)
print("MASK_LEAK_TEST  B vs C on NPU cos_min=%.6f  -> %s"
      % (mn_bc, "마스크 정상 (내용 무관)" if mn_bc > 0.9999 else "마스크가 샌다"), flush=True)
mn_bc_cpu, _ = cos(out_cpu["B_1041_zero"], out_cpu["C_1041_garbage"])
print("MASK_LEAK_TEST  B vs C on CPU cos_min=%.6f" % mn_bc_cpu, flush=True)
print("PADDISC_DONE", flush=True)

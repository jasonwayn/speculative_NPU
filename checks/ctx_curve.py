"""드래프트 NPU-vs-CPU 오차의 실제 컨텍스트 의존성 (통제 실험).

원 진단의 스윕은 길이마다 프롬프트가 달라 통제가 안 됐다. 여기서는 같은
target_hidden 을 앞에서부터 잘라 쓰므로 길이만 변한다.

같이 재는 것:
  - 위치별 cos (블록 16칸 중 어디가 나쁜가)
  - 레이어별 누적 (draft 5개 레이어를 통과하며 커지는가)
  - dtype (float32 로 컴파일한 그래프 vs float16)
"""
import os, time, torch, rebel
import torch.nn as nn

D = "/home/work/npu_work/dflash_work"
DRF = os.path.join(D, "Qwen3-4B-DFlash-b16")
DEV = int(os.environ.get("DEV", "0"))
DTS = os.environ.get("DTYPE", "float32")
DT = getattr(torch, DTS)
B = 16
LENS = [int(x) for x in os.environ.get("LENS", "64,128,256,512,768,1024").split(",")]

import sys
sys.path.insert(0, os.path.join(D, "dflash"))
from transformers import AutoConfig
from dflash.model import DFlashDraftModel

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True)
cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size
NT = len(draft.target_layer_ids)

blob = torch.load(os.path.join(D, "diag_draft_input_1024.pt"))
TH_FULL = blob["target_hidden"].to(DT)
NOI = blob["noise_embedding"].to(DT)
print("dtype=%s  target_hidden=%s" % (DTS, tuple(TH_FULL.shape)), flush=True)


class Stateless(nn.Module):
    """레이어별 출력도 같이 내보낸다 (오차가 어디서 생기는지 보려고)."""

    def __init__(self, m):
        super().__init__(); self.draft = m

    def forward(self, noise_embedding, target_hidden, position_ids, attention_mask):
        return self.draft(position_ids=position_ids, attention_mask=attention_mask,
                          noise_embedding=noise_embedding, target_hidden=target_hidden,
                          past_key_values=None, use_cache=False, is_causal=False)


def cosstat(x, y):
    c = torch.nn.functional.cosine_similarity(x.reshape(-1, x.shape[-1]).float(),
                                              y.reshape(-1, y.shape[-1]).float(), dim=-1)
    return c


rows = []
for C in LENS:
    th = TH_FULL[:, :C].contiguous()
    pos = torch.cat([torch.arange(C, dtype=torch.int64),
                     torch.arange(C, C + B, dtype=torch.int64)]).unsqueeze(0)
    mask = torch.zeros(1, 1, B, C + B, dtype=DT)
    info = [("noise_embedding", [1, B, H], DTS), ("target_hidden", [1, C, NT * H], DTS),
            ("position_ids", [1, C + B], "int64"), ("attention_mask", [1, 1, B, C + B], DTS)]
    ex = [torch.zeros(*s, dtype=getattr(torch, d)) for _, s, d in info]
    t = time.time()
    cm = rebel.compile_from_torch(Stateless(draft).eval(), input_info=info, example_inputs=ex)
    rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
    o = rt(NOI.contiguous(), th, pos.contiguous(), mask.contiguous())
    hn = torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o).float()
    with torch.no_grad():
        oc = draft(position_ids=pos, attention_mask=mask, noise_embedding=NOI,
                   target_hidden=th, past_key_values=None, use_cache=False, is_causal=False)
    hc = (oc[0] if isinstance(oc, (list, tuple)) else oc).float()
    c = cosstat(hc, hn)
    rows.append((C, float(c.min()), float(c.mean())))
    print("CTX %-5d cos_min=%.6f cos_mean=%.6f  compile=%.0fs" % (C, c.min(), c.mean(),
                                                                  time.time() - t), flush=True)
    if C == LENS[-1]:
        print("  per-position cos: " + " ".join("%.3f" % v for v in c.tolist()), flush=True)
    del rt, cm

print("", flush=True)
print("CURVE " + " ".join("%d:%.4f" % (a, b) for a, b, _ in rows), flush=True)
print("CTXCURVE_DONE", flush=True)

"""오차가 컨텍스트 '길이'(K) 때문인가, 위치 '값'(RoPE 각도) 때문인가.

길이를 256 으로 고정하고 position_ids 만 0 / 384 / 768 로 옮긴다.
  - 오프셋을 옮겨도 cos 가 그대로면 -> K(리덕션 길이)가 원인
  - 오프셋이 커질수록 cos 가 떨어지면 -> RoPE 위치 값이 원인 (훨씬 고치기 쉬움)
비교 대조로 길이 1024/오프셋 0 도 같이 잰다.
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
blob = torch.load(os.path.join(D, "diag_draft_input_1024.pt"))
TH = blob["target_hidden"].to(DT); NOI = blob["noise_embedding"].to(DT)

class S(nn.Module):
    def __init__(s, m): super().__init__(); s.draft = m
    def forward(s, n, t, p, a):
        return s.draft(position_ids=p, attention_mask=a, noise_embedding=n,
                       target_hidden=t, past_key_values=None, use_cache=False, is_causal=False)

def build_rt(C):
    info = [("noise_embedding", [1, B, H], DTS), ("target_hidden", [1, C, NT*H], DTS),
            ("position_ids", [1, C+B], "int64"), ("attention_mask", [1, 1, B, C+B], DTS)]
    ex = [torch.zeros(*s_, dtype=getattr(torch, d)) for _, s_, d in info]
    cm = rebel.compile_from_torch(S(draft).eval(), input_info=info, example_inputs=ex)
    return rebel.Runtime(cm, tensor_type="pt", device=DEV)

def measure(rt, C, th, off, tag):
    pos = torch.cat([torch.arange(off, off + C, dtype=torch.int64),
                     torch.arange(off + C, off + C + B, dtype=torch.int64)]).unsqueeze(0)
    mask = torch.zeros(1, 1, B, C + B, dtype=DT)
    o = rt(NOI.contiguous(), th.contiguous(), pos.contiguous(), mask.contiguous())
    hn = torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o).float()
    with torch.no_grad():
        oc = draft(position_ids=pos, attention_mask=mask, noise_embedding=NOI,
                   target_hidden=th, past_key_values=None, use_cache=False, is_causal=False)
    hc = (oc[0] if isinstance(oc, (list, tuple)) else oc).float()
    c = torch.nn.functional.cosine_similarity(hc.reshape(-1, hc.shape[-1]),
                                              hn.reshape(-1, hn.shape[-1]), dim=-1)
    print("%-28s K=%-5d pos=%4d..%-5d cos_min=%.6f cos_mean=%.6f"
          % (tag, C + B, off, off + C + B, c.min(), c.mean()), flush=True)

rt256 = build_rt(256)
for off in (0, 384, 768, 3000):
    measure(rt256, 256, TH[:, :256], off, "len256 (앞 256 토큰 고정)")
print("", flush=True)
# 내용까지 그 위치의 것으로 바꿔서 한 번 더 (내용+위치 동시)
measure(rt256, 256, TH[:, 768:1024], 768, "len256 (뒤 256 토큰 내용)")
print("", flush=True)
del rt256
rt1024 = build_rt(1024)
measure(rt1024, 1024, TH[:, :1024], 0, "len1024 대조군")
print("POSLEN_DONE", flush=True)

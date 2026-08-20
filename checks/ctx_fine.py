"""768~1024 구간 세분 + 내용 vs 길이 판별."""
import os, sys, time, torch, rebel
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

def run(C, th, tag):
    pos = torch.cat([torch.arange(C, dtype=torch.int64),
                     torch.arange(C, C + B, dtype=torch.int64)]).unsqueeze(0)
    mask = torch.zeros(1, 1, B, C + B, dtype=DT)
    info = [("noise_embedding", [1, B, H], DTS), ("target_hidden", [1, C, NT*H], DTS),
            ("position_ids", [1, C+B], "int64"), ("attention_mask", [1, 1, B, C+B], DTS)]
    ex = [torch.zeros(*s_, dtype=getattr(torch, d)) for _, s_, d in info]
    cm = rebel.compile_from_torch(S(draft).eval(), input_info=info, example_inputs=ex)
    rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
    o = rt(NOI.contiguous(), th.contiguous(), pos.contiguous(), mask.contiguous())
    hn = torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o).float()
    with torch.no_grad():
        oc = draft(position_ids=pos, attention_mask=mask, noise_embedding=NOI,
                   target_hidden=th, past_key_values=None, use_cache=False, is_causal=False)
    hc = (oc[0] if isinstance(oc, (list, tuple)) else oc).float()
    c = torch.nn.functional.cosine_similarity(hc.reshape(-1, hc.shape[-1]),
                                              hn.reshape(-1, hn.shape[-1]), dim=-1)
    print("%-22s C=%-5d cos_min=%.6f cos_mean=%.6f" % (tag, C, c.min(), c.mean()), flush=True)
    del rt, cm
    return float(c.min())

for C in [800, 832, 896, 960, 992, 1008, 1016, 1020, 1024]:
    run(C, TH[:, :C], "len")
print("", flush=True)
run(768, TH[:, :768], "content_head")
run(768, TH[:, 256:1024], "content_tail")
print("CTXFINE_DONE", flush=True)

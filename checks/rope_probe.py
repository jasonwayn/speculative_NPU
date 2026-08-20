"""드래프터 RoPE 를 단독으로 NPU 에 올려 CPU 와 비교.

Qwen3 는 rope_theta=1e6. 최고주파 차원의 inv_freq 는 1 이므로 각도 = position(라디안).
position 3000 이면 3000 라디안의 sin/cos 를 구해야 하는데, 범위 축소(range reduction)를
제대로 안 하거나 내부 정밀도가 낮으면 여기서 무너진다.
"""
import os, sys, torch, rebel
import torch.nn as nn
D = "/home/work/npu_work/dflash_work"; DRF = os.path.join(D, "Qwen3-4B-DFlash-b16")
DEV = int(os.environ.get("DEV", "0")); DT = torch.float32
sys.path.insert(0, os.path.join(D, "dflash"))
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation = "eager"
m = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
rot = m.rotary_emb
H = cfg.hidden_size

class Rope(nn.Module):
    """x 는 dtype/device 용도로만 쓰이므로 그래프 입력에서 빼면 DCE 로 잘린다."""

    def __init__(s, r): super().__init__(); s.r = r
    def forward(s, position_ids):
        x = torch.zeros(1, 1, 1, dtype=torch.float32)
        cos, sin = s.r(x, position_ids)
        return cos, sin

N = 272
info = [("position_ids", [1, N], "int64")]
ex = [torch.zeros(1, N, dtype=torch.int64)]
cm = rebel.compile_from_torch(Rope(rot).eval(), input_info=info, example_inputs=ex)
rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
x = torch.zeros(1, N, H, dtype=DT)

print("%-8s %-12s %-12s %-12s %-12s" % ("offset", "cos_maxabs", "sin_maxabs", "cos_cos", "sin_cos"), flush=True)
for off in (0, 128, 384, 768, 1024, 2048, 3000, 4000):
    pos = torch.arange(off, off + N, dtype=torch.int64).unsqueeze(0)
    o = rt(pos.contiguous())
    cn, sn = [torch.as_tensor(t).float() for t in o]
    with torch.no_grad():
        cc, sc = rot(x, pos)
    cc = cc.float(); sc = sc.float()
    def cs(a, b):
        return float(torch.nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]),
                                                           b.reshape(-1, b.shape[-1]), dim=-1).min())
    print("%-8d %-12.3e %-12.3e %-12.8f %-12.8f"
          % (off, (cc - cn).abs().max(), (sc - sn).abs().max(), cs(cc, cn), cs(sc, sn)), flush=True)
print("ROPE_DONE", flush=True)

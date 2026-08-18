"""lm_head 비용 분해: 디바이스 연산+전송 vs 호스트 argmax.

bench2.py 는 로짓 전체(151936 x 16)를 호스트로 가져와 fp32 로 변환 후 argmax 한다.
그 호스트 시간은 T["lmh"] 에 안 잡혀 있어 라운드 비용에서 누락된다. 얼마인지 잰다.
"""
import os, sys, time, torch, rebel
sys.path.insert(0, "/home/work/npu_work/dflash_work")
from torch import nn
from safetensors import safe_open
import glob as _glob

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
DT = torch.float16; DTS = "float16"; B = 16
DEV = int(os.environ.get("DEV", "1"))
NT = int(os.environ.get("NTHREADS", "2"))
torch.set_num_threads(NT)

_W = None
for f in sorted(_glob.glob(SRC + "/*.safetensors")):
    with safe_open(f, framework="pt") as h:
        for k in h.keys():
            if k.endswith("model.embed_tokens.weight"):
                _W = h.get_tensor(k).to(DT); break
    if _W is not None: break
V, H = _W.shape
print("vocab=%d hidden=%d threads=%d" % (V, H, NT), flush=True)
lmw = nn.Linear(H, V, bias=False).to(DT)
with torch.no_grad(): lmw.weight = nn.Parameter(_W, requires_grad=False)
lmw = lmw.eval()

CACHE = "/home/work/npu_work/dflash_work/rbln_cache"
p = os.path.join(CACHE, "lmhead_B%d_%s.rbln" % (B, DTS))
cm = rebel.RBLNCompiledModel(p) if os.path.exists(p) else rebel.compile_from_torch(
    nn.Sequential(lmw).eval(), input_info=[("x", [1, B, H], DTS)])
rt = cm.create_runtime(device=DEV)
x = torch.zeros(1, B, H, dtype=DT).contiguous().numpy()

for _ in range(3):
    o = torch.as_tensor(rt(x)); torch.argmax(o.float(), dim=-1)

dev, host = [], []
for _ in range(20):
    s = time.time(); o = torch.as_tensor(rt(x)); dev.append((time.time() - s) * 1000)
    s = time.time(); torch.argmax(o[:, :B].float(), dim=-1); host.append((time.time() - s) * 1000)
dev.sort(); host.sort()
logits_mb = B * V * 2 / 1e6
print("logits payload  %.1f MB" % logits_mb, flush=True)
print("device call     %6.2f ms   (연산 + 호스트로 전송)" % dev[10], flush=True)
print("host argmax     %6.2f ms   (fp16->fp32 변환 포함, 측정에서 누락돼 있던 부분)" % host[10], flush=True)
print("per lm_head     %6.2f ms   -> 라운드당 2회 = %.2f ms" % (dev[10] + host[10], 2 * (dev[10] + host[10])), flush=True)

# fp32 변환 없이 fp16 로 argmax 하면?
s = time.time()
for _ in range(20): torch.argmax(o[:, :B], dim=-1)
h16 = (time.time() - s) / 20 * 1000
print("host argmax fp16 %6.2f ms  (float() 생략 시)" % h16, flush=True)
print("AUDIT_DONE", flush=True)

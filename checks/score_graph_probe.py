"""A' 타당성 확인: 레이어 35 점수 계산을 NPU 그래프에 넣을 수 있는가.

타깃 전체를 재컴파일하기 전에 위험한 부분만 격리해서 본다.
  1. q @ K^T + softmax 를 S=20480 에 대해 컴파일할 수 있는가
  2. 마스킹을 그래프 안 arange 비교로 할 수 있는가 (안 되면 입력으로 받는다)
  3. 청크 평균까지 접어 [1, S/32] 로 내보낼 때 정확도와 지연
"""
import os, time, math, torch, rebel
import torch.nn as nn

S = int(os.environ.get("S", "20480"))       # KV 블록 크기 = MAXC
L = int(os.environ.get("L", "16"))          # verify 블록
NH, NKV, HD = 32, 8, 128
REP = NH // NKV
CH = 32                                     # CMR 청크
DEV = int(os.environ.get("DEV", "0"))
SCALE = 1.0 / math.sqrt(HD)


def core(q, K, add_mask):
    """q [1,NH,L,HD]  K [1,NKV,S,HD]  add_mask [1,1,1,1,S] -> [1, S/CH]"""
    qg = q.view(1, NKV, REP, L, HD)
    kb = K.view(1, NKV, 1, S, HD)
    a = torch.matmul(qg, kb.transpose(-1, -2)) * SCALE + add_mask
    p = torch.softmax(a, dim=-1)                 # [1,NKV,REP,L,S]
    sc = p.mean(dim=(1, 2, 3))                   # [1,S]
    return sc.view(1, S // CH, CH).mean(-1)      # [1, S/CH]


class MaskIn(nn.Module):
    """마스크를 호스트에서 만들어 입력으로 받는다 (안전한 쪽)."""
    def forward(self, q, K, add_mask):
        return core(q, K, add_mask)


class MaskGraph(nn.Module):
    """마스크를 그래프 안에서 만든다 (되면 입력 하나 절약)."""
    def forward(self, q, K, seq_pos):
        idx = torch.arange(S, dtype=torch.int32).view(1, 1, 1, 1, S)
        m = torch.where(idx < seq_pos.view(1, 1, 1, 1, 1),
                        torch.zeros((), dtype=q.dtype),
                        torch.full((), -1e4, dtype=q.dtype))
        return core(q, K, m)


torch.manual_seed(0)
q = torch.randn(1, NH, L, HD)
K = torch.randn(1, NKV, S, HD)
NPOS = S // 2
am = torch.where(torch.arange(S) < NPOS, 0.0, -1e4).view(1, 1, 1, 1, S)
sp = torch.tensor([NPOS], dtype=torch.int32)

ref = MaskIn()(q, K, am)
print("S=%d L=%d  ref [1,%d]  sum=%.6f" % (S, L, S // CH, float(ref.sum())), flush=True)

cases = [
    ("mask_input", MaskIn(),
     [("q", [1, NH, L, HD], "float32"), ("K", [1, NKV, S, HD], "float32"),
      ("add_mask", [1, 1, 1, 1, S], "float32")], (q, K, am)),
    ("mask_in_graph", MaskGraph(),
     [("q", [1, NH, L, HD], "float32"), ("K", [1, NKV, S, HD], "float32"),
      ("seq_pos", [1], "int32")], (q, K, sp)),
]

for name, mod, info, args in cases:
    t0 = time.time()
    try:
        ex = [torch.zeros(*s, dtype=getattr(torch, d)) for _, s, d in info]
        cm = rebel.compile_from_torch(mod.eval(), input_info=info, example_inputs=ex)
        rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
        ct = time.time() - t0
    except Exception as e:
        print("%-14s COMPILE FAIL after %.0fs: %s" % (name, time.time() - t0, repr(e)[:300]), flush=True)
        continue
    o = rt(*[a.contiguous() for a in args])
    out = torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o).float()
    err = float((out - ref).abs().max())
    rel = err / float(ref.abs().max())
    for _ in range(3): rt(*[a.contiguous() for a in args])
    t = time.time()
    for _ in range(10): rt(*[a.contiguous() for a in args])
    ms = 1000 * (time.time() - t) / 10
    print("%-14s compile=%.0fs  run=%.2f ms  max_err=%.3e (rel %.2e)"
          % (name, ct, ms, err, rel), flush=True)
    del rt, cm

print("PROBE_DONE", flush=True)

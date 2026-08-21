"""probe 1 의 8.7 ms 는 K 업로드(84 MB)였다. 실제로는 K 가 타깃 KV 캐시라
디바이스에 상주한다. K 를 buffer 로 넣어 연산 비용만 잰다.

같이 보는 것: fp16 캐시, 쿼리 수 L, 그리고 컨텍스트를 다 훑을 필요가 있는지.
"""
import os, time, math, torch, rebel
import torch.nn as nn

NH, NKV, HD = 32, 8, 128
REP = NH // NKV
CH = 32
DEV = int(os.environ.get("DEV", "0"))
SCALE = 1.0 / math.sqrt(HD)


class Score(nn.Module):
    def __init__(s, K, S, L):
        super().__init__()
        s.register_buffer("K", K)          # 상주 (KV 캐시 대역)
        s.S, s.L = S, L
    def forward(s, q, add_mask):
        qg = q.view(1, NKV, REP, s.L, HD)
        kb = s.K.view(1, NKV, 1, s.S, HD).to(q.dtype)
        a = torch.matmul(qg, kb.transpose(-1, -2)) * SCALE + add_mask
        p = torch.softmax(a, dim=-1)
        sc = p.mean(dim=(1, 2, 3))
        return sc.view(1, s.S // CH, CH).mean(-1)


print("%-24s %10s %10s %12s" % ("case", "compile", "run ms", "max_err"), flush=True)
torch.manual_seed(0)

for S, L, kdt in [(20480, 16, torch.float32), (20480, 16, torch.float16),
                  (20480, 4, torch.float32), (20480, 1, torch.float32),
                  (8192, 16, torch.float32), (4096, 16, torch.float32)]:
    q = torch.randn(1, NH, L, HD)
    K = torch.randn(1, NKV, S, HD).to(kdt)
    am = torch.where(torch.arange(S) < S // 2, 0.0, -1e4).view(1, 1, 1, 1, S)
    mod = Score(K, S, L).eval()
    with torch.no_grad():
        ref = mod(q, am)
    info = [("q", [1, NH, L, HD], "float32"), ("add_mask", [1, 1, 1, 1, S], "float32")]
    t0 = time.time()
    try:
        cm = rebel.compile_from_torch(mod, input_info=info,
                                      example_inputs=[torch.zeros(*s, dtype=getattr(torch, d))
                                                      for _, s, d in info])
        rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
    except Exception as e:
        print("%-24s FAIL %s" % ("S%d L%d %s" % (S, L, kdt), repr(e)[:160]), flush=True); continue
    ct = time.time() - t0
    o = rt(q.contiguous(), am.contiguous())
    out = torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o).float()
    for _ in range(3): rt(q.contiguous(), am.contiguous())
    t = time.time()
    for _ in range(20): rt(q.contiguous(), am.contiguous())
    ms = 1000 * (time.time() - t) / 20
    print("%-24s %8.0fs %10.2f %12.2e"
          % ("S%d L%d %s" % (S, L, str(kdt)[6:]), ct, ms, float((out - ref).abs().max())), flush=True)
    del rt, cm

print("PROBE2_DONE", flush=True)

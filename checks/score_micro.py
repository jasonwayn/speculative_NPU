"""Scorer.scores 의 GQA 확장 비용 격리 측정.

현행은 K 를 repeat_interleave 로 4배 물리 복사한다 (16K 에서 67 -> 268 MB).
einsum 첨자/배치 matmul 로 바꾸면 그 복사가 사라지는지, 결과가 같은지 본다.
"""
import os, time, math, torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
NH, NKV, HD = 32, 8, 128
REP = NH // NKV
M = int(os.environ.get("M", "3"))          # 라운드당 수락 토큰 수 (tau ~2.4)
REPEAT = int(os.environ.get("REPEAT", "5"))


def v1(q, K):                               # 현행
    Kr = K.repeat_interleave(REP, dim=1)
    a = torch.einsum("mhd,nhd->hmn", q, Kr) / math.sqrt(HD)
    return torch.softmax(a, dim=-1).mean(dim=(0, 1))


def v2(q, K):                               # 배치 matmul, 확장 없음
    m = q.shape[0]
    qg = q.view(m, NKV, REP, HD).permute(1, 2, 0, 3).reshape(NKV, REP * m, HD)
    Kg = K.permute(1, 2, 0)                 # [nkv, hd, N] view
    a = torch.matmul(qg, Kg) / math.sqrt(HD)
    return torch.softmax(a, dim=-1).mean(dim=(0, 1))


def v3(q, K):                               # einsum 첨자로만
    m = q.shape[0]
    a = torch.einsum("mgrd,ngd->grmn", q.view(m, NKV, REP, HD), K) / math.sqrt(HD)
    return torch.softmax(a, dim=-1).mean(dim=(0, 1, 2))


print("threads=%d  OMP_WAIT_POLICY=%s  M=%d"
      % (torch.get_num_threads(), os.environ.get("OMP_WAIT_POLICY", "-"), M), flush=True)
print("%8s %10s %10s %10s %9s %9s" % ("N", "v1 ms", "v2 ms", "v3 ms", "v2 err", "v3 err"), flush=True)

for N in (1024, 4096, 8192, 16384):
    q = torch.randn(M, NH, HD)
    K = torch.randn(N, NKV, HD)
    r1 = v1(q, K); r2 = v2(q, K); r3 = v3(q, K)
    e2 = float((r1 - r2).abs().max()); e3 = float((r1 - r3).abs().max())
    ts = []
    for f in (v1, v2, v3):
        f(q, K)                             # warm
        t = time.time()
        for _ in range(REPEAT): f(q, K)
        ts.append(1000 * (time.time() - t) / REPEAT)
    print("%8d %10.2f %10.2f %10.2f %9.2e %9.2e" % (N, ts[0], ts[1], ts[2], e2, e3), flush=True)

print("MICRO_DONE", flush=True)

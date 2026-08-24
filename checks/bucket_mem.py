"""핵심 질문: bucketing 이 디바이스 메모리에서도 가중치를 복제하는가.

파일 크기는 2 배였다. 디바이스 할당도 2 배면 bucketing 은 우리 문제
(가중치 2 벌이면 TP1 카드 15.7 GiB 초과)를 못 푼다.
비교 대상으로 CompileContext(use_weight_sharing=True) 2 회 컴파일도 같이 잰다.
"""
import os, torch, rebel
import torch.nn as nn

H = 4096
OUT = "/home/work/npu_work/dflash_work/bucket_probe"
os.makedirs(OUT, exist_ok=True)


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.fc = nn.Linear(H, H, bias=False)
    def forward(s, x):
        return s.fc(x)


def info(L):
    return [("x", [1, L, H], "float32")]


def show(tag, cm):
    fns = {}
    for n in ("get_total_device_alloc", "get_num_device_context", "get_executor_count"):
        if hasattr(cm, n):
            try: fns[n] = getattr(cm, n)()
            except Exception as e: fns[n] = "err:%s" % repr(e)[:60]
    tot = fns.get("get_total_device_alloc")
    mb = (tot / 1e6) if isinstance(tot, (int, float)) else tot
    print("%-26s device_alloc=%s MB  ctx=%s  exec=%s" % (
        tag, ("%.1f" % mb) if isinstance(mb, float) else mb,
        fns.get("get_num_device_context"), fns.get("get_executor_count")), flush=True)
    return tot


torch.manual_seed(0)
m = M().eval()

print("=== bucketing ===", flush=True)
a = show("1 bucket  (L17)", rebel.compile_from_torch(m, input_info=[info(17)]))
b = show("2 buckets (L17,L256)", rebel.compile_from_torch(m, input_info=[info(17), info(256)]))

print("", flush=True)
print("=== CompileContext 공유 2 회 컴파일 (optimum-rbln 방식) ===", flush=True)
from rebel import CompileContext
ctx = CompileContext(use_weight_sharing=True)
c17 = rebel.compile_from_torch(m, input_info=info(17), compile_context=ctx)
c256 = rebel.compile_from_torch(m, input_info=info(256), compile_context=ctx)
s1 = show("shared #1 (L17)", c17)
s2 = show("shared #2 (L256)", c256)

print("", flush=True)
if isinstance(a, (int, float)) and isinstance(b, (int, float)):
    print("bucketing 2개/1개 비율 = %.2f  -> %s" % (
        b / a, "가중치 복제됨" if b / a > 1.5 else "가중치 공유됨"), flush=True)
if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
    print("shared #2/#1 비율      = %.2f" % (s2 / s1), flush=True)
print("BUCKET_MEM_DONE", flush=True)

"""Bucketing 검증 (docs.rbln.ai 튜토리얼 방식).

확인할 것:
  1. 서로 다른 시퀀스 길이 두 개를 한 번의 compile_from_torch 로 넣을 수 있는가
  2. 런타임 하나가 두 shape 을 자동으로 받는가
  3. **가중치가 한 벌인가** — 버킷 1개짜리와 2개짜리 파일 크기를 비교
  4. 출력 shape 이 버킷마다 달라도 되는가 (우리 케이스: chunk17 은 전체 위치
     로짓, chunk256 은 1 위치. 이게 되면 DFlash 타깃도 버킷화 가능)
"""
import os, time, torch, rebel
import torch.nn as nn

H = 4096                      # 가중치 67 MB (fp32) — 공유 여부가 크기로 드러나게
DEV = int(os.environ.get("DEV", "0"))
OUT = "/home/work/npu_work/dflash_work/bucket_probe"
os.makedirs(OUT, exist_ok=True)


class Same(nn.Module):
    """출력 shape 이 입력을 따라감 (버킷 차원만 다름)."""
    def __init__(s):
        super().__init__(); s.fc = nn.Linear(H, H, bias=False)
    def forward(s, x):
        return s.fc(x)


class LastOnly(nn.Module):
    """출력이 항상 1 위치 — 버킷마다 출력 shape 이 같음."""
    def __init__(s):
        super().__init__(); s.fc = nn.Linear(H, H, bias=False)
    def forward(s, x):
        return s.fc(x[:, -1:])


def info(L):
    return [("x", [1, L, H], "float32")]


def build(name, mod, infos):
    t = time.time()
    try:
        cm = rebel.compile_from_torch(mod.eval(), input_info=infos)
    except Exception as e:
        print("%-22s COMPILE FAIL: %s" % (name, repr(e)[:200]), flush=True)
        return None
    p = os.path.join(OUT, name + ".rbln")
    cm.save(p)
    sz = os.path.getsize(p) / 1e6
    print("%-22s compile=%4.0fs  file=%8.1f MB  executors=%s" % (
        name, time.time() - t, sz,
        cm.get_executor_names() if hasattr(cm, "get_executor_names") else "?"), flush=True)
    return cm


torch.manual_seed(0)
m = Same()

print("=== 1. 버킷 1개 vs 2개, 파일 크기 ===", flush=True)
c1 = build("same_L17", m, [info(17)])
c2 = build("same_L17_L256", m, [info(17), info(256)])

if c2 is not None:
    print("", flush=True)
    print("=== 2. 런타임 하나로 두 shape 실행 ===", flush=True)
    rt = rebel.Runtime(c2, tensor_type="pt", device=DEV)
    ok = True
    for L in (17, 256):
        try:
            o = rt(torch.randn(1, L, H).contiguous())
            o = torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o)
            print("   L=%-4d -> out %s" % (L, tuple(o.shape)), flush=True)
        except Exception as e:
            ok = False
            print("   L=%-4d RUN FAIL: %s" % (L, repr(e)[:180]), flush=True)
    del rt
    print("   자동 버킷 선택: %s" % ("OK" if ok else "실패"), flush=True)

print("", flush=True)
print("=== 3. 출력 shape 이 버킷 간 동일한 경우 ===", flush=True)
c3 = build("lastonly_L17_L256", LastOnly(), [info(17), info(256)])

print("", flush=True)
print("BUCKET_PROBE_DONE", flush=True)

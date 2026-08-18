"""두 청크 그래프 교차 사용의 수치 검증 + verify 크기 호출 비용 측정.

A) 교차: [0:64]->청크64, [64:192]->청크128, [192:256]->청크64
B) 기준: 전부 청크64 로 4회
마지막 구간의 hidden 이 같으면 캐시가 두 그래프 사이에서 일관되게 공유된 것.
"""
import os, time, torch, rebel
from transformers import AutoTokenizer

D = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
DEV = int(os.environ.get("DEV", "1"))
rt = {}
for ch in (64, 128):
    cmod = rebel.RBLNCompiledModel(os.path.join(D, "dual_graphs", "prefill_%d.rbln" % ch))
    rt[ch] = rebel.Runtime(cmod, tensor_type="pt", device=DEV)

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids
BT = torch.zeros(1, dtype=torch.int16)


def call(ch, ids, off):
    n = ids.shape[1]
    cp = torch.arange(off, off + n, dtype=torch.int32).unsqueeze(0)
    qp = torch.tensor(n - 1, dtype=torch.int16)
    return rt[ch](ids, cp, BT, qp)


def last_hidden(out):
    return torch.as_tensor(out[-1]).float()


# B) 기준 -- 전부 청크64
for off in (0, 64, 128, 192):
    ref = call(64, full[:, off:off + 64].contiguous(), off)
h_ref = last_hidden(ref)

# A) 교차 -- 캐시를 새로 비우기 위해 offset 0 부터 다시
outs = []
outs.append(call(64, full[:, 0:64].contiguous(), 0))
outs.append(call(128, full[:, 64:192].contiguous(), 64))
outs.append(call(64, full[:, 192:256].contiguous(), 192))
h_mix = last_hidden(outs[-1])

cos = float(torch.nn.functional.cosine_similarity(h_ref.flatten(), h_mix.flatten(), dim=0))
print("MIXED vs ALL64  cos=%.6f  %s" % (cos, "MATCH" if cos > 0.999 else "MISMATCH"), flush=True)

# verify 크기 호출 비용
for ch in (64, 128):
    seg = full[:, :17].contiguous()
    for _ in range(3):
        call(ch, seg, 0)
    ts = []
    for _ in range(7):
        s = time.time(); call(ch, seg, 0); ts.append((time.time() - s) * 1000)
    ts.sort()
    print("chunk%-4d  L=17  %.1f ms" % (ch, ts[3]), flush=True)
print("CHECK_DONE", flush=True)

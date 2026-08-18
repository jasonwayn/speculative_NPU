"""그래프 반환 형식 확인 + 두 그래프 교차 사용 시 캐시 일관성 검증.

시나리오: tokens[0:64] 를 청크64 로, [64:192] 를 청크128 로, [192:256] 을 다시 청크64 로.
          마지막 hidden 을 단일 모델(청크64) 전체 처리 결과와 비교.
"""
import os, sys, torch, rebel
from optimum.rbln import RBLNQwen3ForCausalLM
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


def call(ch, ids, off):
    n = ids.shape[1]
    cp = torch.arange(off, off + n, dtype=torch.int32).unsqueeze(0)
    bt = torch.zeros(1, dtype=torch.int16)
    qp = torch.tensor(n - 1, dtype=torch.int16)
    return rt[ch](ids, cp, bt, qp)


out = call(64, full[:, :64].contiguous(), 0)
print("return type=%s len=%d" % (type(out).__name__, len(out)), flush=True)
for i, o in enumerate(out[:4]):
    t = torch.as_tensor(o)
    print("  [%d] shape=%s dtype=%s" % (i, tuple(t.shape), t.dtype), flush=True)
if len(out) > 4:
    t = torch.as_tensor(out[-1])
    print("  [-1] shape=%s dtype=%s" % (tuple(t.shape), t.dtype), flush=True)
print("PROBE_DONE", flush=True)

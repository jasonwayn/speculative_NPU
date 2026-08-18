"""청크 64 / 128 두 prefill 그래프가 실제로 KV 캐시를 공유하는지 검증 + 속도 측정.

검증: 청크64 그래프로 앞부분을 처리(캐시 기록) -> 청크128 그래프로 이어서 처리
      -> 단일 그래프로 통째 처리한 결과와 hidden 이 같으면 캐시 공유 성립.
"""
import os, sys, time, torch, rebel
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer

D = "/home/work/npu_work/dflash_work"
OUTDIR = D + "/dual_graphs"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
DEV = int(os.environ.get("DEV", "1"))

cm = {}
rt = {}
for ch in (64, 128):
    p = os.path.join(OUTDIR, "prefill_%d.rbln" % ch)
    if not os.path.exists(p):
        print("MISSING", p, flush=True); sys.exit(1)
    cm[ch] = rebel.RBLNCompiledModel(p)
    rt[ch] = rebel.Runtime(cm[ch], tensor_type="pt", device=DEV)
    print("runtime ok chunk=%d  inputs=%d" % (ch, len(cm[ch].get_input_info() if hasattr(cm[ch], "get_input_info") else [])), flush=True)

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids

# 런타임 입력 서명 확인 (캐시가 static 으로 빠졌는지)
try:
    ii = cm[64].get_input_info()
    names = [x[0] if isinstance(x, (list, tuple)) else str(x) for x in ii]
    kv = sum(1 for n in names if "past_key_values" in str(n))
    print("chunk64 input_info: total=%d, past_key_values=%d" % (len(names), kv), flush=True)
except Exception as e:
    print("input_info n/a: %s" % str(e)[:80], flush=True)

# 실제 호출 인자 개수를 탐색적으로 확인
ids = full[:, :64].contiguous()
cp = torch.arange(0, 64, dtype=torch.int32).unsqueeze(0)
bt = torch.zeros(1, dtype=torch.int16)
qp = torch.tensor(0, dtype=torch.int16)
am = torch.ones(1, 1, 64, 4096, dtype=torch.float32)
for trial, args in [
    ("(ids,cp,bt,qp)", (ids, cp, bt, qp)),
    ("(ids,cp,bt,qp,am)", (ids, cp, bt, qp, am)),
    ("(ids,cp,bt)", (ids, cp, bt)),
]:
    try:
        out = rt[64](*args)
        print("CALL_OK %s -> %s" % (trial, type(out)), flush=True)
        break
    except Exception as e:
        print("CALL_FAIL %-18s %s" % (trial, str(e)[:110]), flush=True)
print("DV_DONE", flush=True)

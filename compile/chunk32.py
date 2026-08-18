"""prefill_chunk_size 64 하한이 커널 제약인가, 파이썬 검증문일 뿐인가.

configuration_decoderonly.py:224 의 `% 64 != 0` 검사를 우회해 32 로 컴파일해 본다.
컴파일까지 되면 수치 정확성도 확인한다 (chunk64 모델과 hidden 비교).
"""
import os, sys, time, traceback, torch

import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as M
C = M.RBLNDecoderOnlyModelForCausalLMConfig
_orig = C.__init__


def _patched(self, *a, **kw):
    want = kw.get("prefill_chunk_size")
    if want is not None and want % 64 != 0:
        kw = dict(kw); kw["prefill_chunk_size"] = 64
        _orig(self, *a, **kw)
        self.prefill_chunk_size = want          # 검증 통과 후 원하는 값으로 되돌림
    else:
        _orig(self, *a, **kw)


C.__init__ = _patched
print("validation patched", flush=True)

from optimum.rbln import RBLNQwen3ForCausalLM
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
OUT = "/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c32"
CH = int(os.environ.get("CH", "32"))

if not os.path.exists(OUT):
    t = time.time()
    try:
        m = RBLNQwen3ForCausalLM.from_pretrained(
            SRC, export=True, rbln_batch_size=1, rbln_max_seq_len=2048,
            rbln_prefill_chunk_size=CH, rbln_output_hidden_states=True)
        m.save_pretrained(OUT)
        print("COMPILE_OK chunk=%d %.1fs" % (CH, time.time() - t), flush=True)
        del m
    except Exception as e:
        print("COMPILE_FAIL chunk=%d %.1fs %s: %s" % (CH, time.time() - t, type(e).__name__, str(e)[:300]), flush=True)
        traceback.print_exc()
        print("CHUNK32_DONE", flush=True)
        sys.exit(0)
else:
    print("SKIP compile (exists)", flush=True)

# 수치 검증: chunk64 모델과 같은 hidden 을 내는가
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open("/home/work/npu_work/dflash_work/pg1342.txt", encoding="utf-8").read(),
           return_tensors="pt").input_ids
ids = full[:, :300].contiguous()


def run(path):
    m = RBLNQwen3ForCausalLM.from_pretrained(path, export=False, rbln_device=1)
    ch = m.prefill_decoder.rbln_config.prefill_chunk_size
    L = ids.shape[1]
    pad = 1 if L % ch == 0 else 0
    i2 = torch.cat([ids, ids[:, -1:]], dim=1) if pad else ids
    t = time.time()
    h = m(input_ids=i2, attention_mask=torch.ones_like(i2)).hidden_states[-1][:, :L].float()
    dt = (time.time() - t) * 1000
    del m
    return ch, h, dt


try:
    c64, h64, t64 = run("/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64")
    c32, h32, t32 = run(OUT)
    cos = float(torch.nn.functional.cosine_similarity(h64.flatten(), h32.flatten(), dim=0))
    print("chunk%-3d N=300 %.1fms | chunk%-3d N=300 %.1fms | cos=%.6f %s"
          % (c64, t64, c32, t32, cos, "MATCH" if cos > 0.999 else "MISMATCH"), flush=True)
except Exception as e:
    print("RUN_FAIL %s: %s" % (type(e).__name__, str(e)[:250]), flush=True)
print("CHUNK32_DONE", flush=True)

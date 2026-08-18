"""두 prefill 그래프(청크 64 + 128)가 KV 캐시를 공유하도록 컴파일.

optimum-rbln 은 이미 prefill + image_prefill 두 그래프를 같은 CompileContext /
static_tensors 로 컴파일한다. 청크만 다르게 못 하도록 NotImplementedError 로 막아뒀는데,
그 검사만 우회해서 되는지 본다.

되면: seg <= 64 는 청크64 그래프, seg > 64 는 청크128 그래프  -> verify 약 11% 개선
"""
import os, sys, time, traceback, torch

import optimum.rbln.transformers.models.decoderonly.modeling_decoderonly as MD
import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as CF

# 1) 청크 64 배수 검증 우회 (128 은 배수라 사실 불필요하지만 실험 여지를 위해)
_C = CF.RBLNDecoderOnlyModelForCausalLMConfig
_orig_init = _C.__init__


def _init(self, *a, **kw):
    want = kw.get("prefill_chunk_size")
    if want is not None and want % 64 != 0:
        kw = dict(kw); kw["prefill_chunk_size"] = 64
        _orig_init(self, *a, **kw); self.prefill_chunk_size = want
    else:
        _orig_init(self, *a, **kw)


_C.__init__ = _init

# 2) "두 prefill 청크가 다르면 불가" 검사 우회
#    _update_rbln_config 안에 인라인된 raise 라 소스 패치로 지운다.
import inspect, textwrap, types
src = inspect.getsource(MD.RBLNDecoderOnlyModelForCausalLM._update_rbln_config)
if "different prefill chunk sizes" not in src:
    print("GUARD_NOT_FOUND"); sys.exit(1)
lines = src.split("\n")
out = []
skip = 0
for i, l in enumerate(lines):
    if "prefill_chunk_size != rbln_config.image_prefill_chunk_size" in l:
        out.append(l.replace("!=", "is not None and False and"))   # 조건을 항상 거짓으로
        continue
    out.append(l)
patched = textwrap.dedent("\n".join(out))
ns = dict(MD.__dict__)
exec(compile(patched, "<patched>", "exec"), ns)
MD.RBLNDecoderOnlyModelForCausalLM._update_rbln_config = classmethod(ns["_update_rbln_config"].__func__
                                                                    if hasattr(ns["_update_rbln_config"], "__func__")
                                                                    else ns["_update_rbln_config"])
print("guards patched", flush=True)

from optimum.rbln import RBLNQwen3ForCausalLM
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
OUT = "/home/work/npu_work/dflash_work/rbln-Qwen3-4B-dual-64-128"
t = time.time()
try:
    m = RBLNQwen3ForCausalLM.from_pretrained(
        SRC, export=True,
        rbln_batch_size=1, rbln_max_seq_len=4096,
        rbln_prefill_chunk_size=64,
        rbln_image_prefill_chunk_size=128,
        rbln_use_image_prefill=True,
        rbln_output_hidden_states=True,
    )
    m.save_pretrained(OUT)
    print("COMPILE_OK %.0fs -> %s" % (time.time() - t, OUT), flush=True)
except Exception as e:
    print("COMPILE_FAIL %.0fs %s: %s" % (time.time() - t, type(e).__name__, str(e)[:400]), flush=True)
    traceback.print_exc()
print("DUAL_DONE", flush=True)

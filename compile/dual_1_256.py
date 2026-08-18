"""청크 17(verify 전용) + 청크 256(초기 prefill 전용) 두 그래프를 KV 캐시 공유로 컴파일.

verify 는 항상 16토큰(bonus+제안15)만 보내므로 17이면 패딩 1칸.
초기 prefill 은 프롬프트 전체를 한 번에 처리하므로 큰 청크가 유리.
`% 64` 검증은 방어적 검사일 뿐임을 앞서 확인(청크32 컴파일 + cos 0.999922).
"""
import os, sys, time, traceback, torch

import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as CF
_C = CF.RBLNDecoderOnlyModelForCausalLMConfig
_orig = _C.__init__


def _init(self, *a, **kw):
    want = kw.get("prefill_chunk_size")
    if want is not None and want % 64 != 0:
        kw = dict(kw); kw["prefill_chunk_size"] = 64
        _orig(self, *a, **kw); self.prefill_chunk_size = want
    else:
        _orig(self, *a, **kw)


_C.__init__ = _init

from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.configuration_utils import RBLNCompileConfig
from transformers import AutoConfig, AutoModelForCausalLM

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
OUTDIR = "/home/work/npu_work/dflash_work/dual_17_256"
CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "17,256").split(",")]
os.makedirs(OUTDIR, exist_ok=True)

cls = RBLNQwen3ForCausalLM
mcfg = AutoConfig.from_pretrained(SRC)
print("loading HF ...", flush=True)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)

rc, _ = cls.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CHUNKS[0],
    "output_hidden_states": True, "create_runtimes": False})
rc.max_seq_len = 4096
rc = cls._update_rbln_config(preprocessors=None, model=hf, model_config=mcfg, rbln_config=rc)
print("config ok chunk=%s maxlen=%s" % (rc.prefill_chunk_size, rc.max_seq_len), flush=True)
wrapped = cls._wrap_model_if_needed(hf, rc)

ctx = static = None
for ch in CHUNKS:
    dst = os.path.join(OUTDIR, "prefill_%d.rbln" % ch)
    info = cls.get_input_info(batch_size=1, query_length=ch, rbln_config=rc, model_config=mcfg)
    cc = RBLNCompileConfig(compiled_model_name="prefill_%d" % ch, input_info=info)
    meta = [n for n, _, _ in cc.input_info if "past_key_values" in n]
    if ctx is None:
        ex = cc.get_dummy_inputs(fill=0, meta_tensor_names=meta)
        ctx, static = cls._get_compile_context(cc, ex)
    else:
        ex = cc.get_dummy_inputs(fill=0, static_tensors=static)
    t = time.time()
    try:
        cm = cls._compile_model(wrapped, cc, ex, ctx, rc, phase="prefill")
        cm.save(dst)
        print("COMPILE_OK chunk=%d %.0fs" % (ch, time.time() - t), flush=True)
    except Exception as e:
        print("COMPILE_FAIL chunk=%d %s: %s" % (ch, type(e).__name__, str(e)[:300]), flush=True)
        traceback.print_exc(); break
print("D17_DONE", flush=True)

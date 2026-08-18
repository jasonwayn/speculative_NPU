"""hidden state 출력을 37개 -> 6개로 줄인 청크 17/256 그래프 컴파일.

벤더 래퍼는 `return logits, all_hidden_states` 로 37개(embedding + 36 layer)를
전부 호스트로 내보낸다. DFlash 가 실제로 쓰는 건

    target_layer_ids = [1, 9, 17, 25, 33] 인데 extract_context_feature 가
    hidden_states[layer_id + 1] 로 읽으므로 실제 인덱스는 [2, 10, 18, 26, 34]
    hs[-1] = 인덱스 36                      lm_head 입력

6개뿐이다. CMR 을 쓰려면 hs[-2](인덱스 35)도 필요하므로 KEEP 에 35 를 넣어 다시 컴파일할 것.

출력 순서는 KEEP 순서 그대로다. 벤치에서 LID 를 [0,1,2,3,4] 로 바꿔야 한다.
"""
import os, sys, time, traceback, torch
import torch.nn as nn

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
OUTDIR = os.environ.get("OUTDIR", "/home/work/npu_work/dflash_work/slim_17_256")
CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "17,256").split(",")]
KEEP = tuple(int(x) for x in os.environ.get("KEEP", "2,10,18,26,34,36").split(","))
os.makedirs(OUTDIR, exist_ok=True)


class SliceHS(nn.Module):
    """벤더 래퍼 출력에서 필요한 hidden state 만 남긴다."""

    def __init__(self, m, keep):
        super().__init__()
        self.m = m
        self._keep = keep

    @property
    def phase(self):
        return self.m.phase

    @phase.setter
    def phase(self, v):
        self.m.phase = v

    def forward(self, *a):
        logits, hs = self.m(*a)
        return logits, tuple(hs[i] for i in self._keep)


cls = RBLNQwen3ForCausalLM
mcfg = AutoConfig.from_pretrained(SRC)
print("loading HF ... KEEP=%s" % (KEEP,), flush=True)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)

rc, _ = cls.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CHUNKS[0],
    "output_hidden_states": True, "create_runtimes": False})
rc.max_seq_len = 4096
rc = cls._update_rbln_config(preprocessors=None, model=hf, model_config=mcfg, rbln_config=rc)
print("config ok chunk=%s maxlen=%s" % (rc.prefill_chunk_size, rc.max_seq_len), flush=True)

wrapped = SliceHS(cls._wrap_model_if_needed(hf, rc), KEEP)

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
        print("COMPILE_OK chunk=%d %.0fs -> %s" % (ch, time.time() - t, dst), flush=True)
    except Exception as e:
        print("COMPILE_FAIL chunk=%d %s: %s" % (ch, type(e).__name__, str(e)[:400]), flush=True)
        traceback.print_exc(); break
print("SLIM_DONE", flush=True)

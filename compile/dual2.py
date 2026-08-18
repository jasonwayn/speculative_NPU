"""prefill 그래프를 청크 64 / 128 두 개로, KV 캐시를 공유해 컴파일.

image_prefill 가드를 우회하는 대신 벤더의 하위 API 를 직접 호출한다:
  _wrap_model_if_needed -> get_input_info(query_length=N) -> _get_compile_context
  -> _compile_model (같은 context, 두 번째는 static_tensors 재사용)
이게 optimum-rbln 이 prefill + image_prefill 을 만들 때 쓰는 바로 그 경로다.
"""
import os, sys, time, traceback, json, torch

from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.configuration_utils import RBLNCompileConfig
from transformers import AutoConfig, AutoModelForCausalLM

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
OUTDIR = "/home/work/npu_work/dflash_work/dual_graphs"
CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "64,128").split(",")]
os.makedirs(OUTDIR, exist_ok=True)

cls = RBLNQwen3ForCausalLM
mcfg = AutoConfig.from_pretrained(SRC)

# HF 모델을 먼저 올린다 (_update_rbln_config 가 model 을 본다)
print("loading HF model ...", flush=True)
_t = time.time()
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
print("loaded %.0fs" % (time.time() - _t), flush=True)

print("building rbln_config ...", flush=True)
try:
    rc, _rest = cls.prepare_rbln_config(rbln_config={
        "batch_size": 1, "max_seq_len": 4096,
        "prefill_chunk_size": CHUNKS[0],
        "output_hidden_states": True, "create_runtimes": False})
    rc.max_seq_len = 4096          # _update 가 모델 config(40960) 로 덮는 것을 막는다
    rc = cls._update_rbln_config(preprocessors=None, model=hf, model_config=mcfg, rbln_config=rc)
    print("  max_seq_len=%s chunk=%s" % (rc.max_seq_len, rc.prefill_chunk_size), flush=True)
except Exception as e:
    print("CFG_FAIL %s: %s" % (type(e).__name__, str(e)[:250]), flush=True)
    traceback.print_exc(); sys.exit(1)
print("rbln_config ok  chunk=%s  cfgs=%d" % (rc.prefill_chunk_size, len(rc.compile_cfgs)), flush=True)

wrapped = cls._wrap_model_if_needed(hf, rc)

ctx = None
static = None
for ch in CHUNKS:
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
        cm.save(os.path.join(OUTDIR, "prefill_%d.rbln" % ch))
        print("COMPILE_OK chunk=%d %.0fs" % (ch, time.time() - t), flush=True)
    except Exception as e:
        print("COMPILE_FAIL chunk=%d %.0fs %s: %s" % (ch, time.time() - t, type(e).__name__, str(e)[:300]), flush=True)
        traceback.print_exc()
        break
print("DUAL2_DONE", flush=True)

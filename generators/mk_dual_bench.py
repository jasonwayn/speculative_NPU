"""bench_sf_ua.py -> bench_sf_dual.py : 이중 그래프(청크 17 verify / 256 prefill) 적용.

- 단일 청크64 모델 로딩을 두 개의 래핑된 런타임으로 교체
- twh(초기 prefill) -> 청크256,  tpre(verify) -> 청크17
- 비정렬 재개 가드 제거는 원본 프리앰블에 이미 들어 있음
"""
import io, sys

SETUP = '''# --- 이중 그래프: verify=청크17 / prefill=청크256, KV 캐시 공유 --------------
import copy as _copy, rebel as _rebel
import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as _CF
_CC = _CF.RBLNDecoderOnlyModelForCausalLMConfig
_cc_orig = _CC.__init__
def _cc_init(self, *a, **kw):
    w = kw.get("prefill_chunk_size")
    if w is not None and w % 64 != 0:
        kw = dict(kw); kw["prefill_chunk_size"] = 64
        _cc_orig(self, *a, **kw); self.prefill_chunk_size = w
    else:
        _cc_orig(self, *a, **kw)
_CC.__init__ = _cc_init
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNPageTableManager as _PTM
from transformers import AutoModelForCausalLM as _AMC, AutoConfig as _AC
CH_V, CH_P = 17, 256
_mcfg = _AC.from_pretrained(SRC)
_hf = _AMC.from_pretrained(SRC, dtype=torch.float32)
_rc, _ = RBLNQwen3ForCausalLM.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CH_V,
    "output_hidden_states": True, "create_runtimes": False})
_rc.max_seq_len = 4096
_rc = RBLNQwen3ForCausalLM._update_rbln_config(preprocessors=None, model=_hf,
                                               model_config=_mcfg, rbln_config=_rc)
_ptm = _PTM(_rc)
_dam = torch.zeros(_rc.batch_size, 1, 1, _rc.max_seq_len, dtype=torch.float32)
_common = dict(main_input_name="input_ids", embed_tokens=_hf.model.embed_tokens,
               dec_attn_mask=_dam, page_table_manager=_ptm, config=_mcfg)
DUAL = {}
for _ch in (CH_V, CH_P):
    _cm = _rebel.RBLNCompiledModel(
        "/home/work/npu_work/dflash_work/dual_17_256/prefill_%d.rbln" % _ch)
    _cfg = _copy.deepcopy(_rc); _cfg.prefill_chunk_size = _ch
    DUAL[_ch] = _RU.RBLNRuntimeModel(runtime=_rebel.Runtime(_cm, tensor_type="pt", device=DEV),
                                     phase="prefill", batch_size=_rc.batch_size,
                                     rbln_config=_cfg, **_common)
del _hf
CHUNK = CH_V
print("dual graphs ready: verify=%d prefill=%d" % (CH_V, CH_P), flush=True)
# ------------------------------------------------------------------------------
'''

src = "/home/work/npu_work/dflash_work/bench_sf_ua.py"
dst = "/home/work/npu_work/dflash_work/bench_sf_dual.py"
lines = io.open(src, encoding="utf-8").read().splitlines()

i_m = next(i for i, l in enumerate(lines) if l.startswith("m = RBLNQwen3ForCausalLM.from_pretrained(TGT"))
i_p = next(i for i, l in enumerate(lines) if l.startswith("pdec = m.prefill_decoder"))
lines = lines[:i_m] + SETUP.split("\n") + lines[i_p + 1:]

n = 0
for i, l in enumerate(lines):
    if "r = pdec.prefill_forward(seg," in l:
        lines[i] = l.replace("pdec.prefill_forward", "DUAL[CH_V].prefill_forward"); n += 1
    elif "hs = m(input_ids=i2, attention_mask=torch.ones_like(i2)).hidden_states" in l:
        ind = len(l) - len(l.lstrip())
        lines[i] = (" " * ind + "hs = DUAL[CH_P].prefill_forward(i2, "
                    "cache_position=torch.arange(i2.shape[1], dtype=torch.int32).unsqueeze(0), "
                    "attention_mask=torch.ones(i2.shape[1], dtype=torch.int64), batch_idx=0, "
                    "block_tables=BT, is_external_block_tables=False).hidden_states")
        n += 1
    elif "pad = 1 if L % CHUNK == 0 else 0" in l:
        ind = len(l) - len(l.lstrip())
        lines[i] = " " * ind + "pad = 1 if L % CH_V == 0 else 0"
        n += 1
    elif "if ids.shape[1] % CHUNK == 0" in l or "pad = 1 if L % CHUNK == 0" in l:
        n += 0
if n < 3:
    print("PATCH_MISS n=%d" % n); sys.exit(1)

# twh 안의 pad 판정은 prefill 청크(256) 기준이어야 한다
out = []
in_twh = False
for l in lines:
    if l.strip().startswith("def twh("):
        in_twh = True
    elif in_twh and l.strip().startswith("def "):
        in_twh = False
    if in_twh and "pad = 1 if L % CH_V == 0 else 0" in l:
        ind = len(l) - len(l.lstrip())
        l = " " * ind + "pad = 1 if L % CH_P == 0 else 0"
    out.append(l)

io.open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("WROTE", dst, "patched", n)

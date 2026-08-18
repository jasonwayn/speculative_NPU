"""청크 17 / 256 이중 그래프: 캐시 공유 검증 + verify 비용 측정.

시나리오: 청크256 으로 프롬프트를 넣고(초기 prefill), 청크17 로 16토큰 검증(비정렬 재개).
기준: 전부 청크256 으로 처리한 결과.
"""
import os, sys, inspect, textwrap, copy, time, torch, rebel

# 비정렬 재개 가드 제거
import optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils as RU
_M = RU.RBLNRuntimeModel
_ls = inspect.getsource(_M.prefill_forward).split("\n")
_o, _i, _rm = [], 0, False
while _i < len(_ls):
    _l = _ls[_i]
    if "prefix_cached_len % self.rbln_config.prefill_chunk_size != 0" in _l:
        _ind = len(_l) - len(_l.lstrip()); _i += 1
        while _i < len(_ls) and (not _ls[_i].strip() or
                                 len(_ls[_i]) - len(_ls[_i].lstrip()) > _ind):
            _i += 1
        _rm = True; continue
    _o.append(_l); _i += 1
assert _rm, "guard not found"
_b = textwrap.dedent("\n".join(_o)).replace("super()", "super(_Base, self)")
_ns = dict(RU.__dict__); _ns["_Base"] = _M
exec(compile(_b, "<patched>", "exec"), _ns)
_M.prefill_forward = _ns["prefill_forward"]

# %64 검증 우회 (로딩 때도 필요)
import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as CF
_C = CF.RBLNDecoderOnlyModelForCausalLMConfig
_oi = _C.__init__


def _init(self, *a, **kw):
    w = kw.get("prefill_chunk_size")
    if w is not None and w % 64 != 0:
        kw = dict(kw); kw["prefill_chunk_size"] = 64
        _oi(self, *a, **kw); self.prefill_chunk_size = w
    else:
        _oi(self, *a, **kw)


_C.__init__ = _init

from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNPageTableManager
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

D = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
DEV = int(os.environ.get("DEV", "1"))
CH = [17, 256]
cls = RBLNQwen3ForCausalLM
mcfg = AutoConfig.from_pretrained(SRC)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
rc, _ = cls.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CH[0],
    "output_hidden_states": True, "create_runtimes": False})
rc.max_seq_len = 4096
rc = cls._update_rbln_config(preprocessors=None, model=hf, model_config=mcfg, rbln_config=rc)

ptm = RBLNPageTableManager(rc)
dam = torch.zeros(rc.batch_size, 1, 1, rc.max_seq_len, dtype=torch.float32)
common = dict(main_input_name="input_ids", embed_tokens=hf.model.embed_tokens,
              dec_attn_mask=dam, page_table_manager=ptm, config=mcfg)
dec = {}
for ch in CH:
    cm = rebel.RBLNCompiledModel(os.path.join(D, "dual_17_256", "prefill_%d.rbln" % ch))
    cfg = copy.deepcopy(rc); cfg.prefill_chunk_size = ch
    dec[ch] = RU.RBLNRuntimeModel(runtime=rebel.Runtime(cm, tensor_type="pt", device=DEV),
                                  phase="prefill", batch_size=rc.batch_size, rbln_config=cfg, **common)
    print("wrapped chunk=%d" % ch, flush=True)

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids
BT = torch.tensor([0], dtype=torch.int16)


def pf(ch, seg, off):
    L = seg.shape[1]
    pad = 1 if L % ch == 0 else 0
    if pad:
        seg = torch.cat([seg, seg[:, -1:]], dim=1)
    r = dec[ch].prefill_forward(seg, cache_position=torch.arange(off, off + seg.shape[1],
                                dtype=torch.int32).unsqueeze(0),
                                attention_mask=torch.ones(seg.shape[1], dtype=torch.int64),
                                batch_idx=0, block_tables=BT, is_external_block_tables=False)
    hs = r.hidden_states
    return tuple(h[:, :L] for h in hs) if pad else hs


P, NEW = 300, 16
ids = full[:, :P].contiguous(); nxt = full[:, P:P + NEW].contiguous()
ref = pf(256, torch.cat([ids, nxt], dim=1), 0)[-1][:, P:P + NEW].float()
pf(256, ids, 0)
mix = pf(17, nxt, P)[-1][:, :NEW].float()
cos = float(torch.nn.functional.cosine_similarity(ref.flatten(), mix.flatten(), dim=0))
print("MIX(256 prefill -> 17 verify) cos=%.6f  %s" % (cos, "MATCH" if cos > 0.999 else "MISMATCH"), flush=True)

for ch in CH:
    pf(ch, ids, 0)
    for _ in range(3): pf(ch, nxt, P)
    ts = []
    for _ in range(7):
        s = time.time(); pf(ch, nxt, P); ts.append((time.time() - s) * 1000)
    ts.sort(); print("verify(16tok) chunk%-4d %.1f ms" % (ch, ts[3]), flush=True)
print("D17CHK_DONE", flush=True)

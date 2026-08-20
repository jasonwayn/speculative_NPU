"""타깃 verify 를 TP1 / TP2 / TP4 로 돌렸을 때 출력이 같은가.

같은 1024 토큰 프롬프트를 청크 256 으로 prefill 한 뒤, 오프셋 1024 에서 16 토큰을
청크 17 로 검증하고 hidden state 를 저장한다. 세 구성을 따로 돌린 뒤 비교한다.
TP 는 카드마다 부분합을 내고 all-reduce 하므로 리덕션 순서가 달라진다 —
이게 눈에 띄는 차이를 만드는지 보는 것이 목적.
"""
import os, copy, torch, rebel

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

import inspect, textwrap
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
assert _rm
_b = textwrap.dedent("\n".join(_o)).replace("super()", "super(_Base, self)")
_ns = dict(RU.__dict__); _ns["_Base"] = _M
exec(compile(_b, "<patched>", "exec"), _ns)
_M.prefill_forward = _ns["prefill_forward"]

from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNPageTableManager
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

D = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
TP = int(os.environ.get("TP", "1"))
GDIR = os.environ.get("GDIR", os.path.join(D, "fused_17_256"))
DEV0 = int(os.environ.get("DEV0", "0"))
DEVS = list(range(DEV0, DEV0 + TP)) if TP > 1 else DEV0
CH_V, CH_P = 17, 256
P, NEW = 1024, 16
BT = torch.tensor([0], dtype=torch.int16)

mcfg = AutoConfig.from_pretrained(SRC)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
rc, _ = RBLNQwen3ForCausalLM.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CH_V,
    "output_hidden_states": True, "create_runtimes": False})
rc.tensor_parallel_size = TP
rc.max_seq_len = 4096
rc = RBLNQwen3ForCausalLM._update_rbln_config(preprocessors=None, model=hf,
                                              model_config=mcfg, rbln_config=rc)
ptm = RBLNPageTableManager(rc)
dam = torch.zeros(rc.batch_size, 1, 1, rc.max_seq_len, dtype=torch.float32)
dec = {}
for ch in (CH_V, CH_P):
    cm = rebel.RBLNCompiledModel("%s/prefill_%d.rbln" % (GDIR, ch))
    cfg = copy.deepcopy(rc); cfg.prefill_chunk_size = ch
    if ch == CH_V: cfg.logits_to_keep = CH_V
    gcfg = copy.deepcopy(mcfg); gcfg.num_hidden_layers = 5 if ch == CH_V else 4
    dec[ch] = RU.RBLNRuntimeModel(runtime=rebel.Runtime(cm, tensor_type="pt", device=DEVS),
                                  phase="prefill", batch_size=rc.batch_size, rbln_config=cfg,
                                  main_input_name="input_ids", embed_tokens=hf.model.embed_tokens,
                                  dec_attn_mask=dam, page_table_manager=ptm, config=gcfg)
print("TP=%d devices=%s dir=%s ready" % (TP, DEVS, GDIR), flush=True)
del hf

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids
ids = full[:, :P].contiguous(); nxt = full[:, P:P + NEW].contiguous()


def pf(ch, seg, off):
    L = seg.shape[1]
    pad = 1 if L % ch == 0 else 0
    if pad: seg = torch.cat([seg, seg[:, -1:]], dim=1)
    r = dec[ch].prefill_forward(seg, cache_position=torch.arange(off, off + seg.shape[1],
                                dtype=torch.int32).unsqueeze(0),
                                attention_mask=torch.ones(seg.shape[1], dtype=torch.int64),
                                batch_idx=0, block_tables=BT, is_external_block_tables=False)
    hs = r.hidden_states
    return tuple(h[:, :L] for h in hs) if pad else hs


hp = pf(CH_P, ids, 0)
hv = pf(CH_V, nxt, P)
out = dict(prefill_last=hp[-1].float().clone(),
           verify=[h.float().clone() for h in hv])
torch.save(out, "/tmp/tp%d.pt" % TP)
print("TP%d saved  n_prefill_out=%d n_verify_out=%d" % (TP, len(hp), len(hv)), flush=True)
print("TPCHK_DONE", flush=True)

"""slim(6개 출력) vs dual(37개 출력) 정확도/속도 비교.

카드 1장에 타깃 두 벌이 안 올라가므로 같은 카드에서 두 번 순차 실행한다.
  TAG=full DIR=dual_17_256 KEEPN=37 python3 slim_check.py
  TAG=slim DIR=slim_17_256 KEEPN=6  python3 slim_check.py
두 번째 실행이 첫 번째가 저장한 텐서와 cos 를 비교한다.
"""
import os, copy, time, torch, rebel

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

# 비정렬 재개 가드 제거
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
assert _rm, "guard not found"
_b = textwrap.dedent("\n".join(_o)).replace("super()", "super(_Base, self)")
_ns = dict(RU.__dict__); _ns["_Base"] = _M
exec(compile(_b, "<patched>", "exec"), _ns)
_M.prefill_forward = _ns["prefill_forward"]

from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNPageTableManager
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

D = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
DEV = int(os.environ.get("DEV", "0"))
DIR = os.environ.get("DIR", "slim_17_256")
TAG = os.environ.get("TAG", "slim")
KEEPN = int(os.environ.get("KEEPN", "6"))
CH = [17, 256]

mcfg = AutoConfig.from_pretrained(SRC)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
rc, _ = RBLNQwen3ForCausalLM.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CH[0],
    "output_hidden_states": True, "create_runtimes": False})
rc.max_seq_len = 4096
rc = RBLNQwen3ForCausalLM._update_rbln_config(preprocessors=None, model=hf,
                                              model_config=mcfg, rbln_config=rc)

# 런타임이 잡는 출력 버퍼 개수 = config.num_hidden_layers + 1 -> 잘린 개수에 맞춘다
wcfg = copy.deepcopy(mcfg); wcfg.num_hidden_layers = KEEPN - 1
ptm = RBLNPageTableManager(rc)
dam = torch.zeros(rc.batch_size, 1, 1, rc.max_seq_len, dtype=torch.float32)
common = dict(main_input_name="input_ids", embed_tokens=hf.model.embed_tokens,
              dec_attn_mask=dam, page_table_manager=ptm, config=wcfg)
dec = {}
for ch in CH:
    cm = rebel.RBLNCompiledModel(os.path.join(D, DIR, "prefill_%d.rbln" % ch))
    cfg = copy.deepcopy(rc); cfg.prefill_chunk_size = ch
    dec[ch] = RU.RBLNRuntimeModel(runtime=rebel.Runtime(cm, tensor_type="pt", device=DEV),
                                  phase="prefill", batch_size=rc.batch_size,
                                  rbln_config=cfg, **common)
    print("wrapped %s chunk=%d" % (DIR, ch), flush=True)
del hf

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids
BT = torch.tensor([0], dtype=torch.int16)
# extract_context_feature 는 hidden_states[layer_id + 1] 를 읽는다
LID = [2, 10, 18, 26, 34] if KEEPN == 37 else [0, 1, 2, 3, 4]


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

pf(256, ids, 0)
hv = pf(17, nxt, P)
sig = dict(last=hv[-1].float().clone(),
           feat=torch.cat([hv[i].float() for i in LID], dim=-1).clone())
torch.save(sig, "/tmp/sig_%s.pt" % TAG)
print("nout=%d  saved /tmp/sig_%s.pt" % (len(hv), TAG), flush=True)

ref_path = "/tmp/sig_full.pt"
if TAG != "full" and os.path.exists(ref_path):
    ref = torch.load(ref_path)
    for k in ("last", "feat"):
        c = float(torch.nn.functional.cosine_similarity(
            ref[k].flatten(), sig[k].flatten(), dim=0))
        print("COS %-5s %.6f  %s" % (k, c, "MATCH" if c > 0.9999 else "MISMATCH"), flush=True)

# --- 속도 ---
for _ in range(2):
    pf(256, ids, 0)
ts = []
for _ in range(7):
    s = time.time(); pf(256, ids, 0); ts.append((time.time() - s) * 1000)
ts.sort()
print("TIME %-5s prefill(300tok, chunk256) %.1f ms" % (TAG, ts[3]), flush=True)

pf(256, ids, 0)
for _ in range(3):
    pf(17, nxt, P)
ts = []
for _ in range(9):
    s = time.time(); pf(17, nxt, P); ts.append((time.time() - s) * 1000)
ts.sort()
print("TIME %-5s verify(16tok, chunk17)    %.1f ms" % (TAG, ts[4]), flush=True)
print("SLIMCHK_DONE", flush=True)

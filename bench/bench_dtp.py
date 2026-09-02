"""상태 유지 드래프트 위에서 stock vs CMR 재측정.

드래프트가 디바이스 KV 캐시를 재사용하므로(무상태 대비 16K 에서 11.7배 빠름) 이제
GPU 와 공정한 비교가 된다. CMR 은 선택된 위치를 압축 위치로 캐시에 다시 기록해
SpecExtend 의 위치 재인덱싱을 유지하고 어텐션 범위도 실제로 줄인다.
"""
import sys, os
# RBLN 런타임은 8 코어 머신에서 OS 스레드를 32 개 띄운다. torch 의 OMP 스레드가 병렬
# 구간이 끝난 뒤에도 busy-wait 으로 코어를 붙잡으면 NPU 에 호출을 넣어줄 CPU 가 없어져
# 호출이 스케줄링 대기에 걸린다. NTHREADS=2 는 그 곡선의 최악점이었다 (CMR 3.2 배 손해).
# import torch 보다 먼저 설정해야 OMP 런타임이 읽는다. -> docs/HOST_THREAD_STARVATION.md
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
import json, time, threading, subprocess, math, torch, rebel
sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")
from torch import nn
from rebel import CompileContext
from transformers import AutoTokenizer, AutoConfig
from optimum.rbln import RBLNQwen3ForCausalLM
from dflash.model import DFlashDraftModel, extract_context_feature
from draft_two import Append, Block, _Base, PAGED, rope
from draft_two_rr import AppendRR, BlockRR, host_angle
from safetensors import safe_open
import glob as _glob

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
TGT = os.environ.get("TGTDIR", "/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-16k")
DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
B = 16
APPEND_WIDTH = int(os.environ.get("APPEND_WIDTH", str(B)))
DEV = int(os.environ.get("DEV", "0"))
DEV_TARGET = int(os.environ.get("DEV_TARGET", str(DEV)))
DEV_DRAFT = int(os.environ.get("DEV_DRAFT", str(DEV)))
TARGET_TP = int(os.environ.get("TARGET_TP", "1"))
# 드래프터·lm_head 도 가중치를 쪼갠다. 배치 1 에서 비용은 계산이 아니라 가중치를
# 읽는 시간이므로, 일을 나누는 게 아니라 읽을 바이트를 나눠야 시간이 준다.
DRAFT_TP = int(os.environ.get("DRAFT_TP", "1"))
# lm_head 는 캐시도 CompileContext 도 없는 단독 그래프라 드래프터와 달리 TP4 가
# 통합에서도 산다. 동일 조건 A/B 로 178.62 -> 204.78 tok/s (+14.6%), J/tok -14.7%.
# 그래서 드래프터와 따로 준다 (VENDOR_LIMITS §14).
LMH_TP = int(os.environ.get("LMH_TP", str(DRAFT_TP)))
TARGET_DEVICES = (list(range(DEV_TARGET, DEV_TARGET + TARGET_TP))
                  if TARGET_TP > 1 else DEV_TARGET)
DRAFT_DEVICES = (list(range(DEV_DRAFT, DEV_DRAFT + DRAFT_TP))
                 if DRAFT_TP > 1 else DEV_DRAFT)
LMH_DEVICES = (list(range(DEV_DRAFT, DEV_DRAFT + LMH_TP))
               if LMH_TP > 1 else DEV_DRAFT)
TARGET_GRAPH_DIR = os.environ.get(
    "TARGET_GRAPH_DIR", "/home/work/npu_work/dflash_work/fused_17_256")
MAXNEW = int(os.environ.get("MAXNEW", "256")); NSAMP = int(os.environ.get("NSAMP", "3"))
WARMNEW = int(os.environ.get("WARMNEW", "32"))
INLEN = int(os.environ.get("INLEN", "4096"))
CMR = int(os.environ.get("CMR", "0"))
# (CMR 가드 제거: 레이어 35 를 내보내는 그래프를 쓴다)
CHUNK_SZ = int(os.environ.get("CMR_CHUNK", "32")); TOPK = int(os.environ.get("CMR_TOPK", "32"))
EVERY = int(os.environ.get("CMR_EVERY", "4")); BUDGET = int(os.environ.get("CMR_BUDGET", "1024"))
# 프리필에서 한 번만 검색하고 이후 스코어러를 전부 끈다. every 곡선이 64 까지
# 꺾이지 않아(tau -0.4%, 처리량 1.59 배) 동적 검색의 값을 확인하려는 극한 케이스.
CMR_ONCE = int(os.environ.get("CMR_ONCE", "0"))
MAXC = int(os.environ.get("MAXC", "16384"))
# 타깃 그래프를 컴파일할 때 쓴 max_seq_len. 기본은 드래프터 캐시와 동일.
TARGET_MAX_SEQ = int(os.environ.get("TARGET_MAX_SEQ", str(MAXC)))
KV_BLOCK_SIZE = int(os.environ.get("KV_BLOCK_SIZE", str(MAXC)))
if MAXC % KV_BLOCK_SIZE:
    raise ValueError("MAXC must be divisible by KV_BLOCK_SIZE")
KV_NUM_BLOCKS = MAXC // KV_BLOCK_SIZE
DTYPE_NAME = os.environ.get("DRAFT_DTYPE", "float16")
DT, DTS = {
    "float16": (torch.float16, "float16"),
    "bfloat16": (torch.bfloat16, "bfloat16"),
    "float32": (torch.float32, "float32"),
}[DTYPE_NAME]
print("draft dtype=%s" % DTYPE_NAME, flush=True)
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
sys.path.insert(0, "/home/work/npu_work/dflash_work")
from bench_cmr_parts import Power, Scorer, pick_chunks   # 재사용
# --- 비정렬 오프셋 prefix caching 가드 제거 -------------------------------------
import inspect as _insp, textwrap as _tw
import optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils as _RU
_M = _RU.RBLNRuntimeModel
_ls = _insp.getsource(_M.prefill_forward).split("\n")
_o, _i, _rm = [], 0, False
while _i < len(_ls):
    _l = _ls[_i]
    if "prefix_cached_len % self.rbln_config.prefill_chunk_size != 0" in _l:
        _ind = len(_l) - len(_l.lstrip())
        _i += 1
        while _i < len(_ls) and (not _ls[_i].strip() or
                                 len(_ls[_i]) - len(_ls[_i].lstrip()) > _ind):
            _i += 1
        _rm = True
        continue
    _o.append(_l); _i += 1
if not _rm:
    raise RuntimeError("guard not found")
_body = _tw.dedent("\n".join(_o)).replace("super()", "super(_Base, self)")
_ns = dict(_RU.__dict__); _ns["_Base"] = _M
exec(compile(_body, "<patched>", "exec"), _ns)
_M.prefill_forward = _ns["prefill_forward"]
# -------------------------------------------------------------------------------



tok = AutoTokenizer.from_pretrained(SRC)
_W = None
for _f in sorted(_glob.glob(SRC + "/*.safetensors")):
    with safe_open(_f, framework="pt") as _h:
        for _k in _h.keys():
            if _k.endswith("model.embed_tokens.weight"):
                _W = _h.get_tensor(_k).to(DT); break
    if _W is not None: break
embed = nn.Embedding.from_pretrained(_W, freeze=True)
lmw = nn.Linear(_W.shape[1], _W.shape[0], bias=False).to(DT)
with torch.no_grad(): lmw.weight = nn.Parameter(_W, requires_grad=False)
lmw = lmw.eval()

# --- 이중 그래프: verify=청크17 / prefill=청크256, KV 캐시 공유 --------------
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
CH_V = 17
CH_P = int(os.environ.get("TARGET_PREFILL_CHUNK", "256"))
_mcfg = _AC.from_pretrained(SRC)
_hf = _AMC.from_pretrained(SRC, dtype=torch.float32)
_rc, _ = RBLNQwen3ForCausalLM.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": TARGET_MAX_SEQ, "prefill_chunk_size": CH_V,
    "output_hidden_states": True, "create_runtimes": False})
_rc.tensor_parallel_size = TARGET_TP
_rc.max_seq_len = TARGET_MAX_SEQ
_rc = RBLNQwen3ForCausalLM._update_rbln_config(preprocessors=None, model=_hf,
                                               model_config=_mcfg, rbln_config=_rc)
_ptm = _PTM(_rc)
_dam = torch.zeros(_rc.batch_size, 1, 1, _rc.max_seq_len, dtype=torch.float32)
_wcfg = _copy.deepcopy(_mcfg); _wcfg.num_hidden_layers = 6   # 출력 7개
_common = dict(main_input_name="input_ids", embed_tokens=_hf.model.embed_tokens,
               dec_attn_mask=_dam, page_table_manager=_ptm, config=_wcfg)
DUAL = {}
for _ch in dict.fromkeys((CH_V, CH_P)):
    _cm = _rebel.RBLNCompiledModel(
        "%s/prefill_%d.rbln" % (TARGET_GRAPH_DIR, _ch))
    _cfg = _copy.deepcopy(_rc); _cfg.prefill_chunk_size = _ch
    if _ch == CH_V: _cfg.logits_to_keep = CH_V
    _gcommon = dict(_common)
    _gcfg = _copy.deepcopy(_mcfg); _gcfg.num_hidden_layers = 6 if _ch == CH_V else 5
    _gcommon["config"] = _gcfg
    DUAL[_ch] = _RU.RBLNRuntimeModel(runtime=_rebel.Runtime(
                                     _cm, tensor_type="pt", device=TARGET_DEVICES),
                                     phase="prefill", batch_size=_rc.batch_size,
                                     rbln_config=_cfg, **_gcommon)
del _hf
CHUNK = CH_V
print("dual graphs ready: verify=%d prefill=%d" % (CH_V, CH_P), flush=True)
# ------------------------------------------------------------------------------

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
if os.environ.get("ACCURATE_DRAFT_SILU") == "1":
    class _AccurateSilu(nn.Module):
        def forward(self, value):
            return value / (1.0 + torch.exp(-torch.clamp(value, -20.0, 20.0)))

    for _draft_layer in draft.layers:
        _draft_layer.mlp.act_fn = _AccurateSilu()
    print("accurate draft SiLU enabled layers=%d" % len(draft.layers), flush=True)
H = cfg.hidden_size; NT = len(draft.target_layer_ids); LID = draft.target_layer_ids
NL = cfg.num_hidden_layers; NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
INV_HOST = 1.0 / (cfg.rope_theta ** (torch.arange(0, HD, 2).float() / HD))
MASK = draft.mask_token_id
BT = torch.arange(KV_NUM_BLOCKS, dtype=torch.int16)
TARGET_BT = torch.tensor([0], dtype=torch.int16)
scorer = Scorer(SRC) if CMR else None
DRAFT_TRACE = int(os.environ.get("DRAFT_TRACE", "0"))
READ_DRAFT_CACHE = int(os.environ.get("READ_DRAFT_CACHE", "0"))
NONCAUSAL_DRAFT = int(os.environ.get("NONCAUSAL_DRAFT", "0"))
STATELESS_CONTEXT = int(os.environ.get("STATELESS_CONTEXT", "0"))
USE_STATELESS_DRAFT = int(os.environ.get("USE_STATELESS_DRAFT", "0"))
BULK = int(os.environ.get("BULK", "256"))   # prefill op 의 q_len 한계 회피

caches = [
    torch.zeros(KV_NUM_BLOCKS, NKV, KV_BLOCK_SIZE, HD, dtype=DT)
    for _ in range(2 * NL)
]
cinfo = [
    (
        "past_key_values_%d" % i,
        [KV_NUM_BLOCKS, NKV, KV_BLOCK_SIZE, HD],
        DTS,
    )
    for i in range(2 * NL)
]
ctxc = CompileContext(use_weight_sharing=True)
for (n, _, _), t in zip(cinfo, caches): ctxc.mark_static_address(t, n)


def ainfo(A):
    return [("th_new", [1, A, NT * H], DTS), ("angle_ctx", [1, A, HD // 2], DTS),
            ("seq_ctx", [1, 1], "int32"),
            ("block_tables", [KV_NUM_BLOCKS], "int16")] + cinfo


binfo = [("noise_emb", [1, B, H], DTS), ("angle_blk", [1, B, HD // 2], DTS),
         ("seq_blk", [1, 1], "int32"),
         ("block_tables", [KV_NUM_BLOCKS], "int16")]
if NONCAUSAL_DRAFT:
    binfo.append(("block_mask", [1, 1, B, MAXC], DTS))
binfo += cinfo


class _BlockNoncausal(_Base):
    def forward(
        s, noise_emb, pos_blk, seq_blk, block_tables, block_mask, *caches
    ):
        paged_noncausal = torch.ops.rbln_custom_ops.paged_attn_prefill
        cb, sb = s.cs(pos_blk)
        hs = noise_emb
        for i, layer in enumerate(s.d.layers):
            at = layer.self_attn
            kc, vc = caches[2 * i], caches[2 * i + 1]
            resid = hs
            x = layer.input_layernorm(hs)
            q = at.q_norm(
                at.q_proj(x).view(1, B, -1, s.hd)
            ).transpose(1, 2)
            q = rope(q, cb, sb).view(1, s.nkv, s.rep, B, s.hd)
            k = at.k_norm(
                at.k_proj(x).view(1, B, -1, s.hd)
            ).transpose(1, 2)
            k = rope(k, cb, sb).unsqueeze(2)
            v = (
                at.v_proj(x)
                .view(1, B, -1, s.hd)
                .transpose(1, 2)
                .unsqueeze(2)
            )
            o = paged_noncausal(
                q=q,
                k=k,
                v=v,
                mask=block_mask.unsqueeze(2),
                kcache=kc.unsqueeze(2),
                vcache=vc.unsqueeze(2),
                seq=seq_blk,
                scale=s.scale,
                block_table=block_tables,
                block_size=s.max_ctx,
            )
            o = o.view(1, s.nh, B, s.hd).transpose(1, 2).reshape(
                1, B, s.nh * s.hd
            )
            hs = resid + at.o_proj(o)
            hs = hs + layer.mlp(layer.post_attention_layernorm(hs))
        return s.d.norm(hs)


class _BlockTrace(_Base):
    def forward(s, noise_emb, pos_blk, seq_blk, block_tables, *caches):
        cb, sb = s.cs(pos_blk)
        hs = noise_emb
        outputs = []
        for i, layer in enumerate(s.d.layers):
            at = layer.self_attn
            kc, vc = caches[2 * i], caches[2 * i + 1]
            resid = hs
            x = layer.input_layernorm(hs)
            q = at.q_norm(at.q_proj(x).view(1, B, -1, s.hd)).transpose(1, 2)
            q = rope(q, cb, sb).view(1, s.nkv, s.rep, B, s.hd)
            k = at.k_norm(at.k_proj(x).view(1, B, -1, s.hd)).transpose(1, 2)
            k = rope(k, cb, sb).unsqueeze(2)
            v = at.v_proj(x).view(1, B, -1, s.hd).transpose(1, 2).unsqueeze(2)
            o = PAGED(
                q=q, k=k, v=v, kcache=kc.unsqueeze(2), vcache=vc.unsqueeze(2),
                seq=seq_blk, scale=s.scale, block_table=block_tables,
                block_size=s.max_ctx, is_bidirectional=True,
            )
            o = o.view(1, s.nh, B, s.hd).transpose(1, 2).reshape(1, B, s.nh * s.hd)
            hs = resid + at.o_proj(o)
            outputs.append(hs)
            hs = hs + layer.mlp(layer.post_attention_layernorm(hs))
            outputs.append(hs)
        return tuple(outputs) + (s.d.norm(hs),)


class _StatelessDraft(nn.Module):
    def __init__(self, draft_model, context_length):
        super().__init__()
        self.draft = draft_model
        self.context_length = context_length

    def forward(self, noise_embedding, target_hidden, position_ids, attention_mask):
        return self.draft(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )


class _CacheReader(nn.Module):
    def forward(self, key_cache, value_cache):
        # Referencing the buffers through the custom op makes RBLN retain their
        # shared static-address semantics. The write is outside the inspected
        # prompt range.
        output = PAGED(
            q=torch.zeros(1, NKV, cfg.num_attention_heads // NKV, B, HD, dtype=DT),
            k=torch.zeros(1, NKV, 1, B, HD, dtype=DT),
            v=torch.zeros(1, NKV, 1, B, HD, dtype=DT),
            kcache=key_cache.unsqueeze(2),
            vcache=value_cache.unsqueeze(2),
            seq=torch.tensor([[MAXC - B]], dtype=torch.int32),
            scale=torch.tensor(HD**-0.5),
            block_table=BT,
            block_size=KV_BLOCK_SIZE,
            is_bidirectional=True,
        )
        return key_cache, value_cache, output


def exs(info):
    o = []
    for n, sh, dt in info:
        o.append(caches[int(n.rsplit("_", 1)[1])] if n.startswith("past_key_values_")
                 else torch.zeros(*sh, dtype=getattr(torch, dt)))
    return o


t0 = time.time()
_dtp = {"tensor_parallel_size": DRAFT_TP} if DRAFT_TP > 1 else {}
cm_a = rebel.compile_from_torch(AppendRR(draft, KV_BLOCK_SIZE).eval(), input_info=ainfo(APPEND_WIDTH),
                                example_inputs=exs(ainfo(APPEND_WIDTH)), compile_context=ctxc,
                                **_dtp)
# 같은 컨텍스트로 세 번째 그래프를 만들면 컴파일이 실패한다 -> Append 하나만 쓰고 반복 호출
if NONCAUSAL_DRAFT:
    _block_module = _BlockNoncausal(draft, KV_BLOCK_SIZE).eval()
else:
    _block_module = (_BlockTrace(draft, KV_BLOCK_SIZE).eval()
                     if DRAFT_TRACE else BlockRR(draft, KV_BLOCK_SIZE).eval())
cm_b = rebel.compile_from_torch(_block_module, input_info=binfo,
                                example_inputs=exs(binfo), compile_context=ctxc,
                                **_dtp)
cm_r = None
if READ_DRAFT_CACHE:
    cm_r = rebel.compile_from_torch(
        _CacheReader().eval(),
        input_info=cinfo[:2],
        example_inputs=caches[:2],
        compile_context=ctxc,
    )
_ltp = {"tensor_parallel_size": LMH_TP} if LMH_TP > 1 else {}
lcm = rebel.compile_from_torch(nn.Sequential(lmw).eval(),
                              input_info=[("x", [1, B, H], DTS)], **_ltp)
scm = None
if STATELESS_CONTEXT:
    scm = rebel.compile_from_torch(
        _StatelessDraft(draft, STATELESS_CONTEXT).eval(),
        input_info=[
            ("noise_embedding", [1, B, H], DTS),
            ("target_hidden", [1, STATELESS_CONTEXT, NT * H], DTS),
            ("position_ids", [1, STATELESS_CONTEXT + B], "int64"),
            ("attention_mask", [1, 1, B, STATELESS_CONTEXT + B], DTS),
        ],
    )
print("COMPILED %.1fs  CMR=%d INLEN=%d" % (time.time() - t0, CMR, INLEN), flush=True)
rt_a = rebel.Runtime(cm_a, tensor_type="pt", device=DRAFT_DEVICES)
rt_b = rebel.Runtime(cm_b, tensor_type="pt", device=DRAFT_DEVICES)
rrt = rebel.Runtime(cm_r, tensor_type="pt", device=DEV_DRAFT) if cm_r is not None else None
lrt = rebel.Runtime(lcm, tensor_type="pt", device=LMH_DEVICES)
srt = rebel.Runtime(scm, tensor_type="pt", device=DEV_DRAFT) if scm is not None else None
STOP = {tok.eos_token_id, 151645}
DSET = os.environ.get("DSET", "gsm8k")
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "0"))
NATURAL_LONG = int(os.environ.get("NATURAL_LONG", "0"))
DRAFT_DIAG = int(os.environ.get("DRAFT_DIAG", "0"))
_ds = [json.loads(l) for l in
       open("/home/work/npu_work/dflash_work/dflash/cache/%s.jsonl" % DSET)]


CORPUS = os.environ.get("CORPUS", "")
CORPUS_LEN = os.environ.get("CORPUS_LEN", "")
_CORPUS_SAMPLES = None
if CORPUS:
    _blob = json.load(open("/home/work/npu_work/dflash_work/corpus_prompts.json"))
    if CORPUS not in _blob:
        raise SystemExit("코퍼스 없음: %s (있는 것: %s)" % (CORPUS, list(_blob)))
    if CORPUS_LEN not in _blob[CORPUS]:
        raise SystemExit("길이 없음: %s (있는 것: %s)" % (CORPUS_LEN, list(_blob[CORPUS])))
    _CORPUS_SAMPLES = _blob[CORPUS][CORPUS_LEN]
    print("corpus=%s len=%s samples=%d 토큰=%s"
          % (CORPUS, CORPUS_LEN, len(_CORPUS_SAMPLES),
             [len(s) for s in _CORPUS_SAMPLES]), flush=True)


def make_prompt(idx, _n=None):
    if _CORPUS_SAMPLES is not None:
        return torch.tensor(_CORPUS_SAMPLES[idx % len(_CORPUS_SAMPLES)],
                            dtype=torch.long).unsqueeze(0)
    ex = _ds[idx % len(_ds)]
    if NATURAL_LONG:
        ordered = [_ds[(idx + offset) % len(_ds)]["turns"][0]
                   for offset in range(len(_ds))]
        document = "Mathematics problem collection:\n\n" + "\n\n".join(
            "%d. %s" % (number + 1, text) for number, text in enumerate(ordered))
        document += ("\n\nSummarize the recurring problem types and the main reasoning "
                     "strategies needed to solve this collection.")
        low, high, best = 1, len(document), None
        while low <= high:
            middle = (low + high) // 2
            candidate = tok.apply_chat_template(
                [{"role": "user", "content": document[:middle]}],
                add_generation_prompt=True, return_tensors="pt", enable_thinking=False)
            if candidate.shape[1] <= PROMPT_LEN:
                best = candidate; low = middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ValueError("PROMPT_LEN is too small")
        return best
    ids = tok.apply_chat_template(
        [{"role": "user", "content": ex["turns"][0]}],
        add_generation_prompt=True, return_tensors="pt", enable_thinking=False)
    if PROMPT_LEN:
        if ids.shape[1] > PROMPT_LEN:
            raise ValueError("prompt already exceeds PROMPT_LEN")
        pattern = torch.tensor(tok.encode(" Background context.", add_special_tokens=False),
                               dtype=ids.dtype).unsqueeze(0)
        need = PROMPT_LEN - ids.shape[1]
        repeats = (need + pattern.shape[1] - 1) // pattern.shape[1]
        prefix = pattern.repeat(1, repeats)[:, :need]
        ids = torch.cat([prefix, ids], dim=1)
    return ids


def run(warm=False):
    T = dict(draft=0.0, verify=0.0, lmh=0.0, score=0.0, prefill=0.0, append=0.0)
    nret = 0; kept = 0

    def lmh(x):
        n = x.shape[1]
        if n < B: x = torch.cat([x, torch.zeros(1, B - n, H, dtype=x.dtype)], dim=1)
        s = time.time()
        _o = lrt(x.contiguous())
        o = torch.as_tensor(_o[0] if isinstance(_o, (list, tuple)) else _o)
        T["lmh"] += time.time() - s
        return torch.argmax(o[:, :n].float(), dim=-1)

    def tpre(seg, off):
        s = time.time(); L = seg.shape[1]; pad = 1 if L % CH_V == 0 else 0
        if pad: seg = torch.cat([seg, seg[:, -1:]], dim=1)
        r = DUAL[CH_V].prefill_forward(seg, cache_position=torch.arange(off, off + seg.shape[1],
                                 dtype=torch.int32).unsqueeze(0),
                                 attention_mask=torch.ones(seg.shape[1], dtype=torch.int64),
                                 batch_idx=0, block_tables=TARGET_BT, is_external_block_tables=False)
        T["verify"] += time.time() - s; hs = r.hidden_states[:6]   # 레이어 35 포함
        hs = tuple(h[:, :L] for h in hs) if pad else hs
        return r.logits[:, :L], hs

    def twh(ids):
        s = time.time(); L = ids.shape[1]
        if L % CH_P == 0:
            # Do not append a duplicate token: that advances the target KV cache
            # one position beyond the draft cache. Fill through L-2 with the
            # prefill graph, then write the final real token with verify-17.
            prefix = ids[:, :-1]
            r0 = DUAL[CH_P].prefill_forward(
                prefix,
                cache_position=torch.arange(L - 1, dtype=torch.int32).unsqueeze(0),
                attention_mask=torch.ones(L - 1, dtype=torch.int64), batch_idx=0,
                block_tables=TARGET_BT, is_external_block_tables=False)
            r1 = DUAL[CH_V].prefill_forward(
                ids[:, -1:],
                cache_position=torch.tensor([[L - 1]], dtype=torch.int32),
                attention_mask=torch.ones(1, dtype=torch.int64), batch_idx=0,
                block_tables=TARGET_BT, is_external_block_tables=False)
            hs = tuple(torch.cat([r0.hidden_states[i], r1.hidden_states[i]], dim=1)
                       for i in range(6))   # 레이어 35 포함
            logits = r1.logits
        else:
            r = DUAL[CH_P].prefill_forward(
                ids, cache_position=torch.arange(L, dtype=torch.int32).unsqueeze(0),
                attention_mask=torch.ones(L, dtype=torch.int64), batch_idx=0,
                block_tables=TARGET_BT, is_external_block_tables=False)
            hs = r.hidden_states[:6]   # 레이어 35 포함
            logits = r.logits
        T["prefill"] += time.time() - s
        return logits, hs

    def append(th_src, positions, at):
        """th_src [1,n,NT*H] 를 캐시 위치 at 부터 기록. n<=BULK 를 자동 분할."""
        n = th_src.shape[1]; done = 0
        while done < n:
            take = min(APPEND_WIDTH, n - done)
            buf = torch.zeros(1, APPEND_WIDTH, NT * H, dtype=DT)
            buf[:, :take] = th_src[:, done:done + take]
            pos = torch.zeros(1, APPEND_WIDTH, dtype=torch.int32)
            pos[:, :take] = positions[done:done + take].to(torch.int32)
            s = time.time()
            _ang = host_angle(pos, INV_HOST, DT)
            rt_a(buf, _ang, torch.tensor([[at + done]], dtype=torch.int32), BT)
            T["append"] += time.time() - s
            done += take

    acc = []; ntok = 0
    for si in range(1 if warm else NSAMP):
        ids = make_prompt(si)
        P = ids.shape[1]
        prefill_logits, hs = twh(ids)
        bonus = int(torch.argmax(prefill_logits[0, -1].float()))
        CAP = P + (WARMNEW if warm else MAXNEW) + B + 16
        THB = torch.zeros(1, CAP, NT * H, dtype=DT)
        VRB = torch.zeros(1, CAP, dtype=torch.long)
        _t0 = torch.cat([hs[i] for i in range(5)], dim=-1).to(DT); TL = _t0.shape[1]
        THB[:, :TL] = _t0; VRB[:, :P] = ids; VL = P
        cached = (P // CHUNK) * CHUNK
        KST = None; sel = None; step = 0; KSTB = None; KLEN = 0
        if CMR:
            s = time.time()
            # 링버퍼로 미리 잡는다. torch.cat 으로 키우면 라운드마다 저장소 전체를
            # 재할당한다 (16K 에서 67 MB). SpecExtend 도 init_caches 에서 미리 잡는다.
            _k0 = scorer.keys(hs[5], torch.arange(P))
            KSTB = torch.empty(CAP, _k0.shape[1], _k0.shape[2], dtype=_k0.dtype)
            KSTB[:_k0.shape[0]] = _k0; KLEN = _k0.shape[0]; KST = KSTB[:KLEN]
            nq = min(B, P)
            q0 = scorer.queries(hs[5][:, P - nq:P], torch.arange(P - nq, P))
            sel, _ = pick_chunks(scorer.scores(q0, KST), TL, BUDGET, CHUNK_SZ, TOPK)
            nret += 1; kept += sel.numel(); T["score"] += time.time() - s
            append(THB[:, sel], torch.arange(sel.numel()), 0)   # 압축 위치로 기록
            ctx_len = sel.numel()
        else:
            if not USE_STATELESS_DRAFT:
                append(THB[:, :TL], torch.arange(TL), 0)
            ctx_len = TL
        if READ_DRAFT_CACHE and not warm and si == 0:
            _reader_output = rrt()
            _read_key, _read_value = [
                torch.as_tensor(_value).float() for _value in _reader_output[:2]
            ]
            with torch.no_grad():
                _read_context = draft.hidden_norm(draft.fc(THB[:, :TL]))
                _read_attn = draft.layers[0].self_attn
                _expected_key = _read_attn.k_norm(
                    _read_attn.k_proj(_read_context).view(1, TL, -1, HD)
                ).transpose(1, 2)
                _read_pos = torch.arange(TL).unsqueeze(0)
                _read_freq = _read_pos.float().unsqueeze(-1) * (
                    1.0
                    / (cfg.rope_theta ** (torch.arange(0, HD, 2).float() / HD))
                ).view(1, 1, -1)
                _read_embedding = torch.cat([_read_freq, _read_freq], dim=-1)
                _expected_key = rope(
                    _expected_key, _read_embedding.cos(), _read_embedding.sin()
                ).float()
                _expected_value = _read_attn.v_proj(_read_context).view(
                    1, TL, -1, HD
                ).transpose(1, 2).float()

            def _cache_metrics(_actual, _expected):
                _actual = _actual[:, :, :TL]
                _cos = torch.nn.functional.cosine_similarity(
                    _actual, _expected, dim=-1
                )
                _rel = (_actual - _expected).norm(dim=-1) / _expected.norm(
                    dim=-1
                ).clamp_min(1e-12)
                return float(_cos.min()), float(_cos.mean()), float(_rel.max())

            _key_metrics = _cache_metrics(_read_key, _expected_key)
            _value_metrics = _cache_metrics(_read_value, _expected_value)
            print(
                "DRAFT_CACHE_DIAG context=%d "
                "key_cos_min=%.8f key_cos_mean=%.8f key_rel_max=%.6g "
                "value_cos_min=%.8f value_cos_mean=%.8f value_rel_max=%.6g"
                % (TL, *_key_metrics, *_value_metrics),
                flush=True,
            )
        while VL < P + (WARMNEW if warm else MAXNEW):
            blk = torch.full((1, B), MASK, dtype=torch.long); blk[0, 0] = bonus
            with torch.no_grad(): noi = embed(blk).detach()
            _save_draft_input = os.environ.get("SAVE_DRAFT_INPUT")
            if _save_draft_input and not warm and si == 0 and VL == P:
                torch.save(
                    {
                        "target_hidden": THB[:, :TL].cpu(),
                        "noise_embedding": noi.cpu(),
                        "prompt_ids": ids.cpu(),
                    },
                    _save_draft_input,
                )
                print("DRAFT_INPUT_SAVED %s" % _save_draft_input, flush=True)
            s = time.time()
            if USE_STATELESS_DRAFT:
                if srt is None or TL > STATELESS_CONTEXT:
                    raise RuntimeError("stateless draft capacity exceeded")
                _stateless_target = torch.zeros(
                    1, STATELESS_CONTEXT, NT * H, dtype=DT
                )
                _stateless_target[:, :TL] = THB[:, :TL]
                _stateless_positions = torch.cat(
                    [
                        torch.arange(STATELESS_CONTEXT, dtype=torch.int64),
                        torch.arange(ctx_len, ctx_len + B, dtype=torch.int64),
                    ]
                ).unsqueeze(0)
                _stateless_mask = torch.zeros(
                    1, 1, B, STATELESS_CONTEXT + B, dtype=DT
                )
                _stateless_mask[:, :, :, TL:STATELESS_CONTEXT] = torch.finfo(DT).min
                _block_result = srt(
                    noi.contiguous(),
                    _stateless_target,
                    _stateless_positions,
                    _stateless_mask,
                )
            else:
                _ang = host_angle(
                    torch.arange(ctx_len, ctx_len + B, dtype=torch.int64).unsqueeze(0),
                    INV_HOST, DT)
                _block_args = [
                    noi,
                    _ang,
                    torch.tensor([[ctx_len]], dtype=torch.int32),
                    BT,
                ]
                if NONCAUSAL_DRAFT:
                    _block_mask = torch.zeros(1, 1, B, MAXC, dtype=DT)
                    _block_mask[:, :, :, :ctx_len + B] = 1
                    _block_args.append(_block_mask)
                _block_result = rt_b(*_block_args)
            T["draft"] += time.time() - s
            _trace = _block_result if isinstance(_block_result, (tuple, list)) else None
            h = torch.as_tensor(_block_result[-1] if _trace is not None else _block_result)
            if DRAFT_DIAG and not warm and si == 0 and VL == P:
                _cpu_layers = []
                _cpu_inputs = []
                _cpu_attentions = []
                _handles = [
                    _layer.register_forward_pre_hook(
                        lambda _module, args, kwargs, _store=_cpu_inputs: _store.append(
                            kwargs["hidden_states"].detach()
                        ),
                        with_kwargs=True,
                    )
                    for _layer in draft.layers
                ] + [
                    _layer.self_attn.register_forward_hook(
                        lambda _module, _args, output, _store=_cpu_attentions: _store.append(
                            output[0].detach()
                        )
                    )
                    for _layer in draft.layers
                ] + [
                    _layer.register_forward_hook(
                        lambda _module, _args, output, _store=_cpu_layers: _store.append(
                            output.detach()
                        )
                    )
                    for _layer in draft.layers
                ]
                with torch.no_grad():
                    ref_h = draft(
                        position_ids=torch.arange(ctx_len + B).unsqueeze(0),
                        attention_mask=None,
                        noise_embedding=noi,
                        target_hidden=THB[:, :TL],
                        past_key_values=None,
                        use_cache=False,
                        is_causal=False,
                    )
                for _handle in _handles:
                    _handle.remove()
                _padded_ref_h = None
                _unmasked_padded_ref_h = None
                if USE_STATELESS_DRAFT:
                    with torch.no_grad():
                        _padded_ref_h = draft(
                            position_ids=_stateless_positions,
                            attention_mask=_stateless_mask,
                            noise_embedding=noi,
                            target_hidden=_stateless_target,
                            past_key_values=None,
                            use_cache=False,
                            is_causal=False,
                        )
                        _unmasked_padded_ref_h = draft(
                            position_ids=_stateless_positions,
                            attention_mask=torch.zeros_like(_stateless_mask),
                            noise_embedding=noi,
                            target_hidden=_stateless_target,
                            past_key_values=None,
                            use_cache=False,
                            is_causal=False,
                        )
                with torch.no_grad():
                    npu_h = h.float()
                    cpu_h = ref_h.float()
                    per_token_cos = torch.nn.functional.cosine_similarity(
                        npu_h[0], cpu_h[0], dim=-1
                    )
                    rel = (npu_h[0] - cpu_h[0]).norm(dim=-1) / cpu_h[0].norm(
                        dim=-1
                    ).clamp_min(1e-12)
                    npu_tokens = torch.argmax(lmw(h.to(DT)).float(), dim=-1)
                    cpu_tokens = torch.argmax(lmw(ref_h.to(DT)).float(), dim=-1)
                print(
                    "DRAFT_DIAG cos_min=%.8f cos_mean=%.8f rel_max=%.6g "
                    "token_matches=%d/%d npu=%s cpu=%s"
                    % (
                        float(per_token_cos.min()),
                        float(per_token_cos.mean()),
                        float(rel.max()),
                        int((npu_tokens == cpu_tokens).sum()),
                        B,
                        npu_tokens.tolist(),
                        cpu_tokens.tolist(),
                    ),
                    flush=True,
                )
                if _padded_ref_h is not None:
                    _padded_cpu_h = _padded_ref_h.float()
                    _pad_vs_exact_cos = torch.nn.functional.cosine_similarity(
                        _padded_cpu_h[0], cpu_h[0], dim=-1
                    )
                    _pad_vs_exact_rel = (
                        _padded_cpu_h[0] - cpu_h[0]
                    ).norm(dim=-1) / cpu_h[0].norm(dim=-1).clamp_min(1e-12)
                    _npu_vs_pad_cos = torch.nn.functional.cosine_similarity(
                        npu_h[0], _padded_cpu_h[0], dim=-1
                    )
                    _npu_vs_pad_rel = (
                        npu_h[0] - _padded_cpu_h[0]
                    ).norm(dim=-1) / _padded_cpu_h[0].norm(dim=-1).clamp_min(1e-12)
                    _padded_tokens = torch.argmax(
                        lmw(_padded_ref_h.to(DT)).float(), dim=-1
                    )
                    _unmasked_cpu_h = _unmasked_padded_ref_h.float()
                    _npu_vs_unmasked_cos = torch.nn.functional.cosine_similarity(
                        npu_h[0], _unmasked_cpu_h[0], dim=-1
                    )
                    _npu_vs_unmasked_rel = (
                        npu_h[0] - _unmasked_cpu_h[0]
                    ).norm(dim=-1) / _unmasked_cpu_h[0].norm(dim=-1).clamp_min(1e-12)
                    print(
                        "STATELESS_PAD_CPU pad_vs_exact_cos_min=%.8f "
                        "pad_vs_exact_rel_max=%.6g pad_token_matches=%d/%d "
                        "npu_vs_pad_cos_min=%.8f npu_vs_pad_rel_max=%.6g "
                        "npu_pad_token_matches=%d/%d npu_vs_unmasked_cos_min=%.8f "
                        "npu_vs_unmasked_rel_max=%.6g padded_tokens=%s"
                        % (
                            float(_pad_vs_exact_cos.min()),
                            float(_pad_vs_exact_rel.max()),
                            int((_padded_tokens == cpu_tokens).sum()),
                            B,
                            float(_npu_vs_pad_cos.min()),
                            float(_npu_vs_pad_rel.max()),
                            int((npu_tokens == _padded_tokens).sum()),
                            B,
                            float(_npu_vs_unmasked_cos.min()),
                            float(_npu_vs_unmasked_rel.max()),
                            _padded_tokens.tolist(),
                        ),
                        flush=True,
                    )
                    for _mask_value in (-1.0e2, -1.0e4, -1.0e9):
                        _alternate_mask = torch.zeros_like(_stateless_mask)
                        _alternate_mask[:, :, :, TL:STATELESS_CONTEXT] = _mask_value
                        _alternate_h = torch.as_tensor(
                            srt(
                                noi.contiguous(),
                                _stateless_target,
                                _stateless_positions,
                                _alternate_mask,
                            )
                        ).float()
                        _alternate_cos = torch.nn.functional.cosine_similarity(
                            _alternate_h[0], cpu_h[0], dim=-1
                        )
                        _alternate_rel = (
                            _alternate_h[0] - cpu_h[0]
                        ).norm(dim=-1) / cpu_h[0].norm(dim=-1).clamp_min(1e-12)
                        _alternate_tokens = torch.argmax(
                            lmw(_alternate_h.to(DT)).float(), dim=-1
                        )
                        print(
                            "STATELESS_MASK value=%g cos_min=%.8f rel_max=%.6g "
                            "token_matches=%d/%d"
                            % (
                                _mask_value,
                                float(_alternate_cos.min()),
                                float(_alternate_rel.max()),
                                int((_alternate_tokens == cpu_tokens).sum()),
                                B,
                            ),
                            flush=True,
                        )
                if srt is not None and TL == STATELESS_CONTEXT:
                    _s0 = time.time()
                    _stateless_h = torch.as_tensor(
                        srt(
                            noi.contiguous(),
                            THB[:, :TL].contiguous(),
                            torch.arange(TL + B, dtype=torch.int64).unsqueeze(0),
                            torch.zeros(1, 1, B, TL + B, dtype=DT),
                        )
                    ).float()
                    _stateless_ms = (time.time() - _s0) * 1000
                    _stateless_cos = torch.nn.functional.cosine_similarity(
                        _stateless_h[0], cpu_h[0], dim=-1
                    )
                    _stateless_rel = (_stateless_h[0] - cpu_h[0]).norm(
                        dim=-1
                    ) / cpu_h[0].norm(dim=-1).clamp_min(1e-12)
                    _stateless_tokens = torch.argmax(
                        lmw(_stateless_h.to(DT)).float(), dim=-1
                    )
                    print(
                        "STATELESS_DIAG ms=%.3f cos_min=%.8f cos_mean=%.8f "
                        "rel_max=%.6g token_matches=%d/%d tokens=%s"
                        % (
                            _stateless_ms,
                            float(_stateless_cos.min()),
                            float(_stateless_cos.mean()),
                            float(_stateless_rel.max()),
                            int((_stateless_tokens == cpu_tokens).sum()),
                            B,
                            _stateless_tokens.tolist(),
                        ),
                        flush=True,
                    )
                if _trace is not None:
                    _layer0 = draft.layers[0]
                    _attn0 = _layer0.self_attn
                    with torch.no_grad():
                        _context = draft.hidden_norm(draft.fc(THB[:, :TL]))
                        _x0 = _layer0.input_layernorm(noi)
                        _q0 = _attn0.q_norm(
                            _attn0.q_proj(_x0).view(1, B, -1, HD)
                        ).transpose(1, 2)
                        _kc0 = _attn0.k_proj(_context).view(1, TL, -1, HD)
                        _kn0 = _attn0.k_proj(_x0).view(1, B, -1, HD)
                        _k0 = _attn0.k_norm(
                            torch.cat([_kc0, _kn0], dim=1)
                        ).transpose(1, 2)
                        _vc0 = _attn0.v_proj(_context).view(1, TL, -1, HD)
                        _vn0 = _attn0.v_proj(_x0).view(1, B, -1, HD)
                        _v0 = torch.cat([_vc0, _vn0], dim=1).transpose(1, 2)
                        _all_pos = torch.arange(TL + B).unsqueeze(0)
                        _cos0, _sin0 = draft.rotary_emb(_x0, _all_pos)
                        _q0 = rope(_q0, _cos0[:, -B:], _sin0[:, -B:])
                        _k0 = rope(_k0, _cos0, _sin0)
                        _k0 = _k0.repeat_interleave(_attn0.num_key_value_groups, dim=1)
                        _v0 = _v0.repeat_interleave(_attn0.num_key_value_groups, dim=1)
                        _scores = torch.matmul(_q0, _k0.transpose(-2, -1)) * _attn0.scaling
                        _weights_bidir = torch.softmax(_scores, dim=-1, dtype=torch.float32)
                        _future = torch.triu(
                            torch.ones(B, B, dtype=torch.bool), diagonal=1
                        )
                        _causal_scores = _scores.clone()
                        _causal_scores[:, :, :, TL:] = _causal_scores[:, :, :, TL:].masked_fill(
                            _future.unsqueeze(0).unsqueeze(0), -torch.inf
                        )
                        _weights_causal = torch.softmax(
                            _causal_scores, dim=-1, dtype=torch.float32
                        )

                        def _attention_residual(_weights):
                            _value = torch.matmul(_weights, _v0)
                            _value = _value.transpose(1, 2).reshape(1, B, -1)
                            return noi + _attn0.o_proj(_value)

                        _manual_bidir = _attention_residual(_weights_bidir)[0]
                        _manual_causal = _attention_residual(_weights_causal)[0]
                        _npu_post0 = torch.as_tensor(_trace[0]).float()[0]
                        _official_post0 = (
                            _cpu_inputs[0] + _cpu_attentions[0]
                        ).float()[0]
                    for _name, _manual in (
                        ("bidir", _manual_bidir),
                        ("causal", _manual_causal),
                        ("official", _official_post0),
                    ):
                        _manual_cos = torch.nn.functional.cosine_similarity(
                            _npu_post0, _manual.float(), dim=-1
                        )
                        _manual_rel = (_npu_post0 - _manual.float()).norm(
                            dim=-1
                        ) / _manual.float().norm(dim=-1).clamp_min(1e-12)
                        print(
                            "DRAFT_ATTN_MODE mode=%s cos_min=%.8f cos_mean=%.8f rel_max=%.6g"
                            % (
                                _name,
                                float(_manual_cos.min()),
                                float(_manual_cos.mean()),
                                float(_manual_rel.max()),
                            ),
                            flush=True,
                        )
                    for _layer_index, _cpu_layer in enumerate(_cpu_layers):
                        _npu_attention = torch.as_tensor(
                            _trace[2 * _layer_index]
                        ).float()[0]
                        _cpu_attention = (
                            _cpu_inputs[_layer_index] + _cpu_attentions[_layer_index]
                        ).float()[0]
                        _attn_cos = torch.nn.functional.cosine_similarity(
                            _npu_attention, _cpu_attention, dim=-1
                        )
                        _attn_rel = (_npu_attention - _cpu_attention).norm(
                            dim=-1
                        ) / _cpu_attention.norm(dim=-1).clamp_min(1e-12)
                        _npu_layer = torch.as_tensor(
                            _trace[2 * _layer_index + 1]
                        ).float()[0]
                        _cpu_layer = _cpu_layer.float()[0]
                        _cos = torch.nn.functional.cosine_similarity(
                            _npu_layer, _cpu_layer, dim=-1
                        )
                        _rel = (_npu_layer - _cpu_layer).norm(dim=-1) / _cpu_layer.norm(
                            dim=-1
                        ).clamp_min(1e-12)
                        print(
                            "DRAFT_LAYER layer=%d attn_cos_min=%.8f attn_rel_max=%.6g "
                            "full_cos_min=%.8f full_cos_mean=%.8f full_rel_max=%.6g"
                            % (
                                _layer_index,
                                float(_attn_cos.min()),
                                float(_attn_rel.max()),
                                float(_cos.min()),
                                float(_cos.mean()),
                                float(_rel.max()),
                            ),
                            flush=True,
                        )
            blk[:, 1:] = lmh(h[:, 1 - B:, :].to(DT))
            vl = VL; c0 = VL   # 재개 지점 = 현재 위치 (vl-c0 = 0)
            seg = blk          # 비정렬 재개: 꼬리 없이 블록만
            post_logits, hs2 = tpre(seg, c0)
            post = torch.argmax(post_logits[:, vl - c0:vl - c0 + B].float(), dim=-1)
            a = int((blk[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
            acc.append(a + 1)
            nf = torch.cat([hs2[i] for i in range(5)], dim=-1)[:, vl - c0:vl - c0 + a + 1, :].to(DT)
            THB[:, TL:TL + nf.shape[1]] = nf
            newpos = torch.arange(ctx_len, ctx_len + a + 1)
            if not USE_STATELESS_DRAFT:
                append(nf, newpos, ctx_len)
            TL += nf.shape[1]; ctx_len += a + 1
            VRB[:, VL:VL + a + 1] = blk[:, :a + 1]; VL += a + 1
            bonus = int(post[0, a]); cached = (VL // CHUNK) * CHUNK
            if CMR and not CMR_ONCE:
                s = time.time()
                _kn = scorer.keys(hs2[5][:, vl - c0:vl - c0 + a + 1],
                                  torch.arange(vl, vl + a + 1))
                KSTB[KLEN:KLEN + _kn.shape[0]] = _kn
                KLEN += _kn.shape[0]; KST = KSTB[:KLEN]
                step += 1
                if step % EVERY == 0:
                    q = scorer.queries(hs2[5][:, vl - c0:vl - c0 + a + 1],
                                       torch.arange(vl, vl + a + 1))
                    sel, _ = pick_chunks(scorer.scores(q, KST), TL, BUDGET, CHUNK_SZ, TOPK)
                    nret += 1; kept += sel.numel()
                    T["score"] += time.time() - s
                    append(THB[:, sel], torch.arange(sel.numel()), 0)
                    ctx_len = sel.numel()
                else:
                    T["score"] += time.time() - s
            if any(int(t) in STOP for t in VRB[0, P:VL]): break
            if ctx_len >= MAXC - 2 * B: break
        ntok += VL - P
    return acc, ntok, T, nret, kept


run(warm=True)
_PWR_DEVS = list(dict.fromkeys(
    (TARGET_DEVICES if isinstance(TARGET_DEVICES, list) else [TARGET_DEVICES])
    + (DRAFT_DEVICES if isinstance(DRAFT_DEVICES, list) else [DRAFT_DEVICES])))
print("power sampling devices: %s" % _PWR_DEVS, flush=True)
with Power(_PWR_DEVS) as p0: time.sleep(5)
IDLE = p0.stats().get("P_mean", 0.0)
with Power(_PWR_DEVS) as pw:
    acc, ntok, T, nret, kept = run()
st = pw.stats(idle=IDLE)
n = max(len(acc), 1)
r = dict(mode="stateful", dset=DSET, cmr=CMR, inlen=INLEN,
         target_device=TARGET_DEVICES, target_tp=TARGET_TP,
         draft_device=DRAFT_DEVICES, draft_tp=DRAFT_TP, lmh_tp=LMH_TP,
         prompt_len=PROMPT_LEN or None,
         natural_long=NATURAL_LONG,
         samples=NSAMP, tau=round(sum(acc) / n, 3),
         rounds=len(acc), tokens=ntok,
         draft_ms=round(1000 * T["draft"] / n, 1), verify_ms=round(1000 * T["verify"] / n, 1),
         append_ms=round(1000 * T["append"] / n, 1), score_ms=round(1000 * T["score"] / n, 1),
         lmh_ms=round(1000 * T["lmh"] / n, 1), prefill_s=round(T["prefill"], 2),
         retrievals=nret, kept_mean=round(kept / max(nret, 1), 1), cmr_once=CMR_ONCE,
         decode_tok_s=round(ntok / max(st["wall_s"] - T["prefill"], 1e-6), 2), **st)
print("SF " + json.dumps(r), flush=True)
print("SF_DONE", flush=True)

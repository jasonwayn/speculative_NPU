"""Compile shared-cache target graphs with fused all-position verify logits."""
import os
import time
import traceback

import torch
import torch.nn as nn
import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as CF

_Config = CF.RBLNDecoderOnlyModelForCausalLMConfig
_orig_init = _Config.__init__


def _init(self, *args, **kwargs):
    requested = kwargs.get("prefill_chunk_size")
    if requested is not None and requested % 64:
        kwargs = dict(kwargs)
        kwargs["prefill_chunk_size"] = 64
        _orig_init(self, *args, **kwargs)
        self.prefill_chunk_size = requested
    else:
        _orig_init(self, *args, **kwargs)


_Config.__init__ = _init

from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.configuration_utils import RBLNCompileConfig
from transformers import AutoConfig, AutoModelForCausalLM

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
MAXC = int(os.environ.get("MAXC", "4096"))
TP = int(os.environ.get("TP", "1"))
OUTDIR = os.environ.get(
    "OUTDIR", "/home/work/npu_work/dflash_work/fused_17_256"
)
CHUNKS = tuple(int(value) for value in os.environ.get("CHUNKS", "17,256").split(","))
KEEP = tuple(int(value) for value in os.environ.get("KEEP", "2,10,18,26,34").split(","))
WSHARE = int(os.environ.get("WSHARE", "1"))   # 0 이면 가중치 공유를 끈다
SCORE_LAYER = int(os.environ.get("SCORE_LAYER", "35"))
SCORE_CHUNK = int(os.environ.get("SCORE_CHUNK", "32"))
SCORE_MODE = os.environ.get("SCORE_MODE", "bmm")    # bmm 이 유일하게 컴파일된다.
# full(5D 브로드캐스트 matmul) 은 IndexError: map::at 로 죽는다 -> checks/ 의 이분 기록
SCORE = {}          # 트레이스 중 계산된 점수를 여기 담아 FusedOutputs 가 집어간다
# 점수는 verify 그래프(청크 17)에만 붙인다. 프리필(청크 256)은 프롬프트당 한 번이라
# CMR 검색과 무관하고, 거기에 넣으면 중간 텐서가 [8, 4*256, 20480] = 671 MB 로 커진다.
SCORE_ON = [True]
os.makedirs(OUTDIR, exist_ok=True)


class AccurateSilu(nn.Module):
    def forward(self, value):
        return value / (1.0 + torch.exp(-torch.clamp(value, -20.0, 20.0)))


def chunk_scores(q, K, seq_pos, scale, use_mask=True, use_softmax=True):
    """q [1,nh,L,hd] (RoPE 후), K 캐시 -> [1, S/SCORE_CHUNK] 청크 평균 어텐션 확률.

    호스트 Scorer.scores 와 같은 정의: 헤드·쿼리에 대한 softmax 확률의 평균.
    fused op 호출 **전**의 캐시를 읽으므로 현재 블록은 안 들어간다 — 호스트판이
    확정된 prefix 만 스코어링하는 것과 같다.
    """
    b, nh, L, hd = q.shape
    if K.dim() != 4:
        raise RuntimeError("unexpected kv cache rank %d shape %s" % (K.dim(), tuple(K.shape)))
    if K.shape[0] != 1:
        raise RuntimeError("KV_BLOCK_SIZE must equal MAXC (single block); got %s"
                           % (tuple(K.shape),))
    nkv, S = K.shape[1], K.shape[2]
    rep = nh // nkv
    kb = K.view(1, nkv, 1, S, hd).to(q.dtype)
    # 마스크는 스칼라 브로드캐스트로 만든다. torch.full_like(a, ...) 로 하면
    # [1,nkv,rep,L,S] 크기(17 x 20480 에서 1100 만 원소)의 상수가 생겨
    # 컴파일러가 IndexError: map::at 로 죽는다. checks/score_graph_probe.py 형태.
    a = torch.matmul(q.view(1, nkv, rep, L, hd), kb.transpose(-1, -2)) * scale
    if use_mask:
        idx = torch.arange(S, dtype=torch.int32).view(1, 1, 1, 1, S)
        a = a + torch.where(idx < seq_pos.reshape(1, 1, 1, 1, 1).to(torch.int32),
                            torch.zeros((), dtype=q.dtype),
                            torch.full((), -1.0e4, dtype=q.dtype))
    p = torch.softmax(a, dim=-1) if use_softmax else a
    sc = p.mean(dim=(1, 2, 3))                    # [1,S]
    nc = S // SCORE_CHUNK
    print("SCORE_GRAPH q=%s(%s) K=%s(%s) -> [1,%d]"
          % (tuple(q.shape), q.device, tuple(K.shape), K.device, nc), flush=True)
    return sc[:, : nc * SCORE_CHUNK].view(1, nc, SCORE_CHUNK).mean(-1)


def chunk_scores_bmm(q, K, seq_pos, scale, use_mask=True, use_softmax=True, contig=False):
    """5D 브로드캐스트 대신 평범한 3D bmm. rep 는 쿼리 쪽에 접어 넣는다."""
    b, nh, L, hd = q.shape
    nkv, S = K.shape[1], K.shape[2]
    rep = nh // nkv
    qg = q.view(1, nkv, rep, L, hd).reshape(nkv, rep * L, hd)
    kg = K.view(nkv, S, hd).to(q.dtype).transpose(1, 2)      # [nkv, hd, S]
    if contig:
        kg = kg.contiguous()
    a = torch.bmm(qg, kg) * scale                            # [nkv, rep*L, S]
    if use_mask:
        idx = torch.arange(S, dtype=torch.int32).view(1, 1, S)
        a = a + torch.where(idx < seq_pos.reshape(1, 1, 1).to(torch.int32),
                            torch.zeros((), dtype=q.dtype),
                            torch.full((), -1.0e4, dtype=q.dtype))
    p = torch.softmax(a, dim=-1) if use_softmax else a
    sc = p.mean(dim=(0, 1)).view(1, S)
    nc = S // SCORE_CHUNK
    return sc[:, : nc * SCORE_CHUNK].view(1, nc, SCORE_CHUNK).mean(-1)


def _attn_tail(self, query_states, key_states, value_states, attention_mask,
               past_key_values, seq_positions, block_tables, lora_int_id):
    k_scale, v_scale = self.maybe_get_kvcache_scale()
    attn_output = self.get_attention_op()(
        query_states, key_states, value_states, attention_mask,
        past_key_state=past_key_values[self.layer_idx][0],
        past_value_state=past_key_values[self.layer_idx][1],
        seq_position=seq_positions, scale=self.scale, block_tables=block_tables,
        block_size=self.kvcache_block_size, k_scale=k_scale, v_scale=v_scale,
        s_aux=getattr(self, "sinks", None))
    if self.lora_config:
        return self.o_proj(attn_output, lora_int_id)
    return self.o_proj(attn_output)


def scoring_forward(self, hidden_states, attention_mask, seq_positions, past_key_values,
                    cos=None, sin=None, block_tables=None, lora_int_id=None):
    """DecoderOnlyAttention.forward (optimum-rbln 0.10.2) + CMR 점수 계산."""
    batch_size, query_length, _ = hidden_states.size()
    query_states, key_states, value_states = self.projection(
        hidden_states=hidden_states, lora_int_id=lora_int_id)
    query_states = query_states.view(batch_size, query_length, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, query_length, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, query_length, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
    if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
    if cos is not None and sin is not None:
        query_states, key_states = self.apply_rotary_pos_embed(query_states, key_states, cos, sin)
    if batch_size > 1 and "prefill" in self.phase:
        raise NotImplementedError("batch size should be 1 if prefill phase")

    if not SCORE_ON[0]:
        return _attn_tail(self, query_states, key_states, value_states,
                          attention_mask, past_key_values, seq_positions,
                          block_tables, lora_int_id)
    K = past_key_values[self.layer_idx][0]
    NC = MAXC // SCORE_CHUNK
    if SCORE_MODE == "out":            # 추가 출력 배관만 시험 (캐시 안 읽음)
        SCORE["v"] = hidden_states.reshape(1, -1)[:, :NC]
    elif SCORE_MODE == "read":         # 정적 캐시를 일반 op 로 읽는 것만 시험
        SCORE["v"] = K[0, :, :NC, 0].mean(0).view(1, NC)
    elif SCORE_MODE == "nosm":         # softmax 제거
        SCORE["v"] = chunk_scores(query_states, K, seq_positions, self.scale, use_softmax=False)
    elif SCORE_MODE == "nomask":       # arange/where 제거
        SCORE["v"] = chunk_scores(query_states, K, seq_positions, self.scale, use_mask=False)
    elif SCORE_MODE == "bare":         # matmul + mean 만
        SCORE["v"] = chunk_scores(query_states, K, seq_positions, self.scale,
                                  use_mask=False, use_softmax=False)
    elif SCORE_MODE == "small":        # 컨텍스트 2048 만 (크기 의존 확인)
        SCORE["v"] = chunk_scores(query_states, K[:, :, :2048], seq_positions, self.scale)
    elif SCORE_MODE == "clone":        # 정적 텐서 별칭 회피
        SCORE["v"] = chunk_scores(query_states, torch.clone(K), seq_positions, self.scale)
    elif SCORE_MODE == "bmmbare":      # 3D bmm, 마스크/softmax 없음
        SCORE["v"] = chunk_scores_bmm(query_states, K, seq_positions, self.scale,
                                      use_mask=False, use_softmax=False)
    elif SCORE_MODE == "bmm":          # 3D bmm, 전체
        SCORE["v"] = chunk_scores_bmm(query_states, K, seq_positions, self.scale)
    elif SCORE_MODE == "bmmc":         # 3D bmm, K 전치본을 contiguous 로
        SCORE["v"] = chunk_scores_bmm(query_states, K, seq_positions, self.scale, contig=True)
    else:
        SCORE["v"] = chunk_scores(query_states, K, seq_positions, self.scale)
    print("SCORE_MODE=%s -> %s" % (SCORE_MODE, tuple(SCORE["v"].shape)), flush=True)

    return _attn_tail(self, query_states, key_states, value_states, attention_mask,
                      past_key_values, seq_positions, block_tables, lora_int_id)


def pack_like(sc, like):
    """[1,nc] 를 hidden state 와 같은 [1,L,H] 에 담는다. 런타임 출력 버퍼가
    hidden state 와 동일 shape 을 가정할 수 있어서 (기존 contract 해킹 참고)."""
    L, H = like.shape[1], like.shape[2]
    nc = sc.shape[1]
    if nc > L * H:
        raise RuntimeError("score %d does not fit in [1,%d,%d]" % (nc, L, H))
    pad = torch.zeros(1, L * H - nc, dtype=like.dtype)
    return torch.cat([sc.to(like.dtype), pad], dim=1).view(1, L, H)


class FusedOutputs(nn.Module):
    def __init__(self, wrapped, keep, all_position_logits):
        super().__init__()
        self.wrapped = wrapped
        self.keep = keep
        self.all_position_logits = all_position_logits

    @property
    def phase(self):
        return self.wrapped.phase

    @phase.setter
    def phase(self, value):
        self.wrapped.phase = value

    def forward(self, *args):
        if not self.all_position_logits:
            logits, hidden_states = self.wrapped(*args)
            return logits, tuple(hidden_states[i] for i in self.keep)

        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            rotary_emb,
        ) = self.wrapped.prepare_forward_args(*args)
        hidden, hidden_states = self.wrapped.model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=True,
        )
        logits = self.wrapped.model.lm_head(hidden)
        # Keep query_position in the compiled input contract expected by
        # RBLNRuntimeModel. Match the hidden-state output shape and dtype so the
        # stock runtime can allocate its output buffer normally.
        contract = torch.ones_like(hidden_states[self.keep[0]]) * query_position.to(
            hidden_states[self.keep[0]].dtype
        ).reshape(1, 1, 1)
        return logits, (tuple(hidden_states[i] for i in self.keep)
                        + (contract, pack_like(SCORE["v"], hidden_states[self.keep[0]])))


cls = RBLNQwen3ForCausalLM
mcfg = AutoConfig.from_pretrained(SRC)
print("loading target", flush=True)
MODEL_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}[os.environ.get("MODEL_DTYPE", "float32")]
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=MODEL_DTYPE)
if os.environ.get("ACCURATE_SILU") == "1":
    for decoder_layer in hf.model.layers:
        decoder_layer.mlp.act_fn = AccurateSilu()
    print("ACCURATE_SILU enabled layers=%d" % len(hf.model.layers), flush=True)
rc, _ = cls.prepare_rbln_config(rbln_config={
    "batch_size": 1,
    "max_seq_len": MAXC,
    "prefill_chunk_size": CHUNKS[0],
    "output_hidden_states": True,
    "create_runtimes": False,
    "tensor_parallel_size": TP,
})
rc.max_seq_len = MAXC
rc = cls._update_rbln_config(
    preprocessors=None, model=hf, model_config=mcfg, rbln_config=rc
)
base = cls._wrap_model_if_needed(hf, rc)

# --- 레이어 SCORE_LAYER 의 어텐션만 교체 -------------------------------------
def find_layers(root):
    for path in ("model.model.layers", "model.layers", "layers"):
        obj = root
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if len(obj) > SCORE_LAYER:
                print("layers at %s (n=%d)" % (path, len(obj)), flush=True)
                return obj
        except AttributeError:
            continue
    raise RuntimeError("decoder layers not found on %s" % type(root).__name__)

_layers = find_layers(base)
_attn = _layers[SCORE_LAYER].self_attn
print("patching %s.self_attn = %s" % (SCORE_LAYER, type(_attn).__name__), flush=True)
_attn.forward = scoring_forward.__get__(_attn, type(_attn))

ctx = None
static = None
for chunk in CHUNKS:
    dst = os.path.join(OUTDIR, "prefill_%d.rbln" % chunk)
    info = cls.get_input_info(
        batch_size=1, query_length=chunk, rbln_config=rc, model_config=mcfg
    )
    cc = RBLNCompileConfig(
        compiled_model_name="prefill_%d" % chunk,
        input_info=info,
        tensor_parallel_size=TP,
    )
    meta = [name for name, _, _ in cc.input_info if "past_key_values" in name]
    if ctx is None:
        examples = cc.get_dummy_inputs(fill=0)
        if WSHARE:
            ctx, static = cls._get_compile_context(cc, examples)
        else:
            # 점수 그래프가 캐시를 일반 op 로 읽으면 두 그래프 사이의 가중치 공유
            # 맵이 어긋나 두 번째 컴파일이 실패한다. 공유를 끄고 각자 컴파일한다.
            from rebel import CompileContext as _CC
            ctx = _CC(use_weight_sharing=False)
            static = {}
            for (_n, _sh, _dt), _t in zip(cc.input_info, examples):
                if "past_key_values" in _n:
                    static[_n] = _t
                    ctx.mark_static_address(_t, _n)
    else:
        examples = cc.get_dummy_inputs(fill=0, static_tensors=static)
    SCORE_ON[0] = (chunk == 17)      # verify 그래프에만 점수를 붙인다
    wrapped = FusedOutputs(base, KEEP, all_position_logits=(chunk == 17))
    started = time.time()
    try:
        compiled = cls._compile_model(wrapped, cc, examples, ctx, rc, phase="prefill")
        compiled.save(dst)
        print("COMPILE_OK chunk=%d sec=%.1f path=%s" % (
            chunk, time.time() - started, dst
        ), flush=True)
    except Exception as exc:
        print("COMPILE_FAIL chunk=%d %s: %s" % (
            chunk, type(exc).__name__, str(exc)[:500]
        ), flush=True)
        traceback.print_exc()
        raise
print("FUSED_COMPILE_DONE", flush=True)

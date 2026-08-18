"""tau 변화 원인 진단: prefill 청크 64 vs 256 이 첫 토큰을 바꾸는가.

dual 실행에서 tau 가 gsm8k 5.699->5.84, math500 3.81->4.295 로 변했다.
비정렬 재개(ua) 단계에서는 tau 가 sfp 와 완전히 동일했으므로, 바뀐 요인은
① 초기 prefill 청크 64 -> 256, ② verify 청크 64 -> 17 둘 중 하나다.

여기서는 ①만 격리한다: 같은 프롬프트를 두 그래프로 prefill 해서
마지막 위치 hidden state 와 그로부터 나오는 argmax 토큰을 비교한다.
토큰이 갈리면 그 시점부터 생성 궤적 전체가 달라지고 tau 도 달라진다 (버그 아님).
"""
import os, json, copy, torch, rebel

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
import optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils as RU
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNPageTableManager
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

D = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
TGT = os.path.join(D, "rbln-Qwen3-4B-h-c64")
DEVA = int(os.environ.get("DEVA", "1"))   # 청크64 모델
DEVB = int(os.environ.get("DEVB", "2"))   # 청크256 그래프 (카드 1장에 둘 다 안 올라감)
DSET = os.environ.get("DSET", "gsm8k")
N = int(os.environ.get("N", "20"))
BT = torch.tensor([0], dtype=torch.int16)

tok = AutoTokenizer.from_pretrained(SRC)
mcfg = AutoConfig.from_pretrained(SRC)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)

# lm_head 를 벤치와 동일하게 컴파일
H = mcfg.hidden_size
lmw = torch.nn.Linear(H, mcfg.vocab_size, bias=False)
lmw.weight.data = hf.lm_head.weight.data.to(torch.float16)
lrt = rebel.compile_from_torch(torch.nn.Sequential(lmw.eval()),
                               input_info=[("x", [1, 1, H], "float16")]).create_runtime(device=DEVA)
print("lm_head ready", flush=True)

# A: 기존 경로 (청크 64 컴파일 모델)
m64 = RBLNQwen3ForCausalLM.from_pretrained(TGT, export=False, rbln_device=DEVA)
p64 = m64.prefill_decoder
print("c64 ready chunk=%d" % p64.rbln_config.prefill_chunk_size, flush=True)

# B: dual 의 청크 256 그래프
rc, _ = RBLNQwen3ForCausalLM.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": 17,
    "output_hidden_states": True, "create_runtimes": False})
rc.max_seq_len = 4096
rc = RBLNQwen3ForCausalLM._update_rbln_config(preprocessors=None, model=hf,
                                              model_config=mcfg, rbln_config=rc)
ptm = RBLNPageTableManager(rc)
dam = torch.zeros(rc.batch_size, 1, 1, rc.max_seq_len, dtype=torch.float32)
cm = rebel.RBLNCompiledModel(os.path.join(D, "dual_17_256", "prefill_256.rbln"))
cfg = copy.deepcopy(rc); cfg.prefill_chunk_size = 256
p256 = RU.RBLNRuntimeModel(runtime=rebel.Runtime(cm, tensor_type="pt", device=DEVB),
                           phase="prefill", batch_size=rc.batch_size, rbln_config=cfg,
                           main_input_name="input_ids", embed_tokens=hf.model.embed_tokens,
                           dec_attn_mask=dam, page_table_manager=ptm, config=mcfg)
del hf
print("c256 ready", flush=True)

ds = [json.loads(l) for l in open(os.path.join(D, "dflash", "cache", "%s.jsonl" % DSET))]


def last_hs_64(ids):
    L = ids.shape[1]
    pad = 1 if L % 64 == 0 else 0
    i2 = torch.cat([ids, ids[:, -1:]], dim=1) if pad else ids
    hs = m64(input_ids=i2, attention_mask=torch.ones_like(i2)).hidden_states
    return hs[-1][:, L - 1:L]


def last_hs_256(ids):
    L = ids.shape[1]
    pad = 1 if L % 256 == 0 else 0
    i2 = torch.cat([ids, ids[:, -1:]], dim=1) if pad else ids
    hs = p256.prefill_forward(
        i2, cache_position=torch.arange(i2.shape[1], dtype=torch.int32).unsqueeze(0),
        attention_mask=torch.ones(i2.shape[1], dtype=torch.int64), batch_idx=0,
        block_tables=BT, is_external_block_tables=False).hidden_states
    return hs[-1][:, L - 1:L]


def argmax_tok(h):
    o = torch.as_tensor(lrt(h.to(torch.float16).contiguous().numpy()))
    return int(torch.argmax(o[0, 0].float()))


same = 0; cos_min = 1.0
for i in range(N):
    ids = tok.apply_chat_template([{"role": "user", "content": ds[i]["turns"][0]}],
                                  add_generation_prompt=True, return_tensors="pt")
    a = last_hs_64(ids).float(); b = last_hs_256(ids).float()
    c = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    ta, tb = argmax_tok(a), argmax_tok(b)
    cos_min = min(cos_min, c)
    same += (ta == tb)
    print("p%-3d len=%-5d cos=%.6f  tok64=%-7d tok256=%-7d %s"
          % (i, ids.shape[1], c, ta, tb, "same" if ta == tb else "DIFF"), flush=True)
print("SUMMARY same=%d/%d  cos_min=%.6f" % (same, N, cos_min), flush=True)
print("TAUDIAG_DONE", flush=True)

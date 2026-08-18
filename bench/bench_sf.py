"""상태 유지 드래프트 위에서 stock vs CMR 재측정.

드래프트가 디바이스 KV 캐시를 재사용하므로(무상태 대비 16K 에서 11.7배 빠름) 이제
GPU 와 공정한 비교가 된다. CMR 은 선택된 위치를 압축 위치로 캐시에 다시 기록해
SpecExtend 의 위치 재인덱싱을 유지하고 어텐션 범위도 실제로 줄인다.
"""
import sys, os, json, time, threading, subprocess, math, torch, rebel
sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")
from torch import nn
from rebel import CompileContext
from transformers import AutoTokenizer, AutoConfig
from optimum.rbln import RBLNQwen3ForCausalLM
from dflash.model import DFlashDraftModel, extract_context_feature
from draft_two import Append, Block
from safetensors import safe_open
import glob as _glob

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
TGT = os.environ.get("TGTDIR", "/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-16k")
DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
B = 16
DEV = int(os.environ.get("DEV", "0"))
MAXNEW = int(os.environ.get("MAXNEW", "256")); NSAMP = int(os.environ.get("NSAMP", "3"))
INLEN = int(os.environ.get("INLEN", "4096"))
CMR = int(os.environ.get("CMR", "0"))
CHUNK_SZ = int(os.environ.get("CMR_CHUNK", "32")); TOPK = int(os.environ.get("CMR_TOPK", "32"))
EVERY = int(os.environ.get("CMR_EVERY", "4")); BUDGET = int(os.environ.get("CMR_BUDGET", "1024"))
MAXC = int(os.environ.get("MAXC", "16384"))
DT = torch.float16; DTS = "float16"
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
sys.path.insert(0, "/home/work/npu_work/dflash_work")
from bench_cmr_parts import Power, Scorer, pick_chunks   # 재사용


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

m = RBLNQwen3ForCausalLM.from_pretrained(TGT, export=False, rbln_device=DEV)
pdec = m.prefill_decoder; CHUNK = pdec.rbln_config.prefill_chunk_size
cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size; NT = len(draft.target_layer_ids); LID = draft.target_layer_ids
NL = cfg.num_hidden_layers; NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
MASK = draft.mask_token_id; BT = torch.tensor([0], dtype=torch.int16)
scorer = Scorer(SRC) if CMR else None
BULK = int(os.environ.get("BULK", "256"))   # prefill op 의 q_len 한계 회피

caches = [torch.zeros(1, NKV, MAXC, HD, dtype=DT) for _ in range(2 * NL)]
cinfo = [("past_key_values_%d" % i, [1, NKV, MAXC, HD], DTS) for i in range(2 * NL)]
ctxc = CompileContext(use_weight_sharing=True)
for (n, _, _), t in zip(cinfo, caches): ctxc.mark_static_address(t, n)


def ainfo(A):
    return [("th_new", [1, A, NT * H], DTS), ("pos_ctx", [1, A], "int32"),
            ("seq_ctx", [1, 1], "int32"), ("block_tables", [1], "int16")] + cinfo


binfo = [("noise_emb", [1, B, H], DTS), ("pos_blk", [1, B], "int32"),
         ("seq_blk", [1, 1], "int32"), ("block_tables", [1], "int16")] + cinfo


def exs(info):
    o = []
    for n, sh, dt in info:
        o.append(caches[int(n.rsplit("_", 1)[1])] if n.startswith("past_key_values_")
                 else torch.zeros(*sh, dtype=getattr(torch, dt)))
    return o


t0 = time.time()
cm_a = rebel.compile_from_torch(Append(draft, MAXC).eval(), input_info=ainfo(B),
                                example_inputs=exs(ainfo(B)), compile_context=ctxc)
# 같은 컨텍스트로 세 번째 그래프를 만들면 컴파일이 실패한다 -> Append 하나만 쓰고 반복 호출
cm_b = rebel.compile_from_torch(Block(draft, MAXC).eval(), input_info=binfo,
                                example_inputs=exs(binfo), compile_context=ctxc)
lcm = rebel.compile_from_torch(nn.Sequential(lmw).eval(), input_info=[("x", [1, B, H], DTS)])
print("COMPILED %.1fs  CMR=%d INLEN=%d" % (time.time() - t0, CMR, INLEN), flush=True)
rt_a = rebel.Runtime(cm_a, tensor_type="pt", device=DEV)
rt_b = rebel.Runtime(cm_b, tensor_type="pt", device=DEV)
lrt = lcm.create_runtime(device=DEV)
STOP = {tok.eos_token_id, 151645}
_full = tok(open("/home/work/npu_work/dflash_work/pg1342.txt", encoding="utf-8").read(),
            return_tensors="pt").input_ids


def make_prompt(off, n):
    body = tok.decode(_full[0, off:off + n], skip_special_tokens=True)
    return tok.apply_chat_template([{"role": "user", "content": "Summarize the following passage.\n\n" + body}],
                                   add_generation_prompt=True, return_tensors="pt",
                                   enable_thinking=False)


def run(warm=False):
    T = dict(draft=0.0, verify=0.0, lmh=0.0, score=0.0, prefill=0.0, append=0.0)
    nret = 0; kept = 0

    def lmh(x):
        n = x.shape[1]
        if n < B: x = torch.cat([x, torch.zeros(1, B - n, H, dtype=x.dtype)], dim=1)
        s = time.time(); o = torch.as_tensor(lrt(x.contiguous().numpy())); T["lmh"] += time.time() - s
        return torch.argmax(o[:, :n].float(), dim=-1)

    def tpre(seg, off):
        s = time.time(); L = seg.shape[1]; pad = 1 if L % CHUNK == 0 else 0
        if pad: seg = torch.cat([seg, seg[:, -1:]], dim=1)
        r = pdec.prefill_forward(seg, cache_position=torch.arange(off, off + seg.shape[1],
                                 dtype=torch.int32).unsqueeze(0),
                                 attention_mask=torch.ones(seg.shape[1], dtype=torch.int64),
                                 batch_idx=0, block_tables=BT, is_external_block_tables=False)
        T["verify"] += time.time() - s; hs = r.hidden_states
        return tuple(h[:, :L] for h in hs) if pad else hs

    def twh(ids):
        s = time.time(); L = ids.shape[1]; pad = 1 if L % CHUNK == 0 else 0
        i2 = torch.cat([ids, ids[:, -1:]], dim=1) if pad else ids
        hs = m(input_ids=i2, attention_mask=torch.ones_like(i2)).hidden_states
        T["prefill"] += time.time() - s
        return tuple(h[:, :L] for h in hs) if pad else hs

    def append(th_src, positions, at):
        """th_src [1,n,NT*H] 를 캐시 위치 at 부터 기록. n<=BULK 를 자동 분할."""
        n = th_src.shape[1]; done = 0
        while done < n:
            take = min(B, n - done)
            buf = torch.zeros(1, B, NT * H, dtype=DT); buf[:, :take] = th_src[:, done:done + take]
            pos = torch.zeros(1, B, dtype=torch.int32)
            pos[:, :take] = positions[done:done + take].to(torch.int32)
            s = time.time()
            rt_a(buf, pos, torch.tensor([[at + done]], dtype=torch.int32), BT)
            T["append"] += time.time() - s
            done += take

    acc = []; ntok = 0
    for si in range(1 if warm else NSAMP):
        ids = make_prompt(si * (INLEN + 512), INLEN)
        P = ids.shape[1]
        hs = twh(ids)
        bonus = int(lmh(hs[-1][:, -1:].to(DT))[0, 0])
        CAP = P + (32 if warm else MAXNEW) + B + 16
        THB = torch.zeros(1, CAP, NT * H, dtype=DT)
        VRB = torch.zeros(1, CAP, dtype=torch.long)
        _t0 = extract_context_feature(hs, LID).to(DT); TL = _t0.shape[1]
        THB[:, :TL] = _t0; VRB[:, :P] = ids; VL = P
        cached = (P // CHUNK) * CHUNK
        KST = None; sel = None; step = 0
        if CMR:
            s = time.time()
            KST = scorer.keys(hs[-2], torch.arange(P))
            nq = min(B, P)
            q0 = scorer.queries(hs[-2][:, P - nq:P], torch.arange(P - nq, P))
            sel, _ = pick_chunks(scorer.scores(q0, KST), TL, BUDGET, CHUNK_SZ, TOPK)
            nret += 1; kept += sel.numel(); T["score"] += time.time() - s
            append(THB[:, sel], torch.arange(sel.numel()), 0)   # 압축 위치로 기록
            ctx_len = sel.numel()
        else:
            append(THB[:, :TL], torch.arange(TL), 0)
            ctx_len = TL
        while VL < P + (32 if warm else MAXNEW):
            blk = torch.full((1, B), MASK, dtype=torch.long); blk[0, 0] = bonus
            with torch.no_grad(): noi = embed(blk).detach()
            s = time.time()
            h = rt_b(noi, torch.arange(ctx_len, ctx_len + B, dtype=torch.int32).unsqueeze(0),
                     torch.tensor([[ctx_len]], dtype=torch.int32), BT)
            T["draft"] += time.time() - s
            h = torch.as_tensor(h)
            blk[:, 1:] = lmh(h[:, 1 - B:, :].to(DT))
            vl = VL; c0 = cached      # tpre 에 사용한 오프셋 (아래 CMR 슬라이싱에 필요)
            seg = torch.cat([VRB[:, c0:VL], blk], dim=1)
            hs2 = tpre(seg, c0)
            post = lmh(hs2[-1][:, vl - c0:vl - c0 + B].to(DT))
            a = int((blk[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
            acc.append(a + 1)
            nf = extract_context_feature(hs2, LID)[:, vl - c0:vl - c0 + a + 1, :].to(DT)
            THB[:, TL:TL + nf.shape[1]] = nf
            newpos = torch.arange(ctx_len, ctx_len + a + 1)
            append(nf, newpos, ctx_len)
            TL += nf.shape[1]; ctx_len += a + 1
            VRB[:, VL:VL + a + 1] = blk[:, :a + 1]; VL += a + 1
            bonus = int(post[0, a]); cached = (VL // CHUNK) * CHUNK
            if CMR:
                s = time.time()
                KST = torch.cat([KST, scorer.keys(hs2[-2][:, vl - c0:vl - c0 + a + 1],
                                                  torch.arange(vl, vl + a + 1))], dim=0)
                step += 1
                if step % EVERY == 0:
                    q = scorer.queries(hs2[-2][:, vl - c0:vl - c0 + a + 1],
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
with Power(DEV) as p0: time.sleep(5)
IDLE = p0.stats().get("P_mean", 0.0)
with Power(DEV) as pw:
    acc, ntok, T, nret, kept = run()
st = pw.stats(idle=IDLE)
n = max(len(acc), 1)
r = dict(mode="stateful", cmr=CMR, inlen=INLEN, samples=NSAMP, tau=round(sum(acc) / n, 3),
         rounds=len(acc), tokens=ntok,
         draft_ms=round(1000 * T["draft"] / n, 1), verify_ms=round(1000 * T["verify"] / n, 1),
         append_ms=round(1000 * T["append"] / n, 1), score_ms=round(1000 * T["score"] / n, 1),
         lmh_ms=round(1000 * T["lmh"] / n, 1), prefill_s=round(T["prefill"], 2),
         retrievals=nret, kept_mean=round(kept / max(nret, 1), 1),
         decode_tok_s=round(ntok / max(st["wall_s"] - T["prefill"], 1e-6), 2), **st)
print("SF " + json.dumps(r), flush=True)
print("SF_DONE", flush=True)

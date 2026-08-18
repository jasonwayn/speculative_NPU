"""DFlash stock vs DFlash+CMR on ATOM+ NPU, long-document inputs.

NPU 포팅의 핵심: 드래프트 그래프는 무상태이고 컨텍스트(target_hidden)를 매 호출 입력으로 받는다.
따라서 CMR 의 gather/재인덱싱/마스킹이 전부 호스트 텐서 구성으로 표현된다 (온디바이스 gather 불필요).
점수는 타깃 마지막 레이어(35) 가중치를 호스트에 올려 hidden_states[-2] 로부터 직접 계산한다.
"""
import sys, os, json, time, threading, subprocess, math, torch, torch.nn as nn, rebel, bisect
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
TGT = os.environ.get("TGTDIR", "/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-16k")
DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
B = int(os.environ.get("B", "16")); DEV = int(os.environ.get("DEV", "0"))
MAXNEW = int(os.environ.get("MAXNEW", "128")); NSAMP = int(os.environ.get("NSAMP", "3"))
INLEN = int(os.environ.get("INLEN", "4096"))
CMR = int(os.environ.get("CMR", "0"))
CHUNK_SZ = int(os.environ.get("CMR_CHUNK", "32")); TOPK = int(os.environ.get("CMR_TOPK", "32"))
EVERY = int(os.environ.get("CMR_EVERY", "4")); BUDGET = int(os.environ.get("CMR_BUDGET", "1024"))
PREC = os.environ.get("PREC", "float16"); DT = getattr(torch, PREC); DTS = PREC
BUCKETS = [int(x) for x in os.environ.get("BUCKETS", "2048,4096,8192,16384").split(",")]
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
from transformers import AutoTokenizer, AutoConfig
from optimum.rbln import RBLNQwen3ForCausalLM
from dflash.model import DFlashDraftModel, extract_context_feature
from safetensors import safe_open
import glob as _glob


class Power:
    def __init__(s, dev, hz=5): s.dev = dev; s.hz = hz; s.on = False; s.rows = []
    def _loop(s):
        while s.on:
            try:
                j = json.loads(subprocess.run(["rbln-stat", "--json"], capture_output=True,
                                              text=True, timeout=3).stdout)
                d = j["devices"][s.dev]
                s.rows.append((time.time(),
                               float(str(d["card_power"]).replace("uW", "")) / 1e6,
                               float(str(d.get("temperature", "0C")).replace("C", ""))))
            except Exception:
                pass
            time.sleep(1.0 / s.hz)
    def __enter__(s):
        s.on = True; s.th = threading.Thread(target=s._loop, daemon=True)
        s.th.start(); s.t0 = time.time(); return s
    def __exit__(s, *a): s.on = False; s.th.join(timeout=3); s.t1 = time.time()
    def stats(s, idle=0.0):
        w = round(s.t1 - s.t0, 1)
        if len(s.rows) < 2: return dict(wall_s=w)
        E = Ed = 0.0
        for i in range(1, len(s.rows)):
            dt = s.rows[i][0] - s.rows[i - 1][0]
            pm = 0.5 * (s.rows[i][1] + s.rows[i - 1][1])
            E += pm * dt; Ed += max(pm - idle, 0) * dt
        P = [r[1] for r in s.rows]; TT = [r[2] for r in s.rows]
        return dict(wall_s=w, P_mean=round(sum(P) / len(P), 2), E_J=round(E, 1),
                    E_dyn_J=round(Ed, 1), T_max=round(max(TT), 1),
                    T_mean=round(sum(TT) / len(TT), 1))


class Scorer:
    """타깃 마지막 디코더 레이어의 어텐션 점수를 호스트에서 재계산.

    hidden_states[-2] (레이어 35 입력) 만 있으면 되므로 새 그래프 출력이 필요 없다.
    GPU 포트가 monkeypatch 로 하던 _scoring_attn_forward 를 대체한다.
    """
    def __init__(s, src, L=35):
        cfg = json.load(open(os.path.join(src, "config.json")))
        s.nh = cfg["num_attention_heads"]; s.nkv = cfg["num_key_value_heads"]
        s.hd = cfg.get("head_dim", cfg["hidden_size"] // s.nh)
        s.eps = cfg["rms_norm_eps"]; s.rep = s.nh // s.nkv
        idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
        def get(n):
            with safe_open(os.path.join(src, idx[n]), framework="pt") as h:
                return h.get_tensor(n).float()
        p = "model.layers.%d." % L
        s.ln = get(p + "input_layernorm.weight")
        s.wq = get(p + "self_attn.q_proj.weight")
        s.wk = get(p + "self_attn.k_proj.weight")
        s.qn = get(p + "self_attn.q_norm.weight")
        s.kn = get(p + "self_attn.k_norm.weight")
        th = cfg.get("rope_theta", 1e6)
        s.inv = 1.0 / (th ** (torch.arange(0, s.hd, 2).float() / s.hd))

    def _rms(s, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * w

    def _rope(s, x, pos):
        f = pos.float().unsqueeze(1) * s.inv.unsqueeze(0)
        e = torch.cat([f, f], dim=-1)
        c = e.cos().unsqueeze(1); si = e.sin().unsqueeze(1)
        h = s.hd // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return x * c + rot * si

    def keys(s, h, pos):
        x = s._rms(h[0].float(), s.ln)
        k = (x @ s.wk.T).view(-1, s.nkv, s.hd)
        return s._rope(s._rms(k, s.kn), pos)

    def queries(s, h, pos):
        x = s._rms(h[0].float(), s.ln)
        q = (x @ s.wq.T).view(-1, s.nh, s.hd)
        return s._rope(s._rms(q, s.qn), pos)

    def scores(s, q, K):
        """q:(M,nh,hd)  K:(N,nkv,hd) -> (N,) 헤드·쿼리 평균 softmax 확률"""
        Kr = K.repeat_interleave(s.rep, dim=1)
        a = torch.einsum("mhd,nhd->hmn", q, Kr) / math.sqrt(s.hd)
        return torch.softmax(a, dim=-1).mean(dim=(0, 1))


def pick_chunks(sc, ctx_len, budget, chunk_sz, topk):
    """청크 평균 점수 상위 top_k + 최신 청크 강제. budget 이내로 절단."""
    n = (ctx_len + chunk_sz - 1) // chunk_sz
    st = torch.arange(n) * chunk_sz
    en = torch.clamp(st + chunk_sz, max=ctx_len)
    cum = torch.cat([torch.zeros(1), torch.cumsum(sc[:ctx_len], 0)])
    means = (cum[en] - cum[st]) / (en - st).float()
    keep = set(torch.topk(means, min(topk, n)).indices.tolist())
    keep.add(n - 1)
    idx = []
    for c in sorted(keep):
        idx.extend(range(int(st[c]), int(en[c])))
    return torch.tensor(idx[:budget], dtype=torch.long), len(keep)


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
with torch.no_grad():
    lmw.weight = nn.Parameter(_W, requires_grad=False)
lmw = lmw.eval()
m = RBLNQwen3ForCausalLM.from_pretrained(TGT, export=False, rbln_device=DEV)
pdec = m.prefill_decoder; CHUNK = pdec.rbln_config.prefill_chunk_size
cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size; NT = len(draft.target_layer_ids); LID = draft.target_layer_ids
MASK = draft.mask_token_id; BT = torch.tensor([0], dtype=torch.int16)
scorer = Scorer(SRC) if CMR else None
print("loaded  CMR=%d  INLEN=%d  buckets=%s" % (CMR, INLEN, BUCKETS), flush=True)


class Wrap(nn.Module):
    def __init__(s, d): super().__init__(); s.d = d
    def forward(s, n, t, p, mk):
        return s.d(position_ids=p, attention_mask=mk, noise_embedding=n,
                   target_hidden=t, past_key_values=None, use_cache=False, is_causal=False)


WR = Wrap(draft).eval()
CACHE = "/home/work/npu_work/dflash_work/rbln_cache"


def get_cm(name, build):
    p = os.path.join(CACHE, name + ".rbln")
    if os.path.exists(p): return rebel.RBLNCompiledModel(p)
    cm = build(); cm.save(p); return cm


BK = {}
for C in BUCKETS:
    cm = get_cm("draft_B%d_C%d_%s" % (B, C, DTS),
                lambda C=C: rebel.compile_from_torch(WR, input_info=[
                    ("noise_emb", [1, B, H], DTS),
                    ("target_hidden", [1, C, NT * H], DTS),
                    ("position_ids", [1, C + B], "int64"),
                    ("attn_mask", [1, 1, B, C + B], DTS)]))
    BK[C] = dict(rt=cm.create_runtime(device=DEV),
                 TH=torch.zeros(1, C, NT * H, dtype=DT),
                 PO=torch.zeros(1, C + B, dtype=torch.long),
                 MK=torch.zeros(1, 1, B, C + B, dtype=DT))
lcm = get_cm("lmhead_B%d_%s" % (B, DTS),
             lambda: rebel.compile_from_torch(nn.Sequential(lmw).eval(),
                                              input_info=[("x", [1, B, H], DTS)]))
lrt = lcm.create_runtime(device=DEV)
BKEYS = sorted(BK); STOP = {tok.eos_token_id, 151645}
print("buckets ready", flush=True)

_full = tok(open("/home/work/npu_work/dflash_work/pg1342.txt", encoding="utf-8").read(),
            return_tensors="pt").input_ids


def make_prompt(off, n_tok):
    body = tok.decode(_full[0, off:off + n_tok], skip_special_tokens=True)
    msg = [{"role": "user", "content": "Summarize the following passage.\n\n" + body}]
    return tok.apply_chat_template(msg, add_generation_prompt=True,
                                   return_tensors="pt", enable_thinking=False)


def run(warm=False):
    T = dict(draft=0.0, verify=0.0, lmh=0.0, score=0.0, prefill=0.0)
    nret = 0; kept_sum = 0

    def lmh(x):
        n = x.shape[1]
        if n < B: x = torch.cat([x, torch.zeros(1, B - n, H, dtype=x.dtype)], dim=1)
        s = time.time(); o = torch.as_tensor(lrt(x.contiguous().numpy()))
        T["lmh"] += time.time() - s
        return torch.argmax(o[:, :n].float(), dim=-1)

    def tpre(seg, off):
        s = time.time(); L = seg.shape[1]; pad = 1 if L % CHUNK == 0 else 0
        if pad: seg = torch.cat([seg, seg[:, -1:]], dim=1)
        r = pdec.prefill_forward(seg,
                                 cache_position=torch.arange(off, off + seg.shape[1],
                                                             dtype=torch.int32).unsqueeze(0),
                                 attention_mask=torch.ones(seg.shape[1], dtype=torch.int64),
                                 batch_idx=0, block_tables=BT, is_external_block_tables=False)
        T["verify"] += time.time() - s
        hs = r.hidden_states
        return tuple(h[:, :L] for h in hs) if pad else hs

    def twh(ids):
        s = time.time(); L = ids.shape[1]; pad = 1 if L % CHUNK == 0 else 0
        i2 = torch.cat([ids, ids[:, -1:]], dim=1) if pad else ids
        hs = m(input_ids=i2, attention_mask=torch.ones_like(i2)).hidden_states
        T["prefill"] += time.time() - s
        return tuple(h[:, :L] for h in hs) if pad else hs

    acc = []; ntok = 0; nsam = 1 if warm else NSAMP
    mx = 32 if warm else MAXNEW
    for si in range(nsam):
        ids = make_prompt(si * (INLEN + 512), INLEN)
        P = ids.shape[1]
        if (not CMR) and P > BKEYS[-1]:
            print("SKIP prompt %d > max bucket %d" % (P, BKEYS[-1]), flush=True); continue
        hs = twh(ids)
        bonus = int(lmh(hs[-1][:, -1:].to(DT))[0, 0])
        # 스토어는 working set 이 아니라 전체 컨텍스트를 담아야 한다
        # (CMR 은 큰 버킷을 쓰지 않으므로 버킷 크기로부터 유도하면 안 됨)
        CAP = P + mx + B + 16
        THB = torch.zeros(1, CAP, NT * H, dtype=DT)
        VRB = torch.zeros(1, CAP, dtype=torch.long)
        _t0 = extract_context_feature(hs, LID).to(DT); TL = _t0.shape[1]
        THB[:, :TL] = _t0; VRB[:, :P] = ids; VL = P
        cached = (P // CHUNK) * CHUNK
        sel = None; step = 0
        KST = None
        if CMR:
            s = time.time()
            KST = scorer.keys(hs[-2], torch.arange(P))     # (P, nkv, hd)
            # prefill 직후 즉시 첫 retrieval -> 큰 드래프트 버킷을 아예 안 쓴다
            nq = min(B, P)
            q0 = scorer.queries(hs[-2][:, P - nq:P], torch.arange(P - nq, P))
            sel, _ = pick_chunks(scorer.scores(q0, KST), TL, BUDGET, CHUNK_SZ, TOPK)
            nret += 1; kept_sum += sel.numel()
            T["score"] += time.time() - s
        cur = None; prev = None
        while VL < P + mx:
            if CMR and sel is not None:
                take = sel
                if TL > int(sel[-1]) + 1:                   # refresh_tail
                    tail = torch.arange(int(sel[-1]) + 1, TL)
                    take = torch.cat([sel, tail])
                th = THB[:, take]; L = th.shape[1]
            else:
                L = TL; th = THB[:, :TL]
            if L > BKEYS[-1]: break
            k = BKEYS[bisect.bisect_left(BKEYS, L)]
            b = BK[k]
            if k != cur:
                b["TH"].zero_(); b["PO"].zero_(); b["MK"].zero_()
                b["MK"][:, :, :, :k] = torch.finfo(DT).min; prev = k; cur = k
            pn = k - L
            b["TH"][:, pn:] = th
            b["PO"][:, pn:] = torch.arange(L + B)            # compact_positions
            if pn < prev: b["MK"][:, :, :, pn:prev] = 0.0
            elif pn > prev: b["MK"][:, :, :, prev:pn] = torch.finfo(DT).min
            prev = pn
            blk = torch.full((1, B), MASK, dtype=torch.long); blk[0, 0] = bonus
            with torch.no_grad(): noi = embed(blk).detach()
            s = time.time()
            h = torch.as_tensor(b["rt"](noi.numpy(), b["TH"].numpy(),
                                        b["PO"].numpy(), b["MK"].numpy()))
            T["draft"] += time.time() - s
            blk[:, 1:] = lmh(h[:, 1 - B:, :].to(DT))
            vl = VL
            seg = torch.cat([VRB[:, cached:VL], blk], dim=1)
            hs2 = tpre(seg, cached)
            post = lmh(hs2[-1][:, vl - cached:vl - cached + B].to(DT))
            a = int((blk[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
            acc.append(a + 1)
            nf = extract_context_feature(hs2, LID)[:, vl - cached:vl - cached + a + 1, :].to(DT)
            THB[:, TL:TL + nf.shape[1]] = nf; TL += nf.shape[1]
            if CMR:
                s = time.time()
                newk = scorer.keys(hs2[-2][:, vl - cached:vl - cached + a + 1],
                                   torch.arange(vl, vl + a + 1))
                KST = torch.cat([KST, newk], dim=0)
                step += 1
                if step % EVERY == 0:
                    q = scorer.queries(hs2[-2][:, vl - cached:vl - cached + a + 1],
                                       torch.arange(vl, vl + a + 1))
                    scv = scorer.scores(q, KST)
                    sel, nch = pick_chunks(scv, TL, BUDGET, CHUNK_SZ, TOPK)
                    nret += 1; kept_sum += sel.numel()
                T["score"] += time.time() - s
            VRB[:, VL:VL + a + 1] = blk[:, :a + 1]; VL += a + 1
            bonus = int(post[0, a]); cached = (VL // CHUNK) * CHUNK
            if any(int(t) in STOP for t in VRB[0, P:VL]): break
            if TL >= CAP - B - 2: break
        ntok += VL - P
    return acc, ntok, T, nret, kept_sum


run(warm=True)
with Power(DEV) as p0: time.sleep(5)
IDLE = p0.stats().get("P_mean", 0.0)
with Power(DEV) as pw:
    acc, ntok, T, nret, kept = run()
st = pw.stats(idle=IDLE)
r = dict(cmr=CMR, inlen=INLEN, samples=NSAMP, maxnew=MAXNEW,
         tau=round(sum(acc) / len(acc), 3) if acc else None, rounds=len(acc), tokens=ntok,
         draft_s=round(T["draft"], 2), verify_s=round(T["verify"], 2),
         lmh_s=round(T["lmh"], 2), score_s=round(T["score"], 2),
         prefill_s=round(T["prefill"], 2),
         draft_ms=round(1000 * T["draft"] / max(len(acc), 1), 1),
         verify_ms=round(1000 * T["verify"] / max(len(acc), 1), 1),
         score_ms=round(1000 * T["score"] / max(len(acc), 1), 1),
         retrievals=nret, kept_mean=round(kept / max(nret, 1), 1),
         tok_s=round(ntok / st["wall_s"], 2),
         decode_tok_s=round(ntok / max(st["wall_s"] - T["prefill"], 1e-6), 2), **st)
print("CMRB " + json.dumps(r), flush=True)
print("CMRB_DONE", flush=True)

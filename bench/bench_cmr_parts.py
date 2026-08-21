"""bench_cmr.py 에서 Power / Scorer / pick_chunks 만 분리 (상태 유지 벤치에서 재사용)."""
import os, json, time, threading, subprocess, math, torch
from safetensors import safe_open

class Power:
    def __init__(s, dev, hz=5):
        # dev 는 int 또는 list. TP 구성에서 카드 1장만 읽으면 에너지가 N배 과소 보고된다.
        s.devs = [dev] if isinstance(dev, int) else list(dict.fromkeys(dev))
        s.dev = s.devs[0]; s.hz = hz; s.on = False; s.rows = []
    def _loop(s):
        while s.on:
            try:
                j = json.loads(subprocess.run(["rbln-stat", "--json"], capture_output=True,
                                              text=True, timeout=3).stdout)
                ds = [j["devices"][i] for i in s.devs]
                s.rows.append((time.time(),
                               sum(float(str(d["card_power"]).replace("uW", "")) / 1e6
                                   for d in ds),
                               max(float(str(d.get("temperature", "0C")).replace("C", ""))
                                   for d in ds)))
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



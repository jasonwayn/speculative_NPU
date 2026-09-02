"""① 드래프터 TP 가 실제로 쪼개졌는지 실행으로 확인  ② lm_head 4등분 (캐스트 제거)

①의 배경: compile_from_torch 시그니처에 **kwargs 가 있어서 tensor_parallel_size 가
그냥 삼켜졌을 수 있다. COMPILE_OK 는 증거가 아니다. 4 장에서 돌려서 draft 시간이
줄어야 진짜 샤딩이다. 출력도 TP1 과 맞아야 한다.

②의 배경: 앞선 시도가 idx.to(torch.int32) 에서 TVM 프론트엔드 캐스트로 죽었다.
캐스트를 빼고 인덱스를 그대로 반환한다.
"""
import os
import sys
import time

import rebel
import torch
from torch import nn

sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")

DT = torch.float32
DTS = "float32"
B = 16

# ======================= ① 드래프터 TP 실행 확인 =======================
print("=" * 60, flush=True)
print("① 드래프터 TP — 컴파일이 아니라 실행으로 판정", flush=True)
try:
    from rebel import CompileContext
    from transformers import AutoConfig
    from dflash.model import DFlashDraftModel
    from draft_two_rr import BlockRR

    DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
    MAXC = int(os.environ.get("MAXC", "4096"))
    KVB = int(os.environ.get("KV_BLOCK_SIZE", str(MAXC)))
    NB = MAXC // KVB

    cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
    H = cfg.hidden_size
    NL = cfg.num_hidden_layers
    NKV = cfg.num_key_value_heads
    HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)

    caches = [torch.zeros(NB, NKV, KVB, HD, dtype=DT) for _ in range(2 * NL)]
    cinfo = [("past_key_values_%d" % i, [NB, NKV, KVB, HD], DTS)
             for i in range(2 * NL)]
    binfo = ([("noise_emb", [1, B, H], DTS),
              ("angle_blk", [1, B, HD // 2], DTS),
              ("seq_blk", [1, 1], "int32"),
              ("block_tables", [NB], "int16")] + cinfo)

    def exs():
        out = []
        for n, sh, dt in binfo:
            if n.startswith("past_key_values_"):
                out.append(caches[int(n.rsplit("_", 1)[1])])
            else:
                out.append(torch.zeros(*sh, dtype=getattr(torch, dt)))
        return out

    torch.manual_seed(0)
    # 정적 주소로 표시된 KV 캐시는 호출 때 넘기지 않는다 (벤치와 동일하게 4 개만).
    args = [torch.randn(1, B, H, dtype=DT),
            torch.randn(1, B, HD // 2, dtype=DT),
            torch.tensor([[64]], dtype=torch.int32),
            torch.arange(NB, dtype=torch.int16)]

    def build(tp):
        ctx = CompileContext(use_weight_sharing=True)
        for (n, _, _), t in zip(cinfo, caches):
            ctx.mark_static_address(t, n)
        kw = dict(input_info=binfo, example_inputs=exs(), compile_context=ctx)
        if tp > 1:
            kw["tensor_parallel_size"] = tp
        return rebel.compile_from_torch(BlockRR(draft, KVB).eval(), **kw)

    def timeit(rt, n=20):
        for _ in range(3):
            rt(*[a.contiguous() for a in args])
        ts = []
        for _ in range(n):
            s = time.time()
            o = rt(*[a.contiguous() for a in args])
            ts.append((time.time() - s) * 1000)
        ts.sort()
        arr = o[0] if isinstance(o, (list, tuple)) else o
        return ts[n // 2], torch.as_tensor(arr).float()

    cm1 = build(1)
    rt1 = rebel.Runtime(cm1, tensor_type="pt", device=0)
    ms1, out1 = timeit(rt1)
    print("  TP1  device=0        %6.2f ms" % ms1, flush=True)
    del rt1

    for tp in (2, 4):
        try:
            cm = build(tp)
            rt = rebel.Runtime(cm, tensor_type="pt", device=list(range(tp)))
            ms, out = timeit(rt)
            cos = torch.nn.functional.cosine_similarity(
                out1.reshape(-1), out.reshape(-1), dim=0)
            verdict = "샤딩됨" if ms < ms1 * 0.85 else "샤딩 안 된 듯 (시간 동일)"
            print("  TP%d  device=%s  %6.2f ms   %.4fx   cos=%.6f   -> %s"
                  % (tp, list(range(tp)), ms, ms1 / ms, float(cos), verdict), flush=True)
            del rt
        except Exception as exc:  # noqa: BLE001
            print("  TP%d  실행 실패  %s: %s" % (tp, type(exc).__name__, str(exc)[:250]),
                  flush=True)
except Exception as exc:  # noqa: BLE001
    print("  ① 전체 실패  %s: %s" % (type(exc).__name__, str(exc)[:300]), flush=True)

# ======================= ② lm_head 4등분 =======================
print("=" * 60, flush=True)
print("② lm_head 어휘 4등분 + 카드별 로컬 argmax", flush=True)
try:
    import glob as _glob
    from safetensors import safe_open

    SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
    LDT, LDTS = torch.float16, "float16"
    NSHARD = int(os.environ.get("NSHARD", "4"))
    W = None
    for f in sorted(_glob.glob(SRC + "/*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if k.endswith("model.embed_tokens.weight"):
                    W = h.get_tensor(k).to(LDT)
                    break
        if W is not None:
            break
    V, HH = W.shape
    VS = V // NSHARD
    print("  vocab=%d hidden=%d shards=%d (구간 %d)" % (V, HH, NSHARD, VS), flush=True)

    class Full(nn.Module):
        def __init__(s, w):
            super().__init__()
            s.lin = nn.Linear(HH, w.shape[0], bias=False)
            with torch.no_grad():
                s.lin.weight = nn.Parameter(w, requires_grad=False)

        def forward(s, x):
            return s.lin(x)

    class Shard(nn.Module):
        """캐스트 없이 (최대값, 인덱스) 반환. .to(int32) 가 TVM 에서 죽었다."""

        def __init__(s, w):
            super().__init__()
            s.lin = nn.Linear(HH, w.shape[0], bias=False)
            with torch.no_grad():
                s.lin.weight = nn.Parameter(w, requires_grad=False)

        def forward(s, x):
            logits = s.lin(x)
            v = torch.amax(logits, dim=-1)
            i = torch.argmax(logits, dim=-1)
            return v, i

    torch.manual_seed(0)
    x = torch.randn(1, B, HH, dtype=LDT)
    xn = x.contiguous().numpy()
    with torch.no_grad():
        ref = torch.argmax(Full(W).eval()(x).float(), dim=-1)

    cmf = rebel.compile_from_torch(Full(W).eval(), input_info=[("x", [1, B, HH], LDTS)])
    rtf = cmf.create_runtime(device=0)
    for _ in range(3):
        torch.argmax(torch.as_tensor(rtf(xn)).float(), dim=-1)
    d, hs = [], []
    for _ in range(20):
        s = time.time(); o = torch.as_tensor(rtf(xn)); d.append((time.time() - s) * 1000)
        s = time.time(); gf = torch.argmax(o.float(), dim=-1); hs.append((time.time() - s) * 1000)
    d.sort(); hs.sort()
    print("  FULL     device %5.2f + host %5.2f = %5.2f ms   match=%s"
          % (d[10], hs[10], d[10] + hs[10], bool((gf == ref).all())), flush=True)
    del rtf, cmf

    rts = []
    for i in range(NSHARD):
        cm = rebel.compile_from_torch(
            Shard(W[i * VS:(i + 1) * VS].contiguous()).eval(),
            input_info=[("x", [1, B, HH], LDTS)])
        rts.append(cm.create_runtime(device=i))
        print("    shard %d -> device %d" % (i, i), flush=True)

    def sharded():
        vs, ix = [], []
        for i, rt in enumerate(rts):
            o = rt(xn)
            vs.append(torch.as_tensor(o[0]).float())
            ix.append(torch.as_tensor(o[1]).long() + i * VS)
        Vs, Is = torch.stack(vs, 0), torch.stack(ix, 0)
        return Is.gather(0, Vs.argmax(0, keepdim=True)).squeeze(0)

    for _ in range(3):
        sharded()
    t = []
    for _ in range(20):
        s = time.time(); gs = sharded(); t.append((time.time() - s) * 1000)
    t.sort()
    print("  SHARDED  total %5.2f ms   (%d 장 순차 호출 + 호스트 합치기)  match=%s"
          % (t[10], NSHARD, bool((gs == ref).all())), flush=True)
    print("  payload  %.2f MB -> %.4f MB" % (B * V * 2 / 1e6, NSHARD * B * 6 / 1e6),
          flush=True)
except Exception as exc:  # noqa: BLE001
    import traceback
    print("  ② 실패  %s: %s" % (type(exc).__name__, str(exc)[:300]), flush=True)
    traceback.print_exc()

print("PROBE2_DONE", flush=True)

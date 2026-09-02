"""드래프터 TP 가 왜 막히는가 — AppendRR 인가, 런타임 두 개인가.

앞선 프로브는 BlockRR 만 쟀다 (TP4 에서 3.06배). 그런데 벤치가 죽는 지점은
rt_a, 즉 AppendRR 이다. AppendRR 은 TP 로 만들어본 적이 없다.

타깃을 안 올린 상태에서 셋을 순서대로 본다.
  A  AppendRR 단독 TP1/TP4        -> 실패하면 AppendRR 자체가 원인
  B  BlockRR  단독 TP4 (재확인)
  C  둘을 동시에 살려두고 TP4      -> 여기서만 실패하면 "정적 캐시 공유 + TP 런타임 2개"
"""
import os
import sys
import time

import rebel
import torch

sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")

from rebel import CompileContext
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
from draft_two_rr import AppendRR, BlockRR

DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
DT, DTS = torch.float32, "float32"
B = 16
A_W = int(os.environ.get("APPEND_WIDTH", str(B)))
MAXC = int(os.environ.get("MAXC", "4096"))
KVB = int(os.environ.get("KV_BLOCK_SIZE", str(MAXC)))
NB = MAXC // KVB

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True)
cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size
NT = len(draft.target_layer_ids)
NL = cfg.num_hidden_layers
NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
print("drafter: layers=%d hidden=%d n_kv=%d hd=%d  append_width=%d blocks=%d"
      % (NL, H, NKV, HD, A_W, NB), flush=True)

caches = [torch.zeros(NB, NKV, KVB, HD, dtype=DT) for _ in range(2 * NL)]
cinfo = [("past_key_values_%d" % i, [NB, NKV, KVB, HD], DTS) for i in range(2 * NL)]


def ainfo():
    return ([("th_new", [1, A_W, NT * H], DTS),
             ("angle_ctx", [1, A_W, HD // 2], DTS),
             ("seq_ctx", [1, 1], "int32"),
             ("block_tables", [NB], "int16")] + cinfo)


def binfo():
    return ([("noise_emb", [1, B, H], DTS),
             ("angle_blk", [1, B, HD // 2], DTS),
             ("seq_blk", [1, 1], "int32"),
             ("block_tables", [NB], "int16")] + cinfo)


def exs(info):
    out = []
    for n, sh, dt in info:
        if n.startswith("past_key_values_"):
            out.append(caches[int(n.rsplit("_", 1)[1])])
        else:
            out.append(torch.zeros(*sh, dtype=getattr(torch, dt)))
    return out


def new_ctx():
    ctx = CompileContext(use_weight_sharing=True)
    for (n, _, _), t in zip(cinfo, caches):
        ctx.mark_static_address(t, n)
    return ctx


def compile_one(kind, tp, ctx):
    info = ainfo() if kind == "append" else binfo()
    mod = (AppendRR(draft, KVB) if kind == "append" else BlockRR(draft, KVB)).eval()
    kw = dict(input_info=info, example_inputs=exs(info), compile_context=ctx)
    if tp > 1:
        kw["tensor_parallel_size"] = tp
    return rebel.compile_from_torch(mod, **kw)


def args_for(kind):
    if kind == "append":
        return [torch.randn(1, A_W, NT * H, dtype=DT),
                torch.randn(1, A_W, HD // 2, dtype=DT),
                torch.tensor([[64]], dtype=torch.int32),
                torch.arange(NB, dtype=torch.int16)]
    return [torch.randn(1, B, H, dtype=DT),
            torch.randn(1, B, HD // 2, dtype=DT),
            torch.tensor([[64]], dtype=torch.int32),
            torch.arange(NB, dtype=torch.int16)]


def timeit(rt, kind, n=15):
    a = [t.contiguous() for t in args_for(kind)]
    for _ in range(3):
        rt(*a)
    ts = []
    for _ in range(n):
        s = time.time()
        rt(*a)
        ts.append((time.time() - s) * 1000)
    ts.sort()
    return ts[n // 2]


print("=" * 62, flush=True)
print("A  AppendRR 단독", flush=True)
for tp in (1, 4):
    try:
        ctx = new_ctx()
        cm = compile_one("append", tp, ctx)
        rt = rebel.Runtime(cm, tensor_type="pt",
                           device=(list(range(tp)) if tp > 1 else 0))
        print("   TP%d  %6.2f ms   OK" % (tp, timeit(rt, "append")), flush=True)
        del rt, cm
    except Exception as exc:  # noqa: BLE001
        print("   TP%d  실패  %s: %s" % (tp, type(exc).__name__, str(exc)[:200]), flush=True)

print("=" * 62, flush=True)
print("B  BlockRR 단독 (재확인)", flush=True)
for tp in (1, 4):
    try:
        ctx = new_ctx()
        cm = compile_one("block", tp, ctx)
        rt = rebel.Runtime(cm, tensor_type="pt",
                           device=(list(range(tp)) if tp > 1 else 0))
        print("   TP%d  %6.2f ms   OK" % (tp, timeit(rt, "block")), flush=True)
        del rt, cm
    except Exception as exc:  # noqa: BLE001
        print("   TP%d  실패  %s: %s" % (tp, type(exc).__name__, str(exc)[:200]), flush=True)

print("=" * 62, flush=True)
print("C  둘을 같은 컨텍스트로 만들고 런타임 2 개를 동시에 유지", flush=True)
for tp in (1, 4):
    rt_a = rt_b = None
    try:
        ctx = new_ctx()
        cm_a = compile_one("append", tp, ctx)
        cm_b = compile_one("block", tp, ctx)
        dev = list(range(tp)) if tp > 1 else 0
        rt_a = rebel.Runtime(cm_a, tensor_type="pt", device=dev)
        print("   TP%d  rt_a OK" % tp, flush=True)
        rt_b = rebel.Runtime(cm_b, tensor_type="pt", device=dev)
        print("   TP%d  rt_b OK   append %.2f ms / block %.2f ms   둘 다 살아있음"
              % (tp, timeit(rt_a, "append"), timeit(rt_b, "block")), flush=True)
    except Exception as exc:  # noqa: BLE001
        where = "rt_a 생성 전/중" if rt_a is None else "rt_b 생성 중"
        print("   TP%d  실패 (%s)  %s: %s"
              % (tp, where, type(exc).__name__, str(exc)[:200]), flush=True)
    finally:
        del rt_a, rt_b

print("APPENDTP_DONE", flush=True)

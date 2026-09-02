"""드래프터 TP 두 번째 런타임 죽음 — 순서와 TP 수준을 가른다.

지금까지 확인: 각 그래프 단독 TP4 OK. append->block 순서로 둘을 만들면 TP4 에서만
두 번째(rt_b)가 INIT_INTERNAL. TP1 쌍은 공존. 저장/재로드 무관. 디바이스 배치 무관.

안 해본 두 가지:
  E  순서 뒤집기 (block 먼저, append 나중) @ TP4
     -> block 이 살면 "두 번째라서" 죽는 것. append 가 또 죽으면 append 특정.
  F  TP2 쌍 (append+block)
     -> 살면 block 4.63 -> ~2.4 ms 는 회수 가능. TP4 만의 문제로 좁혀짐.
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
print("drafter: layers=%d hidden=%d n_kv=%d hd=%d blocks=%d" % (NL, H, NKV, HD, NB),
      flush=True)

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


def pair(order, tp):
    """order: [(이름, kind), ...] 생성 순서대로. 어디까지 살았는지 보고한다."""
    rts = {}
    try:
        ctx = new_ctx()
        cms = {k: compile_one(k, tp, ctx) for _, k in order}
        dev = list(range(tp)) if tp > 1 else 0
        for nm, k in order:
            rts[nm] = rebel.Runtime(cms[k], tensor_type="pt", device=dev)
            print("   TP%d  %s OK" % (tp, nm), flush=True)
        ms = {nm: timeit(rts[nm], k) for nm, k in order}
        print("   TP%d  둘 다 생존   %s" %
              (tp, "  ".join("%s %.2f ms" % (nm, ms[nm]) for nm, _ in order)), flush=True)
    except Exception as exc:  # noqa: BLE001
        dead = order[len(rts)][0] if len(rts) < len(order) else "?"
        print("   TP%d  실패 (%s 생성 중)  %s: %s"
              % (tp, dead, type(exc).__name__, str(exc)[:200]), flush=True)
    finally:
        for r in rts.values():
            del r


print("=" * 62, flush=True)
print("E  순서 뒤집기: block 먼저 -> append 나중 @ TP4", flush=True)
pair([("block", "block"), ("append", "append")], 4)

print("=" * 62, flush=True)
print("F  TP2 쌍 (append -> block, 기존 순서)", flush=True)
pair([("append", "append"), ("block", "block")], 2)

print("PROBE3_DONE", flush=True)

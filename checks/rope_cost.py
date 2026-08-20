"""RoPE 수정판 드래프트 그래프의 호출 비용 — 타깃 없이 격리 측정.

수정판은 cos/sin 을 그래프 입력으로 받는다(텐서 2개 추가). 그래프 안의 sin/cos 계산은
사라졌으므로 더 빨라야 정상인데, 벤치에서는 느려졌다. 어디서 오는지 본다.
  ① 그래프 자체 호출 시간
  ② 호스트 cos/sin 계산 시간
"""
import os, sys, time, torch, rebel
sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")
D = "/home/work/npu_work/dflash_work"
DRF = os.path.join(D, "Qwen3-4B-DFlash-b16")
DEV = int(os.environ.get("DEV", "0")); DT = torch.float32; DTS = "float32"
B = 16; MAXC = 4096
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
from rebel import CompileContext
from draft_two import Append, Block
from draft_two_rope import AppendR, BlockR, host_cs
from draft_two_rr import AppendRR, BlockRR, host_angle

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size; NT = len(draft.target_layer_ids)
NL = cfg.num_hidden_layers; NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
INV = 1.0 / (cfg.rope_theta ** (torch.arange(0, HD, 2).float() / HD))
BT = torch.tensor([0], dtype=torch.int16)


def build(kind):
    caches = [torch.zeros(1, NKV, MAXC, HD, dtype=DT) for _ in range(2 * NL)]
    cinfo = [("past_key_values_%d" % i, [1, NKV, MAXC, HD], DTS) for i in range(2 * NL)]
    ctx = CompileContext(use_weight_sharing=True)
    for (n_, _, _), t in zip(cinfo, caches): ctx.mark_static_address(t, n_)
    if kind == "orig":
        ai = [("th_new", [1, B, NT*H], DTS), ("pos_ctx", [1, B], "int32"),
              ("seq_ctx", [1, 1], "int32"), ("block_tables", [1], "int16")] + cinfo
        bi = [("noise_emb", [1, B, H], DTS), ("pos_blk", [1, B], "int32"),
              ("seq_blk", [1, 1], "int32"), ("block_tables", [1], "int16")] + cinfo
        A, BL = Append(draft, MAXC).eval(), Block(draft, MAXC).eval()
    elif kind == "rr":
        ai = [("th_new", [1, B, NT*H], DTS), ("angle_ctx", [1, B, HD // 2], DTS),
              ("seq_ctx", [1, 1], "int32"), ("block_tables", [1], "int16")] + cinfo
        bi = [("noise_emb", [1, B, H], DTS), ("angle_blk", [1, B, HD // 2], DTS),
              ("seq_blk", [1, 1], "int32"), ("block_tables", [1], "int16")] + cinfo
        A, BL = AppendRR(draft, MAXC).eval(), BlockRR(draft, MAXC).eval()
    else:
        ai = [("th_new", [1, B, NT*H], DTS), ("cos_ctx", [1, B, HD], DTS),
              ("sin_ctx", [1, B, HD], DTS), ("seq_ctx", [1, 1], "int32"),
              ("block_tables", [1], "int16")] + cinfo
        bi = [("noise_emb", [1, B, H], DTS), ("cos_blk", [1, B, HD], DTS),
              ("sin_blk", [1, B, HD], DTS), ("seq_blk", [1, 1], "int32"),
              ("block_tables", [1], "int16")] + cinfo
        A, BL = AppendR(draft, MAXC).eval(), BlockR(draft, MAXC).eval()

    def exs(info):
        return [caches[int(n_.rsplit("_", 1)[1])] if n_.startswith("past_key_values_")
                else torch.zeros(*s_, dtype=getattr(torch, d)) for n_, s_, d in info]
    ca = rebel.compile_from_torch(A, input_info=ai, example_inputs=exs(ai), compile_context=ctx)
    cb = rebel.compile_from_torch(BL, input_info=bi, example_inputs=exs(bi), compile_context=ctx)
    return (rebel.Runtime(ca, tensor_type="pt", device=DEV),
            rebel.Runtime(cb, tensor_type="pt", device=DEV))


def bench(kind, rt_a, rt_b, ctx_len=1024, reps=20):
    th = torch.randn(1, B, NT * H, dtype=DT) * 0.1
    noi = torch.randn(1, B, H, dtype=DT) * 0.1
    pos = torch.arange(ctx_len, ctx_len + B, dtype=torch.int32).unsqueeze(0)
    seq = torch.tensor([[ctx_len]], dtype=torch.int32)
    if kind == "orig":
        aargs = lambda: (th, pos, seq, BT)
        bargs = lambda: (noi, pos, seq, BT)
        host_ms = 0.0
    elif kind == "rr":
        p64 = pos.to(torch.int64)
        t0 = time.time()
        for _ in range(reps): host_angle(p64, INV, DT)
        host_ms = (time.time() - t0) / reps * 1000
        ang = host_angle(p64, INV, DT)
        aargs = lambda: (th, ang, seq, BT)
        bargs = lambda: (noi, ang, seq, BT)
    else:
        p64 = pos.to(torch.int64)
        t0 = time.time()
        for _ in range(reps): host_cs(p64, INV, DT)
        host_ms = (time.time() - t0) / reps * 1000
        c, s_ = host_cs(p64, INV, DT)
        aargs = lambda: (th, c, s_, seq, BT)
        bargs = lambda: (noi, c, s_, seq, BT)
    for _ in range(3): rt_a(*aargs()); rt_b(*bargs())
    ta = []
    for _ in range(reps):
        t0 = time.time(); rt_a(*aargs()); ta.append((time.time() - t0) * 1000)
    tb = []
    for _ in range(reps):
        t0 = time.time(); rt_b(*bargs()); tb.append((time.time() - t0) * 1000)
    ta.sort(); tb.sort()
    print("%-5s  Append %.2f ms   Block %.2f ms   host_cos/sin %.3f ms"
          % (kind, ta[reps // 2], tb[reps // 2], host_ms), flush=True)


for kind in ("orig", "rope", "rr"):
    a, b = build(kind)
    bench(kind, a, b)
    del a, b
print("ROPECOST_DONE", flush=True)

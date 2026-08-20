"""bench_sf_fused_diag.py -> bench_sf_rope.py : 드래프터 RoPE 를 호스트 계산으로 교체.

바꾸는 곳
  ① Append/Block 을 AppendR/BlockR 로
  ② 그래프 입력 pos_ctx / pos_blk -> cos_*/sin_*  [1, N, HD]
  ③ 호출부에서 호스트가 fp64 로 cos/sin 계산해 전달
"""
import io, sys

src = "/home/work/npu_work/dflash_work/bench_sf_fused_diag.py"
dst = "/home/work/npu_work/dflash_work/bench_sf_rope.py"
t = io.open(src, encoding="utf-8").read()
n = {}

# ① 클래스 교체
n["imp"] = t.count("from draft_two import Append, Block, _Base, PAGED, rope")
t = t.replace("from draft_two import Append, Block, _Base, PAGED, rope",
              "from draft_two import Append, Block, _Base, PAGED, rope\n"
              "from draft_two_rope import AppendR, BlockR, host_cs")
n["cls_a"] = t.count("Append(draft, KV_BLOCK_SIZE)")
t = t.replace("Append(draft, KV_BLOCK_SIZE)", "AppendR(draft, KV_BLOCK_SIZE)")
n["cls_b"] = t.count("Block(draft, KV_BLOCK_SIZE)")
t = t.replace("Block(draft, KV_BLOCK_SIZE)", "BlockR(draft, KV_BLOCK_SIZE)")

# ② 입력 서명
n["ainfo"] = t.count('("pos_ctx", [1, A], "int32")')
t = t.replace('("pos_ctx", [1, A], "int32")',
              '("cos_ctx", [1, A, HD], DTS), ("sin_ctx", [1, A, HD], DTS)')
n["binfo"] = t.count('("pos_blk", [1, B], "int32")')
t = t.replace('("pos_blk", [1, B], "int32")',
              '("cos_blk", [1, B, HD], DTS), ("sin_blk", [1, B, HD], DTS)')

# 호스트용 inv_freq (그래프 밖)
t = t.replace("HD = getattr(cfg, \"head_dim\", H // cfg.num_attention_heads)",
              "HD = getattr(cfg, \"head_dim\", H // cfg.num_attention_heads)\n"
              "INV_HOST = 1.0 / (cfg.rope_theta ** (torch.arange(0, HD, 2).float() / HD))")

# ③ 호출부
n["call_a"] = t.count("rt_a(buf, pos, torch.tensor([[at + done]], dtype=torch.int32), BT)")
t = t.replace("rt_a(buf, pos, torch.tensor([[at + done]], dtype=torch.int32), BT)",
              "_ca, _sa = host_cs(pos, INV_HOST, DT)\n"
              "            rt_a(buf, _ca, _sa, torch.tensor([[at + done]], dtype=torch.int32), BT)")
n["call_b"] = t.count("""                _block_args = [
                    noi,
                    torch.arange(ctx_len, ctx_len + B, dtype=torch.int32).unsqueeze(0),
                    torch.tensor([[ctx_len]], dtype=torch.int32),
                    BT,
                ]""")
t = t.replace("""                _block_args = [
                    noi,
                    torch.arange(ctx_len, ctx_len + B, dtype=torch.int32).unsqueeze(0),
                    torch.tensor([[ctx_len]], dtype=torch.int32),
                    BT,
                ]""",
"""                _cb, _sb = host_cs(
                    torch.arange(ctx_len, ctx_len + B, dtype=torch.int64).unsqueeze(0),
                    INV_HOST, DT)
                _block_args = [
                    noi,
                    _cb,
                    _sb,
                    torch.tensor([[ctx_len]], dtype=torch.int32),
                    BT,
                ]""")

need = ["imp", "cls_a", "cls_b", "ainfo", "binfo", "call_a", "call_b"]
bad = [k for k in need if n.get(k, 0) < 1]
if bad:
    print("PATCH_MISS %s  counts=%s" % (bad, n)); sys.exit(1)

io.open(dst, "w", encoding="utf-8", newline="\n").write(t)
print("WROTE %s  %s" % (dst, n))

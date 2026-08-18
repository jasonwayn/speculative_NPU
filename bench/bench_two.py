"""상태 유지 드래프트 지연 vs 캐시 채움 정도. 무상태(45/91/159ms)와 비교."""
import sys, os, time, json, torch, rebel
sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")
from rebel import CompileContext
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
from draft_two import Append, Block

DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
DT = torch.float16; DTS = "float16"
B = 16; A = 16
MAXC = int(os.environ.get("MAXC", "16384"))
DEV = int(os.environ.get("DEV", "0"))
torch.set_num_threads(8)

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size; NT = len(draft.target_layer_ids)
NL = cfg.num_hidden_layers; NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
kv_mb = 2 * NL * NKV * MAXC * HD * 2 / 1e6
print("MAXC=%d  device KV = %.1f MB" % (MAXC, kv_mb), flush=True)

caches = [torch.zeros(1, NKV, MAXC, HD, dtype=DT) for _ in range(2 * NL)]
cinfo = [("past_key_values_%d" % i, [1, NKV, MAXC, HD], DTS) for i in range(2 * NL)]
ctx = CompileContext(use_weight_sharing=True)
for (n, _, _), t in zip(cinfo, caches):
    ctx.mark_static_address(t, n)
a_info = [("th_new", [1, A, NT * H], DTS), ("pos_ctx", [1, A], "int32"),
          ("seq_ctx", [1, 1], "int32"), ("block_tables", [1], "int16")] + cinfo
b_info = [("noise_emb", [1, B, H], DTS), ("pos_blk", [1, B], "int32"),
          ("seq_blk", [1, 1], "int32"), ("block_tables", [1], "int16")] + cinfo


def exs(info):
    o = []
    for n, sh, dt in info:
        o.append(caches[int(n.rsplit("_", 1)[1])] if n.startswith("past_key_values_")
                 else torch.zeros(*sh, dtype=getattr(torch, dt)))
    return o


t0 = time.time()
cm_a = rebel.compile_from_torch(Append(draft, MAXC).eval(), input_info=a_info,
                                example_inputs=exs(a_info), compile_context=ctx)
cm_b = rebel.compile_from_torch(Block(draft, MAXC).eval(), input_info=b_info,
                                example_inputs=exs(b_info), compile_context=ctx)
print("COMPILED %.1fs" % (time.time() - t0), flush=True)
rt_a = rebel.Runtime(cm_a, tensor_type="pt", device=DEV)
rt_b = rebel.Runtime(cm_b, tensor_type="pt", device=DEV)
print("RUNTIME_OK", flush=True)

bt = torch.zeros(1, dtype=torch.int16)
th = (torch.randn(1, A, NT * H) * 0.02).to(DT)
noise = (torch.randn(1, B, H) * 0.02).to(DT)
res = []
fill = 0
for target in [1024, 4096, 8192, 16000]:
    while fill + A <= target:                       # 캐시를 target 까지 채운다
        rt_a(th, torch.arange(fill, fill + A, dtype=torch.int32).unsqueeze(0),
             torch.tensor([[fill]], dtype=torch.int32), bt)
        fill += A
    pb = torch.arange(fill, fill + B, dtype=torch.int32).unsqueeze(0)
    sb = torch.tensor([[fill]], dtype=torch.int32)
    pc = torch.arange(fill, fill + A, dtype=torch.int32).unsqueeze(0)
    sc = torch.tensor([[fill]], dtype=torch.int32)
    for _ in range(3):
        rt_a(th, pc, sc, bt); rt_b(noise, pb, sb, bt)
    ta, tb = [], []
    for _ in range(10):
        s = time.time(); rt_a(th, pc, sc, bt); ta.append((time.time() - s) * 1000)
        s = time.time(); rt_b(noise, pb, sb, bt); tb.append((time.time() - s) * 1000)
    ta.sort(); tb.sort()
    ma, mb = ta[5], tb[5]
    print("ctx=%6d   append=%6.2f ms   block=%6.2f ms   total=%6.2f ms"
          % (fill, ma, mb, ma + mb), flush=True)
    res.append(dict(ctx=fill, append_ms=round(ma, 2), block_ms=round(mb, 2),
                    total_ms=round(ma + mb, 2)))
print("BTWO " + json.dumps(res), flush=True)
print("BTWO_DONE", flush=True)

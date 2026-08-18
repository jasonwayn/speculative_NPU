"""두 그래프(Append/Block) 방식의 수치 검증 + 지연 측정."""
import sys, os, time, torch, rebel
sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")
from rebel import CompileContext
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
from draft_two import Append, Block

DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
DT = torch.float16; DTS = "float16"
B = 16; A = 16
MAXC = int(os.environ.get("MAXC", "1024"))
R = int(os.environ.get("R", "3"))
DEV = int(os.environ.get("DEV", "0"))
torch.set_num_threads(8); torch.manual_seed(0)

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True); cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size; NT = len(draft.target_layer_ids)
NL = cfg.num_hidden_layers; NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
print("NL=%d NKV=%d HD=%d MAXC=%d" % (NL, NKV, HD, MAXC), flush=True)

# 두 그래프가 공유할 캐시 텐서 (동일 객체를 두 컴파일에 모두 사용)
caches = [torch.zeros(1, NKV, MAXC, HD, dtype=DT) for _ in range(2 * NL)]
cache_info = [("past_key_values_%d" % i, [1, NKV, MAXC, HD], DTS) for i in range(2 * NL)]
ctx = CompileContext(use_weight_sharing=True)
for (n, _, _), t in zip(cache_info, caches):
    ctx.mark_static_address(t, n)

a_info = [("th_new", [1, A, NT * H], DTS), ("pos_ctx", [1, A], "int32"),
          ("seq_ctx", [1, 1], "int32"), ("block_tables", [1], "int16")] + cache_info
b_info = [("noise_emb", [1, B, H], DTS), ("pos_blk", [1, B], "int32"),
          ("seq_blk", [1, 1], "int32"), ("block_tables", [1], "int16")] + cache_info


def exs(info):
    out = []
    for name, sh, dt in info:
        if name.startswith("past_key_values_"):
            out.append(caches[int(name.rsplit("_", 1)[1])])
        else:
            out.append(torch.zeros(*sh, dtype=getattr(torch, dt)))
    return out


t0 = time.time()
print("compiling Append", flush=True)
cm_a = rebel.compile_from_torch(Append(draft, MAXC).eval(), input_info=a_info,
                                example_inputs=exs(a_info), compile_context=ctx)
print("compiling Block", flush=True)
cm_b = rebel.compile_from_torch(Block(draft, MAXC).eval(), input_info=b_info,
                                example_inputs=exs(b_info), compile_context=ctx)
print("COMPILED %.1fs" % (time.time() - t0), flush=True)
rt_a = rebel.Runtime(cm_a, tensor_type="pt", device=DEV)
rt_b = rebel.Runtime(cm_b, tensor_type="pt", device=DEV)
print("RUNTIME_OK", flush=True)

bt = torch.zeros(1, dtype=torch.int16)
noise = (torch.randn(1, B, H) * 0.02).to(DT)
th_all = []
ctx_len = 0
ok = True
for r in range(R):
    th_new = (torch.randn(1, A, NT * H) * 0.02).to(DT)
    th_all.append(th_new)
    pos_ctx = torch.arange(ctx_len, ctx_len + A, dtype=torch.int32).unsqueeze(0)
    new_len = ctx_len + A
    pos_blk = torch.arange(new_len, new_len + B, dtype=torch.int32).unsqueeze(0)
    rt_a(th_new, pos_ctx, torch.tensor([[ctx_len]], dtype=torch.int32), bt)
    out = rt_b(noise, pos_blk, torch.tensor([[new_len]], dtype=torch.int32), bt)
    with torch.no_grad():
        ref = draft(position_ids=torch.arange(new_len + B).unsqueeze(0), attention_mask=None,
                    noise_embedding=noise, target_hidden=torch.cat(th_all, dim=1),
                    past_key_values=None, use_cache=False, is_causal=False)
    a = torch.as_tensor(out).float().flatten(); b = ref.float().flatten()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    if cos <= 0.99: ok = False
    print("round %d  ctx=%d->%d  cos=%.6f  %s" % (r, ctx_len, new_len, cos,
          "OK" if cos > 0.99 else "MISMATCH"), flush=True)
    ctx_len = new_len

if ok:
    args_a = (torch.zeros(1, A, NT * H, dtype=DT), torch.zeros(1, A, dtype=torch.int32),
              torch.zeros(1, 1, dtype=torch.int32), bt)
    args_b = (noise, torch.arange(B, dtype=torch.int32).unsqueeze(0),
              torch.zeros(1, 1, dtype=torch.int32), bt)
    for _ in range(3): rt_a(*args_a); rt_b(*args_b)
    ts = []
    for _ in range(10):
        s = time.time(); rt_a(*args_a); rt_b(*args_b); ts.append((time.time() - s) * 1000)
    ts.sort()
    print("STATEFUL_TOTAL_MS %.2f" % ts[len(ts) // 2], flush=True)
print("VERIFY2_DONE", flush=True)

"""상태 유지 드래프트가 원본(eager, CPU)과 같은 출력을 내는지 검증.

시나리오: 컨텍스트를 A개씩 R라운드에 걸쳐 누적하면서 블록 출력을 비교.
원본은 매번 전체 컨텍스트를 받는 stateless eager 호출.
"""
import sys, os, time, math, torch, rebel
sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")
from rebel import CompileContext
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
from draft_stateful import StatefulDraft

DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
DT = torch.float16; DTS = "float16"
B = 16; A = 16
MAXC = int(os.environ.get("MAXC", "1024"))
R = int(os.environ.get("R", "3"))
DEV = int(os.environ.get("DEV", "0"))
torch.set_num_threads(8)
torch.manual_seed(0)

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True)
cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size; NT = len(draft.target_layer_ids)
NL = cfg.num_hidden_layers; NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)

mod = StatefulDraft(draft, MAXC).eval()
info = [("noise_emb", [1, B, H], DTS), ("th_new", [1, A, NT * H], DTS),
        ("pos_ctx", [1, A], "int32"), ("pos_blk", [1, B], "int32"),
        ("seq_ctx", [1, 1], "int32"), ("seq_blk", [1, 1], "int32"),
        ("block_tables", [1], "int16")]
for i in range(NL):
    info += [("past_key_values_%d" % (2 * i), [1, NKV, MAXC, HD], DTS),
             ("past_key_values_%d" % (2 * i + 1), [1, NKV, MAXC, HD], DTS)]
ex = [torch.zeros(*sh, dtype=getattr(torch, dt)) for _, sh, dt in info]
ctx = CompileContext(use_weight_sharing=True)
for (n, _, _), t in zip(info, ex):
    if "past_key_values" in n:
        ctx.mark_static_address(t, n)
print("compiling", flush=True)
cm = rebel.compile_from_torch(mod, input_info=info, example_inputs=ex, compile_context=ctx)
rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
print("RUNTIME_OK", flush=True)

bt = torch.zeros(1, dtype=torch.int16)
noise = (torch.randn(1, B, H) * 0.02).to(DT)
th_all = []
ctx_len = 0
for r in range(R):
    th_new = (torch.randn(1, A, NT * H) * 0.02).to(DT)
    th_all.append(th_new)
    pos_ctx = torch.arange(ctx_len, ctx_len + A, dtype=torch.int32).unsqueeze(0)
    new_len = ctx_len + A
    pos_blk = torch.arange(new_len, new_len + B, dtype=torch.int32).unsqueeze(0)
    out_npu, _ = rt(noise, th_new, pos_ctx, pos_blk,
                    torch.tensor([[ctx_len]], dtype=torch.int32),
                    torch.tensor([[new_len]], dtype=torch.int32), bt)
    # 원본: 전체 컨텍스트로 stateless 호출
    th_full = torch.cat(th_all, dim=1)
    with torch.no_grad():
        ref = draft(position_ids=torch.arange(new_len + B).unsqueeze(0),
                    attention_mask=None, noise_embedding=noise,
                    target_hidden=th_full, past_key_values=None,
                    use_cache=False, is_causal=False)
    a = torch.as_tensor(out_npu).float().flatten()
    b = ref.float().flatten()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    rel = float((a - b).abs().max() / (b.abs().max() + 1e-6))
    print("round %d  ctx=%d -> %d   cos=%.6f  max_rel_err=%.4f  %s"
          % (r, ctx_len, new_len, cos, rel, "OK" if cos > 0.99 else "MISMATCH"), flush=True)
    ctx_len = new_len
print("VERIFY_DONE", flush=True)

"""상태 유지 드래프트 컴파일 + 지연 측정. 무상태 버전(171ms @16K)과 비교."""
import sys, os, time, json, math, torch, rebel
sys.path.insert(0, "/home/work/npu_work/dflash_work")
sys.path.insert(0, "/home/work/npu_work/dflash_work/dflash")
from torch import nn
from rebel import CompileContext
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
from draft_stateful import StatefulDraft

DRF = "/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
DT = torch.float16; DTS = "float16"
B = 16
A = int(os.environ.get("A", "16"))
MAXC = int(os.environ.get("MAXC", "4096"))
DEV = int(os.environ.get("DEV", "0"))
torch.set_num_threads(8)

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True)
cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size
NT = len(draft.target_layer_ids)
NL = cfg.num_hidden_layers
NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
print("NL=%d NKV=%d HD=%d NT=%d MAXC=%d" % (NL, NKV, HD, NT, MAXC), flush=True)

mod = StatefulDraft(draft, MAXC).eval()

info = [("noise_emb", [1, B, H], DTS),
        ("th_new", [1, A, NT * H], DTS),
        ("pos_ctx", [1, A], "int32"),
        ("pos_blk", [1, B], "int32"),
        ("seq_ctx", [1, 1], "int32"),
        ("seq_blk", [1, 1], "int32"),
        ("block_tables", [1], "int16")]
for i in range(NL):
    info.append(("past_key_values_%d" % (2 * i), [1, NKV, MAXC, HD], DTS))
    info.append(("past_key_values_%d" % (2 * i + 1), [1, NKV, MAXC, HD], DTS))


def mk(sh, dt):
    return torch.zeros(*sh, dtype=getattr(torch, dt))


ex = [mk(sh, dt) for _, sh, dt in info]
ctx = CompileContext(use_weight_sharing=True)
for (name, _, _), t in zip(info, ex):
    if "past_key_values" in name:
        ctx.mark_static_address(t, name)

t0 = time.time()
print("compiling ...", flush=True)
cm = rebel.compile_from_torch(mod, input_info=info, example_inputs=ex, compile_context=ctx)
print("COMPILED %.1fs" % (time.time() - t0), flush=True)
rt = rebel.Runtime(cm, tensor_type="pt", device=DEV)
print("RUNTIME_OK", flush=True)

args = (torch.zeros(1, B, H, dtype=DT),
        torch.zeros(1, A, NT * H, dtype=DT),
        torch.zeros(1, A, dtype=torch.int32),
        torch.arange(B, dtype=torch.int32).unsqueeze(0),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int16))
try:
    for _ in range(3):
        rt(*args)
    ts = []
    for _ in range(10):
        s = time.time(); rt(*args); ts.append((time.time() - s) * 1000)
    ts.sort()
    print("STATEFUL_DRAFT_MS %.2f   (ctx payload %.1f MB vs stateless %.1f MB)"
          % (ts[len(ts) // 2], A * NT * H * 2 / 1e6, MAXC * NT * H * 2 / 1e6), flush=True)
except Exception as e:
    print("CALL_FAIL %s: %s" % (type(e).__name__, str(e)[:250]), flush=True)
print("BUILD_DONE", flush=True)

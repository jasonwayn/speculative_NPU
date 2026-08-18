"""사다리 ④: DFlash 드래프트 헤드를 rebel 로 컴파일"""
import sys, torch, torch.nn as nn, traceback, json, time
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
CKPT="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"

from transformers import AutoConfig
from dflash.model import DFlashDraftModel

cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
cfg._attn_implementation = "eager"          # 디스패치 고정
print("layers",cfg.num_hidden_layers,"hidden",cfg.hidden_size,
      "block",getattr(cfg,"block_size",None),"tgt_ids",cfg.dflash_config.get("target_layer_ids"),flush=True)

draft = DFlashDraftModel.from_pretrained(CKPT, config=cfg, dtype=torch.float32).eval()
print("loaded params:", sum(p.numel() for p in draft.parameters())/1e9, "B", flush=True)

H  = cfg.hidden_size
NT = len(cfg.dflash_config["target_layer_ids"])
B  = cfg.block_size          # 16
CTX = 32                     # 컨텍스트 길이(고정) — 정적 shape

class Wrap(nn.Module):
    """plain tensor 만 받는 래퍼 (Cache 제거, use_cache=False)"""
    def __init__(s, d): super().__init__(); s.d = d
    def forward(s, noise_emb, target_hidden, position_ids, attn_mask):
        return s.d(position_ids=position_ids, attention_mask=attn_mask,
                   noise_embedding=noise_emb, target_hidden=target_hidden,
                   past_key_values=None, use_cache=False)

w = Wrap(draft).eval()
noise = torch.randn(1, B, H)
tgt   = torch.randn(1, CTX, NT*H)
pos   = torch.arange(CTX+B).unsqueeze(0)   # rotary 는 ctx+block 전체 길이 필요
mask  = torch.zeros(1, 1, B, CTX+B)          # 양방향 = 전부 보이게

print("\n=== CPU 레퍼런스 ===", flush=True)
with torch.no_grad(): ref = w(noise, tgt, pos, mask)
print("  out", tuple(ref.shape), flush=True)

print("\n=== rebel 컴파일 ===", flush=True)
import rebel
t0=time.time()
try:
    cm = rebel.compile_from_torch(w, input_info=[
        ("noise_emb",     [1,B,H],       "float32"),
        ("target_hidden", [1,CTX,NT*H],  "float32"),
        ("position_ids",  [1,CTX+B],     "int64"),
        ("attn_mask",     [1,1,B,CTX+B], "float32"),
    ])
    print(f"  COMPILE OK ({time.time()-t0:.1f}s)", flush=True)
    rt = cm.create_runtime(); print("  runtime OK", flush=True)
    got = rt(noise.numpy(), tgt.numpy(), pos.numpy(), mask.numpy())
    got = torch.as_tensor(got if not isinstance(got,(list,tuple)) else got[0])
    d = (got.float()-ref.float()).abs()
    print(f"  run OK shape={tuple(got.shape)}  max={d.max():.3e} mean={d.mean():.3e} ref|max|={ref.abs().max():.3e}", flush=True)
    ok = torch.allclose(got.float(), ref.float(), rtol=3e-2, atol=3e-2)
    print(f"  NUMERIC: {'PASS' if ok else 'MISMATCH'}", flush=True)
    print("RUNG4_RESULT " + json.dumps({"compile":True,"numeric":bool(ok)}), flush=True)
    cm.save("/home/work/npu_work/dflash_work/draft_b16.rbln")
    print("  saved draft_b16.rbln", flush=True)
except Exception as e:
    print("  COMPILE/RUN FAILED", flush=True)
    traceback.print_exc()
    print("RUNG4_RESULT " + json.dumps({"compile":False,"err":str(e)[:300]}), flush=True)

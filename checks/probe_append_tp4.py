"""block-먼저 생성은 통과. 이제 '실행 순서'를 가린다.

벤치는 append(프리필 컨텍스트 기록) -> block 순서로 실행하다 첫 block 에서
SYS_TASK_ABORTED 가 났다. 프로브 E 는 block -> append 순서로만 실행했다.

G  생성: block, append (E 와 동일)
   실행: append 여러 번 -> block   (벤치 패턴)
   이어서 block/append 교대 3회 (라운드 패턴)
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
A_W = int(os.environ.get("APPEND_WIDTH", "64"))
MAXC = int(os.environ.get("MAXC", "4096"))
KVB = int(os.environ.get("KV_BLOCK_SIZE", str(MAXC)))
NB = MAXC // KVB
TP = int(os.environ.get("TP", "4"))

cfg = AutoConfig.from_pretrained(DRF, trust_remote_code=True)
cfg._attn_implementation = "eager"
draft = DFlashDraftModel.from_pretrained(DRF, config=cfg, dtype=DT).eval()
H = cfg.hidden_size
NT = len(draft.target_layer_ids)
NL = cfg.num_hidden_layers
NKV = cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", H // cfg.num_attention_heads)

caches = [torch.zeros(NB, NKV, KVB, HD, dtype=DT) for _ in range(2 * NL)]
cinfo = [("past_key_values_%d" % i, [NB, NKV, KVB, HD], DTS) for i in range(2 * NL)]
ainfo = ([("th_new", [1, A_W, NT * H], DTS), ("angle_ctx", [1, A_W, HD // 2], DTS),
          ("seq_ctx", [1, 1], "int32"), ("block_tables", [NB], "int16")] + cinfo)
binfo = ([("noise_emb", [1, B, H], DTS), ("angle_blk", [1, B, HD // 2], DTS),
          ("seq_blk", [1, 1], "int32"), ("block_tables", [NB], "int16")] + cinfo)


def exs(info):
    out = []
    for n, sh, dt in info:
        if n.startswith("past_key_values_"):
            out.append(caches[int(n.rsplit("_", 1)[1])])
        else:
            out.append(torch.zeros(*sh, dtype=getattr(torch, dt)))
    return out


ctx = CompileContext(use_weight_sharing=True)
for (n, _, _), t in zip(cinfo, caches):
    ctx.mark_static_address(t, n)

kw = {"tensor_parallel_size": TP} if TP > 1 else {}
cm_b = rebel.compile_from_torch(BlockRR(draft, KVB).eval(), input_info=binfo,
                                example_inputs=exs(binfo), compile_context=ctx, **kw)
cm_a = rebel.compile_from_torch(AppendRR(draft, KVB).eval(), input_info=ainfo,
                                example_inputs=exs(ainfo), compile_context=ctx, **kw)
dev = list(range(TP)) if TP > 1 else 0
rt_b = rebel.Runtime(cm_b, tensor_type="pt", device=dev)
rt_a = rebel.Runtime(cm_a, tensor_type="pt", device=dev)
print("G  생성 OK (block -> append, TP%d, A_W=%d)" % (TP, A_W), flush=True)

torch.manual_seed(0)


def run_append(off):
    a = [torch.randn(1, A_W, NT * H, dtype=DT),
         torch.randn(1, A_W, HD // 2, dtype=DT),
         torch.tensor([[off]], dtype=torch.int32),
         torch.arange(NB, dtype=torch.int16)]
    s = time.time()
    rt_a(*[t.contiguous() for t in a])
    return (time.time() - s) * 1000


def run_block(off):
    a = [torch.randn(1, B, H, dtype=DT),
         torch.randn(1, B, HD // 2, dtype=DT),
         torch.tensor([[off]], dtype=torch.int32),
         torch.arange(NB, dtype=torch.int16)]
    s = time.time()
    rt_b(*[t.contiguous() for t in a])
    return (time.time() - s) * 1000


try:
    # 가설: 첫 실행 그래프가 캐시의 디바이스 배치를 굳힌다. block 을 먼저 한 번
    # 실행해 배치를 잡아두면 이후 append -> block 순서도 살아야 한다.
    if os.environ.get("BLOCK_WARM", "0") == "1":
        ms = run_block(0)
        print("   block 워밍업 (off=0)  %.2f ms" % ms, flush=True)
    # 벤치의 프리필: append 를 여러 번 (컨텍스트 256 = 폭 64 x 4)
    for i in range(4):
        ms = run_append(i * A_W)
        print("   append #%d (off=%d)  %.2f ms" % (i, i * A_W, ms), flush=True)
    # 첫 block — 벤치가 죽은 지점
    ms = run_block(256)
    print("   block  (off=256)  %.2f ms   <- 벤치가 죽던 지점 통과" % ms, flush=True)
    # 라운드 패턴: block <-> append 교대
    for i in range(3):
        mb = run_block(256 + i * 8)
        ma = run_append(256 + i * 8)
        print("   round %d  block %.2f / append %.2f ms" % (i, mb, ma), flush=True)
    print("G  전부 생존", flush=True)
except Exception as exc:  # noqa: BLE001
    print("G  실행 실패  %s: %s" % (type(exc).__name__, str(exc)[:200]), flush=True)

print("PROBE4_DONE", flush=True)

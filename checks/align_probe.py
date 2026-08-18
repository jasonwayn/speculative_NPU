"""prefill 재개 오프셋이 64의 배수가 아니어도 되는가?

되면 꼬리 재계산(verify 계산의 약 40%)을 없앨 수 있다.
  정렬 방식 : 마지막 64경계부터 다시 보냄  -> 꼬리 재계산
  비정렬    : 새 토큰만 보냄               -> 재계산 0

두 방식의 hidden 을 비교해 값이 같은지 본다.
"""
import torch
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer

D = "/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
tok = AutoTokenizer.from_pretrained(SRC)
m = RBLNQwen3ForCausalLM.from_pretrained(D, export=False, rbln_device=1)
pdec = m.prefill_decoder
CH = pdec.rbln_config.prefill_chunk_size
BT = torch.tensor([0], dtype=torch.int16)
print("chunk=%d" % CH, flush=True)

txt = open("/home/work/npu_work/dflash_work/pg1342.txt", encoding="utf-8").read()
full = tok(txt, return_tensors="pt").input_ids
P = 200            # 64 의 배수가 아님 (200 = 3*64 + 8)
NEW = 16
ids = full[:, :P].contiguous()
nxt = full[:, P:P + NEW].contiguous()


def pre(seg, off):
    L = seg.shape[1]
    pad = 1 if L % CH == 0 else 0
    if pad:
        seg = torch.cat([seg, seg[:, -1:]], dim=1)
    r = pdec.prefill_forward(
        seg,
        cache_position=torch.arange(off, off + seg.shape[1], dtype=torch.int32).unsqueeze(0),
        attention_mask=torch.ones(seg.shape[1], dtype=torch.int64),
        batch_idx=0, block_tables=BT, is_external_block_tables=False)
    hs = r.hidden_states
    return tuple(h[:, :L] for h in hs) if pad else hs


# 기준: 전체를 한 번에 (P + NEW)
ref = pre(torch.cat([ids, nxt], dim=1), 0)[-1][:, P:P + NEW].float()

# A) 정렬 방식: 0..P 넣고, 마지막 64경계(192)부터 꼬리+새것 재전송
pre(ids, 0)
al = (P // CH) * CH
tail = torch.cat([full[:, al:P], nxt], dim=1)
hA = pre(tail, al)[-1][:, P - al:P - al + NEW].float()

# B) 비정렬 방식: 0..P 넣고, P(=200) 에서 바로 새 토큰만
pre(ids, 0)
try:
    hB = pre(nxt, P)[-1][:, :NEW].float()
    okB = True
except Exception as e:
    okB = False
    print("UNALIGNED_FAIL %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)


def cmp(name, a, b):
    c = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    print("%-22s cos=%.6f  %s" % (name, c, "MATCH" if c > 0.999 else "MISMATCH"), flush=True)


cmp("aligned vs ref", hA, ref)
if okB:
    cmp("UNALIGNED vs ref", hB, ref)
print("ALIGN_DONE", flush=True)

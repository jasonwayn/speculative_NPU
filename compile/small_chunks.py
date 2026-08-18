"""청크 8 / 16 / 32 / 64 의 호출당 비용.

목적: "여러 크기 그래프를 두고 seg 에 맞춰 고르면 얼마나 이득인가" 의 상한 계산.
호출당 비용 = 가중치 스트리밍(고정) + 위치수 x 단가 이므로, 작은 청크는 고정비에 수렴한다.
"""
import os, sys, time, torch

import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as _M
_C = _M.RBLNDecoderOnlyModelForCausalLMConfig
_orig = _C.__init__


def _patched(self, *a, **kw):
    want = kw.get("prefill_chunk_size")
    if want is not None and want % 64 != 0:
        kw = dict(kw); kw["prefill_chunk_size"] = 64
        _orig(self, *a, **kw); self.prefill_chunk_size = want
    else:
        _orig(self, *a, **kw)


_C.__init__ = _patched
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
D = "/home/work/npu_work/dflash_work"
DEV = int(os.environ.get("DEV", "1"))
CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "8,16,32,64").split(",")]

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids

for ch in CHUNKS:
    path = D + ("/rbln-Qwen3-4B-h-c64" if ch == 64 else "/rbln-Qwen3-4B-h-c%d" % ch)
    if not os.path.exists(path):
        t = time.time()
        try:
            m = RBLNQwen3ForCausalLM.from_pretrained(
                SRC, export=True, rbln_batch_size=1, rbln_max_seq_len=2048,
                rbln_prefill_chunk_size=ch, rbln_output_hidden_states=True)
            m.save_pretrained(path); del m
            print("compiled chunk=%d %.0fs" % (ch, time.time() - t), flush=True)
        except Exception as e:
            print("COMPILE_FAIL chunk=%d %s: %s" % (ch, type(e).__name__, str(e)[:150]), flush=True)
            continue
    try:
        m = RBLNQwen3ForCausalLM.from_pretrained(path, export=False, rbln_device=DEV)
    except Exception as e:
        print("LOAD_FAIL chunk=%d %s" % (ch, str(e)[:90]), flush=True); continue
    pd = m.prefill_decoder
    BT = torch.tensor([0], dtype=torch.int16)
    row = []
    for L in [9, 17, 25, 33, 49]:
        if L % ch == 0:
            row.append((L, "skip")); continue
        seg = full[:, :L].contiguous()
        cp = torch.arange(0, L, dtype=torch.int32).unsqueeze(0)
        am = torch.ones(L, dtype=torch.int64)

        def call():
            pd.prefill_forward(seg, cache_position=cp, attention_mask=am, batch_idx=0,
                               block_tables=BT, is_external_block_tables=False)
        try:
            call(); call()
            ts = []
            for _ in range(5):
                s = time.time(); call(); ts.append((time.time() - s) * 1000)
            ts.sort()
            n = (L + ch - 1) // ch
            row.append((L, "%.1f/%d" % (ts[2], n)))
        except Exception as e:
            row.append((L, "FAIL"))
    print("chunk=%-4d %s" % (ch, "  ".join("L=%-3s %s" % (a, b) for a, b in row)), flush=True)
    del m
print("SMALL_DONE", flush=True)

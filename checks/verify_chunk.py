"""verify 크기(17~65 토큰) 호출을 chunk64 vs chunk32 로 비교.

verify 는 짧은 세그먼트를 매 라운드 부르므로, 여기서 어느 쪽이 빠른지가 실제 지표다.
프리픽스 캐싱 없이 offset 0 에서 호출해 ceil(L/chunk) x 청크비용 만 본다.
길이는 전부 홀수로 잡아 "chunk 배수면 크래시" 함정을 피한다.
"""
import torch, time
# 로딩 때도 64 배수 검증이 걸리므로 우회한다
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
tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids
LENS = [17, 33, 49, 65, 129]

for name, path in [("chunk64", D + "/rbln-Qwen3-4B-h-c64"),
                   ("chunk32", D + "/rbln-Qwen3-4B-h-c32")]:
    try:
        m = RBLNQwen3ForCausalLM.from_pretrained(path, export=False, rbln_device=1)
    except Exception as e:
        print("%-8s LOAD_FAIL %s" % (name, str(e)[:80]), flush=True); continue
    pd = m.prefill_decoder
    ch = pd.rbln_config.prefill_chunk_size
    BT = torch.tensor([0], dtype=torch.int16)
    row = []
    for L in LENS:
        seg = full[:, :L].contiguous()
        cp = torch.arange(0, L, dtype=torch.int32).unsqueeze(0)
        am = torch.ones(L, dtype=torch.int64)

        def call():
            pd.prefill_forward(seg, cache_position=cp, attention_mask=am,
                               batch_idx=0, block_tables=BT, is_external_block_tables=False)
        try:
            call(); call()
            ts = []
            for _ in range(5):
                s = time.time(); call(); ts.append((time.time() - s) * 1000)
            ts.sort()
            row.append((L, round(ts[2], 1), (L + ch - 1) // ch))
        except Exception as e:
            row.append((L, "FAIL", 0))
    print("%-8s chunk=%-3d  %s" % (name, ch,
          "  ".join("L=%-4s %sms(%s청크)" % (a, b, c) for a, b, c in row)), flush=True)
    del m
print("VC_DONE", flush=True)

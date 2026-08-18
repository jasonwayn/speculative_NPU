"""prefill_chunk_size 64 / 128 / 256 비교.

핵심 질문: 큰 청크가 가중치 스트리밍을 상각하는가?
  상각된다면  -> 청크 256 의 1회 호출이 청크 64 의 1회와 비슷 -> 긴 prefill 이 유리
  안 된다면   -> 위치 수에 비례 -> 큰 청크는 손해

N=17  : verify 1라운드에 해당
N=1024/4096 : 초기 prefill 에 해당
"""
import os, sys, time, json, torch
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
D = "/home/work/npu_work/dflash_work"
DEV = int(os.environ.get("DEV", "1"))
torch.set_num_threads(8)

C256 = D + "/rbln-Qwen3-4B-h-c256"
if not os.path.exists(C256):
    print("compiling chunk=256 ...", flush=True)
    t = time.time()
    m = RBLNQwen3ForCausalLM.from_pretrained(
        SRC, export=True, rbln_batch_size=1, rbln_max_seq_len=4096,
        rbln_prefill_chunk_size=256, rbln_output_hidden_states=True)
    m.save_pretrained(C256)
    print("compiled %.1fs" % (time.time() - t), flush=True)
    del m

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids

MODELS = [("chunk64", D + "/rbln-Qwen3-4B-h-c64"),
          ("chunk128", D + "/rbln-Qwen3-4B-hidden"),
          ("chunk256", C256)]
LENS = [17, 256, 1024, 4000]
res = {}
for name, path in MODELS:
    try:
        m = RBLNQwen3ForCausalLM.from_pretrained(path, export=False, rbln_device=DEV)
        ch = m.prefill_decoder.rbln_config.prefill_chunk_size
    except Exception as e:
        print("%-9s LOAD_FAIL %s" % (name, str(e)[:90]), flush=True); continue
    row = []
    for L in LENS:
        ids = full[:, :L].contiguous()
        pad = 1 if L % ch == 0 else 0
        i2 = torch.cat([ids, ids[:, -1:]], dim=1) if pad else ids
        try:
            m(input_ids=i2, attention_mask=torch.ones_like(i2))          # warm
            ts = []
            for _ in range(3):
                s = time.time()
                m(input_ids=i2, attention_mask=torch.ones_like(i2))
                ts.append((time.time() - s) * 1000)
            ts.sort()
            row.append((L, round(ts[1], 1)))
        except Exception as e:
            row.append((L, "FAIL"))
    print("%-9s chunk=%-4d  %s" % (name, ch, "  ".join("N=%-5s %sms" % (a, b) for a, b in row)), flush=True)
    res[name] = row
    del m
print("CHUNK_AB " + json.dumps(res), flush=True)
print("CHUNK_AB_DONE", flush=True)

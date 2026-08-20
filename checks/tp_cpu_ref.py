"""타깃 hidden state 의 CPU 기준값. TP1/TP2/TP4 결과와 비교.

fused 그래프의 KEEP = (2,10,18,26,34) + verify 는 마지막(36) 하나 더 = 6개.
CPU 는 같은 프롬프트 1024 토큰을 넣고 이어서 16 토큰을 넣어 같은 위치의 hidden 을 뽑는다.
"""
import os, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

D = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
P, NEW = 1024, 16
KEEP = (2, 10, 18, 26, 34, 36)

torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids
ids = full[:, :P + NEW].contiguous()

m = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32).eval()
print("cpu model loaded, forwarding %d tokens ..." % ids.shape[1], flush=True)
with torch.no_grad():
    o = m(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
hs = o.hidden_states
print("n_hidden=%d" % len(hs), flush=True)

ref = dict(prefill_last=hs[KEEP[-2]][:, :P].clone(),      # prefill 그래프 출력 5개 중 마지막 = idx 34
           verify=[hs[k][:, P:P + NEW].clone() for k in KEEP])
torch.save(ref, "/tmp/tpcpu.pt")
print("saved /tmp/tpcpu.pt", flush=True)


def cos(a, b):
    c = torch.nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]).float(),
                                              b.reshape(-1, b.shape[-1]).float(), dim=-1)
    return float(c.min()), float(c.mean())


for t in (1, 2, 4):
    p = "/tmp/tp%d.pt" % t
    if not os.path.exists(p):
        print("missing %s" % p, flush=True); continue
    d = torch.load(p)
    mn, me = cos(ref["prefill_last"], d["prefill_last"])
    print("PREFILL_LAST(idx34)  CPU vs TP%d  cos_min=%.6f cos_mean=%.6f" % (t, mn, me), flush=True)
    for i, (a, b) in enumerate(zip(ref["verify"], d["verify"])):
        mn, me = cos(a, b)
        print("VERIFY idx%-3d        CPU vs TP%d  cos_min=%.6f cos_mean=%.6f"
              % (KEEP[i], t, mn, me), flush=True)
print("TPCPU_DONE", flush=True)

"""twh/tpre 가 hidden state 를 5개로 잘라 반환해서 레이어 35(인덱스 5)가 사라진다.
CMR 은 그 레이어를 쓰므로 6개까지 넘긴다. 특징 추출(range(5))은 그대로 둔다.
"""
import io, sys
p = "/home/work/npu_work/dflash_work/bench_sf_rr_cmr.py"
t = io.open(p, encoding="utf-8").read()
n = {}

n["a"] = t.count("T[\"verify\"] += time.time() - s; hs = r.hidden_states[:5]")
t = t.replace("T[\"verify\"] += time.time() - s; hs = r.hidden_states[:5]",
              "T[\"verify\"] += time.time() - s; hs = r.hidden_states[:6]   # 레이어 35 포함")

n["b"] = t.count("""            hs = tuple(torch.cat([r0.hidden_states[i], r1.hidden_states[i]], dim=1)
                       for i in range(5))""")
t = t.replace("""            hs = tuple(torch.cat([r0.hidden_states[i], r1.hidden_states[i]], dim=1)
                       for i in range(5))""",
"""            hs = tuple(torch.cat([r0.hidden_states[i], r1.hidden_states[i]], dim=1)
                       for i in range(6))   # 레이어 35 포함""")

n["c"] = t.count("            hs = r.hidden_states[:5]\n            logits = r.logits")
t = t.replace("            hs = r.hidden_states[:5]\n            logits = r.logits",
              "            hs = r.hidden_states[:6]   # 레이어 35 포함\n            logits = r.logits")

if n["a"] != 1 or n["b"] != 1 or n["c"] != 1:
    print("MISS %s" % n); sys.exit(1)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patched %s" % n)

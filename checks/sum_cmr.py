import json, glob, os
D = "/home/work/npu_work/dflash_work/cmr_recheck"
rows = {}
for f in sorted(glob.glob(D + "/*.txt")):
    b = os.path.basename(f)[:-4]
    l = open(f).read().strip()
    if not l.startswith("SF "):
        print("%-34s (결과 없음) %s" % (b, l[:80])); continue
    rows[b] = json.loads(l[3:])
print("%-34s %-8s %-7s %-9s %-8s %-8s %s" % ("", "tau", "rounds", "tok/s", "draft", "verify", "kept"))
for b in sorted(rows):
    d = rows[b]
    print("%-34s %-8.3f %-7d %-9.2f %-8.1f %-8.1f %.0f"
          % (b, d["tau"], d["rounds"], d["decode_tok_s"], d["draft_ms"],
             d["verify_ms"], d["kept_mean"]))
def g(k):
    return rows.get(k, {}).get("tau")
print()
for v, tag in (("bench_sf_fused_diag", "RoPE 버그 있음"), ("bench_sf_rr", "RoPE 수정됨")):
    a, b = g(v + "_cmr0"), g(v + "_cmr1")
    if a and b:
        print("%-16s  CMR off %.3f -> on %.3f   =  %+.1f%%" % (tag, a, b, (b / a - 1) * 100))

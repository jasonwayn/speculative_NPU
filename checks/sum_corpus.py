import json, glob, os
D = "/home/work/npu_work/dflash_work/corpus_results"
R = {}
for f in sorted(glob.glob(D + "/*.txt")):
    b = os.path.basename(f)[:-4]
    corp, L, cmr = b.rsplit("_", 2)
    s = open(f).read().strip()
    if not s.startswith("SF "):
        continue
    R[(corp, int(L), int(cmr[-1]))] = json.loads(s[3:])

GPU = {  # spec/results/cmr_*_ab_n10.json  (stock_tau, cmr_tau)
    ("pg19", 1024): (2.558, 2.526), ("pg19", 2048): (2.540, 2.374),
    ("pg19", 4096): (2.301, 2.219), ("pg19", 8192): (1.897, 2.150),
    ("pg19", 16384): (1.732, 2.202),
    ("govreport", 1024): (2.740, 2.700), ("govreport", 2048): (2.851, 2.638),
    ("govreport", 4096): (2.666, 2.557), ("govreport", 8192): (2.130, 2.365),
    ("govreport", 16384): (1.823, 2.405),
}

for corp in ("pg19", "govreport"):
    print("=" * 96)
    print("%s   %-38s | %s" % (corp, "NPU 4장 (RoPE 수정판)", "GPU 1장 (기존)"))
    print("%-7s %-8s %-8s %-8s %-9s %-9s | %-8s %-8s %s"
          % ("길이", "stock", "CMR", "차이", "stock tok/s", "CMR tok/s", "stock", "CMR", "차이"))
    for L in (1024, 2048, 4096, 8192, 16384):
        a, b = R.get((corp, L, 0)), R.get((corp, L, 1))
        if not a or not b:
            print("%-7d (없음)" % L); continue
        g = GPU.get((corp, L), (None, None))
        d = (b["tau"] / a["tau"] - 1) * 100
        gd = (g[1] / g[0] - 1) * 100 if g[0] else 0
        print("%-7d %-8.3f %-8.3f %+-8.1f%% %-9.1f %-9.1f | %-8.3f %-8.3f %+.1f%%"
              % (L, a["tau"], b["tau"], d, a["decode_tok_s"], b["decode_tok_s"],
                 g[0], g[1], gd))
    print("  kept_mean(CMR): " + "  ".join(
        "%d:%.0f" % (L, R[(corp, L, 1)]["kept_mean"]) for L in (1024, 2048, 4096, 8192, 16384)
        if (corp, L, 1) in R))

"""bench_sf_dual.py 의 tpre/twh 앞머리 복구.

패치 과정에서 `s = time.time(); L = ids.shape[1];` 접두가 날아갔다.
"""
import io, sys

p = "/home/work/npu_work/dflash_work/bench_sf_dual.py"
lines = io.open(p, encoding="utf-8").read().splitlines()
fixed = 0
for i, l in enumerate(lines):
    st = l.strip()
    if st == "pad = 1 if L % CH_P == 0 else 0":
        ind = len(l) - len(l.lstrip())
        lines[i] = " " * ind + "s = time.time(); L = ids.shape[1]; pad = 1 if L % CH_P == 0 else 0"
        fixed += 1
    elif st == "pad = 1 if L % CH_V == 0 else 0":
        ind = len(l) - len(l.lstrip())
        lines[i] = " " * ind + "s = time.time(); L = seg.shape[1]; pad = 1 if L % CH_V == 0 else 0"
        fixed += 1
if fixed == 0:
    print("NOTHING_FIXED"); sys.exit(1)
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("fixed", fixed)

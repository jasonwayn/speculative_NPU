#!/usr/bin/env bash
# 원본(카드1) vs 최적화(카드2) 를 동시에 실행 -> 같은 교란 조건에서 비교.
cd /home/work/npu_work/dflash_work || exit 1
COMMON="NTHREADS=2 PREC=float16 B=16 NSAMP=3 MAXNEW=256 DSETS=gsm8k BUCKETS=256,512,1024 TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64"
rm -f ab_orig.txt ab_opt.txt
env $COMMON DEV=1 timeout 2400 python3 bench2.py     > ab_orig.raw 2>&1 &
P1=$!
env $COMMON DEV=2 timeout 2400 python3 bench2_opt.py > ab_opt.raw  2>&1 &
P2=$!
wait $P1; wait $P2
grep -a '^BENCH2 {' ab_orig.raw | head -1 | sed 's|^BENCH2 |ORIG(dev1) |' > ab_orig.txt
grep -a '^BENCH2 {' ab_opt.raw  | head -1 | sed 's|^BENCH2 |OPT (dev2) |' > ab_opt.txt
cat ab_orig.txt ab_opt.txt
echo PAR_DONE

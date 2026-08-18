#!/usr/bin/env bash
# fp16 / int8 / int4 타깃을 동일 조건, 동일 카드에서 순차 측정.
cd /home/work/npu_work/dflash_work
D=/home/work/npu_work/dflash_work
export NSAMP=20 MAXNEW=512 B=16 DEV=0 NTHREADS=2 PREC=float16 \
       BUCKETS=256,512,1024 DSETS=gsm8k
for T in h-c64 h-c64-int8 h-c64-int4; do
  echo "### $T"
  TGTDIR=$D/rbln-Qwen3-4B-$T timeout 3600 python3 bench2.py 2>&1 \
    | grep -E "^BENCH2 |Error|error|Traceback" | head -5
done
echo RUNQ_DONE

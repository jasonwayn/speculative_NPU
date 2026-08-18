#!/bin/bash
cd /home/work/npu_work/dflash_work || exit 1
export NTHREADS=2 OMP_NUM_THREADS=2 PREC=float16 B=16 NSAMP=60 MAXNEW=2048 BUCKETS=2048 DSETS=gsm8k DEV=0
echo "##### chunk128 #####"
TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden \
  python3 bench2.py 2>&1 | grep -E "^BENCH2 |Error" | cut -c1-400
echo "##### chunk64 #####"
TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64 \
  python3 bench2.py 2>&1 | grep -E "^BENCH2 |Error" | cut -c1-400
echo "##### DONE #####"

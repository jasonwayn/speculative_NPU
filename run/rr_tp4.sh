#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
export DRAFT_DTYPE=float32 PROMPT_LEN=1024 NATURAL_LONG=1 MAXC=4096 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4 \
       MAXNEW=256 NSAMP=3 NTHREADS=2
for i in 1 2; do
  for v in bench_sf_fused_diag.py bench_sf_rr.py; do
    L=$(timeout 3600 python3 "$v" 2>&1 | grep -a '^SF ' | head -1)
    echo "r${i} ${v} ${L}"
  done
done
echo RRTP4_DONE

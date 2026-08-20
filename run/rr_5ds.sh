#!/usr/bin/env bash
# 논문 워크로드 5종, 수정 전/후. 짧게: NSAMP=5, MAXNEW=512, TP4 (목표 구성).
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p rr_results
export DRAFT_DTYPE=float32 MAXC=4096 DEV=0 DEV_TARGET=0 DEV_DRAFT=0 \
       TARGET_TP=4 TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4 \
       NSAMP=5 MAXNEW=512 NTHREADS=2
for DS in gsm8k math500 humaneval mbpp mt-bench; do
  for v in bench_sf_fused_diag.py bench_sf_rr.py; do
    out="rr_results/${DS}_${v%.py}.txt"
    if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $out"; continue; fi
    DSET=$DS timeout 3600 python3 "$v" 2>&1 | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
    echo "${DS} ${v} $(cat "$out")"
  done
done
echo RR5_DONE

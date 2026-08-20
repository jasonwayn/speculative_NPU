#!/usr/bin/env bash
# MAXNEW=2048 (원래 프로토콜). 여기서만 position 이 1000 을 넘어 RoPE 결함 구간에 들어간다.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p rr_long
export DRAFT_DTYPE=float32 MAXC=4096 DEV=0 DEV_TARGET=0 DEV_DRAFT=0 \
       TARGET_TP=4 TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4 \
       NSAMP=5 MAXNEW=2048 NTHREADS=2
for DS in math500 gsm8k; do
  for v in bench_sf_fused_diag.py bench_sf_rr.py; do
    out="rr_long/${DS}_${v%.py}.txt"
    DSET=$DS timeout 3600 python3 "$v" 2>&1 | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
    echo "${DS} ${v} $(cat "$out")"
  done
done
echo RRLONG_DONE

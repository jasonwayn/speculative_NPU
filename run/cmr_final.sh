#!/usr/bin/env bash
# RoPE 수정 후 CMR on/off. 그래프는 레이어 35 를 내보내는 fused_tp4_cmr.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p cmr_recheck
export DRAFT_DTYPE=float32 PROMPT_LEN=2048 NATURAL_LONG=1 MAXNEW=256 NSAMP=3 \
       MAXC=4096 DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_cmr \
       CMR_BUDGET=1024 CMR_CHUNK=32 CMR_TOPK=32 CMR_EVERY=4 NTHREADS=2
for C in 0 1; do
  out="cmr_recheck/fixed_cmr${C}.txt"
  CMR=$C timeout 3600 python3 bench_sf_rr_cmr.py 2>&1 | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
  echo "CMR=$C  $(head -c 400 "$out")"
done
echo CMRFIN_DONE

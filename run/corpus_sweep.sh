#!/usr/bin/env bash
# 실제 코퍼스(pg19, govreport) 길이별 stock vs CMR. GPU 실험과 동일 입력·동일 MAXNEW.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p corpus_results
export DRAFT_DTYPE=float32 MAXNEW=256 NSAMP=5 MAXC=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_20k \
       CMR_BUDGET=1024 CMR_CHUNK=32 CMR_TOPK=32 CMR_EVERY=4 NTHREADS=2
for CORP in pg19 govreport; do
  for L in 1024 2048 4096 8192 16384; do
    for C in 0 1; do
      out="corpus_results/${CORP}_${L}_cmr${C}.txt"
      if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $out"; continue; fi
      echo "### $CORP $L cmr$C  $(date +%H:%M:%S)"
      CORPUS=$CORP CORPUS_LEN=$L CMR=$C timeout 7200 python3 bench_sf_corpus.py 2>&1 \
        | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
      head -c 220 "$out"; echo
    done
  done
done
echo CORPSWEEP_DONE

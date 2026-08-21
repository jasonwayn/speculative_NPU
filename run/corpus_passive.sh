#!/usr/bin/env bash
# corpus_sweep.sh 재측정. 유일한 차이는 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0.
# 기존 corpus_results/ 는 NTHREADS=2 + OMP spin(곡선의 최악점)에서 측정됨.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p corpus_passive
export DRAFT_DTYPE=float32 MAXNEW=256 NSAMP=5 MAXC=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_20k \
       CMR_BUDGET=1024 CMR_CHUNK=32 CMR_TOPK=32 CMR_EVERY=4 NTHREADS=2 \
       OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
for CORP in pg19 govreport; do
  for L in 16384 8192 4096 2048 1024; do
    for C in 0 1; do
      out="corpus_passive/${CORP}_${L}_cmr${C}.txt"
      if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $out"; continue; fi
      echo "### $CORP $L cmr$C  $(date +%H:%M:%S)"
      CORPUS=$CORP CORPUS_LEN=$L CMR=$C timeout 7200 python3 bench_sf_diag3.py 2>&1 \
        | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
      head -c 200 "$out"; echo
    done
  done
done
echo CORPPASS_DONE

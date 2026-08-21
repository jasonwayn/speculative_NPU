#!/usr/bin/env bash
# 스코어링에 쓰는 KV 그룹 수를 줄여도 tau 가 유지되는가.
# TP4 에서 카드 하나가 보는 몫은 nkv/4 = 2 그룹(쿼리헤드 8개).
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p heads_ab
export DRAFT_DTYPE=float32 MAXNEW=256 NSAMP=5 MAXC=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_20k \
       CMR_BUDGET=1024 CMR_CHUNK=32 CMR_TOPK=32 CMR_EVERY=4 NTHREADS=2 \
       OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
for CORP in pg19 govreport; do
  for G in 8 2 1; do
    out="heads_ab/${CORP}_g${G}.txt"
    if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $out"; continue; fi
    echo "### $CORP g$G  $(date +%H:%M:%S)"
    CORPUS=$CORP CORPUS_LEN=16384 CMR=1 SCORE_KV_GROUPS=$G timeout 7200 \
      python3 bench_sf_corpus.py 2>&1 | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
    head -c 180 "$out"; echo
  done
done
echo HEADSAB_DONE

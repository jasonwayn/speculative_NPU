#!/usr/bin/env bash
# 2단계: budget 은 1024 로 고정(1024 > 2048 > 4096 확인됨), every 곡선을 끝까지.
# 4->8 이 +24% 인데 tau 는 -0.5% 뿐이었다. 어디서 꺾이는지가 관건.
cd /home/work/npu_work/dflash_work || exit 1
until grep -q CMRTUNE_DONE cmrtune.log 2>/dev/null; do sleep 20; done
mkdir -p cmr_tune
export DRAFT_DTYPE=float32 MAXNEW=256 NSAMP=5 MAXC=20480 TARGET_MAX_SEQ=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_20k \
       CMR_CHUNK=32 CMR_TOPK=32 CMR_BUDGET=1024 NTHREADS=2 \
       CORPUS=pg19 CORPUS_LEN=16384 CMR=1
for E in 16 32 64; do
  out="cmr_tune/k32_e${E}.txt"
  if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $out"; continue; fi
  echo "### every=$E  $(date +%H:%M:%S)"
  CMR_EVERY=$E timeout 7200 python3 bench_sf_corpus.py 2>&1 \
    | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
  head -c 260 "$out"; echo
done
echo CMREVERY_DONE

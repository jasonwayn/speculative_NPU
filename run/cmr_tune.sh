#!/usr/bin/env bash
# 1단계: DFlash 드래프터에 맞는 CMR 설정 탐색.
# budget/every 는 논문의 Vicuna-68M 최적값(1024/4)을 그대로 물려받은 것이고
# 우리 드래프터에 맞춰본 적이 없다. 논문 Table 8 은 드래프터가 강할수록
# 최적 working cache 가 크다고 보고한다 (68M 1024, EAGLE 2048).
# 실제 보존량은 topk x chunk 이므로 TOPK 를 올려야 한다 (BUDGET 은 상한일 뿐).
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p cmr_tune
export DRAFT_DTYPE=float32 MAXNEW=256 NSAMP=5 MAXC=20480 TARGET_MAX_SEQ=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_20k \
       CMR_CHUNK=32 NTHREADS=2 \
       CORPUS=pg19 CORPUS_LEN=16384 CMR=1
for K in 32 64 128; do
  for E in 4 8; do
    B=$(( K * 32 ))
    out="cmr_tune/k${K}_e${E}.txt"
    if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $out"; continue; fi
    echo "### topk=$K budget=$B every=$E  $(date +%H:%M:%S)"
    CMR_TOPK=$K CMR_BUDGET=$B CMR_EVERY=$E timeout 7200 python3 bench_sf_corpus.py 2>&1 \
      | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
    head -c 260 "$out"; echo
  done
done
echo CMRTUNE_DONE

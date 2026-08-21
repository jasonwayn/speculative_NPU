#!/usr/bin/env bash
# 극한: 프리필 검색 한 번, 이후 스코어러 완전 정지. 그리고 중간값 128 / 256.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p cmr_tune
export DRAFT_DTYPE=float32 MAXNEW=256 NSAMP=5 MAXC=20480 TARGET_MAX_SEQ=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_20k \
       CMR_CHUNK=32 CMR_TOPK=32 CMR_BUDGET=1024 NTHREADS=2 \
       CORPUS=pg19 CORPUS_LEN=16384 CMR=1
run () {
  out="cmr_tune/$1.txt"
  if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $out"; return; fi
  echo "### $1  $(date +%H:%M:%S)"
  env "${@:2}" timeout 7200 python3 bench_sf_corpus.py 2>&1 \
    | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
  head -c 260 "$out"; echo
}
run k32_e128 CMR_EVERY=128
run k32_once CMR_EVERY=999999 CMR_ONCE=1
echo CMRONCE_DONE

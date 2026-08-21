#!/usr/bin/env bash
# v2(GQA 확장 제거) + KST 사전할당. 앞 스윕이 끝나면 이어서 돈다.
cd /home/work/npu_work/dflash_work || exit 1
until grep -q CORPV2_DONE corpus_v2.log 2>/dev/null; do sleep 30; done
mkdir -p corpus_v3
export DRAFT_DTYPE=float32 MAXNEW=256 NSAMP=5 MAXC=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_20k \
       CMR_BUDGET=1024 CMR_CHUNK=32 CMR_TOPK=32 CMR_EVERY=4 NTHREADS=2 \
       OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 KST_PRE=1
for CORP in pg19 govreport; do
  for L in 16384 8192 4096 2048 1024; do
    out="corpus_v3/${CORP}_${L}_cmr1.txt"
    if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $out"; continue; fi
    echo "### $CORP $L  $(date +%H:%M:%S)"
    CORPUS=$CORP CORPUS_LEN=$L CMR=1 timeout 7200 python3 bench_sf_diag3.py 2>&1 \
      | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
    head -c 200 "$out"; echo
  done
done
echo CORPV3_DONE

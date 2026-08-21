#!/usr/bin/env bash
# CMR 의 긴 컨텍스트 이득이 RoPE 버그 아티팩트였는지 확인.
# CMR 은 검색할 때마다 드래프터 position 을 0 부터 다시 매긴다(budget 1024).
# 버그가 있으면 position 을 작게 유지하는 것만으로 tau 가 오른다.
# RoPE 를 고친 뒤에도 CMR 이득이 남는지 본다.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p cmr_recheck
export DRAFT_DTYPE=float32 PROMPT_LEN=2048 NATURAL_LONG=1 MAXNEW=256 NSAMP=3 \
       MAXC=4096 DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4 \
       CMR_BUDGET=1024 CMR_CHUNK=32 CMR_TOPK=32 CMR_EVERY=4 NTHREADS=2
for v in bench_sf_fused_diag.py bench_sf_rr.py; do
  for C in 0 1; do
    out="cmr_recheck/${v%.py}_cmr${C}.txt"
    CMR=$C timeout 3600 python3 "$v" 2>&1 | grep -aE '^SF |Traceback|Error' | head -2 > "$out"
    echo "${v%.py} CMR=$C  $(cat "$out" | head -c 400)"
  done
done
echo CMRRE_DONE

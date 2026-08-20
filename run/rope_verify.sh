#!/usr/bin/env bash
# 수정 전/후 드래프트 정확도를 1024 컨텍스트에서 비교 (프로덕션 paged 경로).
cd /home/work/npu_work/dflash_work || exit 1
export DRAFT_DTYPE=float32 PROMPT_LEN=1024 NATURAL_LONG=1 MAXNEW=1 WARMNEW=0 \
       NSAMP=1 MAXC=4096 DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=1 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_17_256 \
       DRAFT_DIAG=1 NTHREADS=2
for v in bench_sf_fused_diag.py bench_sf_rope.py; do
  echo "### $v"
  timeout 2400 python3 "$v" 2>&1 | grep -aE 'DRAFT_DIAG|^SF |Traceback|Error' | head -6
done
echo ROPEVER_DONE

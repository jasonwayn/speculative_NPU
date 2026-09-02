#!/usr/bin/env bash
# 드래프터·lm_head 까지 TP4. 기준선(DRAFT_TP=1)과 A/B.
# 판정: tau 5.115 유지 + 라운드 단축. tau 가 깨지면 샤딩이 수치를 망친 것.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p dtp
export DRAFT_DTYPE=float32 DSET=gsm8k CMR=0 NSAMP=5 MAXNEW=256 \
       MAXC=4096 TARGET_MAX_SEQ=4096 KV_BLOCK_SIZE=1024 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_PREFILL_CHUNK=256 \
       TARGET_TP=4 TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_4k_l35
for D in 1 2 4; do
  out="dtp/draft_tp${D}.txt"
  echo "### DRAFT_TP=$D  $(date +%H:%M:%S)"
  DRAFT_TP=$D timeout 5400 python3 bench_dtp.py 2>&1 \
    | grep -aE '^SF |Traceback|Error|RuntimeError|RBLNCompileError' | head -3 > "$out"
  head -c 500 "$out"; echo
done
echo DTP_DONE

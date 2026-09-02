#!/usr/bin/env bash
# INIT_INTERNAL 원인 판별: TP 런타임 2개가 불가한가, 디바이스가 겹치면 안 되는가.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p dtp
export DRAFT_DTYPE=float32 DSET=gsm8k CMR=0 NSAMP=5 MAXNEW=256 \
       MAXC=4096 TARGET_MAX_SEQ=4096 KV_BLOCK_SIZE=1024 \
       DEV=0 TARGET_PREFILL_CHUNK=256
G1=/home/work/npu_work/dflash_work/fused_17_256_l35
G2=/home/work/npu_work/dflash_work/fused_tp2_l35
run () {  # tag graph TTP DT DTP DD
  echo "### $1  타깃TP=$3 카드$4~ / 드래프터TP=$5 카드$6~   $(date +%H:%M:%S)"
  TARGET_GRAPH_DIR=$2 TARGET_TP=$3 DEV_TARGET=$4 DRAFT_TP=$5 DEV_DRAFT=$6 \
    timeout 5400 python3 bench_dtp.py 2>&1 \
    | grep -aE '^SF |RuntimeError|RBLNCompileError|Error' | head -2 > "dtp/$1.txt"
  head -c 500 "dtp/$1.txt"; echo
}
# X: 타깃 TP1 카드0 / 드래프터 TP2 카드2,3  -> 완전 분리, TP 그룹 1 개
run X_t1d2 $G1 1 0 2 2
# Y: 타깃 TP2 카드0,1 / 드래프터 TP2 카드2,3 -> 완전 분리, TP 그룹 2 개
run Y_t2d2 $G2 2 0 2 2
echo DTP2_DONE

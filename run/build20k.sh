#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
rm -rf fused_tp4_16k
export KEEP=2,10,18,26,34,35 TP=4 CHUNKS=17,256 MAXC=20480
export OUTDIR=/home/work/npu_work/dflash_work/fused_tp4_20k
python3 fused_lmhead_1_256.py

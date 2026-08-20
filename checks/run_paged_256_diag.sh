#!/usr/bin/env bash
set -euo pipefail

cd /home/work/npu_work/dflash_work

export DRAFT_DTYPE=float32
export PROMPT_LEN=1024
export NATURAL_LONG=1
export MAXNEW=1
export NSAMP=1
export MAXC=4096
export KV_BLOCK_SIZE=256
export DEV=0
export DEV_TARGET=0
export DEV_DRAFT=0
export TARGET_TP=1
export TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_17_256
export DRAFT_DIAG=1

python3 bench_sf_fused_diag.py

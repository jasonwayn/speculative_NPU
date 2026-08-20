#!/usr/bin/env bash
set -euo pipefail

cd /home/work/npu_work/dflash_work

for prompt_length in 1008 1023 1025 1040; do
    echo "CONTEXT_SWEEP_BEGIN length=${prompt_length}"
    DRAFT_DTYPE=float32 \
    PROMPT_LEN="${prompt_length}" \
    NATURAL_LONG=1 \
    MAXNEW=1 \
    NSAMP=1 \
    MAXC=4096 \
    DEV=0 \
    DEV_TARGET=0 \
    DEV_DRAFT=0 \
    TARGET_TP=1 \
    TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_17_256 \
    DRAFT_DIAG=1 \
    python3 bench_sf_fused_diag.py
    echo "CONTEXT_SWEEP_END length=${prompt_length}"
done

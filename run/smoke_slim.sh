#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
export CMR=0 NSAMP=3 MAXNEW=256 MAXC=4096 DEV=0 NTHREADS=2 DSET=gsm8k
timeout 1800 python3 bench_sf_slim.py 2>&1 | grep -aE "dual graphs|COMPILED|^SF |Traceback|Error" | head -8
echo SMOKE_DONE

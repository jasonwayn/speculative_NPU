#!/bin/bash
cd /home/work/npu_work/dflash_work || exit 1
pkill -f "bench[2].py" 2>/dev/null
pkill -f "pwr_" 2>/dev/null
sleep 3
rm -f y4.log
export NCARD=4 NSAMP=80 DS=gsm8k TAG=y4 BUCKETS=2048 STAGGER=3 NTHREADS=2 PREC=float16
export TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64
setsid nohup bash multi.sh > y4.log 2>&1 < /dev/null &
disown
echo LAUNCHED

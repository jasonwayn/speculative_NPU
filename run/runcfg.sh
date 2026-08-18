#!/bin/bash
cd /home/work/npu_work/dflash_work || exit 1
pkill -f "bench[2].py" 2>/dev/null
sleep 2
LOG=${LOG:-out.log}
setsid nohup env DEV=${DEV:-0} NSAMP=${NSAMP:-60} MAXNEW=2048 BUCKETS=${BUCKETS:-2048} \
  NTHREADS=2 PREC=${PREC:-float16} SERIAL_LOAD=0 \
  TGTDIR=${TGTDIR:-/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64} \
  DSETS=gsm8k python3 bench2.py > "$LOG" 2>&1 < /dev/null &
disown
echo "LAUNCHED $LOG"

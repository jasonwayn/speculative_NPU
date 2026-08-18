#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
if ps -eo cmd | grep -qE "[b]ench2|[b]ench_sf"; then echo REFUSE; exit 1; fi
mkdir -p slim_results
export CMR=0 NSAMP=20 MAXNEW=2048 MAXC=4096 DEV=0 NTHREADS=2
for DS in gsm8k math500 humaneval mbpp mt-bench; do
  out=slim_results/${DS}.txt
  if [ -s "$out" ] && grep -q "^SF " "$out"; then echo "SKIP $DS"; continue; fi
  echo "### $DS  $(date +%H:%M:%S)"
  DSET=$DS timeout 7200 python3 bench_sf_slim.py 2>&1 | grep -aE "^SF |Traceback|Error" | head -3 > "$out"
  cat "$out"
  sleep 5
done
echo SLIMFULL_DONE

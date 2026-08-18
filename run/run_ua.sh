#!/usr/bin/env bash
# 비정렬 재개판으로 논문 워크로드. 단독 실행.
cd /home/work/npu_work/dflash_work || exit 1
if ps -eo cmd | grep -qE "[b]ench2|[b]ench_sf"; then echo "REFUSE: bench running"; exit 1; fi
mkdir -p ua_results
export CMR=0 NSAMP=20 MAXNEW=2048 MAXC=4096 DEV=0 NTHREADS=2
for DS in gsm8k math500; do
  out=ua_results/${DS}.txt
  if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP $DS"; continue; fi
  echo "### $DS  $(date +%H:%M:%S)"
  DSET=$DS timeout 7200 python3 bench_sf_ua.py 2>&1 \
    | grep -aE '^SF |Traceback|Error' | head -3 > "$out"
  cat "$out"
  sleep 5
done
echo UA_DONE

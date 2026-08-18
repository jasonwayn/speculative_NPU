#!/usr/bin/env bash
# 상태 유지 드래프터로 논문 워크로드 5종. 단독 실행 (다른 카드에서도 아무것도 돌리지 말 것).
cd /home/work/npu_work/dflash_work || exit 1
if [ "$(ps -eo cmd | grep -cE '[b]ench2|[b]ench_sf')" != "0" ]; then echo "REFUSE: bench running"; exit 1; fi
mkdir -p sfp_results
export CMR=0 NSAMP=20 MAXNEW=2048 MAXC=4096 DEV=0 NTHREADS=2
for DS in gsm8k math500 humaneval mbpp mt-bench; do
  out=sfp_results/${DS}.txt
  if [ -s "$out" ] && grep -q '^SF ' "$out"; then echo "SKIP(done) $DS"; continue; fi
  echo "### $DS  $(date +%H:%M:%S)"
  DSET=$DS timeout 7200 python3 bench_sf_paper.py 2>&1 \
    | grep -aE '^SF |Traceback|Error|RuntimeError' | head -3 > "$out"
  cat "$out"
  sleep 5
done
echo SFP_DONE

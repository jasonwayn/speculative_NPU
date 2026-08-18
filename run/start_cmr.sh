#!/usr/bin/env bash
# 분리 실행 전용 래퍼. ssh 한 줄로 호출하고 즉시 반환한다.
cd /home/work/npu_work/dflash_work || exit 1
if ps -eo cmd | grep -q "[b]ench_cmr"; then
  echo "REFUSE: bench_cmr already running"; exit 1
fi
rm -f runcmr.log
rm -rf cmr_results
setsid bash runcmr.sh > runcmr.log 2>&1 < /dev/null &
echo "started pid=$!"

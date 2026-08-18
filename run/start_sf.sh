#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
if ps -eo cmd | grep -q "[b]ench_sf"; then echo "REFUSE: already running"; exit 1; fi
rm -f runsf.log
setsid bash runsf.sh > runsf.log 2>&1 < /dev/null &
echo "started pid=$!"

#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
if ps -eo cmd | grep -qE "[b]ench2|[b]ench_sf"; then echo "REFUSE: already running"; exit 1; fi
rm -f sfp.log
setsid bash run_sf_paper.sh > sfp.log 2>&1 < /dev/null &
echo "started pid=$!"

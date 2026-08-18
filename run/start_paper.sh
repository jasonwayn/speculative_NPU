#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
if ps -eo cmd | grep -q "[b]ench2.py"; then echo "REFUSE: bench2 already running"; exit 1; fi
rm -f paper.log
setsid bash run_paper.sh > paper.log 2>&1 < /dev/null &
echo "started pid=$!"

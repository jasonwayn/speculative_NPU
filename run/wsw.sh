#!/bin/bash
L=/home/work/npu_work/dflash_work/sweep.log
for i in $(seq 1 55); do
  grep -q "SWEEP_ALL" "$L" 2>/dev/null && break
  pgrep -f "swee[p].py" >/dev/null || break
  sleep 30
done
grep -E "idle_W|^=====|^SWEEP " "$L"
echo "---"; pgrep -f "swee[p].py" >/dev/null && echo RUNNING || echo DONE

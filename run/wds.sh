#!/bin/bash
L=/home/work/npu_work/dflash_work/ds5.log
for i in $(seq 1 50); do
  grep -q "BENCH2_ALL" "$L" 2>/dev/null && break
  pgrep -f "bench[2].py" >/dev/null || break
  sleep 30
done
grep -E "^BENCH2 |idle_W" "$L" | cut -c1-400
echo "---"; pgrep -f "bench[2].py" >/dev/null && echo RUNNING || echo DONE

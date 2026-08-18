#!/bin/bash
L=/home/work/npu_work/dflash_work/math500_long.log
for i in $(seq 1 40); do
  grep -q "BENCH2_ALL" "$L" 2>/dev/null && break
  pgrep -f "bench[2].py" >/dev/null || break
  sleep 30
done
grep -E "compiled|idle_W|^BENCH2 " "$L" | cut -c1-420
echo "---"; pgrep -f "bench[2].py" >/dev/null && echo RUNNING || echo DONE

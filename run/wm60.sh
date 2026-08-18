#!/bin/bash
L=/home/work/npu_work/dflash_work/math500_60.log
for i in $(seq 1 45); do
  grep -q "BENCH2_ALL" "$L" 2>/dev/null && break
  pgrep -f "bench[2].py" >/dev/null || break
  sleep 30
done
grep -E "^BENCH2 " "$L" | cut -c1-420
echo "---"; pgrep -f "bench[2].py" >/dev/null && echo RUNNING || echo DONE
tail -3 "$L" | grep -E "Error|error" | head -2

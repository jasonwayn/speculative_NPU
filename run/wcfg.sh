#!/bin/bash
L=/home/work/npu_work/dflash_work/${1:-out.log}
for i in $(seq 1 40); do
  grep -q "BENCH2_ALL" "$L" 2>/dev/null && break
  pgrep -f "bench[2].py" >/dev/null || break
  sleep 20
done
grep -E "^BENCH2 " "$L" | cut -c1-470
exit 0

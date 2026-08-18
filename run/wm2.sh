#!/bin/bash
for i in $(seq 1 45); do
  pgrep -f "mult[i].sh" >/dev/null || break
  sleep 20
done
for L in /home/work/npu_work/dflash_work/s1.log /home/work/npu_work/dflash_work/s4.log; do
  [ -f "$L" ] || continue
  echo "=== $L"; grep -E "MULTI_WALL|^BENCH2 |^PWR4 " "$L" | cut -c1-330
done
echo "---"; pgrep -f "mult[i].sh" >/dev/null && echo RUNNING || echo DONE
exit 0

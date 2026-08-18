#!/bin/bash
L=/home/work/npu_work/dflash_work/confirm.log
for i in $(seq 1 60); do
  grep -q "DONE #####" "$L" 2>/dev/null && break
  pgrep -f "confir[m].sh" >/dev/null || break
  sleep 30
done
grep -E "#####|^BENCH2 " "$L" | cut -c1-400
echo "---"; pgrep -f "confir[m].sh" >/dev/null && echo RUNNING || echo DONE
exit 0

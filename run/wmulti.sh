#!/bin/bash
LOG=${1:-/home/work/npu_work/dflash_work/multi1.log}
for i in $(seq 1 40); do
  pgrep -f "mult[i].sh" >/dev/null || break
  sleep 20
done
grep -E "MULTI_WALL|^BENCH2 " "$LOG" | cut -c1-330
echo "---"; pgrep -f "mult[i].sh" >/dev/null && echo RUNNING || echo DONE
for f in /home/work/npu_work/dflash_work/multi_d*.log; do
  E=$(grep -cE "Error|Traceback" "$f" 2>/dev/null)
  [ "$E" != "0" ] && echo "ERR in $f: $(grep -E 'Error' "$f" | tail -1 | cut -c1-120)"
done
exit 0

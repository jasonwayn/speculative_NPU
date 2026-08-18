#!/bin/bash
L=/home/work/npu_work/dflash_work/tau2048.log
for i in $(seq 1 30); do
  grep -q "NPUFULL" "$L" 2>/dev/null && break
  pgrep -f "npu_ful[l]" >/dev/null || break
  sleep 30
done
grep -E "^\[|NPUFULL|Error" "$L" | tail -12
echo "---"; pgrep -f "npu_ful[l]" >/dev/null && echo RUNNING || echo DONE

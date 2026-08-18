#!/bin/bash
L=/home/work/npu_work/dflash_work/ab.log
for i in $(seq 1 16); do
  grep -q "DONE ##########" "$L" 2>/dev/null && break
  pgrep -f "run_a[b].sh" >/dev/null || break
  sleep 30
done
grep -E "##########|^\[|NPU_LOOP_B|Error" "$L" | tail -10
echo "---"
pgrep -f "run_a[b].sh" >/dev/null && echo RUNNING || echo DONE

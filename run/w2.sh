#!/bin/bash
L=/home/work/npu_work/dflash_work/loopfix.log
for i in $(seq 1 18); do
  grep -q "NPU_LOOP_B" "$L" 2>/dev/null && break
  pgrep -f "npu_loop_[b]" >/dev/null || break
  sleep 30
done
grep -E "compiled|^\[|NPU_LOOP_B|Error" "$L" | tail -8
echo "---"; pgrep -f "npu_loop_[b]" >/dev/null && echo RUNNING || echo DONE

#!/bin/bash
for i in $(seq 1 40); do
  grep -q "NPU_LOOP_B" /home/work/npu_work/dflash_work/loopb.log 2>/dev/null && break
  pgrep -f "npu_loop_[b]" >/dev/null || break
  sleep 30
done
echo "=== 60샘플 CPU 레퍼런스 ==="
grep -E "Acceptance length|speedup|throughput" /home/work/npu_work/dflash_work/bench60.log 2>/dev/null | tail -5
echo "=== (b) NPU 루프 ==="
pgrep -f "npu_loop_[b]" >/dev/null && echo STILL_RUNNING || echo DONE
grep -E "^\[|NPU_LOOP_B" /home/work/npu_work/dflash_work/loopb.log | tail -8

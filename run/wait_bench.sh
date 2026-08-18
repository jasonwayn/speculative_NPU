#!/bin/bash
for i in $(seq 1 40); do
  pgrep -f "dflash.benchmar[k]" >/dev/null || break
  sleep 20
done
echo "=== 완료 여부 ==="
pgrep -f "dflash.benchmar[k]" >/dev/null && echo RUNNING || echo DONE
echo "=== 결과 ==="
sed -e 's/\x1b\[[0-9;]*m//g' /home/work/npu_work/dflash_work/bench.log | grep -viE "it/s\]|examples/s\]" | tail -25

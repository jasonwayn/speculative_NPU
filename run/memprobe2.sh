#!/usr/bin/env bash
# RSS 시계열 -> 로딩 피크인지 정상상태 점유인지 판정
cd /home/work/npu_work/dflash_work
echo "== 컨테이너에서 메모리 쓰는 프로세스 =="
ps -eo rss,comm --sort=-rss | head -6
echo "== 시계열 (초, RSS MB) =="
NSAMP=3 MAXNEW=128 B=16 DEV=0 NTHREADS=2 PREC=float16 BUCKETS=256,512 DSETS=gsm8k \
  TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64 \
  python3 bench2.py > /tmp/mp2.log 2>&1 &
P=$!; t=0
while kill -0 $P 2>/dev/null; do
  R=$(awk '/VmRSS/{print $2}' /proc/$P/status 2>/dev/null)
  [ -n "$R" ] && echo "  t=${t}s  $((R/1024))MB"
  t=$((t+3)); sleep 3
done
wait $P
grep -E "^BENCH2 " /tmp/mp2.log | head -1 | cut -c1-140
echo MP2_DONE

#!/usr/bin/env bash
# 워커 1개의 호스트 RSS 피크 측정 -> 4카드 가능 여부 판정
cd /home/work/npu_work/dflash_work
echo "== container limits =="
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
echo "cur: $(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)"
echo "MemAvailable: $(grep MemAvailable /proc/meminfo)"
echo "cores: $(nproc)"
echo "== single worker peak RSS =="
NSAMP=2 MAXNEW=64 B=16 DEV=0 NTHREADS=2 PREC=float16 BUCKETS=256 DSETS=gsm8k \
  TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64 \
  python3 bench2.py > /tmp/mp.log 2>&1 &
P=$!
PEAK=0
while kill -0 $P 2>/dev/null; do
  R=$(awk '/VmRSS/{print $2}' /proc/$P/status 2>/dev/null)
  T=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
  [ -n "$R" ] && [ "$R" -gt "$PEAK" ] && PEAK=$R
  CG=$T
  sleep 1
done
wait $P
echo "peak_worker_RSS_MB $((PEAK/1024))"
echo "cgroup_at_end_MB $((CG/1048576))"
grep -E "^BENCH2 " /tmp/mp.log | head -1
echo MEMPROBE_DONE

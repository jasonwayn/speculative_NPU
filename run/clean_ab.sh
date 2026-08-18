#!/usr/bin/env bash
# 단독 조건 A/B. 다른 카드/작업 없이 카드 1에서만, 교대 3회씩.
# 호스트를 공유하므로 병렬 측정은 불가 -- 반드시 순차·단독으로 돌린다.
cd /home/work/npu_work/dflash_work || exit 1
if [ "$(ps -eo cmd | grep -c '[b]ench2')" != "0" ]; then echo "REFUSE: bench2 running"; exit 1; fi
export DEV=1 NTHREADS=2 PREC=float16 B=16 NSAMP=5 MAXNEW=512 \
       DSETS=gsm8k BUCKETS=256,512,1024 \
       TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64
for i in 1 2 3; do
  for V in bench2.py bench2_opt.py; do
    L=$(timeout 2400 python3 "$V" 2>&1 | grep -a '^BENCH2 {' | head -1)
    if [ -z "$L" ]; then echo "$V run$i NO_RESULT"; continue; fi
    echo "$L" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read().split(None, 1)[1])
print('%-14s run%s  tau=%-6s tok_s=%-7s wall=%-6s draft_ms=%-6s verify_ms=%-6s lmh_s=%s'
      % ('$V', '$i', d['tau'], d['tok_s'], d['wall_s'], d['draft_ms'], d['verify_ms'], d['lmh_s']))
"
    sleep 5
  done
done
echo CLEAN_AB_DONE

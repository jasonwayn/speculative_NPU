#!/usr/bin/env bash
# eager vs flash_attn 타깃 비교. 단독 실행 (다른 작업 금지).
# 짧은 과제(gsm8k)와 긴 과제(math500) 둘 다 -- 플래시는 컨텍스트가 길수록 유리해야 한다.
cd /home/work/npu_work/dflash_work || exit 1
if [ "$(ps -eo cmd | grep -c '[b]ench2')" != "0" ]; then echo "REFUSE: bench2 running"; exit 1; fi
E=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-16k
F=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-16k-flash
export DEV=1 NTHREADS=2 PREC=float16 B=16 NSAMP=5 MAXNEW=1024 \
       BUCKETS=256,512,1024,2048
for DS in gsm8k math500; do
  for TAG in eager flash; do
    if [ "$TAG" = "eager" ]; then T=$E; else T=$F; fi
    L=$(DSETS=$DS TGTDIR=$T timeout 3600 python3 bench2.py 2>&1 | grep -a '^BENCH2 {' | head -1)
    if [ -z "$L" ]; then echo "$DS $TAG NO_RESULT"; continue; fi
    echo "$L" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read().split(None, 1)[1])
print('%-9s %-6s tau=%-6s tok_s=%-7s verify_ms=%-6s draft_ms=%-5s wall=%s'
      % ('$DS', '$TAG', d['tau'], d['tok_s'], d['verify_ms'], d['draft_ms'], d['wall_s']))
"
    sleep 5
  done
done
echo FLASH_AB_DONE

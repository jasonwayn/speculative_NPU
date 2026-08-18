#!/usr/bin/env bash
# bench2.py(원본) vs bench2_opt.py(.float() 제거) A/B. 카드 1, 동일 조건, 교대 2회.
cd /home/work/npu_work/dflash_work || exit 1
export DEV=1 NTHREADS=2 PREC=float16 B=16 NSAMP=3 MAXNEW=256 \
       DSETS=gsm8k BUCKETS=256,512,1024 \
       TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64
for V in bench2.py bench2_opt.py bench2.py bench2_opt.py; do
  L=$(timeout 1800 python3 "$V" 2>&1 | grep -a '^BENCH2 {' | head -1)
  if [ -z "$L" ]; then echo "$V  NO_RESULT"; continue; fi
  echo "$L" | sed "s|^BENCH2 |$V |"
done
echo AB_DONE

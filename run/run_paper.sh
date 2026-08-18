#!/usr/bin/env bash
# DFlash 논문 워크로드 전체를 NPU 1장(카드 0)에서. prefill + decode 포함.
# 데이터셋별로 결과 파일을 따로 써서 중간에 끊겨도 보존.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p paper_results
export B=16 DEV=0 NTHREADS=2 PREC=float16 NSAMP=20 MAXNEW=2048 \
       BUCKETS=256,512,1024,2048,3072 \
       TGTDIR=/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64
for DS in gsm8k math500 humaneval mbpp mt-bench; do
  out=paper_results/${DS}.txt
  if [ -s "$out" ] && grep -q "^BENCH2 " "$out"; then echo "SKIP(done) $DS"; continue; fi
  echo "### $DS  $(date +%H:%M:%S)"
  DSETS=$DS timeout 7200 python3 bench2.py 2>&1 \
    | grep -aE "^BENCH2 |Traceback|Error|RuntimeError" | head -3 > "$out"
  cat "$out"
  sleep 5
done
echo PAPER_DONE

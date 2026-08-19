#!/usr/bin/env bash
# GPU 가 비어 있는 지금 바로 측정. NPU 와 동일 조건 (NSAMP=20, MAXNEW=2048, B=16, 배치 1).
cd "$HOME/dflash_bench" || exit 1
OUT="$HOME/dflash_bench/paper_gpu"
mkdir -p "$OUT"

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
if [ "$apps" -ne 0 ]; then echo "ABORT: 다른 프로세스 $apps 개 실행 중"; echo GPUNOW_DONE; exit 1; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NSAMP=20 MAXNEW=2048

echo "=== dflash 5종  $(date +%H:%M:%S)"
DSETS=gsm8k,math500,humaneval,mbpp,mt-bench MODE=dflash \
  python3 gpu_bench2.py 2>&1 | grep -aE '^GPU2 |^idle_W|Traceback|Error' > "$OUT/dflash.txt"
cat "$OUT/dflash.txt"

echo "=== AR 기준선 gsm8k  $(date +%H:%M:%S)"
DSETS=gsm8k MODE=ar \
  python3 gpu_bench2.py 2>&1 | grep -aE '^GPU2 |^idle_W|Traceback|Error' > "$OUT/ar_gsm8k.txt"
cat "$OUT/ar_gsm8k.txt"

echo GPUNOW_DONE

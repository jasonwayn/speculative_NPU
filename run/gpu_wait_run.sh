#!/usr/bin/env bash
# GPU 가 완전히 비면 DFlash 논문 워크로드 5종을 NPU 와 동일 조건으로 측정한다.
#
# 다른 사람 학습 작업과 SM 을 나눠 쓰면 tok/s 는 버려야 하는 수치가 된다
# (이전 세션에서 그렇게 측정한 값 전부 폐기했음). 그래서 유휴 상태를 확인하고 시작한다.
#
# 유휴 판정: 다른 compute app 0개 && util < 10%  가 60초 간격으로 5회 연속
# 최대 대기 6시간. 그 안에 안 비면 아무것도 안 하고 종료한다.
cd "$HOME/dflash_bench" || exit 1
OUT="$HOME/dflash_bench/paper_gpu"
mkdir -p "$OUT"

idle_ok() {
  local apps util
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . )
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  [ "$apps" -eq 0 ] && [ "$util" -lt 10 ]
}

streak=0
for i in $(seq 1 360); do
  if idle_ok; then streak=$((streak+1)); else streak=0; fi
  if [ "$streak" -ge 5 ]; then break; fi
  sleep 60
done

if [ "$streak" -lt 5 ]; then
  echo "GPU_BUSY_GAVEUP  6시간 동안 유휴 구간 없음"
  echo GPUWAIT_DONE
  exit 0
fi

echo "GPU_IDLE_START $(date +%H:%M:%S)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NSAMP=20 MAXNEW=2048

# DFlash — NPU 와 같은 5종
DSETS=gsm8k,math500,humaneval,mbpp,mt-bench MODE=dflash \
  python3 gpu_bench2.py 2>&1 | grep -aE '^GPU2 |^idle_W|Traceback|Error' > "$OUT/dflash.txt"
cat "$OUT/dflash.txt"

# AR 기준선 — gsm8k 만 (speedup 배수 확인용)
DSETS=gsm8k MODE=ar \
  python3 gpu_bench2.py 2>&1 | grep -aE '^GPU2 |^idle_W|Traceback|Error' > "$OUT/ar_gsm8k.txt"
cat "$OUT/ar_gsm8k.txt"

echo GPUWAIT_DONE

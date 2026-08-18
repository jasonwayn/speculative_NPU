#!/usr/bin/env bash
# stock vs CMR. 설정별 결과를 개별 파일로 -> 중간에 끊겨도 부분 결과 보존.
cd /home/work/npu_work/dflash_work
mkdir -p cmr_results
export NSAMP=3 MAXNEW=256 B=16 DEV=0 NTHREADS=8 PREC=float16
run () {
  out=cmr_results/len${1}_cmr${2}.txt
  if [ -s "$out" ] && grep -q CMRB "$out"; then echo "SKIP(done) len=$1 cmr=$2"; return; fi
  echo "### len=$1 cmr=$2 buckets=$3 start=$(date +%H:%M:%S)"
  CMR=$2 INLEN=$1 BUCKETS=$3 timeout 5400 python3 bench_cmr.py 2>&1 \
    | grep -E "^CMRB |RuntimeError|Traceback|SKIP" | head -3 > "$out"
  cat "$out"
  sleep 5
}
run 4096  0 8192
run 4096  1 2048
run 8192  0 16384
run 8192  1 2048
run 16000 0 16384
run 16000 1 2048
echo RUNCMR_DONE

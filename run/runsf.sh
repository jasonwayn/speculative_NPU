#!/usr/bin/env bash
# 상태 유지 드래프트 위에서 stock vs CMR. 설정별 결과 파일 분리.
cd /home/work/npu_work/dflash_work
mkdir -p sf_results
export NSAMP=3 MAXNEW=256 DEV=0 NTHREADS=8 MAXC=16384
run () {
  out=sf_results/len${1}_cmr${2}.txt
  if [ -s "$out" ] && grep -q "^SF " "$out"; then echo "SKIP(done) len=$1 cmr=$2"; return; fi
  echo "### len=$1 cmr=$2  $(date +%H:%M:%S)"
  CMR=$2 INLEN=$1 timeout 5400 python3 bench_sf.py 2>&1 \
    | grep -aE "^SF |RuntimeError|Traceback|Error occurred" | head -3 > "$out"
  cat "$out"
  sleep 5
}
run 4096  0
run 4096  1
run 8192  0
run 8192  1
run 16000 0
run 16000 1
echo RUNSF_DONE

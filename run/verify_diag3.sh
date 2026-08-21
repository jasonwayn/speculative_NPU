#!/usr/bin/env bash
# 1) 드리프트 배제용 반복  2) 스레드 수 곡선  3) CMR 끔 기준선의 spin 영향
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p verify_diag
export DRAFT_DTYPE=float32 MAXNEW=128 NSAMP=2 MAXC=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=4 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp4_20k \
       CMR_BUDGET=1024 CMR_CHUNK=32 CMR_TOPK=32 CMR_EVERY=4 NTHREADS=2 \
       CORPUS=pg19 CORPUS_LEN=4096
run () {
  out="verify_diag/$1.txt"
  echo "### $1  $(date +%H:%M:%S)  load=$(cut -d' ' -f1 /proc/loadavg)"
  env "${@:2}" timeout 5400 python3 bench_sf_diag3.py 2>&1 \
    | grep -aE '^SF |Traceback|Error' | head -3 > "$out"
  head -c 800 "$out"; echo
}
run a2 CMR=0 NTHREADS=2
run a3 CMR=0 NTHREADS=2 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
run a8 CMR=0 NTHREADS=8
run t1 CMR=1 NTHREADS=1
run t2 CMR=1 NTHREADS=2
run t3 CMR=1 NTHREADS=3
run t4 CMR=1 NTHREADS=4
run t6 CMR=1 NTHREADS=6
run t8 CMR=1 NTHREADS=8
run t2p CMR=1 NTHREADS=2 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
echo VDIAG3_DONE

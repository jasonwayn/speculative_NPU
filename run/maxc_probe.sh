#!/usr/bin/env bash
# RBLN paged attention 이 실제 컨텍스트만 계산하는가, MAXC 전체를 계산하는가.
# NPUsper 의 controlled unrolling 이 겨냥한 낭비가 우리에게 존재하는지 가른다.
# 같은 짧은 워크로드(gsm8k, 컨텍스트 ~450)를 MAXC 만 다른 두 그래프에서 돌린다.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p maxc_probe
G4=/home/work/npu_work/dflash_work/fused_17_256_l35        # MAXC 4096 (기존)
G20=/home/work/npu_work/dflash_work/fused_tp1_20k_l35      # MAXC 20480 (신규)

if [ ! -f "$G20/prefill_256.rbln" ]; then
  echo "### build TP1 MAXC=20480  $(date +%H:%M:%S)"
  rm -rf "$G20"
  KEEP=2,10,18,26,34,35 TP=1 CHUNKS=17,256 MAXC=20480 OUTDIR=$G20 \
    timeout 5400 python3 fused_lmhead_1_256.py 2>&1 | tail -150 > maxc_build.log
  grep -aE "COMPILE_OK|COMPILE_FAIL|FUSED_COMPILE_DONE" maxc_build.log | head -4
fi

export DRAFT_DTYPE=float32 DSET=gsm8k CMR=0 NSAMP=5 MAXNEW=256 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=1 TARGET_PREFILL_CHUNK=256 NTHREADS=2
run () {
  echo "### MAXC=$1  $(date +%H:%M:%S)"
  MAXC=$1 TARGET_MAX_SEQ=$1 TARGET_GRAPH_DIR=$2 timeout 5400 python3 bench_sf_corpus.py 2>&1 \
    | grep -aE '^SF |Traceback|Error' | head -2 > "maxc_probe/m$1.txt"
  head -c 430 "maxc_probe/m$1.txt"; echo
}
run 4096  $G4
run 20480 $G20
echo MAXCPROBE_DONE

#!/usr/bin/env bash
# 무인 실행. 세 프로브가 순서대로 돌고 결과만 남긴다. 중간 판단 불필요.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p night
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 NTHREADS=2

echo "########## ① 드래프터 TP 프로브   $(date +%H:%M:%S)"
MAXC=4096 TPLIST=2,4 timeout 3600 python3 probe_draft_tp.py 2>&1 | tail -60 > night/draft_tp.log
grep -aE "params:|TP 인자|COMPILE_OK|COMPILE_FAIL|미지원|DRAFTTP_DONE" night/draft_tp.log | head -12

echo "########## ② lm_head 4등분 프로브  $(date +%H:%M:%S)"
NSHARD=4 timeout 3600 python3 probe_lmh_tp.py 2>&1 | tail -40 > night/lmh_tp.log
grep -aE "vocab=|compiled|FULL|SHARDED|payload|LMHTP_DONE|Error" night/lmh_tp.log | head -12

echo "########## ③ TP4 타깃 MAXC 4096 빌드  $(date +%H:%M:%S)"
G4=/home/work/npu_work/dflash_work/fused_tp4_4k_l35
if [ ! -f "$G4/prefill_256.rbln" ]; then
  rm -rf "$G4"
  KEEP=2,10,18,26,34,35 TP=4 CHUNKS=17,256 MAXC=4096 OUTDIR=$G4 \
    timeout 5400 python3 fused_lmhead_1_256.py 2>&1 | tail -150 > night/tp4_build.log
  grep -aE "COMPILE_OK|COMPILE_FAIL|FUSED_COMPILE_DONE" night/tp4_build.log | head -4
fi

echo "########## ④ TP4 기준선 측정  $(date +%H:%M:%S)"
export DRAFT_DTYPE=float32 DSET=gsm8k CMR=0 NSAMP=5 MAXNEW=256 \
       MAXC=4096 TARGET_MAX_SEQ=4096 KV_BLOCK_SIZE=1024 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_PREFILL_CHUNK=256
TARGET_TP=4 TARGET_GRAPH_DIR=$G4 timeout 5400 python3 bench_sf_corpus.py 2>&1 \
  | grep -aE '^SF |Traceback|Error' | head -2 > night/tp4_gsm8k.txt
head -c 470 night/tp4_gsm8k.txt; echo

echo "########## NIGHT_DONE  $(date +%H:%M:%S)"

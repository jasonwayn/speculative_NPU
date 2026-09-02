#!/usr/bin/env bash
# TP1 vs TP2(드래프터 공유) vs TP2(드래프터 전용 카드) — 3장 구성 검증.
# B vs C 가 드래프터 배치 효과만 분리한다 (타깃 분산은 동일).
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p tp_split
G1=/home/work/npu_work/dflash_work/fused_17_256_l35
G2=/home/work/npu_work/dflash_work/fused_tp2_l35

if [ ! -f "$G2/prefill_256.rbln" ]; then
  echo "### build TP2 MAXC=4096  $(date +%H:%M:%S)"
  rm -rf "$G2"
  KEEP=2,10,18,26,34,35 TP=2 CHUNKS=17,256 MAXC=4096 OUTDIR=$G2 \
    timeout 5400 python3 fused_lmhead_1_256.py 2>&1 | tail -150 > tp2_build.log
  grep -aE "COMPILE_OK|COMPILE_FAIL|FUSED_COMPILE_DONE" tp2_build.log | head -4
fi

export DRAFT_DTYPE=float32 DSET=gsm8k CMR=0 NSAMP=5 MAXNEW=256 \
       MAXC=4096 TARGET_MAX_SEQ=4096 KV_BLOCK_SIZE=1024 \
       DEV=0 TARGET_PREFILL_CHUNK=256 NTHREADS=2
run () {   # $1=tag $2=graph $3=TP $4=DEV_TARGET $5=DEV_DRAFT
  echo "### $1  타깃TP=$3 카드$4~  드래프터 카드$5   $(date +%H:%M:%S)"
  TARGET_GRAPH_DIR=$2 TARGET_TP=$3 DEV_TARGET=$4 DEV_DRAFT=$5 \
    timeout 5400 python3 bench_sf_corpus.py 2>&1 \
    | grep -aE '^SF |Traceback|Error|RuntimeError' | head -2 > "tp_split/$1.txt"
  head -c 470 "tp_split/$1.txt"; echo
}
run A_tp1        $G1 1 0 0
run B_tp2_shared $G2 2 0 0
run C_tp2_split  $G2 2 0 2
echo TPSPLIT_DONE

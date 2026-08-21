#!/usr/bin/env bash
# A' 검증용 TP1 그래프. 레이어 35 점수를 그래프 출력으로 뺀다.
# TP1 이면 32 헤드가 한 카드에 있어 헤드 평균이 로컬 연산이다 (all-reduce 불필요).
cd /home/work/npu_work/dflash_work || exit 1
MAXC=${MAXC:-20480}
OUT=/home/work/npu_work/dflash_work/fused_tp1_score_${MAXC}
rm -rf "$OUT"
export KEEP=2,10,18,26,34,35 TP=1 CHUNKS=17,256 MAXC=$MAXC \
       SCORE_LAYER=35 SCORE_CHUNK=32 OUTDIR=$OUT
echo "### build TP1 MAXC=$MAXC -> $OUT  $(date +%H:%M:%S)"
python3 fused_score_1_256.py
echo BUILD_SCORE_DONE

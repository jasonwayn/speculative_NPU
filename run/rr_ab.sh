#!/usr/bin/env bash
# 3자 비교. 정확도(DRAFT_DIAG) 한 번 + tau/처리량 한 번. 교대로 두 바퀴 돌려 노이즈 확인.
cd /home/work/npu_work/dflash_work || exit 1
export DRAFT_DTYPE=float32 PROMPT_LEN=1024 NATURAL_LONG=1 MAXC=4096 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=1 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_17_256 NTHREADS=2

echo "===== 정확도 (1024 컨텍스트, 1라운드) ====="
for v in bench_sf_fused_diag.py bench_sf_rope.py bench_sf_rr.py; do
  echo "--- $v"
  MAXNEW=1 WARMNEW=0 NSAMP=1 DRAFT_DIAG=1 timeout 2400 python3 "$v" 2>&1 \
    | grep -aoE 'cos_min=[0-9.]+ cos_mean=[0-9.]+ rel_max=[0-9.e-]+ token_matches=[0-9]+/[0-9]+' | head -1
done

echo "===== 처리량 (MAXNEW=256, NSAMP=3, 교대 2바퀴) ====="
for round in 1 2; do
  for v in bench_sf_fused_diag.py bench_sf_rope.py bench_sf_rr.py; do
    L=$(MAXNEW=256 NSAMP=3 timeout 3600 python3 "$v" 2>&1 | grep -a '^SF ' | head -1)
    echo "r${round} ${v} ${L}"
  done
done
echo RRAB_DONE

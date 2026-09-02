#!/usr/bin/env bash
# 드래프터 KV 캐시를 블록 단위로 쪼개면 짧은 컨텍스트 페널티가 줄어드는가.
# MAXC=20480 고정, KV_BLOCK_SIZE 만 바꾼다. 컨텍스트는 gsm8k 라 ~450 로 짧다.
# 드래프터 그래프는 벤치 시작 시 컴파일되므로 사전 빌드 불필요.
cd /home/work/npu_work/dflash_work || exit 1
mkdir -p blk_probe
export DRAFT_DTYPE=float32 DSET=gsm8k CMR=0 NSAMP=5 MAXNEW=256 \
       MAXC=20480 TARGET_MAX_SEQ=20480 \
       DEV=0 DEV_TARGET=0 DEV_DRAFT=0 TARGET_TP=1 TARGET_PREFILL_CHUNK=256 \
       TARGET_GRAPH_DIR=/home/work/npu_work/dflash_work/fused_tp1_20k_l35 NTHREADS=2
for BS in 20480 4096 1024 256; do
  out="blk_probe/bs${BS}.txt"
  echo "### KV_BLOCK_SIZE=$BS (blocks=$((20480/BS)))  $(date +%H:%M:%S)"
  KV_BLOCK_SIZE=$BS timeout 5400 python3 bench_sf_corpus.py 2>&1 \
    | grep -aE '^SF |Traceback|Error|RuntimeError' | head -2 > "$out"
  head -c 400 "$out"; echo
done
echo BLKPROBE_DONE

#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
# 앞 스윕이 끝나기를 기다렸다가 실패분(빈 파일/Traceback)만 지우고 재실행
while pgrep -f corpus_sweep.sh > /dev/null; do sleep 20; done
for f in corpus_results/*.txt; do
  if ! grep -q "^SF " "$f" 2>/dev/null; then echo "재실행 대상: $f"; rm -f "$f"; fi
done
export TARGET_MAX_SEQ=20480
bash corpus_sweep.sh

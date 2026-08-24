#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
for C in cumsum max_val argmax topk_val topk_idx kth_mask sort_idx; do
  out=$(timeout 600 python3 topk_one.py "$C" 2>&1)
  rc=$?
  line=$(echo "$out" | grep -a '^RESULT ')
  if [ -n "$line" ]; then echo "$line"
  elif echo "$out" | grep -qa "dumped core\|Segmentation"; then echo "RESULT $C  ***세그폴트(코어덤프)***"
  else echo "RESULT $C  종료코드=$rc  $(echo "$out" | grep -avE 'INFO|Computation graph|^\s*$' | tail -1 | cut -c1-120)"
  fi
done
echo TOPKALL_DONE

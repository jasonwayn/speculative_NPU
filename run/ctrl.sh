#!/bin/bash
cd /home/work/npu_work/dflash_work/dflash || exit 1
export PYTHONPATH=/home/work/npu_work/dflash_work/dflash:$PYTHONPATH
python3 -m dflash.benchmark --backend transformers \
  --model /home/work/npu_work/eagle_test/Qwen3-4B \
  --draft-model /home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k --max-samples 3 --max-new-tokens 96 --temperature 0

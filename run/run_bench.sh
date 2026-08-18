#!/bin/bash
cd /home/work/npu_work/dflash_work/dflash || exit 1
python3 -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2
export PYTHONPATH=/home/work/npu_work/dflash_work/dflash:$PYTHONPATH
python3 -m dflash.benchmark \
  --backend transformers \
  --model /home/work/npu_work/eagle_test/Qwen3-4B \
  --draft-model /home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 10 \
  --max-new-tokens 2048 \
  --temperature 0

#!/bin/bash
cd /home/work/npu_work/dflash_work || exit 1
pkill -f "npu_loop_[b].py" 2>/dev/null; sleep 2
echo "########## (b) + CPU draft ##########"
python3 npu_loop_b.py 320 3 96 cpu 2>&1 | grep -E "^\[|NPU_LOOP_B|Error"
echo "########## (b) + NPU draft ##########"
python3 npu_loop_b.py 320 3 96 npu 2>&1 | grep -E "compiled|^\[|NPU_LOOP_B|Error"
echo "########## DONE ##########"

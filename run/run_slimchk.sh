#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work || exit 1
export DEV=0 NTHREADS=2
rm -f /tmp/sig_full.pt /tmp/sig_slim.pt
TAG=full DIR=dual_17_256 KEEPN=37 python3 slim_check.py 2>&1 | grep -aE "nout=|COS |TIME |Traceback|Error"
sleep 3
TAG=slim DIR=slim_17_256 KEEPN=6 python3 slim_check.py 2>&1 | grep -aE "nout=|COS |TIME |Traceback|Error"
echo SLIMCMP_DONE

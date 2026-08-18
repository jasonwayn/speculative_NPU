import sys
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
from dflash.benchmark import load_and_process_dataset as load_data
for n in ["gsm8k","math500","humaneval","mbpp","mt-bench"]:
    try:
        d=load_data(n); print(f"{n:10s} {len(d):5d}  ex={str(d[0])[:90]}",flush=True)
    except Exception as e:
        print(f"{n:10s} FAIL {type(e).__name__}: {str(e)[:90]}",flush=True)

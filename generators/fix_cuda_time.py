import io
p="/home/work/npu_work/dflash_work/dflash/dflash/model.py"
s=io.open(p).read()
old="def _cuda_time() -> float:\n    torch.cuda.synchronize()\n    return time.perf_counter()"
new="def _cuda_time() -> float:\n    if torch.cuda.is_available():\n        torch.cuda.synchronize()\n    return time.perf_counter()"
if new in s:
    print("ALREADY PATCHED")
elif old in s:
    io.open(p,"w").write(s.replace(old,new)); print("PATCHED")
else:
    print("PATTERN NOT FOUND"); print(repr(s[s.find("_cuda_time")-40:s.find("_cuda_time")+200]))

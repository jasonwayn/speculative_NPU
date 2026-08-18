import io
p="/home/work/npu_work/dflash_work/dflash/dflash/benchmark.py"
s=io.open(p).read()
if "# CPU-PATCHED" in s:
    print("ALREADY PATCHED"); raise SystemExit
reps=[
 ("    torch.cuda.manual_seed_all(0)",
  "    # CPU-PATCHED\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(0)"),
 ("    torch.cuda.set_device(_dist_local_rank())",
  "    if torch.cuda.is_available():\n        torch.cuda.set_device(_dist_local_rank())"),
 ('    device = torch.device(f"cuda:{_dist_local_rank()}")',
  '    device = torch.device(f"cuda:{_dist_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")'),
]
n=0
for a,b in reps:
    if a in s: s=s.replace(a,b,1); n+=1
    else: print("MISS:", a[:60])
io.open(p,"w").write(s)
print(f"PATCHED {n}/3")

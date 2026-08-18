"""드래프트 비용을 [전송] 과 [연산] 으로 분해.

null 그래프: 동일한 [1,C,12800] 입력을 받지만 연산은 sum 하나 -> 기울기가 곧 전송 비용.
real 그래프: 실제 드래프트 -> 기울기 = 전송 + 연산.
차이가 연산 몫.
"""
import sys, os, time, json, torch, rebel
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
from torch import nn
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
DT=torch.float16; DTS="float16"; B=16; DEV=0
torch.set_num_threads(8)
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True)
H=cfg.hidden_size
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=DT).eval()
NT=len(draft.target_layer_ids); W=NT*H
CACHE="/home/work/npu_work/dflash_work/rbln_cache"

class Sink(nn.Module):
    """입력 전체를 읽되 연산은 최소 (합 1회). 입력이 최적화로 사라지지 않게 함."""
    def forward(s,t): return t.sum(dim=1,keepdim=True)

def get_cm(name,build):
    p=os.path.join(CACHE,name+".rbln")
    if os.path.exists(p): return rebel.RBLNCompiledModel(p)
    cm=build(); cm.save(p); return cm

def timeit(rt,args,n=10):
    for _ in range(3): rt(*args)
    ts=[]
    for _ in range(n):
        s=time.time(); rt(*args); ts.append((time.time()-s)*1000)
    ts.sort(); return ts[len(ts)//2]

res=[]
for C in [1024,4096,8192,16384]:
    th=torch.zeros(1,C,W,dtype=DT).numpy()
    mb=C*W*2/1e6
    cm=get_cm("sink_C%d_%s"%(C,DTS), lambda C=C: rebel.compile_from_torch(
        Sink().eval(), input_info=[("target_hidden",[1,C,W],DTS)]))
    rt=cm.create_runtime(device=DEV)
    tnull=timeit(rt,(th,)); del rt
    res.append(dict(C=C,MB=round(mb,1),null_ms=round(tnull,2)))
    print("C=%6d  payload=%7.1fMB  null=%7.2f ms"%(C,mb,tnull),flush=True)
print("SPLIT "+json.dumps(res),flush=True)
print("SPLIT_DONE",flush=True)

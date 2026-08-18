#!/usr/bin/env bash
cd /home/work/npu_work/dflash_work
for W in 1024 1024; do
  echo "### W=$W"
  MAXC=$W timeout 1800 python3 - <<PY 2>&1 | grep -aE "^W=|RuntimeError|Error occurred"
import os, sys, time, torch, rebel
sys.path.insert(0,"/home/work/npu_work/dflash_work"); sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
from rebel import CompileContext
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
from draft_two import Append, Block
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
DT=torch.float16; DTS="float16"; B=16; W=int(os.environ["MAXC"]); NEW=2
torch.set_num_threads(8)
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
d=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=DT).eval()
H=cfg.hidden_size; NT=len(d.target_layer_ids); NL=cfg.num_hidden_layers
NKV=cfg.num_key_value_heads; HD=getattr(cfg,"head_dim",H//cfg.num_attention_heads)
cs=[torch.zeros(1,NKV,W,HD,dtype=DT) for _ in range(2*NL)]
ci=[("past_key_values_%d"%i,[1,NKV,W,HD],DTS) for i in range(2*NL)]
cx=CompileContext(use_weight_sharing=True)
for (n,_,_),t in zip(ci,cs): cx.mark_static_address(t,n)
ai=[("th_new",[1,B,NT*H],DTS),("pos_ctx",[1,B],"int32"),("seq_ctx",[1,1],"int32"),("block_tables",[1],"int16")]+ci
bi=[("noise_emb",[1,B,H],DTS),("pos_blk",[1,B],"int32"),("seq_blk",[1,1],"int32"),("block_tables",[1],"int16")]+ci
def ex(info):
    return [cs[int(n.rsplit("_",1)[1])] if n.startswith("past_key_values_") else torch.zeros(*sh,dtype=getattr(torch,dt)) for n,sh,dt in info]
ca=rebel.compile_from_torch(Append(d,W).eval(),input_info=ai,example_inputs=ex(ai),compile_context=cx)
cb=rebel.compile_from_torch(Block(d,W).eval(),input_info=bi,example_inputs=ex(bi),compile_context=cx)
ra=rebel.Runtime(ca,tensor_type="pt",device=0); rb=rebel.Runtime(cb,tensor_type="pt",device=0)
bt=torch.zeros(1,dtype=torch.int16)
th=(torch.randn(1,B,NT*H)*0.02).to(DT); noi=(torch.randn(1,B,H)*0.02).to(DT)
fill=W-B                      # 창을 거의 채운 상태에서 측정
pc=torch.arange(fill,fill+B,dtype=torch.int32).unsqueeze(0); sc=torch.tensor([[fill]],dtype=torch.int32)
pb=torch.arange(fill,fill+B,dtype=torch.int32).unsqueeze(0); sb=torch.tensor([[fill]],dtype=torch.int32)
for _ in range(5): ra(th,pc,sc,bt); rb(noi,pb,sb,bt)
ts=[]
for _ in range(30):
    s=time.time(); ra(th,pc,sc,bt); rb(noi,pb,sb,bt); ts.append((time.time()-s)*1000)
ts.sort()
print("W=%5d   npu_draft=%6.3f ms"%(W,ts[15]))
PY
done
echo NPUD_DONE

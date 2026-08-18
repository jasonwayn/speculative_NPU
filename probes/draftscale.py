"""드래프트 지연 vs 컨텍스트 버킷 크기. CMR 의 NPU 고유 이득(컨텍스트 재전송 절감)을 정량화."""
import sys, os, time, json, torch, rebel
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
from torch import nn
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
DT=torch.float16; DTS="float16"; B=16; DEV=0
torch.set_num_threads(2)
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True)
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=DT).eval()
H=cfg.hidden_size; NT=len(draft.target_layer_ids)
print(f"H={H} NT={NT} target_hidden_width={NT*H}",flush=True)
class Wrap(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,n,t,p,mk):
        return s.d(position_ids=p,attention_mask=mk,noise_embedding=n,
                   target_hidden=t,past_key_values=None,use_cache=False,is_causal=False)
WR=Wrap(draft).eval()
CACHE="/home/work/npu_work/dflash_work/rbln_cache"; os.makedirs(CACHE,exist_ok=True)
res=[]
for C in [64, 128, 192, 256, 384, 512]:
    nm=f"draft_B{B}_C{C}_{DTS}"; path=os.path.join(CACHE,nm+".rbln")
    t0=time.time()
    try:
        if os.path.exists(path): cm=rebel.RBLNCompiledModel(path); ct=0.0
        else:
            cm=rebel.compile_from_torch(WR,input_info=[("noise_emb",[1,B,H],DTS),
                ("target_hidden",[1,C,NT*H],DTS),("position_ids",[1,C+B],"int64"),
                ("attn_mask",[1,1,B,C+B],DTS)]); cm.save(path); ct=time.time()-t0
        rt=cm.create_runtime(device=DEV)
        noi=torch.zeros(1,B,H,dtype=DT).numpy()
        th =torch.zeros(1,C,NT*H,dtype=DT).numpy()
        po =torch.zeros(1,C+B,dtype=torch.long).numpy()
        mk =torch.zeros(1,1,B,C+B,dtype=DT).numpy()
        for _ in range(3): rt(noi,th,po,mk)
        ts=[]
        for _ in range(10):
            s=time.time(); rt(noi,th,po,mk); ts.append((time.time()-s)*1000)
        ts.sort(); med=ts[len(ts)//2]
        mb=C*NT*H*2/1e6
        print(f"C={C:6d}  draft={med:7.2f} ms   ctx_payload={mb:7.1f} MB   compile={ct:5.1f}s",flush=True)
        res.append(dict(C=C,draft_ms=round(med,2),ctx_MB=round(mb,1)))
        del rt
    except Exception as e:
        print(f"C={C:6d}  FAIL {type(e).__name__}: {str(e)[:200]}",flush=True)
print("DSCALE "+json.dumps(res),flush=True)
print("DSCALE_DONE",flush=True)

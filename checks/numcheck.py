import sys, torch, torch.nn as nn, rebel, json
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
CKPT="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
from transformers import AutoConfig
from dflash.model import DFlashDraftModel
cfg=AutoConfig.from_pretrained(CKPT,trust_remote_code=True); cfg._attn_implementation="eager"
draft=DFlashDraftModel.from_pretrained(CKPT,config=cfg,dtype=torch.float32).eval()
H=cfg.hidden_size; NT=len(cfg.dflash_config["target_layer_ids"]); B=cfg.block_size; CTX=32
class Wrap(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,n,t,p,m): return s.d(position_ids=p,attention_mask=m,noise_embedding=n,
                                       target_hidden=t,past_key_values=None,use_cache=False)
w=Wrap(draft).eval()
rt=rebel.RBLNCompiledModel("/home/work/npu_work/dflash_work/draft_b16.rbln").create_runtime()
torch.manual_seed(1)
stats=[]
for trial in range(3):
    n=torch.randn(1,B,H); t=torch.randn(1,CTX,NT*H)
    p=torch.arange(CTX+B).unsqueeze(0); m=torch.zeros(1,1,B,CTX+B)
    with torch.no_grad(): ref=w(n,t,p,m).float()
    got=torch.as_tensor(rt(n.numpy(),t.numpy(),p.numpy(),m.numpy())).float()
    d=(got-ref).abs()
    rel = d.norm()/ref.norm()
    cos = torch.nn.functional.cosine_similarity(got.flatten(),ref.flatten(),dim=0)
    # 드래프트 출력은 LM head 로 argmax 되므로, 순위 보존이 핵심
    stats.append(dict(rel_l2=float(rel), cos=float(cos),
                      max_abs=float(d.max()), ref_std=float(ref.std())))
    print(f"trial{trial}: rel_L2={rel:.4e}  cos={cos:.6f}  max_abs={d.max():.3e}  ref_std={ref.std():.3e}",flush=True)
print("NUMCHECK "+json.dumps(stats[-1]),flush=True)

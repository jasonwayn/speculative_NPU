"""(b) cache-free NPU draft/verify 루프 → τ 측정.
   타깃은 CPU(정확성 기준), 드래프트는 NPU. cache 롤백 불필요.
   목적: NPU 드래프트로 τ 6.33 이 재현되는가."""
import sys, torch, torch.nn as nn, rebel, json, time
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
TGT="/home/work/npu_work/eagle_test/Qwen3-4B"
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
CTX_MAX=int(sys.argv[1]) if len(sys.argv)>1 else 320
NSAMP  =int(sys.argv[2]) if len(sys.argv)>2 else 5
MAXNEW =int(sys.argv[3]) if len(sys.argv)>3 else 256
USE_NPU_DRAFT = (len(sys.argv)<5) or (sys.argv[4]!="cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dflash.model import DFlashDraftModel, extract_context_feature, sample
torch.set_num_threads(8)
tok=AutoTokenizer.from_pretrained(TGT)
target=AutoModelForCausalLM.from_pretrained(TGT,dtype=torch.float32).eval()
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=torch.float32).eval()
B=draft.block_size; H=cfg.hidden_size; NT=len(draft.target_layer_ids); MASK=draft.mask_token_id

class Wrap(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,n,t,p,m): return s.d(position_ids=p,attention_mask=m,noise_embedding=n,
                                       target_hidden=t,past_key_values=None,use_cache=False)
w=Wrap(draft).eval()

rt=None
if USE_NPU_DRAFT:
    print(f"compiling draft CTX_MAX={CTX_MAX} ...",flush=True); t0=time.time()
    cm=rebel.compile_from_torch(w, input_info=[
        ("noise_emb",[1,B,H],"float32"), ("target_hidden",[1,CTX_MAX,NT*H],"float32"),
        ("position_ids",[1,CTX_MAX+B],"int64"), ("attn_mask",[1,1,B,CTX_MAX+B],"float32")])
    rt=cm.create_runtime(); print(f"  compiled {time.time()-t0:.1f}s",flush=True)

def run_draft(th, blk_ids):
    """th:(1,L,NT*H) 실제 컨텍스트. 좌측 패딩해서 고정 shape 으로."""
    L=th.shape[1]
    with torch.no_grad(): noise=target.model.embed_tokens(blk_ids).detach()
    if not USE_NPU_DRAFT:
        pos=torch.arange(L+B).unsqueeze(0); m=torch.zeros(1,1,B,L+B)
        with torch.no_grad(): return w(noise,th,pos,m)
    assert L<=CTX_MAX, f"ctx {L} > CTX_MAX {CTX_MAX}"
    pad=CTX_MAX-L
    thp=torch.cat([torch.zeros(1,pad,th.shape[2]), th],dim=1)          # 좌측 패딩
    pos=torch.cat([torch.zeros(1,pad,dtype=torch.long),
                   torch.arange(L+B).unsqueeze(0)],dim=1)              # 실제 구간만 유효
    m=torch.zeros(1,1,B,CTX_MAX+B); m[:,:,:,:pad]=float("-inf")        # 패딩 무시
    return torch.as_tensor(rt(noise.numpy(),thp.detach().numpy(),pos.numpy(),m.numpy())).float()

import json as _j
ds=[_j.loads(l) for l in open("/home/work/npu_work/dflash_work/dflash/cache/gsm8k.jsonl")][:NSAMP]
STOP={tok.eos_token_id,151645}
allacc=[]; t_start=time.time()
for si,ex in enumerate(ds):
    txt=ex["turns"][0]          # 캐시에 이미 공식 포맷 적용됨
    ids=tok.apply_chat_template([{"role":"user","content":txt}],add_generation_prompt=True,
                                return_tensors="pt",enable_thinking=False)
    cur=ids; acc_list=[]; n_new=0
    while n_new < MAXNEW:
        with torch.no_grad(): o=target(cur,output_hidden_states=True,use_cache=False)
        th=extract_context_feature(o.hidden_states,draft.target_layer_ids)
        if th.shape[1]>CTX_MAX: break
        bonus=torch.argmax(o.logits[:,-1:,:],dim=-1)
        blk=torch.full((1,B),MASK,dtype=torch.long); blk[0,0]=bonus[0,0]
        h=run_draft(th,blk)
        blk[:,1:]=sample(target.lm_head(h))[:,1:]   # 공식: [:, 1-B:] = 뒤쪽 B-1
        cand=torch.cat([cur,blk],dim=1)
        with torch.no_grad(): o2=target(cand,use_cache=False)
        post=torch.argmax(o2.logits[:,cur.shape[1]-1:cur.shape[1]-1+B,:],dim=-1)
        a=int((blk[:,1:]==post[:,:-1]).cumprod(dim=1).sum(dim=1)[0])
        acc=a+1; acc_list.append(acc); allacc.append(acc)
        newtok=torch.cat([blk[:,:acc],post[:,a:a+1]],dim=1)
        cur=torch.cat([cur,newtok],dim=1); n_new+=newtok.shape[1]
        if any(int(t) in STOP for t in newtok[0]): break
    tau=sum(acc_list)/max(len(acc_list),1)
    print(f"[{si}] new={n_new} cycles={len(acc_list)} tau={tau:.3f} "
          f"running_tau={sum(allacc)/len(allacc):.3f} elapsed={time.time()-t_start:.0f}s",flush=True)
print("NPU_LOOP_B "+json.dumps(dict(mode="npu" if USE_NPU_DRAFT else "cpu",
      tau=round(sum(allacc)/len(allacc),3),cycles=len(allacc),
      samples=len(ds),ctx_max=CTX_MAX,ref_cpu=6.33)),flush=True)

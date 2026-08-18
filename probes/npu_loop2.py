"""cache-free NPU draft/verify 루프 (수정판). trace2 로 등가 확인된 규약 그대로."""
import sys, torch, torch.nn as nn, rebel, json, time
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
TGT="/home/work/npu_work/eagle_test/Qwen3-4B"; DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
CTX_MAX=int(sys.argv[1]); NSAMP=int(sys.argv[2]); MAXNEW=int(sys.argv[3])
MODE=sys.argv[4] if len(sys.argv)>4 else "npu"
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
                                       target_hidden=t,past_key_values=None,use_cache=False,is_causal=False)
w=Wrap(draft).eval()
rt=None
if MODE=="npu":
    print(f"compiling CTX_MAX={CTX_MAX} ...",flush=True); t0=time.time()
    cm=rebel.compile_from_torch(w,input_info=[("noise_emb",[1,B,H],"float32"),
        ("target_hidden",[1,CTX_MAX,NT*H],"float32"),("position_ids",[1,CTX_MAX+B],"int64"),
        ("attn_mask",[1,1,B,CTX_MAX+B],"float32")])
    rt=cm.create_runtime(); print(f"  compiled {time.time()-t0:.1f}s",flush=True)

def draft_fwd(th, blk):
    L=th.shape[1]
    with torch.no_grad(): noise=target.model.embed_tokens(blk).detach()
    if MODE!="npu":
        with torch.no_grad():
            return w(noise,th,torch.arange(L+B).unsqueeze(0),torch.zeros(1,1,B,L+B))
    pad=CTX_MAX-L
    thp=torch.cat([torch.zeros(1,pad,th.shape[2]),th],dim=1)
    pos=torch.cat([torch.zeros(1,pad,dtype=torch.long),torch.arange(L+B).unsqueeze(0)],dim=1)
    m=torch.zeros(1,1,B,CTX_MAX+B); m[:,:,:,:pad]=float("-inf")
    return torch.as_tensor(rt(noise.numpy(),thp.detach().numpy(),pos.numpy(),m.numpy())).float()

ds=[json.loads(l) for l in open("/home/work/npu_work/dflash_work/dflash/cache/gsm8k.jsonl")][:NSAMP]
STOP={tok.eos_token_id,151645}; allacc=[]; t0=time.time()
for si,ex in enumerate(ds):
    ids=tok.apply_chat_template([{"role":"user","content":ex["turns"][0]}],add_generation_prompt=True,
                                return_tensors="pt",enable_thinking=False)
    with torch.no_grad(): o=target(ids,output_hidden_states=True,use_cache=False)
    verified=ids                                              # 0..start-1 (보너스 미포함)
    bonus=int(torch.argmax(o.logits[:,-1,:]))                  # 위치 start 의 토큰
    th=extract_context_feature(o.hidden_states,draft.target_layer_ids)
    acc_list=[]; P=ids.shape[1]; MAXLEN=P+MAXNEW
    while verified.shape[1]<MAXLEN and th.shape[1]<=CTX_MAX:
        blk=torch.full((1,B),MASK,dtype=torch.long); blk[0,0]=bonus
        h=draft_fwd(th,blk)
        with torch.no_grad(): blk[:,1:]=sample(target.lm_head(h[:,1-B:,:]))
        cand=torch.cat([verified,blk],dim=1)
        with torch.no_grad(): o2=target(cand,output_hidden_states=True,use_cache=False)
        vlen=verified.shape[1]
        post=torch.argmax(o2.logits[:,vlen:vlen+B,:],dim=-1)   # logits[vlen+j] -> blk[j+1]
        a=int((blk[:,1:]==post[:,:-1]).cumprod(dim=1).sum(dim=1)[0])
        acc_list.append(a+1); allacc.append(a+1)
        verified=torch.cat([verified,blk[:,:a+1]],dim=1)       # 보너스+채택분 확정
        bonus=int(post[0,a])                                   # 새 보너스
        th=extract_context_feature(o2.hidden_states,draft.target_layer_ids)[:,:vlen+a+1,:]
        if any(int(t) in STOP for t in verified[0,P:]): break   # 공식과 동일
    tau=sum(acc_list)/max(len(acc_list),1)
    print(f"[{si}] new={verified.shape[1]-P} cycles={len(acc_list)} tau={tau:.3f} "
          f"running={sum(allacc)/len(allacc):.3f} t={time.time()-t0:.0f}s",flush=True)
print("LOOP2 "+json.dumps(dict(mode=MODE,tau=round(sum(allacc)/len(allacc),3),
      cycles=len(allacc),samples=len(ds),ref=6.33)),flush=True)

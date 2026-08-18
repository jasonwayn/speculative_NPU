"""타깃·드래프트 모두 NPU. lm_head/embed_tokens 만 CPU."""
import sys, torch, torch.nn as nn, rebel, json, time
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
TGT_NPU="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden"
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
CTX_MAX=int(sys.argv[1]); NSAMP=int(sys.argv[2]); MAXNEW=int(sys.argv[3])
TGT_MODE=sys.argv[4] if len(sys.argv)>4 else "npu"     # npu | cpu
DRF_MODE=sys.argv[5] if len(sys.argv)>5 else "npu"
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from optimum.rbln import RBLNQwen3ForCausalLM
from dflash.model import DFlashDraftModel, extract_context_feature, sample
torch.set_num_threads(8)
tok=AutoTokenizer.from_pretrained(SRC)
cpu=AutoModelForCausalLM.from_pretrained(SRC,dtype=torch.float32).eval()
lm_head=cpu.lm_head; embed=cpu.model.embed_tokens
tgt_npu=RBLNQwen3ForCausalLM.from_pretrained(TGT_NPU,export=False) if TGT_MODE=="npu" else None
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=torch.float32).eval()
B=draft.block_size; H=cfg.hidden_size; NT=len(draft.target_layer_ids); MASK=draft.mask_token_id
LAYER_IDS=draft.target_layer_ids

class Wrap(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,n,t,p,m): return s.d(position_ids=p,attention_mask=m,noise_embedding=n,
                                       target_hidden=t,past_key_values=None,use_cache=False,is_causal=False)
w=Wrap(draft).eval()
rt=None
if DRF_MODE=="npu":
    t0=time.time()
    cm=rebel.compile_from_torch(w,input_info=[("noise_emb",[1,B,H],"float32"),
        ("target_hidden",[1,CTX_MAX,NT*H],"float32"),("position_ids",[1,CTX_MAX+B],"int64"),
        ("attn_mask",[1,1,B,CTX_MAX+B],"float32")])
    rt=cm.create_runtime(); print(f"draft compiled {time.time()-t0:.1f}s",flush=True)

CHUNK=128   # optimum-rbln 0.10.2 버그: 길이가 prefill_chunk_size 배수면 깨짐
def target_fwd(ids):
    if TGT_MODE=="npu":
        L=ids.shape[1]
        if L % CHUNK == 0:                       # 우회: 더미 1토큰 덧붙여 실행 후 잘라냄
            ids2=torch.cat([ids, ids[:,-1:]],dim=1)
            o=tgt_npu(input_ids=ids2,attention_mask=torch.ones_like(ids2))
            return tuple(h[:,:L] for h in o.hidden_states)
        o=tgt_npu(input_ids=ids,attention_mask=torch.ones_like(ids))
        return o.hidden_states
    with torch.no_grad():
        return cpu(ids,output_hidden_states=True,use_cache=False).hidden_states

def draft_fwd(th, blk):
    L=th.shape[1]
    with torch.no_grad(): noise=embed(blk).detach()
    if DRF_MODE!="npu":
        with torch.no_grad(): return w(noise,th,torch.arange(L+B).unsqueeze(0),torch.zeros(1,1,B,L+B))
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
    hs=target_fwd(ids)
    with torch.no_grad(): bonus=int(torch.argmax(lm_head(hs[-1][:,-1].float())))
    verified=ids; th=extract_context_feature(hs,LAYER_IDS)
    P=ids.shape[1]; MAXLEN=P+MAXNEW; acc=[]
    if th.shape[1]>CTX_MAX:
        print(f'[{si}] SKIP prompt {th.shape[1]}>CTX_MAX',flush=True); continue
    while verified.shape[1]<MAXLEN and th.shape[1]<=CTX_MAX:
        blk=torch.full((1,B),MASK,dtype=torch.long); blk[0,0]=bonus
        h=draft_fwd(th,blk)
        with torch.no_grad(): blk[:,1:]=sample(lm_head(h[:,1-B:,:].float()))
        cand=torch.cat([verified,blk],dim=1)
        hs2=target_fwd(cand); vlen=verified.shape[1]
        with torch.no_grad():
            post=torch.argmax(lm_head(hs2[-1][:,vlen:vlen+B].float()),dim=-1)
        a=int((blk[:,1:]==post[:,:-1]).cumprod(dim=1).sum(dim=1)[0])
        acc.append(a+1); allacc.append(a+1)
        verified=torch.cat([verified,blk[:,:a+1]],dim=1); bonus=int(post[0,a])
        th=extract_context_feature(hs2,LAYER_IDS)[:,:vlen+a+1,:]
        if any(int(t) in STOP for t in verified[0,P:]): break
    print(f"[{si}] new={verified.shape[1]-P} cycles={len(acc)} tau={sum(acc)/max(len(acc),1):.3f} "
          f"running={sum(allacc)/len(allacc):.3f} t={time.time()-t0:.0f}s",flush=True)
print("NPUFULL "+json.dumps(dict(tgt=TGT_MODE,drf=DRF_MODE,tau=round(sum(allacc)/len(allacc),3),
      cycles=len(allacc),samples=len(ds),ref=4.785)),flush=True)

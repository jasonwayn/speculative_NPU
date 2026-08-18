"""DFlash on NPU — prefix caching 적용판. 타깃은 꼬리만 재계산."""
import sys, os, json, time, torch, torch.nn as nn, rebel
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
TGT_NPU="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden"
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
CTX_MAX=int(sys.argv[1]); NSAMP=int(sys.argv[2]); MAXNEW=int(sys.argv[3])
CACHED = (len(sys.argv)<5) or (sys.argv[4]!="nocache")
DEV=int(os.environ.get("DEV","0"))
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from optimum.rbln import RBLNQwen3ForCausalLM
from dflash.model import DFlashDraftModel, extract_context_feature, sample
torch.set_num_threads(8)
tok=AutoTokenizer.from_pretrained(SRC)
cpu=AutoModelForCausalLM.from_pretrained(SRC,dtype=torch.float32).eval()
lm_head=cpu.lm_head; embed=cpu.model.embed_tokens
m=RBLNQwen3ForCausalLM.from_pretrained(TGT_NPU,export=False,rbln_device=DEV)
pd=m.prefill_decoder; CHUNK=pd.rbln_config.prefill_chunk_size
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=torch.float32).eval()
B=draft.block_size; H=cfg.hidden_size; NT=len(draft.target_layer_ids); MASK=draft.mask_token_id
LID=draft.target_layer_ids; BT=torch.tensor([0],dtype=torch.int16)

class Wrap(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,n,t,p,mk): return s.d(position_ids=p,attention_mask=mk,noise_embedding=n,
                                        target_hidden=t,past_key_values=None,use_cache=False,is_causal=False)
w=Wrap(draft).eval()
t0=time.time()
cm=rebel.compile_from_torch(w,input_info=[("noise_emb",[1,B,H],"float32"),
    ("target_hidden",[1,CTX_MAX,NT*H],"float32"),("position_ids",[1,CTX_MAX+B],"int64"),
    ("attn_mask",[1,1,B,CTX_MAX+B],"float32")])
rt=cm.create_runtime(device=DEV); print(f"draft compiled {time.time()-t0:.1f}s  CHUNK={CHUNK}",flush=True)

class LMH(nn.Module):
    def __init__(s,l): super().__init__(); s.l=l
    def forward(s,x): return s.l(x)
t0=time.time()
lmh_cm=rebel.compile_from_torch(LMH(lm_head).eval(), input_info=[("x",[1,B,H],"float32")])
lmh_rt=lmh_cm.create_runtime(device=DEV); print(f"lm_head compiled {time.time()-t0:.1f}s",flush=True)
T_LMH=[0.0]
def lmh(x):
    """x:(1,n,H) n<=B -> argmax token ids (1,n)"""
    n=x.shape[1]
    if n<B: x=torch.cat([x, torch.zeros(1,B-n,H)],dim=1)
    s=time.time(); o=torch.as_tensor(lmh_rt(x.contiguous().numpy())); T_LMH[0]+=time.time()-s
    return torch.argmax(o[:,:n].float(),dim=-1)

T_DRAFT=[0.0]; T_VERIFY=[0.0]
# 버퍼 1회 할당 후 재사용 (매 라운드 재할당·복사 제거)
TH_BUF=torch.zeros(1,CTX_MAX,NT*H)
POS_BUF=torch.zeros(1,CTX_MAX+B,dtype=torch.long)
MK_BUF=torch.zeros(1,1,B,CTX_MAX+B)
PREV_PAD=[CTX_MAX]
def reset_buffers():
    TH_BUF.zero_(); POS_BUF.zero_(); MK_BUF.zero_()
    MK_BUF[:,:,:,:CTX_MAX]=float("-inf"); PREV_PAD[0]=CTX_MAX
def draft_fwd(th,blk):
    L=th.shape[1]; pad=CTX_MAX-L
    with torch.no_grad(): noise=embed(blk).detach()
    TH_BUF[:,pad:]=th                       # 꼬리만 갱신, [0:pad] 는 이미 0
    POS_BUF[:,pad:]=torch.arange(L+B)
    if pad<PREV_PAD[0]: MK_BUF[:,:,:,pad:PREV_PAD[0]]=0.0   # 새로 유효해진 구간만 해제
    PREV_PAD[0]=pad
    s=time.time(); r=rt(noise.numpy(),TH_BUF.numpy(),POS_BUF.numpy(),MK_BUF.numpy()); T_DRAFT[0]+=time.time()-s
    return torch.as_tensor(r).float()

def tgt_prefill(seg, off):
    """seg 를 캐시 오프셋 off 에 prefill. -> hidden_states tuple"""
    s=time.time(); L=seg.shape[1]; pad = 1 if L % CHUNK == 0 else 0   # 128배수 버그 우회
    if pad: seg=torch.cat([seg,seg[:,-1:]],dim=1)
    r=pd.prefill_forward(seg,
        cache_position=torch.arange(off,off+seg.shape[1],dtype=torch.int32).unsqueeze(0),
        attention_mask=torch.ones(seg.shape[1],dtype=torch.int64),
        batch_idx=0, block_tables=BT, is_external_block_tables=False)
    T_VERIFY[0]+=time.time()-s
    hs=r.hidden_states
    return tuple(h[:,:L] for h in hs) if pad else hs

def tgt_whole(ids):
    s=time.time()
    if ids.shape[1] % CHUNK == 0:
        i2=torch.cat([ids,ids[:,-1:]],dim=1)
        hs=m(input_ids=i2,attention_mask=torch.ones_like(i2)).hidden_states
        hs=tuple(h[:,:ids.shape[1]] for h in hs)
    else:
        hs=m(input_ids=ids,attention_mask=torch.ones_like(ids)).hidden_states
    T_VERIFY[0]+=time.time()-s
    return hs

ds=[json.loads(l) for l in open("/home/work/npu_work/dflash_work/dflash/cache/gsm8k.jsonl")][:NSAMP]
STOP={tok.eos_token_id,151645}; allacc=[]; W0=time.time(); ntok=0
for si,ex in enumerate(ds):
    ids=tok.apply_chat_template([{"role":"user","content":ex["turns"][0]}],add_generation_prompt=True,
                                return_tensors="pt",enable_thinking=False)
    P=ids.shape[1]
    hs=tgt_whole(ids)                                  # 최초 prefill (캐시 채움)
    with torch.no_grad(): bonus=int(torch.argmax(lm_head(hs[-1][:,-1].float())))
    th=extract_context_feature(hs,LID)                 # 누적 버퍼
    verified=ids; cached=(P//CHUNK)*CHUNK if CACHED else 0
    acc=[]; reset_buffers()
    while verified.shape[1]<P+MAXNEW and th.shape[1]<=CTX_MAX:
        blk=torch.full((1,B),MASK,dtype=torch.long); blk[0,0]=bonus
        h=draft_fwd(th,blk)
        blk[:,1:]=lmh(h[:,1-B:,:].float())
        vlen=verified.shape[1]; cand=torch.cat([verified,blk],dim=1)
        if CACHED:
            seg=cand[:,cached:]                        # 꼬리만 (≤ 128+B)
            hs2=tgt_prefill(seg,cached); base=cached
        else:
            hs2=tgt_whole(cand); base=0
        post=lmh(hs2[-1][:,vlen-base:vlen-base+B].float())
        a=int((blk[:,1:]==post[:,:-1]).cumprod(dim=1).sum(dim=1)[0])
        acc.append(a+1); allacc.append(a+1)
        newfeat=extract_context_feature(hs2,LID)[:,vlen-base:vlen-base+a+1,:]
        th=torch.cat([th,newfeat],dim=1)               # 누적 (재계산 없음)
        verified=torch.cat([verified,blk[:,:a+1]],dim=1); bonus=int(post[0,a])
        if CACHED: cached=(verified.shape[1]//CHUNK)*CHUNK
        if any(int(t) in STOP for t in verified[0,P:]): break
    ntok+=verified.shape[1]-P
    print(f"[{si}] new={verified.shape[1]-P} cycles={len(acc)} tau={sum(acc)/max(len(acc),1):.3f} "
          f"running={sum(allacc)/len(allacc):.3f} t={time.time()-W0:.0f}s",flush=True)
W=time.time()-W0
print("CACHED_RESULT "+json.dumps(dict(cached=CACHED,tau=round(sum(allacc)/len(allacc),3),
      cycles=len(allacc),samples=len(ds),tokens=ntok,wall_s=round(W,1),
      tok_s=round(ntok/W,2),draft_s=round(T_DRAFT[0],1),verify_s=round(T_VERIFY[0],1),
      lmhead_s=round(T_LMH[0],1),other_s=round(W-T_DRAFT[0]-T_VERIFY[0]-T_LMH[0],1))),flush=True)

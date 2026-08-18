"""실제 target hidden 으로 CPU vs NPU 드래프트의 '예측 토큰' 일치율 측정."""
import sys, torch, torch.nn as nn, rebel, json, time
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
TGT="/home/work/npu_work/eagle_test/Qwen3-4B"
DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dflash.model import DFlashDraftModel, extract_context_feature, sample

torch.set_num_threads(8)
tok=AutoTokenizer.from_pretrained(TGT)
target=AutoModelForCausalLM.from_pretrained(TGT, dtype=torch.float32).eval()
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
draft=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=torch.float32).eval()
B=draft.block_size; H=cfg.hidden_size; NT=len(draft.target_layer_ids)
print(f"block={B} hidden={H} n_tgt={NT} layer_ids={draft.target_layer_ids}",flush=True)

# --- 실제 프롬프트로 타깃 prefill → 진짜 target_hidden 확보 ---
q="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
txt=q+"\nPlease reason step by step, and put your final answer within \boxed{}."
ids=tok.apply_chat_template([{"role":"user","content":txt}],add_generation_prompt=True,
                            return_tensors="pt",enable_thinking=False)
with torch.no_grad():
    out=target(ids, output_hidden_states=True, use_cache=False)
th_full=extract_context_feature(out.hidden_states, draft.target_layer_ids)   # (1,L,12800)
first=torch.argmax(out.logits[:,-1:,:],dim=-1)
CTX=th_full.shape[1]
print(f"prompt len={CTX}  target_hidden={tuple(th_full.shape)}  first_tok={first.item()}",flush=True)

# --- 드래프트 입력 구성 (cache 없이 1회 forward) ---
blk=torch.full((1,B), draft.mask_token_id, dtype=torch.long); blk[0,0]=first[0,0]
with torch.no_grad():
    noise=target.model.embed_tokens(blk).detach()
pos=torch.arange(CTX+B).unsqueeze(0)
mask=torch.zeros(1,1,B,CTX+B)

class Wrap(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,n,t,p,m): return s.d(position_ids=p,attention_mask=m,noise_embedding=n,
                                       target_hidden=t,past_key_values=None,use_cache=False)
w=Wrap(draft).eval()
with torch.no_grad():
    h_cpu=w(noise,th_full,pos,mask)
    tok_cpu=sample(target.lm_head(h_cpu))
print("CPU  tokens:", tok_cpu[0].tolist(), flush=True)

# --- 같은 CTX 로 NPU 컴파일 ---
print(f"\ncompiling draft for CTX={CTX} ...",flush=True); t0=time.time()
cm=rebel.compile_from_torch(w, input_info=[
    ("noise_emb",[1,B,H],"float32"), ("target_hidden",[1,CTX,NT*H],"float32"),
    ("position_ids",[1,CTX+B],"int64"), ("attn_mask",[1,1,B,CTX+B],"float32")])
print(f"  compiled {time.time()-t0:.1f}s",flush=True)
rt=cm.create_runtime()
h_npu=torch.as_tensor(rt(noise.detach().numpy(),th_full.detach().numpy(),pos.numpy(),mask.numpy())).float()
with torch.no_grad():
    tok_npu=sample(target.lm_head(h_npu))
print("NPU  tokens:", tok_npu[0].tolist(), flush=True)

same=(tok_cpu==tok_npu).sum().item()
cos=torch.nn.functional.cosine_similarity(h_npu.flatten(),h_cpu.float().flatten(),dim=0)
print(f"\n일치 {same}/{B}   hidden cos={cos:.6f}",flush=True)
print("텍스트 CPU:", repr(tok.decode(tok_cpu[0])),flush=True)
print("텍스트 NPU:", repr(tok.decode(tok_npu[0])),flush=True)
print("FIDELITY "+json.dumps(dict(ctx=CTX,block=B,same=same,cos=float(cos))),flush=True)

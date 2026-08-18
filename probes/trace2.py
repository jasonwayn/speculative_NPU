"""공식(cache) vs 내 방식(cache-free) 을 같은 라운드에서 직접 대조."""
import sys, torch, json
sys.path.insert(0,"/home/work/npu_work/dflash_work/dflash")
TGT="/home/work/npu_work/eagle_test/Qwen3-4B"; DRF="/home/work/npu_work/dflash_work/Qwen3-4B-DFlash-b16"
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DynamicCache
from dflash.model import DFlashDraftModel, extract_context_feature, sample
torch.set_num_threads(8)
tok=AutoTokenizer.from_pretrained(TGT)
target=AutoModelForCausalLM.from_pretrained(TGT,dtype=torch.float32).eval()
cfg=AutoConfig.from_pretrained(DRF,trust_remote_code=True); cfg._attn_implementation="eager"
model=DFlashDraftModel.from_pretrained(DRF,config=cfg,dtype=torch.float32).eval()
ex=[json.loads(l) for l in open("/home/work/npu_work/dflash_work/dflash/cache/gsm8k.jsonl")][0]
ids=tok.apply_chat_template([{"role":"user","content":ex["turns"][0]}],add_generation_prompt=True,
                            return_tensors="pt",enable_thinking=False)
B=model.block_size; MASK=model.mask_token_id
num_input=ids.shape[1]; max_length=num_input+40
output_ids=torch.full((1,max_length+B),MASK,dtype=torch.long)
position_ids=torch.arange(output_ids.shape[1]).unsqueeze(0)
pkv_t=DynamicCache(); pkv_d=DynamicCache()

with torch.inference_mode():
    out=target(ids,position_ids=position_ids[:,:num_input],past_key_values=pkv_t,
               use_cache=True,logits_to_keep=1,output_hidden_states=True)
    output_ids[:,:num_input]=ids
    output_ids[:,num_input:num_input+1]=sample(out.logits,0.0)
    th_cached=extract_context_feature(out.hidden_states,model.target_layer_ids)
    start=num_input
    for cyc in range(3):
        blk=output_ids[:,start:start+B].clone()
        noise=target.model.embed_tokens(blk)
        # ---------- A: 공식 (cache) ----------
        dlen=pkv_d.get_seq_length()
        hA=model(target_hidden=th_cached,noise_embedding=noise,
                 position_ids=position_ids[:,dlen:start+B],
                 past_key_values=pkv_d,use_cache=True,is_causal=False)
        sA=sample(target.lm_head(hA[:,1-B:,:]))
        pkv_d.crop(start)
        # ---------- B: 내 방식 (cache-free, 전체 ctx) ----------
        oB=target(output_ids[:,:start],output_hidden_states=True,use_cache=False)
        th_full=extract_context_feature(oB.hidden_states,model.target_layer_ids)
        L=th_full.shape[1]
        hB=model(target_hidden=th_full,noise_embedding=noise,
                 position_ids=torch.arange(L+B).unsqueeze(0),
                 past_key_values=None,use_cache=False,is_causal=False)
        sB=sample(target.lm_head(hB[:,1-B:,:]))
        same=int((sA==sB).sum())
        print(f"\n--- cycle {cyc} start={start} ---",flush=True)
        print(f"  A(cache)    ctx={tuple(th_cached.shape)} pos=[{position_ids[0,dlen].item()}..{position_ids[0,start+B-1].item()}]",flush=True)
        print(f"  B(cachefree)ctx={tuple(th_full.shape)} pos=[0..{L+B-1}]",flush=True)
        print(f"  A tokens = {sA[0,:8].tolist()}",flush=True)
        print(f"  B tokens = {sB[0,:8].tolist()}",flush=True)
        print(f"  일치 {same}/{sA.shape[1]}   hidden cos={torch.nn.functional.cosine_similarity(hA.flatten(),hB.flatten(),dim=0):.6f}",flush=True)
        # 공식 경로로 진행
        blk[:,1:]=sA
        out=target(blk,position_ids=position_ids[:,start:start+B],past_key_values=pkv_t,
                   use_cache=True,output_hidden_states=True)
        post=sample(out.logits,0.0)
        a=int((blk[:,1:]==post[:,:-1]).cumprod(dim=1).sum(dim=1)[0])
        output_ids[:,start:start+a+1]=blk[:,:a+1]; output_ids[:,start+a+1]=post[:,a]
        start+=a+1; pkv_t.crop(start)
        th_cached=extract_context_feature(out.hidden_states,model.target_layer_ids)[:,:a+1,:]
        print(f"  ACCEPT={a}+1",flush=True)
print("\nTRACE2_DONE",flush=True)

"""공식 dflash_generate 를 그대로 복제하되 라운드별 텐서를 찍는다. 규약 확정용."""
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
block_size=model.block_size; mask_token_id=model.mask_token_id
num_input=ids.shape[1]; max_new=40; max_length=num_input+max_new
print(f"prompt={num_input} block={block_size} mask_id={mask_token_id}",flush=True)

output_ids=torch.full((1,max_length+block_size),mask_token_id,dtype=torch.long)
position_ids=torch.arange(output_ids.shape[1]).unsqueeze(0)
pkv_t=DynamicCache(); pkv_d=DynamicCache()

with torch.inference_mode():
    out=target(ids,position_ids=position_ids[:,:num_input],past_key_values=pkv_t,
               use_cache=True,logits_to_keep=1,output_hidden_states=True)
    output_ids[:,:num_input]=ids
    output_ids[:,num_input:num_input+1]=sample(out.logits,0.0)
    target_hidden=extract_context_feature(out.hidden_states,model.target_layer_ids)
    print(f"[prefill] target_hidden={tuple(target_hidden.shape)}  first_tok={output_ids[0,num_input].item()}",flush=True)

    start=num_input
    for cyc in range(4):
        if start>=max_length: break
        blk=output_ids[:,start:start+block_size].clone()
        dlen_before=pkv_d.get_seq_length()
        pos_slice=position_ids[:,dlen_before:start+block_size]
        noise=target.model.embed_tokens(blk)
        print(f"\n--- cycle {cyc} | start={start} ---",flush=True)
        print(f"  blk_in      = {blk[0,:5].tolist()} ... (len {blk.shape[1]}, mask={mask_token_id})",flush=True)
        print(f"  draft cache before = {dlen_before}",flush=True)
        print(f"  target_hidden = {tuple(target_hidden.shape)}",flush=True)
        print(f"  position_ids  = [{pos_slice[0,0].item()} .. {pos_slice[0,-1].item()}] len={pos_slice.shape[1]}",flush=True)
        h=model(target_hidden=target_hidden,noise_embedding=noise,position_ids=pos_slice,
                past_key_values=pkv_d,use_cache=True,is_causal=False)
        print(f"  draft out     = {tuple(h.shape)}   -> slice [:, {1-block_size}:, :] = {tuple(h[:,1-block_size:,:].shape)}",flush=True)
        print(f"  draft cache after  = {pkv_d.get_seq_length()}",flush=True)
        dl=target.lm_head(h[:,1-block_size:,:])
        s=sample(dl)
        print(f"  sampled(len {s.shape[1]}) = {s[0,:6].tolist()} ...",flush=True)
        pkv_d.crop(start); print(f"  draft cache cropped to {pkv_d.get_seq_length()}  (start={start})",flush=True)
        blk[:,1:]=s
        out=target(blk,position_ids=position_ids[:,start:start+block_size],past_key_values=pkv_t,
                   use_cache=True,output_hidden_states=True)
        post=sample(out.logits,0.0)
        a=int((blk[:,1:]==post[:,:-1]).cumprod(dim=1).sum(dim=1)[0])
        print(f"  blk_guess   = {blk[0,1:7].tolist()} ...",flush=True)
        print(f"  posterior   = {post[0,:6].tolist()} ...",flush=True)
        print(f"  ACCEPT      = {a}  (+1 bonus = {a+1})",flush=True)
        output_ids[:,start:start+a+1]=blk[:,:a+1]
        output_ids[:,start+a+1]=post[:,a]
        start+=a+1
        pkv_t.crop(start); print(f"  target cache cropped to {pkv_t.get_seq_length()} (start={start})",flush=True)
        target_hidden=extract_context_feature(out.hidden_states,model.target_layer_ids)[:,:a+1,:]
        print(f"  next target_hidden = {tuple(target_hidden.shape)}",flush=True)
print("\nTRACE_DONE",flush=True)

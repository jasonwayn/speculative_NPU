"""128 배수 오프셋 prefix caching 이 통짜 prefill 과 같은가?"""
import torch, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
m=RBLNQwen3ForCausalLM.from_pretrained("/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden",
                                       export=False, rbln_device=1)
pd=m.prefill_decoder
torch.manual_seed(0); N=144; ids=torch.randint(1000,5000,(1,N))
hA=m(input_ids=ids,attention_mask=torch.ones_like(ids)).hidden_states[-1]
print("기준 통짜-144:",tuple(hA.shape),flush=True)
def ffwd(x, off):
    cp=torch.arange(off,off+x.shape[1],dtype=torch.int32).unsqueeze(0)
    return pd.prefill_forward(x, cache_position=cp, attention_mask=torch.ones(off+x.shape[1],dtype=torch.int64), batch_idx=0)
try:
    r1=ffwd(ids[:,:128],0);   print("  offset 0  len128 OK",flush=True)
    r2=ffwd(ids[:,128:144],128); print("  offset 128 len16 OK",flush=True)
    hs=getattr(r2,"hidden_states",None)
    if hs is None:
        print("  r2 attrs:",[a for a in dir(r2) if not a.startswith('_')][:10],flush=True)
    else:
        h2=hs[-1].float(); ref=hA[:,128:144].float()
        print(f"  prefix-cached vs whole: max={(h2-ref).abs().max():.3e} "
              f"cos={torch.nn.functional.cosine_similarity(h2.flatten(),ref.flatten(),dim=0):.6f}",flush=True)
except Exception:
    traceback.print_exc()
print("DONE",flush=True)

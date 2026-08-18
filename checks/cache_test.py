"""오프셋 prefill 검증 — 카드 1 사용."""
import torch, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
m=RBLNQwen3ForCausalLM.from_pretrained("/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden",
                                       export=False, rbln_device=1)
pd=m.prefill_decoder; print("prefill_decoder:",type(pd).__name__,flush=True)
torch.manual_seed(0); ids=torch.randint(1000,5000,(1,40))
hA=m(input_ids=ids,attention_mask=torch.ones_like(ids)).hidden_states[-1]
print("A whole-40:",tuple(hA.shape),flush=True)
try:
    r1=pd.prefill_forward(ids[:,:24],cache_position=torch.arange(0,24,dtype=torch.int32).unsqueeze(0),batch_idx=0)
    print("  step1 ->",type(r1).__name__,flush=True)
    r2=pd.prefill_forward(ids[:,24:40],cache_position=torch.arange(24,40,dtype=torch.int32).unsqueeze(0),batch_idx=0)
    print("  step2 ->",type(r2).__name__,flush=True)
    for tag,r in [("r1",r1),("r2",r2)]:
        hs=getattr(r,"hidden_states",None)
        print(f"  {tag}: attrs={[a for a in dir(r) if not a.startswith('_')][:8]} hidden={None if hs is None else f'{len(hs)}x{tuple(hs[0].shape)}'}",flush=True)
    hs2=getattr(r2,"hidden_states",None)
    if hs2 is not None:
        h2=hs2[-1].float(); ref=hA[:,24:40].float()
        print(f"  offset vs whole: max={ (h2-ref).abs().max():.3e} "
              f"cos={torch.nn.functional.cosine_similarity(h2.flatten(),ref.flatten(),dim=0):.6f}",flush=True)
except Exception:
    traceback.print_exc()
print("DONE",flush=True)

"""타깃 Qwen3-4B 를 output_hidden_states=True 로 컴파일 → 5개 층 추출 확인"""
import torch, time, json, traceback
from optimum.rbln import RBLNQwen3ForCausalLM
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
DST="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden"
TGT_IDS=[1,9,17,25,33]
try:
    t0=time.time()
    m = RBLNQwen3ForCausalLM.from_pretrained(
        SRC, export=True,
        rbln_batch_size=1, rbln_max_seq_len=4096,
        rbln_output_hidden_states=True,
    )
    print(f"COMPILED in {time.time()-t0:.1f}s", flush=True)
    m.save_pretrained(DST); print("saved ->", DST, flush=True)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(SRC)
    ids=tok("The capital of France is", return_tensors="pt").input_ids
    out=m.prefill_decoder(input_ids=ids, cache_position=torch.arange(ids.shape[1]).unsqueeze(0)) \
        if hasattr(m,"prefill_decoder") else m(input_ids=ids)
    hs=getattr(out,"hidden_states",None)
    print("hidden_states:", "None" if hs is None else f"{len(hs)} tensors, first={tuple(hs[0].shape)}", flush=True)
    if hs is not None:
        print("target_layer_ids", TGT_IDS, "-> 사용 가능:", all(i < len(hs) for i in TGT_IDS), flush=True)
    print("TARGET_RESULT "+json.dumps({"ok":True,"n_hidden":0 if hs is None else len(hs)}), flush=True)
except Exception as e:
    print("FAILED", flush=True); traceback.print_exc()
    print("TARGET_RESULT "+json.dumps({"ok":False,"err":str(e)[:400]}), flush=True)

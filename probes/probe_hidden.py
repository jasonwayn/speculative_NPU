import torch, inspect, traceback, json
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer
DST="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden"
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
m = RBLNQwen3ForCausalLM.from_pretrained(DST, export=False)
print("loaded. output_hidden_states =", m.rbln_config.output_hidden_states, flush=True)
print("forward sig:", inspect.signature(m.forward), flush=True)
tok = AutoTokenizer.from_pretrained(SRC)
ids = tok("The capital of France is", return_tensors="pt").input_ids
L = ids.shape[1]; print("input len", L, flush=True)

def show(tag, out):
    hs = getattr(out, "hidden_states", None)
    lg = getattr(out, "logits", None)
    print(f"  [{tag}] logits={None if lg is None else tuple(lg.shape)}  "
          f"hidden={'None' if hs is None else f'{len(hs)} x {tuple(hs[0].shape)}'}", flush=True)
    return hs

hs=None
for tag, fn in [
    ("generate", lambda: m.generate(ids, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)),
    ("forward+cache_position", lambda: m(input_ids=ids, cache_position=torch.arange(L).unsqueeze(0))),
    ("forward+attn_mask", lambda: m(input_ids=ids, attention_mask=torch.ones_like(ids))),
]:
    try:
        hs = show(tag, fn()) or hs
    except Exception as e:
        print(f"  [{tag}] FAIL: {type(e).__name__}: {str(e)[:160]}", flush=True)

if hs is not None:
    TGT=[1,9,17,25,33]
    print(f"\nhidden layers = {len(hs)}, target_layer_ids {TGT} 사용가능 = {all(i<len(hs) for i in TGT)}", flush=True)
    sel = torch.cat([hs[i] for i in TGT], dim=-1)
    print("concat(5 layers) shape =", tuple(sel.shape), " (드래프트 fc 입력 5*2560=12800 이어야 함)", flush=True)
    print("PROBE_RESULT "+json.dumps({"ok":True,"n_layers":len(hs),"concat":list(sel.shape)}), flush=True)
else:
    print("PROBE_RESULT "+json.dumps({"ok":False}), flush=True)

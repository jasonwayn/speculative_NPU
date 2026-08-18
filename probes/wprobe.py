"""CMR 점수 계산에 필요한 타깃 마지막 레이어 가중치만 안전텐서에서 꺼낼 수 있는지 확인."""
import json, os, torch
from safetensors import safe_open
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"
idx=json.load(open(os.path.join(SRC,"model.safetensors.index.json")))["weight_map"]
L=35  # 마지막 디코더 레이어 (0-indexed, num_hidden_layers=36)
want=[f"model.layers.{L}.input_layernorm.weight",
      f"model.layers.{L}.self_attn.q_proj.weight",
      f"model.layers.{L}.self_attn.k_proj.weight",
      f"model.layers.{L}.self_attn.q_norm.weight",
      f"model.layers.{L}.self_attn.k_norm.weight"]
tot=0
for w in want:
    f=idx.get(w)
    if f is None: print(f"MISSING {w}"); continue
    with safe_open(os.path.join(SRC,f),framework="pt") as h:
        t=h.get_slice(w); sh=t.get_shape()
    n=1
    for d in sh: n*=d
    tot+=n
    print(f"OK  {w:58s} {sh}")
print(f"total_params={tot/1e6:.1f}M  fp16={tot*2/1e6:.1f}MB")
cfg=json.load(open(os.path.join(SRC,"config.json")))
print("heads",cfg["num_attention_heads"],"kv_heads",cfg["num_key_value_heads"],
      "head_dim",cfg.get("head_dim"),"rope_theta",cfg.get("rope_theta"))

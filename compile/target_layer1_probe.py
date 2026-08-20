"""Compile target layer 1 with intermediate outputs for numerical diagnosis."""
import os

import rebel
import torch
import torch.nn as nn
from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.configuration_utils import RBLNCompileConfig
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    slice_and_unsqueeze_cos_sin,
)
from transformers import AutoConfig, AutoModelForCausalLM


SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
OUT = "/home/work/npu_work/dflash_work/diag_target_layer1_probe_v3.rbln"
MAXC = 4096
CHUNK = 256

model_config = AutoConfig.from_pretrained(SRC)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32).eval()
cls = RBLNQwen3ForCausalLM
rc, _ = cls.prepare_rbln_config(rbln_config={
    "batch_size": 1,
    "max_seq_len": MAXC,
    "prefill_chunk_size": CHUNK,
    "output_hidden_states": True,
    "create_runtimes": False,
})
rc.max_seq_len = MAXC
rc = cls._update_rbln_config(
    preprocessors=None, model=hf, model_config=model_config, rbln_config=rc
)
wrapped = cls._wrap_model_if_needed(hf, rc).eval()
wrapped.phase = "prefill"


class LayerProbe(nn.Module):
    def __init__(self, wrapped_model):
        super().__init__()
        self.layer = wrapped_model.model.model.layers[1]
        self.rotary_emb = wrapped_model.rotary_emb

    def forward(self, hidden, cache_position, block_tables, past_key, past_value):
        cos, sin = self.rotary_emb(hidden, MAXC)
        cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, cache_position)
        pre_norm = self.layer.get_pre_attention_layernorm()(hidden)
        past = [[past_key, past_value], [past_key, past_value]]
        attention = self.layer.self_attn(
            hidden_states=pre_norm,
            attention_mask=None,
            seq_positions=cache_position[:, :1],
            past_key_values=past,
            cos=cos,
            sin=sin,
            block_tables=block_tables,
        )
        post_attention = hidden + attention
        post_norm = self.layer.get_post_attention_layernorm()(post_attention)
        mlp_module = self.layer.get_mlp()
        gate = mlp_module.gate_proj(post_norm)
        up = mlp_module.up_proj(post_norm)
        activated = gate / (1.0 + torch.exp(-torch.clamp(gate, -20.0, 20.0)))
        fused = activated * up
        mlp = mlp_module.down_proj(fused)
        output = post_attention + mlp
        return (
            pre_norm,
            attention,
            post_attention,
            post_norm,
            gate,
            up,
            activated,
            fused,
            mlp,
            output,
        )


probe = LayerProbe(wrapped).eval()
input_info = [
    ("hidden", [1, CHUNK, model_config.hidden_size], "float32"),
    ("cache_position", [1, CHUNK], "int32"),
    ("block_tables", [1], "int16"),
    ("past_key_values_0", [1, model_config.num_key_value_heads, MAXC, model_config.head_dim], "float32"),
    ("past_key_values_1", [1, model_config.num_key_value_heads, MAXC, model_config.head_dim], "float32"),
]
cc = RBLNCompileConfig(compiled_model_name="target_layer1_probe", input_info=input_info)
examples = cc.get_dummy_inputs(fill=0)
context, _ = cls._get_compile_context(cc, examples)
compiled = rebel.compile_from_torch(
    probe,
    input_info=input_info,
    example_inputs=examples,
    compile_context=context,
)
compiled.save(OUT)
print("LAYER1_PROBE_COMPILE_OK", OUT, flush=True)

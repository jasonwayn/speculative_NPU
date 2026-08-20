"""Compile shared-cache target graphs with fused all-position verify logits."""
import os
import time
import traceback

import torch
import torch.nn as nn
import optimum.rbln.transformers.models.decoderonly.configuration_decoderonly as CF

_Config = CF.RBLNDecoderOnlyModelForCausalLMConfig
_orig_init = _Config.__init__


def _init(self, *args, **kwargs):
    requested = kwargs.get("prefill_chunk_size")
    if requested is not None and requested % 64:
        kwargs = dict(kwargs)
        kwargs["prefill_chunk_size"] = 64
        _orig_init(self, *args, **kwargs)
        self.prefill_chunk_size = requested
    else:
        _orig_init(self, *args, **kwargs)


_Config.__init__ = _init

from optimum.rbln import RBLNQwen3ForCausalLM
from optimum.rbln.configuration_utils import RBLNCompileConfig
from transformers import AutoConfig, AutoModelForCausalLM

SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
MAXC = int(os.environ.get("MAXC", "4096"))
TP = int(os.environ.get("TP", "1"))
OUTDIR = os.environ.get(
    "OUTDIR", "/home/work/npu_work/dflash_work/fused_17_256"
)
CHUNKS = tuple(int(value) for value in os.environ.get("CHUNKS", "17,256").split(","))
KEEP = tuple(int(value) for value in os.environ.get("KEEP", "2,10,18,26,34").split(","))
os.makedirs(OUTDIR, exist_ok=True)


class AccurateSilu(nn.Module):
    def forward(self, value):
        return value / (1.0 + torch.exp(-torch.clamp(value, -20.0, 20.0)))


class FusedOutputs(nn.Module):
    def __init__(self, wrapped, keep, all_position_logits):
        super().__init__()
        self.wrapped = wrapped
        self.keep = keep
        self.all_position_logits = all_position_logits

    @property
    def phase(self):
        return self.wrapped.phase

    @phase.setter
    def phase(self, value):
        self.wrapped.phase = value

    def forward(self, *args):
        if not self.all_position_logits:
            logits, hidden_states = self.wrapped(*args)
            return logits, tuple(hidden_states[i] for i in self.keep)

        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            rotary_emb,
        ) = self.wrapped.prepare_forward_args(*args)
        hidden, hidden_states = self.wrapped.model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=True,
        )
        logits = self.wrapped.model.lm_head(hidden)
        # Keep query_position in the compiled input contract expected by
        # RBLNRuntimeModel. Match the hidden-state output shape and dtype so the
        # stock runtime can allocate its output buffer normally.
        contract = torch.ones_like(hidden_states[self.keep[0]]) * query_position.to(
            hidden_states[self.keep[0]].dtype
        ).reshape(1, 1, 1)
        return logits, tuple(hidden_states[i] for i in self.keep) + (contract,)


cls = RBLNQwen3ForCausalLM
mcfg = AutoConfig.from_pretrained(SRC)
print("loading target", flush=True)
MODEL_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}[os.environ.get("MODEL_DTYPE", "float32")]
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=MODEL_DTYPE)
if os.environ.get("ACCURATE_SILU") == "1":
    for decoder_layer in hf.model.layers:
        decoder_layer.mlp.act_fn = AccurateSilu()
    print("ACCURATE_SILU enabled layers=%d" % len(hf.model.layers), flush=True)
rc, _ = cls.prepare_rbln_config(rbln_config={
    "batch_size": 1,
    "max_seq_len": MAXC,
    "prefill_chunk_size": CHUNKS[0],
    "output_hidden_states": True,
    "create_runtimes": False,
    "tensor_parallel_size": TP,
})
rc.max_seq_len = MAXC
rc = cls._update_rbln_config(
    preprocessors=None, model=hf, model_config=mcfg, rbln_config=rc
)
base = cls._wrap_model_if_needed(hf, rc)

ctx = None
static = None
for chunk in CHUNKS:
    dst = os.path.join(OUTDIR, "prefill_%d.rbln" % chunk)
    info = cls.get_input_info(
        batch_size=1, query_length=chunk, rbln_config=rc, model_config=mcfg
    )
    cc = RBLNCompileConfig(
        compiled_model_name="prefill_%d" % chunk,
        input_info=info,
        tensor_parallel_size=TP,
    )
    meta = [name for name, _, _ in cc.input_info if "past_key_values" in name]
    if ctx is None:
        examples = cc.get_dummy_inputs(fill=0, meta_tensor_names=meta)
        ctx, static = cls._get_compile_context(cc, examples)
    else:
        examples = cc.get_dummy_inputs(fill=0, static_tensors=static)
    wrapped = FusedOutputs(base, KEEP, all_position_logits=(chunk == 17))
    started = time.time()
    try:
        compiled = cls._compile_model(wrapped, cc, examples, ctx, rc, phase="prefill")
        compiled.save(dst)
        print("COMPILE_OK chunk=%d sec=%.1f path=%s" % (
            chunk, time.time() - started, dst
        ), flush=True)
    except Exception as exc:
        print("COMPILE_FAIL chunk=%d %s: %s" % (
            chunk, type(exc).__name__, str(exc)[:500]
        ), flush=True)
        traceback.print_exc()
        raise
print("FUSED_COMPILE_DONE", flush=True)

"""Print the installed RBLN Qwen3 implementation and compiled graph contract."""
import inspect
import os
import pkgutil

import rebel
import optimum.rbln
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM


ROOT = os.path.dirname(inspect.getfile(optimum.rbln))
print("OPTIMUM_RBLN_ROOT", ROOT)
print("REBEL_VERSION", getattr(rebel, "__version__", "unknown"))

model_config = AutoConfig.from_pretrained("/home/work/npu_work/eagle_test/Qwen3-4B")
rbln_config, _ = RBLNQwen3ForCausalLM.prepare_rbln_config(rbln_config={
    "batch_size": 1,
    "max_seq_len": 4096,
    "prefill_chunk_size": 256,
    "output_hidden_states": True,
    "create_runtimes": False,
})
print("RBLN_CONFIG_PREPARED", vars(rbln_config))
hf = AutoModelForCausalLM.from_pretrained(
    "/home/work/npu_work/eagle_test/Qwen3-4B", dtype="auto"
).eval()
rbln_config = RBLNQwen3ForCausalLM._update_rbln_config(
    preprocessors=None,
    model=hf,
    model_config=model_config,
    rbln_config=rbln_config,
)
print("RBLN_CONFIG_UPDATED", vars(rbln_config))

for module in pkgutil.walk_packages([ROOT], prefix="optimum.rbln."):
    name = module.name.lower()
    if "qwen3" in name or name.endswith("decoderonly.modeling_decoderonly"):
        try:
            imported = __import__(module.name, fromlist=["_"])
            print("MODULE", module.name, inspect.getfile(imported))
            for symbol, value in vars(imported).items():
                lowered = symbol.lower()
                if inspect.isclass(value) and any(
                    part in lowered for part in ("attention", "decoderlayer", "rmsnorm")
                ):
                    try:
                        print("SOURCE_BEGIN", module.name, symbol)
                        print(inspect.getsource(value))
                        print("SOURCE_END", module.name, symbol)
                    except (OSError, TypeError):
                        pass
        except Exception as error:
            print("MODULE_ERROR", module.name, type(error).__name__, str(error))

graph = rebel.RBLNCompiledModel(
    "/home/work/npu_work/dflash_work/diag_bf16_layers_0_9/prefill_256.rbln"
)
print("GRAPH_TYPE", type(graph))
for name in dir(graph):
    if any(part in name.lower() for part in ("input", "output", "dtype", "tensor")):
        try:
            value = getattr(graph, name)
            if not callable(value):
                print("GRAPH_ATTR", name, repr(value))
        except Exception as error:
            print("GRAPH_ATTR_ERROR", name, type(error).__name__, str(error))

keywords = ("precision", "bfloat16", "float16", "amp", "accum")
for base, _, files in os.walk(ROOT):
    for filename in files:
        if not filename.endswith(".py"):
            continue
        path = os.path.join(base, filename)
        try:
            lines = open(path, errors="ignore").read().splitlines()
        except OSError:
            continue
        matches = [
            "%d:%s" % (index, line.strip())
            for index, line in enumerate(lines, 1)
            if any(keyword in line.lower() for keyword in keywords)
        ]
        if matches:
            print("PRECISION_FILE", path)
            print("\n".join(matches[:80]))

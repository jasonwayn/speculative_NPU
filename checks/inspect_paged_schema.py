import inspect
import torch
import optimum.rbln


operation = torch.ops.rbln_custom_ops.paged_causal_attn_prefill
print("SCHEMAS", operation._schemas)
print("PACKET", operation)

try:
    import optimum.rbln.ops
    print("OPS_MODULE", inspect.getfile(optimum.rbln.ops))
except Exception as error:
    print("OPS_MODULE_ERROR", repr(error))

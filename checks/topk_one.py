"""케이스 하나만 별도 프로세스에서. 세그폴트가 나도 다른 케이스에 영향 없게."""
import os, sys, torch, rebel
import torch.nn as nn

CASE = sys.argv[1]
N, K = 640, 32


class M(nn.Module):
    def forward(s, x):
        if CASE == "topk_idx":   return torch.topk(x, K, dim=-1).indices.to(torch.int32)
        if CASE == "topk_val":   return torch.topk(x, K, dim=-1).values
        if CASE == "argmax":     return torch.argmax(x, dim=-1).to(torch.int32)
        if CASE == "sort_idx":   return torch.sort(x, dim=-1, descending=True).indices.to(torch.int32)
        if CASE == "kth_mask":                       # 인덱스 없이 마스크만
            kth = torch.topk(x, K, dim=-1).values[..., -1:]
            return (x >= kth).to(torch.int32)
        if CASE == "max_val":    return torch.max(x, dim=-1).values
        if CASE == "cumsum":     return torch.cumsum(x, dim=-1)   # 대조군 (평범한 op)
        raise SystemExit("unknown case")


info = [("x", [1, N], "float32")]
try:
    cm = rebel.compile_from_torch(M().eval(), input_info=info,
                                  example_inputs=[torch.zeros(1, N)])
except Exception as e:
    print("RESULT %-10s compile=FAIL  %s" % (CASE, repr(e)[:150]), flush=True); raise SystemExit(0)
try:
    rt = rebel.Runtime(cm, tensor_type="pt", device=int(os.environ.get("DEV", "0")))
    x = torch.randn(1, N)
    o = rt(x.contiguous())
    o = torch.as_tensor(o[0] if isinstance(o, (list, tuple)) else o)
    ok = ""
    if CASE == "topk_idx":
        ref = set(torch.topk(x, K, -1).indices.flatten().tolist())
        ok = "  값일치" if set(o.flatten().tolist()) == ref else "  값불일치"
    if CASE == "argmax":
        ok = "  값일치" if int(o.flatten()[0]) == int(torch.argmax(x, -1)) else "  값불일치"
    print("RESULT %-10s compile=OK run=OK out=%s%s" % (CASE, tuple(o.shape), ok), flush=True)
except Exception as e:
    print("RESULT %-10s compile=OK run=FAIL  %s" % (CASE, repr(e)[:150]), flush=True)

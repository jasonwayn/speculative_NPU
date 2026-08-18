import torch, torch.nn as nn, rebel, traceback
V=151936; H=2560; B=16
x=torch.randn(1,B,V)
cases={
 "argmax":        lambda t: torch.argmax(t,dim=-1).to(torch.int32),
 "max.indices":   lambda t: torch.max(t,dim=-1).indices.to(torch.int32),
 "topk1.indices": lambda t: torch.topk(t,1,dim=-1).indices.to(torch.int32),
 "max.values":    lambda t: torch.max(t,dim=-1).values,
 "amax":          lambda t: torch.amax(t,dim=-1),
}
for name,fn in cases.items():
    class M(nn.Module):
        def __init__(s,f): super().__init__(); s.f=f
        def forward(s,t): return s.f(t)
    try:
        rebel.compile_from_torch(M(fn).eval(), input_info=[("t",[1,B,V],"float32")])
        print(f"  {name:16s} OK",flush=True)
    except Exception as e:
        print(f"  {name:16s} FAIL {str(e)[:70]}",flush=True)

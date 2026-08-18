import torch, torch.nn as nn, rebel
V=151936; H=2560; B=16
W=torch.randn(V,H)*0.02
def mk(fn,dt):
    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.l=nn.Linear(H,V,bias=False)
            with torch.no_grad(): s.l.weight.copy_(W)
            s.fn=fn
        def forward(s,x): return s.fn(s.l(x))
    return M().eval()
cases={
 "Linear+argmax":  lambda t: torch.argmax(t,dim=-1).to(torch.int32),
 "Linear+topk1":   lambda t: torch.topk(t,1,dim=-1).indices.to(torch.int32),
 "Linear+amax":    lambda t: torch.amax(t,dim=-1),
 "Linear only":    lambda t: t,
}
for name,fn in cases.items():
    try:
        rebel.compile_from_torch(mk(fn,"float32"), input_info=[("x",[1,B,H],"float32")])
        print(f"  {name:16s} OK",flush=True)
    except Exception as e:
        print(f"  {name:16s} FAIL {str(e)[:60]}",flush=True)

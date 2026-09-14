# Activation forward/backward (relu, tanh, sigmoid, relu6) on ocl:0 vs CPU, in
# float32 and float16. Expect OK on every line; the half rows were wrong before
# the dtype define reached the activation kernel (OPENCL-PERF.md, Finding 11).
import torch, pytorch_ocl  # noqa
d='ocl:0'
x=torch.randn(1000,257)
for act in ['relu','tanh','sigmoid','relu6']:
    f={'relu':torch.relu,'tanh':torch.tanh,'sigmoid':torch.sigmoid,'relu6':torch.nn.functional.relu6}[act]
    for dt in [torch.float32, torch.float16]:
        xc=x.to(dt).detach().requires_grad_(True); y=f(xc); y.float().sum().backward()
        xo=x.to(d).to(dt).detach().requires_grad_(True); yo=f(xo); yo.float().sum().backward(); torch.ocl.synchronize()
        ef=((yo.detach().cpu().float()-y.detach().float()).abs().max()).item(); eb=((xo.grad.cpu().float()-xc.grad.float()).abs().max()).item()
        print(f"{act:8s} {str(dt):14s} fwd max abs err {ef:.1e}  bwd {eb:.1e}", "OK" if max(ef,eb)<2e-3 else "BAD")

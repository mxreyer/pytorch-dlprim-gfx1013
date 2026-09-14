# conv2d Y/dW/dX relative error vs CPU fp32 for the ResNet-9 shapes; run with
# and without DLPRIM_CONV_FP16=1 to see what the fp16 inner loop costs (Finding 11).
import torch, pytorch_ocl  # noqa
for B,ci,co,H in [(128,64,128,32),(128,128,128,16),(128,128,256,16),(128,256,512,8),(128,512,512,4)]:
    g=torch.Generator().manual_seed(ci+co)
    x=torch.randn(B,ci,H,H,generator=g); w=torch.randn(co,ci,3,3,generator=g)*(1.0/(ci*9))**0.5; gy=torch.randn(B,co,H,H,generator=g)
    xc=x.clone().requires_grad_(True); wc=w.clone().requires_grad_(True); y=torch.nn.functional.conv2d(xc,wc,padding=1); y.backward(gy)
    xo=x.to('ocl:0').requires_grad_(True); wo=w.to('ocl:0').requires_grad_(True); yo=torch.nn.functional.conv2d(xo,wo,padding=1); yo.backward(gy.to('ocl:0')); torch.ocl.synchronize()
    def rel(a,b): d=(a.cpu().double()-b.double()).abs(); return f"max {d.max().item()/b.abs().max().item():.1e} mean {d.mean().item()/b.abs().mean().item():.1e}"
    print(f"{ci}->{co}@{H}:  Y {rel(yo.detach(),y.detach())} | dX {rel(xo.grad,xc.grad)} | dW {rel(wo.grad,wc.grad)}")

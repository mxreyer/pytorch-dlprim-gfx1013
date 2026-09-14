# Backward-data planes on awkward shapes: odd spatial sizes, 1-2 pixel images,
# and gradient accumulation (beta=1). Exercises the border masks in
# winconv_3x3_bwd_data_reduce. Expect OK on every line.
import torch, pytorch_ocl  # noqa
# odd spatial sizes and gradient accumulation (beta=1) exercise the plane border masks
for B,ci,co,H,W in [(4,16,32,7,9),(4,32,32,5,5),(2,64,64,1,3),(8,32,64,31,33),(3,32,32,2,2)]:
    x=torch.randn(B,ci,H,W,requires_grad=True); w=torch.randn(co,ci,3,3,requires_grad=True); gy=torch.randn(B,co,H,W)
    for _ in range(2):
        torch.nn.functional.conv2d(x,w,padding=1).backward(gy)
    xo=x.detach().to('ocl:0').requires_grad_(True); wo=w.detach().to('ocl:0').requires_grad_(True)
    for _ in range(2):
        torch.nn.functional.conv2d(xo,wo,padding=1).backward(gy.to('ocl:0'))
    torch.ocl.synchronize()
    e=((xo.grad.cpu()-x.grad).abs().max()/x.grad.abs().max()).item()
    ew=((wo.grad.cpu()-w.grad).abs().max()/w.grad.abs().max()).item()
    print(f"{ci}->{co}@{H}x{W}: dX {e:.1e} dW {ew:.1e}", "OK" if max(e,ew)<1e-4 else "BAD")

# BatchNorm forward+backward on ocl:0 vs CPU for the ResNet-9 activation shapes.
import torch, pytorch_ocl  # noqa
shapes=[(128,64,32,32),(128,128,32,32),(128,128,16,16),(128,256,16,16),(128,512,8,8),(128,512,4,4),(128,3,32,32),(128,1024,4,4)]
worst=0
for B,C,H,W in shapes:
    g=torch.Generator().manual_seed(C+H)
    x=torch.randn(B,C,H,W,generator=g)*2+1; gy=torch.randn(B,C,H,W,generator=g)
    bn=torch.nn.BatchNorm2d(C); bn.weight.data.uniform_(0.5,1.5,generator=g); bn.bias.data.normal_(generator=g)
    # copy the state BEFORE the CPU forward, or the GPU copy starts from
    # already-updated running stats and gets "two updates" (an earlier version
    # of this script did exactly that and reported a running-stats bug that
    # did not exist)
    bno=torch.nn.BatchNorm2d(C).to('ocl:0'); bno.load_state_dict(bn.state_dict())
    xc=x.clone().requires_grad_(True); y=bn(xc); y.backward(gy)
    xo=x.to('ocl:0').requires_grad_(True); yo=bno(xo); yo.backward(gy.to('ocl:0')); torch.ocl.synchronize()
    def err(a,b): return ((a.cpu().double()-b.double()).abs().max()/b.double().abs().max()).item()
    es={"y":err(yo.detach(),y.detach()),"dx":err(xo.grad,xc.grad),"dw":err(bno.weight.grad,bn.weight.grad),"db":err(bno.bias.grad,bn.bias.grad),"rmean":err(bno.running_mean,bn.running_mean),"rvar":err(bno.running_var,bn.running_var)}
    e=max(es["y"],es["dx"],es["dw"],es["db"]); worst=max(worst,e); print(f"{B}x{C}x{H}x{W}: " + " ".join(f"{k}={v:.1e}" for k,v in es.items()))
print("WORST", f"{worst:.2e}", "OK" if worst < 1e-4 else "BAD")

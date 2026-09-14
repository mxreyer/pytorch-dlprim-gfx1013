# BatchNorm running_mean/running_var after exactly one training call, ocl vs CPU vs
# the momentum formula. Exists because a test once got the state-copy order wrong
# and reported a running-stats bug that was not there.
import torch, pytorch_ocl  # noqa
torch.manual_seed(0)
x=torch.randn(16,4,8,8)*2+3
mean=x.mean((0,2,3)); var_unb=x.var((0,2,3),unbiased=True); var_b=x.var((0,2,3),unbiased=False)
for dev in ['cpu','ocl:0']:
    rm=torch.zeros(4,device=dev); rv=torch.ones(4,device=dev)
    y=torch.nn.functional.batch_norm(x.to(dev),rm,rv,training=True,momentum=0.1)
    torch.ocl.synchronize() if dev!='cpu' else None
    print(dev, "running_mean", rm.cpu().numpy().round(4), "running_var", rv.cpu().numpy().round(4))
print("expected  running_mean", (0.1*mean).numpy().round(4), "running_var", (0.9+0.1*var_unb).numpy().round(4))
print("if 2 updates:         ", (0.19*mean).numpy().round(4), (0.81+0.19*var_unb).numpy().round(4))
print("biased var variant:   ", (0.9+0.1*var_b).numpy().round(4))
# and via the module
for dev in ['cpu','ocl:0']:
    bn=torch.nn.BatchNorm2d(4).to(dev); bn(x.to(dev)); torch.ocl.synchronize() if dev!='cpu' else None
    print(dev, "module: running_mean", bn.running_mean.cpu().numpy().round(4), "running_var", bn.running_var.cpu().numpy().round(4), "nbt", bn.num_batches_tracked.item())

#!/usr/bin/env python3
"""
Per-kernel profile of ResNet-9 training steps on ocl:0, using pytorch_ocl's
built-in profiler. Needs torch + pytorch_ocl (build.sh makes a venv in
scratch/venv):

  RUSTICL_ENABLE=radeonsi scratch/venv/bin/python tools/profile-step.py 128 20        # wall clock only
  RUSTICL_ENABLE=radeonsi scratch/venv/bin/python tools/profile-step.py 128 10 prof   # + prof.csv (or a 4th arg)

Profiling forces per-kernel events and stretches the step by ~50%, so take the
wall-clock number from an unprofiled run and only the breakdown from prof.csv
(columns: section, kernel, start, end, duration in ms). This is the harness
behind the per-patch img/s table in README.md. Synthetic data, so no CPU-side
augmentation - this measures the GPU step alone.

MIT License, Copyright (c) 2026 mxreyer.
"""
import sys, collections, csv, torch, pytorch_ocl  # noqa
from torch import nn


# ResNet-9 exactly as in the benchmark the img/s numbers come from:
# https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s/blob/main/benchmark.py
def conv_bn(ci, co, pool=False):
    layers = [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(conv_bn(c, c), conv_bn(c, c))

    def forward(self, x):
        return x + self.conv(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(3, 64), conv_bn(64, 128, pool=True), Residual(128),
            conv_bn(128, 256, pool=True), conv_bn(256, 512, pool=True), Residual(512),
            nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.net(x) * 0.125


B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
dev = torch.device("ocl:0")
PROF = len(sys.argv) > 3 and sys.argv[3] == "prof"
OUT = sys.argv[4] if len(sys.argv) > 4 else "prof.csv"
if PROF: torch.ocl.enable_profiling(dev)
torch.manual_seed(0)
model = ResNet9().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
lossfn = torch.nn.CrossEntropyLoss()
x = torch.rand(B, 3, 32, 32, device=dev); y = torch.randint(0, 10, (B,), device=dev)
def step():
    loss = lossfn(model(x), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
for _ in range(5): step()
torch.ocl.synchronize()
import time
t0 = time.perf_counter()
for _ in range(N): step()
torch.ocl.synchronize(); wall = (time.perf_counter() - t0) / N
print(f"batch {B}: {'profiled' if PROF else 'unprofiled'} step {wall*1e3:.1f} ms  = {B/wall:.0f} img/s")
if PROF:
    with torch.ocl.profile(dev, OUT):
        for _ in range(N): step()
        torch.ocl.synchronize()
    rows = list(csv.reader(open(OUT)))
    print("columns:", rows[0], "rows:", len(rows), file=sys.stderr)

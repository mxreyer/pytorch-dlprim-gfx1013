#!/usr/bin/env python3
"""
Per-kernel profile of ResNet-9 training steps on ocl:0, using pytorch_ocl's
built-in profiler. Needs torch + pytorch_ocl (build.sh makes a venv in
scratch/venv):

  RUSTICL_ENABLE=radeonsi scratch/venv/bin/python tools/profile-step.py 128 20        # wall clock only
  RUSTICL_ENABLE=radeonsi scratch/venv/bin/python tools/profile-step.py 128 10 prof   # + prof.csv (or a 4th arg)

The same step runs on other devices for comparison, wall clock only; the
device comes from PROFILE_DEVICE (default ocl:0), and pytorch_ocl is only
needed for ocl. On an NVIDIA GPU, e.g. a Colab T4:

  PROFILE_DEVICE=cuda python tools/profile-step.py 128 50

PROFILE_MODE=infer times the forward pass alone, in eval mode under no_grad,
instead of the training step (default PROFILE_MODE=train).

Profiling forces per-kernel events and stretches the step by ~50%, so take the
wall-clock number from an unprofiled run and only the breakdown from prof.csv
(columns: section, kernel, start, end, duration in ms). This is the harness
behind the per-patch img/s table in README.md. Synthetic data, so no CPU-side
augmentation - this measures the GPU step alone.

MIT License, Copyright (c) 2026 mxreyer.
"""
import os, sys, collections, csv, time, torch
DEV = os.environ.get("PROFILE_DEVICE", "ocl:0")
MODE = os.environ.get("PROFILE_MODE", "train")
if MODE not in ("train", "infer"):
    sys.exit(f"PROFILE_MODE must be train or infer, not {MODE}")
if DEV.startswith("ocl"):
    import pytorch_ocl  # noqa: F401  (registers the "ocl" device)
from torch import nn


# ResNet-9, the model the img/s numbers in README.md are measured on.
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
dev = torch.device(DEV)
PROF = len(sys.argv) > 3 and sys.argv[3] == "prof"
if PROF and dev.type != "ocl":
    sys.exit("profiling uses pytorch_ocl's profiler; it only works on ocl")
if dev.type == "cuda":
    sync = torch.cuda.synchronize
elif dev.type == "ocl":
    sync = torch.ocl.synchronize
else:
    sync = lambda: None
OUT = sys.argv[4] if len(sys.argv) > 4 else "prof.csv"
if PROF: torch.ocl.enable_profiling(dev)
torch.manual_seed(0)
model = ResNet9().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
lossfn = torch.nn.CrossEntropyLoss()
x = torch.rand(B, 3, 32, 32, device=dev); y = torch.randint(0, 10, (B,), device=dev)
def train_step():
    loss = lossfn(model(x), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
def infer_step():
    with torch.no_grad(): model(x)
step = train_step if MODE == "train" else infer_step
if MODE == "infer": model.eval()
for _ in range(5): step()
sync()
t0 = time.perf_counter()
for _ in range(N): step()
sync(); wall = (time.perf_counter() - t0) / N
print(f"{dev} batch {B}: {'profiled' if PROF else 'unprofiled'} {MODE} step {wall*1e3:.1f} ms  = {B/wall:.0f} img/s")
if PROF:
    with torch.ocl.profile(dev, OUT):
        for _ in range(N): step()
        torch.ocl.synchronize()
    rows = list(csv.reader(open(OUT)))
    print("columns:", rows[0], "rows:", len(rows), file=sys.stderr)

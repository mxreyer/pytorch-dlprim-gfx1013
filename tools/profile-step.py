#!/usr/bin/env python3
"""
Per-kernel profile of ResNet-9 training steps on ocl:0, using pytorch_ocl's
built-in profiler. Meant to run inside the notebook image with benchmark.py
mounted at /b and an output directory at /out:

  docker run --rm --device /dev/dri/renderD128 --group-add 105 \
    -v $PWD/benchmark.py:/b/benchmark.py:ro -v $PWD/tools:/t -v /tmp/prof:/out -w /t \
    bc250-notebook:latest python profile-step.py 128 20          # wall clock only
    ... python profile-step.py 128 10 prof                          # + /out/prof.csv

Profiling forces per-kernel events and stretches the step by ~50%, so take the
wall-clock number from an unprofiled run and only the breakdown from prof.csv
(columns: section, kernel, start, end, duration in ms). Used for OPENCL-PERF.md
Finding 10; see there for how to read it. Synthetic data, so no CPU-side
augmentation - this measures the GPU step alone.

MIT License, Copyright (c) 2026 mxreyer.
"""
import sys, importlib.util, collections, csv, torch, pytorch_ocl  # noqa
spec = importlib.util.spec_from_file_location("bench", "/b/benchmark.py"); bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
dev = torch.device("ocl:0")
PROF = len(sys.argv) > 3 and sys.argv[3] == "prof"
if PROF: torch.ocl.enable_profiling(dev)
torch.manual_seed(0)
model = bench.ResNet9().to(dev)
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
    with torch.ocl.profile(dev, "/out/prof.csv"):
        for _ in range(N): step()
        torch.ocl.synchronize()
    rows = list(csv.reader(open("/out/prof.csv")))
    print("columns:", rows[0], "rows:", len(rows), file=sys.stderr)

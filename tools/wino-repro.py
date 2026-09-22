#!/usr/bin/env python3
"""
Correctness sweep for dlprimitives' Winograd backward kernels on ocl:0.

Runs the six ResNet-9 3x3 layer shapes forward+backward N times and compares
Y/dW/dX against cached CPU references. The atomics-free backward paths are the
default since 2026-09-10, so this is the check that they, and anything built on
them, still produce the gradients the CPU does.

  # what ships. Expect 0 bad.
  RUSTICL_ENABLE=radeonsi python3 tools/wino-repro.py 300

  # the emulated-atomics kernels instead, for comparison. Expect 0 bad.
  RUSTICL_ENABLE=radeonsi DLPRIM_WINOGRAD_SPLIT_PLANES=0 \
      DLPRIM_WINOGRAD_BWD_PLANES=0 python3 tools/wino-repro.py 300

Needs a torch + pytorch_ocl environment (build.sh at the top of this repository
makes one in scratch/venv). CPU reference gradients are computed once and
cached next to this script. ~1 minute per 100 sweeps.

MIT License, Copyright (c) 2026 mxreyer.
"""
import os, sys, torch, pytorch_ocl  # noqa: F401  (pytorch_ocl registers "ocl")

D = 'ocl:0'
# The ResNet-9 3x3 layer shapes, as (batch, in, out, HxW).
SHAPES = [(128, 3, 64, 32), (128, 64, 128, 32), (128, 128, 128, 16),
          (128, 128, 256, 16), (128, 256, 512, 8), (128, 512, 512, 4)]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wino-repro-refs2.pt')
TOL = 2e-4          # generous: real failures are ~1e7 relative, not marginal


def references():
    if os.path.exists(CACHE):
        return torch.load(CACHE, weights_only=False)
    print("computing CPU reference gradients once ...", flush=True)
    out = []
    for i, (B, ci, co, H) in enumerate(SHAPES):
        g = torch.Generator().manual_seed(100 + i)
        x = torch.randn(B, ci, H, H, generator=g) * 0.5
        w = torch.randn(co, ci, 3, 3, generator=g) * 0.1
        gy = torch.randn(B, co, H, H, generator=g) * 0.5
        xc = x.clone().requires_grad_(True)
        wc = w.clone().requires_grad_(True)
        y = torch.nn.functional.conv2d(xc, wc, padding=1)
        y.backward(gy)
        out.append((x, w, gy, wc.grad.clone(), xc.grad.clone(), y.detach().clone()))
    torch.save(out, CACHE)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    data = references()
    bad = {}
    for _ in range(n):
        for si, (B, ci, co, H) in enumerate(SHAPES):
            x, w, gy, dwref, dxref, yref = data[si]
            # forward output and both gradients
            xo = x.clone().to(D).requires_grad_(True)
            wo = w.clone().to(D).requires_grad_(True)
            yo = torch.nn.functional.conv2d(xo, wo, padding=1)
            yo.backward(gy.clone().to(D))
            torch.ocl.synchronize()
            for nm, got, ref in (("Y", yo.detach().cpu(), yref),
                                 ("dW", wo.grad.cpu(), dwref),
                                 ("dX", xo.grad.cpu(), dxref)):
                e = ((got.double() - ref.double()).abs().max().item()
                     / ref.double().abs().max().item())
                if e > TOL:
                    key = f"{nm} {ci}->{co}@{H}"
                    bad[key] = bad.get(key, 0) + 1
            del xo, wo, yo
    total = sum(bad.values())
    print(f"{total} bad gradients over {n} sweeps" + (f"  -> {bad}" if bad else ""))
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())

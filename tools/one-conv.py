#!/usr/bin/env python3
"""
Run one 3x3 convolution forward+backward on ocl:0 for a given layer shape, so
that the driver compiles exactly the kernels involved. Made for ISA dumps:

  AMD_DEBUG=cs,asm MESA_SHADER_CACHE_DISABLE=true \
      RUSTICL_ENABLE=radeonsi python3 tools/one-conv.py 128 256 512 8 2> isa.log

Mesa needs BOTH a stage flag ("cs") and a type flag ("asm", "nir", "aco" ...) in
AMD_DEBUG, and the shader cache must be off or nothing is compiled and nothing
is printed. Kernels with "LDS: 40960 bytes" in their stats are the Winograd
ones; they appear in the order forward, backward-data, backward-filter, each
twice (two radeonsi variants). The dumps were made to read the Winograd
kernels' register and LDS use, and to rule the compiler out of a wrong-gradient
bug that turned out to be the GPU's idle-floor voltage.

Arguments: batch channels_in channels_out spatial (default 128 256 512 8).

MIT License, Copyright (c) 2026 mxreyer.
"""
import sys, torch, pytorch_ocl  # noqa: F401  (pytorch_ocl registers "ocl")

B, ci, co, H = (int(x) for x in (sys.argv[1:5] or (128, 256, 512, 8)))
g = torch.Generator().manual_seed(0)
x = (torch.randn(B, ci, H, H, generator=g) * 0.5).to('ocl:0').requires_grad_(True)
w = (torch.randn(co, ci, 3, 3, generator=g) * 0.1).to('ocl:0').requires_grad_(True)
gy = (torch.randn(B, co, H, H, generator=g) * 0.5).to('ocl:0')
torch.nn.functional.conv2d(x, w, padding=1).backward(gy)
torch.ocl.synchronize()
print("dW", w.grad.abs().sum().item(), "dX", x.grad.abs().sum().item(), file=sys.stderr)

# dlprimitives: Winograd kernels — atomics-free backward paths, four access-pattern fixes, prefetch (2.1× on ResNet-9 training)

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues — as a
discussion first, then PRs
**Against:** `ff2d590` (2024-09-04); `src/core/conv.cpp`, `src/kernels/winograd_*.cl`, `src/kernels/bn_sums.cl`

## Summary

Profiling ResNet-9 training on an RDNA1 device (AMD BC-250, gfx1013, 40 CU,
Mesa rusticl) found that 89% of a step is convolution and 77% is the three
Winograd kernels. Five changes to them take the step from 972 img/s (after the
split-K fix filed separately) to 2,055, with gradients matching CPU references
over 900 sweeps of the six layer shapes. They are a stacked series of six
patches, one per item below, in `patches/03`–`08` at
https://github.com/mxreyer/pytorch-dlprim-gfx1013 (`git format-patch` output
with the commit messages), with the measurements in that repository's
OPENCL-PERF.md (Findings 3, 10, 11, 12). The per-section numbers below are
from the investigation, in the order the changes were found; re-measured
patch by patch on the stacked series the steps are 984 → 1,477 → 1,634 →
1,815 → 1,949 → 2,068 img/s (README there). I would like to know which of
these you want as PRs and how you would prefer them gated.

## 1. Backward kernels without `atomic_addf` (972 → 1,374 img/s) — `03-winograd-no-atomics`

Both backward kernels accumulate through `atomic_addf`. On any device without
a hardware fp32 atomic add that is the compare-and-swap loop in `atomic.h`,
and it was **39% of the backward-filter kernel** (measured by swapping it for
a racy plain `+=`). That is every RDNA1/RDNA2 GPU — `llc` lowers `atomicrmw
fadd` to `global_atomic_cmpswap` for gfx1010–gfx1036; `global_atomic_add_f32`
arrives with gfx11 — and the Intel path in `atomic.h` is also a CAS loop.

The atomics were only ever combining *across* K slices: within one slice each
output element has exactly one writer. So each slice writes its own plane of
the existing `workspace()` and a small kernel sums the planes (applying
`beta`); backward-data does the same with four parity planes (the overlapping
4×4 input-gradient tiles fall into four row/column parity classes whose members
are disjoint). The reduce costs under 2% of the kernel it replaces.

On NVIDIA and gfx11+ the atomics are free and the planes cost workspace plus a
reduce, so the patch selects the planes when the device is not NVIDIA and does
not advertise `cl_ext_float_atomics`, and keeps the atomic path otherwise.

## 2. Four access patterns (1,374 → 1,816 img/s, inference 4,433 → 6,418) — `04`, `05`, `06`

Each found by timing a wrong-but-cheap variant of one stage:

- `04-winograd-fwd-filter-layout`: the transformed filters are read at a `C × 64`-byte lane
  stride (39% of the kernel). Storing them `[C][N]` instead of `[N][C]` makes
  the 32 lanes that read one input channel for 32 output features contiguous.
  Forward Winograd 24.4 → 15.0 ms per step.
- `05-winograd-bwd-filter-loads`: lanes map to neighbouring tiles of one channel
  plane (`k = lid % 8`) rather than to 32 different planes; edge tiles use one
  bounded `vload4` per row with columns masked instead of a scalar path.
  27.0 → 19.4 ms.
- `06-bn-sums-grid-stride`: grid-stride reduction loop instead of one contiguous chunk per
  work-item, which for 512 channels at 8×8 put every second lane 128 KB apart
  and hit the same cache sets (13 GB/s). BatchNorm 9.6 → ~3 ms per step.

These are plain coalescing changes and should help everywhere, but they touch
the hottest kernels in the library and I have one device, so they need your
numbers on NVIDIA/Intel before merging.

## 3. Register prefetch (1,816 → 2,055 img/s) — `07-winograd-prefetch`

All three kernels issue the next K step's global loads before the current
step's GEMM and transform/store afterwards. Step 65.0 → 59.8 ms. LDS
double-buffering was measured and lost to occupancy; documented, not included.

## 4. Optional fp16 inner loop (2,055 → 2,987 img/s) — `08-winograd-fp16`

Behind `DLPRIM_CONV_FP16=1`: tiles converted to half on the way into LDS,
GEMM as `half2` fma with half accumulators, fp32 tensors in memory. Y/dX/dW
within ~0.5% of fp32 — about 10× looser than TF32, so opt-in. Included in case
you want it as an option; no argument for it being the default.

## Environment

AMD BC-250 (gfx1013, 40 CU, 2.0 GHz), Mesa 26.1.8 and 26.2.2 rusticl on
radeonsi, LLVM 22.1.8; `pytorch_dlprim` `1af48d4` with `dlprimitives`
`ff2d590`. Correctness: `tools/wino-repro.py` (Y, dW, dX vs CPU over the six
ResNet-9 shapes), `tools/bn-check.py`, `tools/odd-shapes.py` in the repository
above.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

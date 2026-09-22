# dlprimitives: Winograd kernels — atomics-free backward paths, four access-pattern fixes, prefetch (2.1× on ResNet-9 training)

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues — as a
discussion first, then PRs
**Against:** `ff2d590` (2024-09-04; the patch also applies to master `b176c15`); `src/core/conv.cpp`, `src/kernels/winograd_*.cl`, `src/kernels/bn_sums.cl`

## Summary

Profiling ResNet-9 training on an AMD BC-250 (gfx1013, 40 CU, Mesa rusticl)
found that 89% of a step is convolution and 77% is the three Winograd
kernels. Five changes to them take the step from 972 img/s (after the
split-K fix filed separately) to 2,055, with gradients matching CPU references
over 900 sweeps of the six layer shapes. They are a stacked series of five
patches, one per item below, in `patches/dlprimitives/04`–`08` at
https://github.com/mxreyer/pytorch-dlprim-gfx1013 (`git format-patch` output
with the commit messages). The per-section numbers below are from the
investigation, in the order the changes were found; re-measured patch by
patch on the stacked series the steps are 984 → 1,477 → 1,634 → 1,815 →
1,949 → 2,068 img/s (the README there has the table). I would like to know
which of these you want as PRs and how you would prefer them gated.

## 1. Backward kernels without `atomic_addf` (972 → 1,374 img/s) — `04-winograd-no-atomics`

Both backward kernels have many work-groups adding into the same output
values, so they accumulate through `atomic_addf`. On a device with no hardware
fp32 atomic add, that expands to the compare-and-swap loop in `atomic.h` —
read the value, add to it, try to write it back, start over if another lane
got there first — and it measured **39% of the backward-filter kernel** (found
by swapping it for a racy plain `+=`). This is not a niche case: `llc` lowers
`atomicrmw fadd` to `global_atomic_cmpswap` for gfx1010–gfx1036, so every
RDNA1 and RDNA2 GPU takes that path, `global_atomic_add_f32` only arrives with
gfx11, and the Intel path in `atomic.h` is a CAS loop as well.

The atomics were only ever combining results *across* K slices. Within one
slice, each output element has exactly one writer, so nothing needs
protecting. Each slice can therefore write into its own plane of the existing
`workspace()` and a small kernel can add the planes afterwards (applying
`beta`).

Backward-data takes the same treatment with a twist: its 4×4 input-gradient
tiles genuinely do overlap, so there is no single-writer split by slice. But
the overlap has structure — tiles whose row and column indices share the same
parity never touch — so the tiles fall into four classes that are each
conflict-free. Four planes, one per class, and the reduce masks the border
elements that no tile of a class reaches (which also makes the old
zero-fill/pre-scale pass unnecessary).

The reduce costs under 2% of the kernel it replaces. On NVIDIA and gfx11+ the
atomics are cheap and the planes would only cost workspace and an extra
kernel, so the patch selects the planes when the device is not NVIDIA and does
not advertise `cl_ext_float_atomics`, and keeps the atomic path otherwise.

## 2. Four access patterns (1,374 → 1,816 img/s, inference 4,433 → 6,418) — `05`, `06`, `07`

Three places where neighbouring lanes were reading far-apart addresses, so
each read pulled in a cache line to use a few bytes of it. Each was found by
timing a wrong-but-cheap variant of one stage:

- `05-winograd-fwd-filter-layout`: the transformed filters are read at a
  `C × 64`-byte lane stride — 39% of the kernel. Storing them `[C][N]` instead
  of `[N][C]` makes the 32 lanes that read one input channel for 32 output
  features contiguous. Forward Winograd 24.4 → 15.0 ms per step.
- `06-winograd-bwd-filter-loads`: lanes map to neighbouring tiles of one
  channel plane (`k = lid % 8`) rather than to 32 different planes, which had
  every lane reading a different image. Edge tiles use one bounded `vload4`
  per row with the columns masked, instead of a scalar path. 27.0 → 19.4 ms.
- `07-bn-sums-grid-stride`: a grid-stride reduction loop instead of one
  contiguous chunk per work-item. With 512 channels at 8×8, the chunked
  version put every second lane 128 KB apart, so they kept landing in the same
  cache sets and the kernel ran at 13 GB/s on a 359 GB/s machine. BatchNorm
  9.6 → ~3 ms per step.

These are plain coalescing changes and should help everywhere, but they touch
the hottest kernels in the library and I have one device, so they need your
numbers on NVIDIA/Intel before merging.

## 3. Register prefetch (1,816 → 2,055 img/s) — `08-winograd-prefetch`

Each K step used to load its data, wait for it, then do its arithmetic. All
three kernels now issue the *next* step's global loads before the current
step's GEMM and do the transform and LDS store afterwards, so the memory
latency overlaps with work instead of stalling on it. Step 65.0 → 59.8 ms.
LDS double-buffering was measured too and lost to the occupancy it costs;
documented, not included.

## Related, filed separately

Two further changes sit on top of this series and are their own reports,
because each asks a different question: `winograd-lds-padding` (the transpose
padding costs a resident work-group in the backward kernels) and
`winograd-fp16-inner-loop` (an opt-in fp16 GEMM, +45% with an accuracy
trade-off to decide on). Both depend on the patches above and say so.

## Environment

AMD BC-250 (gfx1013, 40 CU, 2.0 GHz), Mesa 26.1.8 and 26.2.2 rusticl on
radeonsi, LLVM 22.1.8; `pytorch_dlprim` `1af48d4` with `dlprimitives`
`ff2d590`. Correctness: Y, dW and dX compared against CPU over the six
ResNet-9 3x3 shapes (900 sweeps, all within 1e-5 relative), BatchNorm
forward and backward within 7e-7 of CPU, and backward-data checked on odd
spatial sizes, 1–2 pixel images and gradient accumulation (beta = 1) for the
plane border masks.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

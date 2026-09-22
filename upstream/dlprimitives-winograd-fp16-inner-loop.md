# dlprimitives: optional fp16 inner loop for the Winograd kernels (1.45× on ResNet-9 training, opt-in)

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues — as a
discussion; this one is a proposal, not a fix
**Against:** `ff2d590` (2024-09-04; the patch also applies to master `b176c15`); `src/core/conv.cpp`, `src/kernels/winograd_*.cl`

## Summary

The three Winograd kernels can convert their tiles to `half` on the way into
LDS and run the register-tile GEMM as packed `half2` FMA, while every tensor
in global memory stays fp32. On an AMD BC-250 (gfx1013) that is worth
**2,055 → 2,987 img/s** on ResNet-9 training and 7,064 → 10,279 on inference,
for ~40 lines per kernel behind `#if HALF_GEMM`.

The cost is accuracy: Y, dX and dW land within ~0.5% of the fp32 result, about
ten times looser than the TF32 that PyTorch uses by default for convolutions
on Ampere. So this is **off by default** and fp32 callers get fp32 — in the
patch it is selected by `DLPRIM_CONV_FP16=1` at kernel-compile time, which is
the crudest gate that works and almost certainly not the one you want. The
question I would like answered is whether you want this at all, and if so
where the switch belongs.

Patch: `patches/dlprimitives/10-winograd-fp16.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013.

## Why the inner loop and not the tensors

`pytorch_ocl` already maps `torch.float16` to `half_data` and the pointwise
kernels honour it, but the heavy kernels are fp32-only — matmul, linear,
pooling, softmax, the loss and BatchNorm all refuse a half tensor. A real fp16
tensor path is a port of every one of them. The inner loop is where the time
actually is: the three Winograd kernels are 77% of a ResNet-9 step (the
`winograd-performance` report has the profile), and they are limited by the
GEMM and the LDS traffic feeding it, not by global memory.

The hardware pays for it. gfx1013 has packed fp16 (`v_pk_fma_f16`, two FMAs
per lane per cycle) and ACO emits it for OpenCL `half2`/`half4` through
rusticl — `tools/fp16-micro.c`, clock pinned at 2.0 GHz, measures 18.7
TFLOP/s on `half8` against 4.65 on dependent `float` chains, 91% of the packed
peak. Halving the LDS tiles helps as much as the math: one `ds_read_b128`
brings eight values instead of four, and the 8×8 register-tile update becomes
32 packed FMAs per K step where it was 64 scalar ones. The epilogue converts
back to fp32 for the inverse transform and the stores.

| per ResNet-9 training step | fp32 | fp16 inner loop | |
| --- | ---: | ---: | ---: |
| `winograd_3x3_main` (forward) | 15.0 ms | 8.9 ms | 1.68× |
| `winograd_3x3_main_bwd` | 16.0 | 10.2 | 1.57× |
| `winograd_bwd_filter` | 19.4 | 12.4 | 1.56× |
| whole step, isolated | 65.2 | **44.3** | **1.47×** |

(Kernel times measured before the prefetch change; the end-to-end figures in
the Summary are on the complete series, where the fp32 step is faster.)

## Accuracy, and the one thing that improves it

There is no packed FMA with an fp32 accumulator on this chip, so the products
are summed in fp16 as well. Against CPU fp32 over five ResNet-9 3×3 shapes,
unit-scale inputs (`tools/grad-err.py`, which runs the same shapes with and
without the switch):

| | Y | dX | dW |
| --- | ---: | ---: | ---: |
| fp32 kernels | 1e-6 | 6e-7 | 7e-6 |
| fp16 inner loop | 0.3–0.9% | 0.4–0.8% | 0.6–1.3% |
| fp16 inner loop, 4× more K splits | same | same | **0.3–0.7%** |

`dW` accumulates over batch × tiles — up to 2,048 terms per work-group — which
is why it was the worst of the three. Split-K bounds it directly: each
work-group sums its own slice in fp16 and the slices are summed in fp32 by the
reduce kernel, so more splits mean shorter fp16 sums. Splitting four times
more halves the dW error at no measurable cost, so the patch raises the
backward-filter split target in fp16 mode only — 16 work-groups per CU
instead of 4, cap 64 instead of 16.

Over a 16-epoch ResNet-9 / CIFAR-10 run the difference does not show:
test accuracy 0.926 in fp16 mode against 0.924 in fp32, which is inside the
run-to-run spread.

## What it depends on

Both couplings are with changes filed separately, and neither is optional:

- **The split-K raise above assumes the new split-K heuristic** (the
  `winograd-ksplit-heuristic` report). Applied to the stock rule, which
  compares work-items to lanes, the target constant means something different.
- **The LDS residency rule** (the `winograd-lds-padding` report) has to be
  told the element size, or it makes the fp32 decision about half-sized tiles.
  With `LTYPE half` a padded work-group is 20 KiB, two fit either way, and the
  padding stays — measured both ways in fp16 mode, 2,889–2,901 img/s, i.e. the
  same number. The patch passes the element size into that rule.

All the numbers here were taken with the rest of that series applied. I have
not measured this change against the stock kernels, and I would not expect the
ratio to hold there.

## Measured and rejected: fp16 tiles, fp32 accumulate

Reading `half` from LDS and accumulating in `float` compiles to
`v_fma_mix_f32`. The accuracy is 6e-4 across Y, dX and dW — TF32-class, ten
times better than the mode above — and it still halves the LDS traffic. But
`v_fma_mix_f32` is not packed and issues slower than `v_mac_f32` on this chip:
forward 13.3 → 17.6 ms, backward-filter 19.3 → 16.9 (the LDS-bound kernel
liked it), whole step 59.8 → 62.6. Slower than plain fp32, so not a candidate
here — but on a device where the mixed FMA is packed, this is the variant
worth having, and it needs no accuracy caveat.

## How would you like it gated

`DLPRIM_CONV_FP16=1` read in the constructors is what the patch does because
it needed no API. It is read when the kernels compile, so it has to be set
before the first convolution, and it applies to every 3×3 convolution in the
process — neither is a property you want in a library. The plausible homes are
a field on `Conv2DSettings`, a context-level precision flag, or a value in the
`algo` string that `Conv2D*::create` already takes. Which of those fits the
direction you have in mind for reduced precision generally?

## Environment

AMD BC-250 (gfx1013, 40 CU, 2.0 GHz), Mesa 26.1.8 and 26.2.2 rusticl on
radeonsi, LLVM 22.1.8; `pytorch_dlprim` `1af48d4` with `dlprimitives`
`ff2d590`. Throughput is a 16-epoch ResNet-9 / CIFAR-10 run at batch 128;
kernel times are medians of three profiled runs with the clock pinned.
Accuracy is the table above — note that the fp32 correctness sweeps the other
reports cite (`tools/wino-repro.py`, 2e-4 relative) do not apply in this mode
and are not the right test for it; `tools/grad-err.py` is.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

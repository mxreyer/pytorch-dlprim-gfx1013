# pytorch_dlprim: half tensors — convolution returns NaN silently; hardtanh/relu6/clamp formulas fail for half

**Where to file:** https://github.com/artyom-beilis/pytorch_dlprim/issues (or a PR)
**Against:** `1af48d4` (2025-11-26); `src/vision_ops.cpp`, `src/pointwise_ops.cpp`

## Summary

Three problems found while inventorying which `torch.float16` operations work
on `ocl:0`:

1. **`convolution_overrideable` accepts a half tensor and returns NaN.** The
   `dlprimitives` convolution kernels are fp32-only; the half bytes go through
   unchecked. Fix: `TORCH_CHECK(input.scalar_type() == kFloat && weight.scalar_type() == kFloat, ...)`
   so it fails loudly like the other unsupported ops (matmul, pooling, softmax,
   BatchNorm all raise).

2. **`hardtanh`, `hardtanh_`, `hardtanh_backward` and `clamp` do not compile
   for half.** The pointwise formulas mix the float scalar parameters with the
   `dtype` operand:

   ```c
   y0=max(w0,min(w1,x0));      // w0, w1 float; x0 half -> ambiguous overload
   ```

   Casting the parameters — `max((dtype)(w0),min((dtype)(w1),x0))`, and
   `(dtype)(0)` for the backward's zero — fixes all four. There are likely
   more formulas in `pointwise_ops.cpp` with the same pattern.

3. **`hardtanh_backward` passes a non-contiguous `grad_output` to
   `todp()`.** After `y.sum().backward()` the gradient is an expanded tensor;
   `grad_output.contiguous()` first.

Patch: `pytorch_ocl_half_fixes.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013; `tools/half-probe.py` there
is the per-op inventory and `tools/act-half.py` checks the activations on half
against CPU. (The matching `dlprimitives` issue: `activation.cl` is built
without its `dtype` define, so relu/tanh/sigmoid on half return garbage.)

## Environment

AMD BC-250 (gfx1013), Mesa rusticl 26.1.8 / 26.2.2, torch 2.4.0,
`pytorch_ocl` 0.2.0 rebuilt from `1af48d4`.

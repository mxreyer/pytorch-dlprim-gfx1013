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
   BatchNorm all raise). The GEMM conv path already has this check in
   `dlprimitives` (`GEMM::get_optimal_conv_gemm`, `DLPRIM_CHECK(dtype == float_data)` —
   what #14 hits with `float64`), but `Conv2DForward::create` sends 3×3
   convolutions with ≥8 channels to the Winograd path, which never checks the
   dtype, so a half tensor reaches the kernel unchecked.

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

Patch: `patches/pytorch_dlprim/01-half-fixes.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013. (The matching
`dlprimitives` issue: `activation.cl` is built without its `dtype` define, so
relu, tanh and sigmoid on half return garbage until that is fixed too.)

## Environment

AMD BC-250 (gfx1013), Mesa rusticl 26.1.8 / 26.2.2, torch 2.4.0,
`pytorch_ocl` 0.2.0 rebuilt from `1af48d4`.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

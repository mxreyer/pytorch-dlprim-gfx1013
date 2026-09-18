# dlprimitives: activation kernels are built without `dtype`, so half tensors are processed as float

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues
**Against:** `ff2d590` (2024-09-04; the patch also applies to master `b176c15`); `src/core/activation.cpp`, `src/kernels/activation.cl`

## Summary

`activation_forward` / `activation_backward` build `activation.cl` with the
`ACTIVATION` define and nothing else. The kernel is written in terms of
`dtype`, and `defs.h` defaults `dtype` to `float` — so when the tensor is
`half_data`, the kernel reads and writes its bytes as if they were floats.
`relu`, `tanh`, `sigmoid` and `relu6` on a half tensor return garbage, with no
error anywhere.

Reproducer through `pytorch_ocl`:

```python
x = torch.randn(256, 256).to("ocl:0").half()
torch.relu(x).cpu()          # wrong values, no error
```

## Fix

Pass the dtype through, the way the other kernels do:

```cpp
cl::Program const &prog = gpu::Cache::instance().get_program(ctx, "activation",
        "ACTIVATION", int(activation),
        "dtype", data_type_to_opencl_type(x.dtype()));
```

with a `DLPRIM_CHECK` that the input and output dtypes match. In
`activation_diff` the `beta` argument is then better taken as `float` and cast
inside the kernel, since the host passes a float regardless of the tensor
type. Patch: `patches/dlprimitives/01-activation-dtype.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013. With it, all four
activations on half match the CPU forward and backward to fp16 precision
(sigmoid within 5e-4, the rest exactly).

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

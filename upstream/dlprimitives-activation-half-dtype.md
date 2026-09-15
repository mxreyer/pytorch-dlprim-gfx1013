# dlprimitives: activation kernels are built without `dtype`, so half tensors are processed as float

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues
**Against:** `ff2d590` (2024-09-04); `src/core/activation.cpp`, `src/kernels/activation.cl`

## Summary

`activation_forward` / `activation_backward` build `activation.cl` with only
the `ACTIVATION` define. The kernel is written in terms of `dtype`, which
`defs.h` defaults to `float`, so a `half_data` tensor is read and written as
float bytes: `relu`, `tanh`, `sigmoid` and `relu6` on a half tensor return
garbage, silently.

Reproducer through `pytorch_ocl`:

```python
x = torch.randn(256, 256).to("ocl:0").half()
torch.relu(x).cpu()          # wrong values, no error
```

## Fix

Pass the dtype through, as the other kernels do:

```cpp
cl::Program const &prog = gpu::Cache::instance().get_program(ctx, "activation",
        "ACTIVATION", int(activation),
        "dtype", data_type_to_opencl_type(x.dtype()));
```

with a `DLPRIM_CHECK` that input and output dtypes match. In
`activation_diff` the `beta` argument is then better taken as `float` and cast
inside the kernel, since the host passes a float. Patch:
`patches/01-activation-dtype.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013; `tools/act-half.py` there
checks each activation on half against CPU.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

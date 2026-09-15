# dlprimitives: enable CUSTOM_REDUCE automatically when the device lacks work-group collectives

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues
**Against:** `ff2d590` (2024-09-04; the patch also applies to master `b176c15`); `src/kernels/reduce.h`, `src/program_cache.cpp`

## Summary

On an OpenCL 3.0 platform that does not implement the optional work-group
collective functions, every kernel that reduces across a work-group — softmax,
log-softmax, NLL loss, bias gradients, BatchNorm sums, global pooling — fails to
compile with `use of undeclared identifier 'work_group_reduce_add'`. Plain
matmul, elementwise and unbiased convolution work, so `pytorch_ocl` imports
fine and then fails on the first loss. Seen on Mesa rusticl; clvk (#39)
reports the same shape of platform (OpenCL 3.0, OpenCL C 1.2, no
`__opencl_c_work_group_collective_functions`) and should hit it too.

## Cause

`program_cache.cpp` passes `-cl-std=CL2.0` whenever the platform reports
OpenCL 2.x or 3.x, and `reduce.h` then picks the OpenCL 2.0 built-ins:

```c
#if __OPENCL_VERSION__ >= 200 && !CUSTOM_REDUCE
#define my_work_group_reduce_add(val) do { val = work_group_reduce_add(val); } while(0)
```

In OpenCL 3.0 those built-ins are optional. rusticl reports OpenCL 3.1 but does
not advertise `__opencl_c_work_group_collective_functions` in
`CL_DEVICE_OPENCL_C_FEATURES` (verified on Mesa 26.1.8 and 26.2.2, radeonsi
gfx1013; the feature list contains only `__opencl_c_integer_dot_product_input_4x8bit`),
and no libclc build implements the functions. So the source compiles with
`__OPENCL_VERSION__ >= 200`, and the built-ins are simply not there.

The `CUSTOM_REDUCE` fallback in `reduce.h` (a `__local` tree reduction with
`barrier()`, OpenCL 1.2) works on every device; it is only off by default.

## Suggested fix

Query `CL_DEVICE_OPENCL_C_FEATURES` once per context, and when
`__opencl_c_work_group_collective_functions` is absent (or the device is < 2.0),
pass `-DCUSTOM_REDUCE=1` from `build_program`. That keeps the fast built-ins
where they exist and makes every reduction kernel compile on any conforming
OpenCL 3.0 platform — rusticl (AMD hardware without ROCm support, Intel
integrated graphics on Mesa, Nouveau) and clvk among them.

Workaround in the meantime: `#define CUSTOM_REDUCE 1` in `reduce.h`
(`patches/dlprimitives/00-custom-reduce.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013).
Measured cost on the BC-250 workload: none visible; reductions are a few
percent of a ResNet-9 step.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

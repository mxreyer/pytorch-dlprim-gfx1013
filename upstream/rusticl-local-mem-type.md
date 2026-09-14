# rusticl/radeonsi: CL_DEVICE_LOCAL_MEM_TYPE reports CL_GLOBAL on a device with real LDS

**Where to file:** https://gitlab.freedesktop.org/mesa/mesa/-/issues
**Labels:** rusticl, radeonsi

## Summary

On gfx1013 (AMD BC-250, Cyan Skillfish) rusticl reports
`CL_DEVICE_LOCAL_MEM_TYPE = CL_GLOBAL`, i.e. that `__local` memory is emulated in
global memory. The device has real on-chip LDS, and it measures 8.7× faster than
global memory. `CL_DEVICE_LOCAL_MEM_SIZE` is reported correctly (65536).

## Environment

- Mesa 26.1.8 (Fedora 44, `mesa-libOpenCL-26.1.8-1.fc44.x86_64`)
- `AMD BC-250 (radeonsi, gfx1013, ACO, DRM 3.64, 7.1.13-200.fc44.x86_64)`
- `RUSTICL_ENABLE=radeonsi`

## What is reported

```
$ RUSTICL_ENABLE=radeonsi clinfo | grep -i 'local memory'
  Local memory type                               Global
  Local memory size                               65536 (64KiB)
```

Via the API, `CL_DEVICE_LOCAL_MEM_TYPE` returns `CL_GLOBAL` (2).

## What is measured

A kernel reading from a `__local` array versus the same access pattern against a
`__global` buffer, both at full clock (best of 5, 40 CUs at 2.0 GHz):

| | bandwidth |
| --- | ---: |
| `__local` read | 3,139 GB/s |
| `__global` read (512 MiB working set) | 359 GB/s |

8.7×, which is not something a global-memory emulation of `__local` can produce.
Reproducer: https://github.com/mxreyer/pytorch-dlprim-gfx1013 `tools/ocl-micro.c`,
tests `lds` and `bw`.

## Why it matters

`CL_DEVICE_LOCAL_MEM_TYPE == CL_GLOBAL` is the documented signal that
`__local` buys nothing, and OpenCL BLAS/DNN libraries that auto-tune read it to
decide whether to use a tiled algorithm at all. On this device that decision
would be exactly backwards: a tiled kernel fed from `__local` is the only way to
get near peak, and LDS bandwidth is in fact the binding constraint for such
kernels here.

## Expected

`CL_LOCAL` on radeonsi devices that have LDS.

## Note

I have only this one device, so I cannot say whether this is specific to
radeonsi, to gfx1013, or general to rusticl.

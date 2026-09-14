# rusticl/radeonsi: CL_DEVICE_LOCAL_MEM_TYPE reports CL_GLOBAL on a device with real LDS

**Where to file:** https://gitlab.freedesktop.org/mesa/mesa/-/issues
**Labels:** rusticl, radeonsi

## Summary

On gfx1013 (AMD BC-250, Cyan Skillfish) rusticl reports
`CL_DEVICE_LOCAL_MEM_TYPE = CL_GLOBAL`, i.e. that `__local` memory is emulated
in global memory. The device has real on-chip LDS, and it measures 13.7× faster
than global memory. `CL_DEVICE_LOCAL_MEM_SIZE` is reported correctly (65536).

## Environment

Mesa 26.1.8 (Fedora 44, `mesa-libOpenCL-26.1.8-1.fc44`), `AMD BC-250 (radeonsi,
gfx1013, ACO, DRM 3.64, 7.1.13-200.fc44.x86_64)`, `RUSTICL_ENABLE=radeonsi`.

```
$ RUSTICL_ENABLE=radeonsi clinfo | grep -i 'local memory'
  Local memory type                               Global
  Local memory size                               65536 (64KiB)
```

## Measurement

A kernel reading a `__local` array versus the same access pattern against a
`__global` buffer (best of 5, 40 CUs at 2.0 GHz):

| | bandwidth |
| --- | ---: |
| `__local` read | 4,902 GB/s |
| `__global` read (512 MiB working set) | 359 GB/s |

That is ~122 B/clk/WGP — the hardware LDS figure, not something a
global-memory emulation can produce. Reproducer: `tools/ocl-micro.c` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013, tests `lds` and `bw`.

## Why it matters

`CL_LOCAL_MEM_TYPE == CL_GLOBAL` is the documented signal that `__local` buys
nothing, and OpenCL BLAS/DNN libraries that auto-tune read it to decide whether
to use a tiled algorithm at all. On this device that decision would be exactly
backwards.

## Expected

`CL_LOCAL` on radeonsi devices that have LDS.

I have only this one device, so I cannot say whether this is specific to
gfx1013 or general to rusticl on radeonsi.

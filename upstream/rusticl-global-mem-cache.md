# rusticl/radeonsi: CL_DEVICE_GLOBAL_MEM_CACHE_TYPE reports CL_NONE on a device with a cache

**Where to file:** https://gitlab.freedesktop.org/mesa/mesa/-/issues
**Labels:** rusticl, radeonsi

## Summary

On gfx1013 (AMD BC-250, Cyan Skillfish) rusticl reports no global memory cache:

```
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE      = CL_NONE
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE      = 0
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE  = 0
```

The device has a cache of roughly 2 MB that delivers about 2.5× DRAM bandwidth.

## Environment

Mesa 26.1.8 (Fedora 44, `mesa-libOpenCL-26.1.8-1.fc44`), `AMD BC-250 (radeonsi,
gfx1013, ACO, DRM 3.64, 7.1.13-200.fc44.x86_64)`, `RUSTICL_ENABLE=radeonsi`.

## Measurement

Streaming read bandwidth against working-set size (best of 3, full clock):

```
     256 KiB      767 GB/s
     512 KiB      948 GB/s
    2048 KiB      833 GB/s   <- still cached
    4096 KiB      567 GB/s   <- falling out
    8192 KiB      340 GB/s
   32768 KiB      327 GB/s   <- DRAM
```

Reproducer: `tools/ocl-micro.c` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013, test `cache`.

## Why it matters

These three queries are the standard input to cache blocking. A library that
sizes tiles from `CL_DEVICE_GLOBAL_MEM_CACHE_SIZE` gets 0 and cannot block;
`CL_NONE` states affirmatively that there is nothing to block for, which is
worse than declining to answer.

## Expected

`CL_READ_WRITE_CACHE` with the L2 size and a 64/128 B cacheline, from the
values radeonsi already knows for the chip.

I have only this one device, so I cannot say whether this is specific to
gfx1013 or general to rusticl on radeonsi.

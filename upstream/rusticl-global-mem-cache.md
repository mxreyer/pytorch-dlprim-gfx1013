# rusticl: CL_DEVICE_GLOBAL_MEM_CACHE_TYPE is hardcoded to CL_NONE for every device

**Where to file:** https://gitlab.freedesktop.org/mesa/mesa/-/issues
**Labels:** rusticl

## Summary

rusticl answers the three global-cache queries with constants, regardless of
driver or hardware (`src/gallium/frontends/rusticl/api/device.rs`, `main`):

```rust
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE => v.write::<cl_device_mem_cache_type>(CL_NONE),
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE => v.write::<cl_ulong>(0),
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE => v.write::<cl_uint>(0),
```

These are the same constants clover returned. On gfx1013 (AMD BC-250, Cyan
Skillfish, radeonsi) the device has a cache of roughly 2 MB that delivers
about 2.5× DRAM bandwidth.

## Environment

Mesa 26.1.8 (Fedora 44, `mesa-libOpenCL-26.1.8-1.fc44`), `AMD BC-250 (radeonsi,
gfx1013, ACO, DRM 3.64, 7.1.13-200.fc44.x86_64)`, `RUSTICL_ENABLE=radeonsi`.
Same device as #16042.

```
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE      = CL_NONE
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE      = 0
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE  = 0
```

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

## Why it matters

These three queries are what a program has to size cache blocking from. A
program that reads `CL_DEVICE_GLOBAL_MEM_CACHE_SIZE` gets 0 (hashcat reads it;
John the Ripper prints it), and `CL_NONE` states affirmatively that there is
nothing to block for, which is worse than declining to answer.

## Expected

`CL_READ_WRITE_CACHE` with the L2 size and a 64/128 B cacheline. radeonsi
already has these per chip (`radeon_info::l2_cache_size`, `tcc_cache_line_size`);
the values need a gallium query so rusticl can report them, with `CL_NONE`
kept only for drivers that do not answer.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

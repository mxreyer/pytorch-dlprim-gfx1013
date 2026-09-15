# rusticl: CL_DEVICE_LOCAL_MEM_TYPE is hardcoded to CL_GLOBAL for every device

**Where to file:** https://gitlab.freedesktop.org/mesa/mesa/-/issues
**Labels:** rusticl

## Summary

rusticl answers `CL_DEVICE_LOCAL_MEM_TYPE` with a constant, regardless of
driver or hardware (`src/gallium/frontends/rusticl/api/device.rs`, `main`):

```rust
// TODO add query for CL_LOCAL vs CL_GLOBAL
CL_DEVICE_LOCAL_MEM_TYPE => v.write::<cl_device_local_mem_type>(CL_GLOBAL),
```

`CL_GLOBAL` means `__local` memory is emulated in global memory. On gfx1013
(AMD BC-250, Cyan Skillfish, radeonsi) the device has real on-chip LDS, and it
measures 13.7× faster than global memory. `CL_DEVICE_LOCAL_MEM_SIZE` is
reported correctly (65536).

## Environment

Mesa 26.1.8 (Fedora 44, `mesa-libOpenCL-26.1.8-1.fc44`), `AMD BC-250 (radeonsi,
gfx1013, ACO, DRM 3.64, 7.1.13-200.fc44.x86_64)`, `RUSTICL_ENABLE=radeonsi`.
Same device as #16042.

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
to use a tiled algorithm at all. On any GPU with dedicated shared memory —
which is every radeonsi, iris and nouveau device — that decision is exactly
backwards.

## Expected

`CL_LOCAL` when the driver has dedicated local memory. There is no gallium
compute cap for this today, so it presumably needs one (or a per-driver
default of `CL_LOCAL` for GPU screens, with `CL_GLOBAL` kept for llvmpipe).

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

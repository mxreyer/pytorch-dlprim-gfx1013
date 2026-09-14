> **WITHDRAWN 2026-09-10 — do not file.** Everything below is a real
> measurement of a wrong cause. The stale-LDS read is the GPU running at the
> clock governor's undervolted idle floor (1000 MHz @ 718 mV, set by hand on
> 2026-09-08): pinned there, 66 / 100 sweeps fail with *any* code; pinned at
> 1200 MHz or above, or at 1000 MHz with the governor's default 800 mV, 0 /
> 900. `ACO_DEBUG=force-waitcnt`, `nosched`, and "worse on 26.2" all changed
> the failure rate by changing how often a bursty workload let the clock fall
> to the floor. A `s_waitcnt_depctr` that waits for *nothing* before every
> instruction also "fixes" it, which is what gave the game away. Full account:
> OPENCL-PERF.md, Finding 3, *It was the voltage*. Kept as a record of how a
> voltage problem impersonates a compiler bug.

# ACO/rusticl gfx1013: `__local` read returns stale data at one lane; ACO_DEBUG=force-waitcnt fixes it on 26.1, mostly on 26.2

**Where to file:** https://gitlab.freedesktop.org/mesa/mesa/-/issues
**Labels:** rusticl, radeonsi, ACO

## Summary

On gfx1013 (AMD BC-250, Cyan Skillfish) an OpenCL compute kernel with correct
work-group synchronisation intermittently reads **stale `__local` memory at one
lane**, corrupting exactly that lane's output. It happens in roughly 2% of
kernel invocations.

On Mesa 26.1.8, `ACO_DEBUG=force-waitcnt` eliminates it completely (0 failures
in 2,400 checks, against ~2% without), which points at `s_waitcnt` insertion
around the `ds_read` rather than at the kernel. **On Mesa 26.2.2 the failure is
about ten times more frequent (67 of 300 sweeps against 6 of 300), and
`force-waitcnt` no longer fully masks it (1 failure in 900 sweeps).**

## Environment

- Mesa 26.1.8 (Fedora 44, `mesa-libOpenCL-26.1.8-1.fc44.x86_64`), and
  Mesa 26.2.2 (Fedora 45, `mesa-libOpenCL-26.2.2-4.fc45.x86_64`) in a container
  on the same host kernel — both reproduce, see the 26.2 table below
- `AMD BC-250 (radeonsi, gfx1013, ACO, DRM 3.64, 7.1.13-200.fc44.x86_64)`
- `RUSTICL_ENABLE=radeonsi`, wave64 (default for rusticl compute)

## The kernel

`winconv_3x3_bwd_filter` from DLPrimitives
(https://github.com/artyom-beilis/dlprimitives, `src/kernels/winograd_bwd_filter.cl`),
reached through `pytorch_ocl`. The relevant epilogue, per work-group of 256:

```c
for(int dc_split = 0; dc_split < 2; dc_split++) {
    // every work-item writes its slice of the transpose staging area
    for(int dr=0;dr<PATCH_Y;dr++)
        for(int dc=0;dc<PATCH_X;dc+=2)
            s_img_tile(dr+my_gemm_tile_kr, dc+dc_split+my_gemm_tile_tl,
                       my_gemm_tile_b) = p_C[dr][dc+dc_split];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int dc = 0; dc < 4; dc += 2) {
        float16 s_kern16 = vload16(0,&s_img_tile(s_row_t,s_col_t,0));  // <-- stale
        float s_kern9[9];
        transform_kernel_bwd(s_kern16,s_kern9);
        /* ... store s_kern9 to global ... */
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
```

Checked by hand before filing:

- `k`/`K_limit` are uniform across the work-group, so no barrier is reached
  divergently;
- both the preceding K loop and this epilogue are
  `write __local → barrier → read __local → barrier`;
- the 16 work-items sharing `lid % 16` each write one of the 16 `indx` slots the
  `vload16` reads, so every element read is written before the barrier;
- all accesses stay inside the declared `__local` array.

## Symptom

Exactly **9 consecutive floats** — one work-item's whole output — are garbage
(~2e9 where the correct magnitude is ~10). Across captured failures the affected
element is always at the *same* position within the work-group's 32×32 output
tile, i.e. the same lane, not a random scatter.

Reading the kernel's output buffer back and comparing across runs with identical
inputs (results must be bit-identical) over 150 runs of six layer shapes:

```
    128->128   1 distinct hash   x150
    128->256   1 distinct hash   x150
    3->64      1 distinct hash   x150
    64->128    1 distinct hash   x150
    256->512   1 distinct hash   x146   + 4 one-off hashes   <- the failures
```

## What does and does not change it

| variant | bad / 150 runs |
| --- | ---: |
| as written | 3–7 |
| `atomic_xchg` store instead of a plain store | 6 |
| 16 scalar `__local` reads instead of `vload16` | 9 |
| `ACO_DEBUG=nosched` | 20 |
| `ACO_DEBUG=validateir,validatera` | 4 (validators do not fire) |
| **`ACO_DEBUG=force-waitcnt`** | **0** (and 0 over a further 400-run confirmation) |

Neither the load form nor the store form matters. `nosched` makes it *worse*,
which is consistent with a latent wait-count problem being reshuffled rather
than fixed. `force-waitcnt` fixes it outright.

### Mesa 26.2.2

Same kernel, same host, same reproducer, run in a Fedora 45 container so that
only the Mesa userspace changes:

| Mesa | variant | bad / 300 sweeps (× 6 shapes) |
| --- | --- | ---: |
| 26.1.8 | as written | 6 |
| 26.2.2 | as written | **67** |
| 26.2.2 | `ACO_DEBUG=force-waitcnt` + `MESA_SHADER_CACHE_DISABLE=true` | 1 in 900 sweeps (1 in the first 300, 0 in a further 600) |

The failures are still `dW`, still the same shapes, still one lane. Whatever
changed in ACO's scheduling between 26.1 and 26.2 made the window wider, and
conservative wait-counts no longer close it entirely.

Performance cost of `force-waitcnt` on this workload: **about a third of the
machine** — 975/977/984 images/s without it, 642/659/647 with, on the same
ResNet-9 training step. That is expected for a global conservative-waitcnt
setting, and it is why this is a diagnosis rather than a fix we can ship.

### Second, smaller issue: `ACO_DEBUG` is not in the shader cache key

`ACO_DEBUG=force-waitcnt` has no effect when Mesa's on-disk shader cache is warm
— the cached binaries were compiled without it and are reused. The failure rate
goes straight back to ~2%:

| | bad / 200 runs |
| --- | ---: |
| `force-waitcnt`, shader cache enabled (default) | 4 |
| `force-waitcnt` + `MESA_SHADER_CACHE_DISABLE=true` | 0 |

This cost me a round of measurements that looked like a free fix and were
actually the unfixed binaries. Either `ACO_DEBUG` should participate in the
cache key, or its effect on caching should be documented.

## Why it is easy to miss

A stale `__local` read only becomes visible if the stale contents differ enough
from the correct value. In this workload the same buffer is reused by another
operation that leaves large values in it; with that operation routed elsewhere,
the stale read picks up a plausible number and the corruption passes silently.
Repeating one shape in a loop also hides it, because the stale value is then the
previous iteration's near-identical one. Both cost us a lot of time, and they
are worth knowing if you try to reproduce.

## Reproducer

https://github.com/mxreyer/pytorch-dlprim-gfx1013 — `tools/wino-repro.py`,
which needs `torch` + `pytorch_ocl` (`build.sh` builds them):

```
# the affected kernels are opt-in in that build and sit behind a safety
# interlock; DLPRIM_UNSAFE_FAST_PATHS=1 disables it. This is the bug with no
# workaround applied:
RUSTICL_ENABLE=radeonsi DLPRIM_UNSAFE_FAST_PATHS=1 \
    DLPRIM_WINOGRAD_SPLIT_PLANES=1 DLPRIM_WINOGRAD_BWD_PLANES=1 \
    python3 tools/wino-repro.py 300     # ~2% bad on 26.1.8, ~20% on 26.2.2
RUSTICL_ENABLE=radeonsi DLPRIM_UNSAFE_FAST_PATHS=1 \
    ACO_DEBUG=force-waitcnt MESA_SHADER_CACHE_DISABLE=true \
    DLPRIM_WINOGRAD_SPLIT_PLANES=1 DLPRIM_WINOGRAD_BWD_PLANES=1 \
    python3 tools/wino-repro.py 300     # clean on 26.1.8; ~1 in 900 on 26.2.2
# note: MESA_SHADER_CACHE_DISABLE is required, see below
```

## What I could not do

I have only this one device, so I cannot say whether this is specific to
gfx1013, to RDNA1, or general. I also could not get ISA out to point at the
offending `s_waitcnt`: `AMD_DEBUG=asm`/`cs`/`stats` print internal blit shaders
but not rusticl compute kernels on this build, so the disassembly is missing
from this report. If there is a supported way to dump rusticl kernel ISA I am
happy to add it.

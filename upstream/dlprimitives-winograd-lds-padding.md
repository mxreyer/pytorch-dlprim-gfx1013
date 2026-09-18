# dlprimitives: the Winograd transpose padding costs a resident work-group

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues
**Against:** `ff2d590` (2024-09-04; also applies to master `b176c15`); `src/core/conv.cpp` and `src/kernels/winograd_*.cl`

## Summary

The three Winograd kernels pad their `__local` tiles so that rows do not land
on the same memory bank. Two defines control it, and the host picks them in
the three constructors:

```cpp
int off = ctx.is_amd() ? 0 : 1;
int toff = 1;
```

`TR_STRIDE_OFFSET` pads the transpose-stage scratch. What is easy to miss is
that it does not only change a stride — *any* non-zero offset also sets
`PADDING_FACTOR` in the kernel, which allocates 16 more tile rows:

```c
#if STRIDE_OFFSET > 0 || TR_STRIDE_OFFSET > 0
#define PADDING_FACTOR 1
#else
#define PADDING_FACTOR 0
#endif
__local LTYPE wg_local_memory[(XTILES_IN_WG + YTILES_IN_WG + 16 * PADDING_FACTOR) * WG_K * 16];
```

With `XTILES_IN_WG = YTILES_IN_WG = 32` and `WG_K = 8` that is **40 KiB per
work-group instead of 32 KiB** in fp32. On AMD, where `off` is already 0, the
single `toff = 1` buys one padded stride and pays the whole 8 KiB for it.

## Effect

Those 8 KiB are worth a resident work-group. Each compute unit has a fixed
budget of `__local`, so a work-group that asks for more of it means fewer run
at once. Measured on an AMD gfx1013 (40 CU, 64 KiB `__local`, Mesa 26.1.8
rusticl) with a latency-bound probe that sweeps the allocation of a 256-item
work-group and counts how many run concurrently on one CU: **32 KiB → 2.0,
40 KiB → 1.2**. The padded work-group has a compute unit to itself.

Half the machine is worth more than the bank conflicts the padding avoids —
but only in the backward kernels. ResNet-9, batch 128, median GPU time per
kernel over three profiled runs:

| kernel | padded (`TR_STRIDE_OFFSET=1`) | unpadded |
| --- | ---: | ---: |
| `winograd_bwd_filter` | 19.1 ms | **14.9 ms** |
| `winograd_3x3_main_bwd` | 14.8 ms | **12.1 ms** |
| `winograd_3x3_main` (forward) | **14.1 ms** | 16.8 ms |

The forward kernel goes the other way and should keep its padding. Per
direction, that is 60.6 → 53.8 ms of GPU time per training step, and
2,069 → 2,330 img/s on the same step measured without profiling. Over a full
16-epoch ResNet-9 / CIFAR-10 run, same binary, one define apart: training
**1,993 → 2,244 img/s**, inference 6,732 → 6,774 (unchanged, as expected —
there is no backward kernel in an inference pass), test accuracy 0.924 either
way, 90% reached at epoch 13 either way.

## Suggested fix

Drop the transpose padding in the two backward constructors when it costs
residency, leave the forward kernel alone, and derive the choice from the
device rather than from the vendor:

```cpp
static bool drop_tr_padding(Context &ctx,int off)
{
    int elem = /* fp16 tiles ? 2 : 4 */;
    int unpadded = (XTILES_IN_WG + YTILES_IN_WG) * WG_K * 16 * elem;   // 32 KiB
    int padded   = unpadded + 16 * WG_K * 16 * elem;                   // 40 KiB
    if(off != 0)   // PADDING_FACTOR is on regardless of TR_STRIDE_OFFSET
        return false;
    int lds = ctx.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    return lds < 2 * padded && lds >= 2 * unpadded;
}
```

This is a no-op wherever `STRIDE_OFFSET` is 1 — every non-AMD device, which
allocates the extra rows regardless and so has nothing to win — and wherever
the device reports 80 KiB or more of `__local`. The kernels themselves are
untouched; only the define changes, so results are bit-identical.

A smaller version of the same change, if the rule above is more machinery than
you want, is `toff = ctx.is_amd() ? 0 : 1` in the two backward constructors,
matching the line above it.

Patch: `patches/dlprimitives/09-winograd-tr-offset.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013.

**This one depends on the rest of that series.** The numbers above were taken
with it applied. Against the stock backward kernels — emulated `atomic_addf`
and the current split-K rule — the same change is a large *loss* on the same
device: 1,171 → 849 img/s. Those kernels are limited by the atomics rather
than by how many work-groups are resident, so they pay the bank conflicts and
collect none of the benefit. Whatever you make of the series, this define
should not be changed on its own. The series is the subject of the
`winograd-performance` report.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

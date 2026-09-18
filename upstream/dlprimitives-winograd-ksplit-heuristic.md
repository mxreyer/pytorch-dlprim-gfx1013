# dlprimitives: Winograd backward-filter split-K heuristic compares work-items to cores

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues
**Against:** `ff2d590` (2024-09-04; the patch also applies to master `b176c15`); `src/core/conv.cpp`, `Conv2DBackwardFilterWinograd`

## Summary

The backward-filter kernel can cut its reduction over the batch into slices
and hand each slice to its own work-group, which is how a layer with few
channels fills a large GPU. The rule that decides whether to do it is:

```cpp
int winograd_work_items = (channels_in / 32) * (channels_out / 32) * 256;
reduce_k_ = winograd_work_items < ctx.estimated_core_count() ? 8 : 1;
```

The left side counts *work-items* — 256 of them per work-group.
`estimated_core_count()` returns `CU × 64` on AMD and `CU × 128` on NVIDIA,
i.e. lanes, not work-groups. Dividing both sides by 256, the rule says: split
only when the launch has fewer than `CU/4` work-groups (AMD) or `CU/2`
(NVIDIA) — far below what it takes to fill the device. The image size, which
is what determines how much there is to split, never enters the decision at
all.

## Effect

On a 40-CU AMD device (BC-250, gfx1013) the threshold works out at 2,560
work-items, i.e. 10 work-groups. A 128→128 3×3 layer launches exactly 16, so
it stays unsplit: 16 compute units busy, 24 idle for the duration of the
kernel. A 64→128 layer launches 8 and does get split. A 40-SM T4 would behave
the same way, with its threshold at 20.

Measured on ResNet-9 (batch 128, six 3×3 layers), per training step: the
backward-filter kernels drop from 55.3 ms to 46.1 ms, the two 128→128 layers
nearly halve, and end-to-end throughput goes 909 → 972 img/s (+6.9%) with
gradients unchanged to 1e-5 relative.

## Suggested fix

Compare work-groups to compute units, aim for a few work-groups per CU, and
never split a slice below 16 K items:

```cpp
int wg_count = ((config_.channels_in + 31) / 32)
             * ((config_.channels_out + 31) / 32);
int cu = ctx.device().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
int target = cu * 4;
k_split_ = 1;
while(k_split_ < 16 && wg_count * k_split_ < target)
    k_split_ *= 2;
int k_work = config.shape[0] * ((h + 1) / 2) * ((w + 1) / 2);
while(k_split_ > 1 && k_work / k_split_ < 16)
    k_split_ /= 2;
reduce_k_ = k_split_ > 1;
```

with the launch's z-dimension set to `k_split_` instead of the fixed 8 — the
kernel already derives its slice from `get_global_size(2)`, so it needs no
change. Targets of 8, 16 and 32 per CU measured the same as 4 on this device,
so the constant is not sensitive. Patch:
`patches/dlprimitives/02-winograd-ksplit-heuristic.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

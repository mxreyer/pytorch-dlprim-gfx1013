# dlprimitives: Winograd backward-filter split-K heuristic compares work-items to cores

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues
**Against:** `ff2d590` (2024-09-04); `src/core/conv.cpp`, `Conv2DBackwardFilterWinograd`

## Summary

The rule that decides whether to split the K reduction in the Winograd
backward-filter kernel is

```cpp
int winograd_work_items = (channels_in / 32) * (channels_out / 32) * 256;
reduce_k_ = winograd_work_items < ctx.estimated_core_count() ? 8 : 1;
```

The left side counts *work-items* (256 per work-group); `estimated_core_count()`
returns `CU × 64` on AMD and `CU × 128` on NVIDIA. So the split only turns on
when the launch has fewer than `CU/4` (AMD) or `CU/2` (NVIDIA) work-groups —
far below what fills the device. The image size, i.e. how much K there is to
split, never enters.

## Effect

On a 40-CU AMD device (BC-250, gfx1013) the threshold is 2,560 work-items =
10 work-groups. A 128→128 3×3 layer launches exactly 16 work-groups, so it runs
unsplit on 16 CUs with 24 idle for the whole kernel; 64→128 launches 8 and
does split. A 40-SM T4 would behave the same way with its threshold at 20.

Measured on ResNet-9 (batch 128, six 3×3 layers), per training step: the
backward-filter kernels drop from 55.3 ms to 46.1 ms, the two 128→128 layers
nearly halve, and end-to-end throughput goes 909 → 972 img/s (+6.9%) with
gradients unchanged to 1e-5 relative.

## Suggested fix

Compare work-groups to compute units and aim for a few work-groups per CU,
capped by the K available:

```cpp
int wg = ((C_in + 31) / 32) * ((C_out + 31) / 32);
int cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
int target = cu * 4;                       // ~4 work-groups per CU
k_split_ = wg >= target ? 1 : min(round_up(target, wg), k_available, 16);
```

Targets of 8, 16 and 32 per CU measured the same as 4 on this device, so the
constant is not sensitive. Patch (with env-var overrides for measurement) in
`winograd_ksplit.patch`, https://github.com/mxreyer/pytorch-dlprim-gfx1013;
the full measurements are in that repository's OPENCL-PERF.md, Finding 2.

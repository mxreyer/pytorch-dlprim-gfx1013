# pytorch_dlprim on gfx1013 (AMD BC-250, Mesa rusticl)

Patches, build script and the performance investigation that make
[pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim) (`pytorch_ocl`,
the OpenCL backend for PyTorch) correct and fast on the AsRock BC-250's
gfx1013 GPU under Mesa's rusticl. ResNet-9 training goes from 909 to
**2,055 img/s** at fp32 (a T4 does 2,294) and inference from 4,489 to
**7,064** (T4: 7,217); an opt-in fp16 inner loop reaches 2,987 / 10,279.

This is the `pytorch_ocl` half of [bc250-jupyterhub-opencl-k3s](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s), the
JupyterHub-on-k3s setup for the same board. That repo's notebook image
installs the `pt_ocl.so` published on this repo's
[releases page](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases),
pinned by sha256.

| | |
| --- | --- |
| [OPENCL-PERF.md](OPENCL-PERF.md) | The investigation: every measurement, command and patch, including what was tried and rejected. |
| [HANDOFF.md](HANDOFF.md) | Where things stand, what is still open, and the constraints to carry forward. |
| `tools/` | Microbenchmarks and probes behind the findings (`ocl-micro.c`, `libclc-probe.c`, `fp16-micro.c`), correctness sweeps (`wino-repro.py`, `bn-check.py`, ...), a Mesa-from-source container (`mesa-dev/`). |
| `upstream/` | Draft reports for Mesa/rusticl, including one withdrawn and kept as a record. |

## The patches

Four patches against `pytorch_dlprim` / its `dlprimitives` submodule. The
first is a **correctness** fix - without it nothing that reduces across a
work-group will compile. The second and fourth are **performance** fixes found
by profiling the stack, the third a set of half-tensor correctness fixes found
on the way; the measurements behind them are in
[OPENCL-PERF.md](OPENCL-PERF.md).

| patch | what | effect |
| --- | --- | ---: |
| `custom_reduce.patch` | portable work-group reduction | makes softmax / cross-entropy / bias-grad compile at all |
| `winograd_ksplit.patch` | the `dlprimitives` patch: fill all 40 CUs in Winograd backward-filter; replace both backward kernels' emulated fp32 atomics with plane writes + a reduce; coalesce four memory access patterns (forward filter layout, backward-filter lane mapping and edge loads, BatchNorm reduction); prefetch; opt-in fp16 inner loop; activation kernels built with their dtype | 909 -> 2,055 img/s training (2.26x), 4,489 -> 7,064 inference (1.57x) |
| `pytorch_ocl_half_fixes.patch` | half-tensor correctness in `pytorch_ocl`: reject non-float32 in convolution (returned NaN), dtype-correct hardtanh/relu6/clamp formulas, contiguous grad in `hardtanh_backward` | correctness |
| `gelu_mad.patch` | `fma()` -> `mad()` (rusticl's `fma()` is a software emulation until Mesa 26.2; the image is on 26.2 now, so this is redundant but harmless) | 3.8x on GELU backward (Mesa 26.1) |

---

## 1. `custom_reduce.patch` - the portable reduction path

**Problem:** on the BC-250, `pytorch_ocl` ([OpenCL backend for
PyTorch](https://github.com/artyom-beilis/pytorch_dlprim), built on
[DLPrimitives](https://github.com/artyom-beilis/dlprimitives)) fails to
compile any GPU kernel that reduces values across a work-group —
`softmax`, `log_softmax`, `cross_entropy`, `nll_loss`, the bias-gradient
step used by `Linear`/`Conv2d`, batchnorm sums, global pooling.  Plain
math (matmul, elementwise, convolution without bias) is unaffected.

**Root cause:** those kernels call the OpenCL 2.0 built-ins
`work_group_reduce_add()` / `work_group_reduce_max()`. rusticl does not
implement the OpenCL C feature they belong to
(`__opencl_c_work_group_collective_functions` is absent from
`CL_DEVICE_OPENCL_C_FEATURES`, and neither upstream libclc nor Mesa's fork
contains a `work_group_reduce_*` implementation), so the built-ins aren't
declared and the kernel source fails with "use of undeclared identifier".
This is independent of which libclc is installed (`tools/libclc-probe.c`
checks it directly; the patched libclc in the notebook image fixes a different
problem, see [the image's README](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s/blob/main/k8s/jupyterhub/notebook-image/README.md)).

**Fix:** DLPrimitives already has a fallback for exactly this. In
`dlprimitives/src/kernels/reduce.h`, an `#ifndef CUSTOM_REDUCE` block picks
between the OpenCL 2.0 built-ins and a hand-written reduction using
`__local` memory + `barrier()`, which needs nothing past OpenCL 1.2. That
fallback is off by default (`CUSTOM_REDUCE 0`). The patch forces it on.

There's no need to touch Mesa, libclc, or anything else on the system —
this is entirely a rebuild of one out-of-tree Python package inside its own
virtualenv.

## What's in this repository

- `custom_reduce.patch` — the one-hunk fix against `pytorch_dlprim`'s
  `dlprimitives` submodule (`src/kernels/reduce.h`).
- `winograd_ksplit.patch` — all of the convolution work, against the
  `dlprimitives` submodule (`src/core/conv.cpp`, the Winograd `.cl` kernels,
  `bn_sums.cl`, the activation kernels).
- `gelu_mad.patch` — `fma()` -> `mad()`, against `pytorch_dlprim` itself
  (`src/pointwise_ops.cpp`).
- `pytorch_ocl_half_fixes.patch` — against `pytorch_dlprim`
  (`src/vision_ops.cpp`, `src/pointwise_ops.cpp`): float32 `TORCH_CHECK` in
  `convolution_overrideable`; `(dtype)` casts on hardtanh/relu6/clamp scalar
  parameters; `grad_output.contiguous()` in `hardtanh_backward`. The
  `dlprimitives` side of the half fixes (activation kernels built with their
  `dtype`) is in `winograd_ksplit.patch`.
- `build.sh` — reproduces the whole thing from a clean clone, in a
  self-contained venv under `./scratch/` (gitignored), and leaves the stripped
  extension at `./pt_ocl.so` (gitignored). Each shipped build is attached to a
  GitHub release as `pt_ocl.so`; the notebook image fetches it from there by
  URL + sha256.

## How to rebuild from scratch

```
./build.sh
```

`build.sh` is self-contained:

1. Builds a throwaway Python 3.12 venv in `./scratch/venv/` with the same
   `torch==2.4.0` (CPU) and `pytorch_ocl` 0.2.0 wheel the notebook image
   uses. (Kept between runs; `rm -rf scratch/` to start clean.)
2. Fetches `artyom-beilis/pytorch_dlprim` at the commit pinned in `build.sh`
   (`PYTORCH_DLPRIM_COMMIT`, with the `dlprimitives` submodule checked
   against `DLPRIMITIVES_COMMIT`) into `./scratch/src/`, applies the four
   patches. Bumping the pins is a deliberate step: re-check every
   `git apply`, rebuild, re-run `tools/wino-repro.py` and `tools/bn-check.py`.
3. Builds the extension and installs the stripped result at
   `./pt_ocl.so` next to the script.

The build is reproducible: source and venv paths are mapped out of the binary
(`-ffile-prefix-map`) and no RPATH is embedded, so two runs of `build.sh` — in
any directory, at the pinned upstream commits, with the pinned torch and
pybind11 — give byte-identical output. To publish a build: `sha256sum
pt_ocl.so`, attach the file to a new release (`vX.Y.Z`), and update
`PT_OCL_VER` / `PT_OCL_SHA256` in the notebook image's Dockerfile. Anyone can
then check a release by rebuilding.

To try the rebuild in the scratch venv directly (matmul / softmax checks):

```
cp scratch/build/pytorch_ocl/pt_ocl.so \
   scratch/venv/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so
rm -f ~/.dlprimitives/cache.db   # old cache keyed to the old kernel source
```

**Do not do that while something has the library loaded.** `cp` rewrites the
file in place, which corrupts the mapping in any running process and segfaults
it — including a benchmark that is still downloading its dataset and has
already imported `pytorch_ocl`. Stop the other process first, or `install` to a
new path and move it into place.

Requires `python3.12`, `python3.12-devel`, `cmake`, `git`, `sqlite-devel`,
`OpenCL-ICD-Loader-devel`.

---

## 2. `winograd_ksplit.patch` - fix the Winograd backward-filter split-K path

**Problem:** `dlprimitives` decides whether to split the Winograd
backward-filter reduction with

```cpp
int winograd_work_items = (channels_in / 32) * (channels_out / 32) * 256;
reduce_k_ = winograd_work_items < ctx.estimated_core_count();
```

On a 40 CU AMD device `estimated_core_count()` is `40*64 = 2560`, and the left
side counts *work-items* (256 per work-group), so the split only turns on below
**10 work-groups**. A 128->128 convolution launches exactly 16 work-groups —
over the threshold, so it runs unsplit on 16 of the 40 CUs, with 24 idle for
the whole kernel. The image size never enters the formula, so the amount of
reduction work actually available to split is not considered either.

**Fix:** choose the split from the work-group count versus the CU count (aim
for ~4 work-groups per CU), and never split past the available K work. The
backward-filter kernels drop from 55.3 ms to 46.1 ms per training step (1.20x);
the two 128->128 layers nearly halve. On its own that is **909 -> 972 img/s
(+6.9%)**. Gradients match CPU to better than 1e-5 relative on every ResNet-9
shape.

### The atomics it was paying for

Splitting K means several slices accumulate into the same filter gradient, and
the kernel did that with `atomic_addf`. rusticl has no hardware fp32 atomic add,
so that expands to a compare-and-swap retry loop - **39% of the kernel's
runtime**, measured by swapping the atomics for a (racy, incorrect) plain `+=`.

They were never needed. Within one K slice the (x,y) work-group grid covers each
`(channel_in, channel_out)` pair exactly once, so every filter element is written
by exactly one work-item. The atomics existed only to combine *across* slices.

So each slice now writes its own plane of the workspace - `9*C*N` floats, which
the existing `workspace()` mechanism already knows how to allocate - and a small
second kernel sums the planes into the gradient, applying `beta` as it goes.
That reduction costs under 2% of the kernel it replaces.

`winograd_bwd_filter` drops from **46.1 ms to 27.8 ms** per training step, and
with the matching parity-plane path for backward-data (Winograd's overlapping
4x4 input-gradient tiles fall into four row/column parity classes whose members
are disjoint, so each class gets a plane and a second kernel sums them) the
training step reaches **1,374 img/s against 972**. Both paths are **on by
default**; gradients match CPU references over 900 sweeps of the six ResNet-9
shapes.

These are dense LDS+ALU kernels, and they are the first code on this box to
have exposed a hand-tuned idle undervolt in the clock governor (1000 MHz at
718 mV: 66/100 sweeps wrong with *any* code pinned there; 0/900 with the
governor's default 800 mV). If they ever produce a rare wrong gradient, pin
the clock and run `tools/wino-repro.py` before suspecting the code - the
procedure is in HANDOFF.md, stage 5, and the full account in OPENCL-PERF.md,
Finding 3.

### The access-pattern fixes (2026-09-10, later the same day)

Also in this patch, found by profiling the step once the atomics were gone
(OPENCL-PERF.md, Finding 10); each was measured by timing a wrong-but-cheap
variant of the stage first:

- `winograd_fwd.cl`: the transformed filters are stored `[C][N]` instead of
  `[N][C]`, so the 32 lanes that read one input channel for 32 output features
  read contiguous memory. Forward Winograd 24.4 -> 15.0 ms per step; this is
  the fix that lifts inference 4,433 -> 6,418 img/s.
- `winograd_bwd_filter.cl`: lanes map to neighbouring tiles of one channel
  plane (`k = lid % 8`) rather than to 32 different planes; the 4x4 input tile
  is one bounded `vload4` per row with the columns masked, instead of a scalar
  path for every edge tile; the 2x2 dY tile is two `vload2`. 27.0 -> 19.4 ms.
- `bn_sums.cl`: grid-stride reduction loop instead of one contiguous chunk per
  work-item, which for 512 channels at 8x8 put every second lane 128 KB apart.
  BatchNorm 9.6 -> ~3 ms per step.
- `conv.cpp`: `DLPRIM_WINOGRAD_KSPLIT_TARGET` / `_MAX` knobs for the split
  heuristic (defaults unchanged; more splitting measured no gain).

Correctness: `tools/wino-repro.py` (now checks Y as well as dW/dX) 0 bad / 300
sweeps; `tools/bn-check.py` forward/backward/gradients within 7e-7 of CPU.

### The prefetch (2026-09-10, evening)

All three Winograd kernels issue the next K step's global loads before the
current step's GEMM and transform/store them afterwards (loaders split into a
raw load and the transform). Step 65.0 -> 59.8 ms fp32, 45.1 -> 40.8 fp16;
16 epochs 1,816 -> 2,055 img/s (OPENCL-PERF.md, Finding 12, which also
records the pipelining variants that did not pay off). The backward-data
planes no longer need a zero-fill: the reduce masks the one-pixel border no
tile of a parity class writes (`tools/odd-shapes.py`).

### The fp16 inner loop (opt-in)

`DLPRIM_CONV_FP16=1` makes the three Winograd kernels convert their tiles to
fp16 on the way into LDS and run the 8x8 register-tile GEMM as packed
`v_pk_fma_f16` with fp16 accumulators; the epilogue converts back and the
tensors in global memory stay fp32. Forward 15.0 -> 8.9 ms, backward-data
16.0 -> 10.2, backward-filter 19.4 -> 12.4 per ResNet-9 step; the 16-epoch
benchmark goes 2,055 -> 2,987 img/s at unchanged accuracy, inference 7,064 ->
10,279. Y/dX/dW land within ~0.5% of fp32 (OPENCL-PERF.md, Finding 11), which is
~10x looser than NVIDIA's TF32 default - hence opt-in. `tools/grad-err.py`
prints the errors for the ResNet-9 shapes.

### Environment variables

The patch also adds env vars so these choices can be measured rather than
assumed:

```
DLPRIM_CONV_ALGO             auto | winograd | gemm | depthwise_separable
DLPRIM_CONV_FWD_ALGO         (same, forward only)
DLPRIM_CONV_BWD_DATA_ALGO    (same, backward-data only)
DLPRIM_CONV_BWD_FILTER_ALGO  (same, backward-filter only)
DLPRIM_WINOGRAD_KSPLIT       stock | <n>     stock = reproduce the old rule
DLPRIM_WINOGRAD_STRIDE_OFFSET  <n>           LDS padding (default 0 on AMD)
DLPRIM_WINOGRAD_TR_OFFSET      <n>           LDS padding, transpose stage
DLPRIM_WINOGRAD_SPLIT_PLANES   0 | 1         backward-filter without atomics (default 1)
DLPRIM_WINOGRAD_BWD_PLANES     0 | 1         backward-data without atomics   (default 1)
DLPRIM_WINOGRAD_KSPLIT_TARGET  <n>           backward-filter split-K target, work-groups per CU (default 4)
DLPRIM_WINOGRAD_KSPLIT_MAX     <n>           backward-filter split-K cap (default 16)
DLPRIM_CONV_FP16               0 | 1         fp16 LDS tiles + packed-fp16 GEMM in the three
                                             Winograd kernels, fp32 tensors in/out (default 0).
                                             ~0.5% error vs fp32, 1.47x on the step; raises the
                                             split-K defaults to 16 / 64 to shorten fp16 sums.
                                             Read at kernel compile time: set before the first conv.
```

Setting either `*_PLANES` to `0` selects the original emulated-atomic kernel
(888 img/s with both off), which is the A/B for the planes paths.

An algorithm passed explicitly by the caller always wins over the env var, so
this is inert unless you set it.

## 3. `gelu_mad.patch` - never call `fma()` on rusticl

**Problem:** the libclc that rusticl loads
(`/usr/lib64/clc/spirv64-mesa3d-.spv`) implements OpenCL C's `fma(float,float,
float)` as a call to `__clc_sw_fma` — **563 SPIR-V instructions** of software
float emulation, because the generic SPIR-V libclc target is built without
`__CLC_HAVE_HW_FMA32`. Measured against `mad()`, which lowers to the hardware
`v_fma_f32`: **70.7 GFLOP/s vs 8,451 GFLOP/s, a 120x difference.**

`dlprimitives`' own `.cl` kernels already use `mad()` everywhere. The only
place `pytorch_ocl` calls `fma()` is GELU backward, five times.

**Fix:** use `mad()`. GELU backward on 8M elements goes 3.38 ms -> 0.88 ms
(`approximate="tanh"`) and 1.80 ms -> 1.10 ms (`approximate="none"`), landing
where any memory-bound elementwise op should. `mad()` may round the
intermediate product where `fma()` may not; GELU is an approximation to begin
with and gradients still match CPU to 1.2e-7 relative.

This does not affect the ResNet-9 benchmark (ReLU, no GELU), but GELU is the
activation in every transformer.

**Where the real fix is:** not in libclc. libclc's SPIR-V build deliberately
leaves the choice to the runtime: `__clc_fma` calls a `noinline` stub
`__clc_runtime_has_hw_fma32()` that returns `false`, and the OpenCL runtime is
expected to swap in the device's answer at link time. Mesa 26.1.8 (Fedora 44)
never does, so the stub stays `false` and every `fma()` takes the software
path — with upstream libclc *and* with Mesa's patched fork alike (measured:
`tools/libclc-probe.c`, 15–18x slower than `mad()` under both). Mesa 26.2
adds `rusticl_insert_libclc_config()` (`rusticl_nir.c`), which defines that
stub from `nir_has_ffma()`; Fedora 45 ships Mesa 26.2.0. On that Mesa, `fma()`
becomes the hardware instruction for all code and this patch is redundant.
Confirmed on the Fedora 45 image (Mesa 26.2.2): `fma()` at 1.0× `mad()`. The
patch stays in for the `FEDORA_VERSION=44` build and costs nothing on 45.

---

## Reverting

The stock (unpatched) extension is whatever the `pytorch_ocl` 0.2.0 wheel
ships. To go back, reinstall it into the venv and delete the kernel cache:

```
scratch/venv/bin/pip install --force-reinstall --no-deps \
  "https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"
rm -f ~/.dlprimitives/cache.db
```

For the notebook image, drop the `pt_ocl.so` `ADD`/`RUN` lines from its
Dockerfile and rebuild.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*

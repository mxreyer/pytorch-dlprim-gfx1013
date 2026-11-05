# Patches for dlprimitives/pytorch_dlprim on AMD BC-250

> **Note:** the patches, tools and documentation in this repository were
> heavily authored by Claude ([Claude Code](https://claude.com/claude-code),
> Claude Opus 5). I steered the direction of the investigation and
> validated the results on my BC-250. The performance measurements are real
> and were taken on my board.

Patches and a build script that make
[pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim) (`pytorch_ocl`,
the OpenCL backend for PyTorch) work correctly and run fast on the AsRock
BC-250's gfx1013 GPU under Mesa's rusticl driver. Also contributes and opt-in fp16
inner loop for further speed-up.

The target performance for this investigation was that of a cloud T4, which
the BC-250, operated at 40 CUs / 2.0 GHz, beats by about 25% in peak fp32 on
paper.
Peak fp32 is 40 CUs × 64 lanes × 2 FLOP per FMA × 2.0 GHz = 10.2 TFLOP/s for
the BC-250, against 2,560 CUDA cores × 2 FLOP × 1.59 GHz boost = 8.1 TFLOP/s
for the T4.
Stock `pytorch_ocl`, however, does not even build its loss and BatchNorm kernels
here, and with just the fixes that make it run correctly (`00`–`02`) it trained
ResNet-9 at under half the T4's rate.
The patches from this repo close that gap and pass the T4 in fp32. All
figures below are medians of five runs of `tools/profile-step.py 128 50`, which
times the ResNet-9 training step at batch 128 on random 3×32×32 input with 10
classes, so data loading is not included:

| | training img/s |
| --- | ---: |
| `pytorch_ocl` with `00`–`02` only ([`pt_ocl-fixes-only.so`](#reproducing-the-benchmarks)) | 916 |
| **with these patches** | **2,436** |
| + opt-in fp16 inner loop (`DLPRIM_CONV_FP16=1`) | 3,147 |
| T4 on Google Colab, fp32 (`PROFILE_DEVICE=cuda`) | 1,930 |

In fp32, gradients match CPU references on every ResNet-9 layer shape; with
the fp16 inner loop, values land within 0.9% of fp32, typically ~0.5%. The per-patch
benchmark numbers, for every step in between, are given in the table below.

Each [release](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases)
of this repository has a prebuilt `pt_ocl.so` attached: the native library of
`pytorch_ocl`, compiled by `build.sh` with all the patches below applied. It
is a drop-in replacement for the `pt_ocl.so` inside the stock `pytorch_ocl`
0.2.0 wheel (for `torch==2.4.0`, Python 3.12, x86-64 Linux), so you can get
the fixes and the speed-up without building anything yourself; see
[Using a release build](#using-a-release-build). From v1.3.0 on, a release
also carries `pt_ocl-fixes-only.so`, built by `./build.sh fixes` with only the
build and correctness fixes: the baseline every speed-up here and in the
other BC-250 repositories is measured against, so those comparisons can be
rerun; see [Reproducing the benchmarks](#reproducing-the-benchmarks). The build is reproducible,
so a release can be checked by rebuilding at its tag and comparing `sha256sum`
(kernel comments are compiled in, so later commits can change the hash). To ship
a new build: run `./build.sh` and `./build.sh fixes`, then tag a release with
`pt_ocl.so` and `pt_ocl-fixes-only.so` attached.

## Overview

| file | description |
| --- | --- |
| `tools/` | Microbenchmarks (`ocl-micro.c`, `libclc-probe.c`, `fp16-micro.c`), a Mesa source check for work-group collectives (`check-mesa-collectives.sh`), correctness sweeps (`wino-repro.py`, `bn-check.py`, ...), a Mesa-from-source container (`mesa-dev/`). |
| `upstream/` | Reports ready to file: five bugs and two proposals for `dlprimitives`, one for `pytorch_dlprim`. |
| `patches/` | Everything `build.sh` applies, per target; the `dlprimitives` series is one patch per report. |

## The patches

`patches/<target>/NN-*.patch`, applied in numeric order by `build.sh`.

`patches/dlprimitives/` — the `dlprimitives` submodule. `00` is a
rusticl build fix, `01` a correctness fix it exposes and `02` a second
correctness fix; `03`–`10` are performance changes, stacked in the order
they were measured; `11` is local only. The performance work all goes into
convolution: it is 89% of a ResNet-9 training step before the series, and four
fifths of that is the backward pass. The last column provides benchmark numbers,
in img/s, for the ResNet-9 training step at batch 128 with the series applied up
to that patch, plus `patches/pytorch_dlprim/`.

| patch | what | report in `upstream/` | img/s after |
| --- | --- | --- | ---: |
| `00-custom-reduce` | **Build fix.** Softmax, cross-entropy, bias gradients and BatchNorm sum across a 256-work-item group with `work_group_reduce_add()`, required in OpenCL 2.0 but optional in 3.0. rusticl lacks it whichever libclc is installed: it never lists it in `CL_DEVICE_OPENCL_C_FEATURES`, and its kernel compiler turns off every optional feature it does not list, so clang never declares the built-ins (Mesa 26.2.3 and main). Those kernels fail with "use of undeclared identifier". Switching to Mesa's libclc fork does **not** avoid this patch: the fork fixes `sin`/`cos`/`fma` linking, and as of LLVM 22 no libclc implements the collectives anyway (LLVM 23's adds them). This enables `dlprimitives`' existing `CUSTOM_REDUCE` fallback (shared memory plus a barrier). `tools/libclc-probe.c` checks for the feature. | custom-reduce-autodetect | 921 |
| `01-reduce-barrier` | **Correctness fix.** The `CUSTOM_REDUCE` fallback has a race the built-in never had: every work-item reads the result from shared slot 0 after the last barrier, and work-item 0 can start the *next* reduction and overwrite it first. Softmax reduces twice in a row (max, then sum), so the `dlprimitives` softmax and log_softmax tests fail. Adds the missing barrier. | reduce-barrier | 919 |
| `02-activation-dtype` | **Correctness fix**, not rusticl-specific. The activation kernel is written in terms of `dtype` (default `float`), but is built without the tensor's dtype, so half tensors are read and written as 4-byte floats and `relu`, `tanh`, `sigmoid` and `relu6` silently return garbage. Builds the kernel for the tensor's dtype, checks input and output dtypes match, and takes the backward `beta` as a float (what the host passes) cast in-kernel. All four then match CPU forward and backward to fp16 precision. | activation-half-dtype | 916 |
| `03-winograd-ksplit-heuristic` | **Occupancy.** A rule decides when to split backward-filter work into slices across more of the GPU. It compared *work-items* against a core count, which on a 40-CU AMD device means "split only below 10 work-groups", so a 128→128 layer (16 work-groups) left 24 CUs idle. It now counts work-groups and targets about four per CU. | winograd-ksplit-heuristic | 982 |
| `04-winograd-no-atomics` | **Atomics.** Both backward kernels have many work-groups adding into the same outputs via atomic add. This GPU has no hardware float atomic add (no RDNA before RDNA3 does), so each becomes a read-add-retry loop, 39% of the backward-filter kernel. The atomics only guard collisions *between* slices; within a slice each output has one writer. So each slice now writes its own scratch plane and a small kernel sums the planes, costing under 2% of what it replaces. Backward-data's 4×4 tiles overlap, but grouped by odd/even row and column they form four non-touching sets. Devices with real float atomics (NVIDIA, or `cl_ext_float_atomics`) keep the original path. | winograd-performance §1 | 1,490 |
| `05-winograd-fwd-filter-layout` | **Access patterns.** Neighbouring lanes read far-apart addresses, using a few bytes of each cache line. Here, transformed filters are stored `[C][N]`, so the 32 lanes reading one input channel are contiguous. | winograd-performance §2 | 1,673 |
| `06-winograd-bwd-filter-loads` | **Access patterns.** Backward-filter lanes map to neighbouring tiles of one image plane instead of 32 different planes, with vector loads on edge tiles. | winograd-performance §2 | 1,861 |
| `07-bn-sums-grid-stride` | **Access patterns.** The BatchNorm sum walks memory with a stride instead of per-lane chunks that hit the same cache sets (13 GB/s of a rated 448 GB/s). | winograd-performance §2 | 2,000 |
| `08-winograd-prefetch` | **Prefetch.** Kernels used to load a slice, wait, then multiply. They now issue the next slice's loads *before* the current multiply, overlapping memory waits with arithmetic. | winograd-performance §3 | 2,147 |
| `09-winograd-tr-offset` | **Scratch padding.** Tiles in LDS (shared on-chip memory) are padded to avoid bank conflicts, but any padding also allocates 16 extra tile rows: 40 KiB per work-group instead of 32, just too much for two work-groups per CU, halving parallelism (`tools/ocl-micro.c occ` measures the boundary). Dropping it pays off in the two backward kernels; forward keeps it. | winograd-lds-padding | 2,439 |
| `10-winograd-fp16` | **fp16 inner loop, opt-in.** Tiles are converted to half on entry to shared memory and multiplied two at a time; tensors in memory stay fp32. Y/dX/dW land within 0.9% of fp32, typically ~0.5%, about 10× looser than NVIDIA's TF32 default, hence opt-in. | winograd-fp16-inner-loop | 3,155 |
| `11-local-knobs` | The environment variables and the partials-hash diagnostic used for the measurements; inert unless set. | — | — |

`patches/pytorch_dlprim/` — the extension itself.

| patch | what |
| --- | --- |
| `01-half-fixes` | Makes convolution reject anything but float32 instead of silently returning NaN; fixes hardtanh/relu6/clamp for half tensors; makes the gradient contiguous in `hardtanh_backward`. Report: pytorch_dlprim-half-tensor-fixes. |
| `02-gelu-mad` | `fma()` → `mad()` in GELU backward. On Mesa < 26.2 rusticl's `fma()` is a 563-instruction software emulation (3.8× on GELU backward). Redundant but harmless on 26.2+. |

### Environment variables

`DLPRIM_CONV_FP16` is added by `patches/dlprimitives/10`, the rest by `11`.
Unset, none of them change anything. The four `*_ALGO` variables override the
algorithm choice whenever `dlprimitives` is left to pick one, which
`pytorch_ocl` always does; a model run through `dlprimitives` directly can
still name an algorithm itself, and that wins. The other variables apply
whenever the Winograd kernels run.

```
DLPRIM_CONV_ALGO               auto | winograd | gemm | depthwise_separable
DLPRIM_CONV_FWD_ALGO           (same, forward only)
DLPRIM_CONV_BWD_DATA_ALGO      (same, backward-data only)
DLPRIM_CONV_BWD_FILTER_ALGO    (same, backward-filter only)
DLPRIM_WINOGRAD_KSPLIT         stock | <n>   stock = the original heuristic
DLPRIM_WINOGRAD_KSPLIT_TARGET  <n>           work-groups per CU to aim for (default 4)
DLPRIM_WINOGRAD_KSPLIT_MAX     <n>           split-K cap (default 16)
DLPRIM_WINOGRAD_SPLIT_PLANES   0 | 1         backward-filter without atomics (default 1
                                             unless the device is NVIDIA or has
                                             cl_ext_float_atomics)
DLPRIM_WINOGRAD_BWD_PLANES     0 | 1         backward-data without atomics
DLPRIM_WINOGRAD_STRIDE_OFFSET  <n>           scratch-tile padding (default 0 on AMD)
DLPRIM_WINOGRAD_TR_OFFSET      <n>           scratch-tile padding, transpose stage (default 1,
                                             but 0 in the backward kernels wherever the
                                             padding would cost a resident work-group -
                                             patch 09). One switch for all three kernels,
                                             so setting it also moves the forward kernel,
                                             which wants 1.
DLPRIM_CONV_FP16               0 | 1         fp16 tiles + packed-fp16 multiply, fp32 tensors
                                             in/out (default 0). Raises the split-K defaults
                                             to 16 / 64 to keep fp16 sums short. Read at
                                             kernel compile time, so set it before the first
                                             convolution.
```

Setting both `*_PLANES` to `0` puts the emulated-atomic backward kernels
back, with the rest of the series still on. This brings us down to 1,191
img/s in `tools/profile-step.py` against 2,436 img/s for the full
series. Note that the first number needs `DLPRIM_WINOGRAD_TR_OFFSET=1`;
the atomic kernels stall in their compare-and-swap loop rather than on
occupancy. With `DLPRIM_WINOGRAD_TR_OFFSET` unset, so that patch `09` drops the
backward kernels' padding, we fall to 801 img/s.

## Using a release build

`pt_ocl.so` replaces the native library inside the stock `pytorch_ocl` 0.2.0
wheel, so you install the wheel first, then put the patched file over it. You
need Python 3.12 on x86-64 Linux, with Mesa's rusticl driver and an OpenCL ICD
loader installed.

1. Create a venv with `torch==2.4.0` (CPU build) and the stock wheel:

   ```
   python3.12 -m venv ~/ocl-venv
   ~/ocl-venv/bin/pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cpu
   ~/ocl-venv/bin/pip install \
     "https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"
   ```

2. Download `pt_ocl.so` from the
   [latest release](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases/latest)
   and check that its `sha256sum` matches the digest on the release page:

   ```
   curl -LO https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases/latest/download/pt_ocl.so
   sha256sum pt_ocl.so
   ```

3. Install it over the stock library and clear the kernel cache, whose
   compiled kernels were built from the old source:

   ```
   install -m0644 pt_ocl.so ~/ocl-venv/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so
   rm -f ~/.dlprimitives/cache.db
   ```

4. Run with `RUSTICL_ENABLE=radeonsi` set. Import `pytorch_ocl` to register the
   device, then move tensors and models to `ocl:0`:

   ```python
   import torch, pytorch_ocl

   dev = torch.device("ocl:0")
   model = torch.nn.Linear(64, 10).to(dev)
   x = torch.randn(32, 64).to(dev)
   print(model(x).sum().item())
   ```

   ```
   RUSTICL_ENABLE=radeonsi ~/ocl-venv/bin/python example.py
   ```

To run the fixes-only baseline instead, do the same with
`pt_ocl-fixes-only.so` from the release (v1.3.0 or later), installed under the
name `pt_ocl.so`; see [Reproducing the benchmarks](#reproducing-the-benchmarks).

Convolutions accept float32 tensors only (see `01-half-fixes`). The first run
of each layer shape is slow because the kernels compile on first use; later
runs read them from `~/.dlprimitives/cache.db`. For the optional fp16 inner
loop, set `DLPRIM_CONV_FP16=1` before the first convolution runs.

## Reverting

Reinstall the stock wheel and clear the cache (for a `build.sh` venv, use
`scratch/venv/bin/pip` instead):

```
~/ocl-venv/bin/pip install --force-reinstall --no-deps \
  "https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"
rm -f ~/.dlprimitives/cache.db
```

## Building

```
./build.sh          # all patches                  -> ./pt_ocl.so
./build.sh fixes    # build/correctness fixes only -> ./pt_ocl-fixes-only.so
```

Requires `python3.12`, `python3.12-devel`, `cmake`, `git`, `sqlite-devel`,
`OpenCL-ICD-Loader-devel`. The script is self-contained:

1. Creates a Python 3.12 venv in `./scratch/venv/` with `torch==2.4.0` (CPU)
   and the `pytorch_ocl` 0.2.0 wheel. Reused between runs; `rm -rf scratch/`
   to start clean.
2. Fetches `pytorch_dlprim` at the commit pinned in `build.sh` (submodule
   checked against its own pin), applies the patches. Bumping the pins
   is a deliberate step: re-check every `git apply`, rebuild, re-run
   `tools/wino-repro.py` and `tools/bn-check.py`.
3. Builds and leaves the stripped extension at `./pt_ocl.so`
   (`./pt_ocl-fixes-only.so` for `./build.sh fixes`, which applies only
   `dlprimitives` `00`–`02` and `pytorch_dlprim` `01`).

The build is reproducible: source paths are mapped out (`-ffile-prefix-map`)
and no RPATH is embedded, so two runs at the pinned commits give byte-identical
output. Shipped builds are attached to
[GitHub releases](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases)
as `pt_ocl.so` and, from v1.3.0, `pt_ocl-fixes-only.so`; anyone can check one by rebuilding at its tag and comparing
`sha256sum`.

## Reproducing the benchmarks

Every speed-up quoted here — and in
[pytorch-gpu-bc250](https://github.com/mxreyer/pytorch-gpu-bc250) and the
[write-up](https://mxreyer.github.io/blog/bc250-ml-box/) — compares the full
series against a build with only the fixes that make `pytorch_ocl` build and
run correctly on rusticl. Stock `pytorch_ocl` cannot serve as that baseline:
its loss, softmax and BatchNorm kernels do not compile here (`00`), so a
training step never completes. The baseline is therefore its own build:

| | `pt_ocl-fixes-only.so` (`./build.sh fixes`) | `pt_ocl.so` (`./build.sh`) |
| --- | --- | --- |
| `dlprimitives` | `00`–`02` | `00`–`11` |
| `pytorch_dlprim` | `01-half-fixes` | `01`, `02` |
| ResNet-9 training, `tools/profile-step.py 128 50` | 916 img/s | 2,436 img/s |

`02-gelu-mad` is left out of the baseline because it is a speed workaround
(for Mesa < 26.2); ResNet-9 has no GELU, so it does not touch the measured step.
The `03`–`11` performance patches are all off in the baseline, not just
disabled via environment variables: the variables from `11` only reach part of
the series.

Both files are attached to each release from v1.3.0 on. To compare them on your
own board, set up the venv as in [Using a release build](#using-a-release-build),
then swap the library and clear the kernel cache before each run:

```
REL=https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases/download/v1.3.0
SITE=~/ocl-venv/lib/python3.12/site-packages/pytorch_ocl
curl -fL -O "$REL/pt_ocl.so" -O "$REL/pt_ocl-fixes-only.so"
sha256sum pt_ocl.so pt_ocl-fixes-only.so   # compare with the release page

for so in pt_ocl-fixes-only.so pt_ocl.so; do
  install -m0644 "$so" "$SITE/pt_ocl.so"
  rm -f ~/.dlprimitives/cache.db
  echo "== $so"
  RUSTICL_ENABLE=radeonsi ~/ocl-venv/bin/python tools/profile-step.py 128 50
done
```

The first run of each build compiles its kernels, so repeat it (the README's
figures are the median of five runs).

## Environment these numbers come from

AsRock BC-250 (gfx1013, 40 CU, 2.0 GHz, 16 GB unified), Linux 7.2.7, Mesa
26.2.3 (Fedora 44), LLVM 22.1.8, `RUSTICL_ENABLE=radeonsi`. Throughput in the
tables is the training step from `tools/profile-step.py`; the T4 row comes from
Google Colab. On Mesa 26.1 the system libclc also cannot
link `sin()`/`cos()` (so `torch.randn` on the device fails); Mesa's own libclc
fork fixes that; `tools/mesa-dev/` builds an image with it.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*

# pytorch_dlprim on gfx1013 (AMD BC-250, Mesa rusticl)

> **Note:** the patches, tools and documentation in this repository were
> heavily authored by Claude ([Claude Code](https://claude.com/claude-code),
> Claude Opus 5). I steered the direction of the investigation and
> validated the results on my BC-250. The performance measurements are real
> and were taken on my board.

The `pt_ocl.so` attached to each
[release](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases) of this
repository is what the notebook image in
[bc250-jupyterhub-opencl-k3s](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s)
installs, pinned by URL and sha256 in its Dockerfile. To ship a new build: run
`./build.sh`, tag a release with `pt_ocl.so` attached, then bump the version
and sha256 there.

Patches and a build script that make
[pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim) (`pytorch_ocl`,
the OpenCL backend for PyTorch) work correctly and run fast on the AsRock
BC-250's gfx1013 GPU under Mesa's rusticl driver — plus the investigation
that got there.

Images per second in the
[bc250-jupyterhub-opencl-k3s benchmark](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s/blob/main/benchmark.py)
(ResNet-9 on CIFAR-10, batch 128, 16 epochs; its
[BENCHMARK.md](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s/blob/main/BENCHMARK.md)
has the T4 runs):

| | training | inference |
| --- | ---: | ---: |
| stock `pytorch_ocl` | 909 | 4,489 |
| **with these patches** | **2,055** | **7,064** |
| + opt-in fp16 inner loop (`DLPRIM_CONV_FP16=1`) | 2,987 | 10,279 |
| NVIDIA T4, fp32, for scale | 2,294 | 7,217 |

Accuracy is unchanged throughout. Gradients match CPU references on every
ResNet-9 layer shape.

| file | description |
| --- | --- |
| [OPENCL-PERF.md](OPENCL-PERF.md) | Claude's full investigation: every measurement, dead end and fix. Long by design; the reference for anyone continuing this work. |
| [HANDOFF.md](HANDOFF.md) | Where things stand, what is still open, the constraints to keep in mind. (For a future Claude session.) |
| `tools/` | Microbenchmarks (`ocl-micro.c`, `libclc-probe.c`, `fp16-micro.c`), correctness sweeps (`wino-repro.py`, `bn-check.py`, ...), a Mesa-from-source container (`mesa-dev/`). |
| `upstream/` | Reports ready to file: three bugs and one proposal for `dlprimitives`, one for `pytorch_dlprim`, two for Mesa/rusticl. |

## The patches

Four patches against `pytorch_dlprim` and its `dlprimitives` submodule.

| patch | against | what |
| --- | --- | --- |
| `custom_reduce.patch` | dlprimitives | **Correctness.** Use the portable work-group reduction instead of the OpenCL 2.0 built-ins rusticl lacks. Without it softmax, cross-entropy, bias gradients and BatchNorm sums fail to compile. |
| `winograd_ksplit.patch` | dlprimitives | **Performance.** All of the convolution work: fill all 40 CUs in the Winograd backward-filter kernel, replace emulated fp32 atomics with plane writes plus a reduce, fix four memory access patterns, prefetch the next K step, opt-in fp16 inner loop. Also builds the activation kernels with their dtype. |
| `pytorch_ocl_half_fixes.patch` | pytorch_dlprim | **Correctness for half tensors.** Reject non-float32 in convolution (it silently returned NaN); dtype-correct hardtanh/relu6/clamp; contiguous grad in `hardtanh_backward`. |
| `gelu_mad.patch` | pytorch_dlprim | **Performance.** `fma()` → `mad()` in GELU backward. On Mesa < 26.2 rusticl's `fma()` is a 563-instruction software emulation (3.8× on GELU backward). Redundant but harmless on 26.2+. |

### Why the reduction fails to compile

`dlprimitives`' reduction kernels call `work_group_reduce_add()` /
`work_group_reduce_max()`. rusticl does not implement that OpenCL C feature
(`__opencl_c_work_group_collective_functions` is absent from
`CL_DEVICE_OPENCL_C_FEATURES`, and no libclc build contains them), so the
kernels fail with "use of undeclared identifier". `dlprimitives` already has a
fallback using `__local` memory and `barrier()` behind `CUSTOM_REDUCE`; the
patch simply turns it on. `tools/libclc-probe.c` checks the feature directly.

### What the convolution patch does

Convolution is 89% of a ResNet-9 training step, and three quarters of that is
the backward pass, so that is where the work went. In order of discovery:

- **Occupancy.** The split-K heuristic for Winograd backward-filter compared
  work-items to cores, so a 128→128 layer ran 16 work-groups on 40 CUs. Now
  it aims for ~4 work-groups per CU. 909 → 972 img/s.
- **Atomics.** Split-K slices accumulated into the filter gradient with
  `atomic_addf`, which rusticl expands into a compare-and-swap loop — 39% of
  the kernel. Each slice now writes its own plane and a small kernel sums
  them. Same trick for backward-data (four parity planes). 972 → 1,374 img/s.
- **Access patterns.** Transformed filters stored `[C][N]` so 32 lanes read
  contiguously; backward-filter lanes mapped to neighbouring tiles of one
  plane with vector loads on the edges; a grid-stride BatchNorm reduction
  instead of one that hit the same cache sets every step. 1,374 → 1,816.
- **Prefetch.** Each Winograd kernel loads the next K step's tiles before the
  current GEMM. 1,816 → 2,055.
- **fp16 inner loop (opt-in).** Tiles converted to fp16 on the way into LDS,
  packed `v_pk_fma_f16` GEMM, fp32 tensors in memory. 2,055 → 2,987, with
  Y/dX/dW within ~0.5% of fp32 — about 10× looser than NVIDIA's TF32 default,
  which is why it is not the default.

Each step was found by profiling and confirmed by first timing a
wrong-but-cheap variant; the measurements are in OPENCL-PERF.md, Findings 2,
3, 10, 11 and 12.

These kernels are dense enough to expose an undervolted clock governor: with
the GPU's idle floor at 1000 MHz / 718 mV they produced wrong gradients in 66
of 100 sweeps, with the governor's default 800 mV in 0 of 900. If a rare wrong
gradient ever shows up, pin the clock and run `tools/wino-repro.py` before
suspecting the code (HANDOFF.md has the procedure).

### Environment variables

Added by `winograd_ksplit.patch`. An algorithm passed explicitly by the caller
always wins, so these are inert unless set.

```
DLPRIM_CONV_ALGO               auto | winograd | gemm | depthwise_separable
DLPRIM_CONV_FWD_ALGO           (same, forward only)
DLPRIM_CONV_BWD_DATA_ALGO      (same, backward-data only)
DLPRIM_CONV_BWD_FILTER_ALGO    (same, backward-filter only)
DLPRIM_WINOGRAD_KSPLIT         stock | <n>   stock = the original heuristic
DLPRIM_WINOGRAD_KSPLIT_TARGET  <n>           work-groups per CU to aim for (default 4)
DLPRIM_WINOGRAD_KSPLIT_MAX     <n>           split-K cap (default 16)
DLPRIM_WINOGRAD_SPLIT_PLANES   0 | 1         backward-filter without atomics (default 1)
DLPRIM_WINOGRAD_BWD_PLANES     0 | 1         backward-data without atomics (default 1)
DLPRIM_WINOGRAD_STRIDE_OFFSET  <n>           LDS padding (default 0 on AMD)
DLPRIM_WINOGRAD_TR_OFFSET      <n>           LDS padding, transpose stage
DLPRIM_CONV_FP16               0 | 1         fp16 LDS tiles + packed-fp16 GEMM, fp32 tensors
                                             in/out (default 0). Raises the split-K defaults
                                             to 16 / 64 to keep fp16 sums short. Read at kernel
                                             compile time, so set it before the first convolution.
```

Setting both `*_PLANES` to `0` brings back the original emulated-atomic kernels
(888 img/s), which is the A/B for the planes paths.

## Building

```
./build.sh
```

Requires `python3.12`, `python3.12-devel`, `cmake`, `git`, `sqlite-devel`,
`OpenCL-ICD-Loader-devel`. The script is self-contained:

1. Creates a Python 3.12 venv in `./scratch/venv/` with `torch==2.4.0` (CPU)
   and the `pytorch_ocl` 0.2.0 wheel. Reused between runs; `rm -rf scratch/`
   to start clean.
2. Fetches `pytorch_dlprim` at the commit pinned in `build.sh` (submodule
   checked against its own pin), applies the four patches. Bumping the pins
   is a deliberate step: re-check every `git apply`, rebuild, re-run
   `tools/wino-repro.py` and `tools/bn-check.py`.
3. Builds and leaves the stripped extension at `./pt_ocl.so`.

The build is reproducible: source paths are mapped out (`-ffile-prefix-map`)
and no RPATH is embedded, so two runs at the pinned commits give byte-identical
output. Shipped builds are attached to
[GitHub releases](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases)
as `pt_ocl.so`; anyone can check one by rebuilding and comparing `sha256sum`.

### Using the result

Drop `pt_ocl.so` over the one in any venv that has the `pytorch_ocl` 0.2.0
wheel and `torch==2.4.0`, then clear the kernel cache:

```
install -m0644 pt_ocl.so <venv>/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so
rm -f ~/.dlprimitives/cache.db     # cached binaries are keyed to the old kernel source
```

Do not `cp` over a `pt_ocl.so` that a running process has loaded: `cp`
rewrites the file in place and the process segfaults. Use `install` (it
unlinks first) or stop the process.

Run with `RUSTICL_ENABLE=radeonsi` in the environment. A quick check:

```
RUSTICL_ENABLE=radeonsi scratch/venv/bin/python tools/wino-repro.py 10   # expect 0 bad
```

### Reverting

Reinstall the stock wheel and clear the cache:

```
scratch/venv/bin/pip install --force-reinstall --no-deps \
  "https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"
rm -f ~/.dlprimitives/cache.db
```

## Environment these numbers come from

AsRock BC-250 (gfx1013, 40 CU, 2.0 GHz, 16 GB unified), Linux 7.1, Mesa
26.1.8 (Fedora 44) on the host and Mesa 26.2.2 (Fedora 45) in containers, LLVM
22.1.8, `RUSTICL_ENABLE=radeonsi`. On Mesa 26.1 the system libclc also cannot
link `sin()`/`cos()` (so `torch.randn` on the device fails); Mesa's own libclc
fork fixes that — see OPENCL-PERF.md, Finding 8.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*

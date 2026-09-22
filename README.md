# pytorch_dlprim on AMD BC-250 (gfx1013) via rusticl

> **Note:** the patches, tools and documentation in this repository were
> heavily authored by Claude ([Claude Code](https://claude.com/claude-code),
> Claude Opus 5). I steered the direction of the investigation and
> validated the results on my BC-250. The performance measurements are real
> and were taken on my board.

Patches and a build script that make
[pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim) (`pytorch_ocl`,
the OpenCL backend for PyTorch) work correctly and run fast on the AsRock
BC-250's gfx1013 GPU under Mesa's rusticl driver — plus the investigation
that got there.

The `pt_ocl.so` attached to each
[release](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases) of this
repository is what the notebook image in
[bc250-jupyterhub-opencl-k3s](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s)
installs, pinned by URL and sha256 in its Dockerfile. To ship a new build: run
`./build.sh`, tag a release with `pt_ocl.so` attached, then bump the version
and sha256 there.

Images per second, ResNet-9 training step on the GPU at batch 128
(`tools/profile-step.py`; the per-patch table below has every step in
between):

| | training |
| --- | ---: |
| stock `pytorch_ocl` | 915 |
| **with these patches** | **2,354** |
| + opt-in fp16 inner loop (`DLPRIM_CONV_FP16=1`) | 2,918 |

Accuracy is unchanged throughout. Gradients match CPU references on every
ResNet-9 layer shape.


| file | description |
| --- | --- |
| [OPENCL-PERF.md](OPENCL-PERF.md) | Claude's full investigation: every measurement, dead end and fix. Long by design; the reference for anyone continuing this work. |
| [HANDOFF.md](HANDOFF.md) | Where things stand, what is still open, the constraints to keep in mind. (For a future Claude session.) |
| `tools/` | Microbenchmarks (`ocl-micro.c`, `libclc-probe.c`, `fp16-micro.c`), correctness sweeps (`wino-repro.py`, `bn-check.py`, ...), a Mesa-from-source container (`mesa-dev/`). |
| `upstream/` | Reports ready to file: five bugs and two proposals for `dlprimitives`, one for `pytorch_dlprim`. |
| `patches/` | Everything `build.sh` applies, per target; the `dlprimitives` series is one patch per report. |

## The patches

`patches/<target>/NN-*.patch`, applied in numeric order by `build.sh`.

**`patches/dlprimitives/`** — the `dlprimitives` submodule. `00` is the
rusticl build fix, `01` the correctness fix it exposes and `02` a second
correctness fix; `03`–`10` are the performance changes, stacked in the order
they were measured; `11` is local only. img/s is the ResNet-9 training step
at batch 128 with the series applied up to that patch.

| patch | what | report in `upstream/` | img/s after |
| --- | --- | --- | ---: |
| `00-custom-reduce` | Adds values across a work-group the portable way, since rusticl lacks the OpenCL 2.0 built-in for it; without this, softmax, cross-entropy, bias gradients and BatchNorm sums do not compile at all | custom-reduce-autodetect | — |
| `01-reduce-barrier` | Adds a barrier after the portable reduction hands out its result, so a kernel that reduces twice in a row (softmax: max, then sum) cannot have work-item 0 start the second one and overwrite the first result before slower work-items have read it; the `dlprimitives` softmax and log_softmax tests fail without it | reduce-barrier | — |
| `02-activation-dtype` | Builds the activation kernel for the tensor's own type; relu/tanh/sigmoid/relu6 on a half tensor were reading it as float and returning garbage | activation-half-dtype | 915 (stock) |
| `03-winograd-ksplit-heuristic` | Decides how far to split the backward-filter work by counting work-groups per compute unit; the old rule left 24 of 40 CUs idle on a 128→128 layer | winograd-ksplit-heuristic | 984 |
| `04-winograd-no-atomics` | Gives each parallel slice its own scratch plane and sums the planes afterwards, instead of every slice fighting over the same memory through emulated float atomics | winograd-performance §1 | 1,477 |
| `05-winograd-fwd-filter-layout` | Stores the transformed filters `[C][N]`, so neighbouring lanes read neighbouring addresses | winograd-performance §2 | 1,634 |
| `06-winograd-bwd-filter-loads` | Points the 32 lanes at neighbouring tiles of one image plane rather than at 32 different planes; vector loads on the edge tiles | winograd-performance §2 | 1,815 |
| `07-bn-sums-grid-stride` | Spreads the BatchNorm sum across memory instead of walking it in per-lane chunks that hit the same cache sets every step | winograd-performance §2 | 1,949 |
| `08-winograd-prefetch` | Starts the next slice's loads before the current slice's arithmetic, so the wait for memory overlaps with work | winograd-performance §3 | 2,068 |
| `09-winograd-tr-offset` | Drops the scratch-tile padding in the backward kernels, where the 8 KiB it costs was worth a second resident work-group per CU; forward keeps it | winograd-lds-padding | 2,261 |
| `10-winograd-fp16` | Opt-in: half-precision tiles and multiply-accumulate inside the kernel, fp32 tensors in memory | winograd-fp16-inner-loop | 2,354; 2,918 with `DLPRIM_CONV_FP16=1` |
| `11-local-knobs` | The environment variables and the partials-hash diagnostic used for the measurements; inert unless set | — | — |

**`patches/pytorch_dlprim/`** — the extension itself.

| patch | what |
| --- | --- |
| `01-half-fixes` | Makes convolution reject anything but float32 instead of silently returning NaN; fixes hardtanh/relu6/clamp for half tensors; makes the gradient contiguous in `hardtanh_backward`. Report: pytorch_dlprim-half-tensor-fixes. |
| `02-gelu-mad` | `fma()` → `mad()` in GELU backward. On Mesa < 26.2 rusticl's `fma()` is a 563-instruction software emulation (3.8× on GELU backward). Redundant but harmless on 26.2+. |

### Why the reduction fails to compile

Softmax, cross-entropy, bias gradients and BatchNorm all need to add up one
value across all 256 work-items of a group. `dlprimitives` does that with
`work_group_reduce_add()`, a built-in that OpenCL 2.0 required and OpenCL 3.0
made optional. rusticl does not provide it — the feature is missing from
`CL_DEVICE_OPENCL_C_FEATURES` and no libclc build contains it — so those
kernels fail to compile with "use of undeclared identifier".

`dlprimitives` already carries a portable fallback that computes the same sum
through shared memory and a barrier, behind `CUSTOM_REDUCE`; `00-custom-reduce`
turns it on. `tools/libclc-probe.c` checks the feature directly.

That fallback has a race of its own, which nothing with the built-in would
ever see: after the last barrier every work-item reads the result from slot 0
of the shared array, and nothing stops work-item 0 from starting the *next*
reduction and overwriting slot 0 first. Softmax reduces twice back to back
(max, then sum), and its unit tests fail. `01-reduce-barrier` adds the missing
barrier.

### What the convolution patches do

Convolution is 89% of a ResNet-9 training step, and four fifths of that is
the backward pass, so that is where the work went. In the order the series
applies them:

- **Occupancy** (`03`). The backward-filter kernel can cut its work into
  slices and spread them over more of the GPU, and a rule decides when that is
  worth doing. The rule compared a count of *work-items* against a core count,
  which on a 40-CU AMD device works out as "split only if there are fewer than
  10 work-groups". A 128→128 layer launches 16, so it ran on 16 compute units
  with 24 idle. It now counts work-groups and aims to give each CU about four.
- **Atomics** (`04`). Both backward kernels have many work-groups adding into
  the same output values, so they used an atomic add to keep those additions
  from stepping on each other. This GPU has no hardware float atomic add — no
  RDNA GPU before RDNA3 does — so each one becomes a retry loop: read the
  value, add to it, try to write it back, start over if another lane got
  there first. That loop was 39% of the backward-filter kernel. But
  the atomics were only guarding collisions *between* the parallel slices;
  inside one slice every output has exactly one writer. Each slice now writes
  into its own plane of scratch memory and a small second kernel adds the
  planes together, for under 2% of the kernel it replaces. Backward-data gets
  the same treatment: its 4×4 tiles do overlap, but sorting them by whether
  their row and column are odd or even gives four groups whose members never
  touch. Devices with a real float atomic add (NVIDIA, or anything advertising
  `cl_ext_float_atomics`) keep the original path, where atomics are cheap.
- **Access patterns** (`05`, `06`, `07`). Three places where neighbouring
  lanes read far-apart addresses, so each read pulled in a cache line to use a
  few bytes of it. The transformed filters are now stored `[C][N]`, which makes
  the 32 lanes that read one input channel contiguous; backward-filter lanes
  now map to neighbouring tiles of one image plane instead of to 32 different
  planes, with vector loads on the edges; and the BatchNorm sum walks memory
  with a stride rather than in per-lane chunks that kept landing in the same
  cache sets (13 GB/s out of 359).
- **Prefetch** (`08`). Each kernel used to load a slice of data, wait for it,
  multiply, then load the next. It now issues the next slice's loads *before*
  doing the current multiply, so the wait for memory overlaps with arithmetic
  instead of stalling on it.
- **Scratch padding** (`09`). The kernels pad their tiles in shared on-chip
  memory (LDS) so that rows do not land on the same memory bank. The catch:
  switching any padding on also makes the kernel allocate 16 extra tile rows,
  40 KiB per work-group instead of 32, and that is just over the line where
  two work-groups fit on a compute unit at once — so the padding was costing
  half the parallelism (`tools/ocl-micro.c occ` measures the boundary).
  Giving it up to win the second work-group back pays off in the two backward
  kernels and does not in the forward one, which keeps its padding.
- **fp16 inner loop, opt-in** (`10`). Tiles are converted to half on the way
  into shared memory and multiplied two at a time; the tensors in memory stay
  fp32. Y/dX/dW land within ~0.5% of the fp32 result — about 10× looser than
  NVIDIA's TF32 default, which is why it is opt-in rather than on.

### Environment variables

`DLPRIM_CONV_FP16` is added by `patches/dlprimitives/10`, the rest by `11`.
These are read only when the caller asks for auto; an explicitly passed algorithm wins. Unset, they change nothing.

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
back, with the rest of the series still on. This brings us down to 1,129
img/s in `tools/profile-step.py` against 2,330 img/s for the full
series. Note that the first number needs `DLPRIM_WINOGRAD_TR_OFFSET=1`;
the atomic kernels stall in their compare-and-swap loop rather than on
occupancy. Without the scratch padding patch `09` gives up, we fall to
792 img/s.

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
   checked against its own pin), applies the patches. Bumping the pins
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

# What is actually slow on the BC-250's OpenCL path

[BENCHMARK.md](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s/blob/main/BENCHMARK.md)
in the bc250-jupyterhub-opencl-k3s repository had ResNet-9 training 2.5×
slower than a cloud T4 and ended on a claim: *"The gap is software, not
silicon."* This is that claim taken apart. Every number below was measured on
the box — 40 CU, 2.0 GHz, kernel 7.1.13; Fedora 44 / Mesa 26.1.8 on the host,
and from 2026-09-10 Fedora 45 / Mesa 26.2.2 in a container. The img/s figures
are from that repository's
[benchmark.py](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s/blob/main/benchmark.py)
(ResNet-9 on CIFAR-10, batch 128, 16 epochs, CPU-side augmentation);
"isolated step" numbers are the GPU step alone (`tools/profile-step.py`).

The findings that turned into fixes are worth **2.5×** on training throughput
(909 → 2,244 img/s) and 1.5× on inference (4,489 → 6,774) at fp32 — within
1.02× and 1.07× of a T4; an opt-in fp16 inner loop (Finding 11) takes that to
2,768 and 8,346, past the T4's fp32 numbers. (Those endpoints were measured in
different sessions and environments; README notes which, and every A/B below
is a pair measured together.) The largest of them spent two
weeks switched off behind what looked like a code generation bug in the
graphics driver; it was the clock governor's idle voltage, and the story of
being wrong about that — and how it was finally settled — is the most useful
thing here (Finding 3). Several popular explanations turned out to be wrong,
and those are worth as much as the fixes. The last section accounts for the
gap that remains once they are all subtracted, which is mostly not where it
looks like it is.

> Reproduce the hardware measurements with
> `tools/ocl-micro.c`:
> ```
> gcc -O2 -o ocl-micro tools/ocl-micro.c -lOpenCL
> RUSTICL_ENABLE=radeonsi ./ocl-micro          # or: info alu bw cache lds atomics lat launch xfer
> ```
> Run the full sweep rather than a single test: the GPU idles at 1000 MHz and
> takes a moment to reach 2000 MHz, so a cold single test measures the wrong
> machine (dependent load latency reads ~440 ns cold against ~300 ns at full
> clock, and LDS bandwidth roughly halves).

## Summary

| # | Finding | Where | Status |
| - | --- | --- | --- |
| 1 | `fma()` compiles to a **563-instruction software emulation**, 120× slower than `mad()` | rusticl < 26.2 (never tells libclc the device has hardware fma) | fixed in our kernels (`02-gelu-mad.patch`); native on Mesa 26.2 |
| 2 | Winograd backward-filter launches **16 work-groups onto 40 CUs** | dlprimitives heuristic | fixed (`patches/dlprimitives/02-winograd-ksplit-heuristic.patch`) |
| 3 | Both backward kernels run on **emulated float atomics** — a third of the training step. Removing them is worth **+46%**. The "ACO `s_waitcnt` bug" that kept the fix gated for two weeks was the **clock governor's idle-floor undervolt** (1000 MHz @ 718 mV) | half silicon, half `dlprimitives`, then a config file | **fixed and shipping**: 1,374 img/s, no driver flags; voltage restored to the governor default |
| 4 | rusticl **misreports** LDS and cache as absent | rusticl device info | no impact on this stack; not pursued |
| 5 | Throttling / launch overhead / memory bandwidth / wrong conv algorithm | — | **all ruled out** |
| 6 | The remaining gap is **the backward passes**, running at half the forward pass's efficiency; a T4 reaches 64% of its paper number where this reached 21.6% at the time (45.7% now) | `dlprimitives` kernels | measured (this corrects an earlier wrong conclusion) |
| 7 | Wave size is per-kernel, and rusticl picks one globally: `AMD_DEBUG=w32cs` is **+13% on GEMM, −5% on this conv workload** | rusticl / radeonsi | measured; matters for matmul-heavy work |
| 8 | Fedora's libclc **cannot link `sin()`, `cos()`, `tan()`, `fma()`, `remquo()`** — `torch.randn(device="ocl:0")` fails to compile | Fedora `libclc-spirv` 22.1.8 (F44 and F45) | **fixed**: Mesa's patched libclc as a drop-in (also removes the start-up warning) |
| 9 | Image moved to **Fedora 45 / Mesa 26.2.2**: `fma()` native, everything correct, training within noise of 44 | Fedora 45 | shipping (Finding 3, *On Mesa 26.2…*; Finding 8; HANDOFF.md, *Done, for the record*) |
| 10 | With the profile finally honest, four **memory access patterns**: a strided transformed-kernel read in forward Winograd (39% of that kernel), lane-to-channel mapping and scalar edge loads in backward-filter, and a BatchNorm reduction that hit the same cache sets every step (13 GB/s) | `dlprimitives` kernels | **fixed and shipping**: 1,460 → **1,963 img/s** in the isolated step; **1,816** over 16 epochs, inference **4,433 → 6,418** |
| 11 | **fp16 inner loop for the Winograd kernels** (`DLPRIM_CONV_FP16=1`, opt-in): packed `v_pk_fma_f16` runs at 91% of the 2× fp16 peak through rusticl; fp32 tensors in and out, fp16 LDS tiles and accumulation over one K slice | `dlprimitives` kernels | **shipping, opt-in**: 2,714 img/s training (acc 0.928), **9,584 inference** — both past the T4's fp32 numbers; Y/dX/dW within ~0.5% of fp32 |
| 12 | **Software pipelining**: prefetching the next K step's tiles into registers before the GEMM | `dlprimitives` kernels | **shipping**: 2,055 img/s / 7,064 inference at fp32 (T4: 2,294 / 7,217); 2,987 / 10,279 in fp16 mode. LDS double-buffering and fp32-accumulate mixed precision measured and rejected |
| 13 | The Winograd **transpose padding costs 8 KiB of `__local`**, which is a resident work-group per CU; the two backward kernels are better off without it and the forward kernel is not | `dlprimitives` host code — one `#define` | **shipping**: 60.6 → 53.8 ms of GPU time per step, **1,993 → 2,244 img/s** over 16 epochs, inference unchanged. Against the *stock* backward kernels the same change is a loss |

## First: where does the time actually go?

`pytorch_ocl` has a built-in per-kernel profiler (`torch.ocl.enable_profiling`
→ `torch.ocl.profile`). One ResNet-9 training step, batch 128, on the stock
stack, is **301 kernel launches**. Aggregated over 10 steps:

| aten op | % of GPU busy time |
| --- | ---: |
| `convolution_backward_overrideable` | 71.7% |
| `convolution_overrideable` | 17.5% |
| `native_batch_norm_backward` | 4.5% |
| `native_batch_norm` | 2.3% |
| everything else (activation, pooling, adds, loss, optimizer) | 4.0% |

**89% of the step is convolution**, and three quarters of that is the backward
pass. Anything that isn't a conv kernel is noise. That single table redirected
the whole investigation.

Summing the individual kernel times gives 139.2 ms/step, and an unprofiled step
takes 139.2 ms of wall clock. The GPU is **saturated** — it is not sitting idle
waiting for the host, and there is no dispatch bubble to reclaim. (Profiling
itself is not free: it forces per-kernel events and stretches the wall-clock
step to ~190 ms. Use the profile for the *breakdown*, never for the total.)

## Finding 1 — `fma()` is a software float emulation (120×)

`tools/ocl-micro.c alu` runs the same dependent-FMA loop five ways:

| op | GFLOP/s |
| --- | ---: |
| `fma(x,b,c)` | **70.7** |
| `mad(x,b,c)` | 8,451 |
| `x*b + c` | 8,194 |
| `x*b` | 4,292 |
| `x+c` | 4,367 |

`mad()` reaches 8.45 of the 10.24 TFLOP/s theoretical peak — 83%, the rest being
loop overhead. **The ALU is fine.** But `fma()`, which should be the exact same
`v_fma_f32` instruction, is **120× slower**. That figure is not noise: across
runs `mad()` varies by a few percent while `fma()` sits at 70.7 GFLOP/s every
single time.

The reason is in the libclc that rusticl loads
(`/usr/lib64/clc/spirv64-mesa3d-.spv`, from Fedora's `libclc` package):

```
$ spirv-dis /usr/lib64/clc/spirv64-mesa3d-.spv | grep -A20 '%_Z3fmafff = OpFunction'
  %_Z3fmafff = OpFunction %float DontInline %14621
       ...
      %22446 = OpFunctionCall %float %_Z12__clc_sw_fmafff %22439 %22440 %22441
               OpReturnValue %22446
```

Every `fma()` is an un-inlined call into `__clc_sw_fma` — **563 SPIR-V
instructions** of software float emulation. The reason is a hand-off that never
happens: libclc's SPIR-V build cannot know the device, so `__clc_fma` asks a
`noinline` stub, `__clc_runtime_has_hw_fma32()`, which returns `false` in the
library and is *meant to be redefined by the OpenCL runtime at link time*. Mesa
26.1.8 does not do that; Mesa 26.2 does (`rusticl_insert_libclc_config()` in
`rusticl_nir.c`, driven by `nir_has_ffma()`). So this is a rusticl version
issue, not a libclc packaging one, and swapping in Mesa's patched libclc does
not change it — measured under both, `fma()` stays 15–18× slower than `mad()`
(`tools/libclc-probe.c`). Confirmed on Fedora 45 / Mesa 26.2.2: the same probe
reports `fma()` at 1.0× `mad()` — 893 vs 894 GFLOP/s — with either libclc.
`02-gelu-mad.patch` is redundant there (and harmless). (An earlier version of this section tied it to the *"Patched Mesa
libclc not detected"* warning; see Finding 8 for what that warning actually
covers.)

**Impact here is narrow but real.** `dlprimitives`' own `.cl` kernels never call
`fma()` — its GEMM and Winograd inner loops already use `mad()`. The only place
`pytorch_ocl` uses it is **GELU backward**:

| | before | after |
| --- | ---: | ---: |
| `gelu(approximate="tanh")` backward, 8M elements | 3.38 ms | **0.88 ms** |
| `gelu(approximate="none")` backward, 8M elements | 1.80 ms | **1.10 ms** |
| `tanh` backward (reference, no `fma`) | 0.77 ms | 0.77 ms |

`02-gelu-mad.patch` swaps the five `fma()`
calls for `mad()`, and GELU backward drops to the speed of any other
memory-bound elementwise op, where it belongs. Gradients still match CPU to
1.2e-7 relative.

ResNet-9 uses ReLU, so **this changes nothing in the benchmark** — but GELU is
the activation in every transformer, so it is a trap sitting directly under the
most likely next workload. It is also a trap for anyone writing their own
OpenCL on this box: on rusticl, write `mad()` or `a*b+c`, never `fma()`.

## Finding 2 — the Winograd backward-filter kernel only fills part of the GPU

`dlprimitives` picks Winograd for every 3×3/stride-1/pad-1 convolution. Its
backward-filter kernel launches one work-group per (32 input channels × 32
output channels) tile, optionally splitting the batch×tiles reduction 8 ways to
create more:

```cpp
int winograd_work_items = (config_.channels_in / 32) * (config_.channels_out / 32) * 256;
reduce_k_ = winograd_work_items < ctx.estimated_core_count();
```

On a 40 CU AMD device `estimated_core_count()` is `40*64 = 2560`, and
`winograd_work_items` counts *work-items* (256 per work-group). So the split
turns on only when the launch has **fewer than 10 work-groups**. A 128→128
convolution has `(128/32)*(128/32) = 16` work-groups — over the threshold, so it
runs unsplit, **16 work-groups on 40 CUs, leaving 24 of them idle for the whole
kernel**.

The image size never enters the formula at all, so the amount of reduction work
available to split is not considered either.

`patches/dlprimitives/02-winograd-ksplit-heuristic.patch` picks the
split from how many work-groups the launch actually has versus how many CUs
there are to fill (aiming for ~4 work-groups per CU), and refuses to split
further than there is K work to divide. Per-call, averaged over 10 steps:

| layer (Cin→Cout @ HxW) | work-groups | stock | patched | speedup | stock TF | patched TF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512→512 @4×4 | 256 | 3.48 ms | 3.37 ms | 1.03× | 2.78 | 2.87 |
| 512→512 @4×4 | 256 | 3.45 ms | 3.36 ms | 1.03× | 2.80 | 2.88 |
| 256→512 @8×8 | 128 | 7.44 ms | 7.16 ms | 1.04× | 2.60 | 2.70 |
| 128→256 @16×16 | 32 | 9.20 ms | 8.66 ms | 1.06× | 2.10 | 2.23 |
| **128→128 @16×16** | **16** | **8.12 ms** | **4.48 ms** | **1.81×** | 1.19 | 2.16 |
| **128→128 @16×16** | **16** | **8.20 ms** | **4.47 ms** | **1.83×** | 1.18 | 2.16 |
| 64→128 @32×32 | 8 | 11.79 ms | 12.24 ms | 0.96× | 1.64 | 1.58 |
| 3→64 @32×32 | 2 | 3.64 ms | 2.39 ms | 1.52× | 0.12 | 0.19 |
| **total per step** | | **55.31 ms** | **46.11 ms** | **1.20×** | | |

The two 16-work-group layers behave exactly as predicted: they were running at
16/40 of the machine, and filling the machine nearly doubles them.

The 64→128 layer is a small **regression** (0.96×) — it has 8 work-groups, the
stock rule already split it 8 ways, and the new rule splits it 16, which costs
more in atomic contention than it gains in occupancy. Splitting further always
multiplies the number of atomic accumulations into the same filter, so there is
a real optimum and it is not always "more". Left as-is rather than special-cased
to one network's shapes; it costs 0.45 ms against 9.2 ms gained.

**End to end this is +6.9%.** The full 16-epoch benchmark, both runs from the
same binary with `DLPRIM_WINOGRAD_KSPLIT=stock` vs the new default:

| | train img/s | s/epoch | test acc | epochs to 90% | infer bs128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock heuristic | 909 | 55.0 | 0.925 | 14 | 4,511 img/s |
| adaptive K-split | **972** | **51.4** | 0.925 | 14 | 4,496 img/s |

Accuracy and convergence are unchanged, and the stock run reproduces the
figures published in BENCHMARK.md (919 img/s / 54.4 s / 0.924 accuracy) to
within run-to-run variance. Inference is untouched, as expected — there is no
backward-filter kernel in an inference pass. Gradients match CPU to better than
1e-5 relative for every ResNet-9 layer shape.

`09-local-knobs.patch` adds environment variables to make these choices
measurable instead of assumed — `DLPRIM_CONV_ALGO`, `DLPRIM_CONV_FWD_ALGO`,
`DLPRIM_CONV_BWD_DATA_ALGO`, `DLPRIM_CONV_BWD_FILTER_ALGO`
(`auto`|`winograd`|`gemm`|`depthwise_separable`) and `DLPRIM_WINOGRAD_KSPLIT`
(`stock`|*n*).

## Finding 3 — the emulated atomics, of which half were avoidable

> **Resolved 2026-09-10 — read the last section of this finding first.** The
> "ACO `s_waitcnt` bug" that the middle of this finding builds up to, and that
> kept the atomics-free kernels behind a 30%-tax workaround, was not a compiler
> bug. It was the GPU clock governor's idle-floor voltage: 1000 MHz at 718 mV
> is below what this chip needs, and kernels that land there read one lane of
> LDS wrong. With the shipped default voltage restored the kernels are correct
> (0 bad / 900 sweeps, 300 of them pinned at 1000 MHz) and they now **ship
> enabled**: **1,374 img/s**, up from 972, with no driver flags and no
> inference penalty. The sections between here and *"It was the voltage"* are
> left as written because the elimination work in them is still valid and the
> reasoning error is worth seeing.

Forward and backward-data convolution do the same amount of arithmetic with the
same launch geometry, yet backward-data takes **1.78× as long** as the forward
kernel. The difference is in the epilogue: `winograd_bwd_data.cl` writes **every
output element** through `atomic_addf`, because Winograd F(2×2,3×3) produces
overlapping 4×4 input-gradient tiles that have to be accumulated.
`winograd_fwd.cl` contains no atomics at all.

To find out what that costs, replace the `atomic_addf` with a plain `+=` — a
racy, numerically wrong kernel, but a valid speed-of-light measurement:

| layer (Cin→Cout @ HxW) | forward | backward-data | same, atomics removed | atomics share |
| --- | ---: | ---: | ---: | ---: |
| 512→512 @4×4 | 2.49 ms | 4.13 ms | 1.56 ms | 62% |
| 512→512 @4×4 | 2.49 ms | 4.10 ms | 1.56 ms | 62% |
| 256→512 @8×8 | 4.41 ms | 7.66 ms | 3.02 ms | 61% |
| 128→256 @16×16 | 4.59 ms | 8.06 ms | 3.79 ms | 53% |
| 128→128 @16×16 | 2.32 ms | 4.75 ms | 2.64 ms | 44% |
| 128→128 @16×16 | 2.33 ms | 4.77 ms | 2.62 ms | 45% |
| 64→128 @32×32 | 5.04 ms | 8.70 ms | 3.89 ms | 55% |
| **total per step** | **23.66 ms** | **42.17 ms** | **19.08 ms** | **55%** |

**55% of the backward-data time is the emulated atomics.** And the same test on
`winograd_bwd_filter` says **39% of that kernel** is atomics too (46.1 → 28.0 ms).
Together they are ~41 ms of a 128 ms step — **a third of training**.

Measured as total GPU kernel time per step, from profiled runs so the
denominators match:

| | GPU time per step |
| --- | ---: |
| stock | 137.75 ms |
| with `patches/dlprimitives/02-winograd-ksplit-heuristic.patch` | 128.32 ms |
| …and atomics removed (speed-of-light, incorrect results) | **106.32 ms** |

The atomics cost **22.0 ms, or 17% of GPU time per step**. Strip them and
backward-data drops to 0.81× the forward kernel, which is where the launch
geometry says it belongs. That 17% is what this stack would gain, for free and
with no code change at all, on any GPU with a hardware fp32 atomic add.

`dlprimitives`' `atomic_addf` uses a native fp32 atomic add where one exists and
otherwise falls back to a compare-and-swap retry loop. rusticl advertises only
integer atomics, so on the BC-250 it is always the CAS loop. What that costs
(`ocl-micro atomics`):

| | plain store | CAS-loop float add | native int atomic add |
| --- | ---: | ---: | ---: |
| uncontended | 20.1 Gop/s | 14.4 Gop/s | 19.7 Gop/s |
| 64K slots | 20.7 Gop/s | 14.3 Gop/s | 20.2 Gop/s |
| 4K slots (heavy contention) | 20.5 Gop/s | **3.3 Gop/s** | 19.5 Gop/s |

Native *integer* atomics are free — they run at plain-store speed even under
heavy contention. The emulated float add costs 1.4× uncontended and **6× when
contended**.

The obvious question is whether rusticl is simply failing to expose a hardware
instruction. It is not:

```
$ cat fadd.ll
define amdgpu_kernel void @k(ptr addrspace(1) %p, float %v) {
  %r = atomicrmw fadd ptr addrspace(1) %p, float %v syncscope("agent") monotonic,
       align 4, !amdgpu.no.fine.grained.memory !0, !amdgpu.ignore.denormal.mode !0
  ret void
}
!0 = !{}

$ for g in gfx1013 gfx1030 gfx1100 gfx90a; do
    printf "%-8s " $g; llc -mtriple=amdgcn-amd-amdhsa -mcpu=$g fadd.ll -o - |
    grep -oE 'global_atomic_(add_f32|cmpswap)' | head -1; done
gfx1013  global_atomic_cmpswap
gfx1030  global_atomic_cmpswap
gfx1100  global_atomic_add_f32
gfx90a   global_atomic_add_f32
```

`global_atomic_add_f32` arrives with RDNA3 (gfx11) and exists on CDNA. **gfx1013
genuinely does not have it**, so neither RDNA1 nor RDNA2 can do better than the
CAS loop. rusticl is reporting the truth, and no driver setting will change it.
Fixing it means restructuring the kernel to avoid overlapping accumulation —
upstream `dlprimitives` work. Swapping in the GEMM path instead is worse (see
Finding 5).

### Only half of them are the hardware's fault — but the fix does not ship

> **Correction.** An earlier version of this section reported the change below
> as landed and worth +23% end to end. It is worth that, and it is wrong often
> enough to be unusable. It now ships **disabled**, behind an environment
> variable, and the measurements are kept because the result is real and the
> failure is interesting.

The missing instruction is real. Needing it is not always.

`winograd_bwd_data` genuinely needs accumulation: Winograd F(2×2,3×3) produces
overlapping 4×4 input-gradient tiles, four tiles touch every output pixel, and
they are spread across work-groups. That one is stuck behind the missing
instruction.

`winograd_bwd_filter` is a different story. Its atomics exist **only** to combine
the split-K slices. Within a single slice the (x,y) work-group grid covers each
`(channel_in, channel_out)` pair exactly once, so every filter element is written
by exactly one work-item — a plain store was always sufficient. The atomic was
paying for cross-slice accumulation, which does not need atomics at all: give
each slice its own plane of the workspace and sum the planes afterwards.

That is what `DLPRIM_WINOGRAD_SPLIT_PLANES=1` does:
`Conv2DBackwardFilterWinograd::workspace()` asks for `k_split × 9 × C × N`
floats, each slice writes its plane with plain stores, and a ten-line reduction
kernel sums them into the gradient applying `beta`.

The same argument applies to `winograd_bwd_data`, whose overlap is real but
regular: tiles of equal row- and column-parity are 4 apart in each direction, so
within one of the four parity classes the patches are exactly disjoint. Four
planes, plain stores, one reduction — `DLPRIM_WINOGRAD_BWD_PLANES=1`.

Both work, and both are fast:

| per training step | |
| --- | ---: |
| `winograd_bwd_filter`, atomics | 46.1 ms |
| `winograd_bwd_filter`, split-K planes | **27.8 ms** |
| ResNet-9 step, default | 130 ms (972 img/s) |
| …with backward-filter planes | 112 ms (1,118 img/s) |
| …with both | **88 ms (1,453 img/s)** |

**And both are wrong, rarely.** Running the six ResNet-9 layer shapes against
CPU references, many times:

| build | bad gradients |
| --- | ---: |
| pristine upstream | 0 / 200 sweeps |
| adaptive K-split, atomics (what ships) | 0 / 400 sweeps |
| + backward-filter split planes | 5 / 200, 4 / 250, 7 / 200 sweeps |
| + backward-data parity planes only | 1 / 250 sweeps |

Always `dW`, always on the widest layers, always garbage magnitude (relative
error ~1e7) rather than a small numerical difference — an uninitialised read,
roughly 1–3% of invocations.

### Where it actually is

Reading the partials back and hashing them settles the one question that
mattered: with identical inputs they must be bit-identical, so a divergence
localises the corruption to before or after the reduction.

```
distinct partials hashes over 150 sweeps, per layer:
    128->128   1 distinct   x150
    128->256   1 distinct   x150
    3->64      1 distinct   x150
    64->128    1 distinct   x150
    256->512   9eacf640...  x146     <- and four one-off hashes
```

Four unique hashes, four failures, same sweep. **The corruption is upstream of
the reduction** — the partials are already wrong when the main kernel finishes.

Diffing a bad set against a good one is sharper still:

```
PARTIALS-DIFF 256->512/k2: 9 of 2359296 differ; per-plane[9,0];
  first idx=1037520 (plane 0, ch_out=450 ch_in=80 tap=0);
  got=2.299923e+09 want=-1.853797e+01
```

**Exactly nine elements** — one `(ch_out, ch_in)` pair's 3×3 taps, which is one
work-item's entire output. Across five captured failures the wrong element is
always at the same position *within* the 32×32 channel tile (`ch_out%32 == 2`,
`ch_in%32 == 16`), i.e. always the same lane of the work-group, not a random
scatter. The value is ~2–4e9 garbage where the truth is order 10.

That value comes from `transform_kernel_bwd(s_kern16, s_kern9)`, and `s_kern16`
is a 16-float read out of `__local`. So: **one lane, intermittently, reads stale
LDS.**

The source-level synchronisation around that read is correct — I checked it
rather than assumed:

- the K loop is `write LDS → barrier → read LDS → barrier`, and `k`/`K_limit`
  are uniform across the work-group, so no divergent barriers;
- the epilogue is `write s_img_tile → barrier → read → barrier` per `dc_split`;
- the writes cover all 16 `indx` slots the `vload16` reads (the 16 work-items
  sharing `lid%16` each write one), and the whole access stays inside
  `wg_local_memory`.

And it is not the instruction form on either side:

| variant | bad / 150 sweeps |
| --- | ---: |
| plain store, `vload16` (as written) | 3–7 |
| `atomic_xchg` store instead | 6 |
| 16 scalar LDS reads instead of `vload16` | 9 |

So a correctly-synchronised LDS read returns stale data at one lane, roughly 2%
of the time. That is a hazard **below the source level**, and ACO's own debug
switches say which one:

| variant | bad / 150 sweeps |
| --- | ---: |
| `ACO_DEBUG=validateir,validatera` | 4 — the validators do not fire |
| `ACO_DEBUG=nosched` | 20 — *worse* |
| **`ACO_DEBUG=force-waitcnt`** | **0** |

`force-waitcnt` makes ACO emit conservative `s_waitcnt`, and the failure
disappears outright — **0 bad over 400 sweeps**. `nosched` making it worse fits
the same story: reshuffling the schedule moves a latent wait-count problem
around rather than fixing it.

**This is an ACO code generation bug — an `s_waitcnt` missing before an LDS read
on gfx1013.** Not a `dlprimitives` logic error, and not one I introduced: the
upstream kernel is correct by luck of instruction scheduling, and changing the
store was enough to lose that luck.

### Two things that make the workaround much less attractive

> **Correction.** The first version of this section reported the workaround as
> free, at 1,457 img/s. Both halves of that were wrong, and the reason is worth
> keeping on the record.

**`ACO_DEBUG` is not part of Mesa's on-disk shader cache key.** A warm cache
serves binaries compiled *without* `force-waitcnt`, so the workaround silently
does not apply — and the failure comes straight back:

| | bad / 200 sweeps |
| --- | ---: |
| `force-waitcnt`, shader cache on (default) | 4 |
| `force-waitcnt`, `MESA_SHADER_CACHE_DISABLE=true` | **0** |

Every "fixed" performance figure I first reported was measured through that warm
cache, i.e. with the bug still active. Those numbers were the *unfixed* speed.

**And with the fix genuinely applied, `force-waitcnt` costs about a third of the
machine.** Three repeats each, shader cache disabled throughout:

| | img/s |
| --- | ---: |
| safe path, no `force-waitcnt` | 975, 977, 984 |
| safe path **+ `force-waitcnt`** | 642, 659, 647 |
| atomics-free paths + `force-waitcnt` | 1072, 1071, 1070 |

Conservative wait-counts are a global driver setting: they slow every shader, not
just the one with the bug. The atomics-free kernels are worth roughly +65% on
their own, and `force-waitcnt` gives about a third of that back.

### Where that leaves it

| | img/s | correctness |
| --- | ---: | --- |
| stock | 909 | clean |
| **shipping default** (occupancy fix only) | **972** | 0 bad / 400 sweeps |
| atomics-free paths alone | ~1,453 | **~2% wrong gradients — unusable** |
| atomics-free + `force-waitcnt` + cache disabled | **1,048** | 0 bad / 250 sweeps |

(the last row from the full 16-epoch benchmark; the isolated training step reads
~1,071.) So the honest gain from the whole atomics investigation is **+7.8% over
the shipping default**, not the +50% it looked like — the ACO bug taxes it
heavily.

And it is worse than that, because the tax is driver-wide while the benefit is
not. Inference is forward-only: it never touches the two backward kernels the
atomics work improves, so it pays the cost and gets nothing. Measuring the two
driver flags separately on a pure inference pass, warm, alternating:

| batch-128 inference | run 1 | run 2 |
| --- | ---: | ---: |
| neither flag | 4,512 | 4,532 img/s |
| `MESA_SHADER_CACHE_DISABLE=true` only | 4,528 | 4,510 img/s |
| **+ `ACO_DEBUG=force-waitcnt`** | **3,154** | **3,149 img/s** |

The cache flag is free — it only affects start-up compilation, not steady state.
**The whole 30% is `force-waitcnt`.** (Batch-1 latency swings 3.2–5.5 ms across
*all* configurations; that is the GPU clock-state noise this document warns about
in the `ocl-micro` note, not a signal. Use the batch-128 figure.)

So it is a **training-only trade**, and on a machine that does both it is the
wrong trade. This box is shared by several users whose driver environment is
fixed per session, not per workload — so the default stays the default here,
and the opt-in mode is for a box dedicated to training. It needs four
environment variables:

```
ACO_DEBUG=force-waitcnt MESA_SHADER_CACHE_DISABLE=true \
DLPRIM_WINOGRAD_SPLIT_PLANES=1 DLPRIM_WINOGRAD_BWD_PLANES=1
```

and you would only want it if the box is training rather than serving.

The fast paths default to **off**, and the code refuses to enable them unless
*both* driver variables are set — it warns and falls back. `MESA_SHADER_CACHE_DISABLE`
is in that check precisely because forgetting it produces a configuration that
looks fast and is quietly wrong, which is the worst possible failure mode.

If the ACO bug is fixed upstream, `force-waitcnt` goes away and the full ~65%
becomes available. That was the case for filing it — a report
was drafted, and withdrawn (see below).

### On Mesa 26.2 it looked worse, and `force-waitcnt` looked like less of a fix

Added 2026-09-10, after building a Fedora 45 container (Mesa 26.2.2, same
host kernel, same `pt_ocl.so`; Finding 8 has the container details). The
`tools/wino-repro.py` sweeps, 6 ResNet-9 layer shapes each, in containers with
a fresh kernel cache:

| Mesa | configuration | bad gradients |
| --- | --- | ---: |
| 26.1.8 (F44) | atomics-free paths, no `force-waitcnt` | 6 / 300 sweeps |
| **26.2.2 (F45)** | atomics-free paths, no `force-waitcnt` | **67 / 300 sweeps** |
| 26.2.2 (F45) | atomics-free paths + `force-waitcnt` + cache off (the opt-in mode) | **1 / 900 sweeps** |
| 26.2.2 (F45) | shipping default (safe path) | 0 / 300 sweeps |

At the time this read as: the bug is not fixed in 26.2, it is ten times more
frequent, and `force-waitcnt` is no longer a complete workaround — so for a few
hours the interlock in the conv patch grew a `CL_DRIVER_VERSION` check
that refused the fast paths on anything but Mesa 26.1, and the report gained a
"worse on 26.2" section. Both are gone now; the next section explains why.

(All of the above was measured with the governor's undervolted idle floor in
place; read on. The "ten times worse on 26.2" is real as a measurement and
means nothing about ACO: 26.2's tighter code changes how the bursty reproducer
load interacts with the clock governor, nothing more.)

### It was the voltage

Once the container could dump ISA — `AMD_DEBUG=cs,asm`; Mesa needs *both*
the stage flag and a type flag now, which is why `cs` alone printed nothing in
the earlier session — the barrier sequences in `winconv_3x3_bwd_filter` were
there to read, and ACO's `aco_insert_waitcnt.cpp` was there to compare them
against. The first hypothesis fell within the hour: ACO deliberately does not
emit `s_waitcnt lgkmcnt(0)` before `s_barrier` in CU mode on GFX10+ (it relies
on in-order LDS issue plus `s_waitcnt_depctr vm_vsrc(0)`), and a Mesa build
that forces the full wait changed nothing — 58 vs 59 bad per 300.

Then the discriminating experiment. `ACO_DEBUG=force-waitdeps` puts a
`s_waitcnt_depctr` before **every** instruction and gave 0 bad. A patched Mesa
that lets the immediate be chosen showed that **`s_waitcnt_depctr 0xffff` —
which waits for nothing — also gives 0 bad.** Any extra issue slot between
instructions, anywhere, made the failure disappear; no single instruction
class was responsible (`s_nop` before every VALU: 3/100; before every DS or
VMEM: 11/100; before SALU/SOPP/SMEM: no effect). That is not a compiler
forgetting a specific wait. That is hardware failing when it is fed
instructions at full rate — and the failure is always the same lane.

The BC-250 here runs under a community clock governor
(`cyan-skillfish-governor-smu`) whose voltage curve was hand-tuned on
2026-09-08; the bug was first seen on 2026-09-09. Pinning the clock through
the governor's D-Bus interface, the fast paths with **no** driver workaround:

| governor state | applied | bad / 100 sweeps |
| --- | ---: | ---: |
| automatic, 1000–2000 MHz (normal) | varies | 15–36 |
| automatic, 1000–1500 MHz | mostly 1100 MHz / 787 mV | 24 |
| automatic, 1500–2000 MHz | ≥ 893 mV | 0 |
| test mode, 2000 MHz | 1062 mV | 0 |
| test mode, 2000 MHz | 987 mV (the curve's own) | 0 |
| pinned 1400 MHz | 868 mV | 0 |
| pinned 1200 MHz | 818 mV | 0 |
| **pinned 1000 MHz** | **718 mV** | **66** |

Everything that had looked like evidence for a compiler bug is explained by
one number. The reproducer alternates GPU kernels with CPU work, so the
governor keeps dropping to its floor between convolutions; a kernel that runs
there — 1000 MHz at `voltage = 720` minus the SMU's `scale = -16` offset — reads
one lane of LDS wrong. Anything that changes the load pattern changes how often
that happens: `force-waitcnt` and forced NOPs stretch every kernel; the
emulated-atomics "safe" path is three times longer per kernel and keeps the
clock up, which is why it always measured clean; Mesa 26.2's tighter code is
burstier; `nosched` is different again. The failure rate was never a property
of the code. It was a property of when the code ran.

Restoring the shipped default for the low end of the curve
(`frequency = 1000 → voltage = 800`, `1100 → 830`; the applied value is now
793 mV at the floor) and re-running everything:

| configuration, new curve | bad |
| --- | ---: |
| fast paths, pinned at 1000 MHz — the point that failed 66/100 | 0 / 300 |
| fast paths, automatic 1000–2000 MHz | 0 / 300 |
| emulated-atomics path, automatic | 0 / 300 |
| **shipped default** (fast paths, no env vars), Fedora 45 image | **0 / 300** |
| shipped default, Fedora 44 image | 0 / 100 |

And the performance, with `force-waitcnt` gone for good:

| | training | inference |
| --- | ---: | ---: |
| emulated atomics (the old shipping default) | 940 img/s | 4,543 |
| **atomics-free, no driver flags (ships now)** | **1,374 img/s** | 4,430 |

**+46% training, inference unchanged** — the number this whole finding said
was locked behind an upstream compiler fix. The interlock is gone from
`patches/dlprimitives/03-winograd-no-atomics.patch`; the atomics-free paths are
selected wherever the device has no native fp32 atomic add (not NVIDIA, no
`cl_ext_float_atomics` — so on every device this repository targets), with
`DLPRIM_WINOGRAD_BWD_PLANES=0` / `DLPRIM_WINOGRAD_SPLIT_PLANES=0` as the way
back to the atomic kernels for comparison. The upstream report in `upstream/`
is withdrawn before filing.

Two lessons worth more than the 46%. First: **a correctness failure that
responds to *any* change in timing is a hardware/voltage question before it is
a compiler question**, and the way to ask it is to pin the clock — five minutes
through the governor's D-Bus interface, no root needed for the range calls.
Every piece of evidence that pointed at ACO (`force-waitcnt` fixing it,
`nosched` worsening it, validators silent, "worse on a newer Mesa") also
pointed at load pattern, and I read it only one way. Second: **undervolt
"validation" has to include the densest LDS+ALU kernel you own, at the idle
floor, pinned.** `tools/wino-repro.py` pinned at the minimum frequency
(`busctl … SetRange uu 1000 1000`, 300 sweeps) is now that test for this box.

### What it is not

Each of these looked like the answer, and each one is eliminated by measurement.
The list is the useful part of this section, because it is what the next person
does not have to redo. `tools/wino-repro.py` is the harness.

| hypothesis | test | result |
| --- | --- | --- |
| ordering between my two kernels | `clFinish()` between them | no change (5/200) |
| ordering against the previous op | `clFinish()` before backward-filter | no change (5/200) |
| the split-K slicing arithmetic | same `k_split`, original atomic accumulate | **clean**, 0/150 |
| my index arithmetic | backward-data off, nothing else touches the workspace | **clean**, 0/200 |
| partials never written | zero-fill the region first | no change (5/200) |
| partials never written | poison-fill with 1e30 — unwritten reads would be obvious | no change; **not** poison values |
| the caching allocator recycling the buffer | `OPENCL_NO_MEM_CACHE=1` (also forces `clFinish` per op) | reduced, not fixed (10→3 /150) |
| backward-data merely running | backward-data via the GEMM path instead | **clean**, 0/200 |
| the two regions overlapping in the shared buffer | offset the partials past everything backward-data uses, so the regions are provably disjoint | no fix (9→3 /200, within noise) |
| workspace too small, or shape mismatch between construction and enqueue | runtime assertion on both, over a run containing 3 failures | never fired |
| an operation may not keep data live in the workspace across kernels | read the code: `Conv2DForwardWinograd` already writes the transformed filter in one kernel and reads it in the next | **allowed**; established upstream pattern |

So: it needs **Winograd backward-data using the same workspace**, and it needs
several *different* layer shapes in sequence — repeating one shape 400 times
never fails, because an unwritten or stale element would hold the previous
iteration's near-identical value and stay invisible.

Every workspace theory in that table is dead, including the one two earlier
versions of this document leaned on. The buffer is correctly sized, the regions
can be made disjoint, the lifetime contract permits what the code does, and none
of it matters — because the corruption happens before the workspace is ever read
back, inside the main kernel, in `__local` memory.

The one loose end the workspace story does explain is *why backward-data has to
be running*: it is not that backward-data corrupts anything, it is that
backward-data leaves large values in the shared buffer. With it routed through
GEMM the buffer holds something benign, so the same stale lane read produces a
plausible number instead of a 2e9 one and slips under the tolerance. Same bug,
invisible. That also explains why repeating one shape 400 times never fails: the
stale value is the previous iteration's near-identical one.

That was where it stood before *It was the voltage*: the paths stayed in the
tree, off, until the clock pin showed the stale LDS read was the governor's
idle-floor voltage. They ship on by default now, and *"the gap is software, not
silicon"* no longer needs an asterisk for the atomics —
neither backward kernel emulates the missing instruction any more.

## Finding 4 — rusticl misreports two device properties

| property | rusticl says | measured |
| --- | --- | --- |
| `CL_DEVICE_LOCAL_MEM_TYPE` | `CL_GLOBAL` (LDS is just global memory) | `__local` reads at **4,902 GB/s** vs 359 GB/s global — 13.7×, unmistakably real on-chip LDS |
| `CL_DEVICE_GLOBAL_MEM_CACHE_TYPE` | `CL_NONE`, size 0, cacheline 0 | a ~2 MB cache running at 2.5–2.9× DRAM is plainly visible |

The cache sweep (`ocl-micro cache`), read bandwidth against working-set size:

```
     256 KiB      767.1 GB/s
     512 KiB      947.9 GB/s
    1024 KiB      831.8 GB/s
    2048 KiB      832.7 GB/s   <- still cached
    4096 KiB      567.1 GB/s   <- falling out
    8192 KiB      339.9 GB/s
   32768 KiB      327.2 GB/s   <- DRAM
  262144 KiB      300.5 GB/s
```

Both values are constants in `rusticl/api/device.rs` (the local-memory one
behind a TODO from the 2020 initial drop; the cache ones inherited from
clover) — Gallium has no cap for either. Not pursued: nothing in this stack
reads them, and the one program known to act on `CL_DEVICE_LOCAL_MEM_TYPE`
is hashcat.

Neither one hurts this stack: `dlprimitives` reads `CL_DEVICE_LOCAL_MEM_SIZE`
(correctly reported as 64 KiB) and `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, but never the
type fields. It will mislead any library that auto-tunes off them, which is the
normal thing for an OpenCL BLAS to do — a tuner reading these would conclude
that tiling through `__local` is pointless and that there is no cache to block
for, on a GPU where `__local` is 13.7× faster than global and the cache is worth
2.5×.

## Finding 5 — what turned out *not* to be the problem

Four plausible explanations, all measured and all wrong. These took as long to
rule out as the real findings took to find.

**Not thermal or power throttling.** 90 seconds of sustained ResNet-9 training,
sampled twice a second: throughput flat at 923 img/s from the first sample to
the last, `sclk` pinned at 2000 MHz for 176 of 181 samples, 123 W median
(145 W peak), 70 °C peak. The board holds its clock indefinitely under a real
ML load. The community clock/fan daemon is doing its job.

**Not kernel launch overhead.** Round-trip latency (`enqueue` + `clFinish`) is
66 µs, which is high — but pipelined dispatch is 2.1 µs per kernel, and
`pytorch_ocl` pipelines. 301 launches per step × 2.1 µs = 0.6 ms against a
139 ms step. Confirmed independently by the profile: summed kernel time equals
wall time.

**Not memory bandwidth.** 359 GB/s read, 344 GB/s write, 356 GB/s copy — this
board out-runs a T4 (320 GB/s). PyTorch's elementwise ops are already sitting on
that roof (`a.clone()` 343 GB/s, `a+b` 310 GB/s, `relu` 350 GB/s), so there is
nothing left on the table there. Host↔device is 16.5 GB/s in, 15.7 GB/s out.
(One oddity worth knowing: `clEnqueueMapBuffer` + `memcpy` runs at **1.3 GB/s**,
13× slower than `clEnqueueWriteBuffer`. Nothing in this stack takes that path,
but don't reach for it.)

**Not the wrong convolution algorithm.** The Winograd-vs-GEMM heuristic is
crude — any 3×3/pad-1/stride-1 conv with ≥8 channels gets Winograd, with no cost
model and no per-device tuning, and the backward-filter case is chosen on a
source comment that reads *"no need to test min channels in/out since it is more
optimal in any case"*. It is nonetheless correct on this hardware. Forcing GEMM
with the new env vars, on the full training step:

| | ms/step | img/s |
| --- | ---: | ---: |
| stock auto (Winograd) | 140.8 | 909 |
| all GEMM | 167.7 | 763 |
| backward-filter GEMM only | 145.3 | 881 |
| backward-data GEMM only | 157.2 | 814 |
| forward GEMM only | 142.5 | 898 |

Winograd wins in all three directions. There is no free win in algorithm
selection.

## Hardware ceilings, for reference

Everything the BC-250's compute path can do, measured rather than quoted:

| | measured | note |
| --- | ---: | --- |
| fp32 FMA (`mad`) | 8,451 GFLOP/s | 83% of the 10.24 TFLOP/s theoretical (40 CU × 64 × 2 × 2.0 GHz) |
| DRAM read / write / copy | 359 / 344 / 356 GB/s | beats a T4's 320 GB/s |
| cache read (≤2 MB working set) | 830–948 GB/s | ~2 MB, 2.5–2.9× DRAM |
| `__local` (LDS) read | 4,902 GB/s | ~122 B/clk/WGP — at the hardware spec |
| dependent VRAM load latency | ~270 ns | ~440 ns before the GPU clocks up |
| kernel dispatch, pipelined | 2.1 µs | 66 µs if you `clFinish` each one |
| host→device / device→host | 16.5 / 15.7 GB/s | `map`+`memcpy` is 13× worse — avoid |

Against those ceilings, `dlprimitives`' convolutions land at roughly **40% of
ALU peak in the forward direction and 20–25% in the backward directions**. The
next section works out how much of that is reachable, and gets it wrong once
before getting it right.

## Finding 6 — where the rest of the gap goes

> **Correction.** An earlier version of this section concluded that the
> convolutions were "close to the practical ceiling of this device", pinned by
> LDS bandwidth, and that "there is no large, easy win hiding in the conv
> kernels". That was wrong, and the error is instructive enough to leave on the
> record. Twice I measured something with a kernel I had written quickly,
> treated the result as a property of the hardware, and then wrote a better
> kernel that beat it. The corrected version is below.

The convolutions land at roughly 40% of ALU peak. Two questions follow: what is
actually achievable on this chip, and how would we know?

### The check that settles it

The cleanest evidence is external. The same benchmark, same workload, on a T4 —
a card with **less** paper compute than this board at 2.0 GHz:

| | img/s | effective GFLOP/s | paper | % of paper |
| --- | ---: | ---: | ---: | ---: |
| BC-250, stock | 909 | 2,068 | 10,240 | 20.2% |
| BC-250, occupancy fix only (shipped until 2026-09-10) | 972 | 2,212 | 10,240 | 21.6% |
| BC-250, atomics-free + driver workaround (the former opt-in) | 1,048 | 2,385 | 10,240 | 23.3% |
| BC-250, atomics-free, no workaround (Finding 3 resolved) | 1,378 | 3,136 | 10,240 | 30.6% |
| BC-250, + Finding 10 | 1,816 | 4,133 | 10,240 | 40.4% |
| **BC-250, shipping default now** (+ Finding 12) | **2,055** | **4,677** | 10,240 | **45.7%** |
| T4 (cuDNN, fp32) | 2,294 | 5,220 | 8,100 | **64.4%** |

(ResNet-9 is 291.3 GFLOP per training step at batch 128; both stacks use
Winograd for 3×3, so counting algorithmic FLOPs is fair to both.)

The BC-250 has **1.26× more paper compute** and delivered **2.52× less
throughput** at the start of this investigation, so the efficiency gap was
**~3.2×**. A mature stack reaches 64% of peak on this workload. Whatever capped
this one at 20%, it was not physics — because a comparable chip does three times
better with the same arithmetic to do. Acting on that took the default first to
21.6% (occupancy fix), then, once the "driver bug" in Finding 3 turned out to
be a voltage setting, to 30.6%, with Finding 10's access-pattern fixes to
40.4%, and with Finding 12's prefetch to **45.7%** — the throughput gap to the
T4 is now **1.12×**, from 2.52× at the start, and inference is within 2%.

That number is the reason the "we're near the ceiling" conclusion could not have
been right, and it is worth having on hand before trusting any roofline built
from one's own microbenchmarks.

### Why my roofline was wrong

The original argument went: LDS delivers ~2.7 TB/s, the ALU wants ~34 TB/s of
operands, so each value must be reused ~12 times, an R×R register tile reuses
each value R times, and the register allocator dies at R=11 — so we are stuck
just short. Neat, and wrong at the first step.

**The LDS figure was my kernel's limit, not the device's.** Rewriting the LDS
benchmark with wide `float4` loads and several independent accumulators, instead
of a scalar load feeding one dependency chain:

| LDS read kernel | GB/s |
| --- | ---: |
| scalar, one accumulator (the original) | 4,298 |
| scalar, 8 accumulators | 4,310 |
| `float4`, 8 accumulators | 4,686 |
| `float4`, 16 accumulators | **4,902** |

4.9 TB/s, not 2.7. Across 20 WGPs at 2.0 GHz that is ~122 bytes/clock/WGP,
essentially the 128 B/clk the hardware is specified to deliver — so **LDS is at
its hardware limit and was never the problem.** With 4.9 TB/s the required reuse
is ~7, not ~12, and R=7 fits in registers with room to spare. The register wall
at R=11 is real but it is not binding.

### What does cap my hand-written kernels

Rewriting the tile benchmark's LDS path properly too (`float4`, no masking in
the inner loop) moves the numbers around but does not break 5.6 TFLOP/s:

```
   R      GFLOP/s     LDS roof  % of roof
   5         3913         6125        64%
   6         4145         7350        56%
   7         4646         8575        54%
   9         5284        11025        48%
  10         5592        12250        46%
```

At R≥7 these are nowhere near the LDS roof and nowhere near the 8,451 ALU peak.
What limits them is **latency, not bandwidth**: a big register tile means few
waves resident per SIMD, and there are not enough of them to cover LDS latency
while the accumulators are being updated. Production GEMM kernels solve exactly
this with software pipelining and double-buffered LDS tiles — prefetching the
next tile during the current one's arithmetic. Mine does none of that.

So **~5.6 TFLOP/s is a floor on the ceiling, not the ceiling.** It is what one
can get without pipelining. Given that I have already twice raised a number I
had presented as a hardware limit, the honest position is that I have not
established where the ceiling is, and the T4 result says it is meaningfully
higher than 5.6.

### Where the gap actually sits

> Written at the 972 img/s state; the split below is that state's. Findings
> 10 and 12 have since taken the step past the ~1,790 img/s projected here.

This part does not depend on knowing the ceiling, because it is a comparison
within the stack. The forward convolutions run at **4,074 GFLOP/s**. The whole
training step averages **2,544**. So everything else — the two backward passes,
which are still ~64% of the step — runs at about **60% of the forward pass's
efficiency**.

| | GFLOP/s | % of paper |
| --- | ---: | ---: |
| forward convolution kernels | 4,074 | 40% |
| whole training step (occupancy fix only, the old default) | 2,212 | 21.6% |
| whole training step (atomics-free, Finding 3 resolved) | 3,136 | 30.6% |
| whole training step (+ Finding 10) | 4,133 | 40.4% |
| whole training step (shipping default now, + Finding 12) | 4,677 | 45.7% |
| T4, whole training step | 5,220 | 64% |

If the backward passes merely matched the forward pass, the step would take
71 ms instead of 132 and the board would do **~1,790 img/s** — within **1.28×**
of the T4. The atomics-free paths, now shipping, get most of the way there.

That splits what remains in two:

- **~1.85× — the backward passes are less efficient than the forward pass.**
  The single largest identified item was the emulated atomics in both backward
  kernels, about a third of the step (Finding 3). Removing them is measured at
  1,453 img/s in an isolated training step and 1,374 in the full benchmark,
  and that is what ships now.
- **~1.28× — everything else**: `dlprimitives`' forward kernels against
  cuDNN's, plus the ~13% of the step that is not convolution.

**So the large win was in the backward passes** — located, implemented, and,
after two weeks behind a misdiagnosis, shipped: 972 → 1,374 img/s. What is
left in the backward passes is the ordinary kind of gap: kernel efficiency
against cuDNN, not a hazard.

## Finding 7 — the wave32 experiment

Before the correction above, the working theory was that wave64 pins the
register tile at R=11 and that wave32 would lift the ceiling. The register-tile
story turned out not to be the binding constraint at all — but the wave-size
experiment was run, and it is worth reporting on its own terms.

**First correction: no patched Mesa is needed.** radeonsi has had the knob all
along — it is in the second `AMD_DEBUG` table, which is easy to miss:

```
$ RUSTICL_ENABLE=radeonsi AMD_DEBUG=help clinfo 2>&1 | grep -i wave
|        w32ge [0x0000000000010000] Use Wave32 for vertex, tessellation, and geometry shaders.
|        w32ps [0x0000000000020000] Use Wave32 for pixel shaders.
|        w32cs [0x0000000000040000] Use Wave32 for computes shaders.
|        w64ge [0x0000000000080000] Use Wave64 for vertex, tessellation, and geometry shaders.
|        w64ps [0x0000000000100000] Use Wave64 for pixel shaders.
|        w64cs [0x0000000000200000] Use Wave64 for computes shaders.
```

It takes effect on rusticl kernels — querying a compiled kernel,
`CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE` goes 64 → 32 under `AMD_DEBUG=w32cs`.

**Second correction: the register wall does not move.** Running the tile sweep in
both modes, R=11 spills either way — and wave32 spills slightly *more*:

| | wave64 (`w64cs`) | wave32 (`w32cs`) |
| --- | ---: | ---: |
| R=10, no spill | 4,514 GFLOP/s | 4,528 GFLOP/s |
| R=11 scratch (LDS variant) | 80 B | 96 B |
| R=12 scratch (pure FMA) | 80 B | 96 B |
| ALU peak (`mad`) | 8,437 GFLOP/s | 8,495 GFLOP/s |
| LDS roof | ~634 G reads/s | ~610 G reads/s |

That makes sense in hindsight: on RDNA a wave can address 256 architectural VGPRs
in **both** modes. Wave64 costs twice the *physical* register file, which reduces
how many waves fit per SIMD — it does not reduce how many registers one kernel
may name. Spilling is governed by the architectural limit, so the wall sits in
the same place. **Wave size is not what pins the register tile** — though, per
the correction above, the register tile was not what pinned throughput either.

**What actually changed: it depends entirely on the kernel.**

| workload | wave64 | wave32 | |
| --- | ---: | ---: | ---: |
| `dlprimitives` SGEMM 4096³ | 4,571 GFLOP/s | **5,168** | **+13%** |
| ResNet-9 training step | **978 img/s** | 928 | **−5%** |

Both reproduce across alternating repeats. `dlprimitives` never queries the
subgroup size — it only probes for `cl_intel_subgroups`, which this device does
not have — so the kernel *source* is byte-identical in both modes. The entire
difference is backend codegen and occupancy.

Per kernel, over one training step:

| kernel | wave64 | wave32 | change |
| --- | ---: | ---: | ---: |
| `winograd_bwd_filter` | 46.05 ms | 49.45 ms | +7.4% |
| `winograd_3x3_main_bwd` | 42.36 ms | 45.23 ms | +6.8% |
| `winograd_3x3_main` | 23.65 ms | 24.34 ms | +2.9% |
| `calc_dyx_sums` | 3.69 ms | 3.30 ms | −10.6% |
| **total GPU time per step** | **128.48 ms** | **135.84 ms** | **+5.7%** |

All three Winograd kernels lose under wave32, and they are 89% of the step, so
the step loses. The GEMM path wins — but ResNet-9 barely uses it.

**So the useful finding is not "wave32 is better" or "worse", it is that the
right wave size is per-kernel and rusticl picks one globally.** For this
convolution-dominated workload wave64 is the correct default and rusticl already
picks it. But a GEMM-dominated workload — a transformer, an LLM, anything that
is mostly matmul — should get **+13% for free** by setting:

```
AMD_DEBUG=w32cs
```

Which pairs with Finding 1: the `fma()` trap and this knob both matter for
exactly the workload the ResNet-9 benchmark does not exercise. Numerics are
unaffected — the full gradient and GELU correctness suite passes identically
under `w32cs`.

OpenCL has no portable way for a kernel to request its subgroup size
(`cl_intel_required_subgroup_size` is Intel-only, and rusticl does not expose an
equivalent), so a library cannot ask for wave32 on its GEMM and wave64 on its
convolutions today. That is the thing worth fixing upstream, and it is a feature
request rather than a bug.

## Finding 8 — the libclc that ships cannot link `sin()`, `cos()` or `fma()`

Found while chasing the warning every run prints:

```
=== Rusticl warning: Patched Mesa libclc not detected. Upstream libclc may
contain known bugs or breaking changes and isn't guaranteed to work reliably.
Please visit https://gitlab.freedesktop.org/karolherbst/mesa-libclc ... ===
```

The hope (HANDOFF.md, *Done, for the record*) was that the fork would make two of our three
patches unnecessary. It makes neither unnecessary — but the reason it exists
turned out to be sitting in the stock Fedora container.

**What the warning is.** rusticl (`core/device.rs`) looks for one symbol,
`__clc_mesa_libclc_version`, in the libclc SPIR-V it loaded, and prints the
warning if it is missing. That is all it does with it — in 26.1.8 and in main
nothing is gated on the result. The fork
(`gitlab.freedesktop.org/karolherbst/mesa-libclc`, branch `llvm_22`) is
upstream libclc 22.1.8 plus nine `libclc:` commits: the version symbol, a
`tgamma` precision fix, scalarisation of a few builtins to cut compile time,
and one fix for a genuinely broken build:

> libclc: mark `__clc_flush_denormal_if_not_supported` as static inline
> The compiled SPIR-V libclc binaries ended up with an empty definition for
> `__clc_flush_denormal_if_not_supported`.

**What Fedora ships.** Fedora 44's `libclc-spirv-22.1.8-1.fc44` from
`updates` (built 2026-06-19) has exactly that empty definition:

```
$ spirv-dis spirv64-mesa3d-.spv | grep -A2 '^%_Z37__clc_flush_denormal_if_not_supportedf = OpFunction'
%_Z37__clc_flush_denormal_if_not_supportedf = OpFunction %float None %4639
       %4641 = OpFunctionParameter %float
               OpFunctionEnd
```

Every kernel whose call graph reaches it fails to build with
`Internal compilation error: nir_shader not fully linked`. The callers are
`__clc_sw_fma` (so `fma()`), `__clc_remquo`, and `__clc_argReductionLargeS` —
the large-argument reduction behind `sin`, `cos`, `tan` and `sincos`. Checked
with a one-builtin-per-kernel probe (`tools/libclc-probe.c` is the reusable
part):

| build | `fma` `sin` `cos` `tan` `sincos` `remquo` | `mad` `sinpi` `tanh` `exp` `log` `pow` `erf` `sqrt` `tgamma` `remainder` `fmod` `atan2` … |
| --- | --- | --- |
| Fedora `libclc-spirv` as shipped (2026-06-19) | **all fail to build** | build |
| Mesa fork 22.1.8.3 | build | build |

In PyTorch terms: `dlprimitives`' random kernel does Box–Muller with
`sin`/`cos`, so on a stock Fedora container

```
torch.randn(4096, device="ocl:0")            -> Failed to build program source random ...
torch.empty(4096, device="ocl:0").normal_()  -> same
torch.nn.init.normal_(t_on_device)           -> same
torch.rand(...)                              -> fine (no trig)
```

Nothing in the ResNet-9 benchmark initialises on the device, which is why
nothing here noticed. The host never showed it either: the host's
`libclc-spirv` is a *different* build of the same NVR (installed by hand from
a Koji build dated 2026-08-06, `dnf` reports it as `@commandline`), and that
one has the function body. Same package name and version, different bytes —
the container and the host were never running the same libclc.

Fedora 45's `libclc-spirv-22.1.8-2.fc45` ships the byte-identical `.spv`
(same sha256), so the swap is needed there too — on Mesa 26.2 `fma()` links
anyway because the software branch is dead code once
`__clc_runtime_has_hw_fma32` is defined true, but `sin`/`cos`/`tan`/`sincos`/
`remquo` still fail (6/25 probed builtins).

**Fix.** The fork publishes drop-in binaries per release. The container
image now `ADD`s `spirv64-mesa3d-.spv` and `spirv-mesa3d-.spv` from release
22.1.8.3 over `/usr/lib64/clc/`, pinned by URL and sha256 (`tools/mesa-dev/
Dockerfile` does the same). Same LLVM 22.1.8 the container's clang is. Verified
in a throwaway container with the files bind-mounted over the originals: the warning is
gone, 24/25 probed builtins build (the 25th is `fma(double)`, no fp64 on this
device, fails either way), `randn`/`normal_` work with mean 0.00 and std 1.00,
and a 2-epoch ResNet-9 run in the same container is within noise of the
unpatched container (905 → 908 img/s training, 4,501 → 4,472 img/s inference,
epoch-1 accuracy 0.824 vs 0.821).

**What it does not do**, measured so nobody has to hope again:

- `fma()` is still software (15–18× `mad()`) — see Finding 1; that is Mesa
  26.2's `rusticl_insert_libclc_config`, not libclc.
- `work_group_reduce_*` is still undeclared. rusticl does not advertise
  `__opencl_c_work_group_collective_functions`, and no libclc — upstream or
  fork — implements those builtins. `00-custom-reduce.patch` stays.

## Finding 10 — four access patterns, +34%, one afternoon

With the voltage question closed and `torch.ocl.profile` giving an honest
breakdown of the new 86 ms step, the largest kernels were, per step:
`winograd_bwd_filter` 27.0 ms, `winograd_3x3_main` (forward) 24.4,
`winograd_3x3_main_bwd` 16.1, BatchNorm 9.6 in total. Two things stood out
before touching anything. Forward and backward-data do the same arithmetic
with the same tile configuration, yet forward took **1.5× longer**. And one
BatchNorm launch out of eight took 1.27 ms where its neighbours took 0.1.

The method throughout was the one from Finding 3: replace one stage of a
kernel with a wrong-but-cheap version, time it, and let the difference say
what that stage costs. Correctness was then re-established with
`tools/wino-repro.py` (which now checks the forward output as well as both
gradients) and `tools/bn-check.py`.

**Forward Winograd: the transformed kernel was read at a stride.**
`winconv_calc_gkgt_3x3` stores the 4×4 transformed filters as `[N][C]`
float16s, and `winconv_3x3`'s load stage has 32 consecutive lanes read 32
consecutive output features for one input channel — a stride of `C × 64`
bytes between lanes. Cutting the *input tile* loads by 4× (the classic
Winograd redundancy) gained 5%; reading the kernel at the wrong-but-coalesced
address gained **39%**. Storing the transform as `[C][N]` and indexing
accordingly is a two-line change, and forward went from 24.4 to 15.0 ms —
faster than backward-data. This is the one that matters for inference, which
is all forward.

**Backward-filter: lanes mapped to channels, not tiles.** Its load stage had
`channel = lid % 32`, so every lane read from a different channel plane,
`H×W` floats apart; mapping `k = lid % 8` instead puts eight neighbouring lanes
on eight neighbouring tiles of the same plane. 27.0 → 24.1 ms. Then the tile
loader's fast path (`vload4`) required the whole 4-wide window to be inside
the row, so *every* tile of a 4×4 or 8×8 layer, and half of a 16×16 one, took
the scalar-with-bounds-checks path. Loading the window as one `vload4` bounded
by the buffer and masking the columns — what the forward kernel already did —
took it to 19.6 ms. Vectorising the 2×2 dY loads: 19.35.

**BatchNorm sums: each lane owned a contiguous chunk.** `bn_sums.cl` gave
every work-item `items / wg_size` consecutive elements, so at any instant the
lanes of a wave read addresses a chunk apart — and for a 512-channel 8×8
layer every second lane jumped by `channels × HW` floats, 128 KB, landing in
the same cache sets each time. That launch moved 16 MB in 1.27 ms (13 GB/s)
and the matching backward launch 3.1 ms. A grid-stride loop (`index += wg_size`)
is the textbook form; both launches are now under 0.1 ms and BatchNorm fell
from 9.6 to ~3 ms per step.

**Tried and rejected, so nobody repeats it:** more K-splitting for
backward-filter. The per-layer times looked proportional to loop iterations
per work-group, which suggested a latency-bound serial loop starving the
machine of work-groups; targets of 8, 16 and 32 work-groups per CU (cap 64)
all measured within noise of the existing 4/16. The layers that looked slow
simply had twice the FLOPs. The heuristic keeps its defaults and gained two
environment overrides for the next person.

| step, batch 128 | ms | img/s |
| --- | ---: | ---: |
| start of this finding (Finding 3 resolved, atomics-free) | 87.7 | 1,460 |
| + forward kernel layout `[C][N]` | 77.9 | 1,642 |
| + backward-filter lane mapping | 74.8 | 1,711 |
| + backward-filter vector edge loads | 70.3 | 1,821 |
| + dY loads vectorised | 69.9 | 1,832 |
| + BatchNorm grid-stride reduction | **65.2** | **1,963** |

Over the full 16-epoch benchmark: **1,378 → 1,816 img/s** training (the
benchmark's CPU-side augmentation is now ~8% of the step and shows up as the
difference to the isolated 1,963), accuracy 0.924, and inference **4,433 →
6,418 img/s** — the forward fix alone — which puts inference within **1.12×**
of the T4's 7,217. Training is within 1.26×.

A small follow-up: the backward-data planes no longer need a zero-fill. Each
parity class misses a one-pixel border strip (only tile 0 reaches row 0, only
the last tile reaches an even-sized image's last row), which is why all four
planes were memset before every launch — 1.2 ms per step. The reduce kernel now
knows the geometry and masks those elements instead; verified on odd sizes,
1×3 and 2×2 images and with accumulation (`tools/odd-shapes.py`). Step 65.2 →
64.4 ms.

Convolution is now 79% of the step (50 of 64 ms), split almost evenly between
the three Winograd kernels, and each of them runs at roughly the same rate per
FLOP. What was left in them was the pipelining work Finding 6 describes,
done in Finding 12. The
non-convolution remainder is 14 ms spread over a dozen small kernels, the
largest being 1.5 ms; the SGD optimizer alone is 110 launches of ~10 µs.

One thing that turned up in passing was **not a bug**: an early version of
`bn-check.py` reported `running_mean` / `running_var` off by ~0.9 relative
from PyTorch's CPU BatchNorm. It was the test: it copied the module state to
the GPU *after* the CPU forward had already updated the running stats, so the
GPU copy started one update ahead. With the copy first, running stats match to
1e-7 (`tools/bn-stats.py` checks the momentum arithmetic against the formula
directly). Recorded here because the wrong version of that claim spent a day
in this document.

## Finding 11 — fp16 where it counts: the inner loop

The question was whether half precision could change the ceiling rather than
chip at it. Two measurements answered it in an afternoon, and the answer is
yes, with a caveat about where the halves live.

**The hardware and the compiler deliver.** gfx1013 has packed fp16 math
(`v_pk_fma_f16`: two FMAs per lane per cycle), and ACO emits it for OpenCL
`half2`/`half4` code through rusticl. `tools/fp16-micro.c`, GPU pinned at
2.0 GHz so the governor does not confound it:

| type | GFLOP/s | vs the fp32 chain |
| --- | ---: | ---: |
| `float` (8 dependent mad chains) | 4,653 | 1.0× |
| `half4` | 18,508 | 3.98× |
| `half8` | 18,703 | 4.02× — **91% of the 20.5 TFLOP/s packed peak** |

**The stack does not — yet.** `pytorch_ocl` maps `torch.float16` to
`dlprimitives`' `half_data`, and the pointwise kernels honour it, but every
heavy kernel is fp32-only: matmul, linear, pooling, softmax, the loss and
BatchNorm refuse a half tensor; convolution *accepted* one and returned NaN
(now a `TORCH_CHECK`); `relu`/`tanh`/`sigmoid`/`relu6` on half returned wrong
values because `activation.cl` was built without its `dtype` define and read
the halves as floats (fixed in `patches/dlprimitives/01-activation-dtype.patch`;
`hardtanh`'s float-vs-half formula and a non-contiguous gradient in
`hardtanh_backward` in `patches/pytorch_dlprim/01-half-fixes.patch`; checked by
`tools/act-half.py`); bfloat16 and
`torch.autocast("ocl")` are unsupported. `tools/half-probe.py`
is the inventory. A full fp16 tensor path is a port of every one of those —
weeks.

**So the halves went where the time is.** The three Winograd kernels are 79% of
the step and, per Finding 10, not memory-bound: the cost is the inner GEMM and
the LDS traffic feeding it. Keeping the tensors in global memory fp32 and
converting on the way into LDS gives: half-size LDS tiles (one `ds_read_b128`
brings eight values, not four), a register tile of `half2` accumulators, and
`v_pk_fma_f16` in the 8×8 update — 32 packed FMAs per K step where there were
64 scalar ones. The epilogue converts back to fp32 for the inverse transform
and the stores. Each kernel's change is ~40 lines behind `#if HALF_GEMM`;
`DLPRIM_CONV_FP16=1` selects it at kernel-compile time.

| kernel, per ResNet-9 step | fp32 | fp16 inner loop | |
| --- | ---: | ---: | ---: |
| `winograd_3x3_main` (forward) | 15.0 ms | 8.9 ms | 1.68× |
| `winograd_3x3_main_bwd` | 16.0 | 10.2 | 1.57× |
| `winograd_bwd_filter` | 19.4 | 12.4 | 1.56× |
| **whole step, isolated** | 65.2 | **44.3** | **1.47×** |

**The caveat is accumulation.** There is no packed FMA with an fp32
accumulator on this chip (`v_fma_mix_f32` exists but is not packed, so it
gives up the 2×), so the products are summed in fp16. Against CPU fp32 on the
ResNet-9 shapes, unit-scale inputs (`tools/grad-err.py`):

| | Y | dX | dW |
| --- | ---: | ---: | ---: |
| fp32 kernels | 1e-6 | 6e-7 | 7e-6 |
| fp16 inner loop, default K split | 0.3–0.9% | 0.4–0.8% | 0.6–1.3% |
| fp16 inner loop, 4× more K splits | same | same | **0.3–0.7%** |

`dW` accumulates over batch × tiles — up to 2,048 terms per work-group — so it
was the worst. The split-K planes from Finding 3 bound that: each work-group
sums its slice in fp16 and the planes are summed in fp32 by the reduce kernel,
so more splits mean shorter fp16 sums. Splitting four times more costs nothing
measurable (Finding 10 tried it for speed and found none) and halves the dW
error; the fp16 mode raises the split target accordingly. For comparison,
NVIDIA's TF32 — PyTorch's *default* for convolutions on Ampere — is ~5e-4
relative; this is roughly ten times looser, which is why it is **opt-in** and
fp32 callers keep fp32.

**Does it train?** The 16-epoch benchmark with `DLPRIM_CONV_FP16=1`:

| | training | test acc | inference |
| --- | ---: | ---: | ---: |
| fp32 (shipping default) | 1,816 img/s | 0.924 | 6,418 |
| **fp16 inner loop (opt-in)** | **2,714** | **0.928** | **9,584** |
| T4, fp32 | 2,294 | 0.926 | 7,217 |
| T4, AMP | 4,441 | 0.923 | 7,355 |

Accuracy is unchanged — 0.928 is the highest of any configuration measured,
which is noise, but it is certainly not degradation. Training
is **18% past the T4's fp32 number** and inference **33% past it**; against the
T4 running its own mixed precision the training gap is 1.64×. The halves also
made the CPU-side data pipeline visible: the isolated step is 44 ms (2,844
img/s) and the benchmark 2,714, so ~5% is now augmentation on the host.

**How to use it:** the switch is read when the kernels compile, so it must be in
the environment before the first convolution — `os.environ["DLPRIM_CONV_FP16"]
= "1"` at the top of the script or notebook, before `import torch`, or in
the process environment. `DLPRIM_WINOGRAD_KSPLIT_TARGET` /
`_MAX` (defaults 16 / 64 in this mode) trade dW accuracy against nothing
measurable. It applies to 3×3 convolutions only; everything else in the
network stays fp32. GELU, softmax, attention and the rest of a transformer's
diet gain nothing from it — that is the full fp16 port above.

## Finding 12 — pipelining: what worked, what did not, and why

The remaining item from Finding 10 was overlap: every K step of the three
Winograd kernels was `global load → transform → LDS store → barrier → GEMM →
barrier`, nothing in flight while anything else ran. Three variants, each
measured with the clock pinned at 2.0 GHz so the governor could not colour
the comparison.

**Register prefetch — kept.** The global loads for step k+1 are issued right
after step k's barrier, before the GEMM, into registers; after the GEMM they
are transformed and stored to LDS. The loaders were split into a raw load and
the transform so the latency of the load, not the arithmetic, is what moves.
Costs 20 VGPRs; the kernels were LDS-limited anyway, so occupancy is
unchanged. (The "3 work-groups per CU" this section originally claimed was an
assumption, not a measurement, and it was wrong — at 40 KiB each they were
resident one to a CU. Finding 13 measures it and turns the 40 KiB into 32.)

| kernel, fp32 | before | after | | fp16 mode | before | after | |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| forward | 15.0 | 13.3 | −12% | | 8.5 | 7.6 | −11% |
| backward-data | 16.0 | 13.3 | −16% | | 9.8 | 8.2 | −16% |
| backward-filter | 19.2 | 19.3 | 0 | | 11.6 | 10.8 | −8% |
| **step** | 65.0 | **59.8** | **−8%** | | 45.1 | **40.8** | **−10%** |

Backward-filter in fp32 did not move: its loop is bound by LDS and ALU, not
by load latency, which the fp16 result (where LDS traffic halves) is
consistent with.

**LDS double-buffering — rejected.** Writing step k+1's tiles into a second
buffer while step k's are read removes one of the two barriers per iteration.
In fp32 it needs 80 KB of LDS, which the device refuses (64 KB per work-group);
in fp16 mode it fits but drops occupancy from 6 work-groups per CU to 3 and the
forward kernel goes from 7.5 to 11.5 ms. The barrier was cheaper than the
occupancy. The double-buffer code is not in the tree.

**Mixed precision (fp16 tiles, fp32 accumulate) — rejected, with regret.**
Reading `half` from LDS and accumulating in `float` compiles to
`v_fma_mix_f32`, the accuracy is 6e-4 across Y, dX and dW — TF32-class, ten
times better than the fp16 mode — and LDS traffic halves. But `v_fma_mix_f32`
issues slower than `v_mac_f32` on this chip: forward 13.3 → 17.6 ms,
backward-filter 19.3 → 16.9 (the LDS-bound one liked it), step 59.8 → 62.6.
Not a default candidate. The result is recorded here so the next person does
not have to try it; the code is not.

**A curiosity, not a bug:** every Winograd kernel allocates 16–25 KB of scratch
per wave. It is the `p_C[8][8] = {{0}}` accumulator initialisation going
through private memory — 20 stores and 20 loads at kernel start, once, never
in the loop. Not worth a fix; worth knowing if "Scratch:" in a shader dump
looks alarming.

**Where that leaves the step (isolated, clock pinned):**

| | fp32 | fp16 inner loop |
| --- | ---: | ---: |
| convolution kernels | 45.9 ms | 26.6 ms |
| everything else | 13.9 ms | 14.2 ms |
| **step** | **59.8 ms — 2,139 img/s** | **40.8 ms — 3,134 img/s** |

Over 16 epochs: fp32 **1,816 → 2,055 img/s**, inference **6,418 → 7,064**
(the T4: 2,294 and 7,217 — training is within 1.12×, inference within 2%);
fp16 mode **2,714 → 2,987** and **9,584 → 10,279**. Accuracy 0.924 / 0.926.

The three kernels are now at roughly 45% of fp32 ALU peak. The remaining
structural cost is the two barriers per 8-channel K step and the
transform-and-store phase between them, which no wave overlaps with anything.
Beyond this the standard answers are larger K steps (fewer barriers, more LDS,
lower occupancy — the trade this section kept losing), or a different kernel
structure altogether — which is a rewrite, not tuning. And with convolution at
27 ms in fp16 mode, the 14 ms of "everything else" — BatchNorm, activation,
pooling, the 110-launch optimizer — is now a third of the step and the better
target.

## Finding 13 — the padding that cost a work-group

Everything above changed a kernel or a launch. This one changes a `#define` —
the only fix here that is purely a constant, and the only one `dlprimitives`'
existing tuning vocabulary could state as it stands. It was found by asking
whether that vocabulary could have produced any of the rest.

The three Winograd kernels pad their `__local` tiles against bank conflicts
with `STRIDE_OFFSET` and `TR_STRIDE_OFFSET`, picked in `conv.cpp`:

```cpp
int off = ctx.is_amd() ? 0 : 1;
int toff = 1;
```

Two constants, per vendor, exactly the shape of
[PR #30](https://github.com/artyom-beilis/dlprimitives/pull/30) (which adds
Imagination GEMM tile sizes). Nobody had measured them here, so:

| `TR_STRIDE_OFFSET` | 0 | 1 (default) | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| img/s, isolated step | 2,242 | 2,052 | 2,054 | 2,064 | 2,018 |

**+9% from one constant.** And the profile says it is not uniform: the two
backward kernels get faster and the forward kernel gets slower, so the right
value is per direction. Medians of three profiled runs:

| kernel | padded | unpadded |
| --- | ---: | ---: |
| `winograd_bwd_filter` | 19.08 ms | **14.93** |
| `winograd_3x3_main_bwd` | 14.75 | **12.08** |
| `winograd_3x3_main` (forward) | **14.08** | 16.84 |
| step (GPU time) | 60.57 | **53.83** (per direction) |

### Why a stride offset costs 8 KiB

Because it is not only a stride. Any non-zero offset trips `PADDING_FACTOR`,
which allocates 16 more tile rows:

```c
#if STRIDE_OFFSET > 0 || TR_STRIDE_OFFSET > 0
#define PADDING_FACTOR 1
#endif
__local LTYPE wg_local_memory[(XTILES_IN_WG + YTILES_IN_WG + 16 * PADDING_FACTOR) * WG_K * 16];
```

(32 + 32 + 16) × 8 × 16 floats = **40 KiB per work-group**, against 32 KiB
with both offsets at zero. On AMD `off` is already 0, so `toff = 1` was buying
one padded transpose stride and paying the whole 8 KiB for it.

That is a residency boundary, and it is measurable —
`tools/ocl-micro.c occ` launches eight 256-item work-groups per CU of a
latency-bound kernel and divides the time by the one-per-CU time:

| `__local` per work-group | 8 K | 16 K | 20 K | 24 K | 32 K | 36 K | 40 K | 48 K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| work-groups resident per CU | 6.9 | 3.9 | 2.7 | 2.0 | 2.0 | 1.1 | 1.2 | 1.3 |

A 32 KiB work-group shares its CU with a second; a 40 KiB one has the CU to
itself. (This also corrects Finding 12, which assumed three resident
work-groups per CU in fp32. It was one. The conclusion there — that
prefetch's 20 extra VGPRs cost no occupancy — still holds, because LDS is the
binding limit either way.)

Why the forward kernel goes the other way is not established. It has the same
LDS footprint and the same residency step; it simply pays more for the bank
conflicts than it gains from the second work-group. Left measured, not
explained.

`09-winograd-tr-offset.patch` drops the padding in the two backward
constructors when the device's `__local` is large enough for the unpadded
work-group to double up but not the padded one, which is true on this chip and
false wherever `STRIDE_OFFSET` is already 1 (every non-AMD device pays for the
padding regardless). In fp16 mode the tiles halve, the rule does not
fire, and the measurement agrees that it should not: 2,889–2,901 img/s with
the rule against 2,886–2,903 with the padding forced back on, i.e. the same
number. Forcing the padding *off* in fp16 costs 5%: 2,738.

### It is not a knob you could have set from outside

The tempting summary — "so the win was available by tuning all along" — is
wrong, and the same binary says so. With the backward kernels as they ship
upstream (emulated `atomic_addf`, stock split-K) the *same* define goes the
other way:

| | all padded | backward unpadded |
| --- | ---: | ---: |
| this series | 2,069 img/s | **2,330** |
| stock backward kernels (`*_PLANES=0`, `KSPLIT=stock`) | **1,171** | 849 |

Those kernels are bound by the compare-and-swap loop, not by occupancy, so
they collect the bank conflicts and none of the benefit. The right value of a
tuning constant is a property of the kernel it tunes, not of the device alone
— which is the answer to "could per-device tuning have found this instead of
patching the kernels?": no, because before the kernels changed, the tuned
value was the one already there.

### End to end

16 epochs, same binary, one define apart (2026-09-18, `scratch/venv`):

| | train img/s | s/epoch | test acc | epochs to 90% | infer bs128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| padded (previous default) | 1,993 | 25.1 | 0.924 | 13 | 6,732 |
| per direction (ships) | **2,244** | **22.3** | 0.924 | 13 | 6,774 |

Inference does not move, as expected — an inference pass has no backward
kernel. Gradients: 0 bad over 300 `tools/wino-repro.py` sweeps.

## What is left on the table

[HANDOFF.md](HANDOFF.md) has the same list in priority order, with the
constraints and dead ends a fresh start would otherwise rediscover.

- **Backward-data's atomics.** Done — the parity-plane kernel ships by
  default: Winograd's 4×4 input-gradient tiles overlap, but tiles of the same
  row- and column-parity are 4 apart in each direction, so within one of the
  four parity classes the patches are exactly disjoint (`USE_PLANES` in
  `winograd_bwd_data.cl`). With backward-filter's split-K planes too, 972 →
  1,374 img/s (Finding 3). Worth sending upstream to `dlprimitives`, together
  with the split-K planes, the occupancy fix, and Finding 10's four
  access-pattern fixes — all of which are generic, not BC-250-specific.
- **Per-kernel wave size.** Tested and reported above: the right choice differs
  per kernel and rusticl picks one globally. Nothing more to measure locally;
  the fix is an upstream feature request for a way to pin a kernel's subgroup
  size.
- **The "ACO `s_waitcnt` bug".** There was none — Finding 3, *It was the
  voltage*. The draft report was withdrawn before filing and has since been
  removed from the repository; this Finding is the record.
- **`ACO_DEBUG` not being in the shader cache key** is still true and still a
  small Mesa footgun, but with no `ACO_DEBUG` in use here it no longer matters
  to this box.
- **A real fp16 tensor path.** Finding 11 put fp16 where the time was; the
  rest of the stack (matmul, linear, pooling, softmax, loss, BatchNorm, autocast)
  is fp32-only or broken for half tensors — `tools/half-probe.py` lists it.
  Worth it for transformer workloads, which Finding 11 does not touch.
- **Software pipelining.** Register prefetch is done (Finding 12). Full LDS
  double-buffering loses to occupancy on this chip. What is left in the
  Winograd kernels is structural — barriers per K step — and is a rewrite.
- **The other tuning constants.** Finding 13 found +12% in one of them and a
  probe (`tools/ocl-micro.c occ`) for the resource it trades. The rest of the
  Winograd geometry — `WG_K = 8`, 32×32 tiles, the 8×8 register patch — has
  never been swept on this device, and unlike `TR_STRIDE_OFFSET` those are
  kernel-source constants rather than a host-side define. The lesson from
  Finding 13 is that their right values moved when the kernels changed, so a
  sweep is worth more now than it would have been before the series.
- **The non-convolution 14 ms.** A third of the fp16-mode step now: BatchNorm
  (~3 ms), activation forward/backward (~2.5), pooling (~1.6), the SGD
  optimizer's 110 launches (~1.3), the planes reduce (~1.4). BN+ReLU fusion and
  a fused optimizer are the two obvious moves.
- **Where the real ceiling is.** Not established. Every attempt to measure it
  here produced a number that a better kernel then beat.
- **libclc.** Done — Mesa's patched libclc is the drop-in for any container
  running this (Finding 8). It did *not* do what the previous version of this bullet hoped:
  `fma()` is a rusticl 26.2 fix, and `work_group_reduce_*` is a rusticl
  feature gap that no libclc fills. What it does fix is worse than either:
  `torch.randn` on the device.
- **Mesa 26.2.** Done: everything is verified on Fedora 45 / Mesa 26.2.2. `fma()` is native
  (`02-gelu-mad.patch` redundant but harmless).
- **The governor's voltage curve.** The one thing on this box that can make
  correct code produce wrong numbers, and it lives outside this repository
  (`/etc/cyan-skillfish-governor-smu/config.toml`). Any future retune must be
  validated with `tools/wino-repro.py` pinned at the floor — HANDOFF.md has
  the exact commands.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*

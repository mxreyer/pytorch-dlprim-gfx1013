# Handoff: what is open, and where to start

State as of 2026-09-14, at the end of the performance investigation written up in
[OPENCL-PERF.md](OPENCL-PERF.md).

Nothing here is required to *use* the box. The shipped configuration is correct
and verified. Everything below is either done-and-recorded or a lead for later.

**2026-09-10 changed this document more than any other day.** The "ACO
`s_waitcnt` bug" that the previous version of this file put at the top as
*"worth more than everything else in this repository combined"* was not a
compiler bug. It was the GPU clock governor's idle-floor voltage. Details in
OPENCL-PERF.md, Finding 3, *It was the voltage*; consequences throughout below.

## Where things stand

| | training | inference | correctness |
| --- | ---: | ---: | --- |
| stock `pytorch_ocl`, Fedora 44 image | 909 img/s | 4,511 | clean |
| occupancy fix only (shipped until 2026-09-10) | 972 | 4,496 | 0 bad / 400 sweeps |
| Fedora 45, atomics-free paths, no driver flags (stage 5) | 1,378 | 4,433 | 0 bad / 300 sweeps |
| + four access-pattern fixes (stage 6) | 1,816 | 6,418 | 0 bad / 300 |
| **what ships now: + register prefetch (stage 10)** | **2,055** | **7,064** | **0 bad / 300 sweeps (Y, dW, dX); BN 7e-7; odd shapes OK** |
| opt-in `DLPRIM_CONV_FP16=1` (stage 9) | 2,987 | 10,279 | conv Y/dX/dW within ~0.5% of fp32; 16-epoch acc 0.926 |
| former "opt-in mode" (atomics-free + `force-waitcnt`) | 1,048 | 3,144 | obsolete — the flag was never needed |
| T4 (Colab), fp32, for scale | 2,294 | 7,217 | |

The atomics-free Winograd backward paths are on by default. There is no
interlock any more; `DLPRIM_WINOGRAD_BWD_PLANES=0` / `_SPLIT_PLANES=0` bring the
emulated-atomic kernels back for comparison (888 img/s).

## 1. ~~File the ACO bug~~ — withdrawn, do not file

`upstream/aco-lds-waitcnt-gfx1013.md` carries a withdrawal banner and is kept as
a record. What every one of its observations actually measured was how often a
bursty workload let the clock governor drop the GPU to 1000 MHz at 718 mV.
Pinned at that point, 66 / 100 sweeps fail with any code; pinned anywhere from
1200 MHz up, or at 1000 MHz with the governor's default 800 mV, none do.

The second item in that report — `ACO_DEBUG` not being in Mesa's shader cache
key — is still true, and no longer matters here.

## 2. ~~A standalone reproducer~~ — not needed

There is nothing to reproduce for Mesa. `tools/wino-repro.py` keeps a job,
though: it is now the **voltage-curve health check** for this box (stage 5).

## 3. The patched libclc — done; it fixed something else than expected

Mesa's libclc fork (`gitlab.freedesktop.org/karolherbst/mesa-libclc`, release
22.1.8.3) now ships in the notebook image, `ADD`ed over `/usr/lib64/clc/` with a
pinned URL and sha256. The *"Patched Mesa libclc not detected"* warning is gone.
Full account in OPENCL-PERF.md, Finding 8. Short version:

- **Neither patch became unnecessary.** `work_group_reduce_*` is a rusticl
  feature gap (`__opencl_c_work_group_collective_functions` is not advertised;
  no libclc implements the builtins). `fma()` stays software because Mesa
  26.1.8 never defines libclc's `__clc_runtime_has_hw_fma32()` stub — Mesa 26.2
  does (`rusticl_insert_libclc_config`), and Fedora 45 ships 26.2.0.
- **What it did fix:** the `libclc-spirv` build in the shipped image had an
  empty `__clc_flush_denormal_if_not_supported`, so `sin`/`cos`/`tan`/`sincos`/
  `fma`/`remquo` all failed to link, and with them `torch.randn(...,
  device="ocl:0")`, `.normal_()` and `nn.init.normal_()` on the device. The host
  never showed it because its `libclc-spirv` is a different (hand-installed
  Koji) build of the same version.
- `tools/libclc-probe.c` answers "warning? work-group collectives? hardware
  fma?" in one run; use it against any future Mesa/libclc combination before
  believing a hypothesis about it.

## 4. Fedora 45 image — done, now the default

The notebook image's `Dockerfile` (in
[bc250-jupyterhub-opencl-k3s](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s/tree/main/k8s/jupyterhub/notebook-image))
takes `ARG FEDORA_VERSION`, default **45** (Mesa 26.2.2, LLVM 22.1.8,
python3.12 available; `FEDORA_VERSION=44 ./build.sh` gets the old one back).
Tested in a bare container:

- `fma()` is the hardware instruction (1.0× `mad()`); `gelu_mad.patch` is now
  redundant but harmless. The libclc drop-in is still needed — F45's
  `libclc-spirv` is byte-identical to F44's and `sin`/`cos` still fail without
  it.
- Emulated-atomics path: 0 bad / 300; 944 img/s over 3 epochs vs 966 for the
  F44 image measured the same way — inside the run-to-run band.
- What looked like "the ACO bug is 10× worse on 26.2" was the governor again
  (stage 5): 26.2's tighter kernels make the reproducer burstier.

Imported into k3s on 2026-09-11; new user servers run it. Since 2026-09-14 the
image fetches `pt_ocl.so` from this repository's GitHub releases (URL +
sha256) instead of a checked-in copy.

## 5. The atomics-free paths ship; the governor's voltage curve is the constraint

**What was found.** With ISA dumps finally working (`AMD_DEBUG=cs,asm` — Mesa
needs the stage flag *and* a type flag; `cs` alone prints nothing, which is why
the earlier attempt failed; `tools/one-conv.py` compiles exactly one layer's
kernels for this) and a Mesa built from source inside a container to try
compiler-side fixes (`tools/mesa-dev/`, ~3 minutes per build), every "fix" that worked turned out to work by slowing
the code down: `force-waitcnt`, a `s_waitcnt_depctr` that waits for nothing,
`s_nop` before every VALU. Pinning the GPU clock through the governor's D-Bus
interface then isolated it to one point of the voltage curve:

| governor pinned at | applied | fast paths, bad / 100 |
| --- | ---: | ---: |
| 1000 MHz | 718 mV (config 720, SMU offset −16) | **66** |
| 1200 MHz | 818 mV | 0 |
| 1400 / 1500 / 1800 / 2000 MHz | 868–987 mV | 0 |

The 720 mV floor was set on 2026-09-08 ("validated at 700 with one step of
margin") — validated against something lighter than a dense LDS+ALU kernel. The
bug appeared on 2026-09-09.

**What was done.** `/etc/cyan-skillfish-governor-smu/config.toml` low-end
safe-points back to the shipped defaults, `1000 → 800 mV`, `1100 → 830 mV`
(applied floor now 793 mV). Then: fast paths pinned at 1000 MHz **0 / 300**;
fast paths on the normal 1000–2000 range **0 / 300**; atomic paths **0 / 300**.
`winograd_ksplit.patch` lost its interlock entirely — `fast_path_allowed()`,
the `ACO_DEBUG` token check, `DLPRIM_UNSAFE_FAST_PATHS`, all gone — and the
planes paths are on by default with `=0` as the off switch. `pt_ocl.so` and
both images rebuilt; shipped default verified with no environment at all:
0 / 300 and 1,374 img/s (3-epoch run). Patch still applies to upstream
`dlprimitives` HEAD.

**What to carry forward.** The governor config is the one file outside this
repository that can make correct code produce wrong numbers. Any retune of the
curve must be validated at the floor with the densest kernel available:

```
# unprivileged; pins the governor at its minimum for the duration
busctl --system call com.cyanskillfish.Governor /com/cyanskillfish/Governor \
    com.cyanskillfish.Governor.PerformanceMode SetRange uu 1000 1000
docker run --rm --device /dev/dri/renderD128 --group-add 105 \
    -v $PWD/tools:/t -w /t bc250-notebook:latest python wino-repro.py 300
busctl --system call com.cyanskillfish.Governor /com/cyanskillfish/Governor \
    com.cyanskillfish.Governor.PerformanceMode SetRange uu 1000 2000
```

Expect `0 bad gradients over 300 sweeps`. Anything else is the voltage.

Also worth doing now that they are correct and default: send the three
`dlprimitives` changes (occupancy fix, split-K planes, parity planes) upstream.

## 6. Four access-pattern fixes — done (OPENCL-PERF.md, Finding 10)

Profiling the step after stage 5 (`torch.ocl.profile`, harness in
`tools/profile-step.py`) and testing wrong-but-cheap variants of each kernel
stage found: forward Winograd reading its transformed filters at a `C×64`-byte
lane stride (39% of the kernel — now stored `[C][N]`), backward-filter mapping
lanes to channel planes and taking a scalar path for every edge tile, and the
BatchNorm reduction hitting the same cache sets every step (13 GB/s). All in
`winograd_ksplit.patch`; isolated step 87.7 → 65.2 ms. Tried and rejected:
more K-splitting (no effect; env knobs `DLPRIM_WINOGRAD_KSPLIT_TARGET/_MAX`
left in for measurement). Correctness: `wino-repro.py` now also checks the
forward output; `tools/bn-check.py` checks BatchNorm forward/backward against
CPU.

(An earlier version of this stage flagged a BatchNorm running-stats
discrepancy. It was the test copying module state after the CPU forward; the
stats match to 1e-7. `tools/bn-stats.py` checks the momentum arithmetic.)

## 9. fp16 — the inner loop is done, the tensor path is not

Measured first (`tools/fp16-micro.c`, GPU pinned): packed `v_pk_fma_f16`
reaches 18.7 TFLOP/s through rusticl, 91% of the 2× peak, so the hardware and
ACO are not the obstacle. Inventory of `pytorch_ocl` with half tensors
(`tools/half-probe.py`): pointwise ops work — the activation kernels
(relu/tanh/sigmoid/relu6) returned wrong values for half because they were
built without a `dtype` define, fixed in `pytorch_ocl_half_fixes.patch` along
with hardtanh's formula and a non-contiguous grad in `hardtanh_backward`
(`tools/act-half.py`); matmul/linear/pool/softmax/loss/BatchNorm refuse,
convolution returned NaN silently (now a `TORCH_CHECK`), bf16 and
`torch.autocast("ocl")` unsupported.

Shipped instead (OPENCL-PERF.md, Finding 11): `DLPRIM_CONV_FP16=1` runs the
three Winograd kernels' LDS tiles and register-tile GEMM in fp16 with fp32
tensors in memory. 1.47× on the step, 2,714 img/s / 9,584 inference, accuracy
unchanged; errors ~0.5% vs fp32 (fp16 accumulation, bounded by the split-K
planes; the mode raises the split target for that). Opt-in because it is ~10×
looser than TF32. Must be in the environment before the first convolution.

Open, in order of value:
- The **full fp16 tensor path** (sgemm, pooling, softmax, loss, BN, activation
  in half; autocast registration) — the thing transformer users need. Weeks.
  `sgemm.cl` is the natural first kernel, same recipe as Finding 11.
- Other pointwise formulas in `pytorch_ocl` that mix float scalar params with
  `dtype` operands (the hardtanh pattern) will fail to compile for half; found
  three, there are likely more. `tools/half-probe.py` is the place to add
  cases.
- Whether the fp16 inner loop should be the *default* is a judgement about
  error tolerance, not speed; TF32 precedent says defaults can be lossy, ours
  is 10× lossier. Left opt-in.

## 10. Pipelining — register prefetch shipped; the rest measured and rejected

OPENCL-PERF.md, Finding 12. Prefetching the next K step's global loads before
the GEMM: fp32 step 65.0 → 59.8 ms, fp16 45.1 → 40.8 (clock pinned). LDS
double-buffering: exceeds the 64 KB work-group limit in fp32, and in fp16 loses
more to occupancy than it gains from the saved barrier. Mixed precision (fp16
LDS, fp32 accumulate via `v_fma_mix_f32`): TF32-class accuracy but slower than
fp32 on this chip. Both are documented, neither is in the tree. The planes
zero-fill is gone too (masked in the reduce). What is left in the Winograd
kernels is structural; the non-convolution 14 ms is now the better target
(BN+ReLU fusion, a fused optimizer).

## 7. Where the ceiling really is — unresolved

Every attempt to measure the practical ceiling here produced a number a
better-written program later beat (2.7 → 4.9 TB/s LDS, twice over). The forward
convolutions run at ~40% of ALU peak and a T4 converts 64% of its paper number
against this board's 45.7% (22% before stage 5, 30.6% after, 40.4% after
stage 6), so there is room, but the cheap wins are taken: the three Winograd
kernels are 77% of the fp32 step, run at the same rate per FLOP, and what
remains in them is structural. Software pipelining — prefetching the next `__local` tile
during the current tile's arithmetic — is the standard technique none of the
kernels here use, and the most likely next gain.

## 8. A second GPU

Less urgent than it was — there is no Mesa bug to triangulate. Still
interesting for performance: RDNA3 has the hardware fp32 atomic add this whole
investigation works around, so the planes-vs-atomics trade would look
different there.

## Constraints to carry forward

- **Never replace system Mesa or libclc on this host.** Live JupyterHub on k3s
  plus the display. Build to a local prefix, or test in a container.
- **Do not overwrite `pt_ocl.so` while a process has it mapped** — `cp` rewrites
  in place and segfaults the running process. Cost us a benchmark run.
- **`~/.dlprimitives/cache.db` outlives every experiment.** `dlprimitives`
  caches compiled kernel *binaries*, keyed on kernel source and build options —
  not on `ACO_DEBUG`, `AMD_DEBUG` or anything else in the driver's environment.
  Binaries compiled during a `force-waitcnt` / `w32cs` experiment are served
  back afterwards as if nothing had changed. Found 2026-09-10: the host venv
  measured 478 img/s with the existing cache and 968 with
  `DLPRIM_CACHE_DIR=<empty dir>`, same binary, same GPU. Delete `cache.db` (or
  point `DLPRIM_CACHE_DIR` at a fresh directory) before believing any number,
  and after any driver-flag experiment. In the Hub the cache lives in the
  user's persistent volume, so it survives image updates too.
- **Warm the GPU before measuring.** It idles at 1000 MHz; a cold single test
  measures the wrong machine. Two false alarms in this session came from this.
- **A correctness failure that responds to *any* change in timing is a
  hardware/voltage question first.** Pin the clock before touching the
  compiler. Two weeks went the other way here.
- **Rare failures need 150–400 sweeps** to say anything. Single runs prove
  nothing, and repeating one shape hides it entirely.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*

# Handoff: where things stand, what is open

As of 2026-09-18. Nothing here is needed to *use* the patches; the shipped
build is correct and verified. The details behind every line are in
[OPENCL-PERF.md](OPENCL-PERF.md).

## Where things stand

img/s in the
[bc250-jupyterhub-opencl-k3s benchmark](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s/blob/main/benchmark.py)
(ResNet-9 on CIFAR-10, batch 128, 16 epochs):

| | training | inference | correctness |
| --- | ---: | ---: | --- |
| stock `pytorch_ocl` | 909 | 4,511 | clean |
| occupancy fix only | 972 | 4,496 | 0 bad / 400 sweeps |
| + atomics-free backward paths | 1,378 | 4,433 | 0 bad / 300 |
| + four access-pattern fixes | 1,816 | 6,418 | 0 bad / 300 |
| + register prefetch | 2,055 | 7,064 | 0 bad / 300 (Y, dW, dX); BatchNorm within 7e-7 of CPU |
| **+ backward scratch padding dropped (what ships)** | **2,212** | **6,802** | **0 bad / 300** |
| opt-in `DLPRIM_CONV_FP16=1` | 2,561 | 8,545 | conv within ~0.5% of fp32; 16-epoch accuracy unchanged |
| T4 (Colab), fp32, for scale | 2,294 | 7,217 | |

The bottom two rows are medians of three runs of the shipped build on
2026-09-18; everything above them is a single run from 2026-09-15 in the
notebook image. So the rows are not one clean ladder, and this benchmark
swings several percent between identical runs on this box — it prepares its
images on the same four CPU cores the training loop needs (README has the
numbers). The padding patch's own A/B, same binary and back to back, was
1,993 → 2,244 training with inference unmoved; the GPU-only step for the
same change, 60.6 → 53.8 ms, is the tighter number.

Everything is on by default on this device. The plane-based backward paths
switch themselves on wherever the GPU has no hardware float atomic add;
NVIDIA and anything advertising `cl_ext_float_atomics` keep the atomics.
With the optional knobs patch applied, `DLPRIM_WINOGRAD_BWD_PLANES=0` /
`_SPLIT_PLANES=0` bring the old emulated-atomic kernels back for comparison.

## The one thing to know before debugging a wrong gradient

For two weeks the atomics-free kernels sat behind a safety interlock because
they produced a wrong lane in ~2% of runs, and `ACO_DEBUG=force-waitcnt` made
it go away. It looked exactly like a compiler bug in how the GPU waits for its
own memory operations. It was not: the GPU clock governor's idle floor had
been hand-set to 1000 MHz at 718 mV, and these dense kernels were the first
code on the box to notice the chip was under-volted there. Pinned at that
floor, 66 of 100 sweeps fail with *any* code; at the governor's default
800 mV, 0 of 900. Every "fix" (`force-waitcnt`, extra `s_nop`s, a wait
instruction that waits for nothing) worked only by slowing the code down
enough that the clock did not drop to the floor as often.

So: **a correctness failure that responds to any change in timing is a
voltage question first.** Pin the clock, then run the sweep:

```
# unprivileged; pins the governor at its minimum for the duration
busctl --system call com.cyanskillfish.Governor /com/cyanskillfish/Governor \
    com.cyanskillfish.Governor.PerformanceMode SetRange uu 1000 1000
RUSTICL_ENABLE=radeonsi scratch/venv/bin/python tools/wino-repro.py 300
busctl --system call com.cyanskillfish.Governor /com/cyanskillfish/Governor \
    com.cyanskillfish.Governor.PerformanceMode SetRange uu 1000 2000
```

Expect `0 bad gradients over 300 sweeps`. Anything else is the voltage curve
(`/etc/cyan-skillfish-governor-smu/config.toml`), and any retune of it should
be validated the same way. Full account: OPENCL-PERF.md, Finding 3.

## Done, for the record

- **The "ACO bug" report** was withdrawn before filing and has been
  removed. Its one still-valid observation — `ACO_DEBUG` is not part of
  Mesa's shader cache key, so ACO experiments need
  `MESA_SHADER_CACHE_DISABLE=true` — no longer matters here.
- **libclc.** Fedora's `libclc-spirv` 22.1.8 cannot link `sin()`, `cos()` or
  `fma()`, which takes `torch.randn` on the device down with it. Mesa's libclc
  fork fixes it as a drop-in. It did *not* fix `work_group_reduce_*` (rusticl
  simply does not implement that feature) or the software `fma()` (a rusticl
  issue before 26.2) — both patches stay. `tools/libclc-probe.c` checks all
  three in one run.
- **Mesa 26.2** (Fedora 45): `fma()` is now the hardware instruction, so
  `02-gelu-mad.patch` is redundant; the patches and numbers hold on both 26.1
  and 26.2.
- **ISA dumps** need `AMD_DEBUG=cs,asm` — both the stage flag and a type flag;
  `cs` alone prints nothing. `tools/one-conv.py` compiles one layer's kernels
  for this. `tools/mesa-dev/` builds a patched Mesa in a container (~3 min)
  for compiler-side experiments.
- **Profiling** with `torch.ocl.profile` (`tools/profile-step.py`) found the
  four access-pattern fixes; timing a wrong-but-cheap variant of a stage
  before fixing it turned out to be the fastest way to know what a fix is
  worth.
- **How many work-groups fit on a compute unit is measurable** (2026-09-18).
  Each CU has a fixed budget of fast on-chip scratch memory, and a work-group
  that asks for more of it means fewer of them run at once. `tools/ocl-micro.c
  occ` sweeps that allocation and counts: 16 KiB → 3.9 work-groups per CU,
  32 KiB → 2.0, 40 KiB → 1.2. That boundary is what `08-winograd-tr-offset`
  acts on, and it corrects an assumption in Finding 12 that the fp32 kernels
  ran 3 to a CU — at 40 KiB each they had a CU apiece. Finding 12's
  conclusions still hold (scratch memory was the binding limit either way),
  but any future occupancy claim should come from this probe.
- **The patches are a series now** (2026-09-15): `patches/dlprimitives/01`–`09`
  are one patch per upstream report, each built and swept on its own on the
  way up (README has the per-patch numbers). The measurement env vars moved
  out of the series on 2026-09-18 and now live in
  `patches/dlprimitives/optional/local-knobs.patch`, which `build.sh` does not
  apply — apply it by hand to reproduce an A/B. `build.sh` recreates
  `scratch/` from the patch files, so any work on the kernels should end with
  `git format-patch` into `patches/`, not with edits left in `scratch/`.

## Open

1. **File the `upstream/` reports.** Drafted, not yet filed: for
   `dlprimitives` the `CUSTOM_REDUCE` auto-detection, the split-K heuristic
   and the activation `dtype` bug (small, unambiguous), plus a proposal for
   the atomics-free paths / access patterns / prefetch that needs the
   author's numbers on NVIDIA and Intel; for `pytorch_dlprim` the half-tensor
   fixes. The scratch-padding report (2026-09-18) is small but **not
   independent** — the same `#define` is a loss against the stock backward
   kernels, so it belongs after the proposal, not before it. Checked 2026-09-15:
   nothing related is filed anywhere, and the `dlprimitives` series applies
   cleanly to upstream HEAD (`b176c15`). Two things about the `dlprimitives`
   maintainer worth knowing: PR #42, AI-generated, was closed after a "who
   wrote this?", and rusticl has been called "a very buggy driver" there
   (pytorch_dlprim #97) — hence the disclosure at the end of each report, and the
   CUSTOM_REDUCE report leading with the OpenCL 3.0 spec rather than rusticl.
2. **A full fp16 tensor path.** The hardware is not the obstacle: the packed
   half-precision multiply reaches 18.7 TFLOP/s through rusticl
   (`tools/fp16-micro.c`). But today only pointwise ops accept half tensors —
   matmul, pooling, softmax, loss and BatchNorm all refuse, and
   `torch.autocast("ocl")` is not registered at all (`tools/half-probe.py` is
   the inventory). `sgemm.cl` is the natural first kernel, same recipe as the
   Winograd fp16 loop. Weeks of work. There are also more pointwise formulas
   in `pytorch_ocl` that mix float constants with half operands and will not
   compile — three found, likely more.
3. **Whether the fp16 inner loop should be the default.** A judgement about
   error tolerance, not speed: NVIDIA's TF32 establishes that a default can be
   lossy, but this is 10× lossier than TF32. Left opt-in.
4. **The remaining gap.** The three Winograd kernels are 76% of the fp32 step
   and run at roughly half the rate the chip's arithmetic units could sustain;
   a T4 reaches 64% of its paper number where this reaches ~52%. The cheap
   wins are taken. Double-buffering the scratch tiles and mixing fp16 inputs
   with fp32 accumulation were both measured and both lost (Finding 12). What
   is left in the kernels is structural — a rewrite, not a tweak. The 14 ms a
   step that is *not* convolution (BatchNorm+ReLU fusion, a fused optimizer)
   is the better next target.
5. **A second GPU** would be interesting for performance, not debugging: RDNA3
   has the hardware float atomic add this work routes around, so the
   planes-versus-atomics trade would come out differently there.

## Constraints

- **Do not overwrite `pt_ocl.so` while a process has it mapped.** `cp`
  rewrites in place and segfaults the process; use `install` or stop it first.
- **`~/.dlprimitives/cache.db` outlives every experiment.** It caches compiled
  kernel *binaries*, keyed on the kernel source and build options — not on
  `ACO_DEBUG`, `AMD_DEBUG` or anything else in the driver's environment.
  Delete it (or point `DLPRIM_CACHE_DIR` at a fresh directory) before
  believing any number, and after any driver-flag experiment. A stale cache
  once measured 478 img/s where a fresh one gave 968, same binary, same GPU.
- **Warm the GPU before measuring.** It idles at 1000 MHz; a cold single test
  measures the wrong machine.
- **Rare failures need 150–400 sweeps** to say anything, and repeating one
  shape hides them entirely — a wrong result then just matches the previous,
  nearly identical iteration.
- **Never replace the host's system Mesa or libclc** while anything else
  depends on the GPU. Build to a local prefix or test in a container.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*

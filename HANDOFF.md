# Handoff: where things stand, what is open

As of 2026-09-15. Nothing here is needed to *use* the patches; the shipped
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
| **+ register prefetch (what ships)** | **2,055** | **7,064** | **0 bad / 300 (Y, dW, dX); BatchNorm within 7e-7 of CPU** |
| opt-in `DLPRIM_CONV_FP16=1` | 2,987 | 10,279 | conv within ~0.5% of fp32; 16-epoch accuracy unchanged |
| T4 (Colab), fp32, for scale | 2,294 | 7,217 | |

Everything is on by default on this device (the planes paths select
themselves wherever there is no native fp32 atomic add; NVIDIA and
`cl_ext_float_atomics` devices keep the atomics). `DLPRIM_WINOGRAD_BWD_PLANES=0`
/ `_SPLIT_PLANES=0` bring the old emulated-atomic kernels back for comparison.

## The one thing to know before debugging a wrong gradient

For two weeks the atomics-free kernels sat behind a safety interlock because
they produced a wrong lane in ~2% of runs, and `ACO_DEBUG=force-waitcnt` made
it go away. It looked exactly like a compiler wait-count bug. It was not: the
GPU clock governor's idle floor had been hand-set to 1000 MHz at 718 mV, and
these dense LDS+ALU kernels were the first code to notice. Pinned there, 66 of
100 sweeps fail with *any* code; at the default 800 mV, 0 of 900. Every
"fix" (`force-waitcnt`, extra `s_nop`s, a `s_waitcnt_depctr` that waits for
nothing) worked only by slowing the code down so the clock did not fall to
the floor as often.

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
  fork fixes it as a drop-in. It did *not* fix `work_group_reduce_*` (a
  rusticl feature gap) or the software `fma()` (a rusticl < 26.2 issue) — both
  patches stay. `tools/libclc-probe.c` checks all three in one run.
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
- **The patches are a series now** (2026-09-15): `patches/dlprimitives/01`–`08`
  are one patch per upstream report, each built and swept on its own on the
  way up (README has the per-patch numbers); `09` holds the measurement env
  vars. `build.sh` recreates `scratch/` from the patch files, so any work on
  the kernels should end with `git format-patch` into `patches/`, not with
  edits left in `scratch/`.

## Open

1. **File the `upstream/` reports.** Drafted, not yet filed: for
   `dlprimitives` the `CUSTOM_REDUCE` auto-detection, the split-K heuristic
   and the activation `dtype` bug (small, unambiguous), plus a proposal for
   the atomics-free paths / access patterns / prefetch that needs the
   author's numbers on NVIDIA and Intel; for `pytorch_dlprim` the half-tensor
   fixes. Checked 2026-09-15:
   nothing related is filed anywhere, and the `dlprimitives` series applies
   cleanly to upstream HEAD (`b176c15`). Two things about the `dlprimitives`
   maintainer worth knowing: PR #42, AI-generated, was closed after a "who
   wrote this?", and rusticl has been called "a very buggy driver" there
   (pytorch_dlprim #97) — hence the disclosure at the end of each report, and the
   CUSTOM_REDUCE report leading with the OpenCL 3.0 spec rather than rusticl.
2. **A full fp16 tensor path.** Packed `v_pk_fma_f16` reaches 18.7 TFLOP/s
   through rusticl (`tools/fp16-micro.c`), so the hardware is not the
   obstacle. Today only pointwise ops accept half tensors; matmul, pooling,
   softmax, loss and BatchNorm refuse, and `torch.autocast("ocl")` is not
   registered (`tools/half-probe.py` is the inventory). `sgemm.cl` is the
   natural first kernel, same recipe as the Winograd fp16 loop. Weeks of work.
   There are also more pointwise formulas in `pytorch_ocl` that mix float
   scalars with `dtype` operands and will not compile for half — three found,
   likely more.
3. **Whether the fp16 inner loop should be the default.** A judgement about
   error tolerance, not speed: TF32 says defaults can be lossy, this is 10×
   lossier. Left opt-in.
4. **The remaining gap.** The three Winograd kernels are 77% of the fp32 step
   and run at ~40% of ALU peak; a T4 reaches 64% of its paper number where
   this reaches 46%. The cheap wins are taken. LDS double-buffering and
   fp16-in/fp32-accumulate were measured and rejected (Finding 12). What is
   left in the kernels is structural — a rewrite, not a tweak. The
   non-convolution 14 ms per step (BatchNorm+ReLU fusion, a fused optimizer)
   is the better next target.
5. **A second GPU** would be interesting for performance, not debugging: RDNA3
   has the hardware fp32 atomic add this work routes around, so the
   planes-vs-atomics trade would look different there.

## Constraints

- **Do not overwrite `pt_ocl.so` while a process has it mapped.** `cp`
  rewrites in place and segfaults the process; use `install` or stop it first.
- **`~/.dlprimitives/cache.db` outlives every experiment.** It caches compiled
  kernel *binaries*, keyed on kernel source and build options — not on
  `ACO_DEBUG`, `AMD_DEBUG` or anything else in the driver's environment.
  Delete it (or point `DLPRIM_CACHE_DIR` at a fresh directory) before
  believing any number, and after any driver-flag experiment. A stale cache
  once measured 478 img/s where a fresh one gave 968, same binary, same GPU.
- **Warm the GPU before measuring.** It idles at 1000 MHz; a cold single test
  measures the wrong machine.
- **Rare failures need 150–400 sweeps** to say anything, and repeating one
  shape hides them entirely — the stale value is then the previous iteration's
  near-identical one.
- **Never replace the host's system Mesa or libclc** while anything else
  depends on the GPU. Build to a local prefix or test in a container.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*

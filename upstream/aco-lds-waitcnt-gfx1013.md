# Withdrawn: "ACO/rusticl gfx1013: stale `__local` read at one lane"

**Not filed. Kept as a record of how a voltage problem impersonates a
compiler bug.** Full account: OPENCL-PERF.md, Finding 3, *It was the voltage*.

## What it looked like

On the AMD BC-250 (gfx1013, Mesa 26.1.8 and 26.2.2, `RUSTICL_ENABLE=radeonsi`),
DLPrimitives' `winconv_3x3_bwd_filter` kernel — a plain
`write __local → barrier → vload16 → barrier` epilogue, checked by hand —
intermittently produced garbage in exactly nine consecutive floats: one
work-item's whole output, always the same lane within the work-group's tile.
About 2% of invocations on 26.1.8, about 20% on 26.2.2.

Everything pointed at `s_waitcnt` insertion around the `ds_read`:

| variant | bad / 150 runs |
| --- | ---: |
| as written | 3–7 |
| `atomic_xchg` store, or 16 scalar reads instead of `vload16` | 6–9 |
| `ACO_DEBUG=nosched` | 20 |
| `ACO_DEBUG=force-waitcnt` | **0** (and 0 in a further 400) |

Changing the load or store form did nothing; `nosched` made it worse;
conservative wait-counts made it vanish. `force-waitcnt` cost a third of the
machine (975 → 650 img/s), which is why it was a diagnosis and not a fix.

## What it was

Every one of those numbers measured how often a bursty workload let the clock
governor drop the GPU to its idle floor, which had been hand-set on 2026-09-08
to 1000 MHz at 718 mV. Pinned there, 66 of 100 sweeps fail with *any* code;
pinned at 1200 MHz or above, or at 1000 MHz with the governor's default
800 mV, 0 of 900. `force-waitcnt`, `nosched` and "worse on 26.2" all changed
the failure rate by changing the timing. The giveaway was a Mesa built from
source (`tools/mesa-dev/`) in which a `s_waitcnt_depctr` that waits for
*nothing* before every instruction also "fixed" it.

## The one observation that survives

`ACO_DEBUG` is not part of Mesa's shader cache key. With a warm on-disk cache,
`ACO_DEBUG=force-waitcnt` has no effect at all — the cached binaries were
compiled without it and are reused. Any experiment with ACO flags needs
`MESA_SHADER_CACHE_DISABLE=true`. It cost a round of measurements here that
looked like a free fix and were actually the unchanged binaries.

## Lessons

- A correctness failure that responds to *any* change in timing is a hardware
  or voltage question first. Pin the clock before reading compiler code.
- A stale-read symptom is easy to miss: it is only visible when the stale
  value differs enough from the correct one. Repeating one shape in a loop
  hides it entirely.
- Rare failures need hundreds of sweeps across several shapes before a
  before/after comparison means anything.

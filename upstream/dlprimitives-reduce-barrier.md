# dlprimitives: the CUSTOM_REDUCE fallback races between back-to-back reductions

**Where to file:** https://github.com/artyom-beilis/dlprimitives/issues
**Against:** `ff2d590` (2024-09-04; `reduce.h` is unchanged on master `b176c15`); `src/kernels/reduce.h`

## Summary

With `CUSTOM_REDUCE=1` (or any device below OpenCL 2.0, which takes the same
path) `test_test_case_softmax` and `test_test_case_log_softmax` fail on the
large-channel shapes:

```
-- test for shape [[100,1000,10]] fwd
Comparison failed for tensor :0 at 102866 expecting 0.0022336 got 0.00120649 for esp=0.001
-- test for shape [[120,2000,5]] fwd
Comparison failed for tensor :0 at 470002 expecting -7.13514 got -6.49718 for esp=0.003
```

Every wrong value in a row is off by the same factor (softmax) or the same
offset (log-softmax): the row's normalising sum is wrong, not the elements.
A one-line fix makes both tests pass.

## Cause

`REDUCE_USING_OP` in `reduce.h` reduces through a `__local` array and ends
with every work-item reading the result from slot 0:

```c
        for(int i=WGS / 2;i>0; i>>= 1) {
            if(lid < i) {
                my_reduce[lid] = reduce_op(my_reduce[lid],my_reduce[lid+i]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        myval = my_reduce[0];
    } while(0)
```

There is no barrier after that read. A kernel that reduces twice in a row
enters the macro again, and its first statement is `my_reduce[lid] = myval` —
so work-item 0 overwrites slot 0 with its own partial of the *second*
reduction while slower work-items may still be reading the result of the
first. `softmax.cl` does exactly this: `my_work_group_reduce_max(val)` for
the row maximum, a loop of `exp()` over the work-item's elements, then
`my_work_group_reduce_add(sum)`. Wavefronts that reach the second reduction
early clobber the maximum before late wavefronts have read it, and those
wavefronts then exponentiate against the wrong maximum. Small rows never
show it because the whole group runs in lock-step; the failing shapes are
the ones with enough elements per work-item that wavefronts drift apart
between the two reductions.

The same pair-of-reductions pattern is in `softmax_with_loss.cl`,
`bn_sums.cl`, `global_pooling.cl` and `depthwise_separable_bw_filter.cl`;
they are exposed to the same race and happen not to fail the tests today.

The OpenCL 2.0 path (`work_group_reduce_*`) is unaffected, which is why this
does not show up on platforms that have the built-ins.

## Fix

A barrier after the read:

```c
        myval = my_reduce[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    } while(0)
```

Patch: `patches/dlprimitives/01-reduce-barrier.patch` in
https://github.com/mxreyer/pytorch-dlprim-gfx1013. Cost is one extra barrier
per reduction; nothing measurable on the ResNet-9 step, where the reductions
are a few percent of the total.

## Disclosure

This report, and the patch, tools and measurements it cites, were produced
with Claude (Anthropic) as the coding assistant; it drafted the text and most
of the code. I reviewed the filing and take responsibility for its contents.

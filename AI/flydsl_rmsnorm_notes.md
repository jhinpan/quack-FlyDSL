# FlyDSL RMSNorm backend notes

Findings from building the opt-in ROCm RMSNorm backend (`quack/flydsl/`,
`quack/rmsnorm_flydsl.py`). Measured on MI355X / gfx950 with FlyDSL 0.2.4 and
torch 2.9.1+rocm7.2.0 unless stated otherwise.

## Quack RMSNorm API coverage

The FlyDSL entry point implements the upstream `quack.rmsnorm` signature,
including optional weight and bias, `weight_offset`, independent output dtype,
residual/prenorm output, residual dtype override, and per-head parameters.
One descriptor-safe kernel serves every case, specialized by compile-time
feature flags; the plain weighted case is that kernel with `has_weight=True`
and every other flag off. There is no separate plain path -- see "The plain
path is gone" below for what merging the two cost.

Backward has one weight-gradient reduction at every row count: a persistent
kernel writing one partial per block, then a final reduce. It is a fixed
reduction tree, so the backward is bitwise reproducible with no opt-in.
Per-head workspaces and final parameter-gradient stores use row-scoped buffer
descriptors so neither temporary nor output addressing silently wraps at 4 GiB.

## The baseline is the compiler, not another hand-written kernel

Bias, residual, prenorm, per-head and dtype overrides have no cheaper
hand-written alternative here, so what a caller gives up by using them is
`torch.compile`, which fuses this shape of work close to the roofline. That is
the number to beat, and it is a demanding one: inductor reaches 4.8 TB/s on
fused residual + prenorm, above anything this backend achieves on any other
shape.

A first cut of that kernel was scalar and re-read the row from global memory
after the reduction, which put it well under that bar. Measured at 32768x2048
bf16, forward, against `torch.compile`:

| case | scalar first cut | vectorized | `torch.compile` |
| --- | --- | --- | --- |
| `+bias` | 2.89 TB/s (0.72x) | 4.36 TB/s (1.08x) | 4.02 TB/s |
| `weight_offset=1` | 3.04 TB/s (0.77x) | 4.36 TB/s (1.11x) | 3.93 TB/s |
| `+residual+prenorm` | 2.71 TB/s (0.57x) | 4.75 TB/s (1.00x) | 4.77 TB/s |

Three things closed that gap:

1. **Take the vector width from the activation dtype and let every other
   operand cover that same span.** `vector_access_plan` splits an operand into
   whole accesses of at most 128 bits, so a 32-bit operand under a full 16-bit
   vector takes two accesses and everything else takes one. A row that is not a
   whole number of vectors narrows the vector rather than dropping to scalar,
   and `vecsize == 1` then falls out of the same code as the scalar case
   instead of needing a second kernel body.
2. **Hold the row in registers across the reduction.** RMSNorm reads each row
   twice by construction; only the first read has to touch memory. With a fused
   residual the cached value is the fp32 sum, which is what the second pass and
   `residual_out` both want anyway. In the persistent backward the weight is
   loop-invariant, so it is loaded once per block with `weight_offset` already
   folded in, not once per row.
3. **Do not materialize `residual_out` when nobody reads it.** It is only ever
   read by the caller under `prenorm`, or by backward as the saved source of a
   fused residual. Gating on that instead of on `residual is not None` takes a
   full tensor write off inference: 0.114 ms to 0.098 ms at 32768x2048.

Backward lands at 0.98x-1.44x of `torch.compile` on the same shapes. The `dbias`
case is the closest (0.98x): both implementations sit near 2.3 TB/s there, so
the second full-row parameter reduction, not the kernel, is what bounds it.

## fp32 atomics cost two launches to save one

A small-row atomic backward used to run below 512 rows, on the argument that it
avoided a workspace and a second kernel launch where launch overhead dominates.
Measured against the staged path it was slower at every row count that used it,
and the reason is the numerics it needs rather than the kernel.

fp32 atomic accumulation forces two things on the caller. The accumulator has to
start at zero, and it has to be fp32, so the result has to be cast back to the
weight dtype. In eager torch each of those is a kernel launch of its own:
`torch.zeros(n, fp32)` costs 4.4us of host time and `fp32.to(bf16)` 5.5us, both
for ~2.1us of device work, against 3.2us for the second FlyDSL launch the atomic
path avoids. The staged reduce kernel writes every element in the weight's own
dtype, so it needs neither. Counting what each path dispatches at m=1, n=2048:

| weight dtype | atomic | staged |
| --- | --- | --- |
| bf16 / fp16 | 3 kernels: fill, backward, cast | 2 kernels: partial, reduce |
| fp32 | 2 kernels: fill, backward | 2 kernels: partial, reduce |

So with a weight in the activation dtype -- the ordinary training case -- the
path built to save a launch issued one more than the path it replaced. Removing
it, on the whole backward op, interleaved medians over 21 rounds:

| weight dtype | speedup | absolute |
| --- | --- | --- |
| bf16 (n=1024/2048/8192, m=1..256) | 1.29x-1.33x | 24.8us to 18.8us |
| fp32, same cells | 1.02x-1.05x | 20.0us to 19.2us |

fp32 weight gains too, though less: the cast is already a no-op there, so what
is left is that a FlyDSL launch is cheaper than a torch memset launch.

The workspace argument did not hold either. `num_programs` is
`min(next_power_of_two(m), CU-derived)`, so the grid already shrinks with the
row count and the staged workspace at m=1 is `n * 4` bytes -- 4 to 32 KiB. At
the top of the old atomic range, m=511, it is 2 to 8 MiB, roughly the size of
the input, and transient in the caching allocator.

Device time alone says the same thing, so this does not depend on eager launch
cost: with the launches discounted the atomic path won 1 of 12 cells (fp32
weight, n=2048, m=1, 5.75us against 7.13us). It was scalar, so its device time
grew with `m * n` -- at n=8192, m=1 it was already 2.8x behind.

The threshold was also wrong on its own terms. Comparing the two kernels alone,
the crossover depends on `n`, which a fixed row count cannot express: about
m=370 at n=512, m=250 at n=2048, and m=8 at n=8192, where the atomic kernel was
3.1x slower by m=511.

Two things came out with it. The parameter accumulators were zeroed on every
call, including the `(1,)` placeholder for a gradient nobody asked for, and
`torch.zeros((1,))` is still a full kernel
launch; the reduce writes what it is asked for, so `torch.empty` is enough. The
effect is inside the noise of the autograd path, but the launches are gone.
And the raw MLIR `atomic_add` went with the kernel, which leaves nothing in this
backend importing FlyDSL's private `_mlir` APIs. That was half the reason for
the `flydsl<0.3` pin; the other half was that only 0.2.4 had been run, and 0.3
is still a dev build, so the pin stays until a release is tested.

Unlike the rest of this file, the numbers above were taken on flydsl
0.3.0.dev765, with the suite passing there as well.

## Launch geometry cannot be computed in traced Python

`num_programs` came from the row config, which takes `math.gcd(N, ...)`, and
Dynamo cannot trace a gcd over a symbolic shape. Resolving it in `_rmsnorm_bwd`
therefore broke `torch.compile(fullgraph=True, dynamic=True)` outright for every
row count that reached the staged path -- which the suite missed because it only
compiled small batches, and those returned from the path selector before they
touched the config.

The forward never had the problem because it resolves its tiling inside the
launcher, behind the opaque custom op. The backward now does the same with
`num_programs` and its workspace, which is also where the allocation belongs
for timing: the benchmark had been hoisting it out of the timed region.

The general rule: anything a kernel is specialized on has to be a build-time
constant, so it must be resolved behind the op, not in the graph.

## A buffer descriptor addresses at most 4 GiB

`fx.rocdl.make_buffer_tensor` defaults to `max_size=True`, which sets
`num_records` to `0xFFFFFFFF`. Wrapping a whole operand and *then* slicing a
row resolves the row offset inside that 4 GiB window, so every row past the
mark wraps to the start of the allocation and silently returns another row's
data. No fault, no error.

Measured at the exact boundary, bf16 with N=8192:

| rows | operand bytes | last-row max abs error |
| --- | --- | --- |
| 262143 | just under 4 GiB | 0 |
| 262145 | just over 4 GiB | 5.719 |

The head of the tensor stays exact throughout, which is why it looks fine
until someone trains at that size.

Slice the row first and build the descriptor over the row
(`num_records_bytes=N * elem_bytes`). That also turns the hardware bounds
check into a real per-row guard instead of a 4 GiB no-op. It costs one
descriptor construction per row in the persistent backward: median 0.4%
forward, 1.8% backward, worst single cell 11%.

## FlyDSL has two architecture authorities, and they read different variables

- `flydsl.compiler.backends.get_backend().target.arch` resolves as
  `env.compile.arch or get_rocm_arch()`, and `env.compile.arch` reads **ARCH**.
- `flydsl.runtime.device.get_rocm_arch()` reads **FLYDSL_GPU_ARCH** (or
  `HSA_OVERRIDE_GFX_VERSION`), never ARCH.

Setting the two to different values makes the compiler generate code for one
architecture while any helper built on `get_rocm_arch()` — wavefront size,
the packed bf16 convert — believes it is on the other:

| environment | compiler target | `get_rocm_arch()` |
| --- | --- | --- |
| `ARCH=gfx942 FLYDSL_GPU_ARCH=gfx950` | gfx942 | gfx950 |
| `ARCH=gfx950 FLYDSL_GPU_ARCH=gfx942` | gfx950 | gfx942 |

Resolve the architecture once, validate it against the device, and pass that
one value into every builder.

## The AST rewriter does not follow calls

FlyDSL rewrites the AST of the `@flyc.kernel` / `@flyc.jit` function itself.
A module-level helper is executed as ordinary Python, so any data-dependent
`if` inside it (`if lane == 0:`) raises
`cannot evaluate dynamic 'Boolean' as Python bool during tracing`.

Helpers that only move data or compute on traced values are fine — that is
why `load_vec`, `store_scalar` and friends live in `rmsnorm_common.py`. Block
reductions cannot be shared this way and are written out in each kernel on
purpose.

## Software bf16 rounding is bit-identical to the gfx950 convert

Pre-gfx95x has no packed fp32-to-bf16 convert, so the kernel rounds to
nearest even by hand. Building the same kernel both ways on gfx950 and
comparing over a full tensor gives an exact match, including NaN, the
infinities, signed zero and subnormals. That branch is otherwise dead on
gfx950 and would never be exercised.

This validates the rounding *code path*, not any pre-gfx95x part. Nothing
here has run on gfx942, which is why the backend claims gfx950 only.

## The row is aligned, so the vector size belongs to the dtype

`from_analytical_heuristic` still writes `vecsize = gcd(N, 128 // dtype_width)`,
the rule `quack/rmsnorm.py` uses, but the adapter admits only rows that are a
multiple of `N_ALIGNMENT`, so the gcd always comes back at full width -- 8 for
16-bit, 4 for fp32 -- and the lower rungs of that ladder are unreachable.

The ladder earned its place while the backend took any row. Serving a length
divisible by 4 but not 8 with a narrower access rather than a scalar loop was
worth about 1.5x:

| shape | scalar fallback | gcd rule |
| --- | --- | --- |
| 32768 x 4092 bf16 | 159.5 us | 107.3 us |
| 32768 x 4094 bf16 | 161.9 us | 107.9 us |
| 32768 x 8188 bf16 | 302.3 us | 205.1 us |

What changed is the scope, not the measurement. No hidden size in practice is
anything but a multiple of 64 or 128 -- 3584, 4608, 5120 and 7168 all are -- so
those rungs only ever ran on shapes nobody feeds this kernel, and such shapes
distort what gets optimized: this file gated a whole refactor on 131072x257 for
exactly that reason. The adapter now refuses them and names `quack.rmsnorm`,
which serves any length, as the fallback.

Three things fell out with them. `use_multi_row_kernel` and
`batch_short_rows` were two predicates that could only disagree on an
unaligned row, and are now one. `SMALL_ROW_THRESHOLD` existed to decide what to
do with a long scalar row, and there are none. `_select_rmsnorm_bwd_programs`
branched on whether the row vectorized, and it always does.

The gcd stays in the config rather than being replaced by the constant, so a
row that somehow reaches the kernel unaligned narrows its access instead of
reading past the end of the row.

## Persistent kernels need their block sized to the row too

The staged backward first pinned its block at 512 threads. A 1024-wide bf16
row is 128 vectors, so three quarters of the block had no column and the
kernel fell back to scalar I/O, landing at 29% of the copy roofline. Sizing
the block to the row and launching proportionally more blocks took
32768x1024 fp16 from 137.5 us to 62.4 us (2.20x, 29% -> 62% of roofline).

## A row too short for a block still needs its geometry in vectors

Rows shorter than a block share one, a group of lanes each. That kernel had
never been vectorized: it sized the group in elements, so a 128-element BF16
row was four scalar 16-bit loads per lane across 32 lanes, and it read the row
from memory twice. It was the slowest path in the backend, and slower than
routing the same row to the one-block-per-row kernel, which is at least
vectorized.

The fix is that the vector width is a property of the row, not of who covers
it, so `RmsNormRowConfig` answers for both. `for_lane_group` drops the
block-width floor -- that floor exists so a block is never a partial wave,
which says nothing about a group sharing a block -- and caps at a wavefront,
which is what keeps the row reduction a bare shuffle with no LDS and no
barrier. A 128-element BF16 row is then one 128-bit access per lane across 16
lanes, with 16 rows to a block.

Forward, BF16, against `torch.compile`:

| shape | scalar group | vectorized group | `torch.compile` |
| --- | --- | --- | --- |
| 262144 x 128 | 1.44 TB/s | 3.72 TB/s | 3.78 TB/s |
| 262144 x 256 | 1.47 TB/s | 4.29 TB/s | 3.96 TB/s |

Before any of it existed a short row got a block to itself, which leaves 48 of
64 lanes with nothing to load on a 128-element row. Sharing the block took
`weight_offset` at 262144x128 from 1.86 to 3.79 TB/s, which is parity with
inductor, and 131072x256 from 3.03 to 3.83. That is the QK-norm shape, so it is
worth having.

Batching stops once a lane group would have to loop: a block of its own is
faster then, by 1.4x at 33554432x1021 and 1.9x at 2047. On an aligned row that
condition and "the row has fewer vectors than the block-width floor" are the
same condition, which is why one predicate now answers for both.

Why a batched kernel loses a deep tile loop at all is still open. At 16384x2047
it compiled to the same 28 VGPRs with no scratch and no LDS, launched the same
grid with the same block, and still ran 1.4x behind a block per row;
predication, the tail guard, the fp32 register cache and `weight_offset` were
each measured out. Both shapes that exposed it are unaligned and no longer
admitted, so this is recorded rather than chased.

Rows of 64 remain at 0.79x of inductor, and the plain path sat at the same
2.99 TB/s there while it existed, so that gap is older than the batching and
survived the merge below.

## The tail block's guard depends on how deep the tile loop is

Batching rounds the grid up, so the last block holds groups with no row. There
are three ways to stop them, and which one wins is decided by the tile count,
not by taste. The thing to minimize is exec-mask updates per row.

On a row a group covers in **one pass**, the descriptor wins. Sizing that
group's `num_records` to zero bytes makes the hardware discard its loads and
stores and leaves every lane free to keep taking part in the reduction shuffle.
Predicating the stores instead cost 8% here, and up to 1.5x once the loop was
deep, because it turns one exec-mask update into one per tile. This is what the
forward does, and it is safe precisely because `batch_short_rows` only batches
rows a group covers in one pass.

On a **deep** tile loop, one uniform branch around the whole row wins, and by a
lot. The forward no longer batches such a row -- `batch_short_rows` stops at
the block-width floor, and on an aligned row that already means one pass -- so
this is a case the surviving path cannot reach. Keep it that way: while a
deep batched loop was reachable, leaving the tail to the descriptor made the
compiler predicate every access on its own, 64 `s_cbranch_execnz` against one
and 1311 instructions
against 476, costing 1.6x on 16384x2047 and 131072x257. Neither VGPR count (52
against 49) nor spilling explains it -- there is no spilling either way and both
land in the same occupancy bucket. It is purely the per-access exec-mask
updates. The branch is legitimate because the row index is uniform across a
group, so the shuffles stay collective for every group that has a row.

`rstd` is bounded by its descriptor in both kernels, sized to the real row or
program count instead of left wide open, which is a tighter bound than the guard
needs and costs nothing.

## A parameter reduce has to open on the rows, not just the parameter

Stage 2 of the staged backward reduces a `num_programs x n` workspace down to
one parameter. Its grid was opened on the parameter alone -- one thread per
output element, each walking every partial row -- so a 256-element weight ran
on a single block whatever the row count, and stage 2 became the whole of the
short-row backward: 159us against a 13us stage 1, slower than a bare
`torch.sum` on the same workspace.

Block by column and give the partial rows their own lanes, combining them
through LDS. Two things make it pay. The descriptor has to come out of the
accumulation loop, because the row a lane reads now depends on the lane and a
divergent buffer descriptor costs more than the split saves -- the reduce
therefore takes the workspace flat, one descriptor for all of it, which is safe
because `num_programs` is CU-derived. And the reduce writes each gradient in
the parameter's own dtype, so nothing casts on the way out.

Reduce, bf16, m=32768, against `torch.sum` on the same workspace:

| n | before | after | `torch.sum` |
| --- | --- | --- | --- |
| 256 | 159.2us | 19.6us | 19.8us |
| 1024 | 88.9us | 10.6us | 19.6us |
| 4096 | 21.5us | 4.2us | 30.9us |

Reproducible to a tenth of a microsecond across runs, and dweight comes out bit
for bit what the old plain reduce produced, because the summation order now
matches it.

## The plain path is gone

The feature builder was always a strict superset -- plain is `has_weight=True`
with every other flag off -- so `build_rmsnorm_module`,
`build_rmsnorm_bwd_two_stage_module`, `_RMSNormFunction` and the
`plain_optimized` predicate were a second implementation of a subset, and the
three sources of truth for what the backend supports were down to one.

This file used to say the blocker was coprime rows, where the feature forward
lost 2.3x on 131072x257. That was the wrong measurement to gate on: every
official shape vectorizes, because `MN_PAIRS` is powers of two and
`COMPACT_SHAPES` adds only (4096, 3000), where `gcd(3000, 8) == 8`. On the
shapes the benchmark reports, the feature forward was already within 0.96x-1.04x
of plain across all 21 shape/dtype cells. The real blocker was the backward,
which this file had never compared path against path -- 1.3x-4.8x behind at
every n up to 3000, all of it the reduce above.

What the merge cost, official harness, bf16, medians of alternating runs:

| case | plain | feature |
| --- | --- | --- |
| fwd 32768x4096 | 94.7us | 96.0us |
| bwd 32768x4096 | 196.3us | 164.7us |
| fwd 32768x256 | 11.6us | 13.9us |
| bwd 32768x256 | 46.6us | 39.9us |

The backward is 1.16x-1.19x faster and the long-row forward is parity. The one
regression is the forward on the shortest official row, and it is in the kernel
rather than around it: 7.1us of device time against roughly 5.7us, with the
adapter measured at zero overhead either way. It is not chased down. That is
the standing price of one implementation instead of two.

## Native forward autotuning

The first autotuning stage is forward-only and opt-in through
`quack.rmsnorm_flydsl.rmsnorm_autotuned`. The existing `rmsnorm` entry and its
`_FWD_CACHE` keep the analytical heuristic unchanged. A normal call to the
tuned entry also serves that heuristic without searching; set
`FLYDSL_AUTOTUNE=1` to force FlyDSL's native search. FlyDSL stores scratch
winners under `FLYDSL_AUTOTUNE_CACHE_DIR` (default `~/.flydsl/autotune`) and
portable artifacts under `FLYDSL_AUTOTUNE_CONFIG_DIR`.

The direct `@flyc.jit` entry injects the per-row thread count as a Constexpr and
passes `waves_per_eu` as a compiler option. Candidates are the legal subset of
the heuristic width, its half/double, and 64/128/256 threads, crossed with the
default occupancy and 1/2/4 waves per EU. Generation removes duplicates and
rejects widths that exceed the wave/block limit or the existing 32-elements per
thread register budget. The key includes M/N, all operand dtypes, feature flags,
per-head mode/count, target architecture, and a schema version. Runtime `eps`
and `weight_offset` deliberately do not split winners.

Each timing sample wraps 100 launches in one HIP event pair and divides the
elapsed time by 100; the tuner takes the median of seven samples after an
untimed compile and one warmup batch. Outputs are complete stores, so tuning
does not zero `output`, `residual_out`, or `rstd`.

Validation on gfx950 with FlyDSL 0.3.0 used BF16 `16 x 4096` with FP32 weight.
All eight candidates compiled and ran; the measured winner was 256 threads at
0.073 ms per launch, the emitted artifact recorded the full key, and a fresh
process served the persisted winner with benchmarking replaced by a hard
failure. Both the search result and a cache hit with different `eps` and
`weight_offset` matched the FP32 reference after BF16 rounding exactly in that
run. Tests also cover residual+bias+prenorm+rstd on a non-default stream and a
`torch.compile(fullgraph=True, dynamic=True)` row-count change. Backward remains
on its existing deterministic heuristic and is not tuned in this stage.

## Measuring this backend

The suite figure quoted for this branch -- **626 passed, 2 skipped**, where the
two skips are the two- and eight-device tests -- is this invocation:

```
HIP_VISIBLE_DEVICES=<idle> python -m pytest \
    tests/test_rmsnorm_flydsl.py tests/test_rmsnorm_flydsl_config.py \
    tests/test_import_isolation.py -q
```

`python -m pytest tests/` does *not* work on a ROCm host: 37 modules fail
collection with `ModuleNotFoundError: No module named 'cuda'`, because the
cutedsl tests import `cuda.bindings`. That is also why the cutedsl head-to-head
cannot be run on this machine.

## Against torch.compile on the same part

`benchmarks/benchmark_rmsnorm_flydsl.py`, bf16 activations with fp32 weight on
MI355X, correctness gated per cell before anything is timed. Speedup is
torch.compile's time over ours, so above 1.0 is ours:

| M x N | fwd | bwd |
| --- | --- | --- |
| 4096 x 3000 | 0.86x | 1.21x |
| 4096 x 4096 | 0.97x | 1.05x |
| 32768 x 1024 | 1.01x | 0.99x |
| 32768 x 2048 | 1.07x | 1.51x |
| 32768 x 4096 | 0.99x | 1.49x |
| 32768 x 8192 | 0.95x | 1.67x |

**The forward is at parity, not ahead.** Four of these six cells go to
torch.compile and two to us, all inside ±15%. Inductor generates a good fused
RMSNorm forward, and matching it is the honest result. The backward is where
this backend wins, 1.5-1.7x once the row is wide, because torch's backward
stops scaling: at `32768x8192` we hold 5066 GB/s against its 3031.

The three shapes at or below `M=512` are omitted because they are launch-bound
and do not reproduce. Forward `256x4096` read 294, 379 and 462 GB/s across
consecutive sweeps, and backward `512x4096` swings either side of torch by more
than 1.5x. Treat that band as order-of-magnitude only. The six rows above are
stable to about 0.2% between sweeps.

Two corrections behind these numbers, both of which moved the conclusion:

1. *`torch.compile` needs `dynamic=False` and a `torch._dynamo.reset()` per
   cell.* Without them Dynamo state accumulates across a sweep and the last
   shape reads **9x** slower than the same shape measured alone -- 1.8750 ms
   against 0.1985 ms, reproducible to four digits, so it looks like a finding
   rather than noise. Every cell now agrees with a per-cell run.
2. *The previous figures are withdrawn.* An earlier version of this file quoted
   "38 of 45 cells, median 1.46x forward; 45 of 45, median 4.51x backward"
   against eager `torch.nn.functional.rms_norm`, plus a three-machine table of
   roofline shares against the CuTe kernel. They came from a harness that sized
   its cache-eviction gate from the 4 MiB per-XCD L2 torch reports rather than
   the 256 MiB MALL, so 41 of its 90 cells never evicted and had no
   cold-measurement contract. Beating eager torch by 1.46x and merely matching
   `torch.compile` are consistent claims, but only the second one is measured
   here. The cross-vendor comparison is not measurable on this host at all: the
   CuTe backend cannot be imported on ROCm.

## `MAX_N = 8192` is a real cliff, but not where the constant says

`rmsnorm_config.MAX_N` rejects any row wider than 8192. Monkeypatching it to
`1 << 20` (probe only, nothing committed) shows it is not a correctness bound:
the forward runs to N = 262144 and the backward to 65536 at bf16 accuracy
indistinguishable from the shapes under the cap, with nothing spilling, wrapping
or silently truncating.

There *is* a cliff worth keeping a cap for, and it sits between **49152 and
57344**. Holding `m` at 4096 isolates width against the `two_read_one_write`
ceiling: share of ceiling is flat at 83.1% for N=32768 and 77.5% for N=49152,
then halves to 37.8% at 57344 and stays there, reading 39.6% at 65536 and 39.2%
at 98304. The step is abrupt rather than a slope, and eager and graph-replay
timing agree on all five widths to within 0.9 points.

So the cap stays at 8192 as policy, not as the register bound its name suggests.
cutedsl gates only at `N > 128k with dtype >= 32 bits` and escapes the register
budget above 8k with `reload_from="smem"`, which this backend has no analogue
for.

## Feature parity with the cutedsl backend

| feature | cutedsl | FlyDSL | |
| --- | --- | --- | --- |
| bias, residual (fused add), prenorm | yes | yes | parity |
| weight_offset (`w+1` fusion) | yes | yes | parity |
| per-head affine | yes | yes | parity |
| store_rstd | yes | yes | parity |
| autotune | yes | yes | `quack/flydsl/rmsnorm_autotune.py` |
| persistent backward | yes | yes | different knob name |
| split parameter reduce | `dw_partial` | yes | two-stage, no atomic variant |
| dual dx dtype | yes | partial | narrower |
| **layernorm / mean** | **yes** | **no** | **real gap** |
| cluster / multicast | yes | n/a | SM90+ DSMEM, no gfx950 analogue |

A token-count diff of the two modules gets four of these rows wrong, all
reporting a gap where there is only a different spelling: `autotune` and
`persistent` live in `quack/flydsl/` rather than in `rmsnorm_flydsl.py`, so a
module-scoped grep reads 0 on both; `dw_partial` is spelled
`partial`/`dweight_total` in `rmsnorm_bwd_kernel.py` with the same
per-block-partials mechanism; and `sm_count` is cutedsl's persistent-launch knob
under another name.

`cluster` is a real difference but not a deficit. It is Hopper-and-later
distributed shared memory, which gfx950 does not have, so counting it would make
the backend permanently non-compliant with hardware it does not run on.

**The one genuine feature gap is layernorm.** `quack/rmsnorm.py` carries
`is_layernorm` through the whole stack, forward and backward, with `mean`
alongside `rstd` and a `layernorm_fwd` / `layernorm_bwd` / `layernorm_ref`
surface. FlyDSL has none of it. It is not exported from `quack/__init__.py`, so
nothing in-tree consumes it today, but it is the item to name when asked what
the backend does not yet do.

Second and smaller: cutedsl exposes `rmsnorm_fwd` and `rmsnorm_bwd` as public
entry points where FlyDSL exposes only `rmsnorm`. The top-level `rmsnorm()`
signatures are argument-for-argument identical, so callers of the public API are
unaffected, but it does mean the benchmark has no low-level entry to time and
goes through `rmsnorm()` and autograd for both of its providers.

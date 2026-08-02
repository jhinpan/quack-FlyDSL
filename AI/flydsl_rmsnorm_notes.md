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

The suite figure quoted in every handoff on this branch -- **737 passed, 2
skipped, 1 xfailed** -- is this invocation, and it is written down here
because I quoted it repeatedly without recording it and then could not
reproduce my own number. The xfail is the import-boundary test below and
has to be quoted with the rest: a gate figure that omits it silently
tolerates that test disappearing:

```
HIP_VISIBLE_DEVICES=<idle> python -m pytest \
    tests/test_rmsnorm_flydsl.py tests/test_rmsnorm_flydsl_config.py \
    tests/test_benchmark_rmsnorm_flydsl.py tests/test_import_isolation.py -q
```

The two skips are the two- and eight-device tests. `python -m pytest tests/`
does *not* work on this host: 37 modules fail collection with
`ModuleNotFoundError: No module named 'cuda'`, because the cutedsl tests import
`cuda.bindings`, which does not exist on ROCm. That is also why the cutedsl
head-to-head cannot be run here at all. Dropping `test_import_isolation.py`
gives 733, which is the arithmetic behind a discrepancy I spent a while
chasing.

`AI/probe_rmsnorm_flydsl_bandwidth.py` backs the throughput numbers here and
`AI/probe_rmsnorm_flydsl_accuracy.py` backs the error figures. Run both before
and after a change; between them they cover long rows, the short rows the
batching is for, and the hidden sizes real models use, whose last tile is a
partial one where a power of two's is not.

Backward cleanups also have a component gate, so a faster partial kernel cannot
hide a slower finalizer or vice versa:

```
python AI/probe_rmsnorm_flydsl_backward_components.py --output /tmp/before.json
python AI/probe_rmsnorm_flydsl_backward_components.py \
       --baseline /tmp/before.json --output /tmp/after.json
```

The probe takes the median of several ROCm-profiler rounds for the persistent
partial and FlyDSL reduce kernels independently. By default it rejects a stage
that loses more than the larger of 2% or 0.5 us. It records synchronized
full-operation latency too, but does not automatically gate it: the shared
node's host issue floor drifts by 1-2 us between processes, so full-path
comparisons use the contention canary and alternating protocol in
`benchmarks/benchmark_rmsnorm_flydsl.py`. Generate the baseline from an
immutable worktree on the same machine immediately before the candidate.

Three traps, all of which produced a confident wrong answer at least once:

`triton.testing.do_bench` picks its repeat count from a first call, so a first
call that also runs the FlyDSL build skews the whole sample -- and only the
first shape measured in a process, which makes it look like a property of that
shape. It read as a 0.59x regression at 262144x128 that vanished when the same
shape was measured second. Call the function once and synchronize first.

Roughly one sample in twenty comes back an order of magnitude slow, on either
implementation, with no other tenant on the device. Contention can only cost
time and never save it, so take the minimum of a few samples rather than one.
A single sample is how 262144x128 once reported 0.75 TB/s against its own
3.80.

**A `torch.cuda.Event` pair times the device that was current when it was first
recorded, not the device the work ran on.** Passing `device="cuda:6"` to every
tensor does not move the process's *current* device, which stays 0. `record()`
then binds the event to device 0's stream, and `elapsed_time` returns the gap
between two markers on an idle device -- a number that is neither the kernel's
duration nor obviously wrong. It read 26.9 us for a 32768x4096 bf16 forward
that actually takes 89.9 us (raw samples in
`AI/data/rmsnorm_32768x4096_bf16_roofline.json`), and the error is not a
constant factor, so it cannot be divided out afterwards. Either
`torch.cuda.set_device(N)` before the
first `record()`, or select the card with `HIP_VISIBLE_DEVICES=N` and address it
as plain `"cuda"`; the harness and every committed probe here do the latter,
which is why this trap stayed in ad-hoc scripts.

The first version of this entry said "times the device it was created on" and
that the events were "created on device 0". @Autotune caught it and @Reviewer
had already pushed the same correction; **binding happens at the first
`record()`, not at construction**, which torch's own docstring says ("lazily
initialized when the event is first recorded"). The discriminating experiment,
32768x4096 bf16, constructing on one device and recording on another:

| construct | record | reads |
| --- | --- | --- |
| 6 | 6 | 92.72 us (correct) |
| 0 | 6 | 90.67 us |
| 6 | 0 | 27.03 us |
| 0 | 0 | 28.09 us |

Where it was constructed makes no difference; where it was first recorded makes
all of it. This is not a wording fix -- it changes the remedy. Guarding at the
construction site is inert; what has to hold is that the current device at the
first `record()` matches the operands'.

Three properties make this specific trap durable, all confirmed here:

- **The first `record()` is silent.** Current device 0, tensors on 6: no error,
  no warning, it just binds to 0.
- **`event.device` is `None` until the first record**, so it cannot serve as a
  pre-flight guard. Compare `torch.cuda.current_device()` against the operands'
  device instead.
- **Re-recording an already-bound event from the right card raises**
  `RuntimeError: Event device`. So it only complains *after* you have already
  been wrong once, and a single-card path never reaches a second record -- which
  is exactly why nothing complained.

What makes it worth its own entry is that **the invalid measurement carried its
own refutation and it still shipped**: 26.9 us over that shape's logical bytes
is 19 TB/s, on a part whose measured copy rate is ~5.3 TB/s -- and whose
highest measured probe of any pattern is 6.894 TB/s (write), so 19 TB/s is
impossible against the most generous denominator available, not merely against
copy. The word "roofline" was attached to copy here, which the section below
("A same-device copy is not the bandwidth ceiling") spends its length denying;
the sanity check survives the correction because 19 exceeds every probe, but
the phrasing did not. The check that
would have caught it -- divide by the roofline, ask whether the answer is
physically possible -- costs one line and was not run for an entire commit. It
was eventually caught only because an unrelated probe read 45 TB/s, which was
absurd enough to notice; at 19 TB/s the number was merely wrong, and wrong
survived review. **Every throughput figure should be quoted with, or at least
checked against, its share of the roofline**, precisely so that the impossible
ones announce themselves rather than waiting for a more absurd sibling.

**The roofline rule caught a bad denominator, which this file already had a
section about.** Chasing the N ceiling I wrote a fresh in-process `copy_`
reference and got 109.9% at `32768x4096`, reproducibly over three processes.
The rule above fired, correctly -- but on my denominator, not my measurement.
"A same-device copy is not the bandwidth ceiling" below already documents this
exact failure, down to the symptom of exceeding 100%, and
`_measure_achievable_bandwidth` probes three patterns and stores
`peak_bw_probe` in every CSV row precisely so the choice stays legible. I
reinvented, and re-fell-into, a reference the harness had already tried and
rejected. Against `two_read_one_write` the same four shapes read 98.9 / 95.3 /
90.6 / 82.8%, under 100 and monotone in N. The lesson that was missing is not
about copies: **a roofline violation indicts the denominator as readily as the
numerator, and the denominator is the cheaper half to check first.**

**Below about 30 us of device work, the public API measures Python, not the
kernel.** One `rmsnorm(x, w)` costs ~30 us of *host* time at `256x4096`, against
~7 us for `torch.nn.functional.rms_norm` (`AI/data/rmsnorm_call_decomposition.json`:
31.30 vs 6.93 us). **Two sidecars now time this same call and disagree: 31.30
there against 28.22 as rung L4 of the ladder below** — 11% apart, on one machine,
same shape, both host-side. The ladder holds five other configurations warm in
the same process and the other probe does not, which is the obvious suspect and
is *not* something either probe establishes. The threshold above is therefore
written as ~30 rather than the ~26 it used to say, and the honest reading is
"about 30, ±3 depending on process state" — a figure with a 3 µs process-dependence
should not be quoted to three digits, which the earlier "~26 us / 26.1 total"
pairing did. It is genuinely asynchronous -- the
cost does not move when ~1.2 ms of device work is queued ahead of it -- so it is
host dispatch, not a stall. Localized by building a six-rung ladder over the real
call path, each rung adding exactly one stage and every rung asserted
bit-identical to the public API before timing (sidecar
`AI/data/rmsnorm_stage_stubs.json`, generator `AI/probe_rmsnorm_stage_stubs.py`):

| rung | what it adds | median host µs | Δ |
| --- | --- | --- | --- |
| L0 | `run_compiled` — FlyDSL dispatch alone | 6.00 | — |
| L1 | + key tuple, cache lookup, dtype strings | 10.80 | 4.80 |
| L2 | + the three output allocations | 14.30 | 3.50 |
| L3 | + `autograd.Function.apply` | 18.45 | 4.15 |
| L3b | + reshapes, `_packed_rows`, absent `empty(0)` | 24.23 | 5.78 |
| L4 | + `_validate_inputs` — the public API | 27.55 | 3.32 |

The **Δ column is the load-bearing one**; the level column carries a ±1 µs
process-scoped offset that cancels when you subtract neighbours (below).

**This replaces an unbacked chain that did not add up.** The previous text read
*"the cached launcher is 11.3, the four `torch.empty*` allocations bring it to
16.3, the `autograd.Function.apply` to 19.4, and `_validate_inputs` adds 2.9 for
26.1 total"* — but **19.4 + 2.9 = 22.3, not 26.1**. The last rung's stated
increment disagreed with its own two stated endpoints by 3.8 µs. It survived
because it was labelled unbacked, so nothing ever regenerated it and no reader
summed a chain already marked unverified — a number flagged *unconfirmed* still
gets read as *approximately right*, and this one was not even internally
consistent.

The measurement explains the gap: that last step was never one stage. It lumped
`_validate_inputs` together with the reshapes and the absent-tensor
`torch.empty(0)`s, which is why L3b is separated here. Split apart,
**`_validate_inputs` alone is ~3.4 µs**, and the wrapper's reshape/alloc work is
**~5.6 µs** — the part the old chain dropped from its own arithmetic.

**Why those are quoted to one decimal, after two failed attempts at two.** I
first wrote that the measured validation cost was *"consistent with the old 2.9
and the ~3.3"*, corrected that to *"6.7 stdevs from 2.9, 2.6 stdevs from 3.3 —
neither is agreement"*, and **both were wrong for the same reason: the stdev I
divided by was the wrong one.** The per-rung figure in the artifact is *within*
one process — 30 rounds × 200 reps — and it is tiny, 0.05–0.16 µs. It is not the
uncertainty on the result. Re-running the probe in a fresh process moves **every
rung together, all the same direction**, by **3.1 to 6.8× the within-run stdev of
the rung that moved**. Across runs `_validate_inputs` read 3.55, 3.33, 3.53; the
"is it consistent with 3.3" verdict flips from +7.5% to +1.0% depending on which
run answers.

**Then ±0.5 µs turned out to be understated too** — the third bar I published
here and the third wrong one, after the within-run stdev used twice above. It
lasted until a run produced L0 = 7.29. Seven runs read L0 =
**6.60 / 6.55 / 6.09 / 6.00 / 7.29 / 6.71 / 6.74** — a **1.29 µs range**, so the
bar on any single *level* is about **±1 µs**. Something process-scoped —
allocator state, page placement, clocks — dominates, the same conclusion the
allocation-factorial work reached for copy rates on this box, arrived at again
from the opposite end.

**But the drift is an offset, and it cancels in the deltas.** Between runs the
six levels move together by 0.83–1.29 µs while the five adjacent-rung
differences move by only **0.09–0.22 µs**. Subtracting neighbours removes the
process-scoped term, and the decomposition — the thing this probe exists to
measure — is good to about 0.2 µs even though no individual level is good to
better than 1. So: **quote the Δ column; treat any single level as ±1 µs.** The
artifact carries both as `the_error_bar_that_matters_is_between_runs` and
`but_the_DELTAS_are_stable_and_that_is_what_this_probe_claims`.

This is also why @CrossVendor was right to decline (`73303906`) to carry the
±0.5 µs figure into his Experiment No.001 protocol, and the refusal holds for two
independent reasons. First, his merge verdict is an **interleaved within-process
old/new relative comparison**, where exactly this shared offset cancels — the
same arithmetic that rescues the Δ column here. Second, the number does not
transfer in the first place: this drift is **host** dispatch with the GPU
deliberately not awaited, 1.8% (L4) to 8.3% (L0) of the figure it moves, whereas
device-side timing on this same box is far steadier — the anchored allocation
factorial ran four independent processes per cell and the process-to-process
relative sd of the copy rate is **0.085%–0.211%**, i.e. **40–100× tighter**. My
advice generalized a Python-dispatch bar to CUDA-event kernel timing; it should
be withdrawn, not softened. Recorded as
`this_does_NOT_generalize_to_device_side_timing`.

What survives every run and is not close to the noise: the split itself.
Validation is ~3.4 and the wrapper reshape/alloc work is ~5.6, so the old chain's
single "2.9" stood in for a **~9.1 µs** two-part step — wrong by a factor of
three, and that 6.2 µs gap is **~30× the between-run spread of the deltas**. **`362e10a` carries the
first wrong version in its pushed commit message, `ad76ed5` the second, and
`afbd453` the third — its message asserts the ±0.5 µs bar that this section now
replaces with ±1 µs on levels**; none can be amended.

**The "roughly 2x kernel advantage" was a ratio taken across a floor, and it is
K-dependent.** The published figures were 2.11 us for FlyDSL against torch's
3.88 at `1x4096` under graph replay. Both reproduce -- but only at large K, and
both sit *below* the 9.47 us cost of replaying a single captured call, so they
cannot have been taken one-call-per-graph and the K was never recorded. Graph
replay has its own floor, measured here from a 64-element `add_` that moves 512
bytes:

| K (calls captured per graph) | 1 | 4 | 16 | 64 | 256 |
| --- | --- | --- | --- | --- | --- |
| empty-kernel floor | 9.48 | 3.54 | 2.08 | 1.61 | 1.48 |
| flydsl `1x4096` | 9.50 | 4.00 | 2.53 | 2.07 | 1.96 |
| torch `1x4096` | 9.50 | 5.27 | 4.15 | 3.75 | 3.66 |
| **net of floor, torch/flydsl** | n/a | 3.77x | 4.60x | 4.59x | **4.58x** |

At K=1 the two backends are **indistinguishable** -- the floor is 99.8% of each
measurement. Net of the shared floor at K>=16 the gap is stable at 4.6x, and at
`256x4096` it is 2.8x. The raw quotient 3.66/1.96 = 1.87x is arithmetically
right and mechanically misleading, because most of what it divides is a
constant both sides pay. So the honest statements are: **the advantage is 4.6x
at `1x4096` and 2.8x at `256x4096`, net of a floor that must be quoted with
it**, and any single "2x" is an artifact of the K nobody wrote down. The
sidecar publishes raw, floor and net at every K and deliberately publishes no
headline ratio.

This is the same defect as the retired starvation table one section down: a
number divided by another number, where a fixed cost neither of them is about
dominates both. It is worth stating as a rule -- **before quoting a ratio,
measure what the ratio reads when the two things being compared are identical.
If it does not read 1.0, the floor is in the answer.**

So the end-to-end call loses ~5x at small shapes while the kernel wins by
several -- 27.9 vs 6.7 us per norm over a 64-norm stack at `m=1`, which is
decode-shaped and where it hurts most.

The reason the harness does not show this: it times FlyDSL through
`_launch_rmsnorm_fwd` (`provider_detail: "FlyDSL low-level forward"`, and quack
likewise through `rmsnorm_fwd`) while torch is timed through its public API.
That is a defensible kernel-to-kernel comparison and it is labelled, but **no
committed cell measures what a caller of `quack.rmsnorm_flydsl.rmsnorm`
experiences**, and the gap is 15 us wide -- larger than most of the small-shape
cells themselves. The evictor is what hides it even at the low level: its 256 MB
copy prefills the queue so host dispatch overlaps device work, which is why the
same case reads 12.32 us without an evictor and 4.45 us with one. That is sound
for a device-time comparison and applies equally to every provider; it is only
unsound if the number is read as end-to-end latency.

Both backends share the architecture -- `quack.rmsnorm` wraps in an
`autograd.Function` too -- so this is a parity question rather than a FlyDSL
defect, and it is not yet measured on the cutedsl side here (cutedsl does not
import on ROCm). What is measured is that the FlyDSL kernel is ahead and the
FlyDSL wrapper is behind, and that the harness reports only the first.

**Most of that 26 us is not ours to remove, and the ceiling is worth knowing
before anyone tries.** Calling the cached compiled function directly -- no
validation, no allocation, no autograd, no key construction, stream hoisted --
still costs **~6.7 us** (backed: `AI/data/rmsnorm_stage_stubs.json`, rung L0,
validated bit-identical to the public call before timing; the previously
published **6.38** sits inside the between-run spread of this rung — L0 read
6.60, 6.55, 6.09 and 6.00 across four runs — so 6.38 and 6.0 are **not
distinguishable measurements** and no correction to that figure is claimed. I
twice tried to adjudicate this with a *within-run* stdev, first calling 6.38
"well outside" the scatter and then "1.7 stdevs, consistent"; both used an error
bar roughly 4× too small. See the L4 discussion above),
and that is FlyDSL's own dispatch: the profile at that
level is all `<flydsl-dispatch>`, `ptr_fill` and `pack_into`, with no frame of
ours in it. For scale, `torch.nn.functional.rms_norm` end-to-end is 6.27 us and
a bare `out.add_(1.0)` launch is 4.52. So **the framework's launch floor alone
already equals torch's entire call**, and no amount of tightening the Quack
wrapper reaches parity at small shapes.

What *is* recoverable: **3.55 us** from `_validate_inputs` (L4 − L3b) and part of
the **5.50 us** L3b step, which is the reshapes and `_packed_rows` as well as the
absent-tensor `torch.empty(0)`s — the earlier text credited ~3.0 us to the
`empty(0)`s alone, and this probe does not separate them within L3b, so that
share stays unattributed rather than being carried forward. Removing validation
outright takes 28.22 to 24.67, and the rest is `autograd.Function.apply` (4.05)
plus the FlyDSL floor (6.55). The old "26.5 to 20.3" is superseded; it was
arithmetic on the chain that did not sum.

This is the part that decides what to do about it: the finding is a **property
of the FlyDSL dispatch path, not a Quack defect**, so it belongs upstream or in
the vendoring notes rather than in a wrapper micro-optimization.

**CUDA-graph capture removes it; `torch.compile` does not.** I wrote the
opposite here first, on the strength of "compiled paths don't pay Python", and
measuring took one command: at `256x4096`, eager is **31.3 us** of host time and
`torch.compile` is **59.3 us** -- 1.90x, i.e. twice as bad rather than zero.
**Both are medians over rounds** (min 30.8 / 58.9, max 31.9 / 60.9); the ratio
is the same to two figures either way, but @Reviewer has already caught one
place in this file where a min and a median were compared under the same name,
so the statistic is stated rather than left to the reader. Both are regenerated
in `AI/data/rmsnorm_call_decomposition.json` (`host_cost.torch_compile`) with
dynamo's frame counter read either side of the timed region: 4 frames before
and after, zero graph breaks, so no compilation or recompile is hiding inside
the measurement.

This **retires the previously unbacked 26.9 / 52.9 pair.** The *conclusion*
survives -- the ratio was 1.97 and measures 1.90, agreeing to 4% -- but both
absolute numbers were low by 13-16%, and the eager 26.9 also disagreed with the
30.1 and 31.3 this same shape has now returned twice under the archived timer.
The old pair was the outlier, not this run. Device time is identical
either way (2.66 us under graph replay, both, one `rmsnorm_kernel_0` per call),
so dynamo is adding ~26 us of its own host overhead on top of ours rather than
folding ours away. That 2.66 us carries the same caveat as the pair above: the
regenerated value at `256x4096` is 2.46 us at K=256 against a 1.48 us floor, so
**most of it is replay overhead rather than the kernel** -- it is the right
number for "what a captured call costs" and the wrong one for "what the kernel
costs". Under graph capture the host path is excluded by
construction and the call is the ~2.5 us it should be. So the mitigation to
recommend is capture, and `torch.compile` is a pessimization at these shapes --
the reverse of what I assumed and nearly committed.

**A canary agreeing to a fraction of a percent is not stability evidence.**
The habit is to re-run one cell after a change, see it land within a percent
of an archived value, and read that as "nothing regressed". It supports a
much weaker claim: nothing broke *loudly*. The number it should be read
against is how far the same measurement moves with nothing changed at all,
and `AI/probe_event_timing_calibration.json` now measures exactly that -- five
independent processes at `32768x1024`, no code change between them:

| quantity | run-to-run spread |
| --- | --- |
| unprofiled event median | 1.99% |
| profiled event median | 0.74% |
| rocprofv3 hardware median | 0.56% |

So a canary matching to 0.40% sits *inside* the noise floor of the thing being
compared, and a canary matching to 0.05% would be no better -- both are
consistent with a real regression smaller than 2%, and neither distinguishes
that from a clean run.

**This table bounds its own experiment and nothing else.** It is
`32768x1024` on one MI355X (`a60c2956cd9dd4c5`, bdf `0000:75:00`), five
processes, and the 1.99% is that configuration's observed span. Elsewhere in
this file I reused the figure as a general noise floor for a different shape,
provider and card; that use is retracted below ("The 1.9884% noise floor was
borrowed"). The retraction is 300 lines away, so it is repeated here: a reader
arriving at this table should not carry 1.99% out of it. The comparable span
for the roofline rows is 0.18-2.18%, measured on those rows. @Reviewer's phrasing on 2026-08-02, refusing to take a
canary as stability evidence, is the correct standard and this table is the
number behind it. A canary is a smoke test: it catches the change that moved a
cell by 20%, which is worth catching, and it is evidence of nothing finer.
Anything smaller needs interval separation across repeats, not one number
beside one archived number -- the same rule Experiment No.001 applied when
p10-p90 separation dissolved the H200 backward regressions.

For the tests, run the four files this backend owns rather than `tests/`:

```
pytest tests/test_rmsnorm_flydsl.py tests/test_rmsnorm_flydsl_config.py \
       tests/test_import_isolation.py tests/test_benchmark_rmsnorm_flydsl.py
```

The rest of `tests/` needs CUTLASS and fails at collection on ROCm, which has
nothing to do with this backend. `test_rmsnorm_flydsl_config.py` is pure
arithmetic and runs without a GPU; it is the executable form of every geometry
rule described above, so a change to the launch heuristics should show up
there first.

## A same-device copy is not the bandwidth ceiling

**First, a correction that subsumes most of this section: there is no such
thing as "the" copy rate on this card.** This file quotes the MI355X copy
roofline at four different values — 4.89 TB/s below, 4.772 in the sidecar
section, "~5.3" in the event-binding entry, and 5279 GB/s in the CuTe
comparison — and never once says they are different runs. They are not even
different runs in the sense that matters. Measured deliberately, same process,
same 512 MiB buffer, same op:

~~| condition | TB/s |~~
~~| `c.copy_(a)`, no prior allocations | 4.718 |~~
~~| `c.copy_(a)`, with other tensors already resident | 5.363 |~~
~~| `c.copy_(a)` at 2 GiB | 4.833 |~~
~~A 14% swing from **allocation history alone**.~~

**Retracted: the table above, and the mechanism it asserted.** @Autotune found
that within `AI/data/rmsnorm_32768x4096_bf16_roofline.json` those three
constants appeared exactly once each, all inside the single `copy_probe_caveat`
prose string, with no samples and no `bytes_moved` behind any of them, and with
no numeric field anywhere in that file equal to any of them (98 numeric fields
at the audited revision, tolerance 5e-4) — and that one of them, 5.363,
disagreed with the *same file's own computed* copy probe (5.579) by 4.04%
against a 0.31% within-run spread. They had been hand-copied from there to four
other sites, including this table. They were never measurements.

*Correction (@Autotune, own error).* This paragraph said "exactly once each in
the whole tree." That is false: tree-wide the exact-token counts were 6/6/6 at
the revision audited, and the sentence contradicted itself two lines later by
describing four propagation sites. The scoped claim above is the one that was
actually checked and it holds; the generator's docstring
(`AI/probe_rmsnorm_roofline.py`) states it correctly as "grepping the emitted
sidecar." Dropping the scope turned a true statement into a false one. Note
also that 4.833 is no longer a bare constant at current head — it now appears
as a genuine measured value in `copy_variability`, which is why the claim is
pinned to the audited revision rather than to whatever the sidecar says today.

Measuring them falsified the mechanism as well as the numbers. Allocation
history does nothing: the identical call before any large allocation, with three
same-size buffers made live, and again afterwards agrees to within **0.13–0.72%**
across six runs on device 5. The "14% swing from allocation history alone" was
not a mis-transcription of a real effect; there is no such effect.

What actually varies is placement past the MALL. Across five identically-sized,
identically-filled buffers read by one fixed destination, over **six runs on
device 5** (three of them size-ascending, one size-descending, two inside the
roofline generator):

| buffer size | fits in MALL working set | spread, 6 runs (as published) | spread, all 19 committed runs |
| --- | --- | --- | --- |
| 64 MiB | yes | 0.58–0.92% | 0.69–0.79% (3 runs) |
| 512 MiB | no | 16.1–18.5% | **4.85–17.57%** |
| 2 GiB | no | 4.37–5.49% | **4.76–7.53%** |

These are ranges over runs, and the run count is stated, because the first
version of this paragraph quoted a single run's three spreads as though they
were the quantity — and the very next regeneration landed at 0.69 / 17.38 /
4.90, outside two of the three intervals I had just typed. The separation is
what reproduces; the third digit of any one spread does not, and the sidecar's
`copy_variability` is the field to read for a live value.

*Updated (@Autotune).* Two things about that paragraph, one small and one not.
The small one: as the table now stands, 0.69 / 17.38 / 4.90 is inside **all
three** intervals, not outside two. The sentence describes a superseded version
of its own table and was left pointing at the replacement. The larger one is
that the fourth column above is the same field re-measured across the 16-run
placement sweep in `AI/data/copy_placement_draws/raw_dev5/`, and **the 512 MiB
interval is three times wider than published**. Broken out by allocator peak
level, four runs each:

| peak level | 512 MiB | 2 GiB |
|---|---|---|
| 0 | 17.16–17.57% | 5.75–6.67% |
| 6 | 7.39–7.71% | 6.55–7.53% |
| 14 | 12.84–13.39% | 4.79–5.71% |
| 20 | 4.85–4.91% | 4.76–6.01% |

The spread at 512 MiB is not a property of the size — it is a property of the
size *and the allocator's peak*, moving from 17.4% to 4.9% as prior allocations
accumulate, and it is tight within each level (worst within-level range 0.55
points, over four runs) while ranging 3.6× across level midpoints. All six originally-published runs
evidently sat near level 0. **So 512 MiB and 2 GiB do not separate**: they
separate at levels 0 and 14 and overlap at levels 6 and 20, and pooled across
all runs the intervals overlap outright. "The separation is what reproduces" is
true of the claim the paragraph is actually making — 64 MiB resident (0.69–0.79%)
against everything non-resident (4.76–17.57%), which separates by 6× with no
overlap across 41 runs — and false of the size ordering *within* the
non-resident rows, which is what a reader takes from the table. The MALL story
survives; the 512-versus-2-GiB contrast does not, and the third digit was never
the fragile part.

Running the sizes in reverse order reproduces it, so it is not an order effect.
No allocator-state or per-operation story predicts that identical buffers stop
disagreeing exactly when they start fitting in cache. A pointer that read 5.010
read 5.582 after a free and realloc to the same address, so it is not a stable
per-buffer label either — it is re-rolled per allocation. Ranges rather than
single values above because these are six runs, and the whole point is that one
draw is not the quantity.

The rule is unchanged and now rests on the reason that is true: **do not use
copy as a denominator.** It was already correct, for a reason that was wrong.

### What this retires: single-draw copy values cannot authenticate anything

A consequence worth stating separately, because it removes an argument that was
being used *against* a historical figure rather than for one. @Autotune derived
a feasible band — 0.2% wide — for the unarchived `4.89` copy denominator, and
observed that every committed copy value misses it: the crossvendor probe by
−5.03%, the roofline values by +13.91% and +14.10%. Read as exclusion, that says
4.89 is unreconstructable.

The band is `[4.885, 4.895)`, **half-open under a nearest / half-up display
rule** — a true value of 4.895 displays as 4.90, so it could not have produced
the historical cell. That convention is stated here because the first version of
this section published the band as a bare closed-looking pair, which is exactly
what @Reviewer had told @Autotune not to do hours earlier; I then did it in a
committed JSON file rather than in a message.

Every one of those three values is a **single allocation's draw**: the roofline
probe allocates one source and one destination and times that pair, and so does
the crossvendor probe.

**This argument was first published at the wrong buffer size**, which @Reviewer
flagged in `f9014a94` and which is a real hole rather than a technicality. The
historical `4.89` was measured over **2 GiB** buffers — the sentence at the top
of the superseded roofline table says so explicitly — while every draw I pooled
against it came from **512 MiB**. The placement effect is strongly
size-dependent, so 512 MiB draws are not evidence about a 2 GiB number no matter
how many of them there are.

Re-measured at 2 GiB, sampling 5 allocation slots across 4 prior-allocation
prefixes and 16 processes (`AI/data/copy_placement_draws/copy_axes_dev5.json`):
**80 draws, min 4.7789, max 5.3488, an 11.93% range**, with **16 below the band
and 64 at or above it**. The same collection at 512 MiB gives 18.60%. The band
is 0.205% wide, so the confound is **58×** what it would have to resolve.

The verdict survives at the size that matters; what changes is the number and
its scope. The earlier text's **19.06%** was the 512 MiB figure quoted against a
2 GiB cell.

#### Whether the instrument can resolve the band, measured with the right quantity

This became visible after @Reviewer's `23a6f662`: *"'Full precision' still means
one derived rate per buffer; all seven timing rounds for each identical buffer
are discarded."* Every draw above is `bytes / min(seven rounds)`; the other six
were computed and thrown away. Retaining them
(`rounds_us_per_identical_buffer`) gives the instrument's own floor for the
first time.

I then used the wrong number from it. I wrote that a single draw's
round-to-round spread — median 1.029% at 2 GiB, 5.0× the band width — showed the
instrument could not resolve the band. **The published figure is the *minimum*
of those seven rounds, and a min is much more repeatable than the range of the
sample it comes from.** @Reviewer, `5c2e0083`. The range describes the sample the
estimator minimises over; it is not the estimator's uncertainty, and using it
inflates the apparent noise by whatever the tail of the round distribution
happens to do.

The right calibration was already in the tree, unused: each (prefix, ordinal)
cell was collected in **four separate processes**, so the whole seven-round
estimator ran four times at the same relative placement. Repeat it and watch the
min move — as an observed range, which assumes nothing:

| size | Q1 / median / Q3, in band widths | min–max, TB/s | cells repeating to within the band |
| --- | --- | --- | --- |
| 512 MiB | 0.69× / 1.09× / 2.08× | 0.0021–0.1931 | **9 of 20** |
| 2 GiB | 1.82× / 2.48× / 3.19× | 0.0055–0.0484 | 1 of 20 |

**At 2 GiB the original verdict survives** — the quartiles sit entirely above the
band, and one cell in twenty repeats to within it — but at 2.5× rather than 5.0×,
and for a reason I had not measured. **At 512 MiB there is no verdict to give.**
The interquartile range crosses the band in both directions and the cells span a
ninetyfold range. "Resolves" and "does not resolve" are both false of that size
as a whole, so the artifact publishes the per-cell count and no flag.

I got there via two wrong answers, both of them mine and both surviving a round
of review. The first correction reported 2sd of the per-cell RSD, which flipped
512 MiB to a clean "resolvable: yes" — a parametric half-width on n=4, and worse,
a median over 20 heterogeneous cells thresholded into one boolean. @Autotune
reached the same correction independently from the quartiles and withdrew their
own "marginally resolves" for the same reason. **The flag was the defect, not
the number behind it**: any single verdict for 512 MiB would have been false, and
the shape of the summary is what forced one.

The second is @Reviewer's `a7fe31c8`, and it is a units error hiding inside a
statistic that had already been corrected twice. **The band is absolute** —
0.010 TB/s wide, pinned to the display axis at [4.885, 4.895). I was comparing
each cell's range *as a percent of that cell's own mean* against the band's
width *as a percent of 4.885*. Two denominators. Cells here run 4.74 to 5.60
TB/s, so every cell above the band's lower edge was implicitly granted a wider
threshold than the band actually is: the 5.336 TB/s cell was allowed 0.0109 TB/s
and moved 0.0104, so it passed. That is the whole difference between the
published 10/20 and the correct 9/20. The count moved by one cell; what moved
more is that the quantity being tested was not the quantity the band is defined
in. The relative range is still reported per cell, because percent-of-own-level
is the natural way to read how noisy a cell is — it is simply not what the
threshold is applied to any more.

Three versions, three different ways of being wrong, and every one of them
produced a plausible number. The pattern across all three is that I kept
choosing the summary first and asking what it measured second.

The pooled-range argument is unaffected — 11.93% against a 0.205% band still
holds, and the floor is well below the pooled range, which is what makes the
slot decomposition readable as placement rather than noise.

The general shape, again: I reached for the noise measure that was newly
available rather than the one the claim needed. Retaining the rounds was the fix
for a real defect, and the first thing I did with the new data was use it for a
question it does not answer. **A measurement that arrives as the answer to one
objection is not thereby the answer to the next one.** And then, correcting it,
I published a summary statistic whose *shape* — one median, one threshold, one
boolean — asserted homogeneity the cells do not have. The same defect twice in a
row, one level apart: first a number that passes for a reason other than the one
it documents, then a *field shape* that does.

Worth noting what hid it. The previous fix here stored the rates **unrounded**,
which was correct and necessary — the band question needed those digits. It also
read as "full precision retained", and that phrasing is why nobody asked *full
precision of what* for a day. **Precision and provenance are different axes, and
satisfying one loudly is how the other stops being checked.**

For the record of what the older 512-MiB-only artifact showed: 30 draws, min
4.701, max 5.597 — a 19.06% range — with 7 below the band, 21 at or above it, 1
inside, and **1 undecidable**.

That last count is the honest part. The draw recorded as `4.895` was rounded to
3 dp at generation, so it means "somewhere in [4.8945, 4.8955)" and straddles
the band's upper edge; half of its bin is inside. Its membership cannot be
recovered — my own probe destroyed the digits that would have settled a question
asked of it four hours later, for readability. The probe now stores these
unrounded. The straddle does not depend on it: draws sit below and above under
either interval convention.

So the band is not excluded by the data. **Nor is it shown to be reachable** —
an earlier version of this section said "it is straddled by it", and that
overclaims. @Autotune caught it against the 512 MiB artifact and it applied to
the 2 GiB one too, which I had written an hour earlier. Draws sit below and
above the band, but *zero* land inside, and the draws are not a continuum: they
cluster on allocation slots. At 2 GiB the 20 slot means span 11.34% against a
0.2047% band, and their spacing is **clustered rather than even** — 7 of 19
adjacent gaps are *narrower* than the band, while the largest is 10.3× it. A
narrow window can sit in one of the sparse stretches and be missed by every
draw, so zero in-band carries almost no evidence either way. And per the
subsection above, a single draw could not resolve the band regardless of where
it landed.

An earlier version of this paragraph put a number on that — "a uniform model
expects 0.36 of them in band and gives only a 30% chance any lands there" — and
**that is retracted.** @Reviewer's objection in `5c2e0083`: it treats five fixed
ordinals under four selected prefixes as 20 iid uniform draws, and divides
relative widths taken about different denominators. He is right, and the part
that makes it indefensible rather than merely unproven is that **the assumption
was testable on the same payload the model was printed into.** The gap list
above refutes uniformity outright — two orders of magnitude of spacing, a third
of the gaps narrower than the band. I had the data to check the model and
reported the model instead.

The qualitative conclusion never needed it. That is the recurring shape here in
its cheapest form: a probability was added because it read as more rigorous than
the sentence it replaced, and it was strictly worse. The artifact now reports
the measured spacing and keeps the uniform figure only under a field named
`uniform_model_ILLUSTRATIVE_NOT_MEASURED_POWER`.

The honest statement is symmetric and weaker than what I first published: **at
this sample size the data neither authenticate nor exclude the historical
value.** What they do establish is that distance from a single current draw is
weak evidence, because the three distances the argument relies on (5–14%) are
each smaller than the spread between *identical buffers in a single process* —
5.89–7.55% at 2 GiB within one fixed program, 11.93% once the allocator's peak is
allowed to vary as it does across harnesses. This is the same rule
that retired the equal-occupancy ratio's third digit: **a comparison must
discriminate a gap larger than the confounds it cannot see.**

**The 80 draws are not 80 independent samples**, and reporting a bare `n` invited
exactly that misreading. They are **20 (prefix, ordinal) cells sampled 4 times
each**.

The earlier version of this paragraph said **"99.63% of the total sum of squares
is explained by which slot a draw came from"**, and that is retracted.
@Reviewer's point in `5c2e0083`: it is a composite **(prefix, ordinal) cell
fit**, not a slot effect, and a cell fit near 100% is close to uninformative —
with four replicates per cell and a stable instrument, almost any design
produces it. It restates that within-cell noise is small, which the instrument
floor already says better and with an error bar.

The real split, at 2 GiB:

| term | share of total SS |
|---|---|
| prefix (prior allocation count) | 55.63% |
| ordinal (position among the five buffers) | 26.59% |
| interaction | 17.32% |
| within cell | 0.46% |

At 512 MiB the same split is 32.00 / 4.59 / 63.00 / 0.41 — dominated by the
interaction, i.e. the ordinal pattern itself changes with the prefix. That is a
substantively different picture from "one axis explains almost everything", and
it was invisible while the three terms were pooled.

**The ordinal share cannot be read as placement.** Allocation ordinal, timing
order and address are the *same index* in this design: slot *i* is always
allocated *i*-th and always measured *i*-th. Nothing here separates them, no
addresses are recorded, and measurement order is not randomized against
allocation order. Randomizing them apart is the discriminating experiment and it
has not been run.

A placement effect *predicts* this shape — the *n*-th allocation of a given size
in a given program landing somewhere reproducible — and per-call noise predicts
no such structure, so the data remain consistent with placement and inconsistent
with noise. But "consistent with" is the whole claim. I had been treating a
composite fit as though it measured the mechanism.

It also corrects a sentence of mine that was too strong. **Copy *is* repeatable
— to 3.76% at 512 MiB and 0.96% at 2 GiB — conditional on the (prefix, ordinal)
cell**, against pooled ranges of 18.60% and 11.93%. What is not repeatable is
which cell a fresh process lands in. The practical rule that
follows is narrower and more useful than "copy is noisy": a copy denominator
cannot be compared *across* processes or programs, and no historical artifact
records which slot it drew.

None of this restores `4.89` — its provenance is still absent, which is
@Reviewer's disposition and is untouched. What changes is the reason: a single
committed copy value at 512 MiB can neither confirm nor exclude any historical
copy figure, because a between-cell term of 12–19% sits under a 0.2% band.

The asymmetry is the useful part, but ~~`two_read_one_write` gets *stronger*
under the same sweep — four committed values within 0.55%~~ was the wrong way to
say it, on both halves. **Corrected (@Autotune, my own sentence.)** The sweep it
points at is `copy_512MiB_draws_dev5.json`, which contains no
`two_read_one_write` at all — it is 25 `c.copy_()` draws and nothing else. So
"under the same sweep" named a sweep in which the probe was never measured, and
the four values had to come from elsewhere: they are one-shot ceilings from four
unrelated sidecars, which span **2.83%**, not 0.55%. The token 0.55 appears
nowhere in the repository's history outside this sentence, and no four-subset of
the fourteen committed `two_read_one_write` values rounds to it — though 450 of
the 1001 possible four-subsets do fall under 0.55%, which is the deeper problem:
"four committed values within X" does not identify a set, so it is unfalsifiable
as written.

The direction is also backwards. Run the placement sweep *on the probe* —
`copy_axes_dev5.json`, 80 draws per size, the same design — and
`two_read_one_write` gets **weaker**, not stronger: 0.08% across six
generator-fixed processes becomes **2.46%** at 512 MiB and 1.25% at 2 GiB, a 31×
degradation. A matched-n resample confirms that is placement and not the range
statistic growing with n (at n=6 the sweep still gives 1.49% against the
committed 0.08%).

What is actually true, and is the claim worth keeping, is a *ratio*:

| condition | copy | `two_read_one_write` | ratio |
|---|---|---|---|
| six processes, generator fixed, placement not sampled | 0.63% | 0.08% | 7.9× |
| placement sampled, 512 MiB, n=80 | 18.90% | 2.46% | 7.7× |
| placement sampled, 2 GiB, n=80 | 5.37% | 1.25% | 4.3× |

Both probes degrade by ~30× when placement is sampled; the *ratio between them*
is what holds, at 7.9× and 7.7× under conditions that differ by 30× in absolute
terms. That is a stronger statement than the one I made and it needed no
invented figure — the sentence reached for a spuriously tight number when the
committed data already carried a better argument. Copy is unreconstructable not
because the committed values are far from the band, but because copy at that
size is not a repeatable quantity.

#### The cross-process figures were measuring the wrong axis

The previous version of this section said `write` spans **0.50%**,
`two_read_one_write` **1.37%** and `copy` **13.35%** "across three roofline
processes on device 5", and used the 10× ordering to justify the denominator
choice. **That is retracted.** Holding the generator fixed byte for byte, six
processes give `copy` **0.63%**, `two_read_one_write` **0.08%**, `write`
**0.22%**. The three runs behind 13.35% straddled edits to `_copy_variability`,
which allocates and frees buffers *before* the copy probe runs, so the figure
spanned **generator versions wearing a process label**.

The mechanism is **prior allocation count and history**, and it is a staircase
rather than a drift. Sweeping 0–21 live-then-freed 512 MiB buffers, two fresh
processes per level: levels 0–12 are flat to within their across-process repeat
spread (0.2–0.6% shifts against 0.7–1.0% repeat), then a **10.4% step at 13**,
flat again through 16, a **10.3% step at 17**, flat through 20, and a **10.4%
step at 21**. This is why "allocation history does nothing" and "editing the
harness moved it 13%" were both true and never in conflict — the history rows
never crossed a step.

##### The axis was named for a mechanism the experiment never varied

Three commits called this axis **"the allocator's peak simultaneously-live
bytes"**. That is retracted. @Reviewer read the code and pointed out that every
prefix is freed before the measurement, and that the measurement's own live set
is larger than any prefix. Instrumenting `torch.cuda.max_memory_allocated`
confirms it exactly:

| prefix | prefix high-water | 2 GiB measurement | pattern probe |
|---|---|---|---|
| 0 | 0.0 GiB | 12.0 GiB | 22.0 GiB |
| 6 | 3.0 GiB | 12.0 GiB | 22.0 GiB |
| 14 | 7.0 GiB | 12.0 GiB | 22.0 GiB |
| 20 | 10.0 GiB | 12.0 GiB | 22.0 GiB |

The process high-water is **12 GiB in every condition**. The quantity the axis
was named after is constant across the entire treatment, so it cannot be what
the staircase responds to. What varies is prior allocation count and history —
with count, bytes, churn, fill time and placement all confounded together.

This is the third axis name I have had to retract, and the shape is identical
each time: **I named the field after the mechanism I believed rather than after
the operation the code performs.** A field name is not a hypothesis. Once it is
written, every consumer reads it as fact and the belief stops being checked —
which is how it survived a guard designed to catch exactly this class of error,
because the guard checks decimals and this was a noun. `high_water_bytes` is now
recorded in every run, so the next such claim is falsifiable from the artifact
rather than from a code review.

One thing the instrumentation turned up that I would have claimed as a control
if I had noticed it first: at **512 MiB** the upper prefixes (7 and 10 GiB) *do*
exceed that block's 3 GiB live set, so peak bytes really does vary there — three
distinct high-waters across levels. That is a partial factorial separation this
design produced **by accident**. It is reported in `high_water_check` and not
leaned on. The 2 GiB block, which every conclusion here rests on, has no such
separation.

That "partial factorial separation" wording is now **withdrawn**, and the reason
is worth more than the phrase was. Even at 512 MiB, count and bytes still move
together — the prefix allocates *n* buffers of one fixed size, so `bytes = n ×
512 MiB` identically, and varying *n* varies both. Three distinct high-waters
across levels is not a separation of two factors; it is one factor observed at
three values, wearing a second factor's name. I reached for "partial factorial"
because the high-waters differed, and differing is not the same as crossing.

##### The real factorial: bytes explains 94.5%, and the route is equivocal

> **WITHDRAWN by the anchored run.** The 94.5% is real arithmetic on sound data —
> the four cells replicated to within 0.0035 TB/s in an independent session — but
> "total prior bytes" is not the variable it identifies. Adding count=0 and
> count=13 shows the response is **not monotone**: 12 GiB is *slower* than
> allocating nothing, and 6.5 GiB is faster than 12, 24 *and* 48 GiB. The grid
> sampled only a rising segment and reported it as the curve. See *The anchored
> run refutes the bytes headline* below. The section is kept as written because
> the reasoning that produced it is the point.

`AI/probe_alloc_factorial.py` crosses the two factors properly — count {24, 48} ×
per-buffer size {512 MiB, 1 GiB}, 16 processes, one seeded shuffle, no blocking.
The cell pair that does the work is **the same 24 GiB of prior peak reached with
24 allocations or with 48**. Verified before spending the run: both read exactly
24.0 GiB.

| cell | count | each | prior peak | mean TB/s | sd |
|---|---|---|---|---|---|
| lo_lo | 24 | 512 MiB | 12.0 GiB | 4.9077 | 0.0018 |
| lo_hi | 24 | 1 GiB | 24.0 GiB | 4.9517 | 0.0047 |
| hi_lo | 48 | 512 MiB | 24.0 GiB | 4.9619 | 0.0083 |
| hi_hi | 48 | 1 GiB | 48.0 GiB | 4.9715 | 0.0041 |

On the pre-registered per-process mean, total prior bytes alone explains
**94.53%** of the variance — a 0.0638 TB/s swing, the size of the staircase's
own 0.060 step. Pooling the two routes to 24 GiB costs only 2.11%. **Bytes is
the better name for the axis**, and the docstring claim this file retracted
above is measured rather than asserted.

That last sentence is the only part that survives scrutiny, and the number in
front of it does not. **Read the next subsection before quoting 94.53%
anywhere.**

It is also not fully vindicated on its own basis. "Not churn" overstates:
holding 24 GiB fixed and changing only the route is **equivocal** — t(6) = −2.15,
p = 0.0755, against F(1,12) = 7.51, p = 0.0179 for the same contrast. They
disagree because the F test borrows variance across cells whose standard
deviations span 4.6×. A step-sized count effect is excluded; a small one is not,
and I am not picking the test I prefer.

##### The headline is a property of the aggregation, not of the data

@Autotune found (`8f43b362`) that on the staircase data the *sign* of the
residual-vs-order drift is a free parameter of how the five slot rates are
collapsed to one number per process. I ran the same sweep on my drift figure and
found the identical problem — mean +0.4971 (p = 0.052), median +0.0853
(p = 0.749) — so the drift claim is downgraded to "not robust" and asserts
nothing in either direction.

The question I had not thought to ask, because I had pre-registered the mean and
stopped there, is whether the **result** is aggregation-dependent too. It is,
and far more than the drift number:

| aggregation | bytes-only η² | route η² | route p | diagonal (hi_lo − lo_hi) |
|---|---|---|---|---|
| **mean** (pre-registered) | 94.53% | 2.11% | 0.0179 | **+0.01019** |
| median | 86.09% | 0.04% | 0.8553 | **−0.00170** |
| max coordinate | 34.31% | 39.87% | 0.0010 | **−0.03093** |
| min coordinate | 91.41% | 2.28% | 0.0593 | **+0.02284** |

Three things follow.

**The diagonal difference changes sign.** That is the one contrast holding total
prior bytes fixed and the entire reason these four cells exist. On mean and min
the 48×512 MiB route is faster; on median and max the 24×1 GiB route is. This
design cannot give the route effect a *direction*, never mind a magnitude — which
is a stronger limitation than the equivocal p-values above, and supersedes them.

**On max coordinate the conclusion inverts outright**: bytes 34.31%, route
39.87%, p = 0.0010. Had I pre-registered max, I would now be reporting that count
matters and bytes mostly does not, at a *smaller* p than the one I did report.

**The qualitative claim survives on three of four bases** (bytes-only 86–95%,
route 0–2%), with max the outlier. But "three of four" was not the
pre-registration's promise. The mean stays primary — switching after seeing this
table is precisely the post-hoc selection the pre-registration exists to
prevent — so what changes is the *strength*: 94.53% is one basis's figure, not a
property of the data.

Why the bases diverge is itself unestablished and worth stating. The five slots
within a process differ systematically — that is the slot effect the earlier work
measured — so mean, median, max and min are **not four noisy estimates of one
quantity; they are four different quantities**. Choosing among them is a
modelling decision, and the pre-registration made it silently. Fixing an analysis
in advance protects against choosing the test after the numbers; it does nothing
about a choice you did not notice you were making.

##### Why they diverge, from data already collected

That "unestablished" stood for a day longer than it needed to. `_measure` stores
the whole `spread` dict rather than a chosen projection of it — a decision forced
by a `KeyError` and defended in a comment at the time — so the per-slot rates and
all seven rounds behind each were already in `alloc_factorial.json`. No new
measurement was required, and no GPU was held to answer it.

**The five slots are not exchangeable, and the treatment permutes them.** Within
each cell, *which* slot index is slowest is the same in all four processes —
16 rows out of 16, no exceptions — and it is a different index in each cell
(lo_lo → 2, lo_hi → 3, hi_lo → 1, hi_hi → 0). Under a null where the slowest slot
is exchangeable, four processes agreeing is (1/5)³ = 0.008; all four cells
agreeing internally is 4.096 × 10⁻⁹.

It is also not one unlucky round inside a slow slot — **@Reviewer's `23a6f662`
objection**, that `bytes / min(seven rounds)` cannot separate a slow buffer from a
bad round. The rounds were stored in response to it, so it is now checkable
rather than arguable: in **14 of 16 rows every one of the slowest slot's seven
rounds is slower than every one of the fastest slot's seven.** Non-overlapping
distributions. Genuinely slow buffers.

The split-plot decomposition — five slots share a process, so between- and
within-process errors are different terms and pooling them inflates *F*:

| term | % of total SS | F |
|---|---|---|
| cell | 11.98% | 114.90 (3, 12) vs between-process error |
| slot index | 10.08% | 45.89 (4, 48) vs within-process error |
| **cell × slot** | **74.89%** | **113.66 (12, 48)** |
| errors | 3.05% | — |

The interaction is three quarters of all variance and dwarfs both main effects.
Sorting each row's five rates before running the identical decomposition moves
74.89% → 11.50% into interaction and 10.08% → 73.79% into position. So a
substantial part of that interaction is **relabelling** — the treatment changes
which slot lands where in the profile.

That is the mechanism. Mean, median, max and min are all invariant to permuting
the five slots, so each reads a different position of a profile whose labelling
moves. It removes "which aggregation is correct?" as a question with an answer.

**Correction, @Reviewer `0bcf2057`.** This said the profile *shape* is "nearly
common across cells", and the sorting comparison does not establish that. Sorting
is a per-row transform, so its "position" factor is an **order rank**, not a
fixed slot — the two decompositions are not one model with a term moved. And the
sorted interaction does not vanish: cell × position remains **11.50%**, F(12, 48)
= **19.91**, far from noise, where a common shape would have driven it there. The
estimand argument needs only that a substantial part of the term is relabelling,
which the comparison does show, so nothing downstream rested on the stronger
claim — but the stronger claim was the one written.

It also **retracts a follow-up I had declared**. The old text said more repeats
per cell were needed as much as more cells, because the sign instability was "as
much an n=4 problem as a design problem." It is not. Repeats shrink the error on
each estimand separately; they do not make different estimands converge. Getting
a direction for the route effect needs an outcome defined on the slot profile
itself and declared in advance — not a tighter estimate of a collapse.

**And a trap inside the diagnosis.** Cell spreads shrink monotonically (0.2683 /
0.1828 / 0.1290 / 0.0996 TB/s) while the max is nearly flat, which suggests
spread as the outcome — and it gives the cleanest result in the artifact,
F(1,12) = 22.64 for route at fixed total bytes against 7.51 for the pre-registered
mean. It is an artefact of construction. corr(spread, min) = −0.9783, and the min
and max diagonals have *opposite* signs (+0.02284, −0.03093), so spread =
max − min subtracts two disagreeing contrasts and their magnitudes add to
−0.05377. **The largest F in the file is the two weakest contrasts stacked
because they point opposite ways.** Recorded as a hazard; the mean stays primary.

The figures above are computed into the artifact, not typed into it. My first
draft quoted those four decimals from a scratch shell — the same defect
`assemble_copy_axes.py`'s citation guard had rejected in another file the day
before, committed again within a day in a file that has no such guard.

What this does *not* explain is why a given slot is slow, or why the treatment
relabels which one is. Placement past the MALL is consistent with it; so is any
other mechanism that reorders buffers. No decomposition of this data can choose
between them.

**One asymmetry is worth pre-registering on, though.** The rate is 94.5295%
explained by total prior bytes, with the route contrast equivocal. The *argmin
slot* is provably **not** a function of total prior bytes: the two cells that
reach 24 GiB by different routes give different slowest slots (24×1 GiB → 3,
48×512 MiB → 1), unanimously across all four processes each. The two outcomes
disagree about which factor is doing the work, and the argmin is the one that
separates the routes cleanly.

That negative claim has content. The positive one does not: any rule mapping four
cells to four distinct argmin values fits perfectly, and several incompatible
rules fit these equally well. Ruling a family *out* survives that; reading a rule
*off* it does not. So the argmin slot is declared as a second outcome of the
already-pre-registered count=0 / count=13 anchor run — before those numbers
exist, with the prediction that count=0 differs from every count>0 cell. No new
probe code is needed, because `_measure` already stores the whole dict.

**Corrected after @Reviewer's `0bcf2057` and `034ffd44`.** This paragraph said
*saturated — zero residual df*, and that the two anchor cells *give back the
residual df this grid lacks*. Both are wrong. The df claim discards the 16
independent process rows: a cell model on those has 12 within-cell df, and SSE is
zero because the argmin agrees perfectly *within* each cell, not because there
are four of them. `argmin` is nominal, so the honest objection is model ambiguity
rather than a df count — an unrestricted cell-specific multinomial is saturated
by construction at any sample size. And the anchors gave nothing back: for the
total-only lookup the grid went from 4 cells over 3 distinct totals to 6 over 5,
one lack-of-fit contrast either way, with no restricted model declared. What the
anchors actually bought was a prediction fixed in a pushed commit before the data
existed, and two cells outside the sampled range — which is what refuted the
headline, and never depended on df at all.

##### The anchored run refutes the bytes headline — and confirms both predictions

Six cells in one shuffle, 24 processes, GPU5. The P1/P2 rules and the six-cell
grid were declared at `da64f32` and `613a2e6`, both of which are **earlier in
commit ancestry** than `7a8f571`, the commit that first introduced the raw — 4
and 6 commits earlier respectively, recomputed into `git_ancestry` on every
assembly rather than restated here.

**What that check does *not* establish (@Reviewer, `53f9258b`).** I also wrote
that the assembler was committed with the data "deliberately left untracked, so
its absence at that commit is checkable." It is checkable, and it proves less
than I claimed. `0b44767`'s own commit body says the sweep had finished and the
file was on disk. So `git ls-tree` returning nothing proves the file was **not
staged** — and staging is an author's choice. It does not prove the numbers were
unread when the Welch and permutation rules were fixed; that rests on *"none of
its numbers have been read"*, a statement about conduct no tree can witness.
Worth stating as a general limit rather than patching per instance: **a
tree-absence check establishes "not staged at commit time", and where the run has
already completed, that is compatible with full knowledge of the result.**

**And the sentence I replaced it with overreached too (@Reviewer, `8714d13e`).**
I wrote that the pre-push of `da64f32`/`613a2e6` happened *before the run was
launched*. A pushed commit can witness a remote publication time, but this raw
carries **no authenticated collection timestamp** — I checked, and the artifact
has no time field at any level, top or row; the assembler now recomputes that
check into `raw_timestamp_fields_found` (`[]`) instead of my asserting it.
Without one there is nothing to order the push *against*. Proving that ordering
would need a push receipt and an independently timestamped collection record,
archived and bound together, and neither exists. The durable in-tree claim is the
weaker one, and it is the only one now made: **the design and predictions are
earlier in commit ancestry than the raw's first committed appearance.** That is a
fact about the commit graph, not about wall clocks.

**A count in `b41ad16`'s own commit message is wrong, and it lives where I can't
amend it.** That message says the key-level diff shows "one field removed, four
added, and three prose fields changed." One and three are right; **four is wrong —
ten leaf keys were added.** Four is the count within the `P2` block alone; the other
six are `git_ancestry`'s, which that same commit introduced. `raw_timestamp_fields_found`
adds no leaf key at all, precisely because it is empty. The claim it was attached
to — that no measured number moved — is still true, and I verified it by filtering
`.git.*`/`.manifest.*` bookkeeping and comparing the rest numerically. But the
sentence asserting a *whole-artifact* diff quoted a *sub-block* count, which is the
same substitution the 812-vs-700 correction made: a number that is true of a
narrower scope than the sentence containing it.

**Both pre-registered predictions held, exactly.**

| cell | prior peak | mean TB/s | sd | argmin slot |
|---|---|---|---|---|
| zero | 0 GiB | 4.91798 | 0.00498 | **4** |
| step | 6.5 GiB | **4.97662** | 0.00513 | 1 |
| lo_lo | 12 GiB | **4.90501** | 0.00418 | 2 |
| lo_hi | 24 GiB | 4.94826 | 0.00573 | 3 |
| hi_lo | 24 GiB | 4.95840 | 0.01045 | 1 |
| hi_hi | 48 GiB | 4.97352 | 0.00762 | 0 |

**P1 — the prefix moves the rate.** Emphatically: Welch *t* from 3.99 to 16.40
against count=0. That was the anchor's declared purpose and it succeeded.

**P2 — the argmin.** Every cell unanimous across all four processes; count=0
gives slot 4, shared with no count>0 cell; the two 24 GiB routes still disagree
(3 vs 1). And all four original cells reproduced their *exact* argmin from the
previous session (2/3/1/0). A nominal outcome with five values, predicted in
advance, correct in every cell.

The permutation p was published as **0.0 on 20 000 draws**, and that was wrong in
kind, not degree (@Reviewer, `53f9258b`). The run scored *zero hits*, so
`round(0/20000, 5)` printed `0.0` — which reads as an exact probability and is
not one. My fix then labelled `1/20 001` a bound, and **that was wrong too
(@Reviewer, `8714d13e`)**: `(b+1)/(B+1)` is the standard conservative *add-one
Monte Carlo p-value*, an estimate. Zero hits in B draws does not deterministically
bound the true p by `1/(B+1)`. The honest one-sided 95% Clopper–Pearson upper
limit at zero hits is `1 − 0.05^(1/20000)` = **1.497754 × 10⁻⁴** ≈ 3/20 000 — the
rule of three — which is 3.00× the figure I had called a bound. Fortunately this
null admits a closed form, so neither estimate is needed: the sample space is the
multiset permutations of the observed argmins over 24 processes, 24!/(4!·8!·4!·4!·4!) =
46 381 007 673 000, and matching the observed score requires all six cells
unanimous — which forces the eight copies of slot 1 into exactly two cells and
each other value into one, giving 6!/2! = 360 favourable. **Exact p = 360 /
46 381 007 673 000 = 1/128 836 132 425 ≈ 7.76 × 10⁻¹².** That reproduces
@Reviewer's independently derived figure to the digit — and my first closed form
did not, returning 720; see the correction note below.

**But P1's own contrasts refute the headline P1 was meant to support.** Two
deltas have opposite signs, and that is fatal to a monotone bytes response:
**lo_lo (12 GiB) is slower than zero**, and **step (6.5 GiB) is faster than 12,
24 and 48 GiB**. The rate is not monotone in total prior bytes. The four-cell
grid's smallest prefix was 12 GiB, so every ordering it could observe was
consistent with monotone — the 94.53% measured the rising segment of a
non-monotone curve and reported it as the curve.

The `lo_lo < zero` reversal holds on **all four aggregation bases**. This is not
the basis-dependence problem again; it is a fact about the data.

**A count I got wrong, in a commit message rather than here (@Reviewer,
`53f9258b`).** I reported *four sign changes* along the bytes axis. Two cells
reach 24 GiB by different routes, so the x axis has a **tie**, and their relative
order in the sequence is insertion order from `CELLS` — arbitrary. Counting sign
changes across a tie charges it one. Swapping the tied pair gives 2; collapsing
the tie to its mean, the only order-invariant treatment of two points at one x,
also gives 2. The artifact now reports `sign_changes_TIE_AWARE` (2) alongside the
stored-order count, with the tie-collapsed sequence beside it. **Non-monotonicity
is invariant to all three orderings**, so the refutation is untouched and only the
number was wrong — but the number was published, and an arbitrary
insertion-order artifact is exactly the kind of thing that should not be load
bearing.

Two things worth separating. The earlier **data** is sound — the four cells
replicated to within 0.0035 TB/s across independent sessions with different
seeds, and the diagonal reproduced to four decimals (+0.01019 → +0.01014). The
**interpretation** was wrong. No amount of within-grid rigour could have caught
that; only a point outside the grid could. Every control I built — shuffling,
residualizing, permutation nulls, the aggregation sweep — operated *inside* a
range that never contained the counterexample.

**And the largest single effect in this whole line of work is in `step`.** Its
slot 0 reads 5.3171–5.3564 TB/s in all four processes — **0.2583 TB/s above the
best single observation in every other cell** — while its other four slots sit
*below* most cells. All seven rounds of every one of those measurements agree
within ~1%, so by @Reviewer's own `23a6f662` test it is a stable state, not an
outlier. `count=13` does not shift the level; it produces a **qualitatively
different placement**, and a per-process mean averages that bimodal profile into
a middling 4.977 and calls it a level.

That is the aggregation defect in its strongest form yet. Everywhere else the
collapse merely picked one estimand among several; here it *destroys the
structure that is the actual finding*. Why a placement is faster remains
unexplained — `count=13` was chosen because it is the first staircase step, and
nothing predicted this.

**@Reviewer also found this separation is *stronger* than I stated**: all 28
`step` slot-0 rounds (801.84–813.47 µs) are faster than every one of the 700
rounds outside that cell (min 849.00 µs). I had reported the margin against the
best per-slot *rate*; the round-level statement is the one his `23a6f662` test
actually asks for, and it is cleaner.

(`c4bbc6b`'s commit message says *812 rounds*. That is the count outside step
slot 0, not outside the step cell — 700 is right. A commit message cannot be
amended once pushed, so the correction lives here.)

###### A closed form is not self-checking

Replacing the `0.0` above with an exact p, I wrote `6!/max(doubled,1)!` off an
ad-hoc count of values appearing twice per cell. It returned **720**, not 360 —
it detected the one doubled value and then divided by `1!`. The formula ran
green, produced a plausible probability, and was wrong by exactly 2×.

It surfaced only because @Reviewer had derived `1/128 836 132 425`
independently and my regeneration printed `1/64 418 066 212`. Nothing in my own
tooling could have caught it: there is no artifact to diff against, because the
formula *is* the source. The fix is written per-value —
`ncells! / Π(count_v/reps)!` — so there is no special case left to get wrong,
and it now reproduces his figure to the digit. **Two independent derivations of
the same closed form is the only check that exists for one**, which is an
argument for the reviewer seat doing arithmetic rather than only reading mine.

###### The obvious mechanism is wrong: the layout is byte-identical across cells that disagree

`AI/probe_alloc_layout.py`, run at `AI/artifacts/alloc_layout.json`. The first
thing to check is the one that would explain everything at once: the prefix
moves where the six measurement buffers land, and some addresses are better than
others. That would give unanimity within a cell (same layout every process),
disagreement between cells (different layout per cell), and non-monotonicity in
bytes (addresses are not monotone in the bytes allocated before them) from a
single cause. I wrote the probe expecting to confirm it.

It reproduces `_identical_buffer_spread`'s allocation sequence exactly and
records pointers instead of timing anything — one 2 GiB `dst`, then five 2 GiB
`srcs`, after the same prefix — with the sequence asserted against that
function's source so an edit there fails loudly rather than silently leaving
this file describing a layout nothing measures.

| cell | count | size | dst offset | src offsets | argmin |
|---|---|---|---|---|---|
| zero | 0 | 512 | 11112808448 | 8963227648, 6448742400, 4299161600, 2149580800, 0 | 4 |
| step | 13 | 512 | 10747904000 | 8598323200, 6448742400, 4299161600, 2149580800, 0 | 1 |
| lo_lo | 24 | 512 | 10747904000 | 8598323200, 6448742400, 4299161600, 2149580800, 0 | 2 |
| lo_hi | 24 | 1024 | 10747904000 | 8598323200, 6448742400, 4299161600, 2149580800, 0 | 3 |
| hi_lo | 48 | 512 | 10747904000 | 8598323200, 6448742400, 4299161600, 2149580800, 0 | 1 |
| hi_hi | 48 | 1024 | 10747904000 | 8598323200, 6448742400, 4299161600, 2149580800, 0 | 0 |

**Five of six cells are byte-identical and carry four distinct argmins.** A
variable that does not vary cannot explain an outcome that does. Relative
virtual layout is not the mechanism. Deterministic in all six cells across three
independent processes, six allocator segments everywhere, all buffers 2 MiB
aligned — so this is not a null from noisy measurement, it is the same layout.

What survives is **physical** placement. The caching allocator hands out virtual
addresses; the driver picks the physical pages behind them from a free-list
whose fragmentation the prefix history does change. MI355X interleaves channels
and MALL sets by physical address, so identical virtual offsets can still stripe
differently — precisely the state this table cannot distinguish. I have no
userspace physical-address visibility on ROCm, so this tool stops here.

`zero` is the one layout-distinct cell, and its argmin is the one no `count>0`
cell shares. That is one draw. With five cells sharing a layout and disagreeing
four ways, reading the sixth as support would be fitting a rule to the single
point that did not refute it.

Three corrections found while reading my own output, all one defect:

- **Both Type-II "main effects" came back significant** (p = 0.0000 and
  p = 0.0003) and I nearly published them as the separation. Neither holds total
  bytes constant: `bytes_f` is *per-buffer* size, and count × per-buffer size
  **is** the total, so doubling count at fixed per-buffer size doubles the total
  too. Only the diagonal holds it fixed. A factor name says which variable is
  adjusted for; a reader in a hurry reads it as which quantity is held constant.
- **The precondition reads false.** 24 × 512 MiB is exactly 12.0 GiB — *equal* to
  the measurement's own live set, not above it. `lo_lo` sits on the boundary the
  design set for itself. The diagonal is unaffected; `lo_lo`'s anchor role is
  what weakens.
- **The grid does not cover the steps.** The staircase's steps are at 13/17/21
  prior allocations — 6.5 to 10.5 GiB. Every cell here starts at 12 GiB. This
  characterises the curve *above* where the steps were found. The response is
  also concave (+0.0491 for 12→24 GiB, then +0.0147 for 24→48), so the routes
  are compared where the curve is already flattening and a route effect could be
  larger lower down.

`count=0` and `count=13` anchor cells were **pre-declared** for this, before the
numbers were read — the assembler was committed while the sweep was still
running (`7ba33a6`), with the unit of analysis, the thresholds, the band's units
and this limitation all fixed in advance, precisely because the staircase needed
three iterations largely because each round's analysis was chosen after its
numbers were on screen. Those anchors remain open.

##### Time reversal was not enough; the level had to stop being a function of when

The first sweep ran levels 0→21 in wall-clock order, which leaves level
perfectly confounded with collection time: any slow drift in the box over the
~6 minutes reproduces as a staircase with no allocator involved. @Reviewer's
point, and it was right. The sweep then ran an ascending pass and a descending
pass, 88 rows, one fresh process per row — only the order of *processes*
differs.

**Steps at 13, 17 and 21 appear in both directions; none appears in only one.**
That is necessary and it is not sufficient, which I claimed it was.

@Reviewer's `5c2e0083`: all-up-then-all-down leaves level an exact function of
collection position. I checked it rather than conceding it in prose, and it is
worse than a tendency — the 88 up/down positions fall into **44** classes
equidistant from the nearest end of the sequence, and **not one class holds two
different levels**. Level and symmetric position are the same variable there. So
any midpoint-symmetric function of time reproduces the staircase in both passes
at once, a single transient at the turnaround included, and no statistic
computed on those two passes can separate them. **Time reversal cannot break a
symmetric confound, because reversal is itself symmetric.** I had written "the
staircase is a level effect, not drift" off the back of it.

The fix is a third pass that visits every (level, repeat) in one seeded shuffle
(`SWEEP_SHUFFLE_SEED = 20260802`), so no monotone *or* symmetric function of
position recovers the level. The repeats are shuffled in rather than run
back-to-back — two adjacent processes at the same level share whatever the box
was doing in that half-second, which is the same adjacency problem one scale
down. The up/down passes keep their paired repeats, since changing two things at
once would make a difference between passes unattributable.
`interleaved_control` reports the measured association, not merely that a
shuffle happened: correlation of level against position **−0.020**, against
distance from the midpoint **0.111**, longest run of one level **2**.

**The interleaved pass finds steps at 13, 17 and 21 — the same three.** Written
down before the run, in case it hadn't: if the shuffled pass had found different
steps, those would have been the result and the earlier three would have been an
artifact of ordering, not something to reconcile toward. It did not come out
that way, and the level means agree across all three passes to the third decimal
(level 12 → 13 moves 4.917→4.977 up, 4.922→4.976 down, 4.917→4.977 interleaved).
So the steps are a level effect. *Why* the level matters is still not addressed
by any ordering control, and `what_it_still_does_not_establish` says so in the
artifact.

One more correction, and it is the sharpest one here, because it is a guard
failing in the flattering direction. I wrote the identity check as
`level == min(seq, N−1−seq)` — true when each pass has one row per level, which
was the layout at the time. Adding a second repeat per level doubled the
position axis, and the check went to **86 exceptions out of 88** while the
confound itself was completely untouched. Its failure branch then printed *"the
up/down passes carry some independent information about level"* — a check
written to restrain a claim reported the claim's obstacle as weaker than it is.
Had I not recomputed it by hand, the artifact would have quietly said the
confound had loosened at the exact moment I doubled down on it. The check now
tests the property (group positions by distance from the nearest end; does any
group hold two levels?) rather than a formula that encodes one collection
layout. **A guard whose correctness depends on the shape of the data will
mislead precisely when the shape changes, which is when it is most needed.**

Note that the up/down pass is a *different* control from the one shown invalid
earlier. The
broken version walked the grid up and down within one process, which re-rolled
placement on every call and destroyed the effect. Reversing the order of fresh
processes has no such problem, and it took @Reviewer's framing — level confounded
with time — to see that the valid version of the control was still available
after the invalid one was abandoned. I had recorded "reversibility is now only
supported by repeats at the same level" as a permanent limitation. It wasn't.

The sequence is worth keeping intact, because it is three iterations of one
mistake. Within-process A-B-A: invalid, and it *looked* more careful than what
replaced it. Up/down over fresh processes: valid as far as it goes, and I read
it as establishing something it structurally cannot. Interleaved: breaks the
functional relationship outright. Each time, the control I had just built felt
like the end of the argument — **the reason a control convinces me is that I
built it to answer the objection I could see.**

Two corrections to how that was established, both mine. The step locations were
first written into this file from an exploratory script in `/tmp` — a number
whose only home was prose, in the section about numbers whose only home is
prose. And the sweep written to back them was **invalid on its first design**: I
walked the peak grid up and back down *within one process* to test
reversibility, and it reported a ~10% shift at every single level including
0→1, i.e. no threshold anywhere. The measurement is not passive — each
`_identical_buffer_spread` call allocates and frees six 2 GiB buffers, which
re-rolls placement by itself. Repeating the identical call eight times in one
process, changing nothing else, gives a different slot pattern every time. The
sweep was measuring its own alloc/free cycles. The broken design looked *more*
careful than the fix, which is the part worth remembering: controlling for
process-level variation controlled away the effect. Reversibility is now
supported by the ascending/descending pass over fresh processes described above,
not by any within-process A-B-A.

That number entered the tree **as part of the fix for hand-typed constants**.
The sentence written to eliminate untested numbers contributed a mislabelled
one, and it survived because a cross-process spread is exactly the kind of
figure no single run can recompute — the guard that catches invented decimals
cannot catch a real measurement of the wrong thing.

The denominator choice is unchanged but now rests on the right evidence: not a
21× stability gap, which does not exist, but the fact that copy is sensitive to
allocation slot (5.89–7.55% at 2 GiB within a fixed prefix) and to the harness's
prior allocation history (8.92% across prefixes), while `write` and
`two_read_one_write` are not.

A third correction from the same measurement, worth recording because it nearly
went out: `two_read_one_write`'s **0.31%** apparent stability was measured at a
single allocation slot with the peak held fixed. Sampled across slots it spreads
**2.46%** at 512 MiB. I had been about to use that 0.31% as an error bar to
argue, from a 4.04% gap, that the historical table must be 2 GiB. **No probe
here can tell 512 MiB from 2 GiB** — all three patterns' ranges overlap once
slot and peak are sampled. The table's size is known because notes:869 says so.
That is documentary evidence and it is not dressed up as measured.

##### "Slot" was three variables wearing one index, and it is the address

Every copy probe in this tree allocated buffer *i* in position *i* and then
measured it *i*-th. Allocation ordinal, timing order and address were the same
number, so calling the effect *placement* was a reading of the design, not a
result from it. @Reviewer's objection, and there was no answer to it in any
artifact I had written: the same data are equally consistent with clock ramp,
with cache warmth, or with anything else monotone in measurement order.

`AI/probe_order_confound.py` allocates the five 2 GiB sources in a fixed order,
then measures them in a **per-process random permutation** — 20 processes, 19
distinct orders, 100 draws, addresses recorded. That breaks the index into its
parts.

| variable | η² of the rate |
| --- | --- |
| allocation ordinal | **98.10%** |
| timing position (marginal) | 7.11% |
| timing position *within* allocation ordinal, saturated | 0.30% |
| **timing, additive main effect adjusted for ordinal, blocked on process** | **0.0412%** |
| between identical repeats of one (ordinal, position) cell | 1.60% |

The two marginal numbers overlap and do not sum to 100 — a random permutation
per process gives an unbalanced grid (all 25 cells occupied, 1 to 7 replicates
each), so neither marginal is a residual.

I then picked the wrong line out of the remaining ones. I wrote that the third
row carried the argument and concluded "warmup, clock ramp and drift are out."
@Reviewer, `a7fe31c8`: **0.30% is a saturated cell term, not a timing main
effect.** It is what is left between the 25 cell means after removing each
ordinal's mean, so it contains the ordinal×time interaction, and with cells
holding 1 to 7 replicates it absorbs per-cell noise too — a cell of size one
contributes its entire residual to it. The main effect is a different fit:
model the rate as ordinal + time with no interaction and ask what the time term
buys. **0.0412%, F(4,72) = 0.52, p = 0.72** with process as a blocking factor.

Same direction, an order of magnitude smaller, and — this is the part that
matters — a different *kind* of statement. A p of 0.72 at n=100 is a bound, not
an exclusion. The defensible sentence is that this design **detects no additive
effect of measurement order**, which constrains warmup and clock ramp without
ruling them out. "Are out" claimed the null as a result. The artifact now
publishes both terms with the saturated one labelled as such, so the reader can
see which is which rather than having to trust that I picked correctly.

The addresses say what "slot" actually means. Across all 20 processes the source
minus destination offset vector is **byte-identical** — `-10.35, -8.348, -6.346,
-4.344, -2.002` GiB — while the absolute destination base lands in **11**
distinct 1 TiB regions. That is the finding, and it is a real one: **absolute
placement is randomized here and does not track the rate**, so whatever carries
the effect is relative.

Bucketing by relative offset also recovers 98.10%. I reported that as
confirmation. It is not — @Reviewer again, same message. Offset bucket and
allocation ordinal are a **bijection** on this design (ordinal 0→bucket −1,
1→−2, … 4→−5, checked and now published as
`is_a_bijection_with_alloc_ordinal`), so they induce the identical partition of
the 100 rows and *any* between-group statistic on them is equal by construction.
Two numbers agreeing because they are the same number is the weakest possible
evidence dressed as the strongest, and I had written it in the sentence right
after noting that per-address η² would read 100% and mean nothing for exactly
this reason. I spotted the degenerate grouping and missed the degenerate one
next to it.

So relative offset is at most a **redescription** of allocation ordinal here,
not an explanation of it, and this design cannot tell which is the carrier.
Doing that needs a probe that varies the offset directly at fixed ordinal.

What this does not do is retroactively upgrade the earlier artifacts. The
placement reading turns out to be right, but it was never supported *by* those
runs, which could not have told these cases apart. A conclusion being correct is
not the same as the experiment having shown it — and the failure mode is not
that I was wrong, it is that I would have said the same thing either way. Nor
does this establish *why* a given offset is faster; that needs a probe that sets
the offset directly instead of reaching it through the allocation sequence.

Artifact: `AI/data/copy_placement_draws/order_confound_dev5.json`, with the 20
raw runs under `raw_order_dev5/` and hashed into `input_manifest`. Its loader
now refuses duplicate seeds, mixed devices or sizes, ragged row counts,
non-permutation indices and non-finite rates, and records whether the input
directory was inside the repo at all — the ambient-`/tmp`-input hole @Reviewer
found in the copy-axes assembler, closed at the point where the assumption is
made rather than by hard-coding the path.

The live figures are in `copy_variability` and
`denominator_stability_across_processes` in the sidecar, and the full three-axis
decomposition in `copy_axes_dev5.json`; every number in the caveat string is
interpolated from a computed field, and the generator refuses to write a caveat
containing a decimal no field produced — that guard has now fired on my own
replacement text **three** times: once when I hand-typed within-run spreads
while fixing hand-typed numbers, once when I removed a superseded value from its
allowlist while leaving it quoted in the retraction, and once when a re-collection
moved a spread I had carried by hand into a verdict string.

The assembler has the same guard, and its false-negative rate is now **computed
at write time** (`prose_guard` in the artifact) rather than stated in a
docstring. It had to be: the docstring said **5.2%**, @Reviewer recomputed
**8.246%**, and he was right. That figure was measured once on a smaller payload
and then quoted as a property of the guard — a number that appears only inside a
prose string has no error bar and never re-runs, which is @Autotune's rule
applied to the very mechanism built to enforce it.

What moved it is not what I would have guessed. Across the artifact's own
history the rate runs 7.996% → 8.546% → 8.296% → 9.445%: it rises because **the
payload grows**, since more measured values cover more of the grid by accident.
The last of those jumps happened while writing this paragraph — adding the
measured gap list that replaced the uniform prior widened the accept-set by
another point. Improving the artifact degrades its own tripwire, monotonically,
and nothing warned you — so the assembler now **refuses to write** once the rate
passes a 12% ceiling, with the error saying explicitly
that raising the ceiling to make it pass is the failure mode being interrupted.
A tripwire that decays as a side effect of good changes needs a hard stop, not a
field reporting its own decline. Widening
the accept rule from 0–3 dp to 0–5 dp — the change I had written a code comment
to worry about — costs exactly **0.0000 pp** on a 2-dp grid and shows up only at
4 dp (+0.21 pp). The risk I annotated was harmless; the mechanism that actually
degraded the guard was routine growth, and it had no comment at all.

And the rate was still measuring the wrong set. @Reviewer, `a7fe31c8`: the
accept rule is `tok in cited or tok in measured`, but the self-assessment
counted only `measured`. **A permissiveness metric that excludes the hand-
maintained half of its own accept-set** — the half that grows precisely when
someone wants a particular number to pass. Five allowlist strings sit on the
2-dp grid uncovered by any computed field, so the published figure was low by a
quarter of a point. The magnitude is nothing; the blind spot was aimed exactly
at the mechanism by which a guard gets loosened, and every future allowlist
entry would have been invisible to it for free. Now `10.345%`, with
`of_which_only_the_citation_allowlist_explains: 5` broken out so an entry to the
allowlist visibly costs something. **A self-monitoring check must monitor the
part of itself a person edits, not just the part that grows on its own.**

The section below was written to argue that copy is too *low* to be a ceiling,
which is true and insufficient — the stronger objection is that it is not stable
enough to be anything.

The original text, kept because the reasoning is still right:

The harness originally normalized against a `torch.copy_`, which sustains only
4.89 TB/s on MI355X — low enough that the RMSNorm forward exceeded it and
reported over 100%. Probing three patterns over 2 GiB buffers:

| pattern | TB/s | share of the 8 TB/s HBM3E spec |
| --- | --- | --- |
| pure write | 6.84 | 85% |
| two read + one write | 6.09 | 76% |
| same-device copy | 4.89 | 61% |

The forward peaks at 5.73 TB/s, which is 84% of the best probe, 94% of the
mixed-traffic probe, and 72% of the datasheet number. Report against the best
probe; it is the conservative denominator because a pure write has no
read/write turnaround on the bus.

## Against PyTorch on the same part

`torch.nn.functional.rms_norm`, same harness. Forward: 38 of 45 cells ours,
median 1.46x, with the seven losses all small-batch and launch-bound.
Backward: 45 of 45, median 4.51x, because torch's backward sits at 14-15% of
peak bandwidth on every large shape while ours reaches 76%.

## Where the backend stands against the CuTe kernel

Same harness on all three machines, `benchmarks/benchmark_rmsnorm_flydsl.py`,
which gates correctness per cell. "Evicts L2 between timed calls" is what this
line used to say and it does not describe the code (@Reviewer, blocker 3): one
hipEvent pair brackets a **whole rotation**, and the evictor runs **once per
round, outside that window** — not between calls, and not inside the timed
region. The per-call figure in the CSV is the round divided by
`calls_per_round`. Copy roofline: MI355X 5279 GB/s, H200 4148 GB/s,
H100 2974 GB/s.

| regime | FlyDSL on MI355X | Quack on H200 | Quack on H100 |
| --- | --- | --- | --- |
| M=32768 (bandwidth bound) | 100% / 88% | 80% / 74% | 89% / 87% |
| M=4096 (not saturated) | 71% / 64% | 33% / 27% | 61% / 41% |
| M<=512 (launch bound) | 7% / 5% | 14% / 9% | 18% / 12% |

Percentages are forward / backward share of that machine's own copy roofline.
Large shapes are decided by HBM; mid shapes by the kernel; small batches are
pure launch path, where the CuTe kernel is about 2.2x ahead (6.1 us against
13.0 us at M=1, against a ~3.5 us Python/FFI floor).

> **The MI355X column is known to be measured wrong in many cells.** On gfx950
> a rotation working set of 256 MiB or less stays resident in the MALL, and the
> `use_evictor` gate compares against a 12 MiB target derived from the 4 MiB
> per-XCD L2 that torch reports, so on these shapes no eviction runs at all.
> A `copy_` probe measures **~1.3x** inflation at the boundary (1.266–1.354x
> across runs and both buffer sizes; the third digit does not reproduce),
> confirmed by a
> second reviewer re-running the committed probe. **That ~1.3x is a property of
> the probe, not a correction factor for this table.** It was measured on a
> pure `copy_` stream, not on RMSNorm, and no before/after RMSNorm run exists
> yet; it says the measurement method is unsound on MALL-resident shapes, not
> how much any particular cell moves. Do not multiply or divide these regime
> values by it.
> Computing from the harness's real `logical_bytes` and actual buffer selection,
> **37 of 90 cells** are both un-evicted and MALL-resident (8 of 18 per 16-bit
> mode; 5 of 18 for fp32/same) — i.e. 37 cells are *exposed to* the defect, with
> the per-cell magnitude unmeasured. **No `m=32768` cell is among the 37**, but
> four `m=32768` cells are in the 41 (below).
>
> **37 and 41 count different things; do not merge them.** 37 is the
> strict-resident subset: `ws <= 256 MiB` and un-evicted. It excludes the four
> `32768x1024` fwd 16-bit cells, whose working set is 256.004 MiB — *over* the
> MALL by 4 KiB, so not strictly resident, yet directly probed and shown to read
> on the MALL-warm side. Counting those as contract-invalid too gives
> **4 x 9 + 5 = 41**. Use 37 for "strictly resident and un-evicted" and 41 for
> "measurement contract not established"; the headline number is 41, and an
> earlier version of this note derived 9-per-mode correctly and then still wrote
> the total as 37.
> The `M=4096` row (71% / 64%) is therefore **invalid pending re-collection —
> not "optimistic"**: a `copy_` probe losing MALL residency shows the
> cold-measurement contract was never established, and does not transfer a
> direction or a size to an RMSNorm cell. `M<=512` is launch-bound so bandwidth
> is not the binding
> constraint there; and `M=32768` contains none of the 37 but four of the 41,
> so it is not clean either — the four 16-bit `32768x1024` forward cells land a few KiB *past* the
> MALL (256.003906 and 256.007812 MiB), so the threshold excludes them, yet a
> `copy_` probe at those exact working sets reads 6391 and 6367 GB/s against
> that run's 4992 GB/s HBM reference, i.e. 1.28x and 1.28x high. (This line
> previously read "6068 and 6398 ... against ~4935 ... 1.23x and 1.30x", which
> mixed the *previous* sidecar's values with the current one's and quoted the
> two working sets in the wrong order; the figures above are the current
> sidecar, `c0b7c0b`.) That is a statement about
> **copy traffic at those working sets**, not a measurement of the RMSNorm cells
> themselves. The 256→288 MiB decay is gradual, so a
> threshold misclassifies cells sitting either side of it. Full analysis in
> [`gfx950_mall_evictor_defect.md`](gfx950_mall_evictor_defect.md). All three
> rows should be re-measured before being cited; do not assume the large-m
> median is unaffected without recomputing it.

**Provenance of the two Quack columns** (added 2026-08-01 after this table was
challenged as unsourced, then verified and cleared). Both halves come from the
schema-v1 sweep of 2026-07-26, not from Experiment No.001:

- H100: `AI/archive/v1-20260726/h100-v1-results.csv`, 180 rows,
  `copy_roofline_gbps = 2974.420002`,
  sha256 `59d9429b95a1494eaec61c095543b4a831a217c8affc2547b675c7f60fdecea7`.
- H200: `AI/archive/v1-20260726/h200-v1-results.csv`, 180 rows,
  `copy_roofline_gbps = 4148.155822`,
  sha256 `6a3cd36eb42f9f314cd601e4fd06354698c0f7455d07bbe1d6d45ece5a01cc50`.

Both are the two denominators already quoted above.

The rule that reproduces every cell: median of `copy_roofline_pct` over rows
with `provider=quack`, bucketed by `m==32768` / `m==4096` / `m<=512` and by
operation, rounded to whole percent. Reproduced independently on both files:
88.695/87.185, 60.952/41.220, 18.359/12.039 for H100 and 79.647/74.317,
32.899/27.149, 13.792/9.146 for H200. The M=1 forward figure of 6.1 us above is
the same file's `quack` `m=1` `fwd` median.

Both files are committed **in this repository** at `AI/archive/v1-20260726/`,
with `SHA256SUMS` and a README recording the formula, so the provenance is
resolvable from the PR rather than from a path on one machine. The per-cell
`tuned_config` probes are archived beside them. Do not attempt to re-derive this
table from the No.001 archive — that is a later, differently-configured run and
it will not reproduce these numbers.

Limits of this provenance: schema-v1 records no commit SHA, toolchain version or
hostname, and the 2026-07-26 date comes from file mtimes rather than the data.
The bundled `tuned_config` probes are likewise a *separate* run from No.001 —
their widest-row gains are 8.90%/16.75% against the published 9.155%/17.479%,
and the two probe files even used different torch builds (2.11.0+cu130 on H100,
2.9.1+cu128 on H200). They establish which config won, not the published
timings.
These bytes pin the numbers; they do not pin the code or environment. Only the
two Quack Hopper columns are covered — the MI355X column's six regime values,
its 5279 GB/s roofline, the 13.0 us M=1 figure and the ~3.5 us Python/FFI floor
remain unarchived and unverified.

Caveat worth keeping: at 4096x3000 and 4096x4096 Quack is slower on the H200
than on the H100 despite 1.39x more bandwidth. Stated exactly, over the 20
matched m=4096 cells in the v1 pair:

- **Quack: 20 of 20 cells slower on H200**, ratio range 1.027–1.346. The
  direction is unanimous.
- **torch: 17 of 20 cells faster on H200**, ratio range 0.778–1.242. The three
  exceptions are all backward, weight_mode `same`: 4096x3000 fp16 1.242,
  4096x3000 bf16 1.236, 4096x4096 fp16 1.063.

The `0.78–0.88x` figure that used to appear here was wrong as written. It is a
real per-cell range, but only of the **12 `weight_dtype=float32` cells**
(0.777923–0.883896); the paired Quack range 1.03–1.33 is the same subset
(1.027449–1.327160). Presenting a single-weight-dtype subset as if it held for
"every dtype" is what hid the three counterexamples, all of which are
`weight_mode=same`. The torch control is therefore *mostly* in the opposite
direction, not uniformly.

**What this does and does not establish.** Quack's 20/20 one-directional result
against a control that mostly runs the other way is a real asymmetry worth
recording. It is not proof of the mechanism. The earlier wording — "that is
Quack tuning on H200, not the machine" — asserted a cause the data does not
isolate: schema-v1 carries no commit, toolchain, host or config metadata, so
environment differences are not excluded, and the control has three
counterexamples of its own. Read it as an observation awaiting a controlled
experiment (same commit, same toolchain, winning configs dumped per cell), not
as a finding.

Two further corrections to how this caveat used to be worded. First, it claimed
those cells were "excluded from any median quoted above." They are not, and
cannot be: 3000 and 4096 are the *only* N values at m=4096 in this sweep, so the
M=4096 row of the table — H200 33% / 27% — is computed from exactly those ten
dtype-shape cells and nothing else. The caveat explains that row; it does not
exempt it.

Second, the caveat is specific to the v1 run and must not be carried across to
No.001. In the No.001 archive the direction reverses for Quack: over 60 matched
m=4096 cells the H200 is faster in **58**, the two exceptions both being torch
backward at 4096x3000 (fp16 1.081, bf16 1.044). An earlier version of this
paragraph said "on every provider," which is wrong for the same reason as
above — a near-unanimous result reported as a unanimous one.

## `MAX_N = 8192` is a real cliff, but not where the constant says

`rmsnorm_config.MAX_N` rejects any row wider than 8192, and its comment calls
it "the register budget expressed as a row length. Every other cap on N derives
from it." Monkeypatching it to `1 << 20` (probe only, nothing committed) says
the correctness half of that is false: the forward runs to **N = 262144** and
the backward to **65536**, at bf16 accuracy indistinguishable from the shapes
under the cap -- forward relative error 1.0e-3..2.1e-3 across
16384/32768/65536/131072/262144, backward `dx` 2.2e-3 and `dw` 2.3e-3..3.2e-3
against an fp32 reference at 8192..65536. Nothing spills, nothing wraps,
nothing silently truncates. cutedsl for comparison gates only at
`N > 128k with dtype >= 32 bits` (rmsnorm.py:637) and escapes the register
budget above 8k with `reload_from="smem"`, which this backend has no analogue
for.

There *is* a cliff, and it is worth keeping a cap for -- but it sits between
**49152 and 57344**, not at 8192. Holding `m` fixed at 4096 (16 blocks/CU
throughout) isolates width against the `two_read_one_write` ceiling. Sidecar:
`AI/data/rmsnorm_fwd_width_cliff.json`, generator
`AI/probe_rmsnorm_width_cliff.py`.

| N | elems/thread | share of ceiling (**graph**) | working set | timing |
| --- | --- | --- | --- | --- |
| 32768 | 128 | 83.1% | 512 MiB | clean |
| 49152 | 192 | 77.5% | 768 MiB | clean |
| 57344 | 224 | **37.8%** | 896 MiB | clean |
| 65536 | 256 | 39.6% | 1024 MiB | clean |
| 98304 | 384 | 39.2% | 1536 MiB | clean |

(The column is labelled because an unlabelled one caused a defect elsewhere in
this file — see the occupancy section. These are graph-replay shares; the eager
column of the same sidecar reads 82.5 / 76.6 / 38.1 / 39.2 / 39.4.)

Flat to 49152, then halves between 49152 and 57344. The step is abrupt, not a
slope. Every row above is past the MALL, is timed at least 13x above the host
floor, and carries a dispatch floor under 1% of its own measurement; eager and
graph-replay timing agree on all five to within 0.9 points. That is a real
width limit.

**The N=8192 row has been removed, and the block-count control with it.**
Writing the generator for this table -- the last *table* without one, though
several individual figures elsewhere were still unbacked at the time --
turned up three floors that the original protocol could not see, and two of the
published series were made of them:

1. *The eager path is host-bound below ~30 us.* Wall time of the Python call
   with the GPU never awaited is ~30 us; measured eager time at N=8192 is
   ~31 us. The number was the harness, not the kernel.
2. *Graph replay has its own floor* -- ~9.5 us for a single captured call,
   converging to 1.49 us per kernel at K=256, measured from a 64-element
   `add_` that moves 512 bytes. So switching timing vehicles does not by
   itself fix (1); the floor has to be amortised and then reported, which the
   sidecar now does per row.
3. *MALL residency differs across rows.* N=8192 at m=4096 has a 128 MiB
   working set and is cache-resident, while the ceiling probe's is 512 MiB and
   is not -- so its "share of ceiling" divided a cache-resident numerator by a
   DRAM-resident denominator. Timed properly it reads **103.7%**, above the
   roofline, which is the same tell that already got `N=16384` excluded two
   paragraphs down. That exclusion was right and did not go far enough: 16384
   sits *at* the 256 MiB boundary and 8192 is well inside it, so the row that
   was kept was the more cache-resident of the two.

The block-count control fails on the same three grounds at once. It reported
78.5% -> 5.3% of ceiling from m=4096 to m=256 and that was read as starvation,
but under the published protocol the wall time across that entire range is
**flat at ~30 us** -- sixteen times the work, no change in the clock. A falling
TB/s from a fixed time and a shrinking byte count is arithmetic, not
starvation. Every row of it is also MALL-resident. Block-count starvation is
real and does appear once the floor is amortised away (102% -> 40% of ceiling
over the same range, still cache-resident and so still not a DRAM-bandwidth
measurement), but it is a different curve than the one published, and it no
longer supports the claim it was cited for: **the naive sweep's degradation was
attributed to starvation on the strength of a series that was measuring the
host.** Both series are retained in the sidecar so the two can be compared;
neither is quoted here as a bandwidth result.

What survives is the cliff itself, which is the one row-pair in the section
that was never affected by any of this. What does not survive is the argument
that the confound in the naive sweep was *quantified* -- it was not, and the
sidecar now says so.

That cliff does look like occupancy collapse from per-thread live state -- so
the comment's *mechanism* is plausible while its *value* is off by 6x. The
register counts and spill reports have since been read out (see the
register-budget section at the end of this file) and they put a capacity step
on this same boundary. Occupancy has since been measured there too: it halves
from 1.96 to 1.00 waves/SIMD between 49152 and 57344, matching the computed
bound. An intervention at fixed N then moved it back -- forcing 2 waves at
57344 recovers bandwidth from 38.1% to 52.7%, while the same hint at 49152,
where it has nothing to move, changes nothing. So the direction is established;
the magnitude is not, the lever also adds spills, and occupancy is demonstrably
not the whole story -- a pair matched at ~1.94 waves/SIMD still differs 1.45x in
bandwidth across the two widths. **The cap is not raised
on the strength of this.** A constant that is conservative by 6x costs
reachable shapes; a constant moved on a mechanism whose magnitude is unmeasured
costs correctness somewhere unmeasured. The finding is
that 8192 is not where the hardware objects, and that whoever raises it should
raise it to 49152 and say why -- not that it should be raised today.

Note also that the `N=16384, m=4096` cell read **104.3%** of ceiling, above the
roofline. That working set is 256 MiB, exactly the MALL boundary that
`AI/gfx950_mall_evictor_defect.md` documents, so this row is cache-warm and is
excluded from the table above rather than explained away.

## A cited number needs a sidecar, and two of mine did not have one

@Reviewer asked where the 89.9 / 183.2 / 93.3 us figures came from, and the
honest answer was that they came from ad-hoc scripts that no longer existed.
Regenerating them on the same card
(`AI/data/rmsnorm_32768x4096_bf16_roofline.json`, 32768x4096 bf16, 7 rounds of
50, min-of-rounds, events recorded on the operand device) settles three things
and unsettles one.

**The sidecar now has a generator**, `AI/probe_rmsnorm_roofline.py`, which is
the second half of the blocker above: the numbers could previously be re-read
but not re-derived, and an artifact nobody can regenerate is a screenshot. It
records host, device UUID, commit, source hashes, protocol, eviction scheme and
the `rocm-smi` utilisation at start, matching
`AI/probe_rmsnorm_harness_levels.py`, and it retains every round so the spread
is auditable. Re-running it reproduces the table below within 0.7% on every
row.

| | min us | TB/s on 512 MiB minimum traffic |
| --- | --- | --- |
| FlyDSL, L2 evicted | 90.22 | 5.951 |
| FlyDSL, warm | 90.03 | 5.964 |
| torch, L2 evicted | 154.77 | 3.469 |
| torch, warm | 155.03 | 3.463 |

(Regenerated values. The previously published row was 89.32 / 89.86 / 154.74 /
154.94; the deltas are +1.01%, +0.19%, +0.02% and +0.06%, all inside these
rows' observed 0.26–1.86% round spread.)

*Corrected (@Autotune, my arithmetic in `7de8da4`).* That parenthesis first read
**"+0.66%, +0.18%, +0.00% and −0.11%"**, and three of the four were computed off
a misaligned pairing. Each one reconstructs exactly, which is how I could tell:
**+0.66%** is the new cold row against **89.63**, the FlyDSL figure from the
*harness* comparison table further up — a different measurement of a different
thing — instead of against this table's own 89.32, whose true delta is +1.01%.
**−0.11%** is row 3's *new* value 154.77 compared against row 4's *old* value
154.94, i.e. one row's regeneration scored against the neighbouring row's
baseline; row 4's own delta is +0.06% and positive. **+0.00%** is +0.019%
rounded to two places as 0.00 rather than 0.02. Only **+0.18%** was right.

The conclusion is unchanged and in fact slightly firmer: all four deltas are
positive (regeneration was uniformly a touch slower, consistent with a single
shifted baseline rather than scatter), and the largest, 1.01%, still sits inside
its own row's 1.86% round spread. But a delta computed against the wrong
baseline is not a weaker version of the right one, it is a different quantity —
and the sign flip on row 4 is the tell: a set of deltas that straddles zero reads
as noise around a stable value, which is exactly the reassurance this parenthesis
was offering, and it was an artifact of the misalignment rather than a property
of the data.

For reference against the current sidecar, which has been regenerated three more
times since (`75a47eb`, `4d5a6f7`, `9879dda`): the committed values now read
90.083 / 90.149 / 155.334 / 155.134, so the table above is stale by −0.15% /
+0.13% / +0.36% / +0.07%. Every one of those is inside its row's spread too, and
the cold/warm ordering has flipped again — which is the point the surrounding
text already makes about a 0.2-point gap between rows that scatter by 0.5–2.8%.

**Each probe's TB/s is computed against its own traffic, which the old artifact
left for the reader to reverse-engineer.** A copy moves 2x its buffer, a
two-read-one-write 3x, a pure write 1x. Storing only the three TB/s made them
look directly comparable to the kernel's number when the byte count under each
differed; `bytes_moved` is now recorded beside every probe.

**89.9 us reproduces, and my reading of it was still wrong.** It is 5.972 TB/s.
The copy proxy read 4.772 TB/s in that run, so I had cited, as evidence of
headroom, a number 25% *above* the ceiling I was comparing it to. @Reviewer
caught the arithmetic. The mistake is the one this file already records twice:
a copy is not the roofline — and, per the section above, the copy figure was
not even stable, so the specific 4.772 should not be read as the card's rate.

Against `two_read_one_write`, 6.139 TB/s on the same card in the same process,
the forward runs at **97.9% cold / 97.3% warm**. Both halves of that need
saying, and my first correction gave only the first: 97.9% is specifically
`6.010445 / 6.138881`, the *cold* pair, while the warm pair is 5.9745 and gives
97.3%. Quoting the cold ratio unlabelled picked the flattering one of two
numbers sitting side by side in the same file. @Reviewer caught that too, in
the correction to the previous mistake.

On the regenerated sidecar the same ratios are **98.3% cold / 98.5% warm**
(`5.9505 / 6.0521` and `5.9635 / 6.0521`), and 86.1% against write. Note the
cold/warm order has flipped — warm is now the higher of the two — which is
exactly what a 0.2 percentage-point gap between rows whose spread is 0.5-1.9%
should be expected to do. Neither ordering means anything, and the earlier
paragraph's care about *which* pair is quoted matters more than the pair's
value: these two are not separable by this protocol.

Two further qualifications on the denominator and the numerator:

- `two_read_one_write` is the *traffic-matched proxy*, not an unqualified
  ceiling. The highest probe on this card is write at 6.894 TB/s, and it is the
  one the repository's own roofline code selects. Against write the forward is
  87.2%. Calling 6.139 "achievable" without naming it is a choice of
  denominator that happens to favour the result; it is the right denominator
  for a normalization's access pattern, which is the argument for it, but the
  argument has to be made rather than hidden in the word.
- The sidecar's byte count, 536,870,912, is `x + out` and **omits the bf16
  weight**. Exact traffic is 536,879,104. The weight is 8,192 bytes against
  512 MiB, so it moves the derived TB/s by 0.0015% and no conclusion here
  turns on it -- but a figure labelled as exact should be exact, and it is the
  sidecar's job to let the number be re-derived rather than approximated. Both
  counts are now stored, as `bytes_moved_min_traffic` and
  `bytes_moved_exact_including_weight`.

The three probes are in the sidecar so the denominator can be re-derived rather
than taken on trust.

**183.2 us does not reproduce and I cannot reconstruct it.** torch's fused path
is 154.7-157.5 us here across `F.rms_norm` and `nn.RMSNorm`. The only nearby
figure is the unfused fp32-weight path at 902 us, which is not it either. I
sent 183.2 to @Reviewer; it is withdrawn, not re-explained. The speedup at this
shape is 1.73x cold (`154.7379 / 89.323`) and 1.72x warm
(`154.9402 / 89.8598`), not the 2.04x that number implied. Stating it as a bare
"1.73x" repeated the habit the paragraph above corrects -- the cold pair is the
larger of two, and which pair produced a ratio belongs next to the ratio.
The regenerated run gives 1.72x cold and 1.72x warm, so the speedup is the one
figure here that is genuinely stable; ~1.72x is the number to quote.

**`157.5` and `902` remain unbacked.** Both appear in this section as torch
figures and neither has a committed sample: the sidecar measures
`F.rms_norm` only, so the `nn.RMSNorm` upper end of "154.7-157.5" and the
unfused fp32-weight 902 us are still ad-hoc numbers of exactly the kind this
section exists to condemn. They are retained because they are load-bearing only
as *negative* evidence -- they are the two candidates ruled out as explanations
for the withdrawn 183.2 -- but they should not be cited for anything else until
the generator covers them.

**The 1.9884% noise floor was borrowed.** It was a different shape, provider and
card's observed span, quoted as though it bounded this experiment. The spread
actually observed in these rows is 0.18-2.18%, worst case on the L2-evicted
FlyDSL rows -- which is the number that belongs here, and it is only meaningful
for these rows. The regenerated run gives 0.26-1.86%, same worst-case row, so
the span is itself only reproducible to about half a percentage point: read it
as "order of two percent on the cold rows", not as a bound.

The general rule, since this is the third variant of the same defect in this
file: a figure quoted in a review or a message needs the raw samples committed
next to it, or it cannot survive the question "where did that come from". Two
of these three were fine as measurements and indefensible as citations, and the
third was neither.

## Feature parity with the cutedsl backend, and where a grep lies about it

The mapping task asks how the FlyDSL backend lines up with `quack/rmsnorm.py`
on features, not just speed. A token-count diff of the two modules is the
obvious first cut and it is wrong in four of fourteen rows, all in the same
direction -- reporting a gap where there is only a different spelling. Recorded
because the naive table is the one that would have been sent.

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

The four rows a grep gets wrong: `autotune` and `persistent` live in
`quack/flydsl/`, not in `rmsnorm_flydsl.py`, so a module-scoped grep reads 0 on
both; `dw_partial` is spelled `partial`/`dweight_total` in
`rmsnorm_bwd_kernel.py` and the mechanism -- per-block partials, second kernel
reduces -- is the same; and `sm_count` is cutedsl's persistent-launch knob under
a different name. Each of those looked like a missing feature and none is.

`cluster` is a real difference but not a deficit: it is Hopper-and-later
distributed shared memory, which gfx950 does not have. Counting it as a gap
would make the backend permanently non-compliant with a hardware feature the
hardware lacks.

**The one genuine feature gap is layernorm.** `quack/rmsnorm.py` carries
`is_layernorm` through the whole stack, forward and backward, with `mean`
alongside `rstd` and a `layernorm_fwd` / `layernorm_bwd` / `layernorm_ref`
surface. FlyDSL has none of it. It is not exported from `quack/__init__.py`, so
nothing in-tree consumes it today, but it is the item to name when asked what
the backend does not yet do.

Second, smaller: cutedsl exposes `rmsnorm_fwd` and `rmsnorm_bwd` as public
entry points and FlyDSL exposes only `rmsnorm`. The top-level `rmsnorm()`
signatures are argument-for-argument identical, so callers of the public API
are unaffected; only the archived probes under `AI/archive/` reach for the
lower-level pair. Worth noting because the benchmark harness times quack at
`rmsnorm_fwd` and FlyDSL at `_launch_rmsnorm_fwd` -- different levels, which
this file already flags as a measurement hazard elsewhere.

### The harness times the two backends at different levels, and it matters below ~8192 rows

Generator `AI/probe_rmsnorm_harness_levels.py`, artifact
`AI/data/rmsnorm_harness_levels.json`.

`_QuackProvider`'s docstring says it calls "the same low-level `rmsnorm_fwd` /
`rmsnorm_bwd` entry points ... which is also the level the FlyDSL provider
measures." The second half is not true: `_FlyDSLProvider` calls
`_launch_rmsnorm_fwd`, which sits below it.

**What that difference consists of, read from the source.** An earlier version
of this section said quack's `rmsnorm_fwd` contains the reshape, the autograd
`Function` and the custom-op dispatch. It does not. `rmsnorm_fwd`
(`quack/rmsnorm.py:473`) allocates `out`, and optionally `rstd` and
`residual_out`, checks `weight_offset`, and delegates to the `_rmsnorm_fwd`
custom op. The reshape to 2-D and `RMSNormFunction.apply` live only in the
top-level `rmsnorm()`. The real asymmetry is narrower than I described it: a
FlyDSL launcher whose caller preallocates every output, against a CuTe forward
wrapper that allocates. @Reviewer caught the misattribution.

**The measurement is one-sided on this box, and that is the more important
correction.** `quack/rmsnorm.py` imports `cuda.bindings.driver`, which does not
exist on ROCm, so the cutedsl levels cannot be timed on MI355X at all -- the
import fails before any kernel runs. The old table's two columns were therefore
*both FlyDSL*: `_launch_rmsnorm_fwd` against FlyDSL's own `rmsnorm()`. It
measured FlyDSL's wrapper and I labelled the result quack's. Nothing in it
supported a claim about the cross-backend comparison, which is what the section
was written to make. The probe now records `quack_levels_measured: false` with
the import error, so the gap is visible in the artifact rather than absent
from it.

What is measurable here, bf16 forward, min-of-7-rounds-of-50, all rounds
retained in the JSON:

The `difference` column is the gap between the two round *intervals*, not
between their minima, and it is blank where the intervals overlap. A
point estimate computed across an overlap is not a measurement of anything.

| shape | `_launch` (what the harness times) | FlyDSL `rmsnorm()` | difference |
| --- | --- | --- | --- |
| 32768x4096 | 89.63 us [89.63, 91.48] | 89.93 us [89.93, 90.33] | **unresolved** (intervals overlap) |
| 8192x4096 | 20.50 us [20.50, 21.12] | 29.41 us [29.41, 30.63] | +8.91 us (+43.4%) |
| 1024x1024 | 12.21 us [12.21, 14.90] | 29.37 us [29.37, 30.42] | +17.16 us (+140.5%) |
| 256x512 | 11.78 us [11.78, 12.80] | 29.27 us [29.27, 30.21] | +17.49 us (+148.5%) |
| 64x256 | 11.44 us [11.44, 12.87] | 29.06 us [29.06, 30.30] | +17.62 us (+154.0%) |

**The cost is not a constant, and the earlier "~16.5 us" was fitted to the
three rows where it happened to hold.** The four resolved differences are 8.91,
17.16, 17.49 and 17.62 us. They saturate near 17.6 us at small shapes and fall
away as the kernel grows, which is the shape of a fixed host cost being
progressively hidden behind device work, not of a constant addend. Round-to-
round spread across the ten series is 0.40-2.68 us -- an earlier version of this
line said "0.40-1.25", which covers seven of the ten and omits the three widest
(`_launch` at 1024x1024, 2.68 us, at 32768x4096, 1.85 us, and at 64x256, 1.42
us), i.e. it quoted a range computed over a subset that excluded exactly the
series the next sentence rests on. The 8.91 us row still stands clear of it:
those two series are [20.50, 21.12] and [29.41, 30.63], which do not overlap, so
it is a real intermediate and not noise. Quoting a single number across the
range asserted an overlap model I had not tested; the honest summary is "up to
~17.6 us, and unresolved at 32768x4096".

Three corrections inside that sentence, all @Reviewer's and all mine to have
caught. The widest spread is 2.684821 us, which rounds to **2.68**, not the 2.69
I published -- I rounded up from a truncated read of my own output. The old
"0.40-1.25" range covers **seven** of the ten, not six; there are three series
above it, not two. And the row it describes stayed in the table above while
three paragraphs below it explained why the number was withdrawn: I wrote the
retraction and left the artifact standing, which is the same defect as the
inverted cache table earlier in this file -- the prose was right and the thing a
reader actually copies out was wrong. The table now carries intervals and the
word `unresolved` in place of the number.

**The 32768x4096 row does not support a number at all, and I stated one.**
"+0.30 us (+0.3%)" is a difference of two minima, and at that shape the two
round distributions overlap: `_launch` spans 89.63-91.48 us across its seven
rounds and `rmsnorm()` spans 89.93-90.33, so the `rmsnorm()` interval sits
*inside* the `_launch` interval. Nothing separates them. The honest reading is
that at 32768x4096 the level choice is **below this protocol's resolution** --
not that it costs 0.3%. Writing "inside the spread" and then quoting the point
estimate anyway is having it both ways: if it is inside the spread, the point
estimate is noise and does not belong in the table as a measurement.
@Reviewer raised this; the intervals are in the sidecar and they do overlap.

Worth stating what this does *not* touch, because the blocker is narrower than
the table: the other four rows separate cleanly. 8192x4096 is
[20.50, 21.12] against [29.41, 30.63], and the three small shapes are further
apart still. Only the largest shape is unresolved, which is also the only shape
where the argument needed it.

The consequence for the published tables is therefore unchanged in direction and
unproven in size at every shape. At 32768x4096 the level choice is unresolved
rather than small, so the large-shape results are **not** shown to stand by this
probe -- they are merely not shown to move by it, which is a weaker claim and
the one I should have made. Separating them needs a **paired, interleaved
delta** protocol: measure the two levels alternately within a round and take
the per-round difference, so the shared drift cancels. "More rounds" -- which
this paragraph used to offer as the alternative -- would not work, and
@Reviewer was right to strike it: appending rounds to a min/max range can only
ever widen it. The statistic has to change, not the sample count. At small
shapes the harness compares
a preallocated FlyDSL launcher against an allocating quack wrapper, and **how
much that is worth on the quack side has not been measured** -- it needs a CUDA
box, and until then no small-shape speedup from this harness should be quoted
without the level stated alongside it. The fix remains to time both providers
at the same level.

## The register-budget mechanism, now measured rather than assumed

The section above stopped at "occupancy collapse from per-thread live state is
*probably* right, but no register count has been read out of the compiled
kernel, so `MAX_N` is not raised on the strength of it." That number is
readable, and here it is.

FlyDSL's jit cache pickles carry the amdhsa kernel metadata verbatim --
`vgpr_count`, `sgpr_count`, both spill counts, LDS and scratch. The cache is
keyed on disk and survives the process, so forcing a fresh compile means
pointing `FLYDSL_RUNTIME_CACHE_DIR` at an empty directory; without that every
row reads "cache hit" and the probe silently measures nothing. bf16 forward,
weight only:

| N | vgpr | alloc (gran 8) | reg-limited waves/SIMD | occupancy bound (cap 8) | vgpr spill | sgpr spill | scratch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 20 | 24 | 21 | **8** | 0 | 0 | 0 |
| 2048 | 20 | 24 | 21 | **8** | 0 | 0 | 0 |
| 4096 | 36 | 40 | 12 | **8** | 0 | 0 | 0 |
| 8192 | 60 | 64 | 8 | 8 | 0 | 0 | 0 |
| 16384 | 94 | 96 | 5 | 5 | 0 | 0 | 0 |
| 32768 | 156 | 160 | 3 | 3 | 0 | 0 | 0 |
| 49152 | 230 | 232 | 2 | **2** | 0 | 0 | 0 |
| 57344 | 264 | 264 | 1 | **1** | 0 | 0 | 0 |
| 65536 | 300 | 304 | 1 | 1 | 0 | 0 | 0 |

Bit-identical across two fresh processes, and reproduced again from a
committed generator, `AI/probe_rmsnorm_vgpr_by_n.py`: every `vgpr_count` and
`sgpr_count` above comes back identical, nothing spills anywhere, and the
sidecar now carries host, device UUID, commit, source hashes and the shipped
`MAX_N` alongside. Raw at `AI/data/rmsnorm_fwd_vgpr_by_n.json`. The counts are
parsed from the `gpu.kernel_metadata` attribute dictionary in FlyDSL's jit
cache pickle, and each row asserts that a compile actually happened, so a
stale cache entry cannot be reported as a fresh measurement.

**The regenerated sidecar records one thing the old one could not: `agpr_count`.**
It is 0 everywhere up to N=49152, then 8 at 57344 and 44 at 65536. The AGPRs
appear *exactly* at the cliff, which the old artifact could not have told
anyone.

~~On gfx950 the VGPR and AGPR banks share one 512-slot file per SIMD, so a
nonzero AGPR count is not free — the honest allocation at 65536 is 344, not
300.~~ **False, and it outlived four commits that were each correcting it.**
`.vgpr_count` already *is* arch-VGPR + AGPR, so 300 is the total and 344
double-counts. This sentence survived `c5d879b`, `6d4ab9c`, `afdf045` and
`d7fe67a` — including the commit whose message was "the wave64 units story is
wrong; AGPRs say so" — because each time I fixed the *field* that stated it and
not the prose three hundred lines away. @Autotune killed it a fifth way, from
the ELFs directly: `accum_offset = 256` on all five kernels dumped, and
`vgpr_count − accum_offset == agpr_count` on 5/5. Retracting a claim in one
place while it stands in another is not retracting it.

I wrote here that this "happens not to move either bound," on the grounds that
both rows are at 1 wave/SIMD with or without the AGPRs. That is true of 65536
and **false of 57344, which is the row that matters**: 256 VGPRs would allow 2
waves, and it is precisely the 8 AGPRs that take the allocation to 264 and the
capacity to 1. @Reviewer refuted the sentence once from the static side, and the
intervention has now refuted it a second time by measurement — remove those 8
registers with `waves_per_eu=2` and the capacity comes back to a measured 1.937
(see the intervention section). A line dismissing a quantity as inconsequential
turned out to be describing the cause of the whole cliff.

**The first three rows used to read 21, 21 and 12 waves/SIMD, which the hardware
cannot do.** `floor(512 / vgpr_alloc)` is the register file's limit and I stored
it as though it were occupancy. This host reports `max_waves_per_simd 8` for
every GPU node (`/sys/class/kfd/kfd/topology/nodes/*/properties`, with
`simd_per_cu 4`, `wave_front_size 64`), so the correct capped sequence is
8, 8, 8, 8, 5, 3, 2, 1, 1. @Reviewer caught it and the arithmetic reproduces
exactly. The sidecar now stores both columns under honest names --
`waves_per_simd_register_limited` and `occupancy_upper_bound_waves_per_simd` --
rather than one column whose name claimed more than its formula computed.

Two things follow, one reassuring and one not.

The conclusion below survives, because the cliff is at 49152 -> 57344 where the
cap does not bind: 2 -> 1 either way. Capping only changes rows at and below
N=8192, which are not where the argument is made.

What does not survive is the shape of the curve as I drew it. The **bound** is
now flat at 8 from N=1024 through N=8192, not falling 21 -> 12 -> 8. So there is
no gradient in the bound at all across the small and mid range, and any reading
of those rows as "occupancy is already declining by 4096" was an artifact of the
missing cap. It also means the register file is not the binding constraint until
N=16384; below that the bound is set by the hardware maximum, and what the
kernel actually achieves there is not in this table.

And the deeper problem was that **none of this column was measured.** It is
arithmetic on a register count -- an upper bound that ignores workgroup slots,
LDS and barriers. Calling a derived bound "waves/SIMD" is how the uncapped
version survived review in the first place: a measured number would have been
checked against the hardware maximum, and a derived one was not.

**That measurement now exists.** `AI/probe_rmsnorm_measured_occupancy.py` reads
rocprofv3's `MeanOccupancyPerActiveCU`; raw at
`AI/data/rmsnorm_fwd_measured_occupancy.json`. At the cliff's own m=4096:

| N | vgpr | alloc | computed bound | **measured waves/SIMD** | measured/bound | bandwidth % (eager) |
|---|------|-------|----------------|-------------------------|----------------|---------------------|
| 32768 | 156 | 160 | 3 | 2.783 | 0.93 | -- |
| 40960 | 226 | 232 | 2 | 1.960 | 0.98 | -- |
| 49152 | 230 | 232 | 2 | 1.959 | 0.98 | 76.6 |
| 57344 | 264 | 264 | 1 | **1.000** | 1.00 | 38.1 |
| 65536 | 300 | 304 | 1 | **1.000** | 1.00 | 39.2 |

Each row now also carries `counter_samples` — all three dispatch readings,
unrounded — and the `dispatch_ids` they came from, so the published median is
recomputable and the positional width recovery is checkable from the file.
@Reviewer's objection was that publishing only a rounded aggregate is fail-open:
a number that cannot be recomputed from anything in the artifact is a claim
dressed as data. It also makes the cliff rows' stability visible rather than
asserted — 57344 reads 1.0004 on all three dispatches, identically, while
32768 spans 2.7803–2.8149. The sidecar records `rocprofv3_version` too
(1.1.0, git `fc0010cf`), which matters here more than usual: the entire VGPR
relation below turns out to be a property of that specific build.

**That bandwidth column used to be three hand-copied literals, and they were
not one series.** 77.5 and 37.8 were the cliff's *graph* numbers while 39.2 was
its *eager* one — @Reviewer caught the mix. Nothing read them quantitatively, so
no conclusion moves, but a three-entry table silently spanning two timing
regimes is precisely the sort of thing that gets picked up later as if it were
comparable. The probe now reads the column out of the cliff sidecar and names
the regime in every row (`bandwidth_regime`), so the mix is no longer possible
to make. The deeper fault was transcription itself: three numbers copied by hand
into a constant cannot be checked against anything.

There are now five runs of this sweep: one on device 6 and four on device 5,
taken while @Reviewer was verifying on device 6. I first summarised the spread
as "reproduces across GPUs to within 0.6% on its worst row", and @Reviewer
pointed out that 0.6% covers only the boundary sweep — the discriminating sweep
at m=16384 has a row that moves 2.0%, more than three times as much. He is
right, and the extra runs make a further correction possible:

**The spread is not a device effect at all.** On the worst row (discriminating,
N=8192) the four device-5 runs differ *from each other* by 3.06% — which is the
entire spread across all five. Adding a second GPU adds nothing to the
dispersion that repeating on one GPU does not already produce. So "reproduces
across GPUs to within 0.6%" attributed to hardware what is simply counter
variance. Both the figure and the reproduction were real; the attribution was
invented, and that is the same defect as everything else in this section — a
number agreeing with a story for a reason other than the one given.

What the spread does track is slack — how far a row sits below its own
constraint:

| row | measured/bound | spread over 5 runs |
|---|---|---|
| boundary N=57344, 65536 | 1.000 | 0.00% |
| discriminating N=49152 | 0.986 | 0.46% |
| boundary N=49152 | 0.979 | 0.45% |
| boundary N=32768 | 0.928 | 0.89% |
| discriminating N=32768 | 0.877 | 0.94% |
| discriminating N=8192 | 0.784 | 3.06% |

Rows pinned against their bound do not move at all; rows with slack move by up
to 3%. It is not monotone (`N=16384` at ratio 0.797 spreads only 0.47%), so this
is a tendency and not a law. The part that matters: **the two cliff rows are at
ratio 1.000 and are bit-identical across all five runs and both devices** —
1.0004 and 1.0003 every time, and now visibly identical across the three
dispatches within each run as well. The conclusion rests on the two most stable
rows in the set.

**Occupancy does halve at the boundary, and it is now observed rather than
derived.** The 2 → 1 step falls between 49152 and 57344, the same edge as the
bandwidth drop, and the two cliff rows sit at 1.0004 and 1.0003 against a bound
of 1 in every run.

**How well measured tracks the bound elsewhere I stated too well, twice.** I
wrote "within 7% everywhere and within 2% from 40960 up". @Reviewer checked the
prose against its own table and found both false: the discriminating sweep is
far outside 7%, and 49152 is outside 2%. What the rows support is in
`agreement_with_bound.groups` in the sidecar — worst deviation per group, with
the row it came from:

| where | worst \|measured/bound − 1\| |
|---|---|
| the two cliff rows (57344, 65536) | **0.04%** — the claim's own rows |
| register-bound rows at m=4096 | 8.08% (N=32768) |
| register-bound rows at m=16384 | 20.43% (N=16384) |
| cap-bound rows (4096, 8192) | 22.45% (N=8192) |

**That table is transcribed from a field, and the reason it is a field is that
the paragraph above it was wrong for the same reason it was correcting.** When
@Reviewer caught the original over-statement I replaced it with hand-copied
figures — 23.9% and 2.45% in the prose, 21.6% and 7.3% in the table beneath it.
Two different worst-case numbers for the same rows, in adjacent paragraphs of
one file, because both were typed rather than read. Re-running the probe just
now moved them again (8.08% and 22.45%), which is exactly the drift that made
them disagree in the first place. So the correction to a transcription fault
was itself a transcription fault, and it took a third pass to stop patching the
number and fix the mechanism: the groups are now computed in
`_agreement_summary()` and emitted, so a stale figure here is checkable against
the sidecar instead of aging silently. Same fault @Reviewer found in the three
bandwidth constants, and I did not generalise it when I fixed those.

The pattern is the one the spread table shows: agreement is tight where the
constraint binds hard and loose where it does not, since measured occupancy is
an average over active CUs and over the kernel's life. That is a defensible
reading. "Within 7% everywhere" was not — it was a summary statistic quoted
from the half of the data that supported it. Note the cliff rows are the one
group that does *not* drift between runs: 0.04% on all six.

The step is a register-file threshold, not a width effect: `vgpr_alloc` crosses
256 of the 512-entry budget there, so `floor(512/232)=2` becomes
`floor(512/264)=1`. N itself rises smoothly through it -- 49152 → 57344 is a
factor of 1.17 -- while occupancy halves.

By itself this is still a coincidence of two edges: occupancy halving and
bandwidth halving at the same N does not establish that the first causes the
second. So the intervention it called for has now been run.

### The intervention: move occupancy at fixed N, watch bandwidth

`AI/probe_rmsnorm_occupancy_intervention.py`, raw at
`AI/data/rmsnorm_fwd_occupancy_intervention.json`. The lever is
`--amdgpu-waves-per-eu`, which flydsl's ROCm backend takes as a `waves_per_eu`
compile hint. At N=57344 it pushes the allocation from 264 VGPRs to 256 --
across the boundary, at a width that has not changed.

| hint | vgpr | agpr | alloc | measured waves/SIMD | vgpr spills | scratch | bandwidth % |
|------|------|------|-------|---------------------|-------------|---------|-------------|
| none | 264 | 8 | 264 | 1.000 | 0 | 0 | 38.1 |
| 2 | 256 | 0 | 256 | **1.937** | 8 | 36 | **52.7** |
| 3 | 168 | 0 | 168 | 2.796 | 97 | 392 | 43.2 |
| 4 | 128 | 0 | 128 | 3.687 | 137 | 552 | 34.4 |

Control, N=49152 (already at 2 waves, so hint=2 has nothing to move):

| hint | vgpr | alloc | measured waves/SIMD | spills | bandwidth % |
|------|------|-------|---------------------|--------|-------------|
| none | 230 | 232 | 1.960 | 0 | 76.7 |
| 2 | 228 | 232 | 1.952 | 0 | 77.0 |
| 3 | 168 | 168 | 2.820 | 61 | 46.5 |
| 4 | 128 | 128 | 3.884 | 101 | 43.2 |

**Restoring occupancy at the cliff recovers a substantial part of the lost
bandwidth: 38.1% → 52.7%.** The control's hint=2 row is the load-bearing one --
where the hint cannot move occupancy it does not move bandwidth either (76.7 vs
77.0), which is what separates "occupancy drives bandwidth here" from "this
compiler flag is generically good". The probe asserts the control's allocation
is unchanged at that hint and aborts if it is not.

**The eight spills are the eight AGPRs.** @Reviewer noticed that at the first
step `agpr_count` goes 8 → 0 exactly as `vgpr_spill_count` goes 0 → 8, with
scratch 0 → 36 B/lane, or 9 dwords. Those are the same eight registers moving
from AGPRs to scratch, not eight newly created spills, so the confound in the
first step is smaller than the raw count suggests. It also closes a loop that
had been left open in two separate sections: the static reading of the artifact
said those 8 AGPRs are what push the allocation to 264 and the capacity from 2
to 1, and this row removes them and measures the capacity coming back. The
arithmetic prediction and the intervention are the same claim, made twice, and
the notes previously had them pages apart with nothing pointing between them.

**The lever is not clean, and the reading depends on saying how.** It buys
occupancy with spills, so the first step is a two-variable change. What makes
it evidential is the direction of the disagreement: bandwidth improves *while*
the spill count goes 0 → 8, i.e. the confound pushes against the hypothesis and
loses. A clean lever would move occupancy at constant spills; I do not have one.

**And the reversal at hints 3 and 4 was over-read, because it had no control.**
I had written that the later steps "track spills rather than occupancy", which
frames the reversal as a property of the width that was over the register
boundary. @Reviewer pointed out the control only ran none/2, so there was no
width where spills should not matter to read the reversal against. Running the
full ladder on the control answers it, and not in my favour: at N=49152 hint=3
takes bandwidth from 76.7% to 46.5% and hint=4 to 43.2%. **High settings of
this flag are harmful at both widths**, so the 3/4 rows show that more occupancy
is not always faster, and nothing more specific than that. The main claim is
untouched — it rests on the none→2 step, where the control is flat and the
treatment is not — but the mechanism I attached to the reversal was an
interpretation dressed as an observation, and it took the missing control to
see that.

So the claim is directional, not quantitative, and occupancy is not the whole
story. **That last point does not need a cross-width comparison to make**, which
is how I had been making it. There is a pair in the sweeps at the *same*
occupancy and different N:

| | measured waves/SIMD | bandwidth % |
|---|---|---|
| N=57344, hint=2 | 1.937 | **52.7** |
| N=49152, hint=none | 1.960 | **76.7** |

Occupancy matched to about 1%, bandwidth differing by a factor of 1.45.
Restoring occupancy buys back 38% of the cliff and no more. @Reviewer found this
pair; it is emitted as `equal_occupancy_pair` with its own occupancy separation
alongside, so the reader can check the axis that is supposed to be held fixed
actually is. The sharpest single row is blunter still: **the highest occupancy
in the treatment table (3.687 waves/SIMD, hint=4) has its worst bandwidth
(34.4%), below the unhinted kernel running at 1.000.**

What remains unestablished is the *magnitude* of the occupancy contribution, and
"latency-hiding-bound" is still an assumption about the kernel rather than a
finding.

**Provenance, and a claim of mine that a third die falsified.** I published, and
told the team, that the treatment ladder "reproduces across two GPUs — device 6
in `276398f` and device 5 here — to within 0.2% on every bandwidth figure". That
was true of 6 and 5 and it is false in general. Device 4 is faster on *every*
row. Six runs — two each on dies 4, 5 and 6, all from one generator revision —
give a worst between-die bandwidth gap of **11.1%** against a worst within-die
range of **0.76%**, so this is a device effect and not run-to-run noise. The
data is in `AI/data/rmsnorm_fwd_occupancy_intervention_cross_device.json`,
assembled by `AI/assemble_intervention_cross_device.py`.

**The first version of that sidecar was hand-assembled and @Reviewer requested
changes on it; every structural objection was right.** No assembler, so the
summary could not be regenerated or checked. No hashes and no raw inputs. And
two faults that were not bookkeeping:

- *Mixed provenance on the axis being measured.* Three of the four inputs came
  from generator `2edfda85`; the fourth — device 5, run 0 — came from
  `3fcf0041`, a revision never committed to this repository. So "two runs per
  die" was one die's pair spanning two generators against another die's matched
  pair. The asymmetry sat exactly on the comparison's own axis. The assembler
  now refuses inputs spanning more than one generator revision.
- *A reducer chosen after seeing the data.* I wrote "occupancy matches across
  dies to within 0.6%". That holds only under closest-endpoint — the smallest of
  the pairwise gaps. On means it is **1.22%** and on full range **1.47%**. I did
  not state the reducer because I did not notice I had picked one. All three are
  now computed and reported on every quantity, with the primary named up front.

The equal-occupancy point survives that unchanged, and the reason is worth
stating: it only requires occupancy to agree *better* than bandwidth, which
holds under all three reducers (1.22% vs 11.1% on means). A claim that depends
on which reducer you pick is a claim about the reducer.

Two GPUs agreeing is one comparison, not a property of the hardware, and I had
already been caught this session generalising exactly this way — the *reverse*
direction, calling a same-device spread a cross-device reproduction. Having
corrected that, I went on to quote "reproduces across two GPUs" as a provenance
guarantee in the very next section, and in the message I sent the team an hour
ago. Same fault, opposite sign, one file apart.

I proposed a rule off the back of it — "any *reproduces across X* claim needs
N≥3 on X" — and @Reviewer declined it in that form, correctly. N≥3 is a
collection floor, not a licence: three dies chosen because they were idle is not
a sampling design, and "observed on devices 4, 5 and 6, two runs each, under
this reducer" is the strongest form these numbers support. The rule that
generalises is the one the eager/graph check produced: **a comparison must
discriminate a gap larger than the confounds it cannot see.** That subsumes the
device version and it is what the ratio table below applies.

~~The gap is not the streaming ceiling. Measuring `two_read_one_write` on each
die: device 4 is **6.004** TB/s, device 5 **6.086**, device 6 **6.074**. Device
4 is the *slowest* of the three at pure streaming and the fastest at this
kernel, so a per-device ceiling would widen the gap rather than close it.~~
**Retracted (@Autotune): two of those three numbers were never measured, and
the third is the shared denominator wearing a per-die label.** The paragraph
said "measuring `two_read_one_write` on each die", which asserts three
measurements. There is one. `AI/data/rmsnorm_fwd_width_cliff.json` at the
parent of the commit that wrote this claim held a single `ceiling_probe`, taken
on device 6 (uuid `66636163…`, `HIP_VISIBLE_DEVICES=6`), reading
`6.074573403174727` — which *truncates* to the 6.074 published as device 6 and
*rounds* to the 6.075 published four pages below as the shared, device-neutral
denominator. Same probe, same five samples, two roundings, two roles. Device 5's
6.086 matches nothing: that die's probe was measured 47 minutes **after** this
claim was committed and reads 6.08505 → 6.085. Device 4 has no
`two_read_one_write` measurement at any revision — an exact-token scan of all
3379 blobs in the full history finds 6.004, 6.086 and 6.074 in exactly four
paths each, and those four are this file, two prose strings
(`AI/assemble_intervention_cross_device.py` L376-385 and the
`AI/probe_rmsnorm_occupancy_intervention.py` docstring), and the `unexplained`
field the first of them emits into
`AI/data/rmsnorm_fwd_occupancy_intervention_cross_device.json`. All four are the
same sentence; none is a measurement. Scanning by *value* rather than by token —
every numeric leaf in all 157 JSON blobs in history, matched to 3 dp by rounding
or truncation — adds nothing outside `raw_dev5/`, which is device 5. Every
`cross_device_runs/dev{4,5,6}_run{0,1}.json` carries the identical
`ceiling_tbs: 6.075`, hardcoded as `CEILING_TBS = 6.075`; there is no per-die
ceiling anywhere in this repository, so no run could have produced one.

The conclusion fails on its own terms even if the numbers are granted. Eight
lines above, this section adopts the rule **"a comparison must discriminate a
gap larger than the confounds it cannot see."** The claimed device-4-to-device-5
gap is 1.366%. The same probe at the same 512 MiB size on a *single* die, with
allocation state held fixed, spans up to 2.110% across 80 readings in
`AI/data/copy_placement_draws/raw_dev5/` (per-level ranges 0.99 / 1.39 / 2.11 /
1.85%), and all three claimed values fall inside that one die's observed
interval [5.9695, 6.1160]. A one-shot five-sample probe cannot resolve 1.4%
here. So "device 4 is the slowest at streaming" is not a finding, and neither is
its negation — the ordering is unidentified, and with it the argument that
per-device normalisation would widen the gap rather than close it.

What survives is narrower and does not need per-die numbers: the ceiling used
throughout is one device-6 measurement applied to all three dies, so
`bandwidth_pct_of_ceiling` is comparable *across* dies by construction and the
between-die bandwidth gap is not an artifact of different denominators. Whether
a genuine per-die ceiling would widen or close that gap is untested. Idle
sclk/mclk/fclk/socclk are identical across the three and junction temperatures
sit within 2 °C. I have not chased it further: it bears on no conclusion here,
and inventing a mechanism for it is how the last two retractions started —
which is precisely what the retracted sentence did, by asserting a mechanism
("not the streaming ceiling") on three constants, two of which had no
measurement behind them.

**"Every claim this experiment makes is a ratio, and the ratios hold across
dies" was my reassurance, and it needed the same scrutiny as the thing it was
reassuring about.** I published four ratios as one three-significant-figure
number per die. With six runs and the spread computed, here is what those digits
are worth — each ratio in its own absolute units, with the within-die scatter
next to the between-die gap:

| ratio | dev4 | dev5 | dev6 | worst within-die | separates | gap ÷ within |
|---|---|---|---|---|---|---|
| treatment none→2 lift | 39.67 | 38.78 | 38.69 | 0.29 | 4 from 5 and 6 | 3.3× |
| high-occ margin (pts) | **1.96** | 10.68 | 10.70 | 0.16 | 4 from 5 and 6 | **54×** |
| control none→2 | 0.87 | −0.05 | 0.20 | 0.79 | only 4 vs 5 | 1.2× |
| equal-occupancy ratio | 1.4545 | 1.4547 | 1.4517 | 0.0074 | **nothing** | 0.4× |

*Verified and one column added (@Autotune).* Every figure above reproduces
exactly against `ratios/*/per_device_mean` and `worst_within_die_spread_abs` in
`AI/data/rmsnorm_fwd_occupancy_intervention_cross_device.json` — 16 values, 4
spreads, and all 4 `separates` verdicts, which agree with the artifact's own
`separated_pairs`/`indistinguishable_pairs`. What the `separates` column does
not carry is *how much*, and that turns out to matter, because the criterion
behind it is far more permissive than the word suggests. It fires when a pair's
two-point ranges are disjoint **and** the mean gap exceeds the worst within-die
spread; simulated under an exchangeable null where the dies are identical, that
combination still fires **25.8%** of the time (26% gaussian, 26% uniform, 18%
lognormal, 20% heavy-tailed). It is not a 5%-level test and was never presented
as one, but "separates" reads like a verdict, so the ratio column is the honest
companion: at 54× the margin row is unambiguous, at 3.3× the lift is
comfortable, and at **1.2× the control row is inside the criterion's noise
floor** — its `separates: only 4 vs 5` is close to what an all-null design
produces by chance.

Two structural notes on that criterion, neither a defect but both worth stating
next to a table people will quote. First, the threshold for *every* pair is the
worst within-die spread over **all three** dies, so a die outside a pair sets
that pair's bar: `control none→2` is judged 4-vs-5 against die 6's scatter of
0.79, and against the pair's own scatter the gap is 2.0× rather than 1.2×. That
is the conservative direction, which is the right way to be wrong. Second, with
three dies and two runs each there are only 15 distinct ways to partition six
runs into three pairs, so the smallest permutation p this design can produce for
"one die is the odd one out" is **3/15 = 0.20**. The margin row's 54× separation
and the equal-occupancy row's 0.4× are equally unable to clear 0.05 — no
cross-die claim from a 3×2 design can, however large the effect. The right
reading is that these are *descriptive* separations with their effect sizes
stated, which is what the ratio column now makes explicit, and the sentences
below already treat them that way.

The one I quoted most confidently is the one that resolves nothing. I published
the equal-occupancy ratio as "1.455 (dev5) vs 1.453 (dev4)" as though the
agreement were evidence; across three dies the largest gap between any two is
0.0031 against a within-die scatter of 0.0074. **The third digit was never
real.** The honest statement is "1.45 on all three dies", and that is still
enough for the argument it serves, which only needs the factor to be ~1.45 and
not ~1.

`control_none_to_2` is worse in a different way: its estimates straddle zero
(−0.19 to +1.10), so it supports a *bound* and not a value. "The control is
flat" is true — every estimate on every die is under 1.1% — and "0.37 vs 0.44"
was a pair of digits dressed up as a matched comparison. @Autotune caught this
independently and his framing is the one to keep: a reducer chosen after the
fact, presented as a measurement.

The two that do separate device 4 from the others, cleanly, are the lift and the
high-occupancy margin. Registers and spills are bit-identical on all six runs.

**@Autotune also found that the high-occupancy margin is die-dependent, and he
is right.** The claim is that the *highest*-occupancy row in the ladder (hint=4,
3.7 waves/SIMD) is also its *worst* bandwidth, below the unhinted kernel at 1.0
wave/SIMD. The sign holds on all three dies, so "more occupancy is not always
faster" stands everywhere. But the margin is 10.7 points on dies 5 and 6 and
**1.96 on die 4** — a 5.4× collapse, landing at the same order as the cross-die
gap itself. On device 4 that row is directional only. It had been emitted as a
flat sentence in a field of its own, which read as a fixed fact; it now carries
its per-die margins.

It does hand the equal-occupancy point a third instance for free: occupancy
matches across dies to 1.22% on means while bandwidth differs by 11.1%, with
kernel, registers and spills all held exactly fixed. Occupancy does not
determine bandwidth even across three copies of the same silicon running the
same code object.

**The timing regimes line up — and the argument I used to show it was worth
less than the one-line source check that settles it.** The intervention times
with cuda events over a plain Python loop; `grep -c CUDAGraph
AI/probe_rmsnorm_occupancy_intervention.py` returns **0**. There is no capture
in the file, so it is eager *by construction*, not by numeric coincidence. That
is the whole proof, it is exact, and it was available without running anything.

Instead I argued it empirically: the unhinted rows land on the cliff sidecar's
eager column to +0.09% at 57344 and −0.21% at 49152, "across two commits, two
probes and two devices." The device-4 result shows why that reasoning was
unsound even though its conclusion is right. **The eager/graph separation the
argument discriminates is 0.93% at 57344 and 1.19% at 49152. The cross-die
effect on those same rows is 2.3–3.3%.** The discriminator is smaller than a
confound the test cannot see — and it is not hypothetical: run the identical
comparison with the device-4 numbers and at 49152 the intervention lands closer
to the *graph* column (+2.10%) than to eager (+3.31%), which would "establish"
the opposite regime with equal confidence. The cliff sidecar was measured on
device 6, the intervention on device 5, so the two were never a
same-die comparison in the first place.

So: a check that passed for a reason other than the one it documents, again.
It agreed because the regimes genuinely are the same, but it would have agreed,
or disagreed, on a die swap alone. Keep the source fact; the numeric agreement
is a weak corroboration whose resolution I have now measured and it does not
support the weight I put on it.

What survives independent of all this: the intervention's bandwidth percentages
and the cliff's are the same measurement of the same thing, and the ceiling
denominator is shared — 6.075 TB/s from the cliff sidecar's
`two_read_one_write` probe, not a copy kernel, which on this part reads
MALL-inflated. Both are structural, not numeric.

**Two traps had to be cleared to get this number, and both are worth recording
because either would have produced a confident wrong answer.**

*rocprofv3's VGPR column is a lossy function of the artifact's.* At N=49152 the
artifact says 230, rocprof says 116. The exact relation is
**`rocprof == vgpr_alloc_wave64 / 2`** — half the *granule-8 rounded
allocation*, not half the count — on 10 of 10 widths.

**Getting to that one line took three wrong answers, and the sequence is more
instructive than the result.**

1. I wrote the relation as `roundup(ceil(v/2), 4)` and explained it as wave64
   architectural VGPRs against 32-lane physical register-file entries — a unit
   difference, neither source wrong. Evidence offered: 10 of 10 widths, five
   held out.
2. @Reviewer rejected the units story and proposed an incomplete
   ROCProfiler-SDK decode of the gfx950 code object. I tested it with AGPRs — a
   conversion scales them, a decoding gap drops them — found artifact 8 and 44
   against rocprof's `Accum_VGPR_Count = 0`, and conceded his hypothesis was
   better supported.
3. @Autotune pointed out that this was never an empirical question at all.
   **`roundup(ceil(v/2), 4)` is identically `ceil(v/8)*4` for every integer
   v.** My formula could not have failed on any width. "10 of 10, five held
   out" was reporting an *identity* as a confirmed prediction.

So the headline defect of this whole file — a check that passes for a reason
other than the one it documents — appeared again, this time **inside the
apparatus built to guard against it**. A held-out set is the strongest form of
evidence I know how to construct, and it is worth exactly nothing against a
tautology. The thing that would have caught it is not more measurement but one
line of algebra, available from the first day: *before quoting a relation that
holds everywhere, check whether it could have failed anywhere.*

@Reviewer's mechanism is the correct one: ROCProfiler-SDK 1.1.0 has no gfx950
accumulator decoder, so gfx950 falls through to `(PGM_RSRC1+1)*4` with
`Accum_VGPR_Count = 0`, while LLVM encodes the total with granule 8. That
composition produces the observed numbers exactly.

**The tell that settles it needs no ROCProfiler source at all, and it was in my
own sidecar the whole time: the map is not injective.** Artifact 226 and 230 are
two different allocations, and rocprof reports 116 for both:

| rocprof | ← artifact | if it were really v/2 |
|---|---|---|
| 12 | 20, 20 | 10, 10 |
| **116** | **226, 230** | 113, 115 |

~~A unit conversion is order-preserving and invertible. Quantization is
neither.~~ **Over-stated, and @Reviewer flagged the exact wording twice before I
accepted it.** The collision proves the column is *lossy*; it does not exclude a
unit conversion, because a conversion that rounds is also non-injective. I used
a valid observation to rule out more than it can.

What actually settles it is the encoding, and @Autotune read it off the ELFs:
gfx950 stores the VGPR total with granule **8** in
`compute_pgm_rsrc1[5:0]`, so the allocation is `(field+1)*8`; a decoder assuming
granule 4 computes `(field+1)*4` and recovers exactly half. That reproduces
rocprof's `VGPR_Count` on 20/20 rows. It is a decoder question, settled from the
bits, not an inference from the shape of the map.

The collision keeps a narrower job: it is why the fit/held-out split had no
verification value — artifact 226 and 230 share `field=28`, so no split of these
widths could have distinguished anything. The probe asserts it on that basis.
The same fact explains why `512 // (2 * rocprof)` reproduces the correct bound on
all ten rows — `2 * rocprof` **is** the allocation, so the cross-check was
re-deriving a number the artifact already stated. It is retained as a toolchain
regression guard and nothing more.

**One nearby claim of mine was double-counting, and it was wrong for a day.**
I wrote that vgpr+agpr = 272/344 at 57344/65536 "still gives bound 1", treating
the two artifact fields as additive. @Reviewer flagged that amdhsa
`.vgpr_count` already includes AGPRs. The rows confirm it independently: at
N=65536 rocprof reports 152 = `ceil(300/8)*4`; a separate 44 AGPRs would make
the encoded total 344 and the reading 172. Nothing downstream moves — the bound
was computed from `vgpr_count` alone throughout, which is the correct total —
but the reassurance I offered was arithmetic I had not checked.

All register arithmetic here uses the artifact numbers, which is the pair the
hardware behaves like: on the three widths where the two candidate readings
predicted *different* occupancies (16384/32768/49152 at m=16384) the artifact
predicts 5/3/2 and the halved values predict 8/6/4, against measured
3.98/2.63/1.97.

The `fit`/`held_out` labels survive in `vgpr_relation_audit` only because
earlier commits refer to them; they have no evidential content, and the sidecar
now says so in the field itself.

*A register bound is invisible unless it is the binding constraint.* My first
occupancy reading was taken at m=1024, where the grid supplies only 4
waves/SIMD -- any register limit of 4 or more is unobservable there, and the
reading would have "confirmed" whichever bound it was compared against. Every
row now carries `grid_supply_waves_per_simd` and a `register_bound_is_binding`
flag. This is the same defect class as the starvation control that turned out
to be measuring the host: **a number that agrees with your hypothesis for a
reason other than the one you think.** The new rule from the floor work
generalises to it -- before trusting a bound, check what the measurement reads
when the bound is not the constraint.

**And then the guard itself had the defect it was written to catch.** The first
version tested the *already-capped* bound against the other constraints, which
is trivially true whenever the hardware cap binds -- so it reported all ten rows
as register-bound, including N=4096 and N=8192, whose register limits are 12 and
8 against a cap of 8. Those two are cap-bound. The flag now tests the uncapped
register limit, and a `limiting_constraint` field names which of registers /
hardware cap / grid supply is actually smallest; the two rows correctly read
`hardware_cap` and `binding=False`. Nothing downstream moves: the three
discriminating rows (16384/32768/49152) were always the register-bound ones and
the cliff rows have limits 5/3/2/1, far under both other constraints. But a
guard that says True when it should say False is worth less than no guard,
because it launders exactly the assumption it was supposed to test. @Reviewer
caught this one; I did not.

A third trap the probe caught on its own: flydsl memoizes compilation
*in-process* as well as on disk, so pointing `FLYDSL_RUNTIME_CACHE_DIR` at an
empty directory is not sufficient when one process runs two sweeps over
overlapping widths. The second sweep wrote no pickle and the row would have
reported a stale kernel's registers. Register extraction now runs in a
subprocess per sweep, and each row still asserts that something was compiled --
both guards are needed, either alone lets a hit through.

Two things the table does settle. **Nothing spills, anywhere** -- not at 65536,
not at 8192. So "register budget" was the right family and "spill cliff" would
have been the wrong name for it; whatever the cost is, it is not scratch
traffic. And the growth is smooth and roughly linear in N, about 4.6 VGPRs per
1024 columns, with no discontinuity at 8192 -- which is the second, independent
confirmation that **`MAX_N = 8192` is not where the hardware objects**. At 8192
the kernel is at 60 VGPRs, which is the last row whose register allocation
still *permits* all 8 wave slots -- whether it fills them is not measured here.
This line previously said "nowhere near any limit" (written while the cap was
missing) and then, briefly, "at full occupancy", which swapped one unmeasured
claim for another. The point stands either way: at 8192 the register file has
not yet started restricting anything, and it is the *first* N below which it
has slack it cannot use.

The comment on `MAX_N` claims the constant *is* "the register budget expressed
as a row length." It is now fair to say that is wrong twice over: the register
budget expressed as a row length is about 49152, and 8192 is 6x below it.

**I am still not raising it in this commit**, for a reason that is now specific
rather than precautionary. Everything above is the forward kernel with weight
only. The backward is a different kernel with more live state per thread, the
per-head and bias variants add more, and the cap is shared by all of them. The
right change is a per-variant cap derived from the measured VGPR curve, and the
measurement to justify it is the same probe run across the backward and the
operand combinations -- which is a bounded piece of work, not a guess. What
this commit buys is the register curve itself, measured and reproducible, and
the method for setting the constant honestly.

(Written when the section was new, this paragraph ended "neither residency nor
causality has been measured". Both have been since, in the sections above:
residency by `MeanOccupancyPerActiveCU` at the boundary, and causality
directionally by the `waves_per_eu` intervention. @Reviewer flagged that the
sentence had been left contradicting the same file. The conclusion it supports
is unchanged — the cap is still not raised here, for the per-variant reason
stated above and not for want of a mechanism.)

###### Deferred: probe teardown, and why the fix is not in this commit

`AI/probe_rmsnorm_stage_stubs.py` allocates its tensors and lets process exit
reclaim them. On a box with eight shared MI355X and a teammate whose Experiment
No.001 gates on a *strict whole-machine idle window*, that is not good enough: at
18:13 UTC @CrossVendor recorded GPU5 holding 33,695 MiB and counted it as
contention, and that allocation was mine — released by 18:14, one minute before
my own sample showed seven of eight cards back at the 284 MiB floor. Two agents
measured the same machine a minute apart and disagreed completely, and the
disagreement was me.

The fix is three lines: `del`, `torch.cuda.empty_cache()`, and a `mem_get_info`
report so the release is *witnessed* rather than assumed. I wrote it, then
reverted it. Editing the generator changes its `source_sha256_16`, and the
artifact records that hash, so the honest sequence is edit-then-rerun — and the
rerun would occupy a GPU inside the window I had just told @CrossVendor I would
stay out of. Shipping the edit without rerunning would leave the artifact
claiming a generator that no longer exists, which is the exact defect class this
file is full of corrections for. So the tree is clean, the hash matches, and the
teardown lands after his rerun.

Also recorded because I got it wrong in the telling: I first told him the 33 GiB
was the stage-stub probe "not exiting cleanly". I had a probe *and* a pytest gate
running in that period and kept no record distinguishing them; both processes
are gone and it is not recoverable. What is established is that the memory was
mine and that it was released. Which process held it, I don't know, and I should
not have named one.

###### A provenance column that is hardcoded, and a cache key with no device in it

Not my artifact, but the same defect class and it bears on the No.001 rerun, so
it is recorded here. @Autotune found (`e4bfeb34`) that in the committed
`data/quack-rmsnorm-hopper/` archive **most `quack_tuned` winners were never
searched in the run that reports them**. I recomputed every figure from the
committed bytes independently:

```
h100 quack_tuned cold_compile_ms  min 1.538  p50 2.564   max 10.706 ms   0/90 >1s
h200 quack_tuned cold_compile_ms  min 2.963  p50 28264   max 175220 ms  86/90 >1s
wall: h100 discarded 3649.1 s | h100 kept 6.9 s | h200 3765.8 s
```

`cold_compile_ms` is, per the archive's own README (lines 273–274), "the entire
autotune search". A 400-config search cannot finish in 2.5 ms; `rmsnorm_*_tuned`
are decorated without `cache_results=`, the default is `True`
(`quack/autotuner.py:526`), and `check_disk_cache` returns on a hit without ever
calling `bench_fn` (`:270`). So H100's 90 tuned cells were all supplied by
`~/.quack/cache`, and the run that did search is the *discarded contended* one
(canary `closing_over_opening` = 0.872, `quiet: false`).

**The column named for this says `False` on all 540 rows.** `_cold_launch`
does return a real `reused = key in compile_timings`, but its only two call
sites are inside `_FlyDSLProvider`; `_TorchProvider` and `_QuackProvider`
hardcode the literal at four return sites, and the patch adds two more for the
tuned provider. Line numbers, for **both trees** — the archive is pinned at
`8aabb38` and the three of us spent a day quoting HEAD line numbers at a pinned
artifact, which @Autotune pinned down in `225de0bc`:

| | `8aabb38` (the tree that produced the CSVs) | HEAD |
| --- | --- | --- |
| `_cold_launch` | :352 | :1363 |
| `_FlyDSLProvider` call sites | :414, :472 | :1436, :1517 |
| `_TorchProvider` literals | :535, :571 | :1580, :1616 |
| `_QuackProvider` literals | :623, :652 | :1668, :1697 |

The structure is identical in both, so the narrowing holds on the tree that
actually generated the archive and not merely on HEAD. `sweep-rmsnorm.sh:43`
runs exactly `quack_tuned quack torch` —
**all three of the providers in the archive are on the hardcoded side, and the
one provider that could report the field did not run.** So the honest statement
is narrower than "the field is dead": the column's semantics are defined for one
provider only, and the CSV populates it for every provider anyway. For the tuned
provider it is worse than undefined — the reuse it would need to report happens
in `check_disk_cache`, a layer *below* the harness, which cannot see it.

**A hole neither of them had named: the disk cache key contains no device
identity.** `cache_key = [VERSION, str(tuning_key)] + config_str_list`
(`quack/autotuner.py:264`), and `tuning_key` is the *values* of the declared
keys — `key = [kwargs[k] for k in self.keys if k in kwargs]` at `:305`, which
appends `kwargs[k]`, not `k` — plus each tensor's shape/stride/dtype
(`:305-320`). No GPU name, no arch, no `num_cus`.
Arch enters only indirectly, through the enumeration in `get_all_*_configs()` —
and `_max_cluster_for` returns 16 for every `9 <= arch <= 11`, so on H100 and
H200 (both sm_90) the enumerated config strings are byte-identical. **One
`~/.quack/cache` directory hits interchangeably across the two cards**, and no
field in the archive can say which card a hit came from. "Source unrecorded" is
broader than "no timestamp": the *device* is unrecorded too. This is the same
hole as the missing `num_cus` in the artifact fingerprint, from the other end.

*(Third anchor, added after @Reviewer's `eac5bcef` point 4 — his objection was
to @Autotune citing `:264` **alone**, since that line only shows an opaque
`str(tuning_key)` and cannot by itself prove what is or isn't inside it. The
claim above was already anchored at both `:264` and `:305-320`, so it survives;
but he is right that a third anchor is needed to close it, and the third one
makes the hole worse than stated. The on-disk JSON written at `:283-290` stores
`"key": str(tuning_key)` with the comment **"this field is informational (only
`configs_timings` is read back)"**. So the key material is not merely
device-blind — the copy that *is* persisted is never validated on read at all.
Adding device identity to the key tuple would change which file is opened, but
nothing in the loader would check the file's own recorded key against the
caller's. And the two decorators declare `key=["is_layernorm", "per_head"]`
(`rmsnorm.py:511`) and `key=["per_head", "has_dw_partial", "has_db_partial",
"is_layernorm"]` (`:1527`) — four booleans and shapes, no device term
anywhere.)*

Two things I got wrong in the exchange, both of the standard shape. I wrote
"`cold_compile_reused`: 540/540 = False; torch's 90 rows are blank in that
column" — self-contradictory in one line. 540/540 *are* `False`, torch included;
what is blank is `cold_compile_ms`. And I quoted "the 9th-smallest of the
remaining 86 = 5763.654", which is `warm[8]` *including* the four warm cells —
but my replacement was off by one in the other direction: excluding the four,
6147.442 is the **8th**-smallest and the 9th is **6247.459**. @Autotune caught
that (`225de0bc`) inside his acceptance of my correction of his figure, so the
same index was misreported twice in a row, once by each of us.

**And the sentence that used it had a worse error in it.** I wrote "the searched
minimum (6147 ms)" — 6147.442 is the 8th-smallest, and the actual minimum is
**5259.356** (`fwd 1x4096` f32/f32). The same number served as two different
statistics in adjacent sentences, which made the published gap 30× instead of
**26.2×** (5259.356 / 200.396). @Autotune listed all three independently in
`316d4c64` at 18:52:50; `b232286` fixed them at 18:54:25, so his message came
first and I do not claim to have caught them alone — the second and third I hit
while re-checking `4ed86c8` against his first. All of these are numbers correct
about a set other than the one their label names, the same shape as the error
bar and as @CrossVendor's `e764fd2f`, which corrected my `cold_compile_reused`
line while fixing the identical slip in his own `f9b3108b` sum.

I then scanned my own 14 artifacts for the `cold_compile_reused` shape — a
provenance-ish key constant across every row of a list. One hit,
`static_identical_across_dies`, true on all 8 rows of
`rmsnorm_fwd_occupancy_intervention_cross_device.json`. Checked the generator
(`AI/assemble_intervention_cross_device.py:286`): a genuine three-way
conjunction over per-device VGPR and spill sets, so the constancy is a measured
fact, not a literal. Clean this time, and the scan is worth repeating whenever a
new artifact grows a row list.

**What isolates the anchor.** @Autotune's defensible anchor, H200 fwd
32768x8192 bf16 at −10.66%, is one of the four inherited cells. A permutation
test over the four cells' spread against all 90 gives p = 0.0100 (exact,
C(90,4) = 2,555,190) — real but unlocalizing. The sharper test is that each
inherited cell has same-op/same-shape dtype neighbours that *were* searched this
run:

| cell | inherited | searched dtype-neighbours | |
| --- | --- | --- | --- |
| fwd 32768×2048 | +0.16% | −0.55 … +5.62 | inside |
| bwd 32768×2048 | −0.70% | −0.83 … +3.77 | inside |
| bwd 32768×8192 | +17.48% | +0.16 … +18.53 | inside |
| **fwd 32768×8192** | **−10.66%** | −3.87 … +1.87 | **outside by 6.79 pp** |

Only the anchor falls outside its neighbours' range. This cuts both ways:
"inherited" is not itself a defect — +17.48% is indistinguishable from configs
genuinely searched this run at the same shape — while −10.66% cannot be waved
off as "this shape tunes badly". It is descriptive support only: it identifies
neither the config's origin nor a cause, which is where @CrossVendor's `f9b3108b`
put the boundary and is where it should stay.

I also withdrew a recommendation here before sending it. I was going to propose
asserting `cold_compile_ms < 1s ⇒ did not search` in the rerun; `f9b3108b` ruled
it a diagnostic rather than a semantic assertion, correctly — the 26× gap
between the searched minimum (5259 ms) and the inherited maximum (200 ms) is an
accident of this run, and the 200 ms cell already shows a cache hit need not be
cheap. A cache manifest with hit/miss counts and a per-device fingerprint is the
thing that would actually survive a toolchain change.

**A live inconsistency in the same file, recorded not fixed.** @Autotune found
(`225de0bc`) that `QUACK_FORCE_CACHE_UPDATE` is read as a bare truthy
(`quack/autotuner.py:270`: `not os.environ.get(..., False)`), so `="0"`,
`="false"` and `="no"` all *enable* forced update and there is no assignment
that disables it — only `unset`, or the empty string. Confirmed. What makes it
unambiguously a defect rather than a style choice is that it is the **only**
boolean flag in that file written this way. The denominator matters and I got it
wrong first: I published "the only one of the **eight** env reads", and there are
**seven** — an AST walk over `os.getenv` / `os.environ.get` / `os.environ[]`
gives `:35`, `:46`, `:127`, `:174`, `:270`, `:352`, `:456`. I had folded in
`quack/cache/__init__.py:38`, which is a different file. @Autotune caught it
(`b12b184f`) and supplied the tighter framing: `:35` (`QUACK_HOME`) and `:46`
(`QUACK_CACHE_DIR`) are not flags at all but paths with defaults, so the honest
statement is **1 of 5 flags** — `:127`, `:174`, `:352`, `:456` all compare
`== "1"`, and only `:270` does not. Narrowing the denominator strengthens the
finding rather than weakening it. Anyone writing `QUACK_FORCE_CACHE_UPDATE=0` to
turn the behaviour off gets the opposite, and the surrounding code taught them to
expect otherwise. (`:46` is in fact the sturdiest read in the file:
`os.getenv(..., "").strip() or default_cache_dir()` falls back on empty and on
all-whitespace. `:270` is the lone outlier.)

Related, same file, also untouched: `autotune()`'s module-level default is
`cache_results=True` (`:526`), `Autotuner.__init__`'s is `False` (`:109`), and
the docstring says "Defaults to False" (`:549`) — three answers to one question.
`rmsnorm_fwd_tuned` and `rmsnorm_bwd_tuned` pass neither, so they inherit
`True`, which is what makes the whole cache-hit finding above possible.

Neither is fixed here. `quack/autotuner.py` is inside the region @Reviewer is
holding pending the `dbab028` verdict, and a correctness change to cache-update
semantics is exactly the kind of thing that should not land while a freeze is
open. Recorded so the fix is a decision rather than a rediscovery.

*(Update, `2af35006`: @Reviewer's `312c61b`/`dbab028` exact review returned
**request changes**, and with it the blanket hold ended — a fix to
`quack/autotuner.py` now needs its own exact commit/handoff rather than waiting
on the freeze. He independently listed both defects above as standalone
pre-existing bugs. The hold sentence above is left as written because the
paragraph it belongs to was true when committed; this note is the correction.
Same for `1ef57ab`'s commit message, which says "still under @Reviewer's
dbab028 hold" — he lifted it at 19:09:37 and I committed at 19:15:54, so that
line was already stale in the commit and cannot be amended. The substance is
unaffected: that commit touches no production file.)*

###### Does rmsnorm need `restore_value`? No — and asking cost nothing

@Autotune's five-step plan had "harden the cache key **and add
`restore_value`** before adding configs" as step 2. He then found the thing
that makes step 2 self-defeating: setting `restore_value` installs
`pre_hook`/`post_hook` (`autotuner.py:136`/`:145`), and **both** sites that
choose a benchmark path gate on `has_hooks` —

| site | gate |
| --- | --- |
| `_bench` `:199-204` | `use_l2_cold = (self._do_bench is None and ... and not has_hooks)` |
| `__call__` `:363` | `if self._do_bench is None and not has_hooks:` |

The second is the load-bearing one: it skips `_clone_l2_rotate_inputs`
entirely, so `_l2_cold_arg_sets` is never even built. The comment at `:225-228`
says why — "the clone/restore inside hooks doesn't work under CUDA graph
capture" (the quoted sentence begins on `:227` and ends on `:228`) — so the
exclusion is deliberate, not an oversight. Net effect:
turning on `restore_value` silently moves autotune scoring off the L2-cold
CUDA-graph path and onto `partial(triton.testing.do_bench, warmup=5, rep=25)`,
while the harness keeps reporting the L2-cold regime. Selection and reporting
would disagree by construction.

Both of us then went at the prior question — does rmsnorm need it at all — from
different directions, and landed in the same place. His route (`5c4a9111`) is
the stronger one and it is pure reading:

- Every name in `mutates_args` is a caller-allocated **output** buffer. The
  schemas are hand-written with `(aN!)` alias annotations because they mutate
  optional tensors, which makes this a `torch.library` contract rather than a
  comment: fwd `:367` mutates `out`/`rstd`/`mean`/`residual_out`, bwd `:1210`
  mutates `dx`/`dw_partial`/`db_partial`/`dresidual`. The real inputs — `x`,
  `weight`, `dout`, `rstd`, `residual`, `dresidual_out` — carry **no** `(a!)`.
- The accumulator objection answers itself. If `dw_partial`/`db_partial` were
  accumulated rather than overwritten, the tool for it would be
  `reset_to_zero`, not `restore_value` — and **every** allocation site is
  `torch.empty`, never `torch.zeros` (`:1396`, `:1400`, `:1444`, `:1488`,
  `:1818`, `:1819`). So if the semantics were accumulate, *the existing
  non-autotune production path would already be wrong*, reading uninitialised
  memory on every call. It is overwrite. (This argument is his and it is
  better than mine, which was the kernel-level one: `copy(tXrdW, tXgdW)` at
  `:1175`/`:1200` is rmem→gmem, and the cross-row reduction at `:1155-1174`
  goes through smem without ever re-reading global.)
- `reset_to_zero` **does not exist in quack at all** — `grep -c` on
  `quack/autotuner.py` is 0 and no file under `quack/ tests/ benchmarks/`
  mentions it; it lives upstream at
  `/root/wt-device-key/python/flydsl/autotune.py:667`. Second collision of the
  same kind he flagged for `restore_value` in `20a41d64`. Verified both.
- The one place that *looks* like an input being written is `rmsnorm.py:505`,
  `residual_out = x` (guarded by `if residual_out is None:` on `:504`). It is
  not a write target: the `_rmsnorm_fwd(...)` call spans `:500-502` and has
  already returned, so the assignment is post-call, assembling the return
  tuple. It is also in the
  non-tuned `rmsnorm_fwd` wrapper, which `rmsnorm_fwd_tuned` does not call.
  Worth naming precisely because it is the only thing in the file that would
  make someone think the conclusion is wrong.
- `restore_value` has **zero callers tree-wide**, inherited whole in `b69bb67`.

My addition is the hardware half, in `AI/probe_restore_value_needed.py` /
`AI/data/restore_value_needed.json`: eight replays on the same buffers,
bit-identical every time. That alone would prove nothing — a pure-overwrite
kernel is idempotent trivially, so "stable" could be vacuous. The probe
therefore carries a **falsifier**: a second cell that feeds the output back in
as the residual input, creating genuine read-and-write dataflow. That cell
*does* drift. The check can fail, so its passing means something.

One scope limit, stated in the artifact as `verdict.cutedsl_status` rather than
buried: the two decorators `restore_value` would attach to live in
`quack/rmsnorm.py`, which imports `cuda.bindings.driver` at module scope
(`:7`) and **cannot be imported on MI355X**. The kernel I actually replayed is
the FlyDSL one, whose custom ops declare the same pure-output structure
(`rmsnorm_flydsl.py:400`, `:619`). So the cutedsl half rests on source
argument, not on this box's hardware — which is exactly the half @Autotune's
`mutates_args` reading covers. Closing it empirically wants an H100/H200 run of
the same probe against `quack.rmsnorm._rmsnorm_fwd`.

###### I said that remedy was unavailable. It was available, and the way I got there is the defect this whole file is about

I wrote that gap up as though the H200 run were merely waiting on someone's
time, then went to check and reported something worse: **both tailscale H200
boxes have `nvidia-cutlass-dsl` 4.5.2 while `pyproject.toml:10` pins
`==4.6.1`**, and `quack/pipeline.py:13` imports `alloc_reserved_mbarrier`,
which 4.5.2's `cutlass.pipeline` does not export. I reproduced it on
@CrossVendor's own checkout at `4f36477`, on both machines, and sent it to him
as a blocking prerequisite for his Experiment No.001 rerun.

Every individual fact in that paragraph is true. The conclusion was still
wrong, and it is wrong in exactly the way this document keeps cataloguing: **I
checked each box's *system* python and reported the result as a property of the
*machine*.** hyper00 carries `/root/quack-FlyDSL-h200-test/.venv` on cutlass
**4.6.1**; hyper01 carries `/root/crossvendor-cu-venv` on **4.6.0**. Under
either, at current head:

```
$ cd /root/quack-crossvendor-pr1 && .venv/bin/python -c 'import quack.rmsnorm'
rmsnorm OK at current head
```

The label said "the H200 boxes are blocked." The evidence supported "one
interpreter on each H200 box is blocked." Same shape as the `L2_cache_size`
that isn't the LLC and the buffer count summed across cells — a number correct
about a set other than the one its name claims. I have now made the artifact
record `sys.executable` for this reason, with the reason written into the field.

**So I ran it.** `AI/probe_cutedsl_restore_value.py`, hyper00 GPU 1 (0 MiB / 0%
verified immediately before launch; GPU 0 carried the first run and was taken
by someone else's job before the re-run, so the box was shared throughout and
released after), cutlass 4.6.1 via `/root/quack-FlyDSL-h200-test/.venv`,
scratch clone of `4f36477` — chosen over my own HEAD deliberately, and the
choice is safe because `quack/rmsnorm.py` (`26e30eff…`) and
`quack/autotuner.py` (`67875584…`) are byte-identical at both commits, and they
are the only two files the probe touches.

| check | 8192×2048 | 4096×4096 |
|---|---|---|
| CHECK 1 fwd, 5 replays, same inputs | **bit-identical** | **bit-identical** |
| CHECK 1 bwd, 5 replays, same inputs | **bit-identical** | **bit-identical** |
| CHECK 1b bwd, **one `dw_partial` reused** + sentinel | **bit-identical**, sentinel gone | **bit-identical**, sentinel gone |
| CHECK 2 aliased control | **drifted** (out, rstd, residual_out, residual_in) | **drifted** |

Control drifts, mains hold. The probe is sensitive, and the cutedsl kernels do
not mutate their inputs. **The source reading of schema `:367`/`:1210` is now
confirmed on Hopper hardware rather than argued from the file.** That closes
the half of the `restore_value` verdict that had been resting on reading since
the beginning — and it closed in the direction the reading predicted, which is
the least interesting possible outcome and exactly why it was worth spending an
idle GPU on rather than continuing to assert it.

**Row 1b exists because my first version of row "bwd" tested less than its own
docstring claimed.** I wrote "if `copy(tXrdW, tXgdW)` at `:1175`/`:1200` were a
read-modify-write, replaying on the same `dw_partial` buffer would drift" —
while calling `rmsnorm_bwd`, which allocates `dx` and `dw_partial` *itself* on
every call (`:1382`, `:1395`, and `quack::rmsnorm_bwd_f` at `:1443-1444`). No
buffer was ever reused, so that check structurally could not see the thing it
named. Same defect class again, one layer down: the *test* was correct about a
set other than the one its description named. @Reviewer's `b3053d0a` — that
tuned wrappers' call contracts cannot be inferred from untuned wrappers'
allocation habits — is what made me re-read my own probe with that question.

So 1b drives `_rmsnorm_bwd` directly with caller-owned buffers, hands it the
*same* `dw_partial` on all five replays, and pre-fills it with a sentinel
(12345.0) beforehand. Two distinct failure modes are now separated: drift
across replays would mean read-modify-write, and a surviving sentinel would
mean the kernel accumulates into what was there rather than owning every
element it writes. Neither happens. `dw_partial` is a store.

What this still does **not** establish, and @Reviewer is right to keep it out
of scope: **aliasing**. Every call site anyone has checked hands these kernels
freshly allocated, non-overlapping outputs, and `_clone_l2_rotate_inputs`
(`bench_utils.py:174-176`) clones each tensor argument *independently*, which
actively destroys whatever alias graph the real call had. My probe inherits
that limitation exactly — CHECK 2 aliases `residual` onto `residual_out`
through the public wrapper, but nothing here passes `out is x` or overlapping
views to a tuned entry point. So the no-restore result is **conditional on no
aliasing**, and that condition belongs in the design boundary of any future
production tuned wrapper rather than being extrapolated from today's
fresh-allocation habit. The artifact carries that sentence in a
`what_this_still_does_not_establish` field so it travels with the data.

What does **not** transfer: the *cost* half. `AI/data/restore_value_needed.json`
prices the regime switch on MI355X, where the LLC is a 256 MiB MALL behind a
4 MiB L2; the H200's LLC is the 60 MiB L2 itself. The rotation band sits in a
different place on each, so the two artifacts' timings must not be put in one
table. The new artifact says so in a `scope` field rather than leaving it to
whoever reads it next.

One consequence I should state plainly because I sent the opposite to
@CrossVendor: **his Experiment No.001 rerun is not blocked by this.** The
`pipeline.py:13` / 4.5.2 mismatch is real for the system interpreter and worth
knowing, but a 4.6.1 venv already exists on hyper00 and a 4.6.0 one on hyper01.
I sent him a prerequisite that isn't one. The `__init__.py:6` observation below
still stands on its own facts.

The sharper part that survives. `quack/__init__.py:6` gates the entire cutedsl
import chain on `torch.version.hip is None`, so on a **CUDA** box `import
quack.rmsnorm_flydsl` executes `quack/__init__.py` first and would die inside
`quack/rmsnorm.py` before ever reaching the FlyDSL module — *if* cutedsl were
broken in that interpreter. On the 4.6.1 venv it is not, so the FlyDSL import
gets further and fails on its own missing `flydsl` package instead. It is a
real failure mode all the same: it is precisely what both boxes' *system*
interpreters do, carrying cutlass 4.5.2. The FlyDSL backend has no cutlass
dependency and gets taken down by a package it does not use.

###### And the third instance of the same mistake, in the sentence I wrote about the second

I closed the paragraph above with: `tests/test_import_isolation.py` passes here
because ROCm takes the other branch, so **this direction of the gate is
untestable on this hardware**. That is wrong, and it is the *same* wrong as
everything else in this section — I asserted a property of the hardware from an
observation about the current test file. "Untestable on this hardware" and
"untested on this hardware" are different claims and I made the stronger one
without checking.

The branch is selected by `torch.version.hip`, which the sibling test
**already overrides** to simulate CUDA (at `:150` in the current file; the
`:77` written here was wrong when written — that line is an assertion in a
different test), and a failing cutedsl import is one
`meta_path` loader away. So the test is writable on MI355X, and I wrote it:
`test_simulated_cuda_broken_cutedsl_takes_the_flydsl_path_down_with_it`. It
sets `torch.version.hip = None`, installs a loader that raises the real 4.5.2
error (`cannot import name 'alloc_reserved_mbarrier' from 'cutlass.pipeline'`)
for `quack.dsl` and `quack.rmsnorm`, then asserts that `import
quack.rmsnorm_flydsl` fails **with that message** and that
`quack.rmsnorm_flydsl` never lands in `sys.modules`.

Two details that decide whether it is evidence:

- ~~It intercepts `quack.dsl` too, not just `quack.rmsnorm`. On a real CUDA
  box `quack.dsl` is imported first and dies on the same missing cutlass.~~
  **False, and @Reviewer checked it against the tree when I did not.**
  `alloc_reserved_mbarrier` is imported at exactly one place,
  `quack/pipeline.py:13`, reached via `quack/rmsnorm.py:24`. `quack/dsl/`
  imports `cutlass.cute*` and never `cutlass.pipeline`, so it does not
  request the missing name at all. Measured: the fabricated error was
  raised by `quack.dsl` on every run, i.e. the assertion passed at a
  boundary the real chain does not fail at — the wrong-reason pass the
  bullet claims to prevent, committed by the bullet itself. The successor
  lets `quack.dsl` succeed and routes the failure through
  `quack.rmsnorm` -> `quack.pipeline`, checking the reached chain.
- It asserts on the **message**, not merely that something raised. A bare
  "it raises" would also be satisfied by FlyDSL simply being absent, which is
  a different fact about a different package.

Mutation-checked rather than assumed: wrapping the cutedsl chain in
`try/except ImportError` inside `__init__.py` — the obvious repair — fails
**only** this test (`1 failed, 4 passed`), which is the signal it exists to
give. Disabling the gate entirely fails this one and the `:77` sibling, the
right blast radius for removing the branch both depend on. Both mutations
reverted; `quack/__init__.py` is untouched in the commit.

Suite is now **737 passed, 2 skipped, 1 xfailed**. (The `738 passed, 2
skipped` originally written here was this commit's own transient state; the
test has since been converted to a strict xfail, twice, and the number is
corrected in place rather than left to read as present tense. @Reviewer
measured the exact-tip figure and the count above is his.)

###### The above shipped the defect I had just audited in someone else's file

@Reviewer, on the version described above: a plain green test asserting that
the import **fails** makes the obvious repair show up as a red test. The fix
would look like the regression. That is the same objection I had spent the
afternoon establishing against `f747907`'s eight green tests — and I wrote it
into my own file, in the commit whose whole subject is noticing this class of
mistake. Finding it in someone else's work does not inoculate you.

So the test is now a **strict xfail stating the desired behaviour**: FlyDSL has
no cutlass dependency, therefore `import quack.rmsnorm_flydsl` *should* survive
a broken cutedsl chain. It XFAILs today. When the import boundary is repaired
it XPASSes, and `strict=True` converts that into a failure that says "flip
me" — a signal that the successor landed, not a wall in front of it. Renamed
accordingly: `test_simulated_cuda_flydsl_import_survives_a_broken_cutedsl_chain`.

Both directions measured, not argued:

| state of `quack/__init__.py` | outcome |
| --- | --- |
| unmutated (today) | `4 passed, 1 xfailed` |
| cutedsl chain wrapped in `try/except ImportError` | `1 failed, 4 passed` — `XPASS(strict)` |

The child process now signals through its **exit code** (3 = failed for exactly
the cutedsl reason, 0 = imported and `rmsnorm` is callable), and the parent
rejects any other code as a third outcome rather than folding it into either
branch. That guard immediately caught a bug in my own success path: under the
repair the import succeeded and then died on `AttributeError: module
'quack.rmsnorm_flydsl' has no attribute 'rmsnorm_flydsl'` — I had asserted a
name that does not exist, so without the guard the fix would have registered as
"still broken." rc=1 was refused, printed, and fixed to `callable(...rmsnorm)`.

And a hole he measured that I had asserted shut. I wrote that the parent
"refuses any other exit code"; that was true of the parent's assertions and
false of what pytest reported. `xfail(strict=True)` with no `raises=` records
**any** failure of the test as the same XFAIL, so he replaced the child body
with an unrelated `RuntimeError` and still got `4 passed, 1 xfailed` — the
third-outcome guard fired and was absorbed by the marker that was supposed to
be reporting something else entirely. A guard whose failure is indistinguishable
from the expected result is not a guard, and "the parent refuses it" was a claim
about code I had read rather than about the only output a reader sees.
Reproduced before fixing: same `1 xfailed`. Now `raises=` is narrowed to a
purpose-built `CutedslGateStillCouplesFlydsl(AssertionError)`, and the guard
raises a plain `AssertionError`, which is no longer the expected type. Three
states, all measured:

| state | outcome |
| --- | --- |
| unmutated | `4 passed, 1 xfailed` |
| @Reviewer's unrelated `RuntimeError` in the child | `1 failed, 4 passed` — real failure, traceback shown |
| cutedsl chain wrapped in `try/except ImportError` | `1 failed, 4 passed` — `XPASS(strict)` |

Two of @Reviewer's points I checked rather than accepted. The independence
question — whether this test only passes because of `sys.modules` ordering from
its sibling — does not arise: `_run_python` spawns a **subprocess**, so each
test gets a fresh interpreter. Verified by running the test alone and the
file three times. (Those runs read `1 passed` / `5 passed` when written;
at the exact tip they are `1 xfailed` / `4 passed, 1 xfailed`, because the
test is now a strict xfail. Superseded counts, corrected in place.)

A stronger caveat on that answer, from @Reviewer: a fresh interpreter was
never the same thing as a fresh **package state**. It inherited
`PYTHONPATH`, so a `sitecustomize.py` containing only `import quack`
preloaded the unmodified package and the test reported XPASS with nothing
repaired. Fixed by running every child under `-I`.
And his caveat on the repair itself is right and is *not* what this test
endorses: `try/except ImportError` around the whole chain would swallow genuine
import bugs inside the dependency. It is a mutation to show the test responds,
not a proposed fix; the real repair is a package-init dependency boundary.

Conclusion, agreed both ways: **step 2 should be deleted, not rewritten.**
`restore_value` buys rmsnorm zero correctness and costs the measurement regime.
His option 2 — making restore and the graph path coexist — is a real design
problem but has no customer: `_clone_l2_rotate_inputs` already clones args *and*
kwargs per rotation buffer, which solves what restore solves in a way CUDA graph
capture accepts. Hardening the cache key stays, and is the more urgent half.

###### What that probe got wrong first, and what the 2×2 actually shows

I tried to price the regime switch as one number and got a result pointing the
wrong way: `do_bench` on a single buffer is **slower** than the rotating graph
(0.1110 vs 0.0906 ms at 32768×4096; 0.0165 vs 0.0125 at 8192×2048). If that
were an L2 measurement, L2-hot being slower would be nonsense. It isn't one —
the two paths differ in **launch mechanism** as well as cache state, so the
comparison was confounded from the start. Splitting it (32768×4096, ms, medians
of 5 from `AI/data/restore_value_needed.json`):

| | graph | `do_bench` |
| --- | --- | --- |
| rotate (4 sets) | **0.0911** | 0.1129 |
| single | 0.0897 | 0.1151 |

- **Mechanism** = 1.240× here, 1.629× at 8192×2048. `do_bench` pays an event
  pair and a fresh output allocation per launch; the graph replays 200 recorded
  calls with no Python in the window. This term is essentially the whole
  end-to-end gap (1.264× and 1.388×).
- **Rotation, within one mechanism** = 1.00× (0.985× and 0.997×).

(Cells move in the third decimal between runs — 0.0904/0.0907/0.1112 several
runs earlier. Message `f75cd8b8` quotes that earlier run; the table here is the
current artifact's, regenerated after the `cutedsl_status` correction below.
The mechanism term at 8192×2048 is the least stable figure in the set,
1.629–1.665× across runs, and the 8192×2048 end-to-end number moves with it
(1.323–1.388×); the rotation term is 0.98–1.01× throughout. Ratios are what
this section argues from, for that reason. The rotation term here is ~1.00× for
a reason established two subsections down — **both of these shapes lie outside
the band where rotation can change cache state at all**, so this 2×2 is not
evidence about rotation in general.)

###### …and the explanation I first gave for the 1.00× was the very defect this repo already documents

I wrote, and committed in `2566fb9`: "**MI355X reports `L2_cache_size` = 4 MiB**,
while one 32768×4096 bf16 tensor is 256 MiB and one 8192×2048 is 32 MiB — 64×
and 8× the cache before any rotation. The single-buffer cell was never L2-hot;
the working set blows the cache on the first pass, and rotating four clones
cannot make an already-cold read colder." **That reasoning is wrong, and wrong
in the specific way `AI/gfx950_mall_evictor_defect.md` exists to warn about.**

The last-level cache on gfx950 is not the 4 MiB L2. The hierarchy (ROCm Kernel
Wiki `hw-chiplet-xcd`, quoted in that file) is

    per-CU L1D -> per-XCD 4 MiB L2 (x8 XCDs in SPX) -> 256 MiB MALL -> HBM

and `torch.cuda.get_device_properties().L2_cache_size` returning 4 MiB is
precisely the value that document calls the **fallback** — defect 1 in its own
table is the harness sizing its rotation target against `L2_cache_size` instead
of the MALL, fixed in `31c1fd4` ("size the rotation target and the evictor gate
against the MALL, not L2"). I read the same property the old harness read, drew
the same conclusion the old harness drew, and did it in a tree that contains a
written record of why that is a mistake. Working the numbers against the right
cache:

| shape | one tensor `t` | single set `2t` | rotated set `(n+1)t` | fits 256 MiB MALL? |
| --- | --- | --- | --- | --- |
| 32768×4096 bf16 | 256.0 MiB | 512.0 MiB | 1280.0 MiB | **neither** |
| 8192×2048 bf16 | 32.0 MiB | 64.0 MiB | 160.0 MiB | **both** |

(These are the *measured* sets — see the next subsection. I published two wrong
denominators before arriving at them. The first counted only cloned inputs and
missed the output. The second "fixed" it to `2n·t`, assuming each rotation slot
carries its own output; that was the same counting error @Autotune corrected in
his own transition-point claim an hour later (`b01d0609`), where he compared
probe buffers against sweeps whose rotation slot is a src+dst pair and got his
direction *and* magnitude the wrong way round. Same trap, adjacent numbers,
found independently — but in my case the fix overshot, because I inferred the
rotation from source instead of measuring it. It is `(n+1)t`, not `2n·t`.)

Which left the honest statement of the observation much narrower than what I
published: **the within-graph single-vs-rotate ratio came out ~1.00× on both
shapes, and I did not have an explanation for it that survives contact with
this document.** The reason I offered explains a null result on the small shape
by appeal to a cache that is not the relevant one, and predicts a *non*-null on
the large shape, which is not what the probe measured.

###### The explanation that did survive — found because a reviewer refused the fix I was proud of

@Reviewer's `dbe0206d` declined to accept the `in+out` column I had just added
as a correction, on the grounds that it *infers* per-slot outputs from
`torch.empty` in the source, when under graph capture the output's address is
an allocator question, not a source question. He was right — my correction was
itself wrong — and testing it settled the ~1.00× as well.

`AI/probe_rotation_band.py` records `data_ptr()` on every returned tensor
inside the captured region. **The output does not rotate: `n=4` input buffers,
`1` output buffer.** The caller discards each returned tensor, so the caching
allocator hands the same block back every call. So both published denominators
were wrong, in opposite directions — `n*t` missed the output; `2*n*t`
overstated the rotated set by 1.6×. The measured sets are `(n+1)*t` rotated and
`2*t` single. (Getting this scoped right took three tries in the probe itself:
accumulating pointers across all four cells reported 8; counting one whole
`bench_graph` call reported 2, because eager warmup allocates from the normal
pool and capture from the graph's private pool, and the timed window is the
replay alone. Only the third names the set it claims to.)

That makes the null result predictable rather than mysterious. Rotation can
only change cache state where the single set is MALL-resident and the rotated
set is not — `2t <= 256 MiB < (n+1)t`, i.e. `t` in (51.2, 128] MiB. **Both
shapes I benchmarked sit outside it**: 8192×2048 has `t`=32 MiB (64 vs 160 MiB,
both fit); 32768×4096 has `t`=256 MiB (512 vs 1280 MiB, neither fits). Neither
cell was ever a contrast. A scan at N=1024 agrees on all eight coarse points:

| `t` | 32 | 48 | **56** | **64** | **96** | **128** | 160 | 192 MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single/rotate | 0.989 | 0.999 | **0.919** | **0.862** | **0.860** | **0.856** | 1.000 | 1.004 |

Bold = inside the predicted band. 8/8, and the null holds on *both* sides,
which is what distinguishes the model from "bigger is slower".

So the corrected reading is the opposite of the one I nearly published:
**rotation does exactly what it was built to do — I picked two shapes where it
cannot show.** "Rotation buys nothing on this card" would have been a
device-wide claim resting on two badly chosen points.

###### The band's two edges are not the same kind of claim, and I stated both as if they were

@Reviewer's `afd7d2b5` audits @Autotune's commit, not mine, but one line in it
is checkable here: "`_pick_l2_rotate_count(..., target_ratio: int = 3, ...)` is
at `bench_utils.py:1274`". True in his frozen file and true in my tree at
`:180`. Following that default through my own band exposes a limit I had
missed.

**`n=4` is not a property of this device.** `_pick_l2_rotate_count` sizes
`target_ratio * L2_cache_size` = 12 MiB against tensors of tens of MiB, so
`n_by_l2` is always 1 and `n` clamps to `min_buffers` for *every* shape in my
scan. That is defect 1 again — the same wrong-cache read, this time landing on
the buffer count instead of the eviction target. I derived the band holding
`n=4` and then wrote it as though (51.2, 128] MiB were a fact about gfx950.

Under an effective-LLC fix `n` scales with the MALL (`n`=24 at `t`=32 MiB,
`n`=16 at `t`=48 MiB), so the rotated set crosses capacity at far smaller `t`
and the low edge moves **down**. Forcing the `n` such a fix would pick:

| `t` MiB | 32 | 32 | 32 | 48 | 48 | 192 | 192 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `n` | 4 | 8 | 24 | 4 | 16 | 4 | 12 |
| single/rotate | 1.004 | **0.879** | **0.868** | 0.992 | **0.867** | 0.975 | 1.001 |

7/7. The two edges are different in kind:

- **Low edge — a claim about this harness.** Two shapes I filed under "rotation
  cannot help here" gain a clear effect at a larger `n`. Nothing about the
  device forbade it; the buffer count did.
- **High edge — a claim about the device.** `2t > MALL` contains no `n`, so no
  buffer count can rescue it, and `t`=192 MiB stays ~1.00× at both `n`=4 and
  `n`=12.

The coarse scan cannot distinguish these, because every one of its points was
taken at the same clamped `n`. 8/8 agreement made a harness artifact look like
a hardware law. Recorded as `EDGE_IS_N_DEPENDENT` in the artifact so the low
edge is not quotable as a gfx950 property.

There is a second-order consequence for @Autotune: **his effective-LLC fix does
not merely re-target the evictor, it changes which shapes rotation can affect
at all** — `t`=32 MiB moves from no-effect to ~0.87× purely by `n` going 4→24.
Any before/after comparison across that fix is comparing two different
experiments at the small end, not the same experiment with a better constant.

I first sent that alongside a claim that his MALL-cap failure counts (5/7/8 for
caps 128/64/32) were "two sides of the same mechanism" as my forced-`n` result.
@Reviewer's `d43c819d` refused the merge, and he is right — **and the reason is
stronger than the one he gave.** His counts come from
`tests/test_bench_utils_llc.py`, which runs against a synthetic `gfx950`
fixture with monkeypatched cache values: 155 `monkeypatch`/fixture references,
no kernel launched, no timing taken. They are assertions about a **capacity
model's arithmetic**. My 7/7 is wall-clock `single/rotate` on a real MI355X.
The two can agree numerically while sharing no evidence at all, because one
never touches the device the other measures. Citing their agreement as
cross-validation would be the label-vs-set defect one level up: two things
named "the effect of changing the cap" that are about different sets.
Retracted; each stands on its own footing.

Two things keep this short of settled. The MALL size is hardcoded, the same
caveat that document flags about its own 256 MiB. And a tight scan across the
edges shows the transition is **soft, not a step** — at `t`=51 MiB the ratio is
already 0.949 where the model says 1.00, and at 129/132 MiB it is 0.885/0.959
rather than snapping back. The capacity rule predicts *where* the band is, not
a threshold at its edges; that is consistent with partial residency in a
set-associative cache, and sharpening it is Experiment No.002 territory
(`gfx950_mall_evictor_defect.md` blocks it pending re-collection). The earlier
candidate list — HBM-bound, evictor gap, `graph_single` not single-buffer — is
withdrawn: the third is now measured false (it *is* single-buffer), and the
first two are not needed to explain a result the capacity model predicts.

What survives unchanged is the part the section is actually for: the
**mechanism** term (1.240×/1.629×) accounts for essentially the whole
`graph_rotate` → `do_bench_single` gap, so the end-to-end number is not an L2
measurement and must not be quoted as one. That conclusion never depended on
which cache is last-level. The `restore_value` verdict does not depend on any
of this either — it rests on the mutation contract and the replay test, and as
of `AI/data/cutedsl_restore_value.json` that replay test has now been run on
the cutedsl kernel too, on Hopper, with a control that drifts.

One contamination of my own, flagged rather than quietly dropped: the
`do_bench` rotate cell rotates via a Python closure doing a modulo and an index
**inside the timed region**, so its 0.987×/0.828× is that overhead plus
allocator behaviour, not a cache effect. Only the within-graph ratio is
like-for-like,
because there the rotation is baked into the recorded graph on both sides.

None of these numbers bear on `bench_utils.py:98` ("the L2-hot number favours
wider layouts / deeper smem stages that don't actually win once data has to
come from HBM"). That is a claim about **which config wins** — relative ranking
across layouts — not about the magnitude of any single config. Confirming or
refuting it requires scoring the whole config set under both regimes, which is
a different experiment. Quoting these ratios at it would be the same
label-vs-set error this file keeps cataloguing.

## I proposed a guard against the defect, and shipped the defect in the guard

@Autotune corrected a prescription of mine and was right to. I had written
that per-test `AssertionError` subclasses were **necessary** to keep a shared
helper's bug from being absorbed as a converted test's expected outcome. His
S1–S4 show one shared `ExpectedDefect` gives identical results, and his S5
shows per-test subclasses do not close the door either: point the helper at
whichever class the test names and the absorption comes right back
(`4 failed, 6 passed, 2 xfailed`, both converted tests swallowing). Counting
subclasses was never the answer.

His replacement is an AST meta-test stating the invariant directly — raisers
of the marker class must be a subset of the tests marked with it — and he is
right that *saying* the invariant beats implying it through a naming
convention. I measured what the check actually covers. Helper mutated to
`len(paths) != 99` so the helper itself is broken, on his frozen `730b7775`:

| | AST guard | runtime |
|---|---|---|
| clean | `stray=[]` | `11 passed, 1 xfailed` |
| S5 `raise ExpectedDefect(...)` | `stray=['_sole_artifact']` ✓ | `4 failed, 6 passed, 2 xfailed` |
| E1 `_E = ExpectedDefect; raise _E(...)` | `stray=[]` | `4 failed, 7 passed, 1 xfailed` |
| E2 helper moved to a sibling module | `stray=[]` | `4 failed, 7 passed, 1 xfailed` |
| E3 `getattr(mod, "ExpectedDefect")` | `stray=['_sole_artifact']` | `4 failed, 7 passed, 1 xfailed` |

E1 adds one line. E2 changes nothing about the helper at all — it is what a
shared helper looks like after one ordinary refactor, not an adversarial
construction. Both read `stray=[]`, "clean", while that surviving `1 xfailed`
*is* the absorption. E3 goes red by coincidence: `"ExpectedDefect"` happens to
appear as a string literal inside the `getattr`, so `ast.dump` collides with
it. The check also fires on innocent code — `raise ValueError("not an
ExpectedDefect at all")` and a local named `ExpectedDefectSeen` both land in
`stray`. Substring matching is wrong in both directions.

Which is this file's recurring defect one more time: **a check correct about
one set (raises whose source text spells the class name) wearing a label that
names another (raises of that class).** The invariant is right; the
enforcement point is source text, and source text is not where raising
happens.

Moving the check to runtime fixes it. In `pytest_runtest_makereport`, when the
`xfail(raises=E)` marker matches, walk to the innermost traceback frame and
ask whether that code object was compiled inside the test. Alias, `getattr`
and cross-module all fail that question identically, because it asks who
raised rather than how the source spelled it. All three evasions go red, clean
stays green.

**And the first version of my fix had the same defect it was written to
catch.** I compared `co_name != item.function.__name__`, so a test that
defines a closure and raises from inside it was reported as absorption — but a
closure written inside the test *is* the test's own code. "Innermost frame is
the test function" is a set; "the raise came from the test's own body" is the
set I meant. Recursing `co_consts` covers closures, comprehensions and
generator expressions, and the three false-positive probes go `3 xfailed`.
Fourth time this session I have produced this defect class while in the middle
of writing about it, which is the argument for the runtime check being
mechanical rather than a rule anyone has to remember.

Guard checked in at `AI/proposed_xfail_absorption_guard.py`, unwired (it does
nothing where it sits), no dependencies;
whether it lands in `tests/conftest.py` is @Autotune's call, since it is his
file. Reported as `695fb4f0`. All runs in a throwaway clone of `730b7775`,
since deleted.

## Four ways that test was still wrong, none of which I found

@Reviewer's verdict on `e3853d7` was REQUEST CHANGES with three blockers plus
a stale-counts finding. I reproduced all four on the frozen tip before
changing anything, and every one holds.

**1. A cross-process exit code authenticates nothing.** The child reserved
exit 3 for "failed, and failed for exactly the cutedsl reason". He replaced
the child's `raise ImportError(MESSAGE)` with a bare `raise SystemExit(3)` —
an unrelated third outcome that collides with the reserved code — and the
parent converted it into `CutedslGateStillCouplesFlydsl`. Result:
`4 passed, 1 xfailed`, **pytest exit 0**. Reproduced exactly. This is the
third time in a row this same test has been green for a reason it does not
describe, and each time the previous fix was real but narrower than I claimed:
`raises=` genuinely closed rc=1 and unrelated exception types, and I then
wrote that "an unrelated failure cannot be absorbed", which is a statement
about a larger set than the one I had closed.

The fix is that the child now prints `SENTINEL {json}` naming its own branch,
the reached chain, and whether the message matched; the parent requires
exactly one sentinel line and checks the branch name, not the number.

**2. The simulation was attached to the wrong boundary.** The test raised the
missing-symbol error from `quack.dsl` *and* `quack.rmsnorm`, with a comment
justifying `quack.dsl` on the grounds that it "would fail on the same missing
cutlass". Checked against the tree: `alloc_reserved_mbarrier` is imported at
exactly one place, `quack/pipeline.py:13`, reached through
`quack/rmsnorm.py:24`. `quack/dsl/` imports `cutlass.cute*`, `cutlass`,
`cutlass.base_dsl` — never `cutlass.pipeline`. So it does not request the
missing name at all, and being imported first it was where the fabricated
error always came from. Instrumented the loaders to record who raised:
`WHO_RAISED=['quack.dsl']` on every run. **The bullet written to prevent a
wrong-reason pass was itself the wrong-reason pass**, which is this file's
recurring defect once more — a justification correct about a set (modules
that fail on a CUDA box) other than the one it names (modules that request
`alloc_reserved_mbarrier`).

The successor lets `quack.dsl` succeed, routes through `quack.rmsnorm` →
`quack.pipeline`, and asserts the reached chain is
`['quack.dsl', 'quack.rmsnorm', 'quack.pipeline']`. Mutating it back to the
predecessor's shape (error raised at `quack.dsl`) now fails: `1 failed, 4
passed`.

**3. A fresh interpreter is not a fresh package state.** `_run_python`
inherited the environment and prepended ROOT to the inherited `PYTHONPATH`.
He put a `sitecustomize.py` containing nothing but `import quack` on that
path and the test reported `XPASS(strict)` — "the boundary is repaired" —
against a completely untouched `quack/__init__.py`. Reproduced: `2 failed, 3
passed`, the target among them as a false XPASS. My defence of this test's
independence ("`_run_python` spawns a subprocess, so each test gets a fresh
interpreter") was true and answered a smaller question than the one that
mattered.

Every child now runs under `-I`, which ignores `PYTHONPATH`, `sitecustomize`
and the user site directory; ROOT is injected into `sys.path` inside the
child instead, which also pins the tests to *this* tree rather than whichever
`quack` the ambient environment resolves first. Under the same preload the
whole file now reads `4 passed, 1 xfailed`, identical to clean.

He also found the success path depended on this host having the optional
FlyDSL package installed — with `flydsl` blocked, the repaired boundary
produced an ordinary rc=1 instead of the intended XPASS, which would make the
test unable to demonstrate the repair on the very CUDA machines it is about
(the notes record the 4.6.1 CUDA venv as missing `flydsl`). The target module
is now supplied by the stub loader, so what is measured is the package
boundary alone. Verified: repair mutation + `flydsl` blocked in every child →
still `XPASS(strict)`.

**4. Stale counts and a wrong anchor.** `738 passed, 2 skipped` (this
commit's own transient state), `1 passed` / `5 passed` for the target and
file, a gate figure omitting the xfail, and `:77` for an override that lives
at `:150`. All corrected in place rather than left reading as present tense.
The marker's "turns this GREEN" was also wrong in the same direction as
everything else here: `strict=True` makes the repair a *failing* XPASS, which
the docstring three lines down described correctly.

Mutation matrix on the successor, all `HIP_VISIBLE_DEVICES=5`:

| | result |
|---|---|
| clean | `4 passed, 1 xfailed`, exit 0 |
| unrelated child `SystemExit(3)` | `1 failed, 4 passed`, exit 1 |
| child `RuntimeError` | `1 failed, 4 passed` |
| wrong ImportError message | `1 failed, 4 passed` |
| error raised at `quack.dsl` (predecessor's shape) | `1 failed, 4 passed` |
| `sitecustomize: import quack`, no repair | `4 passed, 1 xfailed` |
| boundary repaired | `XPASS(strict)` |
| boundary repaired + `flydsl` blocked | `XPASS(strict)` |

Gate unchanged: `737 passed, 2 skipped, 1 xfailed`. Pinned ruff 0.11.13 check
and format clean.

What I take from four rounds on one test: each of my fixes was real, and each
time I described it in terms of a set larger than the one I had actually
closed. The pattern is not that the fixes were wrong — it is that the
*claim* moved one quantifier out past the evidence, every single time.

### Addendum: he then checked it against the real wheel, and it broke again

@Reviewer installed genuine PyPI `nvidia-cutlass-dsl==4.5.2` and compared the
real failure against the synthetic one. I reproduced his setup: downloaded
both wheels and confirmed the hashes byte-for-byte —
`nvidia-cutlass-dsl==4.5.2`
`68ed1b63ca74aae87955012da9dfd7fdaae471329d0028b229b841c7192ccf52`,
`nvidia-cutlass-dsl-libs-base==4.5.2` (cp310)
`386e832427e3670479049a1560e4d8d2e565d8c0f37a6852c6d7043d046548f1`.
Installed to an isolated target with `cuda-python==12.9.7`,
`cuda-bindings`, `cuda-pathfinder`, the real traceback is:

```
quack/__init__.py:22 -> quack/rmsnorm.py:24 -> quack/pipeline.py:13
ImportError: cannot import name 'alloc_reserved_mbarrier' from 'cutlass.pipeline'
  exc.name = 'cutlass.pipeline'
  exc.path = '.../cutlass/pipeline/__init__.py'
```

Confirming his chain independently, on real bytes rather than by reading
source. My successor set `name=` but not `path=`, and his point is that the
difference is not cosmetic: a *narrow* repair may legitimately key on the
metadata. Built the one he describes — tolerate the skew only when
`exc.name == "cutlass.pipeline" and exc.path is not None` — and measured both
sides:

| | real 4.5.2 wheel | my successor's test |
|---|---|---|
| path-keyed repair, `path=None` simulation | `rmsnorm` callable | `1 xfailed` |
| path-keyed repair, `path=` simulation | `rmsnorm` callable | `XPASS(strict)` |

The first row is the failure mode: the interpreter the test exists to protect
is *already fixed*, and the test still reports the defect. So the simulation
now carries `name` and `path`, taken from the child's own resolution of
`cutlass.pipeline` where it exists, and the parent asserts the shape of both.
Dropping `path=` is itself now a failing mutation (`1 failed, 4 passed`).

Which is the same lesson a fifth time, and this one is worth stating exactly:
I fixed "the error is raised at the right module" and wrote that the failure
now travels the real chain. It travels the real *modules*. The **exception
object** was still not the real one, and "the real chain" names a set that
includes the thing being thrown. Every round of this has been the claim
covering one more attribute than the evidence.

Full matrix on the amended successor:

| mutation | result |
|---|---|
| clean | `4 passed, 1 xfailed` |
| unrelated child `SystemExit(3)` | `1 failed, 4 passed` |
| child `RuntimeError` | `1 failed, 4 passed` |
| wrong message | `1 failed, 4 passed` |
| wrong `exc.name` | `1 failed, 4 passed` |
| `path=` dropped | `1 failed, 4 passed` |
| path-keyed repair | `XPASS(strict)`, and real wheel callable |

Gate `737 passed, 2 skipped, 1 xfailed`; pinned ruff clean over `tests/` and
`quack/`.

One thing this environment settles that the simulation never could. Every
claim on this branch about cutlass 4.5.2 breaking the FlyDSL import — the
reason the test exists — was inferred from reading source, because this box
had no cutlass at all. Against the real wheel, at branch head, unpatched:

```
import quack.rmsnorm_flydsl
  -> ImportError, exc.name='cutlass.pipeline'
```

So the defect is real on real bytes, not only in a meta_path fiction. That
was worth knowing before shipping five rounds of tests about it, and I did
not know it until @Reviewer's addendum made me install the wheel. The
environment is kept at `/tmp/c452env` and the recipe is in the message to
@CrossVendor: the whole chain is import-level, so it needs no GPU and no
NVIDIA hardware, which means every "what does a CUDA box do here" question on
this branch was answerable on g23 the entire time.

### The real wheel found a second defect nobody knew about

@Reviewer's merge boundary: the simulation must let the real `quack.dsl`,
`quack.rmsnorm` and `quack.pipeline` code execute and produce the failure at
the `cutlass.pipeline` dependency edge, because a whole-module loader hides
any narrow repair made *inside* those files. He is right, and with a real
4.5.2 install the answer is simpler than any simulation: **intercept nothing
at all.** Real 4.5.2 genuinely lacks the symbol, so the whole chain runs and
Python raises the error itself. That is
`test_real_cutlass_452_flydsl_import_survives_the_missing_symbol`, gated on
`QUACK_CUTLASS_452_ENV` and skipped without it.

Then the mutation he asked for — a narrow repair *inside* `quack/pipeline.py:13`:

```
try:
    from cutlass.pipeline import agent_sync, alloc_reserved_mbarrier
except ImportError:
    from cutlass.pipeline import agent_sync
    alloc_reserved_mbarrier = None
```

It is visible to the test, which is what he wanted proven. But it did not
produce XPASS — it moved the failure to a **different, previously unknown
one**:

```
quack/rounding.py:25 -> from cutlass._mlir_helpers.arith import bitcast
ImportError: No module named 'cutlass._mlir_helpers'
```

4.5.2 ships `cutlass/_mlir`, not `cutlass/_mlir_helpers`. So **the 4.5.2 gap
is at least two independent symbols in two different files**, and every
statement on this branch describing it as "the missing `alloc_reserved_mbarrier`"
— including the marker reason, the docstrings and the notes above — has been
describing one member of a set with more than one member. Nobody knew because
nobody had the wheel; it was inferred from source the whole time.

That also falsified an assertion I had just written. The real-wheel test
pinned `exc_name == "cutlass.pipeline"` and the exact three-frame chain, so
the inner repair read as "the real failure was not the cutlass.pipeline one"
— a repair reported as an unrelated breakage. Widened to
`exc_name.startswith("cutlass")`, with the specific edge reported rather than
asserted: what this test is about is the *coupling*, a FlyDSL import dying
anywhere inside a cutedsl bootstrap it does not use, and that is now what it
asserts. With both inner repairs applied it goes `XPASS(strict)`.

| | real-wheel test |
|---|---|
| unpatched | `1 xfailed` |
| repair inside `quack/pipeline.py` only | `1 xfailed` (moved to rounding.py) |
| repairs in `pipeline.py` + `rounding.py` | `XPASS(strict)` |
| no `QUACK_CUTLASS_452_ENV` | `1 skipped` |

Gate with the env var set: `4 passed, 2 xfailed`; without it,
`4 passed, 1 skipped, 1 xfailed`.

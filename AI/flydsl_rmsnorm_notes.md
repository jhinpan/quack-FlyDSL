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
skipped** -- is this invocation, and it is written down here because I quoted
it repeatedly without recording it and then could not reproduce my own number:

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

**Below about 26 us of device work, the public API measures Python, not the
kernel.** One `rmsnorm(x, w)` costs ~26 us of *host* time at `256x4096`, against
~6.4 us for `torch.nn.functional.rms_norm` (regenerated: 30.14 vs 5.73 us,
sidecar `AI/data/rmsnorm_call_decomposition.json`, generator
`AI/probe_rmsnorm_call_decomposition.py`). It is genuinely asynchronous -- the
cost does not move when ~1.2 ms of device work is queued ahead of it -- so it is
host dispatch, not a stall. Localized by stubbing each stage in situ: the cached
launcher is 11.3 us, the four `torch.empty*` allocations bring it to 16.3, the
`autograd.Function.apply` to 19.4, and `_validate_inputs` adds 2.9 for 26.1
total. (**Still unbacked**: the stage split needs in-situ stubbing of quack
internals and is not in the sidecar. The 26 us total it sums to *is* backed.)

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
still costs **6.38 us** (**unbacked**: not regenerated in the sidecar, which
covers the host total and the graph series but not the stubbed-stage figures),
and that is FlyDSL's own dispatch: the profile at that
level is all `<flydsl-dispatch>`, `ptr_fill` and `pack_into`, with no frame of
ours in it. For scale, `torch.nn.functional.rms_norm` end-to-end is 6.27 us and
a bare `out.add_(1.0)` launch is 4.52. So **the framework's launch floor alone
already equals torch's entire call**, and no amount of tightening the Quack
wrapper reaches parity at small shapes.

What *is* recoverable, measured by neutralizing each in situ: ~3.3 us from
`_validate_inputs` and ~3.0 us from the per-call `torch.empty(0)` absent-tensor
allocations, taking 26.5 us to 20.3. Real, worth doing eventually, and about a
third of the gap. The rest is `autograd.Function.apply` plus the FlyDSL floor.

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

| condition | TB/s |
| --- | --- |
| `c.copy_(a)`, no prior allocations | 4.718 |
| `c.copy_(a)`, with other tensors already resident | 5.363 |
| `c.copy_(a)` at 2 GiB | 4.833 |

A 14% swing from **allocation history alone**. Standalone `c.copy_`,
`a.clone()` and `c[:] = a` all agree at 4.71–4.72, so it is placement, not the
operation. That is the mechanism behind the four values: each was measured in a
process with a different heap, and each got written down as a property of the
card. The regenerated sidecar reads 5.579 because the probe runs after the
kernel's operands are allocated; `AI/probe_rmsnorm_roofline.py` records the
caveat inline so the number cannot be lifted out of context again.

The consequence is a rule, not a number: **do not use copy as a denominator.**
`write` (6.909) and `two_read_one_write` (6.052) reproduce across runs to
better than 1%; copy does not reproduce against itself. The section below was
written to argue that copy is too *low* to be a ceiling, which is true and
insufficient — the stronger objection is that it is not stable enough to be
anything.

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
154.94; the deltas are +0.66%, +0.18%, +0.00% and -0.11%, all inside these
rows' observed 0.26-1.86% round spread.)

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
It is 0 everywhere up to N=49152, then 8 at 57344 and 44 at 65536. On gfx950
the VGPR and AGPR banks share one 512-slot file per SIMD, so a nonzero AGPR
count is not free -- the honest allocation at 65536 is 344, not 300. The AGPRs
appear *exactly* at the cliff, which the old artifact could not have told
anyone.

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

Provenance: the treatment ladder now reproduces across two GPUs — device 6 in
`276398f` and device 5 here — to within 0.2% on every bandwidth figure, with
identical register and spill counts. The control's 3/4 rows are single-device so
far. This distinction is worth stating because the previous version of this
section sat next to a cross-reproduced occupancy table and was easy to read as
having the same backing.

**The timing regimes line up, and that is checkable rather than asserted.**
@Reviewer verified it on the device-6 run and it holds here: the intervention
times with cuda events, best of 10 × 20 inner calls, which is the eager regime,
and its unhinted rows land on the width-cliff sidecar's *eager* column to
+0.09% at 57344 and −0.21% at 49152, across two commits, two probes and two
devices. The cliff's graph column at the same shapes is 409.5 and 171.1 µs, so
at 57344 the intervention is unambiguously reading eager (0.09% vs 0.84%); at
49152 the two regimes are only 1.2% apart and the separation there is weaker
evidence than the first. This matters twice over: the intervention's bandwidth
percentages are directly comparable to the cliff's because they are the same
measurement of the same thing, and the unhinted baseline is demonstrably not
drifting. Worth noting the ceiling denominator is shared too — 6.075 TB/s from
the cliff sidecar's `two_read_one_write` probe, not a copy kernel, which on this
part reads MALL-inflated.

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

A unit conversion is order-preserving and invertible. Quantization is neither.
@Autotune found this, and the probe now *asserts* the collision rather than
leaving it to be noticed. The same fact explains why `512 // (2 * rocprof)`
reproduces the correct bound on all ten rows — `2 * rocprof` **is** the
allocation, so the cross-check was re-deriving a number the artifact already
stated. It is retained as a toolchain regression guard and nothing more.

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

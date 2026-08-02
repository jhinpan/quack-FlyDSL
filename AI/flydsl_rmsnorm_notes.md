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

Two traps, both of which produced a confident wrong answer at least once:

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
that from a clean run. @Reviewer's phrasing on 2026-08-02, refusing to take a
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

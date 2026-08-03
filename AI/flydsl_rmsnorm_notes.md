# FlyDSL RMSNorm backend notes

Findings from building the opt-in ROCm RMSNorm backend (`quack/flydsl/`,
`quack/rmsnorm_flydsl.py`). Most historical measurements below used MI355X /
gfx950, FlyDSL 0.2.4, and torch 2.9.1+rocm7.2.0. The current backend requires
FlyDSL 0.3 or newer; the controlled wide-row reproduction uses
0.3.0.dev765 and names its complete environment.

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
the old `flydsl<0.3` pin. The released 0.3 API is now required
(`flydsl>=0.3,<0.4`): the RMSNorm tuner relies on its public
`CompiledFunction` fast callable and the thread-local `CompilationContext`
hint overlay.

The numbers above were taken on FlyDSL
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

The backward is 1.16x-1.19x faster and the long-row forward is
parity. The one regression is the forward on that narrowest row, and it is in
the kernel
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

Each config is first compiled outside the timing region, and every gate,
warmup, graph capture and timed launch calls its `CompiledFunction` directly.
One search call builds and reuses a set of different tensor addresses across
all candidates. Exact shape/dtype/stride/device metadata is preserved, object
aliases (including the absent-residual alias to `x`) remain aliases, empty
placeholders remain empty, and every write target has independent storage.
Before timing a candidate, every address set is run and its output,
`residual_out` and `rstd` (when active) are checked against deterministic rows
of an independent fp32 reference.

The ROCm cache authority is Triton's active benchmark driver's eviction buffer:
256 MiB on the tested ROCm stack. This deliberately overrides the 4 MiB
`torch.cuda.get_device_properties().L2_cache_size` reported on gfx950, which is
the per-XCD L2 and not the chip's roughly 256 MiB MALL. Rotation is sized from
read operands only -- write-only outputs are still cloned but are not trusted
to allocate in MALL -- so the normal read-address plus eviction working set is
at least 3x the target. The plan is capped at 200 address sets and 4 GiB / 25%
of free HBM; a reduced-memory plan explicitly falls back to per-call eviction
rather than silently becoming cache-hot.

The timed graph records 200 round-robin kernel launches. A replay-based warmup
contributes about 200 ms of GPU work, then three graph replays are timed and
their median is returned as ms/call. Compilation, cloning, correctness checks,
warmup and cache-eviction writes are outside the event pair. Unsupported graph
capture and graph OOM use an event-timed multi-address fallback; if the address
pool alone cannot turn over one cache, that fallback evicts before each
individually timed call. Graph capture uses a device-local non-default stream,
and forced searches are serialized because CUDA/HIP permits one process-wide
capture at a time.

The selected config, defaults, disk winners, and portable artifacts retain
FlyDSL's normal semantics; a second process-local cache maps them to fast
callables. Compiler hints use `CompilationContext.compile_hints`, not
`flyc.compile[hints]`, because FlyDSL 0.3 implements the latter by mutating the
shared JIT's persistent hints.

The process cache includes the physical device index, tensor ABI/layout,
constexprs, config and compiler hints, toolchain and invalidating environment.
Each device gets a cloned JIT instance as well as a distinct cache entry, so two
same-architecture devices never share a loaded function pointer. Compilation
and cache publication use `FLYDSL_BUILD_LOCK`; `CompiledFunction`'s CallState is
itself thread-local. Schema version 3 prevents either dispatcher-scored schema
1 winners or same-address cache-hot schema 2 winners from being loaded.

The analytical heuristic lands on the best width or within 6% of it across the
candidates it may consider, and `waves_per_eu=None` won every forced search.
Within that space the search has almost nothing to find.

The exception is 512 threads per row, which the heuristic may not pick and the
tuner may. Inductor uses 512 at `32768x8192` and this backend could not: a block
above 256 threads needs `known_block_size` declared on the kernel, which
upstream FlyDSL passes and the vendored copy did not, so the AMDGPU default
refused the launch outright. With it declared, 512 measures 202.0us against
213.2us at the widest shape -- 1.06x, and 95% of the ceiling against 90%.

It cannot go in the heuristic. 512 wins only when the row is wide *and* there
are many rows: 1.06x at `32768x8192`, 0.98x at `4096x4096`, 0.21x at
`32768x256`. The heuristic sees only `N`, and `M` cannot enter it because `M` is
symbolic under `dynamic=True`. The tuner does see `M`, so 512 is offered there
and nowhere else. `MAX_TUNED_NUM_THREADS` is the separate ceiling that says so;
`MAX_NUM_THREADS` stays at 256 and no default choice changes.

Offering it needed one more guard. The tuner's own timing does not reliably
reject a block wider than the row has vectors: given 512 at `N=1024`, where 384
of the 512 lanes have nothing to load, it picked it and ran 0.70x. Candidates
now require `num_vecs >= threads`, which also drops several pre-existing
candidates that idled lanes for the same reason.

Validation on gfx950 with released FlyDSL 0.3.0 used BF16 activations and FP32
weights. At `256 x 4096`, public winner-cache-hit host enqueue fell from
154.997 us to 33.070 us, against 24.200 us for the heuristic entry. The old
128-136 us device-timing floor fell to 9-11 us on the three shortest shapes;
larger shapes now measure kernel work rather than JIT-key construction. Search,
cache hit with dynamic `eps`/`weight_offset`, residual+bias+prenorm+rstd on a
non-default stream, dynamic fullgraph rows, and same-architecture multi-device
isolation all have numerical regression coverage. Backward remains on its
existing deterministic heuristic and is not tuned in this stage.

The schema-3 retest used physical GPU 3 while its activity was 0%, five
provider-order-rotated rounds, and a 64 MiB add canary. The canary stayed within
0.56% (35.84-36.04 us). Values are public-call device latency in microseconds;
Triton's 256 MiB cache clear is outside each timed event:

| M x N | heuristic | L2-cold-tuned winner | `torch.compile` |
| --- | ---: | ---: | ---: |
| 1 x 4096 | 6.44 | 7.96 | 6.32 |
| 256 x 4096 | 6.60 | 7.40 | 6.56 |
| 512 x 4096 | 7.00 | 7.40 | 6.92 |
| 4096 x 3000 | 16.08 | 15.40 | 18.60 |
| 4096 x 4096 | 18.88 | 19.44 | 21.84 |
| 32768 x 1024 | 38.52 | 37.44 | 35.80 |
| 32768 x 2048 | 67.88 | 68.96 | 70.24 |
| 32768 x 4096 | 118.68 | 120.52 | 128.72 |
| 32768 x 8192 | 221.96 | 222.04 | 207.24 |

`32768 x 2048` produced a useful objective distinction. The requested
multi-address graph objective chose 64 threads in all three forced searches,
even after the read-only pool was enlarged to seven sets: 939.6 MiB of rotating
reads plus a 256 MiB eviction buffer. Default-occupancy timings were:

| threads | multi-address graph | clear-before-every-call |
| ---: | ---: | ---: |
| 64 | 49.91 us | 70.04 us |
| 128 | 51.54 us | 68.60 us |
| 256 (heuristic) | 54.61 us | 67.44 us |

So the expected 256-thread cold winner is true for an independently measured
clear-before-every-call objective, but not for the CuTe-style graph objective
implemented here. The latter is stable and is the persisted schema-3
definition, so the tuner correctly records 64 rather than forcing the
heuristic. Interposing a 256 MiB memset immediately before every launch changes
the ranking even though its own time is outside the event pair. At
`32768 x 8192`, 256 threads won 3/3 searches (the occupancy hint was 4, 4,
default); `256 x 4096` was a near-tie and changed from 128/default to 256/4 in
the next two searches.

The short-shape candidate objective still excludes public dispatch. On a
same-stream GEMM backlog, `256 x 4096` host enqueue was 23.98 us for the
heuristic entry and 32.71 us for the tuned cache hit: 8.72 us remains in
winner-key/environment handling, separate from candidate kernel latency.

Note that `benchmarks/benchmark_rmsnorm_flydsl.py` imports `rmsnorm`, not
`rmsnorm_autotuned`, so setting `FLYDSL_AUTOTUNE=1` around the benchmark
changes nothing. Comparing tuned against untuned needs a third column the
harness does not have.

## Measuring this backend

The suite figure for this revision is **649 passed, 3 skipped, 1 xfailed** when
one idle GPU is visible. The three skips require multiple visible devices; the
xfail is `test_simulated_cuda_flydsl_import_survives_a_broken_cutedsl_chain`,
which pins a real limitation rather than a passing behaviour (see the import
coupling note below):

```
HIP_VISIBLE_DEVICES=<idle> python -m pytest \
    tests/test_rmsnorm_flydsl.py tests/test_rmsnorm_flydsl_config.py \
    tests/test_import_isolation.py -q
```

`python -m pytest tests/` does *not* work on a ROCm host: 37 modules fail
collection with `ModuleNotFoundError: No module named 'cuda'`, because the
cutedsl tests import `cuda.bindings`. That is also why the cutedsl head-to-head
cannot be run on this machine.

## Pre-wide diagnostic sweep against torch.compile

Before the benchmark adopted the full CuTe shape ladder, this diagnostic sweep
used bf16 activations with fp32 weight on MI355X. Every cell was correctness
gated before timing. Speedup is torch.compile's time over ours:

| M x N | moved | fwd | bwd |
| --- | --- | --- | --- |
| 1 x 4096 | 40 KiB | 1.34x | 1.07x |
| 256 x 4096 | 4 MiB | 1.21x | 1.09x |
| 512 x 4096 | 8 MiB | 1.10x | 1.06x |
| 4096 x 3000 | 47 MiB | 0.88x | 1.13x |
| 4096 x 4096 | 64 MiB | 0.98x | 0.62x |
| 32768 x 256 | 32 MiB | 0.92x | 1.09x |
| 32768 x 512 | 64 MiB | 1.07x | 0.84x |
| 32768 x 1024 | 128 MiB | 1.00x | 0.99x |
| 32768 x 2048 | 256 MiB | 1.06x | 1.55x |
| 32768 x 4096 | 512 MiB | 0.98x | 1.48x |
| 32768 x 8192 | 1536 MiB | 0.94x | 1.66x |

The three diagnostic groups answered different questions: fixed `N` sweeping `M` is
the launch-bound end, `M=4096` is the band that is neither launch-bound nor
saturated, and fixed `M=32768` sweeping `N` is the bandwidth-bound end.
`4096x3000` is the only shape here that is not a power of two while still being
a multiple of `N_ALIGNMENT` (`gcd(3000, 8) == 8`), so it exercises the
predicated final tile. The current benchmark instead matches all eleven CuTe
shapes exactly; its widest target cell is reevaluated in the wide-row section
below.

Backward cells below `M=32768` are under the autograd host floor and should not
be read as kernel results -- `4096x4096` reads 0.62x here and 1.08x in the run
before it. The floor is measured in the next section.

**The forward is at parity and most of what is left is not in the kernel.**
Splitting our own forward time into the kernel and everything above it:

| M x N | `rmsnorm()` | kernel only | host | kernel GB/s | ceiling | % of ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| 1 x 4096 | 30.3us | 6.7us | **78%** | 5 | 5 | 101% |
| 256 x 4096 | 29.5us | 6.8us | **77%** | 621 | 614 | 101% |
| 512 x 4096 | 29.5us | 7.3us | **75%** | 1174 | 1180 | 100% |
| 4096 x 3000 | 29.1us | 15.0us | 49% | 3367 | 4101 | **82%** |
| 4096 x 4096 | 29.0us | 18.8us | 35% | 3823 | 4517 | **85%** |
| 32768 x 256 | 29.1us | 11.9us | 59% | 2977 | 3323 | 90% |
| 32768 x 512 | 28.9us | 16.1us | 44% | 4100 | 4417 | 93% |
| 32768 x 1024 | 29.0us | 36.1us | -- | 3703 | 3752 | 99% |
| 32768 x 2048 | 40.1us | 62.6us | -- | 4300 | 4271 | 101% |
| 32768 x 4096 | 94.7us | 114.4us | -- | 4683 | 4976 | 94% |
| 32768 x 8192 | 186.6us | 215.3us | -- | 5208 | 5580 | 93% |

The ceiling is measured per shape rather than taken from one number: the best
of `mul`, `add` and `abs` with `out=`, on tensors of that shape, which is the
same one-read-one-write traffic the forward moves. Earlier revisions of this
file divided by a flat 5279 GB/s taken from a `two_read_one_write` probe. That
is the wrong pattern for this kernel and it is not even the right magnitude:
at the `32768x8192` footprint three independent pointwise probes agree on
**5574 GB/s**, while `copy_` reads 4545 there and is the outlier, not the bound.

`rmsnorm()` has a flat ~29us floor from `M=1` to `M=4096`: validation, the
layout predicates, the launcher key and `autograd.Function.apply`, which
profiles as ~38 `isinstance` calls and three `torch.empty` per launch. Below
`M=4096` that floor is most of the measured time, so those cells report Python
rather than gfx950. Kernel-only numbers above `32768x1024` are pessimistic
instead: `do_bench` clears cache between iterations, which is real work at
those sizes, so the host column is left blank rather than reported negative.

Read against that, the room is not where the earlier revisions said. At
`M <= 512` the kernel is **at the ceiling** -- 100% to 101% -- so nothing there
is a kernel problem; the whole cost is the ~29us host floor, and torch.compile
pays about 25us of one itself, which is why it is only a few microseconds ahead
in that band. The largest kernel gap is `M=4096`, at 82% and 85%, and the two
widest shapes hold 93-94% against torch.compile's 98%, so 6-7% is left there and
Inductor demonstrates it is reachable.

Part of the `M=4096` gap is the row width. Five-repeat medians put 128 threads
per row ahead of the heuristic's 256 by 1.07x at `4096x3000` and 1.06x at
`4096x4096`. It is not a missing `M` term in the heuristic, though: sweeping `M`
at `N=4096` gives 256 the win at 1024, 2048, 8192, 16384 and 32768, and only
`M=4096` -- exactly 16 blocks per CU on 256 CUs -- prefers 128. A search finds
it, a rule does not obviously predict it.

### Against the upstream FlyDSL kernels these were vendored from

`quack/flydsl/` is adapted from `ROCm/FlyDSL` at `ddaa507`, where
`kernels/norm/rmsnorm_kernel.py` is 1970 lines and exports ten builders. This
backend is 479 lines and exports two, so the obvious reading is that it is a
subset. It is not: the two are a rewrite, and on plain RMSNorm they win.

Both builders on the eleven shapes, bf16 with fp32 weight. **The dispatch layer
has to match or the comparison is meaningless**: `build_rmsnorm_module` returns
a raw `@flyc.jit` function, and calling that re-resolves every argument into a
JIT cache key on each call, which is tens of microseconds. This backend routes
through `run_compiled`, which caches the compiled callable and skips that. The
"upstream" column below gives upstream the same treatment; the "raw jit" column
is what upstream's own tests call, kept because it is what a caller who copies
those tests will actually get.

| shape | plain: ours | upstream | raw jit | fused: ours | upstream |
| --- | --- | --- | --- | --- | --- |
| 1 x 4096 | 6.6us | 6.8us | 45.9us | 6.8us | 6.8us |
| 256 x 4096 | 6.8us | 6.9us | 47.9us | 7.2us | 7.2us |
| 512 x 4096 | 7.3us | 7.2us | 48.3us | 8.4us | 8.2us |
| 4096 x 3000 | 15.3us | 19.6us | 46.2us | 29.3us | 40.1us |
| 4096 x 4096 | 18.4us | 19.0us | 43.2us | 39.2us | 38.8us |
| 32768 x 256 | 12.0us | 13.9us | 39.6us | 17.8us | 37.1us |
| 32768 x 512 | 16.4us | 21.1us | 36.7us | 37.2us | 65.3us |
| 32768 x 1024 | 36.0us | 44.0us | 44.0us | 65.0us | 122.8us |
| 32768 x 2048 | 62.5us | 93.0us | 92.9us | 112.5us | 110.9us |
| 32768 x 4096 | 111.7us | 111.5us | 111.5us | 200.5us | 199.8us |
| 32768 x 8192 | 208.1us | 209.9us | 209.8us | 392.8us | 377.5us |

Read against equal dispatch, the two are close on plain RMSNorm and diverge on
the fused path. Plain: ties at `M <= 512` and at the two widest, and 1.16x to
1.49x here across the `M=32768` middle plus 1.28x at `4096x3000`. Fused: ties
at `M <= 512`, **2.08x at `32768x256`, 1.76x at `32768x512` and 1.89x at
`32768x1024`**, 1.37x at `4096x3000`, and a slight loss (0.96x-0.99x) at the
three widest.

The wins concentrate where upstream routes `N <= 2048` to its *dedicated*
narrow-row kernel, `_build_rmsnorm_large_m_small_n_module`, which packs 8-32
rows per block with `THREADS_PER_ROW = min(64, 1024 // BLOCK_M)`. Sizing the
lane group to the row in *vectors* and then packing rows to the block width the
one-row path targets beats that, and beats it hardest when the residual add is
fused in as well.

An earlier revision of this section reported 5.7x to 8.7x at `M <= 512`. That
was the raw-jit column against this backend's `run_compiled` path: entirely
FlyDSL's JIT dispatch, none of it the kernel. The raw-jit column is kept because
it is what upstream's own tests call and therefore what a caller copying them
gets, but it is a property of the dispatcher. The same asymmetry produced the
`rmsnorm_autotuned` regression fixed elsewhere in this file, and it is worth
naming as a pattern: comparing a pre-resolved launcher against a JIT-dispatched
one measures the dispatcher.

For reference on the same eleven shapes, `torch.compile` of a plain PyTorch
RMSNorm forward reads 10.4, 12.0, 12.3, 13.6, 20.5, 9.9, 15.3, 36.3, 66.6,
110.7 and 202.8us. It leads at `4096x3000`, `32768x256` and `32768x8192`, and
trails elsewhere. Upstream FlyDSL is not behind torch.compile once the dispatch
is matched either.

What is genuinely absent is the quantisation family. Upstream fuses RMSNorm
with dynamic and smooth quantisation, with and without the residual add
(`build_rmsnorm_dynamicquant_module`, `build_rmsnorm_smoothquant_module`,
`build_fused_add_rmsnorm_dynamicquant_module`,
`build_fused_add_rmsnorm_smoothquant_module`), and none of that is here. For
inference serving that is a real gap and a bigger one than layernorm; the plain
and fused-add paths are not.

### This backend helps eager callers and costs compiled ones

Wrapping `rmsnorm()` in `torch.compile` makes it *slower*, and the reason is
not a defect in this module. Measured at `512x4096`, cost added by compiling:

| graph contents | eager | compiled | added by compiling |
| --- | --- | --- | --- |
| pure aten | 16.6us | 26.7us | +10.1us |
| a minimal mutating custom op | 11.5us | 32.7us | +21.2us |
| the same with three mutated outputs | 15.3us | 35.0us | +19.6us |
| `rmsnorm()` | 29.7us | 57.9us | +28.2us |
| the custom op alone, no `autograd.Function` | 28.1us | 56.0us | +28.0us |

Any custom op in a compiled graph costs about 20us of Dynamo/AOTAutograd
wrapper on this build; the `autograd.Function` is not the cause, since removing
it changes nothing. Note this is a different fusion from the one the kernels do
well. *Inside* a launch this backend fuses bias, residual add, prenorm store,
per-head and `weight_offset` into one pass, on par with upstream FlyDSL doing
the same. *Across* launches it fuses nothing, because a custom op is opaque to
Inductor, and that is the fusion a compiled model is buying. Layer by layer, `rmsnorm()` under compile is a 12.3us
launcher, +12.0us of custom-op dispatch, +3.7us of Inductor's generated wrapper
and +30.4us of Dynamo/AOT. Inductor's generated code itself is optimal -- one
call to our op, no extra copy.

That microbenchmark understates the real cost, which is fusion. In a
transformer block with two RMSNorms around two matmuls (`512x4096`, three runs):

| | eager | compiled |
| --- | --- | --- |
| this backend | ~99us | ~105us |
| plain PyTorch | ~134us | **~67us** |

**Eager, this backend wins by 1.35x; compiled, it loses by 1.57x.** The kernel
counts are equal at four apiece, but Inductor fuses the reference's RMSNorm
math into the matmul epilogue --
`triton_tem_fused__to_copy_add_mean_mm_mul_pow_rsqrt_silu` -- while our op is
opaque and stands alone as `rmsnorm_kernel_0`. An opaque custom op is a fusion
barrier by construction, and at these shapes the fusion is worth more than the
kernel.

So the honest scope: this backend is for eager ROCm code. A ROCm user who runs
`torch.compile` is better off with plain PyTorch until the kernel is
expressible in something Inductor can fuse across, which is a different
integration than a custom op.

**The backward advantage is host-dependent, so treat the wide-row figure as a
range.** Two MI355X hosts running the same commit agree on our kernel and
disagree on torch.compile's:

| `32768x8192` bwd | ours | torch.compile |
| --- | --- | --- |
| host A | 0.3185 ms | 0.5302 ms (1.66x) |
| host B | 0.3363 ms | 0.3599 ms (1.07x) |

Ours reproduces across hosts to 5%; torch.compile's differs by 47%, stably on
both (host A read 0.5302/0.5357/0.5361/0.5512 across runs). `MAX_AUTOTUNE` is
not the cause -- disabling it made host A slower still -- and the cause is not
identified. The table above is host A, so read its backward column as the
optimistic end: `1.1x` to `1.7x` on wide rows depending on the host.

The narrow backward cells move with the host as well, and in the other
direction: host A shows `256x4096` at 1.19x, host B at 0.59x. Both are under
their own autograd floor (~0.09ms on host A, ~0.028ms on host B), so neither
is measuring the kernel. Check the floor by timing `M=1` before quoting
anything in that band.

Forward cells at `M=32768` reproduce to about 0.2% between sweeps; the
`M<=512` forward cells have read 294, 379, 412 and 462 GB/s across sweeps,
which is the same host floor seen from the other side.

A fused cell (`--features fused`: bias, residual, prenorm) is in the harness so
the feature path is measured rather than assumed. At `32768x4096` it reads 4911
GB/s against torch.compile's 5150 -- again parity, slightly behind.

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

## Wide rows stream instead of spilling

`MAX_N` is now 262144. The old 8192 cap matched the largest row whose whole
per-thread fragment fit the 32-element register-cache budget, but the kernel
continued caching after the cap was lifted. At N=262144 bf16 that meant 1024
fp32 values per forward thread and 512 values plus persistent parameter
accumulators per backward thread.

`rocprofv3` confirms that every optimized N=262144 kernel uses zero scratch:

| kernel | VGPR | scratch/thread |
| --- | ---: | ---: |
| forward | 12 | 0 B |
| backward correction | 32 | 0 B |
| backward partial | 28 | 0 B |
| parameter reduce | 24 | 0 B |

Forward rows above the budget now compute `sum_sq` in a device loop and reload
the activation for the epilogue. Backward first streams one row per block to
write its scalar correction, then a column-tiled persistent kernel computes dx
and bounded dweight/dbias partials; the existing deterministic parameter reduce
is unchanged. There are no fp32 atomics.

The original PR run reported 2.526/2.916 ms forward and 4.131/5.719 ms backward
for FlyDSL/torch.compile at `8192 x 262144`, but it recorded neither raw
samples, provider-order control, a steady-state warmup, nor a bandwidth
contention canary. The forward speedup does not survive a controlled rerun.

`benchmarks/benchmark_rmsnorm_flydsl.py --controlled` correctness-gates each
provider, settles clocks for three seconds, alternates provider order, rotates
two 4 GiB inputs, and measures 512 MiB bandwidth probes before and after:

```bash
PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=0 python benchmarks/benchmark_rmsnorm_flydsl.py \
  --controlled --M 8192 --N 262144 --dtype bfloat16 --weight_dtype float32
PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=0 python benchmarks/benchmark_rmsnorm_flydsl.py \
  --controlled --backward --M 8192 --N 262144 --dtype bfloat16 --weight_dtype float32
```

The same torch, HIP, FlyDSL, and Triton builds on two MI355X nodes produced:

| node | op | FlyDSL | torch.compile | torch / FlyDSL | BW canary |
| --- | --- | ---: | ---: | ---: | ---: |
| `mia1-p02-g23` | fwd | 2.448 ms | 2.392 ms | 0.977x | 0.993 |
| `smci355-ccs-aus-n08-09` | fwd | 2.576 ms | 2.425 ms | 0.941x | 1.004 |
| `mia1-p02-g23` | bwd | 3.957 ms | 5.206 ms | 1.316x | 0.998 |
| `smci355-ccs-aus-n08-09` | bwd | 4.162 ms | 5.409 ms | 1.299x | 1.005 |

Both nodes were healthy: their best bandwidth probes reached 6.47-6.80 TB/s,
all canaries stayed within 0.7% of one, and the fully idle remote node had no
KFD processes. The reproducible conclusion is therefore narrower: backward is
about 1.30x faster, while forward is 2.3%-5.9% slower than torch.compile at the
target cell. Logical GB/s remains provider-independent rather than physical
traffic because the wide FlyDSL forward deliberately performs an extra
streaming read.

## Where this backend stands against the cutedsl one

An earlier version of this section was a single "parity" table, and it was wrong
in four rows and silent about the two limits that matter most to a caller. The
three things it conflated are separated here: what the public API accepts, what
inputs the backend serves, and how the kernels are built.

### 1. Public API

`quack/rmsnorm.py` exports 12 functions plus `QuackRMSNorm`; this backend
exports `rmsnorm` and `rmsnorm_autotuned`.

| surface | cutedsl | FlyDSL |
| --- | --- | --- |
| `rmsnorm()` | yes | yes, argument for argument |
| `rmsnorm_fwd` / `rmsnorm_bwd` | yes | **no** |
| `rmsnorm_ref` / `rmsnorm_bwd_ref` | yes | **no** |
| `QuackRMSNorm` (`torch.nn.RMSNorm` drop-in) | yes | **no** |
| `layernorm_fwd` / `_bwd` / `_ref` / mean | yes | **no** |

Only the top-level entry matches. Three consequences a caller feels:

- **`rstd` cannot be retrieved.** cutedsl's `rmsnorm_fwd(store_rstd=True)`
  returns it; here `store_rstd` is set to `needs_grad` internally and never
  surfaces. The kernel supports it, the API does not expose it, so the old
  "store_rstd | parity" row was wrong.
- **No reference implementation ships.** Tests and the benchmark each carry
  their own, where cutedsl users share `rmsnorm_ref`.
- **The benchmark has no low-level entry to time**, so both of its providers go
  through `rmsnorm()` and autograd. That is internally fair but not comparable
  to `benchmark_rmsnorm.py`, which times `rmsnorm_fwd` directly.

### 2. Inputs served

| input | cutedsl | FlyDSL |
| --- | --- | --- |
| row width | any `N`, to 262144 in its own benchmark | `N <= 262144` |
| row alignment | any `N` (gcd vectorization, predicated tail) | **multiple of 8, or 4 for fp32** |
| architecture | SM80..SM100 | **gfx950 only** |
| activation/weight dtypes | fp16/bf16/fp32 and more | fp16/bf16/fp32, any pairing |
| `eps` | unvalidated | must be finite and positive |

The wide-row path now covers the full benchmark ladder, including ordinary
12288/16384 hidden sizes. Alignment remains stricter than cutedsl: any `N` that
is not a multiple of 8 is still refused.

### 3. Implementation mechanisms

| mechanism | cutedsl | FlyDSL |
| --- | --- | --- |
| bias, residual (fused add), prenorm | yes | yes |
| weight_offset (`w+1` fusion) | yes | yes |
| per-head affine | one kernel, symbolic head extent | one kernel **per head count** |
| autotune | forward **and** backward tuners | **forward only**, and off unless `FLYDSL_AUTOTUNE=1` |
| persistent backward | `sm_count` is a caller knob | computed internally, **no knob** |
| split parameter reduce | `dw_partial` | same mechanism, spelled `partial`/`dweight_total` |
| dual dx dtype | internal `dx_dtype` only | not exposed either -- **neither public API reaches it** |
| layernorm / mean | yes | **no** |
| cluster / multicast | SM90+ DSMEM | n/a on gfx950 |

Two rows worth expanding. **Per-head recompiles**: cutedsl makes the head extent
a `cute.sym_int()`, so one compiled kernel serves every head count, while
`_FWD_CACHE` here keys on the literal `num_heads` and builds again for each --
a cost for anyone serving several model configs in one process. **`dual dx
dtype`** was previously graded "narrower" by comparing this backend's top-level
entry against cutedsl's internal custom op; measured against cutedsl's own
`rmsnorm()`, neither exposes it.

`cluster` is a real difference but not a deficit: it is Hopper-and-later
distributed shared memory, which gfx950 does not have.

**Layernorm remains a genuine gap** -- `quack/rmsnorm.py` carries `is_layernorm`
through the whole stack with `mean` alongside `rstd` -- it is just no longer
the only one worth naming.

### The import isolation only holds in one direction

Importing `quack` does not import FlyDSL, which is the direction the module
docstring claims and the direction that matters on a CUDA host. The reverse is
not true: on a CUDA host, `import quack.rmsnorm_flydsl` runs the CuTe bootstrap
in `quack/__init__.py` first and fails if the installed cutlass is mismatched,
even though this backend has no cutlass dependency. It is moot on ROCm, where
the bootstrap is skipped entirely, and it is pinned by a strict xfail in
`tests/test_import_isolation.py` rather than left undocumented. Fixing it means
making the CuTe exports lazy, which is a change to the CUDA path and out of
scope here.

# FlyDSL RMSNorm backend notes

Findings from building the opt-in ROCm RMSNorm backend (`quack/flydsl/`,
`quack/rmsnorm_flydsl.py`). Measured on MI355X / gfx950 with FlyDSL 0.2.4 and
torch 2.9.1+rocm7.2.0 unless stated otherwise.

## Quack RMSNorm API coverage

The FlyDSL entry point implements the upstream `quack.rmsnorm` signature,
including optional weight and bias, `weight_offset`, independent output dtype,
residual/prenorm output, residual dtype override, and per-head parameters.
The original plain weighted path keeps its vectorized/small-N kernels; feature
combinations use a descriptor-safe kernel specialized by compile-time feature
flags.

Backward keeps the small-row atomic path and uses a deterministic persistent
partial plus final reduction for large row counts or deterministic mode.
Per-head workspaces and final parameter-gradient stores use row-scoped buffer
descriptors so neither temporary nor output addressing silently wraps at 4 GiB.

## The baseline for a feature kernel is the compiler, not the plain path

The plain path cannot express bias, residual, prenorm, per-head or dtype
overrides at all, so it is not what a caller gives up by using them. What they
give up is `torch.compile`, which fuses this shape of work close to the
roofline. That is the number to beat, and it is a demanding one: inductor
reaches 4.8 TB/s on fused residual + prenorm, above anything this backend
achieves on any other shape.

A first cut of the feature path was scalar and re-read the row from global
memory after the reduction, which put it well under that bar. Measured at
32768x2048 bf16, forward, against `torch.compile`:

| case | scalar first cut | vectorized | `torch.compile` |
| --- | --- | --- | --- |
| `+bias` | 2.89 TB/s (0.72x) | 4.36 TB/s (1.08x) | 4.02 TB/s |
| `weight_offset=1` | 3.04 TB/s (0.77x) | 4.36 TB/s (1.11x) | 3.93 TB/s |
| `+residual+prenorm` | 2.71 TB/s (0.57x) | 4.75 TB/s (1.00x) | 4.77 TB/s |

Three things closed that gap, and all three are visible in the plain path
already:

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

The atomic backward is still scalar. It only runs for fewer than 512 rows
outside deterministic mode, where launch overhead dominates anyway.

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

## Vector size follows `quack/rmsnorm.py`

`vecsize = gcd(N, 128 // dtype_width)`, the same rule the CuTe kernel uses. A
row that is not a whole number of 128-bit vectors degrades to a narrower
access rather than collapsing to scalar. Worth about 1.5x on hidden sizes
divisible by 2 or 4 but not 8:

| shape | scalar fallback | gcd rule |
| --- | --- | --- |
| 32768 x 4092 bf16 | 159.5 us | 107.3 us |
| 32768 x 4094 bf16 | 161.9 us | 107.9 us |
| 32768 x 8188 bf16 | 302.3 us | 205.1 us |

## Persistent kernels need their block sized to the row too

The staged backward first pinned its block at 512 threads. A 1024-wide bf16
row is 128 vectors, so three quarters of the block had no column and the
kernel fell back to scalar I/O, landing at 29% of the copy roofline. Sizing
the block to the row and launching proportionally more blocks took
32768x1024 fp16 from 137.5 us to 62.4 us (2.20x, 29% -> 62% of roofline).

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
which gates correctness per cell and evicts L2 between timed calls. Copy
roofline: MI355X 5279 GB/s, H200 4148 GB/s, H100 2974 GB/s.

| regime | FlyDSL on MI355X | Quack on H200 | Quack on H100 |
| --- | --- | --- | --- |
| M=32768 (bandwidth bound) | 100% / 88% | 80% / 74% | 89% / 87% |
| M=4096 (not saturated) | 71% / 64% | 33% / 27% | 61% / 41% |
| M<=512 (launch bound) | 7% / 5% | 14% / 9% | 18% / 12% |

Percentages are forward / backward share of that machine's own copy roofline.
Large shapes are decided by HBM; mid shapes by the kernel; small batches are
pure launch path, where the CuTe kernel is about 2.2x ahead (6.1 us against
13.0 us at M=1, against a ~3.5 us Python/FFI floor).

Caveat worth keeping: at 4096x3000 and 4096x4096 Quack is slower on the H200
than on the H100 despite 1.39x more bandwidth, while torch on the same two
boxes moves the right way. That is Quack tuning on H200, not the machine, and
those cells are excluded from any median quoted above.

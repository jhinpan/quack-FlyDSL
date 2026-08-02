# The benchmark harness does not produce cold reads on gfx950

Status: **confirmed by measurement**, fix not yet written. Blocks Experiment
No.002 (MI355X flydsl-vs-torch matrix) — any MI355X numbers taken before this
is fixed will overstate bandwidth on every shape below `32768x2048`.

## Summary

`benchmarks/benchmark_rmsnorm_flydsl.py` sizes both its rotation buffers and its
L2 evictor from `torch.cuda.get_device_properties().L2_cache_size`. On gfx950
that property reports **4 MiB**, which is the *per-XCD* L2. It does not report
the device-wide **256 MiB MALL / Infinity Cache** that sits behind it. Two
independent consequences follow, and they compound.

The cache hierarchy on this part (ROCm Kernel Wiki `hw-chiplet-xcd`):

    per-CU L1D -> per-XCD 4 MiB L2 (x8 XCDs in SPX) -> 256 MiB MALL -> HBM

## Defect 1: the evictor never runs on the shapes that need it

    l2_target_bytes = properties.L2_cache_size * args.l2_target_ratio   # 4 MiB * 3 = 12 MiB
    use_evictor = rotation_working_set_bytes < l2_target_bytes

The evictor is gated to run only when the rotation working set is *smaller than
12 MiB*. Every mid-size cell has a working set far above 12 MiB but far below
the 256 MiB MALL, so it lands in the gap: too big to trigger the evictor, small
enough to stay resident in MALL. For `4096x4096` bf16 forward at 2 buffers the
working set is 128 MiB — the gate evaluates `128 MiB < 12 MiB` = False, so no
eviction happens, and 128 MiB fits comfortably in the 256 MiB MALL.

The evictor is also only 12 MiB, which cannot flush a 256 MiB cache even when
it does run.

## Defect 2: default rotation counts cannot clear the MALL

Defaults are `--min-rotation-buffers 2 --max-rotation-buffers 4`. Working set is
`rotation_buffers * bytes_per_call`. Buffers needed to exceed 256 MiB, bf16:

| shape | fwd bytes/call | bwd bytes/call | fwd bufs needed | bwd bufs needed |
| --- | --- | --- | --- | --- |
| `1x4096` | 16 KiB | 24 KiB | 16385 | 10924 |
| `256x4096` | 4.0 MiB | 6.0 MiB | 65 | 44 |
| `512x4096` | 8.0 MiB | 12.0 MiB | 33 | 23 |
| `4096x3000` | 46.9 MiB | 70.3 MiB | 7 | 5 |
| `4096x4096` | 64.0 MiB | 96.0 MiB | 5 | 4 |
| `32768x1024` | 128.0 MiB | 192.0 MiB | 3 | 3 |
| `32768x2048` | 256.0 MiB | 384.0 MiB | 2 | 2 |
| `32768x4096` | 512.0 MiB | 768.0 MiB | 2 | 2 |
| `32768x8192` | 1024.0 MiB | 1536.0 MiB | 2 | 2 |

With the cap at 4, only `m=32768` clears the MALL. `32768x1024` needs 3 and is
borderline-clean; everything at `m<=4096` cannot be fixed by rotation at all —
`256x4096` would need 65 buffers, `1x4096` sixteen thousand. **Rotation count is
not a viable lever below `m=32768`; the evictor is the only one.**

## Measurement

`AI/probe_gfx950_mall_rotation.py`, MI355X (gfx950), one GPU, SPX/NPS1.
Kernel held *exactly* fixed (`copy_` between rotating buffer pairs); only the
number of rotation buffers varies, so any systematic difference is cache
residency and not kernel selection.

    ### buffer = 64 MiB
     buffers    working set   vs MALL      GB/s
           1         128 MiB     0.50x      6825
           2         256 MiB     1.00x      6834
           3         384 MiB     1.50x      5124     <- step here
           4         512 MiB     2.00x      5053
           8        1024 MiB     4.00x      5032
          32        4096 MiB    16.00x      4961

The honest figure is the **boundary step between adjacent rotation counts**:
2 buffers (256 MiB) = 6834 GB/s vs 3 buffers (384 MiB) = 5124 GB/s, one
rotation apart, same kernel — **1.33x inflation**. The 16 MiB sweep shows the
same step at the same place and reaches 1.75x, but its single-buffer row is
additionally inflated by fitting the 32 MiB aggregate L2, so it should not be
the headline.

A correctly-sized evictor recovers the HBM number:

    64 MiB x2 bufs (WS=256 MiB), evictor=0 MiB      6003 GB/s   <- current behaviour
    64 MiB x2 bufs (WS=256 MiB), evictor=12 MiB     4856 GB/s   <- current evictor size
    64 MiB x2 bufs (WS=256 MiB), evictor=256 MiB    4456 GB/s
    64 MiB x2 bufs (WS=256 MiB), evictor=512 MiB    4346 GB/s
    64 MiB x8 bufs (WS=1024 MiB), no evictor        4565 GB/s   <- HBM reference

A 256–512 MiB evictor lands at 4346–4456 GB/s against a 4565 GB/s HBM
reference. The 12 MiB evictor recovers roughly two thirds of the gap, which is
why the defect is easy to miss rather than glaring.

## A false start worth recording

The first probe swept `x.sum()` over buffer sizes and found a *sharper* cliff at
256 MiB — apparently stronger evidence. It was discarded, for two reasons:

1. `sum()` is a reduction; torch may pick different kernels/strategies at
   different input sizes, so size and kernel varied together and the cliff was
   not attributable to the cache.
2. Its absolute numbers (3796 GB/s at 2 GiB) sat ~40% below both the wiki's
   independent microbench on this part (6192 GB/s at 1 GiB) and this repo's own
   write probe (6587 GB/s). A measurement that far under two independent
   ceilings is kernel-limited, not HBM-limited.

The wiki's microbench holds its kernel fixed and reports 6152 GB/s at 64 MiB vs
6192 GB/s at 1024 MiB — **flat across the boundary**, the opposite shape to my
first probe. That disagreement is what prompted the controlled rerun. The
controlled probe reproduces a step where the uncontrolled one showed a cliff,
and its absolute values (4961–6834 GB/s) are consistent with the two known
ceilings.

Why the wiki's curve is flat while this one steps: the wiki kernel is a
*read-only* non-temporal stream with an `nt` policy hint. Per the wiki's own
caveat, `nt` is a hint that does not promise to bypass MALL, but a read-only
stream with no write traffic has a very different MALL hit profile from a
read+write copy. These do not contradict each other; they are different access
patterns. The harness's pattern is read+write, so the stepping curve is the
relevant one.

## Fix options

The MALL size does **not** need hardcoding. `rocminfo` reports it as `L3` per
agent, and it parses cleanly:

    arch=gfx950  L2=4 MiB (per-XCD)  L3/MALL=256 MiB

1. **Size the evictor from the largest reported cache level, not `L2_cache_size`.**
   Derive an effective LLC (`L3` if present, else `L2`) and target a multiple of
   *that*. Also fix the `use_evictor` gate, which currently compares against the
   same undersized number and so switches eviction off precisely where it is
   needed. This is the option I favour: it is one sizing function, it fixes both
   defects, and it degenerates to current behaviour on NVIDIA where no L3 is
   reported.

2. **Force the rotation working set past the MALL.** Rejected as a primary fix —
   the table above shows it is arithmetically impossible below `m=32768`, and
   raising `max_rotation_buffers` to 65 for `256x4096` would allocate absurd
   amounts of memory to solve a problem the evictor solves directly.

3. **Treat memory-side cache as a separate regime and report it.** Not a fix,
   but worth doing alongside 1: record effective-LLC bytes and the resulting
   working-set ratio per cell in the results CSV so a later reader can tell
   whether a cell was HBM-bound without re-deriving it.

Recommendation: **1 + 3**.

## Consequence for existing numbers

Any MI355X figure in `AI/flydsl_rmsnorm_notes.md` at `m <= 4096` is measured
partly against MALL and is optimistic. The `M=32768` row (100%/88%) is
unaffected — its working sets are >=512 MiB at 2 buffers. The `M=4096` row
(71%/64%) and `M<=512` row (7%/5%) are affected, though at `M<=512` the cells
are launch-bound and bandwidth is not the binding constraint anyway, so the
practical distortion is concentrated in the **`M=4096` row**.

This also means percent-of-roofline is not comparable between the MI355X column
and the two Hopper columns of that table, on top of the already-documented
"different roofline probes, do not compare across hosts" caveat.

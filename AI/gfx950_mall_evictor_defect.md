# The benchmark harness does not produce cold reads on gfx950

Status: **confirmed by measurement**, fix not yet written. Blocks Experiment
No.002 (MI355X flydsl-vs-torch matrix) — any MI355X numbers taken before this
is fixed overstate bandwidth on 11 of the 18 benchmarked cells, including one
`m=32768` cell. A second code path (`quack/autotuner.py`, 8 of 18) shares the
root cause; see below.

## Summary

`benchmarks/benchmark_rmsnorm_flydsl.py` sizes both its rotation buffers and its
L2 evictor from `torch.cuda.get_device_properties().L2_cache_size`. On gfx950
that property reports **4 MiB**, which is the *per-XCD* L2. It does not report
the device-wide **256 MiB MALL / Infinity Cache** that sits behind it. Two
independent consequences follow. They were originally written up as compounding;
the controlled measurement (below) attributes the observed 1.32x to **defect 1**
— the evictor never runs — and does *not* show defect 2's undersizing costing
anything measurable on this access pattern. Defect 2 is still a real
mis-derivation and should be fixed, but it should not be credited with the
inflation.

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

The evictor is also only 12 MiB, which on the face of it cannot flush a 256 MiB
cache even when it does run. That reasoning turns out not to survive
measurement: forcing a 12 MiB evictor to run at the boundary recovers almost
all of the gap (4891 GB/s against a 4964 GB/s HBM reference), within ~2% of
what a 256 MiB evictor achieves. A copy-based evictor evidently disturbs MALL
residency out of proportion to its own footprint. The binding problem is the
gate, not the size.

## Defect 2: default rotation counts cannot clear the MALL

Defaults are `--min-rotation-buffers 2 --max-rotation-buffers 4`. Working set is
`rotation_buffers * bytes_per_call`. What matters is not "buffers needed" in the
abstract but **what the harness actually picks**, since `by_target` is derived
from the same undersized 12 MiB figure. Simulating `_rotation_count` exactly:

Earlier versions of this table used `m*n*2*{2,3}` as an approximation. That was
wrong in a way that mattered — it dropped the weight and `rstd` terms and so
placed cells on the wrong side of the boundary. The table below uses the
harness's real `logical_bytes` (`benchmark_rmsnorm_flydsl.py:116`) and also
accounts for `use_evictor`, which the earlier version ignored entirely. Shown
for **bf16/same**; the other 16-bit modes give the same verdicts.

| shape | op | B/call | picked | working set | vs MALL | regime |
| --- | --- | --- | --- | --- | --- | --- |
| `1x4096` | fwd | 0.023 MiB | 4 | 0.094 MiB | 0.00x | evicted |
| `1x4096` | bwd | 0.039 MiB | 4 | 0.156 MiB | 0.00x | evicted |
| `256x4096` | fwd | 4.008 MiB | 3 | 12.023 MiB | 0.05x | inflated |
| `256x4096` | bwd | 6.017 MiB | 2 | 12.033 MiB | 0.05x | inflated |
| `512x4096` | fwd | 8.008 MiB | 2 | 16.016 MiB | 0.06x | inflated |
| `512x4096` | bwd | 12.018 MiB | 2 | 24.035 MiB | 0.09x | inflated |
| `4096x3000` | fwd | 46.881 MiB | 2 | 93.761 MiB | 0.37x | inflated |
| `4096x3000` | bwd | 70.340 MiB | 2 | 140.679 MiB | 0.55x | inflated |
| `4096x4096` | fwd | 64.008 MiB | 2 | 128.016 MiB | 0.50x | inflated |
| `4096x4096` | bwd | 96.031 MiB | 2 | 192.062 MiB | 0.75x | inflated |
| `32768x1024` | fwd | 128.002 MiB | 2 | **256.004 MiB** | 1.00x | see below |
| `32768x1024` | bwd | 192.129 MiB | 2 | 384.258 MiB | 1.50x | clean |
| `32768x2048` | fwd | 256.004 MiB | 2 | 512.008 MiB | 2.00x | clean |
| `32768x2048` | bwd | 384.133 MiB | 2 | 768.266 MiB | 3.00x | clean |
| `32768x4096` | fwd | 512.008 MiB | 2 | 1024.016 MiB | 4.00x | clean |
| `32768x4096` | bwd | 768.141 MiB | 2 | 1536.281 MiB | 6.00x | clean |
| `32768x8192` | fwd | 1024.016 MiB | 2 | 2048.031 MiB | 8.00x | clean |
| `32768x8192` | bwd | 1536.156 MiB | 2 | 3072.312 MiB | 12.00x | clean |

Counting the **full 90-cell matrix** (9 shapes x 2 ops x 5 dtype/weight modes)
rather than one dtype:

- 47 of 90 cells have a working set `<= 256 MiB`
- of those, **10 actually run the evictor** — the `m=1` cells, whose sets are
  small enough to satisfy `ws < 12 MiB`
- so **37 of 90** are both un-evicted and MALL-resident

Per mode that is **8 of 18** for each of the four 16-bit modes and **5 of 18**
for `float32/same`. The previously published "11 of 18" was produced by the
approximate byte count and by ignoring `use_evictor`; it is withdrawn.

**`32768x1024` forward: measured, not inferred.** @Reviewer was right that the
cell is at **256.004 MiB**, not exactly 256.0 — the extra 4 KiB is the weight
row, which my approximate byte count had dropped — and right that a set
marginally *over* capacity cannot be classified by a `<= 256 MiB` rule when the
probe only sampled 256 and 384 MiB. So I measured it directly, at the exact
harness working set of 268439552 bytes:

    WS bytes       WS MiB      GB/s
    268435456      256.00000   6394
    268437504      256.00195   6385
    268439552      256.00391   6325   <- 32768x1024 bf16/same fwd, x2 buffers
    268439552      256.00391   6295   (repeat)
    268443648      256.00781   6401
    268500992      256.06250   6161
    268697600      256.25000   5960
    269484032      257.00000   5409
    (for reference: 288 MiB 4987, 384 MiB 5055)

The cell reads **6295–6325 GB/s**, firmly on the inflated side — being 4 KiB
over capacity does not evict it. So the verdict "inflated" stands for this cell,
but it now rests on a measurement of that specific working set rather than on a
threshold rule.

Worth recording that the boundary is **not a cliff**: throughput decays across
roughly 256 → 288 MiB rather than stepping at one point. My earlier
"exactly 256.0 MiB, precisely the boundary" framing implied a sharpness the data
does not show, and the adjacent-rotation step figure (1.32x) is a chord across
that decay, not the height of a discontinuity.

Two earlier claims are withdrawn outright: "only `m<=4096` is affected" /
"`m=32768` is unaffected", and the "11 of 18" count.


That earlier version also carried an off-by-one in a "buffers needed" table
(`ceil(MALL/bytes)+1`, which overshoots whenever the division is exact): it
listed 7/5 for `4096x3000` where the correct values are 6/4, and 5/4 for
`4096x4096` where they are 5/3. The table above avoids the issue by simulating
the actual selection rather than computing a requirement.

Rotation count still is not a usable lever at small `m` — `256x4096` would need
65 forward buffers and `1x4096` over 16000 — so **the evictor remains the only
practical fix**. But the reason is the size of the requirement, not an
`m=32768` cutoff.

## Measurement

`AI/probe_gfx950_mall_rotation.py`, MI355X (gfx950), one GPU, SPX/NPS1,
`HIP_VISIBLE_DEVICES=7`, torch 2.9.1+rocm7.2.0. Kernel held *exactly* fixed
(`copy_` between rotating buffer pairs); only the number of rotation buffers
varies, so any systematic difference is cache residency and not kernel
selection. Every figure below is emitted by that script, which also writes
`AI/probe_gfx950_mall_rotation.json` with per-repeat raw samples and the
environment (including `rocminfo`'s `L3: 262144 KB`, i.e. the MALL is
discoverable rather than assumed). Timing follows `_time_rotating_calls`: the
window covers one whole rotation and the evictor runs *outside* it.

    ### buffer = 64 MiB
     buffers    working set   vs MALL      GB/s
           1         128 MiB     0.50x      6090
           2         256 MiB     1.00x      6453
           3         384 MiB     1.50x      4896     <- step here
           4         512 MiB     2.00x      4942
           8        1024 MiB     4.00x      4960
          32        4096 MiB    16.00x      5030

The honest figure is the **boundary step between adjacent rotation counts**:
2 buffers (256 MiB) = 6453 GB/s vs 3 buffers (384 MiB) = 4896 GB/s, one
rotation apart, same kernel — **1.32x inflation**. The 16 MiB sweep steps at
the same place (8 bufs = 256 MiB, 5309 GB/s → 12 bufs = 384 MiB, 4008 GB/s)
and gives **1.32x**, independently.

**Independently reproduced by @Autotune**, with a cleaner control than mine:
bytes moved *per iteration* held constant at 32 MiB while only the buffer count
varies, so iteration cost cannot co-vary with working set. MI355X, bf16 copy,
7 repeats, median (min/max spread <1.5% at every point):

     n_bufs   in+out MiB   GB/s med    min     max
          3          192       6624    6602    6627
          4          256       6555    6494    6562
          5          320       4680    4658    4699
          6          384       4658    4594    4678
          7          448       4708    4661    4723

Same knee at 256 MiB, −28.6% across it, flat thereafter out to 1024 MiB. Their
`rocminfo` node also reports `L1 32 KB / L2 4096 KB / L3 262144 KB`. Note the
256 MiB point reads 6555 — on the *high* side, agreeing with my 6453 — so a
cell landing at exactly 256.0 MiB is MALL-warm, not clean. That is a diagnostic
probe on a shared box, not a PR-grade number.

A correctly-sized evictor recovers the HBM number:

    64 MiB x2 bufs (WS=256 MiB), evictor=0 MiB      6403 GB/s   <- current behaviour
    64 MiB x2 bufs (WS=256 MiB), evictor=12 MiB     4891 GB/s   <- current evictor size
    64 MiB x2 bufs (WS=256 MiB), evictor=256 MiB    4726 GB/s
    64 MiB x2 bufs (WS=256 MiB), evictor=512 MiB    4797 GB/s
    64 MiB x2 bufs (WS=256 MiB), evictor=1024 MiB   4335 GB/s
    64 MiB x8 bufs (WS=1024 MiB), no evictor        4964 GB/s   <- HBM reference

Evicting at all is what matters here: with no evictor the boundary cell reads
6403 GB/s against a 4964 GB/s HBM reference, and *any* of the evictor sizes
brings it to 4335–4891 GB/s, i.e. to within about 12% of that reference. The
12 MiB evictor is not obviously worse than the 256–1024 MiB ones on this
access pattern, so the measured defect is **defect 1** — the `ws < 12 MiB`
gate means no evictor runs on these shapes at all. Defect 2 (the evictor being
sized off the per-XCD L2) is a real mis-derivation but is not, on this
evidence, what costs the 1.32x. An earlier version of this block reported
280–1070 GB/s at the larger evictor sizes; that was my own bug — the evictor
was being run inside the timed window, so those numbers measured the evictor
itself, not the copy.

## A second code path with the same root cause

`quack/autotuner.py` does not share the harness's rotation logic. It calls
`_pick_l2_rotate_count` (`quack/bench/bench_utils.py:179`), which makes the same
`L2_cache_size` mistake — `target_ratio * l2_size` = 12 MiB on gfx950 — but with
different bounds: `min_buffers=4, max_buffers=16` against the harness's
`min=2, max=4`.

> **Retracted pending recomputation (raised by @Reviewer, verified).** This
> section previously gave a per-cell count for the autotuner path. Two errors,
> both mine:
>
> 1. **"`n_by_l2 == 1` in every cell / always picks 4 / `max_buffers=16` is
>    inert" is false, and my own printed output contained the counterexample.**
>    The formula is `ceil(12 MiB / tensor_bytes)`, so it grows as the tensors
>    shrink. At `1x4096` a set is ~0.016 MiB, giving `n_by_l2 = 768`, which
>    clips at **`max_buffers=16`** — the ceiling binds, not the floor. My table
>    printed `nb=16` for that row while the prose beside it said "4 everywhere".
>    The floor does win for the mid and large shapes, but "everywhere" was the
>    same all/almost-all overreach I keep committing.
> 2. **I substituted the harness's `logical_bytes` for what this helper actually
>    counts.** `_pick_l2_rotate_count` sums *every* Tensor in both `args` and
>    `kwargs` of the real tuned call — forward has at least `x`, `weight`, `out`;
>    backward adds `dout`, `rstd`, `dx` and the partials. My 18-cell table used
>    `m*n*2*{2,3}`, an approximation of a different quantity. So `8 of 18` and
>    "`4096x4096` fwd lands at exactly 256.0 MiB" are not established, and the
>    "affected count goes 8 → 6 under an LLC target" figure inherits the same
>    defect.
>
> What survives: the helper does use `L2_cache_size` and therefore inherits the
> per-XCD/MALL confusion; this path has **no evictor at all**, only rotation;
> and `max_buffers` has to move together with any target change (see below,
> which is a statement about the formula rather than a per-cell count). The
> counts must be recomputed from the actual argument sets of a real tuned call,
> instrumented rather than modelled. Not yet done.

**Changing `l2_size` alone is not sufficient here, and `max_buffers` must move
with it.** @Autotune raised this and the mechanism holds independently of the
cell counts: with the target at `3 x 256 MiB = 768 MiB`, any call whose cloned
tensor set is under 48 MiB wants more than 16 buffers and is clipped by
`max_buffers=16` — and the smallest shapes have the smallest sets, so the
clipping lands exactly where the most rotation is needed. Memory is not the
obstacle: the working set at the crossing point is 256 MiB by construction
(~0.26 GiB), so the awkward part is the buffer *count*, not the bytes.

Scope note: `quack/autotuner.py` is not in PR #4's diff (that PR touches
`quack/flydsl/rmsnorm_autotune.py`), so this is not blocked by the PR #4 fence.

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
and its absolute values (4008–6453 GB/s) are consistent with the two known
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

1. **Fix the `use_evictor` gate first; re-size the evictor second.**
   The gate is the part the measurement actually indicts. `use_evictor =
   ws < l2_target_bytes` switches eviction off precisely where it is needed, and
   the evictor-control block shows that *running an evictor at all* is what
   recovers the HBM number — 12 MiB gets within ~2% of the 256/512 MiB results.
   So the gate should be driven by whether the working set fits the effective
   LLC, not by whether it is smaller than a multiple of the per-XCD L2.

   Re-sizing off an effective LLC (`L3` if present, else `L2`) is still correct
   and should ship with it — the current derivation is wrong on its own terms,
   and it degenerates to today's behaviour on NVIDIA where no L3 is reported —
   but it should be presented as correcting a mis-derivation, not as the thing
   that buys back the 1.32x. On this access pattern it does not.

2. **Force the rotation working set past the MALL.** Still rejected for the
   *harness*, where the evictor solves the problem directly — but my original
   reason was wrong and should not be reused. I wrote that raising
   `max_rotation_buffers` to 65 for `256x4096` "would allocate absurd amounts of
   memory". It would not: the working set at the crossing point is 256 MiB by
   construction, so the allocation is ~0.26 GiB regardless of buffer count. What
   is awkward at small `m` is the *count* (65 buffers at `256x4096`, ~16k at
   `1x4096`), not the bytes. For the autotuner path, which has no evictor, this
   is the only available lever and the memory cost is not an objection to it.

3. **Treat memory-side cache as a separate regime and report it.** Not a fix,
   but worth doing alongside 1: record effective-LLC bytes and the resulting
   working-set ratio per cell in the results CSV so a later reader can tell
   whether a cell was HBM-bound without re-deriving it.

Recommendation: **1 + 3**.

## Consequence for existing numbers

Any MI355X figure in `AI/flydsl_rmsnorm_notes.md` whose picked working set is
`<= 256 MiB` *and* which does not trigger the evictor is measured partly against
MALL and is optimistic — **37 of 90 cells** across the full matrix, or 8 of 18
per 16-bit mode and 5 of 18 for fp32/same.
The `M=32768` row (100%/88%) is clean except for one undetermined cell:
`32768x1024` forward sits at 256.004 MiB, marginally *over* MALL capacity, and
the probe has no sample between 256 and 384 MiB to place it. It should be
measured rather than classified. Whether that shifts the
published median depends on how many cells feed it, and should be recomputed
rather than assumed. The `M=4096` row
(71%/64%) and `M<=512` row (7%/5%) are affected, though at `M<=512` the cells
are launch-bound and bandwidth is not the binding constraint anyway, so the
practical distortion is concentrated in the **`M=4096` row**.

This also means percent-of-roofline is not comparable between the MI355X column
and the two Hopper columns of that table, on top of the already-documented
"different roofline probes, do not compare across hosts" caveat.

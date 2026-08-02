# The benchmark harness does not produce cold reads on gfx950

Status: **confirmed by measurement**; the harness-side fix is **written and
measured** (`31c1fd4` sized the rotation target and the gate against the MALL,
`fad422c` and `4bd05e7` made the LLC lookup fail closed -- `fad422c` alone did
not, it only caught whole-topology failure and let a corrupt MALL cache entry
resolve to 4 MiB while still reporting `source: kfd_topology`; before/after cost is committed under
`AI/gate_llc_before_after/`). What is *not* done is re-collection: no MI355X
cell has been re-measured under the fixed harness. Blocks Experiment No.002
(MI355X flydsl-vs-torch matrix) — any MI355X numbers taken before the fix are
**unsound on 41 of the 90 benchmarked cells**.

**41 and 37 count different things and must not be merged.**

- **37** is the *strict-resident* subset: working set `<= 256 MiB` **and** the
  evictor does not fire (47 fit the MALL, 10 of those do run the evictor).
  8 of 18 per 16-bit mode, 5 of 18 for `float32/same`. **No `m=32768` cell is
  in the 37** — an earlier version of this line said one was, conflating the
  two sets.
- **41** is the *contract-invalid* headline: the 37 plus the four
  `32768x1024` fwd 16-bit cells at 256.004 MiB. Those sit a few KiB *past*
  capacity, so a `ws <= MALL` rule excludes them, but they were probed directly
  at their exact working sets and read MALL-warm anyway. Per 16-bit mode that
  is 8 strict-resident + 1 exact-probed = 9, so `4 x 9 + 5 = 41`.

An earlier version of this note derived the 9-per-mode figure correctly and
then still wrote the full total as 37; @Reviewer caught the arithmetic not
matching its own premise. Use 37 when the claim is "strictly resident and
un-evicted", 41 when it is "the cold-measurement contract was not established".

Both are exposure counts, not per-cell results: without an RMSNorm before/after
nobody knows how far any individual cell moves. (An earlier "11 of 18" here was computed from approximate
bytes and ignored the `use_evictor` gate; withdrawn — see below.) A second code
path, `_pick_l2_rotate_count` in `quack/bench/bench_utils.py`, shares the root
cause but **has no live consumer on this box**: the FlyDSL autotune path uses
`flydsl.autotune` with `batched_event_bench`, which neither rotates nor evicts,
and the only in-repo consumers of `_pick_l2_rotate_count` are the CuTe and GEMM
paths, which cannot even import here (`cuda.bindings.driver`). @Reviewer
established this. The gate documented below is the *harness's own*
`use_evictor` (`benchmarks/benchmark_rmsnorm_flydsl.py:960`), a separate
implementation that is live and produced the 90-cell matrix.

## Summary

`benchmarks/benchmark_rmsnorm_flydsl.py` sizes both its rotation buffers and its
L2 evictor from `torch.cuda.get_device_properties().L2_cache_size`. On gfx950
that property reports **4 MiB**, which is the *per-XCD* L2. It does not report
the device-wide **256 MiB MALL / Infinity Cache** that sits behind it. Two
independent consequences follow. They were originally written up as compounding;
the controlled measurement (below) attributes the observed ~1.3x to **defect 1**
— the evictor never runs — and does *not* show defect 2's undersizing costing
anything measurable on this access pattern. Defect 2 is still a real
mis-derivation and should be fixed, but it should not be credited with the
inflation.

### Scope of what has been measured

Everything quantitative in this note comes from a **`torch.Tensor.copy_`
bandwidth probe**. No RMSNorm kernel has been run before and after a fix, and
no MI355X 90-cell matrix has been re-collected. The consequences:

- The **~1.3x is a property of the copy probe at one working set**. It is not a
  per-cell correction factor and must not be applied to the 41 exposed cells,
  to the six regime values in `flydsl_rmsnorm_notes.md`, or to any roofline
  percentage. Doing that would report an inferred number as a measured one.
- **41 of 90** is an exposure count — 37 where the gate provably does not fire
  and the working set provably fits the MALL, plus 4 probed directly a few KiB
  past capacity. It says nothing about how far any of those cells is off.
- What the probe does establish is that the *measurement method* is unsound on
  MALL-resident shapes, which is sufficient to justify fixing the gate and
  re-collecting. It is not sufficient to restate existing numbers.

Closing this gap needs an RMSNorm before/after on the exposed cells, which is
Experiment No.002 and has not been run. The gate fix's own cost *is* measured —
`AI/gate_llc_before_after/` holds twelve committed runs — but that measures what
the harness change costs in reported GB/s, which is not the same as knowing the
true HBM number for any cell.

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
all of the gap — 4880.56 GB/s against a 4992.27 GB/s HBM reference, 2.2%
below it, and in fact 2.0% *above* the 4786.65 GB/s that a 256 MiB evictor
reaches. (Percentages are computed from the raw medians in the current sidecar
— stored in `c0b7c0b`, generated by the script at `21e91f6` — not from rounded
GB/s; rounding first shifted these by over a point. An earlier version of this
paragraph quoted 4898.37 / 4935.34 / 4827.98 from a *superseded* sidecar while
the tables below had already moved on, which gave the file two current answers.
The exact values move run to run — see the spread note below — so read them to
the nearest point, not the nearest hundredth; the 0.75%-below figure the old
sidecar gave and the 2.2%-below figure this one gives are the same
qualitative result.) A
copy-based evictor evidently disturbs MALL
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

`contract-invalid` means the cold-measurement contract did not hold for that
cell, so its number needs re-collection. It deliberately does **not** say
"inflated": that word asserts a direction, and nothing here measured an RMSNorm
cell. The column said `inflated` until @Reviewer pointed out that the tail of
this file had been corrected to "invalid, needs re-collection" while the table
still asserted a sign.

Two distinct reasons land a cell in that set, and they must not be merged into
one rule:

* in this mode **10 cells** are strictly `ws <= 256 MiB`; **2 of them** (the two
  `M=1` rows) are small enough that the `ws < l2_target` gate still fires, so
  **8** are both at-or-under capacity *and* un-evicted. An earlier version of
  this line said "ten cells … and un-evicted", which welded a strict-capacity
  count onto an un-evicted count and contradicted the table directly above it;
* **`32768x1024` fwd** is `256.004 MiB` (16-bit weight) or `256.007812 MiB`
  (fp32 weight) — *past* capacity, so the `<=` rule excludes it. It qualifies
  only because the archived fine-boundary copy probe measured those exact
  working sets on the MALL-warm side. "`<= 256 MiB` leaves exactly 11" is not a
  valid derivation and is no longer used.

So this mode contributes **8 + 1 = 9** contract-invalid cells, and the full
matrix total is the independently computed **47 at-or-under capacity − 10
evicted = 37**, not a per-mode count multiplied up.

| shape | op | B/call | picked | working set | vs MALL | regime |
| --- | --- | --- | --- | --- | --- | --- |
| `1x4096` | fwd | 0.023 MiB | 4 | 0.094 MiB | 0.00x | evicted |
| `1x4096` | bwd | 0.039 MiB | 4 | 0.156 MiB | 0.00x | evicted |
| `256x4096` | fwd | 4.008 MiB | 3 | 12.023 MiB | 0.05x | contract-invalid |
| `256x4096` | bwd | 6.017 MiB | 2 | 12.033 MiB | 0.05x | contract-invalid |
| `512x4096` | fwd | 8.008 MiB | 2 | 16.016 MiB | 0.06x | contract-invalid |
| `512x4096` | bwd | 12.018 MiB | 2 | 24.035 MiB | 0.09x | contract-invalid |
| `4096x3000` | fwd | 46.881 MiB | 2 | 93.761 MiB | 0.37x | contract-invalid |
| `4096x3000` | bwd | 70.340 MiB | 2 | 140.679 MiB | 0.55x | contract-invalid |
| `4096x4096` | fwd | 64.008 MiB | 2 | 128.016 MiB | 0.50x | contract-invalid |
| `4096x4096` | bwd | 96.031 MiB | 2 | 192.062 MiB | 0.75x | contract-invalid |
| `32768x1024` | fwd | 128.002 MiB | 2 | **256.004 MiB** | 1.00x | contract-invalid (past MALL; see below) |
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
- plus **4** `32768x1024` fwd 16-bit cells at 256.004 MiB, a few KiB *past*
  capacity so the `<=` test excludes them, but probed directly and reading
  MALL-warm — **41 of 90** contract-invalid in total

Per mode that is **8 of 18** strict-resident for each of the four 16-bit modes
and **5 of 18** for `float32/same`; adding the exact-probed cell gives 9 of 18
contract-invalid per 16-bit mode, i.e. `4 x 9 + 5 = 41`. These are recomputed from the harness's real `logical_bytes`
and actual buffer selection. A separate "8 of 18" figure quoted earlier in this
thread was @Autotune's, derived from a synthetic single-tensor model of
`_pick_l2_rotate_count`; the agreement is coincidental and the two should not be
cited as corroborating each other. The previously published "11 of 18" was produced by the
approximate byte count and by ignoring `use_evictor`; it is withdrawn.

**`32768x1024` forward: measured, not inferred.** @Reviewer was right that the
cell is at **256.004 MiB**, not exactly 256.0 — the extra 4 KiB is the weight
row, which my approximate byte count had dropped — and right that a set
marginally *over* capacity cannot be classified by a `<= 256 MiB` rule when the
probe only sampled 256 and 384 MiB. So I measured it directly, at the exact
harness working set of 268439552 bytes.

> **Withdrawn table, superseded 2026-08-02.** This paragraph originally carried
> a nine-row table reading `268439552 -> 6325 GB/s` and `6295` on repeat, and
> concluded "the cell reads 6295–6325 GB/s". **Those numbers were produced by
> the buggy fine-sweep arithmetic** (`elem_bytes = ws/2`, which doubled every
> point, plus page-rounding that collapsed the 2 KiB steps) — the sweep labelled
> 256.00–257 MiB was really walking 512–514 MiB. They are not measurements of
> this cell and must not be cited. The corrected sweep is in
> "Fine boundary sweep" below; the authoritative value for this working set is
> **6391.418 GB/s** at 268439552 B, from the current sidecar (`c0b7c0b`,
> generated by the script at `21e91f6`). Leaving the old table in place
> alongside the new one gave the file two incompatible "current" answers, which
> is @Reviewer's blocker 1. **This line then reproduced that same failure**: it
> was left reading **6067.802 GB/s**, a value from the *previous* sidecar, so
> the sentence that exists to name one authoritative number named a stale one.
> The two differ by 5.3% — more than the run-to-run spread of the surrounding
> rows — and 6391.418 is the higher, i.e. the correction moves this cell
> *further* from the HBM reference, not closer. Do not read the change as
> tightening the case.

The verdict for the cell is unchanged — a copy probe at its exact working set
shows the cold-measurement contract does not hold there, and being 4 KiB over
capacity does not by itself evict it — but it rests on the corrected sweep, not
on the withdrawn table above or on a threshold rule. No RMSNorm measurement of
this cell exists, so its direction and magnitude remain unknown.

Worth recording that the boundary is **not a cliff**: throughput decays across
roughly 256 → 288 MiB rather than stepping at one point. My earlier
"exactly 256.0 MiB, precisely the boundary" framing implied a sharpness the data
does not show, and the adjacent-rotation step figure (~1.3x) is a chord across
that decay, not the height of a discontinuity.

Two earlier claims are withdrawn outright: "only `m<=4096` is affected" /
"`m=32768` is unaffected", and the "11 of 18" count.


That earlier version also carried an off-by-one in a "buffers needed" table,
and my first correction of it stated the condition backwards. The requirement
is the smallest `n` with `n*B > C` for per-buffer bytes `B` and MALL capacity
`C`, i.e. `floor(C/B)+1`. Writing `ceil(C/B)+1` overshoots by one whenever
`C/B` is **not** an integer — the common case — and is correct only when the
division happens to be exact. I had it the other way round.

Recomputed with the real per-buffer bytes. All three rows below are the
**forward-only** set — `input + output + weight`, no `rstd`, because
`benchmarks/benchmark_rmsnorm_flydsl.py:411` passes `store_rstd=False`. The
autograd path (`store_rstd=need_grad`) adds a 4-byte `rstd`, which only changes
the count at `m=1`; see the note under the table.
`C = 256 MiB`:

| cell | B | C/B | buffers to exceed |
| --- | --- | --- | --- |
| `4096x4096` fwd bf16/same | 67117056 | 3.9995 | **4** (not 5) |
| `256x4096` fwd | 4202496 | 63.875 | **64** (not 65) |
| `1x4096` fwd | 24576 | 10922.67 | **10923** (not >16000) |

`1x4096` is tensor-set dependent and both values below are correct, for
different call sites. `rmsnorm_fwd` takes `store_rstd`, and
`benchmarks/benchmark_rmsnorm_flydsl.py:411` passes `store_rstd=False`, so the
forward-only benchmark set is `x + out + weight` = 24576 B → **10923**. The
autograd path sets `store_rstd=need_grad` (`quack/rmsnorm.py:1632`), which adds
a 4-byte `rstd` → 24580 B → **10921**. At `m=1` the buffer is only 24 KB, so
4 bytes moves the count. @Autotune computed 10921 against the grad set; the
harness cell this table describes is the fwd-only one.

Rotation count still is not a usable lever at small `m` — 64 buffers at
`256x4096` and 10923 at `1x4096` — so **the evictor remains the only
practical fix**. But the reason is the size of the requirement, not an
`m=32768` cutoff.

## Measurement

`AI/probe_gfx950_mall_rotation.py`, MI355X (gfx950), one GPU, SPX/NPS1,
`HIP_VISIBLE_DEVICES=0`, torch 2.9.1+rocm7.2.0.

**Which physical card that is:** `HIP_VISIBLE_DEVICES=0` selects PCI bus
`0x75` (`0000:75:00.0`), which `rocm-smi` lists as **GPU 3**, not GPU 0 —
torch's enumeration order does not match `rocm-smi`'s on this host. So neither
the earlier `HIP_VISIBLE_DEVICES=7` in this line nor my later correction to
"GPU 0" named the right card; the mask value is an index into the visible set
and never denoted a physical GPU. The sidecar records `device_pci_bus_id` so
the card is identifiable regardless of anyone's numbering.

> **Correction, 2026-08-02.** An earlier version of this paragraph said torch's
> `uuid` "does not correspond to `rocm-smi --showuniqueid`, so PCI BDF is the
> only cross-checkable identifier here." The first half is true and the
> conclusion drawn from it was wrong. Torch's `uuid` is not a UUID at all: its
> 16 bytes are the **ASCII text** of a hex string, so
> `61363063-3239-3536-6364-396464346335` decodes to `a60c2956cd9dd4c5`, which
> is exactly KFD's `unique_id` (11964983762810164421) for that node. Verified
> on all 8 GPUs: torch `uuid`, decoded that way, equals KFD `unique_id` in
> 8/8 cases. `rocm-smi --showuniqueid` prints a *different* per-GPU 64-bit
> value that matches neither, which is what misled me — I compared against
> rocm-smi, found no match, and concluded the field was useless instead of
> checking what it actually encodes. So there are **two** cross-checkable
> identifiers, and `unique_id` is the stronger one: it survives PCI
> renumbering and, unlike a BDF, stays distinct when several KFD nodes share
> one PCI address.

Kernel held *exactly* fixed
(`copy_` between rotating buffer pairs); only the number of rotation buffers
varies, so any systematic difference is cache residency and not kernel
selection. Every figure below is emitted by that script, which also writes
`AI/probe_gfx950_mall_rotation.json` with all 75 round latencies per point and the
environment. The 256 MiB MALL size the script divides by is a **hardcoded
constant, corroborated by `rocminfo`'s `L3: 262144 KB`, not discovered** — the
sidecar says so itself (`mall_bytes_is_hardcoded: true`,
`rocminfo_l3_agrees_with_assumed: true`). An earlier version of this line said
"discoverable rather than assumed", which contradicted the artifact it cited.
Timing follows `_time_rotating_calls`: the
window covers one whole rotation and the evictor runs *outside* it.

All eleven sampled points from the current sidecar, not a subset:

    ### buffer = 64 MiB
     buffers    working set   vs MALL      GB/s
           1         128 MiB     0.50x      5992
           2         256 MiB     1.00x      6385
           3         384 MiB     1.50x      4882     <- step here
           4         512 MiB     2.00x      4882
           6         768 MiB     3.00x      4958
           8        1024 MiB     4.00x      4966
           9        1152 MiB     4.50x      4992
          12        1536 MiB     6.00x      5001
          16        2048 MiB     8.00x      5029
          24        3072 MiB    12.00x      4999
          32        4096 MiB    16.00x      5021

The honest figure is the **boundary step between adjacent rotation counts**:
2 buffers (256 MiB) = 6385.24 GB/s vs 3 buffers (384 MiB) = 4881.77 GB/s, one
rotation apart, same kernel — **1.308x inflation** on this run. An earlier
version of this table printed 6459 / 4787 / **1.349x** from a superseded
sidecar and dropped six of the eleven rows; 1.349 is a real measurement (it is
`c9c10fc` in the run table below), but it is not what the committed sidecar
says today. The run-to-run range is in that table.

The 16 MiB sweep crosses the boundary at 8 → 9 buffers, but the script never
computed that step: `bench_rotation` looks for `lo + 1 = 9` and `ROTATIONS`
jumped 8 → 12, so the 16 MiB sweep contributed nothing to the printed verdict
while this note quoted a hand-computed 1.32x from the 8 → 12 pair — four
rotations apart, which does not meet the adjacency standard the 64 MiB figure
is held to. `ROTATIONS` now samples 9, and the script prints an explicit
`NOT COMPUTED` line rather than silently omitting the step.

With 9 sampled, the 16 MiB sweep does produce an adjacent step:

     buffers    working set   vs MALL      GB/s
           1          32 MiB     0.12x      4173
           2          64 MiB     0.25x      4609
           3          96 MiB     0.38x      4954
           4         128 MiB     0.50x      5069
           6         192 MiB     0.75x      5226
           8         256 MiB     1.00x      5280
           9         288 MiB     1.12x      4096     <- step here
          12         384 MiB     1.50x      4035
          16         512 MiB     2.00x      4087
          24         768 MiB     3.00x      4084
          32        1024 MiB     4.00x      4108

5279.90 / 4096.39 = **1.289x**, the low end of the run range. (The previous
version of this block quoted 5322 / 4022 / 1.32x from the superseded sidecar.)
Note the 16 MiB sweep climbs steadily from 1 to 8 buffers before the step —
below ~256 MiB the copy is *gaining* from residency, which is the same effect
seen from the other side.

A second sampling defect turned up on the way here, and it is worth stating
because it also came from my own code. `rounds` was `iters // n_buffers`, so a
2-buffer row got 15 rounds per repeat while a 32-buffer row got 1 — the two
sides of every boundary step were sampled unequally, and the *high* buffer
count side, which is where the post-boundary rows live, always got less data.
`ROUNDS` is now a constant 15 (75 rounds per point) regardless of buffer count.

Across runs the same step measures **1.266x, 1.289x, 1.298x, 1.308x, 1.318x,
1.324x, 1.328x, 1.334x (x2), 1.341x, 1.344x, 1.345x, 1.349x, 1.354x**, so the
committed-run range is **1.266-1.354x**. Each is recomputable from a JSON
sidecar committed in this repo's history. Sidecar-holding commit → the script
commit that generated it (`environment.git_commit`), which are *different
commits* and which an earlier version of this list conflated:

| sidecar stored in | generated by script at | 16 MiB 8/9 | 64 MiB 2/3 |
| --- | --- | --- | --- |
| `7780def` | `50c54b6` | (not sampled) | 1.317957 |
| `0198dee`, `46d9095` | `7780def` | 1.297679 | 1.323657 |
| `85ade0c` | `46d9095` | 1.327690 | 1.353909 |
| `250502a` | `2bd5624` | 1.341131 | **1.345494** |
| `d853d7b`, `1b53896` | `250502a` | 1.343588 | 1.333958 |
| `c9c10fc` | `1b53896` | 1.323130 | 1.349374 |
| `f4e36e9` | `742196f` | **1.266049** | 1.334622 |
| `c0b7c0b` | `21e91f6` | 1.288916 | 1.307976 |

> **Retraction of a retraction, 2026-08-02.** I previously "withdrew" `1.345x`
> as appearing in no committed artifact. **That withdrawal was wrong** —
> @Reviewer checked and `250502a`'s sidecar contains `6434.215277 / 4782.047493
> = 1.345494x`. I had listed only four commits, treated each as both storage
> and source, and did not recompute from all of them; the value I declared
> phantom was in a sidecar I never opened. The known `elem_bytes` bug in that
> commit affects its **`fine_boundary`** block only — the `rotation_sweep`
> block it comes from is unaffected. Withdrawing a real measurement on a bad
> audit is worse than the original error, because it destroys evidence while
> looking like diligence.
>
> **And the same omission recurred, 2026-08-02.** @Reviewer then found that
> this very table was still missing `f4e36e9` (source `742196f`), whose sidecar
> gives 16 MiB **1.266049x** and 64 MiB 1.334622x — recomputed here and
> confirmed. 1.266 is the *lowest* value on record, so leaving it out narrowed
> the published range to 1.289-1.354x in the direction that flattered the
> claim. That the fix for an omission left another omission of the same kind in
> the same table is the point: I was correcting entries rather than
> regenerating the table from the set of sidecars in history.

Quote
this as **~1.3x**; the third significant figure is not reproducible and my
earlier "1.32x at two buffer sizes, independently" claimed a precision and an
agreement the data never supported. The qualitative result — a step of roughly
30% at the 256 MiB crossing, in both sweeps and on every run — is solid.

**Corroborating diagnostic from @Autotune** — not an independent confirmation
until a script, raw samples and environment are committed alongside it, which
they are not yet. The design is a cleaner control than mine:
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

    64 MiB x2 bufs (WS=256 MiB), evictor=0 MiB      6415.76 GB/s  <- pre-fix behaviour
    64 MiB x2 bufs (WS=256 MiB), evictor=12 MiB     4880.56 GB/s  <- old evictor size
    64 MiB x2 bufs (WS=256 MiB), evictor=256 MiB    4786.65 GB/s
    64 MiB x2 bufs (WS=256 MiB), evictor=512 MiB    4756.04 GB/s
    64 MiB x2 bufs (WS=256 MiB), evictor=1024 MiB   4683.10 GB/s
    64 MiB x8 bufs (WS=1024 MiB), no evictor        4992.27 GB/s  <- HBM reference

Evicting at all is what matters here: with no evictor the boundary cell reads
6415.76 GB/s against a 4992.27 GB/s HBM reference (1.285x), and *any* of the
evictor sizes brings it to 4683.10–4880.56 GB/s. The worst of those, the
1 GiB evictor, is 6.19% below the reference; the best, the 12 MiB one, is
2.24% below it. Which
of the large evictors comes last is not stable across runs (the 1 GiB one is
worst here and was mid-pack on the previous sidecar), so read only the
grouping, not the order. These six rows are the current sidecar; the previous
version of this block quoted the superseded one (6458.99 / 4898.37 / 4827.98 /
4637.79 / 4709.39 / 4935.34), where the 512 MiB row rather than the 1 GiB row
came last — which is exactly why the ordering is not a result. The
12 MiB evictor is not obviously worse than the 256–1024 MiB ones on this
access pattern, so the measured defect is **defect 1** — the `ws < 12 MiB`
gate means no evictor runs on these shapes at all. Defect 2 (the evictor being
sized off the per-XCD L2) is a real mis-derivation but is not, on this
evidence, what costs the ~1.3x. An earlier version of this block reported
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
obstacle *for crossing the MALL*: the working set at that crossing point is
256 MiB by construction, so the awkward part is the buffer *count*, not the
bytes. Note this is not the same as the autotuner's own target — at
`target_ratio=3` against a 256 MiB LLC the target is 768 MiB, so a clone sized
to reach it is ~0.75 GiB, not 0.26 GiB. @Reviewer flagged the conflation.

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
   recovers the HBM number — the 12 MiB evictor lands 1.46% above the 256 MiB
   one and 0.75% below the HBM reference.
   So the gate should be driven by whether the working set fits the effective
   LLC, not by whether it is smaller than a multiple of the per-XCD L2.

   Re-sizing off an effective LLC (`L3` if present, else `L2`) is still correct
   and should ship with it — the current derivation is wrong on its own terms,
   and it degenerates to today's behaviour on NVIDIA where no L3 is reported —
   but it should be presented as correcting a mis-derivation, not as the thing
   that buys back the ~1.3x. On this access pattern it does not.

2. **Force the rotation working set past the MALL.** Still rejected for the
   *harness*, where the evictor solves the problem directly — but my original
   reason was wrong and should not be reused. I wrote that raising
   `max_rotation_buffers` to 64 for `256x4096` "would allocate absurd amounts of
   memory". It would not: the working set needed to *cross the MALL* is 256 MiB
   by construction, so that allocation is ~0.26 GiB regardless of buffer count.
   (Reaching the autotuner's own `3 * LLC` target is a different and larger
   number, ~0.75 GiB.) What
   is awkward at small `m` is the *count* (64 buffers at `256x4096`, 10923 at
   `1x4096`), not the bytes. For the autotuner path, which has no evictor, this
   is the only available lever and the memory cost is not an objection to it.

3. **Treat memory-side cache as a separate regime and report it.** Not a fix,
   but worth doing alongside 1: record effective-LLC bytes and the resulting
   working-set ratio per cell in the results CSV so a later reader can tell
   whether a cell was HBM-bound without re-deriving it.

Recommendation: **1 + 3**.

## Consequence for existing numbers

Any MI355X figure in `AI/flydsl_rmsnorm_notes.md` whose picked working set is
`<= 256 MiB` *and* which does not trigger the evictor was taken **without the
cold-measurement contract holding** — **37 of 90 cells** across the full matrix,
or 8 of 18 per 16-bit mode and 5 of 18 for fp32/same. Adding the four
`32768x1024` fwd 16-bit cells, which were probed directly at their exact
working sets rather than classified by the threshold, gives the **41 of 90**
headline in the status line above.

Those 41 cells are **invalid and need re-collection; they are not "optimistic by
a known amount"**. Everything measured here is a `copy_` probe. A copy kernel
losing MALL residency says the *contract* was not established; it does not
transfer a direction or a magnitude to an RMSNorm cell, whose arithmetic
intensity, access pattern and launch configuration all differ. An earlier
version of this line called the cells "optimistic", which asserts a signed error
this evidence cannot support.
The `M=32768` row (100%/88%) contains no cell in the 37 — but it contributes
four of the 41, and it is not clean.
The four 16-bit `32768x1024` forward cells sit at 256.003906 MiB (16-bit
weight) and 256.007812 MiB (fp32 weight) — a few KiB *past* MALL capacity, so
the `ws <= MALL` test excludes them, yet a copy probe at those exact working
sets still shows the contract failing. The
fine-boundary block measures those exact working sets:

    268435456   256.000000 MiB   6397 GB/s
    268437504   256.001953 MiB   6295
    268439552   256.003906 MiB   6391   <- 32768x1024 fwd, 16-bit weight
    268443648   256.007812 MiB   6367   <- 32768x1024 fwd, fp32 weight
    268500992   256.062500 MiB   5584
    268697600   256.250000 MiB   5898
    269484032   257.000000 MiB   5031
    301989888   288.000000 MiB   4871
    402653184   384.000000 MiB   4991

Against this run's 4992 GB/s HBM reference the **copy probe** at both working
sets reads high (6391 and 6367 GB/s, i.e. 1.28x and 1.28x). That is a statement
about copy traffic at those sizes, and it establishes only that a set a few KiB
past capacity is not self-evicting. It is **not** a measurement of the RMSNorm
cells, whose direction and magnitude are unknown until they are re-collected.

The ordering among the four points from 256.000 to 256.008 MiB is **not
established**, and two successive attempts to say so were themselves wrong.
The first claimed the interquartile ranges "overlap almost completely" while
quoting 6019–6191 against 6254–6453 — intervals that are *disjoint*, as
@Reviewer caught. Per-run IQRs, recomputed from all 75 raw rounds:

| WS | run `1b53896` | run `742196f` | run `21e91f6` |
| --- | --- | --- | --- |
| 256.000000 MiB | 6446.6 / 6421.7–6465.2 | 6391.3 / 6379.0–6409.5 | 6415.6 / 6379.2–6428.1 |
| 256.001953 MiB | 6331.1 / 6266.0–6391.4 | 6236.9 / 6179.4–6354.9 | 6248.5 / 6202.3–6331.1 |
| 256.003906 MiB | 6078.7 / 6018.8–6190.9 | 6128.8 / 6045.9–6173.9 | 6391.3 / 6373.2–6422.0 |
| 256.007812 MiB | 6379.4 / 6254.4–6453.0 | 6156.8 / 6089.9–6214.0 | 6367.3 / 6289.7–6391.5 |

Within any single run the last two IQRs are disjoint, so no run can call the
difference noise on its own. **Across** runs the comparison inverts outright:
256.003906 reads *below* 256.007812 by 300 and by 28 GB/s in the first two runs
and *above* it in the third, having moved 262 GB/s — many times its own ~50 GB/s
IQR width — between runs taken minutes apart on an otherwise idle card. Whatever
orders these four points is not the working set. That is a claim about
reproducibility, not a statistical test, and it is the strongest one available
here: back-to-back blocks cannot separate a working-set effect from drift.
Interleaved repeats would be needed. The only claim these four points support is
that copy traffic at all four of those working sets reads above the HBM
reference, i.e. the cold-measurement contract is not established there. The decay from 256 to 288 MiB is gradual, not a cliff, which is why a
threshold test misclassifies cells sitting a few KiB either side of it — and
why these were measured rather than classified. Whether this shifts the
published median depends on how many cells feed it, and should be recomputed
rather than assumed. The `M=4096` row
(71%/64%) and `M<=512` row (7%/5%) are affected, though at `M<=512` the cells
are launch-bound and bandwidth is not the binding constraint anyway, so the
practical distortion is concentrated in the **`M=4096` row**.

This also means percent-of-roofline is not comparable between the MI355X column
and the two Hopper columns of that table, on top of the already-documented
"different roofline probes, do not compare across hosts" caveat.

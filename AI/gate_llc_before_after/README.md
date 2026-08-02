# Raw before/after for the MALL-sized rotation target and evictor gate

Committed because the cost figures for `31c1fd4` previously existed only in that
commit's message and in chat. They are re-collected here from scratch, with the
raw `results.csv` and `environment.json` of every run, so the numbers can be
recomputed rather than taken on trust. The figures below supersede the three
lines quoted in `31c1fd4`'s body: those came from single runs of each side, and
one of them (`4096x4096`, `-10.0%`) is a 1-vs-1 comparison of a cell whose
before-side varies by 4%.

## What was run

Host `mia1-p02-g23`, `HIP_VISIBLE_DEVICES=0` -> `0000:75:00.0` -> rocm-smi
GPU[3], MI355X / gfx950. `torch 2.9.1+rocm7.2.0`, HIP `7.2.26015`.

Provider `torch` only, `fwd` only, bf16 activations. Using the reference
provider rather than FlyDSL is deliberate: the change under test is in the
harness, not in a kernel, and `torch.nn.functional.rms_norm` holds still while
the harness moves. `--weight-modes same` for the isolate runs; the four
before/after pairs also carry the `float32` weight rows.

* `before_{A,B,C,D}` -- harness at `94d3f8c`, the commit before the gate change,
  extracted with `git show` and run unmodified.
* `after_{A,B,C,D}` -- harness at `d167701`, which was head when these ran.
* `isolate_gateonly` -- current harness with the target line reverted to
  `properties.L2_cache_size * ratio`. New gate, old 12 MiB target.
* `after_v3_schema` -- head after `fad422c`, two shapes, to show the schema
  change did not move the numbers.
* `isolate_targetonly` -- current harness with the gate reverted to
  `rotation_working_set_bytes < l2_target_bytes`. New 768 MiB target, old gate.

### Executed source

Each run's `environment.json` records its own `command`. Those commands name
four scratch copies which were **deleted after the runs and not committed** --
so for a week this artifact asked you to take on trust that the deleted files
were what the prose said they were. @Reviewer refused, correctly. The exact
bytes are now vendored in `sources/`, with `sources/SHA256SUMS`:

| file | sha256 | used by |
| --- | --- | --- |
| `_bench_pre_gate_tmp.py` | `44be2278...` | `before_{A,B,C,D}` |
| `_v_gateonly.py` | `473cd47e...` | `isolate_gateonly` |
| `_v_targetonly.py` | `380c82f7...` | `isolate_targetonly` |
| `_v_nomargin.py` | `6ce25f00...` | `isolate_nomargin` |

Each is regenerable, and the recipe is the audit:

```
git show 94d3f8c:benchmarks/benchmark_rmsnorm_flydsl.py           # _bench_pre_gate_tmp.py
git show d167701:benchmarks/benchmark_rmsnorm_flydsl.py           # base for the three isolates
```

then in that base substitute exactly one line (each anchor occurs exactly once,
which is what makes the substitution unambiguous):

* `_v_gateonly.py` — `l2_target_bytes = llc_bytes * args.l2_target_ratio`
  → `l2_target_bytes = properties.L2_cache_size * args.l2_target_ratio`
* `_v_targetonly.py` — `use_evictor = rotation_working_set_bytes <= 2 * llc_bytes`
  → `use_evictor = rotation_working_set_bytes < l2_target_bytes`
* `_v_nomargin.py` — same anchor
  → `use_evictor = rotation_working_set_bytes <= llc_bytes`

**What this does and does not establish.** The vendored files reproduce
byte-exactly from immutable commits, and their hashes match the ones @Reviewer
derived independently. That authenticates the *intended* source. It does **not**
authenticate the *executed* source: the generator did not record a hash of the
file it ran, so nothing in these twelve artifacts rules out an additional
uncommitted edit in the deleted originals. **That gap is not closable
retroactively** — treat every isolate here as reproducible-in-intent, not as
certified, and re-run them if the attribution ever needs to be load-bearing.

It is closed going forward. The harness now records `script_sha256` (a hash of
the file actually executing, so a scratch copy is distinguishable from the
committed harness), `git_dirty` and `git_dirty_paths` in every
`environment.json`. Recording only `git_commit` was the root cause: HEAD was
`d167701` for the before run, both isolates and all four after runs, so the
artifact's own provenance field was constant across the arms of the experiment
it was supposed to identify. Any future run of this comparison will be
checkable in a way these are not.

The `before` runs are the one exception: `94d3f8c` is a commit, `git show` is
byte-exact, and `before_{A,B,C,D}`'s numbers are reproducible by anyone.

These snapshots predate `fad422c`, which renamed `l2_target_bytes` to
`rotation_target_bytes`, added `evictor_threshold_bytes`, and bumped the schema
to 3. The isolate recipes are against `d167701`, not current head; running them
against head will not apply, because the anchor lines have been renamed.

`before_{A,B}` and `after_{A,B}` carry the `float32` weight rows as well as
`same`; `before_{C,D}`, `after_{C,D}`, `after_v3_schema` and the three isolate runs are
`same` only.
Every table here is `same`, so the extra rows are context, not inputs.

Exact counts, so no one has to infer them: **12 run directories, 59 CSV rows.**

| runs | rows each | total |
| --- | --- | --- |
| `before_A`, `before_B`, `after_A`, `after_B` | 8 (4 shapes × `same`+`float32`) | 32 |
| `before_C`, `before_D`, `after_C`, `after_D` | 4 (4 shapes, `same`) | 16 |
| `isolate_gateonly`, `isolate_targetonly` | 4 | 8 |
| `after_v3_schema` | 2 | 2 |
| `isolate_nomargin` | 1 | 1 |

The contention canary is `quiet: true` in all twelve runs, with
`closing_over_opening` in 0.975-1.020, so nothing here is a neighbour on the
node. `last_level_cache_bytes` is `268435456` with `torch_l2_cache_size`
`4194304` in the after runs; both keys are absent from the before runs because
that commit did not record them -- which is itself the defect.

These artifacts are **eleven schema v2 and one schema v3** -- `after_v3_schema/`
is the v3 one, which is what it was collected to demonstrate. Calling the whole
set "v2" was wrong, and the directory name refutes it on its own; @Reviewer's
blocker 2. Current head emits **v4**. A fresh run will not match any of them
field for field: `fad422c` renamed `l2_target_bytes` (which had been carrying
`3 x LLC` under an L2 name) to `rotation_target_bytes`, added
`evictor_threshold_bytes` per row, and added `last_level_cache_provenance`,
`rotation_target_bytes` and `evictor_gate` to the environment; v4 then renamed
`l2_eviction_between_calls` to `evictor_ran_per_rotation`.

**All twelve** `methodology.steady_state` strings here -- the eleven v2 and the
one v3 alike -- assert timing with "per-call events", which the code has never
done in these runs. The `methodology.cache` string likewise still describes the
old "below the L2 target / between individually timed calls" rule, wrong on both
counts. These are frozen artifacts and are not being edited; the strings are
wrong *in the files*, which is why they are named here. Any consumer reading
their methodology should read this paragraph instead.
**None of this changed the timing path** -- the numbers below stand, and
the v3 rerun in `after_v3_schema/` reproduces `4096x4096` and `32768x1024`
to 0.12% and 0.14% of the four-run v2 medians.

## Cost of the change

Median of four runs per side. Working set and buffer count are from run A;
they are deterministic given the shape.

**Read the "what moved" column before attributing any delta.** The commit
changes two things at once -- the rotation target and the evictor gate -- and
they do not both reach every cell. Two of the four cells below never turn the
evictor on at all; their deltas are entirely the larger rotation. Saying "the
gate cost 13.8%" at `32768x1024` is false: that cell's evictor is OFF on both
sides.

| shape | rotation ws MiB, before -> after | buffers | evictor | what moved | before GB/s | after GB/s | delta | run-to-run range, before / after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `512x4096` | 16.016 -> 32.031 | 2 -> 4 | F -> T | rotation + gate | 842 | 1050 | +24.7% | 17.7% / 0.1% |
| `4096x4096` | 128.016 -> 256.031 | 2 -> 4 | F -> T | rotation + gate | 3338 | 3013 | -9.7% | 4.2% / 0.7% |
| `32768x1024` | 256.004 -> 512.008 | 2 -> 4 | F -> F | rotation only | 3109 | 2679 | -13.8% | 2.3% / 1.9% |
| `32768x2048` | 512.008 -> 768.012 | 2 -> 3 | F -> F | rotation only | 3148 | 3239 | +2.9% | 2.7% / 2.1% |

Over the full 90-cell matrix the same split is 53 cells whose rotation count
changes, 37 whose gate flips false->true, and 47 with the evictor on afterwards.
Those are three different counts over three different sets of cells.

The 37 here and the 37 in `gfx950_mall_evictor_defect.md` are, checked cell by
cell, **the same set** — every cell that was MALL-resident and un-evicted under
the old harness is exactly a cell whose gate now flips on. That is an empirical
fact about this 90-cell matrix, not an identity: the two predicates are
`ws_old <= 256 MiB and not ev_old` and `not ev_old and ws_new <= 512 MiB`, and
nothing forces them to agree on a different shape list. Do not carry the
equality over to a new matrix without re-checking it. Separately, the note's
headline count is **41**, which adds the four `32768x1024` fwd 16-bit cells;
those are contract-invalid by direct probe but do *not* flip the gate, so they
are not in either 37.

Two rows need further care.

`512x4096` is not a clean +24.7%. Its before-side range is 17.7% (860 / 961 /
812 / 823) and the four before runs have p90-p10 spreads of **24.8 / 92.2 /
21.5 / 28.2%** of their own medians, against **2.6 / 3.1 / 6.6 / 2.2%** after.
(An earlier version of this line said "26% and 95%, against 3%", quoting two of
the four and rounding; @Reviewer recomputed all eight. The full set is worse
for the before side, not better, but quoting a subset was the error either
way.) The after medians are 8.000 / 8.000 / 8.000 / 8.005 us -- three of them
identical to the microsecond across independent runs.

**The "event quantum" explanation for that was wrong, and so was the paragraph
that used to follow it.** It said to read the after side as "at the event-timer
floor, *not* stable at 1050 GB/s", and then two lines later said "the honest
statement is that the after side is stable at 1050". Those are the two
positions on offer and it asserted both; @Reviewer's blocker 6.

Measured rather than argued, by `AI/probe_event_timing_calibration.py` (sidecar
committed beside it): 400 `elapsed_time` reads of a single `512x4096`
`rms_norm` return **110 distinct values** with a typical spacing of 0.04 us,
not 1 us. There is no microsecond quantum here, so three identical 8.000 us
medians are not a quantization artifact. They are three medians of 40 samples
each landing on the same value, which is what a tight distribution does.

What the same probe *does* support: this cell is launch-dominated. Under
rocprofv3 the hardware kernel median at `512x4096` is 5.16 us while
one-pair-per-rotation event timing reads 11.67 us -- an over-read of
**+104..127%** at `512x4096`, against **+5..6%** at `32768x1024`. (Per-call,
for scale: +135..144% and +21..23%.) These are ranges over five independent
runs of each phase; see the withdrawal note below for why they are not single
numbers, and why the launch-bound one should be read as "order 100% and
unstable" rather than as its endpoints. So the after side is
repeatable at ~8 us, a large fraction of which is not the kernel, and the
before side (p90-p10 spreads of 24.8-92.2%) was not repeatable at all. Both of
those are observations. The +24.7% is a ratio between a stable number and an
unstable one and should still not be quoted as a speedup -- but the reason is
the before side's instability and this cell's launch overhead, not a timer
floor.

**The `+16%` / `+1%` this paragraph used to quote were withdrawn on
2026-08-02** and are the same arithmetic error @Reviewer blocked in the
`_time_rotating_calls` docstring against `43ffc5b`: they divided the
*unprofiled* event median (6.6001, 40.1601) by the *profiled* phase's hardware
median (5.68, 39.58), crossing profiler regimes -- one number from a process
with rocprofv3 attached, the other from a process without it. The sidecar
stores `over_read_vs_hardware` as the profiled pair, which is the only pairing
where both halves come from the same process. Fixing the docstring and leaving
this file was itself an instance of the thing: the wrong figure survived in the
artifact a reader is more likely to reach for, precisely because the fix was
scoped to where the blocker pointed rather than to everywhere the number went.
`tests/test_benchmark_rmsnorm_flydsl.py` now reads this file, so a stale
recurrence fails rather than waiting for someone to notice.

The direction of the correction is worth stating plainly, because it is not
the flattering one: the over-read at this shape is not a 16% garnish on a
kernel measurement, it is larger than the kernel. That does not change the
conclusion -- launch-dominated is launch-dominated, more so now -- but the
earlier figure understated it by a factor of six.

**And the replacement was still quoted more precisely than it was measured.**
The `+103%` / `+5%` that stood here until 2026-08-02 were single runs, because
the probe ran each phase once and so could not show its own reproducibility.
Repeats were added on 2026-08-02 to close @Reviewer's point that the sidecar
was checkable but unauthenticated; they promptly showed the launch-bound cell
moving by tens of points run to run. Two repeats were not enough to see it --
the first two-run check put that spread at 3.7pp, and only five exposed the
tail -- which is a useful thing to know about how many repeats "reproducible"
needs. The figures above are now ranges over five runs.

**And five runs bound this artifact, not the machine.** Successive
regenerations gave `512x4096` spreads of 25.0pp, 7.7pp, and 21.5pp: the spread
*itself* is unstable, so the endpoints above are exact for the committed
sidecar and are not a property anyone should expect to reproduce. Read that row
as "order 100% and unstable". The `32768x1024` figures are tight across every
set, and the ordering the design rests on -- per-rotation below per-call, both
shrinking as the kernel grows -- holds in every individual repeat.

Note the pattern this file keeps re-instantiating, now three deep: the first
correction fixed a *remembered* number, the second fixed a *derivation*, and
the third fixed a *precision*. Each left the next level unexamined. A number
quoted to three significant figures from one sample is an assumed value
presented as an observed one, which is the same defect one level down -- and a
range quoted from one set of five is that defect one level down again, which is
why the paragraph above says so instead of stopping at the ranges.

`32768x1024` is the cell the investigation started from, and its `-13.8%` is the
headline: 3109 GB/s was measured against a resident MALL. It is fixed by the
rotation growing 2 -> 4 buffers, which pushes the working set to 512.008 MiB and
past the MALL on its own. **The gate does not touch it** -- 512.008 MiB is above
the `2 x 256 MiB` threshold, so the evictor stays off. An earlier version of the
source comment claimed the margin was what rescued this cell; that was wrong,
and @Reviewer caught it against the code.

## Which half of the change did it

**Withdrawn as a causal decomposition, 2026-08-02.** This section used to be
titled as above and read the four columns as a 2x2 factorial. It is not one.
@Reviewer called the attribution unsupported; checking the artifact, the
isolates are confounded in two ways, and the second one I had not noticed
either.

**Confound 1: `isolate_targetonly` also flips the gate.** It keeps the *old*
predicate `ws < l2_target_bytes` but feeds it the *new* 768 MiB target. The old
predicate against a 768 MiB target is true for every shape here except
`32768x2048`. So the arm labelled "target only" turns the evictor on at three
of four shapes. It never isolated the target.

**Confound 2: the two arms do not run the same evictor.** `_L2Evictor` is
allocated at `rotation_target_bytes`, so `isolate_gateonly` (12 MiB target)
performs a 12 MiB copy between rotations while `after` performs a 768 MiB one.
"Evictor on" is not one treatment across the columns; it is two different
kernels moving 64x different bytes.

What the runs actually sample is the physical configuration `(rotation working
set, evictor on/off)`, and the four arms do not cover its corners:

| shape | corners covered | duplicated arms |
| --- | --- | --- |
| `512x4096` | 3 of 4 | target-only and after are the same config |
| `4096x4096` | 4 of 4 (with `isolate_nomargin`) | target-only ~ after |
| `32768x1024` | 4 of 4 | none |
| `32768x2048` | 2 of 4 | gate-only ~ before; target-only ~ after |

At `32768x2048` neither "isolate" changed the evictor at all, so its three
columns are three samples of two configurations and the +2.9% is run-to-run
variation, not an effect of either half.

The numbers themselves are unchanged and remain in the artifact; only the
attribution is withdrawn:

| shape | before | gate only | target only | after |
| --- | --- | --- | --- | --- |
| `512x4096` | 860 (F, 16.016 MiB) | 932 (T, 16.016 MiB) | 1048 (T, 32.031 MiB) | 1050 (T, 32.031 MiB) |
| `4096x4096` | 3316 (F, 128.016 MiB) | 3498 (T, 128.016 MiB) | 3019 (T, 256.031 MiB) | 3009 (T, 256.031 MiB) |
| `32768x1024` | 3118 (F, 256.004 MiB) | 2635 (T, 256.004 MiB) | 2673 (T, 512.008 MiB) | 2703 (F, 512.008 MiB) |
| `32768x2048` | 3082 (F, 512.008 MiB) | 3172 (F, 512.008 MiB) | 3190 (F, 768.012 MiB) | 3273 (F, 768.012 MiB) |

**"`32768x1024` is all gate" is retracted.** It appears in this file and in
`1486fda`'s commit body and it is not supported. Reading the row as a factorial:
gate-only is -482.9 GB/s, target-only is -444.3, and if those were independent
effects the after cell would read 2190. It reads 2703. The interaction term is
**+512.8 GB/s** — larger than either main effect — which is the signature of two
arms reaching the same outcome by different routes, not of one dominant factor.
And the shipped configuration has the evictor **off** at this cell. A claim that
the gate is doing the work is refuted by the shipped run's own metadata.

Every arm is also a **single run**, against before-side run-to-run ranges of
1.9-17.7% on these same cells. Even were the design clean, one run per corner
would not support attribution.

What the artifact does support, stated without a causal claim: at all four
shapes the before configuration and the after configuration differ, the
directions are not uniform (`512x4096` and `32768x2048` up, `4096x4096` and
`32768x1024` down), and at `4096x4096` an intermediate configuration
(128 MiB resident + evictor on) reads *higher* than either endpoint, which is
enough to show the two changes cannot be reasoned about one at a time. That
last point was the useful content of this section and it survives.

A real decomposition needs the rotation count and the evictor forced
independently by flags rather than by editing predicates, the evictor
allocation held fixed across arms, and repeats per corner. That is not done.

## What the 2x margin actually decides

The margin only matters for cells whose working set lands between `1x` and `2x`
the MALL. Over the 90-cell matrix that is **13 cells**, all of them 4096-row
shapes between 256.03 and 384.19 MiB: `4096x4096` fwd (4 modes), `4096x3000` bwd
(4), `4096x4096` bwd (4), and `4096x3000` fwd fp32. Everything else is decided
identically by `<= llc` and `<= 2 * llc`.

Of the four shapes measured here, exactly one is in that band: `4096x4096` fwd.
Measured directly, with only the margin changed:

```
bare `<= llc_bytes`   evictor OFF   3535 GB/s
shipped `<= 2 * llc`  evictor ON    3010 GB/s   (4-run median)
```

So the margin is load-bearing -- worth 17% at the one cell here that it decides,
and in the direction that matters, since 3535 is the MALL-resident reading. This
does not make `2` a measured constant. The 256->288 MiB decay is gradual, so any
threshold in that region is a choice; what is measured is that a bare `1x` is
too low at 256.03 MiB.

Note what this comparison is and is not. Both arms hold the rotation at
256.031 MiB and differ only in whether the evictor runs, so unlike the four-arm
table above it is a genuine one-variable contrast -- but `isolate_nomargin` is
a **single run** against a four-run median, and the evictor allocation is
768 MiB in both, so it says nothing about the 12 MiB evictor. It is one clean
point, not a characterisation of the band.

The other boundary, for completeness: `32768x1024` after the change sits at
512.0078 MiB and clears the 512 MiB threshold by 8 KiB, so the evictor is off.
The `isolate_targetonly` run has the identical working set with the evictor on
and reads 2673 GB/s against a four-run after-median of 2679 (range 1.9%). The
knife-edge is worth 0.2% there, i.e. nothing measurable. @Reviewer's reading of
this pair is the right one and stronger than mine: it means **this single point
did not detect a gate cost**, which is not the same as bounding one. It does
not generalise to the 2x threshold and does not validate the margin.

## Not covered

Only four of the nine `COMPACT_SHAPES` and only bf16 forward. The remaining
cells, the backward pass, and the FlyDSL and quack providers are the subject of
the pending re-collection, not of this artifact. In particular none of the
published FlyDSL-vs-torch ratios have been re-measured under the fixed gate
here; the ones in `AI/flydsl_rmsnorm_notes.md` are still flagged as needing
re-collection.

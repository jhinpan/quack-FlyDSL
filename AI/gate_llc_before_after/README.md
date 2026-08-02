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

Each run's `environment.json` records its own `command`, so which binary
produced which directory is checkable from the artifact and not only from this
file. Three of those commands name scratch copies that were deleted after the
run, and cannot be re-read from the artifact -- reproduce them like this:

* `benchmarks/_bench_pre_gate_tmp.py` (the before runs) is
  `git show 94d3f8c:benchmarks/benchmark_rmsnorm_flydsl.py` byte for byte.
* `benchmarks/_v_gateonly.py` is the current file with
  `l2_target_bytes = llc_bytes * args.l2_target_ratio` replaced by
  `properties.L2_cache_size * args.l2_target_ratio`.
* `benchmarks/_v_targetonly.py` is the current file with
  `use_evictor = rotation_working_set_bytes <= 2 * llc_bytes` replaced by
  `rotation_working_set_bytes < l2_target_bytes`.
* `benchmarks/_v_nomargin.py` (the `isolate_nomargin` run) is the same line
  replaced by `rotation_working_set_bytes <= llc_bytes` -- margin dropped from
  2 to 1, everything else current.

These snapshots predate `fad422c`, which renamed `l2_target_bytes` to
`rotation_target_bytes`, added `evictor_threshold_bytes`, and bumped the schema
to 3. Reproducing them means reverting the named line in the version at
`d167701`, not in current head.

`before_{A,B}` and `after_{A,B}` carry the `float32` weight rows as well as
`same`; `before_{C,D}`, `after_{C,D}`, `after_v3_schema` and the three isolate runs are
`same` only.
Every table here is `same`, so the extra rows are context, not inputs.

The contention canary is `quiet: true` in all twelve runs, with
`closing_over_opening` in 0.975-1.020, so nothing here is a neighbour on the
node. `last_level_cache_bytes` is `268435456` with `torch_l2_cache_size`
`4194304` in the after runs; both keys are absent from the before runs because
that commit did not record them -- which is itself the defect.

These artifacts are schema **v2** and current head emits **v3**. A fresh run will
not match them field for field: `fad422c` renamed `l2_target_bytes` (which had
been carrying `3 x LLC` under an L2 name) to `rotation_target_bytes`, added
`evictor_threshold_bytes` per row, and added `last_level_cache_provenance`,
`rotation_target_bytes` and `evictor_gate` to the environment. The
`methodology.cache` string in these files also still describes the old "below
the L2 target / between individually timed calls" rule, which was wrong on both
counts. **None of this changed the timing path** -- the numbers below stand, and
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
812 / 823) and the individual before runs have p90/p10 spreads of 26% and 95%,
against 3% after. At 8-10 us per call this cell is close to launch overhead and
the before configuration is simply unstable; the honest statement is that the
after side is stable at 1050 and the before side was not measuring anything
repeatable. It should not be quoted as a speedup.

`32768x1024` is the cell the investigation started from, and its `-13.8%` is the
headline: 3109 GB/s was measured against a resident MALL. It is fixed by the
rotation growing 2 -> 4 buffers, which pushes the working set to 512.008 MiB and
past the MALL on its own. **The gate does not touch it** -- 512.008 MiB is above
the `2 x 256 MiB` threshold, so the evictor stays off. An earlier version of the
source comment claimed the margin was what rescued this cell; that was wrong,
and @Reviewer caught it against the code.

## Which half of the change did it

Single runs, `--weight-modes same`. Cell format: GB/s (evictor, rotation ws).

| shape | before | gate only | target only | after |
| --- | --- | --- | --- | --- |
| `512x4096` | 860 (F, 16.016 MiB) | 932 (T, 16.016 MiB) | 1048 (T, 32.031 MiB) | 1050 (T, 32.031 MiB) |
| `4096x4096` | 3316 (F, 128.016 MiB) | 3498 (T, 128.016 MiB) | 3019 (T, 256.031 MiB) | 3009 (T, 256.031 MiB) |
| `32768x1024` | 3118 (F, 256.004 MiB) | 2635 (T, 256.004 MiB) | 2673 (T, 512.008 MiB) | 2703 (F, 512.008 MiB) |
| `32768x2048` | 3082 (F, 512.008 MiB) | 3172 (F, 512.008 MiB) | 3190 (F, 768.012 MiB) | 3273 (F, 768.012 MiB) |

The two halves are not separable into "one matters and one does not", and they
do not act in the same direction everywhere:

* At `32768x1024` eviction is the whole story -- but note what delivers it. Hold
  the rotation at 256.004 MiB and only switch the evictor on: 3118 -> 2635. So
  the drop is a cache effect, not a rotation-size effect. In the shipped
  configuration, though, the evictor is *off* at this cell (512.008 MiB is over
  the threshold) and the same ~2679 is reached by the rotation being large
  enough to self-evict. Two different mechanisms, nearly the same number.
* At `4096x4096` the gate alone *raises* the number, 3316 -> 3498, and the
  target is what costs: 128 MiB of rotation is comfortably MALL-resident, and
  the 3498 figure is the evictor running while the operands stay cached, which
  is worse than either endpoint. Only doubling the rotation to 256.031 MiB
  brings it down to ~3010. This is the case that would have been missed by
  changing the gate alone.
* At `32768x2048` neither half flips the evictor and the cell moves +2.9%,
  inside twice its own run-to-run range. Cells already far past the MALL are
  approximately unaffected, as expected.

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

The other boundary, for completeness: `32768x1024` after the change sits at
512.0078 MiB and clears the 512 MiB threshold by 8 KiB, so the evictor is off.
The `isolate_targetonly` run has the identical working set with the evictor on
and reads 2673 GB/s against a four-run after-median of 2679 (range 1.9%). The
knife-edge is worth 0.2% there, i.e. nothing measurable.

## Not covered

Only four of the nine `COMPACT_SHAPES` and only bf16 forward. The remaining
cells, the backward pass, and the FlyDSL and quack providers are the subject of
the pending re-collection, not of this artifact. In particular none of the
published FlyDSL-vs-torch ratios have been re-measured under the fixed gate
here; the ones in `AI/flydsl_rmsnorm_notes.md` are still flagged as needing
re-collection.

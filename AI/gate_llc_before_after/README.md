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
* `after_{A,B,C,D}` -- harness at `d167701` (current).
* `isolate_gateonly` -- current harness with the target line reverted to
  `properties.L2_cache_size * ratio`. New gate, old 12 MiB target.
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

`before_{A,B}` and `after_{A,B}` carry the `float32` weight rows as well as
`same`; `before_{C,D}`, `after_{C,D}` and the two isolate runs are `same` only.
Every table here is `same`, so the extra rows are context, not inputs.

The contention canary is `quiet: true` in all ten runs, with
`closing_over_opening` in 0.983-1.020, so nothing here is a neighbour on the
node. `last_level_cache_bytes` is `268435456` with `torch_l2_cache_size`
`4194304` in the after runs; both keys are absent from the before runs because
that commit did not record them -- which is itself the defect.

The `methodology.cache` string in these `environment.json` files still describes
the old "below the L2 target" rule. The same commit that adds this directory
corrects that string, so a fresh run will not match the artifacts on that one
field. Nothing in the timing path changed with it.

## Cost of the change

Median of four runs per side. Working set and buffer count are from run A;
they are deterministic given the shape.

| shape | rotation ws MiB, before -> after | buffers | evictor | before GB/s | after GB/s | delta | run-to-run range, before / after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `512x4096` | 16.016 -> 32.031 | 2 -> 4 | F -> T | 842 | 1050 | +24.7% | 17.7% / 0.1% |
| `4096x4096` | 128.016 -> 256.031 | 2 -> 4 | F -> T | 3338 | 3013 | -9.7% | 4.2% / 0.7% |
| `32768x1024` | 256.004 -> 512.008 | 2 -> 4 | F -> F | 3109 | 2679 | -13.8% | 2.3% / 1.9% |
| `32768x2048` | 512.008 -> 768.012 | 2 -> 3 | F -> F | 3148 | 3239 | +2.9% | 2.7% / 2.1% |

Two of these need care before being read as "the gate cost N%".

`512x4096` is not a clean +24.7%. Its before-side range is 17.7% (860 / 961 /
812 / 823) and the individual before runs have p90/p10 spreads of 26% and 95%,
against 3% after. At 8-10 us per call this cell is close to launch overhead and
the before configuration is simply unstable; the honest statement is that the
after side is stable at 1050 and the before side was not measuring anything
repeatable. It should not be quoted as a speedup.

`32768x1024` is the cell the investigation started from, and its `-13.8%` is the
headline: 3109 GB/s was measured against a resident MALL.

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

* At `32768x1024` the gate is the whole story. Holding the rotation at
  256.004 MiB and only switching the evictor on drops 3118 -> 2635. The larger
  rotation then recovers a little (2673, 2703) because a 512 MiB set is past the
  MALL on its own.
* At `4096x4096` the gate alone *raises* the number, 3316 -> 3498, and the
  target is what costs: 128 MiB of rotation is comfortably MALL-resident, and
  the 3498 figure is the evictor running while the operands stay cached, which
  is worse than either endpoint. Only doubling the rotation to 256.031 MiB
  brings it down to ~3010. This is the case that would have been missed by
  changing the gate alone.
* At `32768x2048` neither half flips the evictor and the cell moves +2.9%,
  inside twice its own run-to-run range. Cells already far past the MALL are
  approximately unaffected, as expected.

## The 2x margin at its own boundary

`32768x1024` after the change sits at 512.0078 MiB against a `2 * llc_bytes`
threshold of exactly 512 MiB -- it clears by 8 KiB, so the evictor is off. The
`isolate_targetonly` run has the identical working set with the evictor on and
reads 2673 GB/s against a four-run after-median of 2679 (range 1.9%). The
knife-edge is therefore worth 0.2% at this cell, i.e. nothing measurable. That
is a bound on the cost of the arbitrary side of the margin at one point, not a
general result; it does not make 2x measured rather than chosen.

## Not covered

Only four of the nine `COMPACT_SHAPES` and only bf16 forward. The remaining
cells, the backward pass, and the FlyDSL and quack providers are the subject of
the pending re-collection, not of this artifact. In particular none of the
published FlyDSL-vs-torch ratios have been re-measured under the fixed gate
here; the ones in `AI/flydsl_rmsnorm_notes.md` are still flagged as needing
re-collection.

# schema-v1 sweep, 2026-07-26 — source of the Hopper regime table

These two files are the source for the `regime | Quack on H200 | Quack on H100`
table in `AI/flydsl_rmsnorm_notes.md`. They were nearly retracted as unsourced
on 2026-08-01 because the search had been scoped to the later Experiment No.001
archive, which is a different run and does not reproduce them. Kept here with
hashes so that does not happen again.

| file | rows | copy_roofline_gbps | sha256 (short) |
| --- | --- | --- | --- |
| `h100-v1-results.csv` | 180 (90 quack + 90 torch) | 2974.420002 | `59d9429b` |
| `h200-v1-results.csv` | 180 (90 quack + 90 torch) | 4148.155822 | `6a3cd36e` |

Full hashes in `SHA256SUMS`. All 90 Quack correctness gates pass in each file,
90 unique cell keys, no duplicates. Verified independently by two seats on
different runtimes.

**What these files do not carry.** schema-v1 has no commit SHA, toolchain
version, hostname or per-cell winning config. The "2026-07-26" date comes from
file mtimes and surrounding context, not from the data. So these bytes pin the
*numbers* in the table; they do not pin the *code or environment* that produced
them, and no causal claim about tuning-versus-machine can be settled from them
alone.

## The formula that reproduces the table

Filter `provider == quack`. Separately per operation (`fwd`/`bwd`), take the
median of `copy_roofline_pct` over each m-regime — `m == 32768`, `m == 4096`,
`m <= 512` — and round to whole percent.

| regime | H200 fwd/bwd | H100 fwd/bwd |
| --- | --- | --- |
| m == 32768 | 79.647 / 74.317 → 80 / 74 | 88.695 / 87.185 → 89 / 87 |
| m == 4096 | 32.899 / 27.149 → 33 / 27 | 60.952 / 41.220 → 61 / 41 |
| m <= 512 | 13.792 / 9.146 → 14 / 9 | 18.359 / 12.039 → 18 / 12 |

The prose figure of ~6.1 us for M=1 Quack forward is also from here: 5.936–6.128
us on H200 and 5.904–6.144 us on H100.

Reproduced independently twice, by two seats on different runtimes, from these
exact bytes.

## Two things not to do with these files

1. **Do not re-derive the table from the Experiment No.001 archive.** No.001 is a
   later schema-v2 run with different code, different probes and different
   results. Its natural Quack medians are incompatible with the values above.
   That mismatch is what triggered the false "unsourced" finding.

2. **Do not carry the H200 m=4096 caveat across datasets, and state it
   exactly.** In *these* files Quack is slower on H200 than H100 at 4096x3000
   and 4096x4096 in **20 of 20** matched cells (1.027–1.346x). The torch control
   mostly runs the other way but not uniformly: **17 of 20** faster on H200,
   range 0.778–1.242, with three backward `same` exceptions (4096x3000 fp16
   1.242, 4096x3000 bf16 1.236, 4096x4096 fp16 1.063). Do not quote
   "0.78–0.88x" as the range over all cells — it is the per-cell range of the 12
   `weight_dtype=float32` cells only (0.777923–0.883896), and the paired Quack
   1.03–1.33 is that same subset. All three counterexamples are
   `weight_mode=same`, which that subset excludes.

   In No.001 the direction reverses: H200 faster in **58 of 60** matched cells,
   the two exceptions being torch backward at 4096x3000 (fp16 1.081, bf16
   1.044). Not "every provider."

   The asymmetry is an observation, not an explained result. These files carry
   no commit, toolchain, host or config metadata, so a tuning explanation is not
   isolated from an environment one.

   Also note those two shapes are the only N values at m=4096, so they *are* the
   m==4096 row — they were never excluded from it.

## Related

`probe-tuned-h100.json` and `probe-tuned-h200.json` are archived here alongside
the CSVs. They record per-cell `tuned_config` strings from the same period and
are the source for the `use_tma=True, smem_stages=3` claim about the widest
backward row, which was also briefly and wrongly retracted as unsourced:

| host | 32768x8192 bwd winner | tuned/analytical | implied gain |
| --- | --- | --- | --- |
| H100 | `use_tma=True, smem_stages=3` | 0.911 | 8.90% |
| H200 | `use_tma=True, smem_stages=3` | 0.8325 | 16.75% |

At 32768x2048 bwd the H100 winner is `use_tma=True, smem_stages=2` (0.9657) and
the H200 winner is the analytical config unchanged (1.0042).

**Two things these probes do not establish.**

*Not knob isolation.* The winners change several knobs simultaneously — H100
four (`reload_wdy`, `reload_x`, `use_tma`, `smem_stages`), H200 six (those plus
`num_threads`, `threads_per_row`, `cluster_n`). Crediting the win to
`use_tma`/`smem_stages` is a selection, not a measurement. A one-knob-at-a-time
ablation would be needed.

*Coverage.* Each JSON holds **4 cells only**: `32768x2048` and `32768x8192`,
fwd and bwd, **bfloat16 only**. No other shape, no fp16, no fp32, no
weight-mode variation. Any statement drawn from these files is a statement
about those four cells.

*Not a mechanism boundary.* The probe sampled only N=2048 and N=8192, so it
cannot speak to N=4096. In the No.001 archive H200 `32768x4096` bwd bf16/same
goes 241.44 -> 219.57 us, a 9.06% gain. The gain therefore falls off gradually
(N=8192 +17.5%, N=4096 +9.1%, N=2048 -0.7%, N=1024 -0.9%) rather than being
confined to the widest row.

**These probes are a separate run from No.001 — do not equate them.** The
published No.001 gains at the same cell are 9.155% (H100) and 17.479% (H200),
against these probes' 8.90% and 16.75%. Close, but not identical, so the probes
corroborate *which config wins* and roughly how much it wins by; they are not
the timing record behind the published numbers. The two probe files also differ
from each other in toolchain — H100 ran torch 2.11.0+cu130, H200 torch
2.9.1+cu128 — so they are not even a matched pair between hosts, and neither
records a Quack commit, command line, date or cache identity. The generating
script is archived here as `probe_tuned_vs_analytical.py`
(sha256 `d2cd0915496f296f2ef9f8366aaa4ec7e5a6e7d05bb4bde3c8e550ed62d1faa7`),
but nothing cryptographically links either JSON to a source commit.

## Still unarchived

This directory covers the two **Quack Hopper** columns only. The MI355X column
of the same table (six regime values, the 5279 GB/s copy roofline, the 13.0 us
M=1 figure and the ~3.5 us Python/FFI floor) has no archived source here and
remains unverified. Do not treat the presence of this directory as provenance
for the whole table.

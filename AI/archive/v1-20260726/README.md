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
   "0.78–0.88x" as a per-cell range — those are aggregated dtype medians.

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

| host | 32768x8192 bwd winner | tuned/analytical |
| --- | --- | --- |
| H100 | `use_tma=True, smem_stages=3` | 0.911 |
| H200 | `use_tma=True, smem_stages=3` | 0.8325 |

At 32768x2048 bwd the H100 winner is `use_tma=True, smem_stages=2` (0.9657) and
the H200 winner is the analytical config unchanged (1.0042), which independently
supports the gain being confined to the widest row.

## Still unarchived

This directory covers the two **Quack Hopper** columns only. The MI355X column
of the same table (six regime values, the 5279 GB/s copy roofline, the 13.0 us
M=1 figure and the ~3.5 us Python/FFI floor) has no archived source here and
remains unverified. Do not treat the presence of this directory as provenance
for the whole table.

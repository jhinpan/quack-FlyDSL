# Cross-vendor rmsnorm forward: FlyDSL/MI355X against CuTeDSL/H200

Collected 2026-08-02. Same harness source on both hosts, same shapes, same
`--sample-rounds 40`, fwd only, bfloat16, `weight_mode=same`. One GPU each,
selected idle: MI355X device 6 (`HIP_VISIBLE_DEVICES=6`), H200 device 5
(`CUDA_VISIBLE_DEVICES=5`) on `hyper00` where four of eight cards were running
someone else's work and were left alone.

These are **v4** artifacts: they were produced by the harness immediately
before the `peak_bw_probe` column was added, which is the column the finding
below is about. They are frozen at v4 for that reason.

| shape | cutedsl us (H200) | flydsl us (MI355X) |
| --- | --- | --- |
| 1x4096 | 17.1 | 4.2 |
| 256x4096 | 16.7 | 4.5 |
| 512x4096 | 17.0 | 5.1 |
| 4096x3000 | 39.9 | 13.7 |
| 4096x4096 | 38.4 | 16.2 |
| 32768x1024 | 49.9 | 28.8 |
| 32768x2048 | 84.7 | 52.3 |
| 32768x4096 | 146.0 | 94.9 |
| 32768x8192 | 276.2 | 186.8 |

**Microseconds across vendors are not a kernel comparison** and this table is
not one. Two different parts with different memory systems; the H200's
`two_read_one_write` ceiling here measured 4314 GB/s against the MI355X's
6664 GB/s `write`. The table is recorded because it is what was run, not
because the ratio means what it looks like.

## The finding: `peak_bw_pct` is not the cross-vendor fix either

`comparison_scope` in every artifact before schema v5 said, of exactly this
comparison, *"compare peak_bw_pct instead"*. That advice is wrong, and wrong
in a way that changes the answer rather than blurring it.

`peak_bw_pct` divides by `achievable_bandwidth.median_gbps`, which is the best
of three probes -- and **a different probe won on each host**:

| probe | H200 GB/s | MI355X GB/s |
| --- | --- | --- |
| copy | 4147 | 4644 |
| two_read_one_write | **4314** (winner) | 6068 |
| write | 3269 | **6664** (winner) |

So the two `peak_bw_pct` columns are ratios against different references. Held
to one common probe, the same rows give three different stories:

| shape | vs copy | vs two_read_one_write | vs write |
| --- | --- | --- | --- |
| 4096x3000 | +47.5 | +30.5 | +16.1 |
| 4096x4096 | +47.1 | +27.8 | +8.7 |
| 32768x1024 | +35.6 | +14.5 | -12.3 |
| 32768x2048 | +34.1 | +11.1 | -19.9 |
| 32768x4096 | +33.1 | +8.0 | -27.6 |
| 32768x8192 | +30.0 | +4.6 | -32.7 |

(Percentage points, FlyDSL minus cutedsl.) At 32768x8192 the denominator moves
the result from FlyDSL ahead by 30.0 points to behind by 32.7. **It inverts the
conclusion, it does not merely scale it.** Under `copy` the MI355X figures
exceed 100%, which is its own signal that `copy` understates that memory
system rather than that the kernel beat the hardware.

Schema v5 adds `peak_bw_probe` to every row. That does not make the columns
comparable -- nothing in the artifact can. It makes the incomparability visible
to a reader who has only `results.csv`, which is the part that was missing:
before v5 the sign error above was reachable from published columns with no
indication anything was wrong.

## What can be said

Nothing here is a like-for-like kernel comparison, because no like-for-like
hardware was available. What the run does support:

- On its own part, FlyDSL reaches 86.3% of the best measured MI355X ceiling at
  32768x8192 and 84.9% at 32768x4096. cutedsl reaches 90.1% and 85.2% of the
  best measured H200 ceiling at the same shapes. Both are near their machine's
  roofline at long rows, each measured against its own host's winning probe --
  which is the one reading where using each host's own winner is the right
  choice, since the question is "how close to this machine's ceiling".
- The small-shape column is a launch-overhead comparison, not a bandwidth one:
  cutedsl sits at a flat ~17 us from M=1 to M=512 while the logical bytes grow
  512x, so that number is dispatch, not memory. FlyDSL's ~4.2-5.1 us over the
  same span is the same kind of floor at a different height. Neither figure is
  evidence about the kernels' memory behaviour, and the M=1 row -- 1.4 GB/s
  against 5.8 GB/s, both ~0% of peak -- is the clearest statement of that.

## Not established

- **bwd is not covered here.** Forward only.
- **fp16 and fp32, and `weight_mode=float32`, are not covered.**
- **No repeat run.** Single collection per host, so the run-to-run spread is
  unmeasured and no stability claim is made. The `4096x4096` MI355X cell was
  re-read once afterwards at 16.30 us against 16.19 (0.7%), which is one
  canary and not a spread.
- **The 32768x2048 row has identical medians for both providers on H200**
  (84.727999 us). p10/p90 differ, so it is a tie at this timer's resolution
  rather than a duplicated row, but it should not be read as an exact match.

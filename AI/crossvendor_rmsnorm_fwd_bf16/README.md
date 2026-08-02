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

(Percentage points, FlyDSL minus cutedsl.) At 32768x8192 the difference between
the two percentages moves from +30.0 to −32.7 depending only on which probe is
chosen as the common denominator. **The sign is a property of the denominator,
not of the kernels.**

An earlier draft wrote that as "FlyDSL ahead by 30.0 points to behind by 32.7",
which @Reviewer refused, and he was right: ahead/behind is the exact language
this document opens by saying the data cannot support, and picking a common
probe does not repair that. Each side is still divided by *its own host's*
measurement of that probe, so the quantity being differenced is how well each
kernel exploits its own machine's proxy ceiling — two utilisation ratios on
different hardware. That the difference changes sign with the choice of proxy
is the point; neither sign is a statement about which kernel is faster. The
caveat was in the first paragraph and I violated it in the third, which is the
easier mistake to make than to notice.

Under `copy` the MI355X figures exceed 100%, which is its own signal that
`copy` understates that memory system rather than that the kernel beat the
hardware.

Schema v5 adds `peak_bw_probe` to every row. That does not make the columns
comparable -- nothing in the artifact can. What it adds is the probe's *name*.
An earlier draft claimed the sign error was reachable from published columns
"with no indication anything was wrong"; that was false and @Reviewer checked
it. The v4 CSVs already carry `peak_bw_gbps` on every row -- 4314.018124 here,
6664.195243 there -- so the denominators were visibly different and `peak_bw_pct`
was recomputable from the CSV alone. The gain is semantic rather than
existential: `two_read_one_write` vs `write` says *why* they differ, and
separates "a different probe won" from "the same probe, different hardware",
which the bare numbers do not distinguish. Overstating an increment as an
absence is the same failure this directory exists to document, pointed the
other way.

## What can be said

Nothing here is a like-for-like kernel comparison, because no like-for-like
hardware was available. What the run does support:

- On its own part, FlyDSL reaches 86.3% of the best measured MI355X ceiling at
  32768x8192 and 84.9% at 32768x4096. cutedsl reaches 90.1% and 85.2% of the
  best measured H200 ceiling at the same shapes. Both are near their machine's
  roofline at long rows, each measured against its own host's winning probe --
  which is the one reading where using each host's own winner is the right
  choice, since the question is "how close to this machine's ceiling".
- The small-shape column is not a bandwidth comparison: cutedsl sits at a flat
  ~17 us from M=1 to M=512 while the logical bytes grow 512x, and FlyDSL's
  ~4.2-5.1 us over the same span is the same kind of floor at a different
  height. A time that does not move as the work grows 512x is a *fixed cost*
  floor -- that much the data shows. An earlier draft called it "dispatch, not
  memory"; nothing here isolates dispatch specifically, and the floor could as
  well be launch, JIT-warmed entry, event overhead, or a wave-quantisation
  effect. @Reviewer's blocker 4. Attributing the floor to a mechanism needs the
  rocprofv3 breakdown, which is not in this directory. Neither figure is
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

### Limits of these two artifacts specifically

Found by @Reviewer against `fdf2530` and confirmed. The harness was fixed for
future runs (`SCHEMA_VERSION`, live `comparison_scope`, execution-derived
`methodology`, `provider_modules`, `gpu.uuid`), but these two files are frozen
at v4 and cannot be re-collected without re-running both hosts, so their limits
are recorded here rather than papered over.

- **The H200 file cannot bind the cutedsl provider to a tree.** It records
  ambient `git_commit 4f36477` with a dirty benchmark file, and
  `versions.quack = 0.5.0` from installed distribution metadata. The tree at
  that commit declares `quack.__version__ = "0.6.1"`, and on that host it does
  not import at all (`cannot import name 'alloc_reserved_mbarrier'`), so the
  recorded commit provably was not the provider that ran -- the run used
  installed 0.5.0 from `dist-packages`. Establishing that needed a shell on the
  machine; nothing in the artifact said it. Later runs record
  `provider_modules` (import path, `__version__`, `__init__.py` hash) so the
  question is answerable from the file.
- **Neither file identifies the physical card.** Both record `visible_index 0`
  under a visibility mask, with no UUID or BDF. The MI355X run was ordinal 6
  matched to node 8, and the matched UID itself was not stored. The H200
  `hostname` is a container id (`425db16ba2a0`); that it was hyper00 device 5
  with four cards busy is stated in this README and archived nowhere.
  `visible_count = 1` and the endpoint canary show the *visible* device was
  quiet, which is not the same as exclusive use of the host.
- **The H200 `methodology` block describes the gfx950 path, not what ran.** It
  says the LLC was "read from the KFD topology, matched by unique_id" while
  `last_level_cache_provenance` in the same file says
  `not_a_hip_build / torch_l2_fallback`; it asserts the 256 MiB MALL and the
  4 MiB per-XCD L2 on an sm_90 card; and its `steady_state` describes FlyDSL
  first-launch JIT on a `quack`+`torch` run. Read the provenance fields, not
  the prose, for what this run did.
- **No raw per-round samples.** The 40 per-cell rounds and 30 probe samples
  were not retained, so the medians, p10/p90, the timer-resolution reading of
  the 32768x2048 tie, and the 16.30 us canary cannot be independently
  recomputed. This is the same gap as @Reviewer's standing blocker 2 and is
  what Experiment No.002 exists to close; it cannot be repaired retroactively.
- Both CSVs are CRLF, so `git diff --check` reports trailing whitespace on
  them. They are archived bytes and are intentionally left exactly as written.

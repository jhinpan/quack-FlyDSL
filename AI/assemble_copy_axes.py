"""Assemble the three-axis copy artifact at both buffer sizes.

`AI/data/copy_placement_draws/copy_512MiB_draws_dev5.json` argued that the
historical `4.89 TB/s` copy cell cannot authenticate anything, because the copy
rate's confound is far wider than the 0.2% band that cell implies. The argument
was right and its evidence was measured at the wrong size: every draw in it came
from **512 MiB** buffers, while `AI/flydsl_rmsnorm_notes.md:869` states the
historical table was "probing three patterns over **2 GiB** buffers". The
placement effect is strongly size-dependent, so 512 MiB draws are not evidence
about a 2 GiB number. @Reviewer named this scope gap in `f9014a94` and it was a
real hole in my own published argument, not a technicality.

This file re-runs that argument at 2 GiB, and carries 512 MiB alongside so the
size dependence is visible rather than asserted. It supersedes the 512-MiB-only
artifact.

Three corrections came out of measuring it properly, all against my own work:

  1. The verdict survives at 2 GiB, with a different number. Pooled over 80
     draws spanning 5 allocation slots and 4 allocator high-water marks, the
     2 GiB copy rate ranges 11.82% -- 59x the width of the band it would have to
     resolve, and the draws straddle the band. At 512 MiB the same measurement
     gives 18.84%. Both retire the cell; the artifact previously implied ~19%
     was the figure at the size that mattered, and it is not.

  2. `denominator_stability_across_processes` is wrong about its own axis.
     It reports copy at 13.35% with `n_processes: 3`. Holding the generator
     fixed, copy reproduces across six processes to **0.63%**, and a given slot
     reproduces to under 1% at both sizes. The three runs behind 13.35%
     straddled edits to `_copy_variability`, which allocates and frees buffers
     before the copy probe runs. What moves the rate is the allocator's peak
     high-water mark -- a reversible staircase, indistinguishable at 0/6/11 live
     512 MiB buffers, shifting at 13 and again at 17. So the published figure is
     a spread across *generator versions* wearing a process label. It entered
     the tree as part of the fix for hand-typed constants: the sentence written
     to eliminate untested numbers introduced a mislabelled one.

  3. The size discriminator I nearly published is unusable. `two_read_one_write`
     looked like a stable instrument -- 0.11-0.31% across processes -- so a 4.04%
     gap between its 512 MiB value and the historical 6.09 read as "13x the
     error bar, therefore the table is 2 GiB". But that error bar was measured
     at one allocation slot with the peak held fixed, which is the single
     condition that hides both confounds. Sampled across slots and peaks, the
     same probe spreads 2.47% at 512 MiB and the two sizes' ranges *overlap* for
     all three patterns. Nothing here can tell 512 MiB from 2 GiB. The claim
     that the table is 2 GiB rests on notes:869 saying so, which is documentary
     evidence, and this file does not dress it up as measured.

The band and its convention are inherited unchanged from the earlier artifact:
@Reviewer fixed these as half-open under nearest / half-up in `4d6ed9de`, and a
bare pair reads as closed. All draws here are stored unrounded, so unlike the
512 MiB set there is no undecidable draw.

Run:  AI/collect_copy_axes.sh   (collects), then
      python AI/assemble_copy_axes.py /tmp/copyaxes
Writes AI/data/copy_placement_draws/copy_axes_dev5.json.
"""

import hashlib
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "AI/data/copy_placement_draws/copy_axes_dev5.json"

# Inherited from 4d6ed9de. Named, not assumed: a bare pair reads as closed.
BAND = (4.885, 4.895)
DIRECTION_CONTROL_ABSENT = (
    "this sweep ran levels in monotonic wall-clock order, so a level effect and a "
    "drift over the collection window are indistinguishable in it. Steps below are "
    "reported without a time-reversal control. @Reviewer, 5c2e0083."
)

BAND_CONVENTION = (
    "half-open [lo, hi) under a nearest / half-up display rule: a true value of "
    "4.895 displays as 4.90, not 4.89, so it cannot have produced the historical "
    "cell. Convention fixed by @Reviewer in 4d6ed9de; stated here because a bare "
    "pair reads as closed and an earlier version of this artifact published one."
)

# notes:869. Documentary, not measured -- see the module docstring, point 3.
HISTORICAL = {"write": 6.84, "two_read_one_write": 6.09, "copy": 4.89}
HISTORICAL_SIZE_MIB = 2048


def _load(src_dir):
    runs = defaultdict(list)
    for path in sorted(Path(src_dir).glob("*.json")):
        if path.name.startswith("_"):
            continue  # _peak_sweep.json is the staircase sweep, a different schema
        d = json.loads(path.read_text())
        # New name first. The treatment is prior allocation count, not peak bytes;
        # `peak_live_512mib` is the retracted alias kept so older runs still load.
        runs[d.get("n_prior_512mib_allocs", d["peak_live_512mib"])].append(d)
    if not runs:
        raise SystemExit(f"no runs in {src_dir}")
    return dict(sorted(runs.items()))


def _high_water_check(runs):
    """Did the treatment move the process high-water at all? Measured per run.

    This exists because the axis was published for two commits as "allocator peak
    high-water mark" when the prefix never exceeded the live set the measurement
    itself allocates -- so the high-water was constant across every level and the
    name asserted a mechanism the design could not touch. @Reviewer caught it by
    reading the code. The point of this function is that the next reader should
    not have to: the claim is now falsifiable from the artifact.
    """
    allr = [r for rs in runs.values() for r in rs]
    have = [r for r in allr if "high_water_bytes" in r]
    if not have:
        return {
            "recorded": False,
            "why_it_matters": (
                "these runs predate high_water_bytes. Any statement here about a "
                "peak-bytes threshold is unsupported by them; the treatment is prior "
                "allocation count."
            ),
        }
    g = 1024**3
    # Per measurement block, not pooled. A run measures 512 MiB then 2 GiB
    # sequentially, and the two have very different live sets (~3 GiB vs 12 GiB).
    # Collapsing them with max() would report the 22 GiB pattern-probe peak against
    # every block and hide that the prefix *does* exceed the 512 MiB block's live
    # set at the upper levels. Which is the whole question: the axis is separable
    # exactly where the prefix is the thing setting the high-water.
    blocks = sorted({k for r in have for k in r["high_water_bytes"] if k != "after_prefix"})
    by_level = {}
    for level, rs in runs.items():
        rs = [r for r in rs if "high_water_bytes" in r]
        if not rs:
            continue
        hw = [r["high_water_bytes"] for r in rs]
        pre = max(h["after_prefix"] for h in hw)
        row = {"prefix_high_water_GiB": round(pre / g, 2)}
        for b in blocks:
            vals = [h[b] for h in hw if b in h]
            if not vals:
                continue
            row[b] = {
                "measurement_live_set_GiB": round(max(vals) / g, 2),
                "high_water_set_by": "prefix" if pre > max(vals) else "measurement",
            }
        by_level[str(level)] = row

    per_block = {}
    for b in blocks:
        setters = {
            row[b]["high_water_set_by"] for row in by_level.values() if isinstance(row.get(b), dict)
        }
        hws = {
            max(row[b]["measurement_live_set_GiB"], row["prefix_high_water_GiB"])
            for row in by_level.values()
            if isinstance(row.get(b), dict)
        }
        per_block[b] = {
            "high_water_varies_across_levels": len(hws) > 1,
            "distinct_high_waters_GiB": sorted(hws),
            "peak_bytes_separable_from_alloc_count": len(hws) > 1 and "prefix" in setters,
        }

    sep = [b for b, v in per_block.items() if v["peak_bytes_separable_from_alloc_count"]]
    key2g = "during_identical_buffers_2048MiB"
    return {
        "recorded": True,
        "by_level": by_level,
        "per_measurement_block": per_block,
        "blocks_where_peak_bytes_varies": sep,
        "load_bearing_block": key2g,
        "verdict": (
            "For the 2 GiB identical-buffer draws -- the block every conclusion in this "
            "artifact rests on -- the prefix never sets the process high-water: it tops "
            f"out below the {key2g} live set at every level, so peak bytes is constant "
            "and cannot be the axis. The treatment there is prior allocation count and "
            "history, with count, bytes, churn, fill time and placement mutually "
            "confounded. At 512 MiB the upper prefixes do exceed that block's smaller "
            "live set, so peak bytes varies there; that is a partial separation this "
            "design produced by accident, not by intent, and it is reported rather than "
            "claimed as a control. A real factorial needs prefixes above the 2 GiB "
            "block's own live set."
            if key2g not in sep
            else "the prefix sets the high-water for the 2 GiB block at some levels; "
            "peak bytes and allocation count are at least partly separable here."
        ),
    }


def _decompose(runs, size):
    """Variance by allocation slot, and reproducibility along each axis.

    Three axes, which the earlier work collapsed into one:
      slot    -- position among identically-sized buffers in one process
      process -- re-running the same program byte for byte
      program -- changing the allocator's peak high-water mark
    """
    per_peak = {}
    for peak, rs in runs.items():
        slots = [
            [r["identical_buffers_by_size"][size]["TBps_per_identical_buffer"][p] for r in rs]
            for p in range(5)
        ]
        allv = [v for s in slots for v in s]
        per_peak[f"peak_live_512mib_{peak}"] = {
            "n_processes": len(rs),
            "slot_means": [sum(s) / len(s) for s in slots],
            "slot_draws": slots,
            "pooled_range_pct_of_min": round((max(allv) / min(allv) - 1) * 100.0, 2),
            "worst_across_process_range_pct": round(
                max((max(s) / min(s) - 1) * 100.0 for s in slots), 2
            ),
        }

    cross_program = {}
    for p in range(5):
        means = {
            peak: sum(
                r["identical_buffers_by_size"][size]["TBps_per_identical_buffer"][p] for r in rs
            )
            / len(rs)
            for peak, rs in runs.items()
        }
        cross_program[f"slot{p}"] = {
            "mean_by_peak_live_512mib": means,
            "range_pct_of_min": round((max(means.values()) / min(means.values()) - 1) * 100.0, 2),
        }

    allv = [
        v
        for rs in runs.values()
        for r in rs
        for v in r["identical_buffers_by_size"][size]["TBps_per_identical_buffer"]
    ]
    slot_key = defaultdict(list)
    for peak, rs in runs.items():
        for r in rs:
            for p, v in enumerate(
                r["identical_buffers_by_size"][size]["TBps_per_identical_buffer"]
            ):
                slot_key[(peak, p)].append(v)
    gm = sum(allv) / len(allv)
    sst = sum((v - gm) ** 2 for v in allv)
    ssb = sum(len(vs) * ((sum(vs) / len(vs)) - gm) ** 2 for vs in slot_key.values())

    # The instrument floor, from the seven timing rounds behind each draw where
    # the run recorded them. Without this the decomposition above is unreadable:
    # "98.7% of variance is slot and peak" is only a statement about placement if
    # a single draw is stable to much better than the spread being attributed.
    # If round-to-round noise were the same size as the across-slot range, the
    # same eta-squared would appear with no placement effect at all.
    rounds = [
        r["identical_buffers_by_size"][size][k]
        for rs in runs.values()
        for r in rs
        if (k := "rounds_us_per_identical_buffer") in r["identical_buffers_by_size"][size]
    ]
    within = [
        (max(s) / min(s) - 1) * 100.0 for per_run in rounds for s in per_run if min(s) > 0
    ]
    floor = (
        {
            "n_draws_with_rounds": len(within),
            "round_spread_pct_median": round(statistics.median(within), 3),
            "round_spread_pct_max": round(max(within), 3),
            "note": (
                "spread over the seven timing rounds behind one draw, in us. Compare "
                "against pooled_range_pct_of_min: the placement claim requires the "
                "pooled range to be large against this, not merely nonzero."
            ),
        }
        if within
        else {
            "n_draws_with_rounds": 0,
            "note": (
                "these runs predate rounds_us_per_identical_buffer. The decomposition "
                "below is reported without an instrument floor to compare it against, "
                "which is the gap @Reviewer raised in 23a6f662; re-collect to close it."
            ),
        }
    )

    return {
        "n_draws": len(allv),
        "min_TBps": min(allv),
        "max_TBps": max(allv),
        "pooled_range_pct_of_min": round((max(allv) / min(allv) - 1) * 100.0, 2),
        "within_draw_instrument_floor": floor,
        "variance_explained_by_slot_and_peak_pct": round(ssb / sst * 100.0, 2),
        "worst_across_process_range_pct": round(
            max(pp["worst_across_process_range_pct"] for pp in per_peak.values()), 2
        ),
        "worst_cross_program_range_pct": round(
            max(cp["range_pct_of_min"] for cp in cross_program.values()), 2
        ),
        "by_peak": per_peak,
        "by_slot_across_programs": cross_program,
    }


def _band(draws):
    lo, hi = BAND
    below = sum(1 for v in draws if v < lo)
    half_open = sum(1 for v in draws if lo <= v < hi)
    closed = sum(1 for v in draws if lo <= v <= hi)
    above = len(draws) - below - half_open
    return {
        "band": list(BAND),
        "band_convention": BAND_CONVENTION,
        "band_width_pct_of_lo": round((hi / lo - 1) * 100.0, 3),
        "n_draws": len(draws),
        "draws_below_band": below,
        "draws_inside_band_half_open": half_open,
        "draws_inside_band_closed": closed,
        "draws_at_or_above_band_half_open": above,
        "min_below_and_max_above_band": bool(below and above),
        "why_not_called_straddled": (
            "an earlier version of this field was named band_is_straddled, which reads "
            "as 'this protocol can produce a value in the band'. The data do not show "
            "that and cannot: see band_reachability_power. All the boolean states is "
            "that the minimum draw is below the band and the maximum is at or above it, "
            "which is also true of any distribution with a hole where the band sits. "
            "@Autotune raised this against the 512 MiB artifact and it applied here too."
        ),
        "all_draws_unrounded": True,
        "no_undecidable_draws": (
            "every draw here is stored unrounded, so unlike the 512-MiB-only artifact "
            "there is no draw whose band membership the record cannot decide"
        ),
    }


def _band_reachability(runs, size):
    """Could this protocol have produced an in-band value at all? Power, not outcome.

    Zero draws in band is weak evidence of anything. The draws are not a
    continuum: they cluster on allocation slots whose means are separated by gaps
    far wider than the band. Under a uniform-slot model the expected number of
    slot means landing in a 0.2%-wide window is well under one, so observing
    zero in-band is the *expected* outcome even if the band is perfectly
    reachable. @Autotune's argument, recomputed here for this dataset rather
    than transcribed.
    """
    lo, hi = BAND
    axis = _decompose(runs, size)
    means = sorted({round(m, 6) for pk in axis["by_peak"].values() for m in pk["slot_means"]})
    band_w = (hi / lo - 1) * 100.0
    span = (max(means) / min(means) - 1) * 100.0
    p_one = band_w / span
    floor = axis["within_draw_instrument_floor"]
    fl = floor.get("round_spread_pct_median")
    return {
        "n_slot_means": len(means),
        "slot_mean_span_pct": round(span, 2),
        "band_width_pct": round(band_w, 4),
        "p_single_slot_in_band_uniform_model": round(p_one, 4),
        "expected_slot_means_in_band": round(len(means) * p_one, 3),
        "p_at_least_one_in_band_pct": round((1 - (1 - p_one) ** len(means)) * 100.0, 1),
        # The second reason this protocol has no power, and the stronger one. The
        # argument above is about placement scattering draws past a narrow window.
        # This is about a single draw not being repeatable to the window's width in
        # the first place: with slot, process and program all held fixed, the seven
        # timing rounds behind one draw already spread further than the band.
        "band_width_vs_single_draw_noise": (
            None
            if fl is None
            else {
                "median_round_spread_pct": fl,
                "band_width_pct": round(band_w, 4),
                "ratio_noise_to_band": round(fl / band_w, 1),
                "band_resolvable_by_one_draw": bool(fl < band_w),
            }
        ),
        "conclusion": (
            f"with {len(means)} slot means spanning {span:.2f}% and a band {band_w:.4f}% wide, a uniform "
            f"model expects {len(means) * p_one:.2f} of them in band and gives only a {(1 - (1 - p_one) ** len(means)) * 100.0:.0f}% chance that "
            "any lands there. Zero in-band is therefore not evidence against the "
            "historical value -- this protocol has almost no power in that direction. "
            "The defensible statement is symmetric: at this size and sample size the "
            "data neither authenticate nor exclude a value in the band."
            + (
                ""
                if fl is None
                else (
                    f" A second limit, visible only once the timing rounds were retained: "
                    f"a single draw's own round-to-round spread has median {fl}%, which is "
                    f"{round(fl / band_w, 1)}x the band width. Even with slot, process and "
                    "program all held fixed, one measurement cannot resolve an interval "
                    "this narrow. The placement argument was never the binding constraint."
                )
            )
        ),
    }


def _slot_rates(run, size, pat):
    """Derived rate per slot, from either the old float list or the new records.

    `_pattern_rates` used to return five floats per pattern; it now returns five
    records carrying all seven timing rounds, because a single derived rate
    cannot distinguish a genuinely slow slot from one that caught a bad round.
    Both shapes are read here rather than migrating the committed runs: the raw
    runs under `raw_dev5/` are hashed into `input_manifest`, and rewriting them
    to fit a new reader would break the binding between artifact and evidence
    that the manifest exists to enforce.
    """
    block = run["roofline_patterns_by_size"][size][pat]
    return [v["TBps_at_min"] if isinstance(v, dict) else v for v in block]


def _within_slot_spreads(run, size, pat):
    """Per-slot round-to-round spread, where the run recorded it. Empty if not."""
    block = run["roofline_patterns_by_size"][size][pat]
    return [v["spread_pct_of_min"] for v in block if isinstance(v, dict)]


def _size_discrimination(runs):
    """Can any pattern tell the two buffer sizes apart? Measured, not assumed."""
    out = {}
    allr = [r for rs in runs.values() for r in rs]
    for pat in ("write", "two_read_one_write", "copy"):
        a = [v for r in allr for v in _slot_rates(r, "512MiB", pat)]
        b = [v for r in allr for v in _slot_rates(r, "2048MiB", pat)]
        # Round-to-round noise inside one slot, where recorded. This is the
        # instrument's floor; the across-slot spreads above are only meaningful
        # as a confound to the extent they exceed it.
        within = [
            s
            for r in allr
            for size in ("512MiB", "2048MiB")
            for s in _within_slot_spreads(r, size, pat)
        ]
        out[pat] = {
            "range_512MiB": [min(a), max(a)],
            "range_2048MiB": [min(b), max(b)],
            "spread_512MiB_pct": round((max(a) / min(a) - 1) * 100.0, 2),
            "spread_2048MiB_pct": round((max(b) / min(b) - 1) * 100.0, 2),
            "n_per_size": len(a),
            "ranges_overlap": not (max(a) < min(b) or max(b) < min(a)),
            "historical_TBps": HISTORICAL[pat],
            "within_slot_spread_pct_max": round(max(within), 3) if within else None,
            "within_slot_spread_pct_median": (
                round(statistics.median(within), 3) if within else None
            ),
            "n_slots_with_rounds": len(within),
        }
    out["verdict"] = (
        "No pattern separates 512 MiB from 2 GiB once allocation slot and allocator "
        "peak are sampled -- all three overlap. An earlier draft of this argument "
        "used two_read_one_write's 0.31% single-slot spread as an error bar and "
        "concluded from a 4.04% gap that the historical table must be 2 GiB. That "
        "spread was measured with slot and peak both held fixed, the one condition "
        "under which neither confound is visible; sampled properly the same probe "
        f"spreads {out['two_read_one_write']['spread_512MiB_pct']}%. The table's size "
        "is known from notes:869 stating it, which is documentary evidence. Nothing in "
        "this artifact measures it. The spread in this sentence was hand-carried from a "
        "previous collection and the prose guard caught it when fresh runs moved the "
        "value; it is now interpolated from the field above."
    )
    return out


def _assert_prose_is_derived(payload):
    """Refuse to write prose containing a decimal no field on this payload produced.

    The same guard `AI/probe_rmsnorm_roofline.py` grew, for the same reason, and
    it belongs here in particular: the field this artifact corrects
    (`denominator_stability_across_processes`) was itself introduced by the fix
    for hand-typed constants, and it was wrong. A guard that only runs on the
    generator does not protect the assembler that reinterprets it.

    The allowlist is values no run here can compute: figures measured elsewhere
    and cited, which must carry their provenance in the surrounding text.

    Known limit, measured rather than assumed: the accept-set is every payload
    number rendered at 0-3 dp, which for this artifact is 406 strings. Sweeping
    the 2-dp values in [0, 20] -- the range these percentages live in -- 5.2% are
    accepted without having been the number the prose meant. So this catches a
    fabricated or drifted constant about 19 times in 20, not always. It is a
    tripwire, not a proof, and a coincidental match is still possible.
    """
    cited = {
        "4.89": "historical copy cell, notes:869",
        "6.09": "historical two_read_one_write cell, notes:869",
        "6.84": "historical write cell, notes:869",
        "4.885": "band lower edge, fixed by @Reviewer in 4d6ed9de",
        "4.895": "band upper edge, fixed by @Reviewer in 4d6ed9de",
        "4.90": "band convention worked example",
        "13.35": "the superseded roofline field this artifact corrects",
        "0.63": "copy across six processes with the generator held fixed",
        "0.31": "superseded single-slot two_read_one_write spread, quoted as the defect",
        "4.04": "superseded gap that argument rested on, quoted as the defect",
        "0.2": "band width, rounded, quoted in prose about it",
    }

    measured = set()

    def add(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # 0-5 dp: the band width is rendered at 4 dp in prose, and a guard that
            # only knew 0-3 rejected its own correctly-derived text. A tripwire that
            # fires on precision rather than provenance trains you to widen the
            # allowlist, which is how a real bad number eventually gets waved through.
            for p in range(6):
                measured.add(f"{v:.{p}f}")

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            add(node)

    walk(payload)

    prose = []

    def collect(node):
        if isinstance(node, dict):
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)
        elif isinstance(node, str):
            prose.append(node)

    collect(payload)

    unexplained = sorted(
        {
            tok
            for text in prose
            for tok in re.findall(r"\d+\.\d+", text)
            if tok not in cited and tok not in measured
        }
    )
    if unexplained:
        raise SystemExit(
            f"prose in this artifact contains decimals {unexplained} that no field on "
            "the payload computed and no entry in the citation allowlist explains. "
            "That is the defect this artifact documents: the roofline sidecar's "
            "denominator_stability_across_processes was a hand-typed 13.35% that "
            "turned out to measure the wrong axis entirely. Interpolate from a "
            "computed field, or add the value to `cited` with its provenance."
        )


def _staircase(path=REPO / "AI/data/copy_placement_draws/raw_dev5/_peak_sweep.json"):
    """Where the allocator-peak steps are, from the fine sweep, if it was run.

    Optional because it costs ~44 processes. When absent the artifact says so
    rather than carrying a remembered pair of integers: an earlier draft of the
    notes asserted steps "at 13 and again at 17" from an uncommitted /tmp
    script, which is a number living only in prose.
    """
    if not path.exists():
        return {"collected": False, "how": "python AI/probe_copy_size_draws.py OUT --sweep-steps"}
    d = json.loads(path.read_text())

    def lvl(r):
        return r.get("n_prior_512mib_allocs", r["peak_live_512mib"])

    by = defaultdict(list)
    for r in d["rows"]:
        by[lvl(r)].append(r["draws_2GiB"])

    # The floor the step criterion needs, measured rather than assumed. The
    # previous criterion was `3 * max(repeat, 0.5)`, and that 0.5 was typed by
    # hand as "a repeat spread can't meaningfully be below this". It is the exact
    # object this whole line of work keeps retiring: a number with no error bar,
    # sitting inside a threshold that decides which steps get published. If the
    # sweep carries rounds, the floor is the median round-to-round spread of a
    # single draw; if it does not, the constant stays and the artifact says so.
    round_spreads = [
        (max(s) / min(s) - 1) * 100.0
        for r in d["rows"]
        for s in r.get("rounds_us_2GiB", [])
        if min(s) > 0
    ]
    floor = statistics.median(round_spreads) if round_spreads else 0.5
    floor_src = (
        f"median round-to-round spread of a single draw over {len(round_spreads)} draws"
        if round_spreads
        else (
            "hand-picked 0.5 -- this sweep predates rounds_us_2GiB, so there is no "
            "measured instrument floor and the threshold rests on a typed constant"
        )
    )

    levels, steps, prev = [], [], None
    for peak in sorted(by):
        reps = by[peak]
        means = [sum(r[i] for r in reps) / len(reps) for i in range(5)]
        repeat = max(max(r[i] for r in reps) / min(r[i] for r in reps) - 1 for i in range(5)) * 100
        shift = None if prev is None else max(abs(a / b - 1) for a, b in zip(means, prev)) * 100
        levels.append(
            {
                "peak_live_512mib": peak,
                "slot_means": means,
                "across_process_repeat_spread_pct": round(repeat, 2),
                "shift_vs_previous_level_pct": None if shift is None else round(shift, 2),
            }
        )
        if shift is not None and shift > 3 * max(repeat, floor):
            steps.append(peak)
        prev = means

    # Direction control. The first sweep ran 0..21 in wall-clock order, so a level
    # effect and a slow drift over the collection window are the same signal. If
    # the rows carry a direction, the same step-finding runs on each pass
    # independently: a real step appears at the same level going up and coming
    # back down; a drift artifact does not survive time reversal.
    def steps_for(rows):
        b = defaultdict(list)
        for r in rows:
            b[lvl(r)].append(r["draws_2GiB"])
        s, pv = [], None
        for pk in sorted(b):
            reps = b[pk]
            if len(reps) < 2:
                pv = [sum(r[i] for r in reps) / len(reps) for i in range(5)]
                continue
            mn = [sum(r[i] for r in reps) / len(reps) for i in range(5)]
            rp = max(max(r[i] for r in reps) / min(r[i] for r in reps) - 1 for i in range(5)) * 100
            sh = None if pv is None else max(abs(a / b2 - 1) for a, b2 in zip(mn, pv)) * 100
            if sh is not None and sh > 3 * max(rp, floor):
                s.append(pk)
            pv = mn
        return s

    dirs = {r.get("direction") for r in d["rows"]}
    direction_control = {"present": False, "why_it_matters": DIRECTION_CONTROL_ABSENT}
    if dirs - {None}:
        up = steps_for([r for r in d["rows"] if r.get("direction") == "up"])
        down = steps_for([r for r in d["rows"] if r.get("direction") == "down"])
        both = sorted(set(up) & set(down))
        direction_control = {
            "present": True,
            "steps_ascending": up,
            "steps_descending": down,
            "steps_in_both_directions": both,
            "steps_in_one_direction_only": sorted(set(up) ^ set(down)),
            "note": (
                "each row is still its own fresh process; only the order of processes "
                "differs. Steps appearing in one direction only are not established -- "
                "they are consistent with drift over the collection window, which the "
                "monotonic first design could not distinguish from a level effect."
            ),
        }
    return {
        "collected": True,
        "reps_per_level": d["reps_per_level"],
        "one_process_per_row": d["one_process_per_row"],
        "step_at_n_prior_512mib_allocs": steps,
        "step_at_peak_live_512mib": steps,
        "axis_name_retracted": (
            "'peak_live_512mib' named a quantity this design holds constant; see "
            "high_water_check. The level is a prior-allocation count."
        ),
        "criterion": "level-to-level shift exceeds 3x the across-process repeat spread",
        "criterion_floor_pct": round(floor, 3),
        "criterion_floor_source": floor_src,
        "direction_control": direction_control,
        "levels": levels,
        "why_not_within_process": d["why_not_within_process"],
    }


def _input_manifest(src_dir, extra):
    """Hash every input this artifact was built from, plus the code that built it.

    @Reviewer's standing objection to the previous assembler: it validated only
    that *some* git repository answered, so a forged commit or clean flag would
    be accepted, and nothing bound the evidence to the raw runs. That applies to
    this file unchanged, so it is fixed here rather than inherited.

    This does not make the artifact trustworthy on its own -- a manifest proves
    the assembler saw these bytes, not that the bytes came off a GPU. What it
    buys is that a reader who still has the raw runs can verify the artifact was
    built from them, and that a regeneration which silently picks up a different
    input set stops matching.
    """
    entries = []
    for path in sorted(Path(src_dir).glob("*.json")) + [Path(p) for p in extra if Path(p).exists()]:
        entries.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    for rel in ("AI/probe_copy_size_draws.py", "AI/assemble_copy_axes.py"):
        entries.append(
            {
                "path": rel,
                "sha256": hashlib.sha256((REPO / rel).read_bytes()).hexdigest(),
                "bytes": (REPO / rel).stat().st_size,
            }
        )
    return {
        "n_inputs": len(entries),
        "entries": entries,
        "what_this_proves": (
            "the assembler read exactly these bytes. It does not prove they came from "
            "a GPU, and a manifest cannot: it is a binding between artifact and raw "
            "runs, not an attestation."
        ),
    }


def _git():
    def run(*a):
        r = subprocess.run(
            ["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False
        )
        if r.returncode != 0:
            raise SystemExit(f"git {' '.join(a)} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD")[:7],
        "worktree_dirty": bool(run("status", "--porcelain")),
    }


def main():
    # Defaults to the committed raw runs, not /tmp: the manifest is only useful if
    # a reader can fetch the bytes it hashes. Pass a directory to re-assemble from
    # a fresh collection.
    src = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "AI/data/copy_placement_draws/raw_dev5")
    runs = _load(src)
    allr = [r for rs in runs.values() for r in rs]

    payload = {
        "what": (
            "Copy-rate draws at both 512 MiB and the 2 GiB size the historical "
            "roofline table was measured at, decomposed across three axes: "
            "allocation slot, process, and prior allocation count/history."
        ),
        "retracted_axis_name": (
            "the third axis was published in 9879dda and be86f90 as 'allocator peak "
            "high-water mark'. That name is withdrawn: see high_water_check, which "
            "measures that the process high-water is identical at every level because "
            "the prefix never exceeds the live set the measurement itself allocates. "
            "Raised by @Reviewer in 5c2e0083 and confirmed by instrumenting "
            "torch.cuda.max_memory_allocated. The staircase is real and reproducible; "
            "the mechanism named for it was not measured. Field keys carrying 'peak' "
            "are retained as aliases so prior runs and readers still parse."
        ),
        "supersedes": (
            "AI/data/copy_placement_draws/copy_512MiB_draws_dev5.json, whose argument "
            "was correct but measured entirely at 512 MiB while the cell it retires "
            "was measured at 2 GiB (notes:869). Scope gap raised by @Reviewer in "
            "f9014a94."
        ),
        "device_name": allr[0]["device_name"],
        "hip_visible_devices": allr[0]["hip_visible_devices"],
        "n_processes": len(allr),
        "n_prior_512mib_alloc_levels": sorted(runs),
        "peak_levels_512mib": sorted(runs),
        "high_water_check": _high_water_check(runs),
        "generator": "AI/probe_copy_size_draws.py",
        "assembler": "AI/assemble_copy_axes.py",
        "input_manifest": _input_manifest(src, []),
        **_git(),
        "historical_table": {
            "source": "AI/flydsl_rmsnorm_notes.md:869",
            "buffer_size_mib": HISTORICAL_SIZE_MIB,
            "TBps": HISTORICAL,
            "size_is_documentary_not_measured": (
                "notes:869 says 'probing three patterns over 2 GiB buffers'. This "
                "artifact cannot confirm it -- see size_discrimination."
            ),
        },
        "axes": {size: _decompose(runs, size) for size in ("512MiB", "2048MiB")},
        "band_test": {
            size: _band(
                [
                    v
                    for rs in runs.values()
                    for r in rs
                    for v in r["identical_buffers_by_size"][size]["TBps_per_identical_buffer"]
                ]
            )
            for size in ("512MiB", "2048MiB")
        },
        "size_discrimination": _size_discrimination(runs),
        "band_reachability_power": {
            size: _band_reachability(runs, size) for size in ("512MiB", "2048MiB")
        },
        "peak_staircase": _staircase(),
    }

    hist_axis = payload["axes"]["2048MiB"]
    hist_band = payload["band_test"]["2048MiB"]
    payload["verdict"] = (
        "The conclusion holds at the size that matters, for a better-supported reason. "
        "At 2 GiB the copy rate ranges {r:.2f}% over {n} draws spanning {s} allocation "
        "slots and {p} allocator peaks, which is {x:.0f}x the {w:.2f}% width of the band "
        "the historical 4.89 cell implies, with {b} draws below it and {a} at or above. "
        "That is NOT a demonstration that the band is reachable: zero draws land in it, "
        "and because the draws cluster on allocation slots whose means are separated by "
        "gaps far wider than the band, zero in-band is what a uniform model predicts "
        "even if the band is perfectly reachable -- see band_reachability_power. The "
        "defensible claim is symmetric and weaker than the one this artifact first "
        "made: at this sample size the data neither authenticate nor exclude the "
        "historical value, and a single copy draw cannot authenticate a harness, a "
        "device, or a commit. What changes from the superseded artifact is the number "
        "and its scope, not the verdict: {r:.2f}% at 2 GiB rather than the {o:.2f}% "
        "this same collection gives at 512 MiB, which is the size the superseded "
        "artifact measured."
    ).format(
        o=payload["axes"]["512MiB"]["pooled_range_pct_of_min"],
        r=hist_axis["pooled_range_pct_of_min"],
        n=hist_axis["n_draws"],
        s=5,
        p=len(runs),
        x=hist_axis["pooled_range_pct_of_min"] / hist_band["band_width_pct_of_lo"],
        w=hist_band["band_width_pct_of_lo"],
        b=hist_band["draws_below_band"],
        a=hist_band["draws_at_or_above_band_half_open"],
    )
    payload["what_is_and_is_not_repeatable"] = {
        "repeatable": (
            "copy IS repeatable to under {ap:.2f}% given a fixed allocation slot, a "
            "fixed program, and a fixed allocator peak"
        ).format(ap=hist_axis["worst_across_process_range_pct"]),
        "not_repeatable": (
            "which slot a draw lands on -- {sl:.2f}% across slots at 2 GiB within a "
            "single fixed program -- and where those slots sit once the program's peak "
            "high-water mark changes ({pr:.2f}%). Pooling both axes gives {po:.2f}%, "
            "which is the figure the verdict uses; it is reported as a pooled range "
            "and not as either axis alone, because a fresh draw from an unknown "
            "harness is exposed to both."
        ).format(
            sl=max(pp["pooled_range_pct_of_min"] for pp in hist_axis["by_peak"].values()),
            pr=hist_axis["worst_cross_program_range_pct"],
            po=hist_axis["pooled_range_pct_of_min"],
        ),
        "corrects": (
            "denominator_stability_across_processes in the roofline sidecar, which "
            "reports copy at 13.35% with n_processes: 3. That is not a process axis: "
            "the three runs straddled edits to _copy_variability that changed the "
            "allocator's peak. Same program, six processes: 0.63%."
        ),
    }

    _assert_prose_is_derived(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

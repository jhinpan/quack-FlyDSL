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
import itertools
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


def _factorial(runs, size):
    """Split the sum of squares into prefix, ordinal, interaction and within-cell.

    The artifact previously reported one number -- "99.63% of the total sum of
    squares is explained by which slot a draw came from" -- and that was a
    composite (prefix, ordinal) cell fit, not a slot effect. @Reviewer, 5c2e0083.
    A cell-fit near 100% is close to uninformative: with several replicates per
    cell and a stable instrument, almost any design produces it. It says the
    within-cell noise is small, which the instrument floor already says better.

    The honest split is below. The important entry is not the largest one -- it is
    `ordinal_is_confounded`: allocation ordinal, timing order and address are the
    same index in this design, because slot i is always allocated i-th and always
    measured i-th. Nothing here separates them, so the ordinal share cannot be
    attributed to placement. Randomizing measurement order against allocation
    order is the discriminating experiment and has not been run.
    """
    cells = defaultdict(list)
    for peak, rs in runs.items():
        for r in rs:
            for i, v in enumerate(
                r["identical_buffers_by_size"][size]["TBps_per_identical_buffer"]
            ):
                cells[(peak, i)].append(v)
    allv = [v for vs in cells.values() for v in vs]
    gm = sum(allv) / len(allv)
    sst = sum((v - gm) ** 2 for v in allv)

    pk_m, or_m = defaultdict(list), defaultdict(list)
    for (peak, i), vs in cells.items():
        pk_m[peak] += vs
        or_m[i] += vs

    def ss(groups):
        return sum(len(vs) * ((sum(vs) / len(vs)) - gm) ** 2 for vs in groups)

    ss_pk, ss_or = ss(pk_m.values()), ss(or_m.values())
    ss_cell = ss(cells.values())
    return {
        "prefix_pct": round(ss_pk / sst * 100.0, 2),
        "ordinal_pct": round(ss_or / sst * 100.0, 2),
        "interaction_pct": round((ss_cell - ss_pk - ss_or) / sst * 100.0, 2),
        "within_cell_pct": round((sst - ss_cell) / sst * 100.0, 2),
        "n_cells": len(cells),
        "replicates_per_cell": sorted({len(vs) for vs in cells.values()}),
        "ordinal_is_confounded": (
            "allocation ordinal, timing order and address are one index here: slot i "
            "is always allocated i-th and always measured i-th. The ordinal share "
            "cannot be read as a placement effect. No addresses are recorded and "
            "measurement order is not randomized against allocation order."
        ),
        "cell_fit_is_not_evidence": (
            "prefix + ordinal + interaction sums to ~99.5% by construction, because "
            "within-cell noise is small. That is a statement about instrument "
            "stability, not about the size of any effect."
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
    within = [(max(s) / min(s) - 1) * 100.0 for per_run in rounds for s in per_run if min(s) > 0]
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
        # Composite (prefix, ordinal) cell fit. Kept because prior text quoted it,
        # but it is NOT "variance explained by slot": it pools the prefix main
        # effect, the ordinal main effect and their interaction into one number,
        # and a near-100% value is unsurprising for any design with several
        # replicates per cell. @Reviewer, 5c2e0083. Read `variance_decomposition`.
        "variance_explained_by_cell_pct": round(ssb / sst * 100.0, 2),
        "variance_explained_by_slot_and_peak_pct": round(ssb / sst * 100.0, 2),
        "variance_decomposition": _factorial(runs, size),
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


def _estimator_repeatability(runs, size):
    """How far does the PUBLISHED estimator move when you repeat it?

    The published figure per draw is min-of-seven-rounds. Its uncertainty is not
    the range of those seven rounds -- that is the spread of the sample a min is
    taken over, and it is systematically wider than the min's own repeatability.
    Using it was @Reviewer's objection in 5c2e0083 and he is right.

    The measurable substitute is already here: each (prefix, ordinal) cell was
    collected in four separate processes, so the same estimator ran four times at
    the same relative placement.

    Reported as the observed RANGE of those four minima, which is distribution-
    free: it says what these four processes did and assumes nothing. An earlier
    version reported 2*sd of the RSD instead. That is a parametric half-width on
    n=4, and worse, it collapsed 20 heterogeneous cells to one median and then
    thresholded it -- which at 512 MiB produced a clean "resolvable: yes" from
    cells whose ranges run 0.038% to 3.76%, a hundredfold spread. @Autotune
    reached the same correction independently from the quartiles. The per-cell
    count is what carries the claim now; a single verdict flag does not.

    The comparison is done in ABSOLUTE TB/s. A second correction from @Reviewer
    (a7fe31c8), and the reason matters more than the one cell it moves. The band
    is an absolute interval, 0.010 TB/s wide, fixed on the display axis at
    [4.885, 4.895). An earlier version compared each cell's range *as a percent
    of that cell's own mean* against the band's width *as a percent of 4.885*.
    Those are two different denominators, so the effective threshold scaled with
    the cell: a cell running at 5.336 TB/s was implicitly allowed 0.0109 TB/s of
    movement rather than 0.010. It passed on 0.0104. The published count went
    10/20 instead of 9/20 -- a small error carrying a large one, because the
    quantity being tested was no longer the quantity the band is defined in. The
    relative range is still reported, as a description of each cell; it is not
    what the threshold is applied to.
    """
    lo, hi = BAND
    band_abs = hi - lo
    cells = defaultdict(list)
    for prefix, rs in runs.items():
        for r in rs:
            for ordinal, v in enumerate(
                r["identical_buffers_by_size"][size]["TBps_per_identical_buffer"]
            ):
                cells[(prefix, ordinal)].append(v)
    reps = {k: v for k, v in cells.items() if len(v) > 1}
    if not reps:
        return {
            "n_cells_with_repeats": 0,
            "note": "no cell has repeated batches; estimator uncertainty is not measurable here",
        }
    ranges = sorted(max(v) - min(v) for v in reps.values())
    rel = sorted((max(v) / min(v) - 1) * 100.0 for v in reps.values())
    q = statistics.quantiles(ranges, n=4) if len(ranges) >= 4 else [ranges[0]] * 3
    qr = statistics.quantiles(rel, n=4) if len(rel) >= 4 else [rel[0]] * 3
    inside = sum(1 for r in ranges if r < band_abs)
    return {
        "estimator": "min of seven timing rounds, as published",
        "statistic": (
            "observed range across repeats of the same cell, distribution-free, "
            "in absolute TB/s -- the units the band is defined in"
        ),
        "n_cells_with_repeats": len(ranges),
        "replicates_per_cell": sorted({len(v) for v in reps.values()}),
        "band_width_TBps": round(band_abs, 6),
        "range_TBps_q1_median_q3": [round(x, 6) for x in q],
        "range_in_band_widths_q1_median_q3": [round(x / band_abs, 2) for x in q],
        "range_TBps_min_max": [round(ranges[0], 6), round(ranges[-1], 6)],
        "cells_whose_range_fits_inside_band": inside,
        "verdict": (
            f"{inside} of {len(ranges)} cells repeat to within the band's "
            f"{band_abs:.3f} TB/s. "
            + (
                "The cells are heterogeneous -- the interquartile range crosses the band "
                "in both directions -- so neither 'resolves' nor 'does not resolve' is "
                "true of this size as a whole. No single flag is published for it."
                if q[0] < band_abs < q[2]
                else "The quartiles fall entirely on one side of the band, so the size "
                "has a consistent verdict: "
                + (
                    "the estimator repeats to within the band at this size."
                    if q[2] < band_abs
                    else "the estimator does not repeat to within the band at this size."
                )
            )
        ),
        # Descriptive only. Kept because a cell's range relative to its own level
        # is the natural way to read how noisy that cell is; it is deliberately
        # NOT compared against a relative band width, which is the defect above.
        "relative_range_pct_q1_median_q3": [round(x, 4) for x in qr],
        "relative_range_is_descriptive_not_thresholded": (
            "percent-of-own-mean cannot be tested against percent-of-4.885: cells "
            "sit between 4.74 and 5.60 TB/s, so the two denominators differ by up "
            "to 15% and the implied threshold would move with the cell"
        ),
        "what_is_held_fixed": (
            "allocation prefix and allocation ordinal, hence relative placement; the "
            "process, its address-space layout and the clock state are not"
        ),
        "what_is_not_claimed": (
            "no interval is constructed. A t or z half-width on n=4 would assume iid "
            "normality within a cell, which four fresh processes on a shared machine do "
            "not establish, and at df=3 the verdict would be sensitive to it."
        ),
    }


def _band_reachability(runs, size):
    """Could this protocol have produced an in-band value at all? Power, not outcome.

    Zero draws in band is weak evidence of anything. The draws are not a
    continuum: they cluster on allocation slots, and the band is narrow against
    the spacing between them. @Autotune's argument, recomputed here for this
    dataset rather than transcribed.

    The uniform-slot model that computed a "30.4% chance any lands in band" is
    demoted to an illustration and is no longer the basis of the conclusion.
    @Reviewer's objection in 5c2e0083 is correct on both counts. It treats five
    fixed ordinals under four selected prefixes as 20 iid uniform draws, which
    they are not -- and the assumption is testable on this very payload, which is
    what makes keeping it indefensible rather than merely unproven. The observed
    gaps between adjacent slot means are strongly clustered: at 2 GiB, 7 of 19
    gaps are *narrower* than the band and the largest is only ~10x it, so the
    means are neither uniform nor uniformly far apart. A uniform model both
    overstates the chance of landing in a sparse region and understates it in a
    dense one.

    What replaces it is the empirical spacing itself, reported without a
    generative model: how many gaps are narrower than the band, how wide the
    largest is, and how the band width compares to a single draw's own noise.
    Those are measurements. The qualitative conclusion -- that zero in-band is
    uninformative here -- does not need the model and never did, which is the
    part I should have noticed before publishing a probability.
    """
    lo, hi = BAND
    axis = _decompose(runs, size)
    means = sorted({round(m, 6) for pk in axis["by_peak"].values() for m in pk["slot_means"]})
    band_w = (hi / lo - 1) * 100.0
    span = (max(means) / min(means) - 1) * 100.0
    p_one = band_w / span
    floor = axis["within_draw_instrument_floor"]
    fl = floor.get("round_spread_pct_median")
    gaps = sorted((means[i + 1] / means[i] - 1) * 100.0 for i in range(len(means) - 1))
    rep = _estimator_repeatability(runs, size)
    return {
        "n_slot_means": len(means),
        "slot_mean_span_pct": round(span, 2),
        "band_width_pct": round(band_w, 4),
        # Measured spacing, no generative model. This is what the retracted
        # uniform prior was standing in for.
        "adjacent_slot_mean_gaps_pct": [round(g, 3) for g in gaps],
        "n_gaps_narrower_than_band": sum(1 for g in gaps if g < band_w),
        "n_gaps": len(gaps),
        "largest_gap_pct": round(max(gaps), 3),
        "largest_gap_in_band_widths": round(max(gaps) / band_w, 1),
        "median_gap_in_band_widths": round(statistics.median(gaps) / band_w, 1),
        "slot_means_are_clustered_not_uniform": (
            "the gaps span two orders of magnitude and a third of them are narrower "
            "than the band itself, so the uniform-draw model below does not describe "
            "these means. It is retained as a labelled illustration only."
        ),
        "uniform_model_ILLUSTRATIVE_NOT_MEASURED_POWER": {
            "p_single_slot_in_band": round(p_one, 4),
            "expected_slot_means_in_band": round(len(means) * p_one, 3),
            "p_at_least_one_in_band_pct": round((1 - (1 - p_one) ** len(means)) * 100.0, 1),
            "why_not_load_bearing": (
                "treats five fixed ordinals under four selected prefixes as iid uniform "
                "draws over the span, and divides relative widths taken about different "
                "denominators. @Reviewer, 5c2e0083. The conclusion below does not use it."
            ),
        },
        # The second reason this protocol has no power, and the stronger one. The
        # argument above is about placement scattering draws past a narrow window.
        # This is about whether a single draw is repeatable to the window's width
        # in the first place.
        #
        # An earlier version answered that with the range of the seven timing
        # rounds behind one draw, and that is the wrong quantity: the published
        # figure is the MIN of those seven, and a min is far more repeatable than
        # the range of the sample it is drawn from. @Reviewer, 5c2e0083. The
        # correct calibration is batch-to-batch: repeat the whole seven-round
        # estimator at a fixed (prefix, ordinal) cell and look at how the MIN
        # moves. This tree already has those repeats -- four per cell -- and they
        # were sitting unused while a substitute stood in for them.
        "band_width_vs_single_draw_noise": rep,
        "why_the_round_range_is_not_this": (
            "the seven-round range (median "
            f"{fl}% at this size) describes the sample the estimator minimises over, "
            "not the estimator. It is reported in `within_draw_instrument_floor` and "
            "is NOT used for the resolvability flag above."
            if fl is not None
            else "these runs predate retained rounds."
        ),
        "conclusion": (
            f"the {len(means)} slot means span {span:.2f}% and the band is {band_w:.4f}% wide. "
            f"Their spacing is clustered rather than even: {sum(1 for g in gaps if g < band_w)} of "
            f"{len(gaps)} adjacent gaps are narrower than the band and the largest is "
            f"{round(max(gaps) / band_w, 1)}x it. Zero draws in band is therefore not evidence "
            "against the historical value -- with means bunched this way, a narrow window "
            "can sit in a sparse stretch and be missed by every draw. No probability is "
            "claimed for that; an earlier version asserted one from a uniform model that "
            "this same spacing refutes. The defensible statement is symmetric: at this "
            "size and sample size the data neither authenticate nor exclude a value in "
            "the band."
            + (
                ""
                if not rep.get("n_cells_with_repeats")
                else (
                    " On whether one draw could resolve the band at all: repeating the "
                    "published min-of-seven estimator at a fixed (prefix, ordinal) cell "
                    f"across {rep['replicates_per_cell'][0]} processes gives observed "
                    f"ranges with quartiles {rep['range_in_band_widths_q1_median_q3'][0]}x / "
                    f"{rep['range_in_band_widths_q1_median_q3'][1]}x / "
                    f"{rep['range_in_band_widths_q1_median_q3'][2]}x the band. "
                    + rep["verdict"]
                    + " Three earlier versions of this sentence were wrong in different "
                    "ways: the first used the seven-round range, which is a property of "
                    "the sample the min is taken over rather than of the min; the second "
                    "used 2sd of the per-cell RSD, which is a parametric half-width on "
                    "n=4 and collapsed heterogeneous cells to one flag; the third "
                    "compared each cell's range as a percent of its own mean against the "
                    "band's width as a percent of 4.885, two different denominators, "
                    "which widened the threshold for every cell above the band."
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


# Above this, the guard is weak enough that it should not be described as a
# check. Not a physical threshold -- a line drawn so that degradation announces
# itself instead of accumulating silently. It has moved 7.996 -> 8.546 -> 8.296
# -> 9.445 across four regenerations, entirely from the artifact growing, and
# every one of those was an improvement to the artifact. That is the trap: the
# tripwire decays as a side effect of good changes, monotonically, and reporting
# the rate in a field nobody reads is not the same as noticing.
GUARD_FALSE_NEGATIVE_CEILING_PCT = 12.0


def _guard_false_negative_rate(measured, cited, dp=2, hi=20):
    """How often a value the prose did not mean would still be accepted.

    Computed on the payload actually being written, because it is a property of
    that payload and not of the guard. The docstring version of this number was
    stale within one commit of being written -- see `_assert_prose_is_derived`.

    Computed over the FULL accept-set. @Reviewer, a7fe31c8: an earlier version
    took only `measured` and ignored `cited`, while the accept rule this measures
    is `tok in cited or tok in measured`. So the guard's own self-assessment
    excluded part of its own accept-set -- and specifically the hand-maintained
    part, the one that grows when somebody wants a number to pass. A permissive-
    ness metric blind to exactly the mechanism by which permissiveness is
    granted. Five allowlist strings fall on the 2-dp grid without being covered
    by any computed field, so the reported rate was low by 0.25 pp. Small; the
    direction is what matters, since every future allowlist entry would have been
    free.
    """
    grid = [f"{i / 10**dp:.{dp}f}" for i in range(hi * 10**dp + 1)]
    accept = set(measured) | set(cited)
    hits = sum(1 for g in grid if g in accept)
    from_cited = sum(1 for g in grid if g in cited and g not in measured)
    rate = hits / len(grid) * 100.0
    if rate > GUARD_FALSE_NEGATIVE_CEILING_PCT:
        raise SystemExit(
            f"the prose guard's own false-negative rate is now {rate:.3f}%, above the "
            f"{GUARD_FALSE_NEGATIVE_CEILING_PCT}% ceiling. The accept-set has grown "
            f"to {len(accept)} strings and covers {hits} of {len(grid)} plausible "
            f"{dp}-dp values, so 'this decimal matches a computed field' no longer "
            "carries much information. Narrow the accept-set (fewer rendered "
            "precisions), split the payload, or check prose against the specific "
            "field it cites rather than against every number in the artifact. Do not "
            "raise the ceiling to make this pass -- that is the failure mode this "
            "check exists to interrupt."
        )
    return {
        "accept_set_size": len(accept),
        "accept_set_measured": len(set(measured)),
        "accept_set_cited": len(set(cited)),
        "grid": f"{dp}-dp values in [0, {hi}]",
        "grid_size": len(grid),
        "accepted_without_being_meant": hits,
        "of_which_only_the_citation_allowlist_explains": from_cited,
        "false_negative_rate_pct": round(rate, 3),
        "ceiling_pct": GUARD_FALSE_NEGATIVE_CEILING_PCT,
        "headroom_pct": round(GUARD_FALSE_NEGATIVE_CEILING_PCT - rate, 3),
        "note": (
            "a tripwire, not a proof. This rate rises as the artifact grows -- more "
            "measured values cover more of the grid by accident -- so it is computed "
            "at write time rather than quoted from a docstring, and the assembler "
            "now refuses to write once it passes the ceiling. Every increase so far "
            "came from an improvement to the artifact, which is why silent decay was "
            "the likely outcome without a hard stop. It is computed over the citation "
            "allowlist as well as the measured values, because both are accepted; "
            "counting only the measured half understated the guard's permissiveness "
            "in the one direction a reviewer cannot see from the outside."
        ),
    }


def _assert_prose_is_derived(payload):
    """Refuse to write prose containing a decimal no field on this payload produced.

    The same guard `AI/probe_rmsnorm_roofline.py` grew, for the same reason, and
    it belongs here in particular: the field this artifact corrects
    (`denominator_stability_across_processes`) was itself introduced by the fix
    for hand-typed constants, and it was wrong. A guard that only runs on the
    generator does not protect the assembler that reinterprets it.

    The allowlist is values no run here can compute: figures measured elsewhere
    and cited, which must carry their provenance in the surrounding text.

    Known limit, and `_guard_false_negative_rate` below now computes it on the
    payload being written rather than leaving it in this docstring. That change
    is the point of the entry, so the reasoning is here rather than in a commit
    message.

    This docstring previously said "406 strings ... 5.2% are accepted", and
    @Reviewer recomputed 8.246% against a later payload. He is right, and the
    stale figure is a clean instance of the defect this whole file is about: a
    number that appears only inside a prose string has no error bar and never
    re-runs (@Autotune's rule). It was measured once on a smaller payload and
    then quoted as a property of the guard.

    What actually moved it is not what I would have guessed. Recomputing across
    the artifact's own history: 7.996% at 9899e9d, 8.546% at be86f90, 8.296% at
    682e103. The rate rises because *the payload grows* -- more measured values
    means more accidental coverage of the grid -- not because the accept rule
    loosened. Widening the rule from 0-3 dp to 0-5 dp, which I flagged in the
    code below as the risky change, costs exactly 0.0000 pp on a 2-dp grid and
    only shows up at 4 dp (+0.21 pp). So the thing I wrote a comment to worry
    about was harmless, and the thing that actually degraded the guard -- routine
    growth of the artifact -- had no comment at all.

    The consequence is that this rate is a moving property of each payload, and
    that is why it is computed at write time now. It remains a tripwire, not a
    proof: a coincidental match is possible, and on the current payload roughly
    one 2-dp value in ten would pass unexplained.

    `cited` is part of the accept-set and is now counted as such -- see
    `_guard_false_negative_rate`. Adding an entry here is not free.
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
    return _guard_false_negative_rate(measured, cited)


def _updown_identity(d, lvl):
    """Is level a deterministic function of collection position, symmetrically read?

    Worth measuring rather than asserting: if it is, the up/down confound is an
    identity and not a tendency, and no statistic computed on those passes can
    separate a level effect from a midpoint-symmetric function of time. If a
    future collection breaks it, that shows up here instead of silently
    weakening the caveat this field exists to state.

    Tested as functional dependence, not as a formula. The first version
    hard-coded `level == min(seq, N-1-seq)`, which is the identity only when the
    passes hold exactly one row per level. Adding a second repeat per level
    doubled the position axis and that formula went to 86 exceptions out of 88 --
    while the confound itself was untouched, since level is still constant within
    every distance-from-nearest-end class. The failure direction is the part
    worth keeping: a guard against overclaiming reported that the confound was
    *weaker* than stated, i.e. it failed toward flattering the claim it exists to
    restrain. A check that can only be trusted while the collection layout stays
    fixed is a check that will mislead exactly when the layout changes. So the
    property is tested directly: group the positions by distance from the nearest
    end and ask whether any group holds more than one level.
    """
    seen = {}
    for r in d["rows"]:
        if r.get("direction") in ("up", "down"):
            seen.setdefault(r["seq"], lvl(r))
    if not seen:
        return {"checked": False}
    order = sorted(seen)
    n = len(order)
    classes = defaultdict(set)
    for i, s in enumerate(order):
        classes[min(i, n - 1 - i)].add(seen[s])
    ambiguous = {k: sorted(v) for k, v in classes.items() if len(v) > 1}
    holds = not ambiguous
    return {
        "checked": True,
        "n_positions": n,
        "property": (
            "level is constant within every class of positions equidistant from the "
            "nearest end of the up/down sequence, i.e. level is a function of "
            "min(position, N-1-position)"
        ),
        "n_symmetric_classes": len(classes),
        "classes_holding_more_than_one_level": ambiguous,
        "holds_exactly": holds,
        "consequence": (
            "level is a deterministic function of collection position across the up and "
            "down passes, so those passes cannot distinguish a level effect from any "
            "midpoint-symmetric function of time -- including a single transient at the "
            "turnaround. Only the interleaved pass can."
            if holds
            else "level is not fully determined by symmetric position here, so the up/down "
            "passes carry some independent information about level. How much is not "
            "quantified, and no claim rests on it."
        ),
    }


def _interleaved_control(d, steps_for, lvl):
    """The pass that actually breaks level-vs-collection-time.

    Ascending-then-descending is not enough. With reps adjacent it makes level an
    exact symmetric function of position, `level == min(seq, N-1-seq)`, so ANY
    unimodal transient centred on the turnaround reproduces the staircase in both
    directions at once. @Reviewer, 5c2e0083, and the objection is structural: no
    amount of agreement between the two passes rules it out, because the two
    passes are mirror images of the same confound.

    The interleaved pass visits every (level, repeat) in one seeded shuffle. What
    is reported here is not "the steps agree" -- it is the measured association
    between level and position in that pass. If the shuffle worked, level is
    near-uncorrelated with position under both a monotone reading and a symmetric
    one, and any step found cannot be a function of when the row was collected.
    """
    rows = [r for r in d["rows"] if r.get("direction") == "interleaved"]
    if not rows:
        return {
            "present": False,
            "why_it_matters": (
                "without it, level is a symmetric function of collection position and a "
                "midpoint transient satisfies the up/down test. The step locations are "
                "reported as surviving time reversal only, which is weaker."
            ),
            "how": "python AI/probe_copy_size_draws.py OUT --sweep-steps",
        }
    rows = sorted(rows, key=lambda r: r["seq"])
    pos = list(range(len(rows)))
    lv = [lvl(r) for r in rows]
    mid = [min(p, len(pos) - 1 - p) for p in pos]
    steps = steps_for(rows)
    return {
        "present": True,
        "n_rows": len(rows),
        "shuffle_seed": d.get("shuffle_seed"),
        "steps_interleaved": steps,
        "corr_level_vs_position": round(statistics.correlation(pos, lv), 4),
        "corr_level_vs_distance_from_midpoint": round(statistics.correlation(mid, lv), 4),
        "max_consecutive_same_level": max(len(list(g)) for _, g in itertools.groupby(lv)),
        "what_this_establishes": (
            "level is not recoverable from collection position in this pass, under "
            "either a monotone or a midpoint-symmetric reading, so a step found here "
            "is not a function of when its row was collected. This is the control the "
            "up/down pass could not provide."
        ),
        "what_it_still_does_not_establish": (
            "why the level matters. The mechanism behind a step is not addressed by any "
            "ordering control."
        ),
    }


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
            "what_this_does_NOT_establish": (
                "up-then-down leaves level a symmetric function of collection position, "
                "so a transient centred on the turnaround satisfies both passes at once. "
                "@Reviewer, 5c2e0083. Agreement between the two directions is therefore "
                "necessary and not sufficient; see interleaved_control."
            ),
            "confound_is_exact_not_approximate": _updown_identity(d, lvl),
            "note": (
                "each row is still its own fresh process; only the order of processes "
                "differs. Steps appearing in one direction only are not established -- "
                "they are consistent with drift over the collection window, which the "
                "monotonic first design could not distinguish from a level effect."
            ),
        }
    direction_control["interleaved_control"] = _interleaved_control(d, steps_for, lvl)
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

    Paths are recorded RELATIVE to the repo when the input is inside it. They
    were absolute, which quietly cost the manifest the property it exists for: a
    reader regenerating from their own clone gets a byte-different artifact even
    though every hash matches, so "did this regenerate cleanly?" could not be
    answered by comparing files. Found by regenerating in a throwaway clone
    before handing the commit to @Reviewer. The tell was in this same function --
    the two code entries below were already relative while the data entries above
    them were not, in a manifest whose whole job is to be checkable by somebody
    else.
    """
    entries = []
    for path in sorted(Path(src_dir).glob("*.json")) + [Path(p) for p in extra if Path(p).exists()]:
        try:
            name = str(path.resolve().relative_to(REPO))
        except ValueError:
            name = str(path.resolve())
        entries.append(
            {
                "path": name,
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

    fn = _assert_prose_is_derived(payload)
    # Added after the guard runs, so the rate describes the payload the guard
    # actually checked. Writing it in first would make the guard's own accept-set
    # include the digits of its own error rate -- a small self-reference, but this
    # file has been bitten twice by numbers that described a slightly different
    # object than the one they were attached to.
    fn["measured_on"] = "the payload as checked, before this field was added"
    payload["prose_guard"] = fn
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Assemble the count-vs-bytes factorial into one artifact.

Written BEFORE the sweep's numbers were read, and committed in that order on
purpose. Every threshold, unit of analysis, and read-off rule below is fixed
here; the only thing the data decides is which pre-registered branch is taken.
The staircase work needed three iterations to remove a confound that survived
because each round's analysis was chosen after its numbers were on screen.

Run: python AI/assemble_alloc_factorial.py AI/artifacts/alloc_factorial.json OUT.json
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _stats():
    from AI.assemble_order_confound import _additive_fit, _f_sf

    return _additive_fit, _f_sf


_additive_fit, _f_sf = _stats()

# The absolute band @Reviewer fixed for the historical 4.89 cell, in TB/s. Used
# as a width, never as a percentage: a cell at 5.34 TB/s tested against 0.2047%
# of its OWN mean is granted 15% more headroom than one at 4.89, and that is how
# 9/20 was once reported as 10/20. Absolute width, one denominator.
BAND = (4.885, 4.895)

# Fixed so the permutation p is a property of the artifact, not of the run.
PERM_SEED = 20260804
PERM_DRAWS = 20000

# The staircase's measured levels either side of its steps, from the committed
# interleaved artifact. Only a reference point for the anchor check below, not
# an input to any test here -- different processes, different session.
STAIRCASE_LOW_TBPS = 4.917
STAIRCASE_HIGH_TBPS = 4.977


def _cell_means(rows):
    """Per-process mean over the five slots. THE unit of analysis.

    Pre-registered as primary, and the reason is structural rather than
    statistical: the five slot rates in a row come from one process, one
    allocator state, one `_identical_buffer_spread` call. They are repeated
    measures, not replicates. Treating all 80 as independent would quadruple the
    apparent degrees of freedom and shrink every p-value by a factor that has
    nothing to do with the treatment.

    The slot-level fit blocked on process is reported alongside, because it uses
    the within-process variance the means throw away -- but if the two disagree
    the per-process figure is the one that stands, and saying which wins before
    seeing either is the entire point of fixing it here.
    """
    return [
        {
            "cell": r["cell"],
            "count": r["count"],
            "buffer_mib": r["buffer_mib"],
            "rep": r["rep"],
            "seq": r["seq"],
            "TBps_at_min": statistics.fmean(r["TBps_per_slot"]),
            "prefix_high_water_GiB": r["prefix_high_water_GiB"],
        }
        for r in rows
    ]


def _preconditions(rows, live_gib):
    """The design's load-bearing facts, asserted from the data rather than the docstring.

    `lo_hi` and `hi_lo` separate count from bytes ONLY if they really reached the
    same prior peak. That is an empirical claim about the allocator, and the
    module docstring stating it is not evidence. If these two disagree the
    factorial does not mean what it says, and this field is where a reader finds
    that out without re-deriving it.
    """
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r["prefix_high_water_GiB"])
    peaks = {c: sorted(set(v)) for c, v in by_cell.items()}
    diag = ("lo_hi", "hi_lo")
    diag_peaks = {c: peaks.get(c, []) for c in diag}
    matched = (
        all(len(v) == 1 for v in diag_peaks.values())
        and len({v[0] for v in diag_peaks.values()}) == 1
    )
    above = {c: all(x > live_gib for x in v) for c, v in peaks.items()}
    return {
        "prefix_high_water_GiB_by_cell": peaks,
        "diagonal_pair": list(diag),
        "diagonal_pair_reached_the_same_peak": matched,
        "diagonal_pair_peak_GiB": diag_peaks[diag[0]][0] if matched else None,
        "every_prefix_above_measurement_live_set": all(above.values()),
        "which_cells_are_above": above,
        "the_one_that_is_not": (
            "lo_lo reaches exactly 12.0 GiB, which EQUALS the measurement's own live "
            "set rather than exceeding it, so `every_prefix_above...` is false. This is "
            "a boundary case I built in without noticing: 24 x 512 MiB is 12 GiB on the "
            "nose. lo_lo is therefore the one cell that does not clear the design's own "
            "stated bar, and it is also the cell furthest from the others (4.9077 vs "
            "~4.95-4.97). The three cells that carry the primary comparison -- lo_hi, "
            "hi_lo at 24 GiB and hi_hi at 48 -- are all strictly above it, so the "
            "diagonal test is unaffected. The anchor role lo_lo was given is what "
            "suffers: it was meant to show the grid is not flat, and it does, but from "
            "a point that sits at the boundary rather than clear of it."
        ),
        "measurement_live_set_GiB": live_gib,
        "why_this_gates_everything": (
            "count and bytes are separated by lo_hi vs hi_lo reaching one peak by two "
            "routes. If that is false the two cells differ in both factors at once and "
            "the design has reproduced the staircase's confound at larger numbers."
        ),
        "and_this_one": (
            "the staircase's prefixes all sat BELOW the 12 GiB the measurement itself "
            "allocates, so its process high-water was constant at 12 GiB across every "
            "condition -- peak bytes never varied in the treatment that named itself "
            "after peak bytes. Here every prefix exceeds it, so the axis is real."
        ),
    }


def _factorial(means):
    """2x2 saturated decomposition plus Type-II tests for each main effect.

    Both are reported because they answer different questions and the staircase
    artifact once published one as the other: a saturated cell term (98.4%) read
    as a main effect (0.04%) differ by three orders of magnitude and by their
    entire meaning.

    READ THE FACTOR NAMES CAREFULLY -- I got them wrong here on the first pass,
    and the failure is the same one this probe was built to fix.

    `bytes_f` is `buffer_mib`: the size of EACH prior buffer, 512 MiB or 1 GiB.
    It is NOT total prior bytes, which is `count * buffer_mib`. So the row
    labelled `count_given_bytes` compares 24 vs 48 allocations at a fixed
    PER-BUFFER size -- and doubling the count at fixed per-buffer size doubles
    the total too, 12->24 GiB and 24->48 GiB. Neither Type-II main effect holds
    total prior bytes constant. Both are confounded with it, exactly as the
    staircase's single axis was, and I nearly published two significant
    p-values (p=0.0000 and p=0.0003) as though the probe had separated them.

    Only ONE contrast in this design holds total prior bytes fixed: the
    diagonal, lo_hi vs hi_lo, 24 GiB either way. That is why it is the primary
    comparison, and it is the reason the four cells were chosen. The factorial
    below is a decomposition of the grid, not four independent questions --
    `route_at_fixed_total_bytes` is the unconfounded one.

    The general form, again: a factor name is a claim about what is held
    constant, and a 2x2 whose two factors multiply into a third quantity does
    not hold that third quantity constant on either margin.
    """
    for r in means:
        r["count_f"] = str(r["count"])
        r["bytes_f"] = str(r["buffer_mib"])
        r["total_f"] = str(r["prefix_high_water_GiB"])

    gm = statistics.fmean(r["TBps_at_min"] for r in means)
    tot = sum((r["TBps_at_min"] - gm) ** 2 for r in means)

    by = defaultdict(list)
    for r in means:
        by[r["cell"]].append(r["TBps_at_min"])
    cells = {
        c: {
            "n": len(v),
            "mean_TBps": statistics.fmean(v),
            "stdev_TBps": statistics.stdev(v) if len(v) > 1 else 0.0,
            "min_TBps": min(v),
            "max_TBps": max(v),
        }
        for c, v in sorted(by.items())
    }
    between = sum(len(v) * (statistics.fmean(v) - gm) ** 2 for v in by.values())

    tests = {}
    for label, base, full in (
        ("count_given_PER_BUFFER_size", ["bytes_f"], ["bytes_f", "count_f"]),
        ("PER_BUFFER_size_given_count", ["count_f"], ["count_f", "bytes_f"]),
    ):
        r0, p0 = _additive_fit(means, base)
        r1, p1 = _additive_fit(means, full)
        extra, df1, df2 = r0 - r1, p1 - p0, len(means) - p1
        f = (extra / df1) / (r1 / df2) if r1 > 0 and df1 > 0 and df2 > 0 else 0.0
        tests[label] = {
            "type_II_SS": extra,
            "eta_squared_pct": round(extra / tot * 100.0, 4) if tot > 0 else 0.0,
            "F": round(f, 4),
            "df": [df1, df2],
            "p": round(_f_sf(f, df1, df2), 4),
        }

    radd, padd = _additive_fit(means, ["count_f", "bytes_f"])
    within = sum((x - cells[c]["mean_TBps"]) ** 2 for c, v in by.items() for x in v)
    inter_ss = radd - within
    df1 = len(by) - padd
    df2 = len(means) - len(by)
    fi = (inter_ss / df1) / (within / df2) if within > 0 and df1 > 0 and df2 > 0 else 0.0

    # The unconfounded contrast, and the only one in the grid. A 3-level model on
    # TOTAL prior bytes (12/24/48 GiB) pools lo_hi and hi_lo into one 24 GiB
    # level; the saturated 4-cell model splits them. The difference between the
    # two is precisely "does the ROUTE to 24 GiB matter, holding 24 GiB fixed" --
    # which is the question the probe was built for, stated as a model comparison
    # rather than as a t-test, so it is on the same footing as the rows above.
    tby = defaultdict(list)
    for r in means:
        tby[r["total_f"]].append(r["TBps_at_min"])
    tmu = {k: statistics.fmean(v) for k, v in tby.items()}
    rss_total = sum((r["TBps_at_min"] - tmu[r["total_f"]]) ** 2 for r in means)
    d1, d2 = len(by) - len(tby), len(means) - len(by)
    fr = ((rss_total - within) / d1) / (within / d2) if within > 0 and d1 > 0 else 0.0

    return {
        "cells": cells,
        "grand_mean_TBps": gm,
        "total_prior_bytes_only_model": {
            "levels_GiB": {
                k: round(v, 5) for k, v in sorted(tmu.items(), key=lambda x: float(x[0]))
            },
            "eta_squared_pct": round((1 - rss_total / tot) * 100.0, 4) if tot > 0 else 0.0,
            "note": (
                "one number per distinct total prior byte count, pooling the two routes "
                "to 24 GiB. Compare to the saturated figure below: the gap between them "
                "is everything the ROUTE explains."
            ),
        },
        "route_at_fixed_total_bytes": {
            "SS": rss_total - within,
            "eta_squared_pct": round((rss_total - within) / tot * 100.0, 4) if tot > 0 else 0.0,
            "F": round(fr, 4),
            "df": [d1, d2],
            "p": round(_f_sf(fr, d1, d2), 4),
            "this_is_the_only_unconfounded_term_in_this_dict": True,
            "why": (
                "it is the sole comparison in the grid where total prior bytes is held "
                "fixed at 24 GiB while the route to it changes (24x1GiB vs 48x512MiB). "
                "Both Type-II 'main effects' below change the total on every contrast "
                "they average over, because count * per-buffer size IS the total."
            ),
        },
        "saturated_between_cell_eta_squared_pct": round(between / tot * 100.0, 4)
        if tot > 0
        else 0.0,
        "within_cell_eta_squared_pct": round((tot - between) / tot * 100.0, 4) if tot > 0 else 0.0,
        "main_effects_type_II": tests,
        "interaction": {
            "SS": inter_ss,
            "eta_squared_pct": round(inter_ss / tot * 100.0, 4) if tot > 0 else 0.0,
            "F": round(fi, 4),
            "df": [df1, df2],
            "p": round(_f_sf(fi, df1, df2), 4),
        },
        "how_to_read_these": (
            "the saturated between-cell figure is what four separate cell means explain, "
            "and it is large by construction whenever cells differ at all. The Type-II "
            "rows are each factor adjusted for the OTHER FACTOR ONLY -- not for total "
            "prior bytes, which neither of them holds fixed. Read "
            "`route_at_fixed_total_bytes` for the question this probe exists to answer; "
            "the Type-II rows describe the grid's shape and must not be quoted as "
            "'count matters independent of bytes'."
        ),
        "the_trap_in_this_dict": (
            "`count_given_PER_BUFFER_size` is significant, and it is NOT evidence that "
            "count matters at fixed bytes. Doubling count at fixed per-buffer size "
            "doubles the total. The name says which variable is adjusted for, and a "
            "reader in a hurry will read it as which quantity is held constant. Those "
            "differ here, and the fields were renamed from `count_given_bytes` / "
            "`bytes_given_count` for exactly that reason."
        ),
    }


def _diagonal(means):
    """The one comparison the probe exists for: same bytes, different count.

    Reported with a difference and an interval, not only a p-value. n=4 per cell
    cannot make a null informative on its own, so the detectable-effect figure
    below is what turns 'no difference found' into a bounded statement.
    """
    a = [r["TBps_at_min"] for r in means if r["cell"] == "lo_hi"]
    b = [r["TBps_at_min"] for r in means if r["cell"] == "hi_lo"]
    if len(a) < 2 or len(b) < 2:
        return {"error": "need >=2 per diagonal cell"}
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    sp = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    se = math.sqrt(sp * (1 / na + 1 / nb)) if sp > 0 else 0.0
    t = (ma - mb) / se if se > 0 else 0.0
    df = na + nb - 2
    p = _f_sf(t * t, 1, df)
    # t crit at 95%, df=6, from the same F machinery: F(1,df) at p=.05.
    lo_f, hi_f = 0.0, 100.0
    for _ in range(200):
        mid = (lo_f + hi_f) / 2
        if _f_sf(mid, 1, df) > 0.05:
            lo_f = mid
        else:
            hi_f = mid
    tcrit = math.sqrt((lo_f + hi_f) / 2)
    half = tcrit * se
    return {
        "lo_hi_mean_TBps": ma,
        "hi_lo_mean_TBps": mb,
        "difference_TBps": ma - mb,
        "difference_pct_of_lo_hi": round((ma - mb) / ma * 100.0, 4),
        "ci95_TBps": [round(ma - mb - half, 5), round(ma - mb + half, 5)],
        "t": round(t, 4),
        "df": df,
        "p": round(p, 4),
        "smallest_difference_this_could_detect_TBps": round(half, 5),
        "smallest_difference_this_could_detect_pct": round(half / ma * 100.0, 4),
        "staircase_step_size_for_scale_TBps": round(STAIRCASE_HIGH_TBPS - STAIRCASE_LOW_TBPS, 5),
        "powered_for_a_staircase_sized_step": half < (STAIRCASE_HIGH_TBPS - STAIRCASE_LOW_TBPS),
        "verdict_against_the_preregistered_rule": (
            "p=0.0755 at alpha=.05 does not reject, but the interval is what carries "
            "the content and it is NOT a clean null: [-0.0218, +0.0014] TB/s excludes "
            "everything below -0.022 and only barely includes zero. The point estimate "
            "-0.0102 TB/s is one sixth of the staircase's 0.060 step and the design "
            "could resolve 0.0116, so a step-sized route effect is firmly excluded "
            "while a small one is not. Read as: at 24 GiB held fixed, the route "
            "(24 x 1 GiB vs 48 x 512 MiB) does not produce a staircase-sized change, "
            "and whether it produces a small one is unresolved at n=4."
        ),
        "the_model_comparison_disagrees_and_that_matters": (
            "the same contrast tested as a model comparison against the pooled "
            "within-cell variance gives F(1,12)=7.51, p=0.0179 -- significant at the "
            "same alpha the t-test misses. They differ because the F test borrows "
            "variance from all four cells (df=12) while the t-test uses only the two "
            "diagonal cells (df=6). Neither is wrong; the F test's extra power is "
            "bought with an equal-variance assumption across cells whose stdevs run "
            "0.0018 to 0.0083, a 4.6x spread. I am not picking the one I like: the "
            "honest statement is that the route effect sits right at the resolution "
            "limit of this design, which is why the follow-up below is not optional."
        ),
        "why_the_interval_and_not_just_p": (
            "with n=4 per cell a large p is compatible with an effect the design simply "
            "could not see. The interval says which differences are excluded; if it is "
            "wider than the staircase step this probe cannot call the null either way, "
            "and that is a result about the probe rather than about the allocator."
        ),
    }


def _anchor(means):
    """Does the treatment do anything at all in this range?

    The declared threat to the design, written down before the run: the
    staircase's steps are at prefix 13, 17 and 21. BOTH counts here (24 and 48)
    are past all three. If the level saturates above 21, all four cells sit at
    one level and a null between them says nothing about count vs bytes -- it
    says the probe was run entirely on the flat part of the curve.

    This is not decidable from the four cells alone, which is why it is a
    limitation field and not a test. A count=0 cell is the missing anchor and is
    declared here as a follow-up rather than added after seeing which way the
    numbers went.
    """
    vals = [r["TBps_at_min"] for r in means]
    gm = statistics.fmean(vals)
    return {
        "all_counts_are_past_every_known_staircase_step": True,
        "known_step_locations_n_prior_allocs": [13, 17, 21],
        "counts_tested": [24, 48],
        "grand_mean_TBps": gm,
        "staircase_low_TBps": STAIRCASE_LOW_TBPS,
        "staircase_high_TBps": STAIRCASE_HIGH_TBPS,
        "grand_mean_is_nearer": "high"
        if abs(gm - STAIRCASE_HIGH_TBPS) < abs(gm - STAIRCASE_LOW_TBPS)
        else "low",
        "no_zero_count_cell_in_this_run": True,
        "what_that_costs": (
            "without a count=0 condition in these same processes there is no internal "
            "evidence that the prefix moves the rate AT ALL here. A null across the four "
            "cells is then ambiguous between 'count and bytes both do not matter' and "
            "'the curve is flat above 21 and this probe never left the plateau'. The "
            "comparison to the staircase levels above is across sessions and processes, "
            "so it is context, not an internal control."
        ),
        "declared_followup": (
            "add count=0 and count=13 cells at 512 MiB in one shuffled run with the "
            "existing four. Declared here, before the numbers were read, so that adding "
            "it later is a pre-registered step rather than a reaction to the result."
        ),
    }


def _rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[s[k]] = r
        i = j + 1
    return out


def _corr(x, y):
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def _perm_p(xs, res, stat):
    rnd = random.Random(PERM_SEED)
    p2, ge = xs[:], 0
    for _ in range(PERM_DRAWS):
        rnd.shuffle(p2)
        if abs(_corr(_rank(p2), _rank(res))) >= abs(stat) - 1e-12:
            ge += 1
    return ge / PERM_DRAWS


def _ols(y, cols):
    """Least squares by Gaussian elimination on the normal equations. Returns RSS, npar."""
    n, p = len(y), len(cols)
    a = [[sum(cols[i][k] * cols[j][k] for k in range(n)) for j in range(p)] for i in range(p)]
    b = [sum(cols[i][k] * y[k] for k in range(n)) for i in range(p)]
    for i in range(p):
        piv = max(range(i, p), key=lambda r: abs(a[r][i]))
        a[i], a[piv] = a[piv], a[i]
        b[i], b[piv] = b[piv], b[i]
        for r in range(i + 1, p):
            m = a[r][i] / a[i][i]
            for c in range(i, p):
                a[r][c] -= m * a[i][c]
            b[r] -= m * b[i]
    x = [0.0] * p
    for i in reversed(range(p)):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, p))) / a[i][i]
    fit = [sum(x[i] * cols[i][k] for i in range(p)) for k in range(n)]
    return sum((y[k] - fit[k]) ** 2 for k in range(n)), p


def _trend_adjustment(raw_rows):
    """Test the bytes effect against an explicit linear time trend, on all four bases.

    Replaces an assertion. The previous wording said the realized shuffle's
    -0.1917 bytes-vs-position correlation meant "drift cannot manufacture a bytes
    effect". @Autotune (8f43b362) and @Reviewer both pointed out the same slip,
    which @Reviewer had just caught @Autotune making: randomization makes drift
    independent of treatment in the ASSIGNMENT DISTRIBUTION, and one realized
    draw's diagnostic cannot carry a statement about the distribution. The fix
    @Autotune proposed is the right one -- put time in the model and look.

    Fitting `y ~ 1 + bytes_dummies + position` and taking each term adjusted for
    the other turns the claim into something that can fail. It does not fail, on
    any basis: the trend adjusted for bytes is never distinguishable from zero,
    and the bytes effect is essentially unmoved.

    Caveat, @Autotune's own and it is the binding one: this fits drift as LINEAR
    in position. Non-monotone or periodic time structure passes it untouched,
    exactly as it passes Spearman. Nothing in this artifact closes that.
    """
    rows = sorted(raw_rows, key=lambda r: r["seq"])
    t = [float(r["seq"]) for r in rows]
    one = [1.0] * len(rows)
    lv = sorted({r["prefix_high_water_GiB"] for r in rows})
    dmy = [[1.0 if r["prefix_high_water_GiB"] == v else 0.0 for r in rows] for v in lv[1:]]
    out = {}
    for name, fn in (
        ("mean_PREREGISTERED_PRIMARY", statistics.fmean),
        ("median", statistics.median),
        ("max_coordinate", max),
        ("min_coordinate", min),
    ):
        y = [fn(r["TBps_per_slot"]) for r in rows]
        gm = statistics.fmean(y)
        tot = sum((v - gm) ** 2 for v in y)
        r_full, p_full = _ols(y, [one, *dmy, t])
        r_t, _ = _ols(y, [one, t])
        r_b, _ = _ols(y, [one, *dmy])
        df2 = len(y) - p_full
        fb = ((r_t - r_full) / len(dmy)) / (r_full / df2)
        ft = ((r_b - r_full) / 1) / (r_full / df2)
        out[name] = {
            "bytes_eta_squared_pct_unadjusted": round((1 - r_b / tot) * 100.0, 2),
            "bytes_eta_squared_pct_adjusted_for_trend": round((r_t - r_full) / tot * 100.0, 2),
            "bytes_F": round(fb, 2),
            "bytes_p": round(_f_sf(fb, len(dmy), df2), 4),
            "trend_given_bytes_eta_squared_pct": round((r_b - r_full) / tot * 100.0, 2),
            "trend_F": round(ft, 2),
            "trend_p": round(_f_sf(ft, 1, df2), 4),
        }
    tp = [v["trend_p"] for v in out.values()]
    out["verdict"] = (
        f"The bytes effect survives explicit adjustment for a linear time trend on ALL "
        f"FOUR aggregations, and the trend adjusted for bytes is never distinguishable "
        f"from zero (p from {min(tp):.4f} to {max(tp):.4f}). On the pre-registered mean "
        "the bytes effect is marginally LARGER adjusted than unadjusted (94.53% -> "
        "95.11%) with the trend at F(1,12)=1.78. This replaces the earlier assertion "
        "that 'drift cannot manufacture a bytes effect', which was a statement about "
        "the assignment distribution resting on one realized draw's diagnostic. NOTE "
        "the model fits drift as LINEAR in position; non-monotone or periodic time "
        "structure passes it untouched, just as it passes Spearman."
    )
    return out


def _headline_sweep(raw_rows):
    """THE HEADLINE RESULT, recomputed on four aggregations. Read this before quoting 94.53%.

    Written after @Autotune's 8f43b362 showed the drift statistic's sign is a
    free parameter of aggregation. The obvious next question -- which I had not
    asked, because I had pre-registered the mean and stopped -- is whether the
    RESULT is too. It is, and considerably more so than the drift number:

      agg      bytes-only   route eta2   route p    diagonal (hi_lo - lo_hi)
      mean        94.53%       2.11%      0.0179       +0.01019
      median      86.09%       0.04%      0.8553       -0.00170
      max         34.31%      39.87%      0.0010       -0.03093
      min         91.41%       2.28%      0.0593       +0.02284

    Three things a reader has to be told rather than left to find.

    First, the diagonal difference CHANGES SIGN across bases. That is the one
    contrast in the design that holds total prior bytes fixed and the whole
    reason the four cells were chosen. On mean and min the 48x512MiB route is
    faster; on median and max the 24x1GiB route is. Whatever "the route effect"
    is, this design cannot even give it a direction.

    Second, on the max-coordinate basis the bytes story inverts: total bytes
    explains 34% and the route explains 40%, with p=0.0010 -- the opposite
    conclusion to the pre-registered one, at a smaller p. If I had pre-registered
    max I would now be reporting that count matters and bytes mostly does not.

    Third, the direction of the qualitative claim does survive on three of four
    bases (bytes-only 86-95%, route 0-2%), and max is the outlier. But "three of
    four" is not the pre-registration's promise. The pre-registration fixed the
    mean and that binding is kept -- switching now, having seen the table, would
    be the exact post-hoc selection the whole file exists to prevent. What has to
    change instead is the STRENGTH of the claim, and it does: 94.53% is one
    basis's figure, not a property of the data.

    Why the bases diverge is itself informative and is not established here: the
    five slots within a process differ systematically (that is the whole slot
    effect from the earlier work), so mean/median/max/min are not four noisy
    estimates of one quantity -- they are four different quantities. Choosing
    among them is a modelling decision that the pre-registration made silently.
    """
    rows = sorted(raw_rows, key=lambda r: r["seq"])
    out = {}
    for name, fn in (
        ("mean_PREREGISTERED_PRIMARY", statistics.fmean),
        ("median", statistics.median),
        ("max_coordinate", max),
        ("min_coordinate", min),
    ):
        y = [fn(r["TBps_per_slot"]) for r in rows]
        gm = statistics.fmean(y)
        tot = sum((v - gm) ** 2 for v in y)
        tby, cby = defaultdict(list), defaultdict(list)
        for r, v in zip(rows, y):
            tby[r["prefix_high_water_GiB"]].append(v)
            cby[r["cell"]].append(v)
        tmu = {k: statistics.fmean(v) for k, v in tby.items()}
        cmu = {k: statistics.fmean(v) for k, v in cby.items()}
        rss_t = sum((v - tmu[r["prefix_high_water_GiB"]]) ** 2 for r, v in zip(rows, y))
        within = sum((v - cmu[r["cell"]]) ** 2 for r, v in zip(rows, y))
        d1, d2 = len(cby) - len(tby), len(y) - len(cby)
        f = ((rss_t - within) / d1) / (within / d2) if within > 0 and d1 > 0 else 0.0
        out[name] = {
            "total_prior_bytes_only_eta_squared_pct": round((1 - rss_t / tot) * 100.0, 2),
            "saturated_eta_squared_pct": round((1 - within / tot) * 100.0, 2),
            "route_eta_squared_pct": round((rss_t - within) / tot * 100.0, 2),
            "route_F": round(f, 3),
            "route_p": round(_f_sf(f, d1, d2), 4),
            "diagonal_hi_lo_minus_lo_hi_TBps": round(cmu["hi_lo"] - cmu["lo_hi"], 5),
        }
    diag = [v["diagonal_hi_lo_minus_lo_hi_TBps"] for v in out.values()]
    byt = [v["total_prior_bytes_only_eta_squared_pct"] for v in out.values()]
    out["verdict"] = (
        f"THE HEADLINE IS AGGREGATION-DEPENDENT. Total-prior-bytes eta^2 ranges "
        f"{min(byt):.2f}% to {max(byt):.2f}%. Worse, the diagonal difference -- the only "
        f"contrast holding total bytes fixed, and the reason these four cells exist -- "
        f"CHANGES SIGN, from {min(diag):+.5f} to {max(diag):+.5f} TB/s. This design "
        "cannot give the route effect a direction. On the max-coordinate basis the "
        "conclusion inverts outright: bytes 34.31%, route 39.87%, p=0.0010. The "
        "qualitative bytes story holds on three of four bases and the pre-registered "
        "mean is kept as primary, but 94.53% is one basis's number and must not be "
        "quoted as a property of the data."
    )
    return out


def _aggregation_sweep(raw_rows):
    """The same drift question asked on four different per-process aggregations.

    This exists because of @Autotune's f31e6406/8f43b362 finding on the staircase
    data: the SIGN of the residual-vs-order association there is a free parameter
    of how the five slot rates are collapsed to one number per process. Max
    coordinate gave -0.04 (p=0.79, reads as a clean null), median gave +0.26
    (p=0.09). Same rows, same residualization, opposite conclusions.

    Running the same sweep here shows this artifact has the identical problem,
    and the basis I happened to pre-register is the one that looks most like a
    finding. That is the worst possible way to be lucky, and the honest response
    is to publish all four rather than to defend the choice.

    The per-process MEAN remains the pre-registered primary for the bytes result,
    and it stays primary -- changing it now, after seeing that another basis
    gives a quieter drift number, would be exactly the post-hoc selection this
    whole file is built to avoid. What changes is the drift CLAIM, which is
    downgraded to "not robust to aggregation" and asserts nothing in either
    direction.
    """
    rows = sorted(raw_rows, key=lambda r: r["seq"])
    xs = [r["seq"] for r in rows]
    out = {}
    for name, fn in (
        ("mean_PREREGISTERED_PRIMARY", statistics.fmean),
        ("median", statistics.median),
        ("max_coordinate", max),
        ("min_coordinate", min),
    ):
        ys = [fn(r["TBps_per_slot"]) for r in rows]
        cmu = defaultdict(list)
        for r, v in zip(rows, ys):
            cmu[r["cell"]].append(v)
        cmu = {k: statistics.fmean(v) for k, v in cmu.items()}
        res = [v - cmu[r["cell"]] for r, v in zip(rows, ys)]
        rs = _corr(_rank(xs), _rank(res))
        out[name] = {
            "residual_spearman": round(rs, 4),
            "residual_pearson": round(_corr(xs, res), 4),
            "permutation_p_two_sided": round(_perm_p(xs, res, rs), 4),
            "raw_spearman": round(_corr(_rank(xs), _rank(ys)), 4),
        }

    # All 80 slot observations, residualized on the cell x slot-index mean, so
    # nothing is collapsed away at all.
    long = [(r["seq"], r["cell"], i, v) for r in rows for i, v in enumerate(r["TBps_per_slot"])]
    cs = defaultdict(list)
    for _s, c, i, v in long:
        cs[(c, i)].append(v)
    cs = {k: statistics.fmean(v) for k, v in cs.items()}
    lr = [v - cs[(c, i)] for _s, c, i, v in long]
    lp = [s for s, _c, _i, _v in long]
    out["no_aggregation_80_slot_obs"] = {
        "residual_spearman": round(_corr(_rank(lp), _rank(lr)), 4),
        "residual_pearson": round(_corr(lp, lr), 4),
        "note": (
            "cell x slot-index residuals over all 80 observations, nothing collapsed. "
            "These are not independent -- five share a process -- so no p is quoted."
        ),
    }
    ss = [v["residual_spearman"] for k, v in out.items() if k != "no_aggregation_80_slot_obs"]
    ps = [v["permutation_p_two_sided"] for k, v in out.items() if k != "no_aggregation_80_slot_obs"]
    out["verdict"] = (
        f"NOT ROBUST. Residual Spearman ranges {min(ss):+.4f} to {max(ss):+.4f} across four "
        f"aggregations of the same 16 processes, with permutation p from {min(ps):.4f} to "
        f"{max(ps):.4f}. No basis reaches p<.05. Nothing here licenses 'there is drift' or "
        "'there is no drift'. The pre-registered mean is the basis that looks MOST like a "
        "finding, which is why all four are published rather than the one."
    )
    return out


def _order_control(means, raw_rows):
    """Is the shuffled collection position predicting the rate anyway?

    The staircase needed three iterations to kill exactly this. One seeded
    shuffle makes position independent of cell in expectation, not in the draw
    that actually happened, so it is checked rather than assumed.

    Two corrections here, both from @Autotune's f31e6406, and the first one this
    function committed in its own first draft:

    RESIDUALS, NOT RAW RATE. The first version correlated position against the
    raw rate and got -0.0362, which reads as a clean null. It is not a null; it
    is the wrong quantity. Cell means differ by up to 0.064 TB/s, so the raw
    series is dominated by which cell each process happened to draw, and a drift
    in the residuals can hide underneath that -- or cancel against it. Removing
    the cell mean first gives +0.4441 Pearson, +0.4971 Spearman. Twelve times the
    magnitude and the opposite sign. Identical in shape to publishing
    `corr_level_vs_position` (a randomization-quality diagnostic) as evidence
    the outcome does not drift.

    SPEARMAN, NOT ONLY PEARSON. Pearson tests linear association; a non-monotone
    or periodic dependence on collection time passes it untouched. The rank
    statistic is what addresses the objection, and the docstring of the first
    draft said "a rank correlation instead" while the code computed Pearson --
    prose describing a check the code did not perform.
    """
    rows = [{**r, "pos": str(r["seq"])} for r in means]
    gm = statistics.fmean(r["TBps_at_min"] for r in rows)
    tot = sum((r["TBps_at_min"] - gm) ** 2 for r in rows)
    # Position as a continuous trend: 16 levels on 16 rows is saturated and
    # would explain 100% by construction. Halves and a rank correlation instead.
    n = len(rows)
    first = [r["TBps_at_min"] for r in rows if r["seq"] < n / 2]
    second = [r["TBps_at_min"] for r in rows if r["seq"] >= n / 2]
    xs = [r["seq"] for r in rows]
    ys = [r["TBps_at_min"] for r in rows]
    corr = _corr(xs, ys)

    # The cell mean removed. This is the series the drift question is about.
    cmu = defaultdict(list)
    for r in rows:
        cmu[r["cell"]].append(r["TBps_at_min"])
    cmu = {k: statistics.fmean(v) for k, v in cmu.items()}
    res = [r["TBps_at_min"] - cmu[r["cell"]] for r in rows]
    rp, rs = _corr(xs, res), _corr(_rank(xs), _rank(res))
    sweep = _aggregation_sweep(raw_rows)

    # Two-sided permutation p on the rank statistic, seeded so it is a property
    # of the artifact rather than of the run.
    #
    # The first version of this used the 16 cyclic rotations plus their reversals
    # as a deterministic reference set, to avoid introducing a seed. That is
    # WRONG and it is worth saying why, because it looked like the more rigorous
    # choice. The permutation null here is exchangeability of positions, and
    # rotations are not exchangeable draws: a rotation of 0..15 is still almost
    # monotone, so every member of that reference set has a large |rank corr|
    # with any trending residual. The statistic is therefore not extreme *within
    # its own reference set* and the test cannot reject regardless of the data.
    # It returned p=0.375 against 0.0519 from an honest shuffle -- a reference
    # set constructed to avoid an arbitrary constant, which instead built the
    # answer into the null. Same defect class as everything else in this file:
    # a check that passes for a reason other than the one it documents.
    pperm = _perm_p(xs, res, rs)

    # Bytes level by position, to show the shuffle decoupled treatment from time.
    tot_rank = [float(r["prefix_high_water_GiB"]) for r in rows]

    cnt_by_half = defaultdict(list)
    for r in rows:
        cnt_by_half["first" if r["seq"] < n / 2 else "second"].append(r["count"])

    return {
        "corr_RESIDUAL_vs_position_pearson": round(rp, 4),
        "corr_RESIDUAL_vs_position_spearman": round(rs, 4),
        "permutation_p_two_sided": round(pperm, 4),
        "permutation_draws": PERM_DRAWS,
        "permutation_seed": PERM_SEED,
        "aggregation_sweep": sweep,
        "residual_drift_verdict": (
            "NOT ROBUST TO AGGREGATION, and therefore NOT A CLAIM in either direction. "
            "On the pre-registered per-process mean the residual drift is Spearman "
            "+0.4971 (p=0.052), which reads as a near-significant finding. On the median "
            "of the same five slot rates it is +0.0853 (p=0.749). Same 16 processes, same "
            "residualization, and the pre-registered basis is the one that looks most "
            "like a result. See `aggregation_sweep` for all four. The bytes effect is "
            "unaffected by the drift, and that is now TESTED rather than asserted -- see "
            "`time_trend_adjustment`."
        ),
        "time_trend_adjustment": _trend_adjustment(raw_rows),
        "corr_RAW_rate_vs_position_pearson_DO_NOT_QUOTE_AS_THE_CONTROL": round(corr, 4),
        "why_that_field_is_named_that": (
            "the raw-rate correlation is -0.0362 and reads as a clean null. It is the "
            "wrong quantity: cell means differ by up to 0.064 TB/s, so the raw series is "
            "dominated by which cell each process drew and a residual drift hides "
            "underneath it. Twelve times the magnitude, opposite sign. It is kept "
            "because it is what the first draft of this function published, and a reader "
            "comparing artifacts should be able to see the corrected pair side by side."
        ),
        "corr_total_prior_bytes_vs_position_spearman": round(_corr(_rank(xs), _rank(tot_rank)), 4),
        "first_half_mean_TBps": statistics.fmean(first),
        "second_half_mean_TBps": statistics.fmean(second),
        "half_difference_TBps": statistics.fmean(first) - statistics.fmean(second),
        "half_difference_pct_of_total_variance": round(
            (statistics.fmean(first) - statistics.fmean(second)) ** 2 * n / 4 / tot * 100.0,
            3,
        )
        if tot > 0
        else 0.0,
        "counts_in_first_half": sorted(cnt_by_half["first"]),
        "counts_in_second_half": sorted(cnt_by_half["second"]),
        "shuffle_balanced_the_counts_across_halves": sorted(cnt_by_half["first"])
        == sorted(cnt_by_half["second"]),
        "why_not_position_as_a_factor": (
            "16 distinct positions on 16 rows is a saturated model: it explains 100% of "
            "the variance by construction and says nothing. The trend and the half-split "
            "are the tests that can fail."
        ),
        "why_spearman_and_not_only_pearson": (
            "Pearson tests linear association only; a non-monotone or periodic "
            "dependence on collection time passes it untouched. The rank statistic is "
            "what the objection actually requires, and the first draft of this function "
            "claimed one in its docstring while computing the other."
        ),
    }


def _band(means, rows):
    """Cell separation measured against the band in the band's own units."""
    lo, hi = BAND
    w = hi - lo
    by = defaultdict(list)
    for r in means:
        by[r["cell"]].append(r["TBps_at_min"])
    cm = {c: statistics.fmean(v) for c, v in by.items()}
    pairs = sorted(cm)
    seps = {
        f"{a}_vs_{b}": {
            "difference_TBps": abs(cm[a] - cm[b]),
            "in_band_widths": round(abs(cm[a] - cm[b]) / w, 3),
        }
        for i, a in enumerate(pairs)
        for b in pairs[i + 1 :]
    }
    ranges = [max(v) - min(v) for v in by.values()]
    return {
        "band_TBps": list(BAND),
        "band_width_TBps": round(w, 6),
        "cell_mean_separations": seps,
        "within_cell_range_TBps_by_cell": {
            c: round(max(v) - min(v), 5) for c, v in sorted(by.items())
        },
        "largest_within_cell_range_in_band_widths": round(max(ranges) / w, 3),
        "why_absolute": (
            "a percentage-of-own-mean threshold grants a fast cell more absolute room "
            "than a slow one. Comparing a cell range as a percent of its own mean "
            "against a band width as a percent of 4.885 is two denominators in one "
            "comparison, and it once turned 9/20 into 10/20."
        ),
        "n_processes": len(rows),
    }


def _manifest(src, extra):
    entries = []
    for p in [Path(src)] + [REPO / e for e in extra]:
        b = p.read_bytes()
        try:
            name = str(p.resolve().relative_to(REPO))
        except ValueError:
            name = str(p.resolve())
        entries.append({"path": name, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)})
    return {
        "n_inputs": len(entries),
        "entries": entries,
        "what_this_proves": (
            "the assembler read exactly these bytes. It does not prove they came from a "
            "GPU, and a manifest cannot."
        ),
    }


def _git():
    def run(*a, strip=True):
        r = subprocess.run(
            ["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False
        )
        if r.returncode != 0:
            raise SystemExit(f"git {' '.join(a)} failed: {r.stderr.strip()}")
        return r.stdout.strip() if strip else r.stdout

    dirty = [ln[3:] for ln in run("status", "--porcelain", strip=False).splitlines()]
    return {
        "commit": run("rev-parse", "HEAD")[:7],
        "worktree_dirty": bool(dirty),
        "worktree_dirty_paths": sorted(dirty),
        "worktree_dirty_note": (
            "cannot read clean in the commit that contains this artifact: writing the "
            "file dirties the tree the flag describes. The path list is what makes the "
            "flag checkable."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    args = ap.parse_args()

    raw = json.loads(Path(args.src).read_text())
    rows = raw["rows"]
    means = _cell_means(rows)

    payload = {
        "what": (
            "Does prior-allocation COUNT or prior-allocation BYTES move the copy rate? "
            "Four cells cross 24/48 allocations with 512 MiB/1 GiB buffers; lo_hi and "
            "hi_lo reach the same 24 GiB prior peak by two different routes."
        ),
        "analysis_was_written_before_the_numbers": (
            "AI/assemble_alloc_factorial.py was authored while the 16-process sweep was "
            "still running and committed before its output was read. Unit of analysis, "
            "thresholds, the band's units, the anchor limitation and its follow-up are "
            "all fixed there. Only which pre-registered branch is taken was decided by "
            "the data."
        ),
        "what_the_author_had_already_seen": (
            "two smoke-test rows, run to check that lo_hi and hi_lo really reach the "
            "same 24 GiB peak before spending 16 processes on a design that rests on "
            "it. Their rates were on screen: lo_hi 4.9648/5.0411/4.9977/4.8784/4.8843 "
            "and hi_lo 4.9869/4.8870/4.9478/5.0067/4.9663 TB/s. That is one process per "
            "diagonal cell, and it is disclosed because 'written before the numbers' is "
            "otherwise a claim a reader cannot check and I cannot honestly make. Those "
            "two processes are NOT in the artifact -- the sweep ran fresh -- so they "
            "biased the choice of analysis, if at all, and not the analysis itself."
        ),
        "and_the_file_mtime_will_not_show_this": (
            "assemble_alloc_factorial.py's mtime is LATER than alloc_factorial.json's, "
            "because ruff format rewrote it after the sweep finished. The ordering "
            "claim above rests on the commit sequence, not on mtimes, and a reader "
            "checking the obvious filesystem evidence would find it says the opposite."
        ),
        "device_name": raw.get("device_name"),
        "shuffle_seed": raw.get("shuffle_seed"),
        "n_processes": len(rows),
        "headline_sweep_READ_BEFORE_QUOTING": _headline_sweep(rows),
        "preconditions": _preconditions(rows, raw.get("measurement_live_set_GiB", 12.0)),
        "primary_comparison": _diagonal(means),
        "factorial": _factorial(means),
        "anchor_limitation": _anchor(means),
        "order_control": _order_control(means, rows),
        "band": _band(means, rows),
        "conclusion": (
            "On the pre-registered per-process mean: prior-allocation TOTAL BYTES moves "
            "the copy rate (12/24/48 GiB -> 4.9077/4.9568/4.9715 TB/s, 94.53% of the "
            "variance, a 0.0638 TB/s swing the size of the staircase's own 0.060 step), "
            "and the ROUTE to a given total adds 2.11%. So the staircase axis is better "
            "described as prior bytes than as prior allocation count. BUT SEE "
            "`headline_sweep_READ_BEFORE_QUOTING` FIRST: that 94.53% is one aggregation's "
            "figure, ranging 34.31% to 94.53% across four, and the diagonal contrast "
            "changes sign between them. The qualitative claim survives on three of four "
            "bases; the number does not survive on any but its own."
        ),
        "three_limits_on_that_conclusion": (
            "FIRST: 'count does not matter' is not established. The diagonal is the only "
            "contrast holding total bytes fixed, and it is equivocal -- t(6)=-2.15 "
            "p=0.0755 against F(1,12)=7.51 p=0.0179 for the same comparison. A "
            "step-sized route effect is excluded; a small one is not. SECOND: the "
            "response is strongly concave (+0.0491 TB/s from 12->24 GiB, then only "
            "+0.0147 from 24->48), so the two routes are being compared at 24 GiB, on "
            "the part of the curve that is already flattening. A route effect could be "
            "larger lower down and this design would not see it. THIRD, and worst: the "
            "diagonal difference changes sign across per-process aggregations "
            "(-0.03093 to +0.02284 TB/s), so this design cannot give the route effect a "
            "direction at all, let alone a magnitude. All three are why the count=0 and "
            "count=13 anchor cells are declared, not optional -- and the third is why "
            "more repeats per cell are needed as much as more cells."
        ),
        "unit_of_analysis": (
            "per-process mean over the five identical-buffer slots, n=4 per cell. The "
            "five slots share one process and one allocator state; they are repeated "
            "measures, not replicates."
        ),
        "what_this_cannot_do": (
            "it separates two candidate axes and identifies no mechanism. Nothing here "
            "explains why a level differs, and no factorial or ordering control can."
        ),
        "manifest": _manifest(
            args.src, ["AI/probe_alloc_factorial.py", "AI/assemble_alloc_factorial.py"]
        ),
        "git": _git(),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

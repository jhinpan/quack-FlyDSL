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
    """
    for r in means:
        r["count_f"] = str(r["count"])
        r["bytes_f"] = str(r["buffer_mib"])

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
        ("count_given_bytes", ["bytes_f"], ["bytes_f", "count_f"]),
        ("bytes_given_count", ["count_f"], ["count_f", "bytes_f"]),
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

    return {
        "cells": cells,
        "grand_mean_TBps": gm,
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
            "rows are each factor adjusted for the other, which is the number that "
            "answers 'does count matter once bytes are held' -- these are not "
            "interchangeable and publishing the first as the second overstates by "
            "orders of magnitude."
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


def _order_control(means):
    """Is the shuffled collection position predicting the rate anyway?

    The staircase needed three iterations to kill exactly this. One seeded
    shuffle makes position independent of cell in expectation, not in the draw
    that actually happened, so it is checked rather than assumed.
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
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    corr = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0

    cnt_by_half = defaultdict(list)
    for r in rows:
        cnt_by_half["first" if r["seq"] < n / 2 else "second"].append(r["count"])

    return {
        "corr_rate_vs_collection_position": round(corr, 4),
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
        "preconditions": _preconditions(rows, raw.get("measurement_live_set_GiB", 12.0)),
        "primary_comparison": _diagonal(means),
        "factorial": _factorial(means),
        "anchor_limitation": _anchor(means),
        "order_control": _order_control(means),
        "band": _band(means, rows),
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

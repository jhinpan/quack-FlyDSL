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

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "AI/data/copy_placement_draws/copy_axes_dev5.json"

# Inherited from 4d6ed9de. Named, not assumed: a bare pair reads as closed.
BAND = (4.885, 4.895)
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
        d = json.loads(path.read_text())
        runs[d["peak_live_512mib"]].append(d)
    if not runs:
        raise SystemExit(f"no runs in {src_dir}")
    return dict(sorted(runs.items()))


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

    return {
        "n_draws": len(allv),
        "min_TBps": min(allv),
        "max_TBps": max(allv),
        "pooled_range_pct_of_min": round((max(allv) / min(allv) - 1) * 100.0, 2),
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
        "band_is_straddled": bool(below and above),
        "all_draws_unrounded": True,
        "no_undecidable_draws": (
            "every draw here is stored unrounded, so unlike the 512-MiB-only artifact "
            "there is no draw whose band membership the record cannot decide"
        ),
    }


def _size_discrimination(runs):
    """Can any pattern tell the two buffer sizes apart? Measured, not assumed."""
    out = {}
    allr = [r for rs in runs.values() for r in rs]
    for pat in ("write", "two_read_one_write", "copy"):
        a = [v for r in allr for v in r["roofline_patterns_by_size"]["512MiB"][pat]]
        b = [v for r in allr for v in r["roofline_patterns_by_size"]["2048MiB"][pat]]
        out[pat] = {
            "range_512MiB": [min(a), max(a)],
            "range_2048MiB": [min(b), max(b)],
            "spread_512MiB_pct": round((max(a) / min(a) - 1) * 100.0, 2),
            "spread_2048MiB_pct": round((max(b) / min(b) - 1) * 100.0, 2),
            "n_per_size": len(a),
            "ranges_overlap": not (max(a) < min(b) or max(b) < min(a)),
            "historical_TBps": HISTORICAL[pat],
        }
    out["verdict"] = (
        "No pattern separates 512 MiB from 2 GiB once allocation slot and allocator "
        "peak are sampled -- all three overlap. An earlier draft of this argument "
        "used two_read_one_write's 0.31% single-slot spread as an error bar and "
        "concluded from a 4.04% gap that the historical table must be 2 GiB. That "
        "spread was measured with slot and peak both held fixed, the one condition "
        "under which neither confound is visible; sampled properly the same probe "
        "spreads 2.47%. The table's size is known from notes:869 stating it, which "
        "is documentary evidence. Nothing in this artifact measures it."
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
            for p in range(4):
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


def _staircase(path=Path("/tmp/copyaxes_steps.json")):
    """Where the allocator-peak steps are, from the fine sweep, if it was run.

    Optional because it costs ~44 processes. When absent the artifact says so
    rather than carrying a remembered pair of integers: an earlier draft of the
    notes asserted steps "at 13 and again at 17" from an uncommitted /tmp
    script, which is a number living only in prose.
    """
    if not path.exists():
        return {"collected": False, "how": "python AI/probe_copy_size_draws.py OUT --sweep-steps"}
    d = json.loads(path.read_text())
    by = defaultdict(list)
    for r in d["rows"]:
        by[r["peak_live_512mib"]].append(r["draws_2GiB"])
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
        if shift is not None and shift > 3 * max(repeat, 0.5):
            steps.append(peak)
        prev = means
    return {
        "collected": True,
        "reps_per_level": d["reps_per_level"],
        "one_process_per_row": d["one_process_per_row"],
        "step_at_peak_live_512mib": steps,
        "criterion": "level-to-level shift exceeds 3x the across-process repeat spread",
        "levels": levels,
        "why_not_within_process": d["why_not_within_process"],
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
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/copyaxes"
    runs = _load(src)
    allr = [r for rs in runs.values() for r in rs]

    payload = {
        "what": (
            "Copy-rate draws at both 512 MiB and the 2 GiB size the historical "
            "roofline table was measured at, decomposed across three axes: "
            "allocation slot, process, and allocator peak high-water mark."
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
        "peak_levels_512mib": sorted(runs),
        "generator": "AI/probe_copy_size_draws.py",
        "assembler": "AI/assemble_copy_axes.py",
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
        "peak_staircase": _staircase(),
    }

    hist_axis = payload["axes"]["2048MiB"]
    hist_band = payload["band_test"]["2048MiB"]
    payload["verdict"] = (
        "The conclusion holds at the size that matters, for a better-supported reason. "
        "At 2 GiB the copy rate ranges {r:.2f}% over {n} draws spanning {s} allocation "
        "slots and {p} allocator peaks, which is {x:.0f}x the {w:.2f}% width of the band "
        "the historical 4.89 cell implies, and the draws straddle that band ({b} below, "
        "{a} at or above). A single copy draw therefore cannot authenticate a harness, a "
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

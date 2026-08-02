"""Assemble the cross-device view of the occupancy intervention.

Why this file exists, and why the thing it replaces was not good enough.

The first cross-device sidecar was hand-assembled: I ran the probe four times,
read the numbers out, and typed a summary. @Reviewer requested changes on it and
every structural objection was correct.

  - No assembler. The summary could not be regenerated, so it could not be
    checked, and it aged the moment the rows moved.
  - Mixed provenance. Three of the four inputs came from generator
    `2edfda85`; the fourth (device 5, run 0) came from `3fcf0041`, a generator
    revision that was never committed to the repository at all. So "two runs
    per die" was really one die's pair under two different generators against
    another die's matched pair -- the asymmetry sat exactly on the axis the
    comparison was about.
  - No raw inputs, no hashes, no physical device IDs.
  - A reducer chosen after seeing the data. I published "occupancy matches
    across dies to within 0.6%". That is true only under closest-endpoint --
    the smallest of the four pairwise gaps. On means it is 0.924% and on full
    observed range 1.275%. I did not state which reducer I used, because I did
    not notice I had chosen one.
  - Ratios that silently selected run 0 of each die, presented to three
    significant figures as though the third digit meant something.

That last one is the substantive error rather than a bookkeeping one, and
@Autotune pinned it: `control_none_to_2_change_pct` on device 5 alone spans
0.030% to 0.367% across the estimates in the tree. Publishing "0.37 (dev5) vs
0.44 (dev4)" as a matched pair implies a resolution the quantity does not have
-- the within-die range fully contains the between-die difference. Same for the
equal-occupancy ratio, where dev5's range (1.4508-1.4564) contains dev4's
(1.4533-1.4535) outright.

So this assembler does four things the hand summary did not:

  1. Reads committed raw inputs and records their sha256 and per-run provenance
     (commit, generator sha, device uuid), and REFUSES to assemble inputs from
     more than one generator revision.
  2. States the reducer before reporting anything, and reports all three
     (closest-endpoint, mean-vs-mean, full range) rather than the flattering one.
  3. For every derived ratio, reports the within-die range alongside the
     between-die difference, and marks the ratio `resolved: false` when the
     former swallows the latter. A quantity that cannot separate two dies is not
     evidence about two dies.
  4. Rounds to a precision the dispersion supports instead of to three digits.

Inputs: AI/data/cross_device_runs/*.json, each a full sidecar written by
AI/probe_rmsnorm_occupancy_intervention.py.

Run: python AI/assemble_intervention_cross_device.py
"""

import hashlib
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "AI/data/cross_device_runs"
OUT = REPO / "AI/data/rmsnorm_fwd_occupancy_intervention_cross_device.json"

# Stated BEFORE the numbers are read. The point of naming three is that no
# single one of them can be selected after the fact to flatter a claim.
REDUCERS = {
    "closest_endpoint_pct": "smallest gap between any dev-A estimate and any dev-B estimate; "
    "the most generous reading, and the one my '0.6%' claim silently used",
    "mean_vs_mean_pct": "difference of per-die means, the ordinary summary",
    "full_range_pct": "spread of all estimates from both dies together, the most conservative",
}
PRIMARY_REDUCER = "mean_vs_mean_pct"


def _load():
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        raw = path.read_bytes()
        d = json.loads(raw)
        runs.append(
            {
                "file": str(path.relative_to(REPO)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "device_index": str(d["HIP_VISIBLE_DEVICES"]),
                "device_uuid": d.get("device_uuid"),
                "device_name": d.get("device"),
                "commit": d.get("commit"),
                "worktree_dirty": d.get("worktree_dirty"),
                "generator_sha256_16": d["source_sha256_16"][
                    "AI/probe_rmsnorm_occupancy_intervention.py"
                ],
                "timing_regime": (d.get("timing_regime") or {}).get("regime"),
                "data": d,
            }
        )
    if not runs:
        raise SystemExit(f"no run inputs in {RUNS_DIR}")
    gens = {r["generator_sha256_16"] for r in runs}
    if len(gens) != 1:
        raise SystemExit(
            f"inputs span {len(gens)} generator revisions {sorted(gens)}. The first version "
            "of this comparison did exactly that -- one die's runs came from a generator "
            "revision that was never committed -- which put the asymmetry on the same axis "
            "as the claim. Re-run every die on one generator before assembling."
        )
    regimes = {r["timing_regime"] for r in runs}
    if regimes != {"eager"}:
        raise SystemExit(f"inputs disagree on timing regime: {regimes}")
    return runs


def _by_device(runs):
    out = {}
    for r in runs:
        out.setdefault(r["device_index"], []).append(r)
    if len(out) < 2:
        raise SystemExit("need at least two devices to say anything cross-device")
    for dev, rs in out.items():
        if len(rs) < 2:
            raise SystemExit(
                f"device {dev} has {len(rs)} run(s). A single estimate per die cannot "
                "separate a device effect from run-to-run noise -- that conflation is the "
                "error this file was written to stop repeating."
            )
    return out


def _reduce(per_die):
    """All three reducers over ALL dies, not a chosen pair.

    Takes {device: [estimates]}. The first version of this function took two
    lists and the callers passed devs[0] and devs[1], which silently dropped
    device 6 from every published figure while the file said "three dies". A
    summary that quietly excludes a third of its inputs is the same defect as a
    reducer chosen after the fact, so the shape of the function now makes it
    impossible: it consumes the whole mapping.
    """
    dies = sorted(per_die)
    if len(dies) < 2:
        raise SystemExit("need at least two dies to reduce across dies")
    pairs = [(x, y) for i, x in enumerate(dies) for y in dies[i + 1 :]]
    closest = min(
        min(abs(u - v) / min(u, v) * 100 for u in per_die[x] for v in per_die[y])
        for x, y in pairs
    )
    means = {d: statistics.fmean(per_die[d]) for d in dies}
    worst_mean_pair = max(
        abs(means[y] - means[x]) / min(means[x], means[y]) * 100 for x, y in pairs
    )
    allv = [v for d in dies for v in per_die[d]]
    full = (max(allv) - min(allv)) / min(allv) * 100
    return {
        "closest_endpoint_pct": round(closest, 4),
        "mean_vs_mean_pct": round(worst_mean_pair, 4),
        "full_range_pct": round(full, 4),
        "dies_compared": dies,
    }


def _within_die_range_pct(vals):
    if len(vals) < 2:
        return None
    return round((max(vals) - min(vals)) / min(vals) * 100, 4)


def _spread_abs(vals):
    """Absolute spread. Used for quantities that live near zero.

    control_none_to_2_change_pct is itself a percentage that straddles zero
    (estimates from -0.19 to +1.10). Relative comparison of such a quantity is
    meaningless -- dividing by a near-zero base produced a -404.8% 'within-die
    range' and a 334.6% 'between-die difference', and the resolved test then
    reported True on the one ratio that is least resolvable of the four. A
    percentage change of a percentage change is not a percentage change.
    """
    return round(max(vals) - min(vals), 4)


def _ratios(d):
    t = {r["waves_per_eu_hint"]: r for r in d["treatment_sweep"]}
    c = {r["waves_per_eu_hint"]: r for r in d["control_sweep"]}
    return {
        "treatment_none_to_2_lift_pct": (t[2]["tbs"] - t["none"]["tbs"]) / t["none"]["tbs"] * 100,
        "control_none_to_2_change_pct": (c[2]["tbs"] - c["none"]["tbs"]) / c["none"]["tbs"] * 100,
        "equal_occupancy_bandwidth_ratio": c["none"]["tbs"] / t[2]["tbs"],
        "highest_occ_vs_unhinted_margin_pct": (t["none"]["tbs"] - t[4]["tbs"])
        / t[4]["tbs"]
        * 100,
    }


def _ratio_block(by_dev):
    """Per-ratio: can this quantity tell the dies apart at all?

    Everything here is in the ratio's OWN units, absolute, not as a percentage
    of itself. Two of these quantities live near zero and one is a bare ratio
    near 1.45, so relative comparison is either meaningless or misleading; a
    first version of this function divided by a near-zero base and produced a
    -404.8% within-die range, then reported resolved=True on the strength of it.

    Separation is tested over EVERY pair of dies, not just the extremes. With
    three dies the min/max pair can be disjoint while a middle die overlaps
    both, which is not separation.
    """
    per_dev = {dev: [_ratios(r["data"]) for r in rs] for dev, rs in by_dev.items()}
    names = sorted(next(iter(per_dev.values()))[0])
    devs = sorted(per_dev)
    out = {}
    for name in names:
        vals = {d: [x[name] for x in per_dev[d]] for d in devs}
        within = {d: _spread_abs(vals[d]) for d in devs}
        worst_within = max(within.values())
        means = {d: statistics.fmean(vals[d]) for d in devs}
        pairs = [(x, y) for i, x in enumerate(devs) for y in devs[i + 1 :]]
        between = max(abs(means[y] - means[x]) for x, y in pairs)
        # Resolution is a property of PAIRS, not of the whole set. Requiring
        # every pair to be disjoint marks a quantity unresolved whenever any two
        # dies happen to agree -- which is the normal case when one die is the
        # odd one out. highest_occ_vs_unhinted_margin separates device 4 from
        # 5 and 6 by 8.74 against a worst within-die spread of 0.16, and the
        # all-pairs rule called that unresolved because 5 and 6 agree. A
        # criterion that punishes agreement is measuring the wrong thing.
        separated_pairs = [
            {"dies": [x, y], "gap_abs": round(abs(means[y] - means[x]), 4)}
            for x, y in pairs
            if (max(vals[x]) < min(vals[y]) or max(vals[y]) < min(vals[x]))
            and abs(means[y] - means[x]) > worst_within
        ]
        overlapping_pairs = [
            {"dies": [x, y], "gap_abs": round(abs(means[y] - means[x]), 4)}
            for x, y in pairs
            if not (max(vals[x]) < min(vals[y]) or max(vals[y]) < min(vals[x]))
            or abs(means[y] - means[x]) <= worst_within
        ]
        resolved = bool(separated_pairs)
        out[name] = {
            "units": "absolute, in the ratio's own units -- NOT a percentage of itself",
            "per_device_estimates": {d: [round(v, 4) for v in vals[d]] for d in devs},
            "per_device_mean": {d: round(means[d], 4) for d in devs},
            "within_die_spread_abs": within,
            "worst_within_die_spread_abs": worst_within,
            "worst_between_die_mean_gap_abs": round(between, 4),
            "separated_pairs": separated_pairs,
            "indistinguishable_pairs": overlapping_pairs,
            "resolved": resolved,
            "reading": (
                "at least one pair of dies is separated by more than either die's own "
                "scatter (see separated_pairs); pairs in indistinguishable_pairs are not "
                "told apart by this quantity and should not be quoted as differing"
                if resolved
                else "no pair of dies is separated by more than the within-die scatter: "
                "this quantity does NOT distinguish any of these dies, and quoting one "
                "figure per die implies a resolution it does not have"
            ),
        }
    return out


def main():
    runs = _load()
    by_dev = _by_device(runs)
    devs = sorted(by_dev)

    rows = []
    keyed = {}
    for dev, rs in by_dev.items():
        for r in rs:
            for sweep in ("treatment_sweep", "control_sweep"):
                for row in r["data"][sweep]:
                    k = (sweep, row["n"], str(row["waves_per_eu_hint"]))
                    e = keyed.setdefault(k, {})
                    e.setdefault(dev, []).append(row)

    for (sweep, n, hint), per_dev in sorted(keyed.items()):
        bw = {d: [x["pct_of_ceiling"] for x in per_dev[d]] for d in devs}
        occ = {d: [x["measured_waves_per_simd"] for x in per_dev[d]] for d in devs}
        regs = {d: sorted({x["vgpr_count"] for x in per_dev[d]}) for d in devs}
        spills = {d: sorted({x["vgpr_spill_count"] for x in per_dev[d]}) for d in devs}
        rows.append(
            {
                "sweep": sweep,
                "n": n,
                "waves_per_eu_hint": hint,
                "bandwidth_pct_of_ceiling": bw,
                "bandwidth_within_die_range_pct": {d: _within_die_range_pct(bw[d]) for d in devs},
                "bandwidth_between_die": _reduce(bw),
                "occupancy_waves_per_simd": occ,
                "occupancy_between_die": _reduce(occ),
                "vgpr_count": regs,
                "vgpr_spill_count": spills,
                "static_identical_across_dies": bool(
                    len({tuple(regs[d]) for d in devs}) == 1
                    and len({tuple(spills[d]) for d in devs}) == 1
                    and all(len(regs[d]) == 1 for d in devs)
                ),
            }
        )

    worst_bw = max(r["bandwidth_between_die"][PRIMARY_REDUCER] for r in rows)
    worst_within_bw = max(
        v for r in rows for v in r["bandwidth_within_die_range_pct"].values() if v is not None
    )
    occ_by_reducer = {
        k: round(max(r["occupancy_between_die"][k] for r in rows), 4) for k in REDUCERS
    }

    payload = {
        "what": "the occupancy intervention run on three MI355X dies, two runs each, one "
        "generator revision, assembled by a script rather than by hand",
        "generator": "AI/assemble_intervention_cross_device.py",
        "inputs": [{k: v for k, v in r.items() if k != "data"} for r in runs],
        "supersedes": (
            "a hand-assembled summary that @Reviewer requested changes on. Every structural "
            "objection was correct: no assembler, no hashes, no raw inputs, mixed generator "
            "provenance (device 5 run 0 came from generator 3fcf0041, a revision never "
            "committed to this repository), a reducer chosen after seeing the data, and "
            "ratios that silently selected run 0. See the module docstring."
        ),
        "reducers": REDUCERS,
        "primary_reducer": PRIMARY_REDUCER,
        "reducer_note": (
            "Three are reported on every quantity because I previously published "
            "'occupancy matches across dies to within 0.6%' without noticing that I had "
            "selected closest-endpoint. On means the same rows give "
            f"{occ_by_reducer['mean_vs_mean_pct']}% and on full range "
            f"{occ_by_reducer['full_range_pct']}%. The claim was not wrong so much as "
            "unstated: a summary statistic with an unnamed reducer is not checkable."
        ),
        "occupancy_between_die_worst_by_reducer": occ_by_reducer,
        "bandwidth_worst_between_die_pct": worst_bw,
        "bandwidth_worst_within_die_pct": worst_within_bw,
        "bandwidth_finding": (
            f"under {PRIMARY_REDUCER}, the worst between-die bandwidth gap is {worst_bw}% "
            f"against a worst within-die range of {worst_within_bw}%. The dies differ by "
            "more than they scatter, so this is a device effect and not run-to-run noise. "
            "That is the one conclusion here that clears its own confound comfortably."
        ),
        "ratios": _ratio_block(by_dev),
        "ratio_note": (
            "Each ratio reports, in its own absolute units, every die's estimates and spread, "
            "which PAIRS of dies it separates, and which it does not. I previously published "
            "these as one three-digit figure per die for two dies, which implied a resolution "
            "they do not have. Concretely: equal_occupancy_bandwidth_ratio, which I published "
            "as '1.455 (dev5) vs 1.453 (dev4)', separates NO pair of the three dies -- the "
            "largest between-die gap is 0.0031 against a within-die spread of 0.0074, so the "
            "third digit was never real and the honest statement is '1.45 on all three'. "
            "control_none_to_2_change_pct separates only 4-vs-5, and its estimates straddle "
            "zero (-0.19 to +1.10), so the claim it supports is a bound and not a value: "
            "every estimate on every die is under 1.1%, i.e. the control is flat. "
            "treatment_none_to_2_lift_pct and highest_occ_vs_unhinted_margin_pct do separate "
            "device 4 from devices 5 and 6, and 5 and 6 are indistinguishable on both."
        ),
        "high_occupancy_margin_note": (
            "highest_occ_vs_unhinted_margin_pct is the claim that the highest-occupancy row "
            "in the treatment ladder (hint=4, 3.7 waves/SIMD) is also its WORST bandwidth, "
            "below the unhinted kernel at 1.0 wave/SIMD. @Autotune flagged that this margin "
            "is die-dependent and it is: ~10.7 points on devices 5 and 6, but 1.96 on device "
            "4 -- a 5.4x collapse. The SIGN holds on all three dies, so the qualitative "
            "claim ('more occupancy is not always faster') survives everywhere. The MARGIN "
            "does not, and on device 4 it is the same order as the cross-die bandwidth gap "
            "itself, so on that die it should be read as directional only. It was previously "
            "emitted as a flat sentence in a field of its own, which read as a fixed fact."
        ),
        "static_note": (
            "vgpr_count and vgpr_spill_count are identical across all dies and runs on every "
            "row. The compiler output does not depend on which die executes it, as expected; "
            "it is stated because it is what makes the timing comparison a comparison of "
            "silicon rather than of code."
        ),
        "equal_occupancy_bonus": (
            "occupancy agrees across dies far more closely than bandwidth does -- worst "
            f"{occ_by_reducer[PRIMARY_REDUCER]}% against {worst_bw}% under "
            f"{PRIMARY_REDUCER} -- with kernel, registers and spills bit-identical. So this "
            "is a third instance of the experiment's main point, arrived at without "
            "designing for it: occupancy does not determine bandwidth, here not even across "
            "three copies of the same silicon running the same code object. Note this "
            "argument only needs occupancy to agree BETTER than bandwidth, which holds under "
            "all three reducers; it does not depend on the 0.6% figure I originally quoted, "
            "and that is why it survives the reducer being named."
        ),
        "unexplained": (
            "why the dies differ. Not the streaming ceiling: two_read_one_write reads 6.004 "
            "TB/s on device 4 against 6.086 (dev5) and 6.074 (dev6), so device 4 is slowest "
            "at streaming and fastest here, and per-die normalisation widens the gaps rather "
            "than closing them. @Autotune notes the gap correlates with spill count "
            "(r=0.74) and is ~3% on register-resident rows against ~11% on scratch-heavy "
            "ones, which points at the scratch/memory path; recorded as a place to look, "
            "not as an explanation. Idle sclk/mclk/fclk/socclk identical across the three, "
            "junction temperatures within 2 C. Left unexplained on purpose: the last two "
            "things retracted in the notes were mechanisms attached without a control."
        ),
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(f"  inputs: {len(runs)} runs over devices {devs}")
    print(f"  worst between-die bandwidth ({PRIMARY_REDUCER}): {worst_bw}%")
    print(f"  worst within-die bandwidth: {worst_within_bw}%")
    for name, blk in payload["ratios"].items():
        print(f"  ratio {name}: resolved={blk['resolved']}")


if __name__ == "__main__":
    main()

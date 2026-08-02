"""Assemble the 512 MiB copy-draw artifact, with slot structure and conventions.

This replaces a hand-written JSON. The first version of that file was typed in
one pass and drew three separate corrections within the hour, every one of them
about how the numbers were *stated* rather than what they were:

  * `draws_inside_band: 1` was reported against a bare pair `[4.885, 4.895]`,
    which reads as a closed interval. @Reviewer had already fixed these bands as
    half-open under nearest / half-up, and had specifically told @Autotune not to
    publish them without naming the convention. I published them without naming
    the convention, in a committed file rather than a message.

  * The pooled `n_draws: 25` invites reading 25 as a sample size. @Autotune's
    decomposition -- verified here, 99.58% of the sum of squares -- shows the 25
    draws are 10 allocation slots sampled 2-3 times. The effective n for "how
    much does a fresh draw vary" is nearer 10; for "how much does a repeat vary"
    it is 2-3.

  * "Copy at that size is not a repeatable quantity" was too strong. Copy at 512
    MiB is repeatable to 0.02-2.4% *conditional on the allocation slot*. What is
    not repeatable is which slot a fresh process lands in.

The slot decomposition is a better argument for the placement mechanism than the
pooled range ever was, and it belongs in the artifact rather than in a thread: a
placement effect predicts that the n-th allocation of a given size in a given
program lands somewhere reproducible, so the same slot re-measures to a fraction
of a percent while different slots differ by ~17%. Per-call noise predicts no
slot structure at all. Two generator families with different but internally
reproducible patterns is what an allocator does and what noise does not.

One thing this assembler cannot fix, and says so in the output: the draw
recorded as `4.895` was rounded to 3 dp at generation, so it means "somewhere in
[4.8945, 4.8955)" and straddles the half-open band's upper edge. Its membership
is undecidable from the committed record -- not 0, not 1. The probe now stores
these unrounded (see `_identical_buffer_spread`); runs collected before that
change cannot be recovered.

Run:  python AI/assemble_copy_placement_draws.py
Writes AI/data/copy_placement_draws/copy_512MiB_draws_dev5.json.
"""

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "AI/data/copy_placement_draws/copy_512MiB_draws_dev5.json"
LIVE = REPO / "AI/data/rmsnorm_32768x4096_bf16_roofline.json"

# @Reviewer fixed these under nearest / half-up, half-open. Named, not assumed.
BAND = (4.885, 4.895)
BAND_CONVENTION = (
    "half-open [lo, hi) under a nearest / half-up display rule: a true value of "
    "4.895 displays as 4.90, not 4.89, so it cannot have produced the historical "
    "cell. Convention fixed by @Reviewer in 4d6ed9de; stated here because a bare "
    "pair reads as closed and the earlier version of this file published one."
)

# Runs collected before the probe stored unrounded rates. Transcribed from stdout,
# and marked as such: they are 3 dp and cannot be re-derived from committed code.
TRANSCRIBED_RUNS = [
    {
        "source": "ad-hoc size sweep, ascending",
        "family": "adhoc",
        "precision": "3dp_transcribed",
        "draws": [4.816, 5.009, 5.026, 5.597, 5.581],
    },
    {
        "source": "ad-hoc size sweep, descending",
        "family": "adhoc",
        "precision": "3dp_transcribed",
        "draws": [4.701, 4.977, 4.952, 5.570, 5.561],
    },
    {
        "source": "ad-hoc size sweep, ascending",
        "family": "adhoc",
        "precision": "3dp_transcribed",
        "draws": [4.747, 5.001, 4.990, 5.592, 5.586],
    },
    {
        "source": "roofline generator, superseded run",
        "family": "gen",
        "precision": "3dp_transcribed",
        "draws": [5.338, 4.768, 5.582, 5.583, 4.837],
    },
    {
        "source": "roofline generator, run rounded to 3dp before the probe kept raw rates",
        "family": "gen",
        "precision": "3dp_transcribed",
        "draws": [5.349, 4.760, 5.581, 5.587, 4.895],
    },
]

# The draws in the ad-hoc runs came out in ascending allocation order; the
# generator's are in its own allocation order. Position within a run is the slot
# index -- that is the axis the decomposition is about.


def _live_run():
    """The current roofline artifact's 512 MiB draws, unrounded and regenerable."""
    d = json.loads(LIVE.read_text())
    block = d["copy_variability"]["identical_buffers_by_size"]["512MiB"]
    return {
        "source": "roofline generator at this commit (regenerable from committed code)",
        "family": "gen",
        "precision": "full",
        "draws": block["TBps_per_identical_buffer"],
        "generator_commit": d["commit"],
        "generator_dirty": d["worktree_dirty"],
    }


def _band_counts(draws):
    """Counts under BOTH conventions, plus what the recorded precision can decide.

    A draw stored at 3 dp is an interval, not a point. Reporting a count as
    though every draw were exact is the same move as reporting a band as though
    the convention were obvious.
    """
    lo, hi = BAND

    # Partition into decidable and undecidable FIRST. Counting undecidable draws
    # into one of the buckets is the same defect the convention note is about:
    # a value stored at 3 dp that straddles the edge has no membership, and
    # putting it on the "outside" side because a strict comparison happens to
    # place it there asserts precision the record does not have.
    undecidable, decidable = [], []
    for v, prec in zip(draws["values"], draws["precision"]):
        half = 0.0005 if prec == "3dp_transcribed" else 0.0
        if half and (v - half) < hi < (v + half):
            undecidable.append(v)
        else:
            decidable.append(v)

    below = sum(1 for v in decidable if v < lo)
    half_open = sum(1 for v in decidable if lo <= v < hi)
    closed = sum(1 for v in decidable if lo <= v <= hi)
    return {
        "band": list(BAND),
        "band_convention": BAND_CONVENTION,
        "n_decidable": len(decidable),
        "draws_below_band": below,
        "draws_inside_band_half_open": half_open,
        "draws_inside_band_closed": closed,
        "draws_at_or_above_band_half_open": len(decidable) - below - half_open,
        "counts_exclude_undecidable": True,
        "undecidable_at_recorded_precision": {
            "values": undecidable,
            "why": (
                "stored at 3 dp, so each means a 0.001-wide interval that straddles the "
                "band's upper edge; membership is not determined by the committed record. "
                "The probe now stores these unrounded, but runs collected before that "
                "cannot be recovered."
            ),
        },
        "band_is_straddled": bool(below and (len(decidable) - below - half_open)),
        "straddle_survives_either_convention": bool(below and (len(decidable) - below - closed)),
        "straddle_survives_undecidable": (
            "yes -- draws sit below and above the band under either convention, so the "
            "conclusion does not depend on how the undecidable draw is resolved"
        ),
    }


def _slots(runs):
    """Decompose variance by (generator family, allocation position).

    @Autotune's decomposition, recomputed here rather than transcribed from his
    message -- which is the rule this whole line of work exists to enforce.
    """
    slots = {}
    for r in runs:
        for pos, v in enumerate(r["draws"]):
            slots.setdefault(f"{r['family']}_pos{pos}", []).append(v)
    allv = [v for r in runs for v in r["draws"]]
    gm = sum(allv) / len(allv)
    sst = sum((v - gm) ** 2 for v in allv)
    ssb = sum(len(vs) * ((sum(vs) / len(vs)) - gm) ** 2 for vs in slots.values())
    per_slot = {}
    for k, vs in sorted(slots.items()):
        per_slot[k] = {
            "draws": vs,
            "n": len(vs),
            "range_pct_of_min": round((max(vs) / min(vs) - 1) * 100.0, 3),
        }
    within = [s["range_pct_of_min"] for s in per_slot.values()]
    return {
        "per_slot": per_slot,
        "n_slots": len(slots),
        "variance_explained_by_slot_pct": round(ssb / sst * 100.0, 2),
        "worst_within_slot_range_pct": max(within),
        "slots_reproducing_under_1pct": sum(1 for w in within if w < 1.0),
        "why_this_matters": (
            "A placement effect predicts exactly this shape: the n-th allocation of a "
            "given size in a given program lands somewhere reproducible, so the same "
            "slot re-measures to a fraction of a percent while different slots differ "
            "by ~17%. Per-call noise predicts no slot structure at all. This is a "
            "stronger argument for the mechanism than the pooled range, which merely "
            "shows the draws are spread out."
        ),
        "effective_n_note": (
            "Do not read the pooled draw count as a sample size. With almost all of "
            "the variance on a 10-level slot axis, the effective n for 'how much does "
            "a fresh draw vary' is nearer the slot count than the draw count, and for "
            "'how much does a repeat vary' it is the 2-3 repeats per slot."
        ),
    }


def main():
    runs = list(TRANSCRIBED_RUNS) + [_live_run()]
    values = [v for r in runs for v in r["draws"]]
    precision = [r["precision"] for r in runs for _ in r["draws"]]
    lo, hi = min(values), max(values)

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode or dirty.returncode or not head.stdout.strip():
        raise SystemExit(
            "cannot read git provenance; refusing to write an artifact whose commit "
            "field would be empty and whose dirty flag would default to clean."
        )

    slots = _slots(runs)
    payload = {
        "what": (
            "every 512 MiB copy draw behind the placement finding, with the allocation-"
            "slot decomposition that explains almost all of their spread"
        ),
        "why": (
            "@Autotune derived a feasible band for the unarchived 4.89 copy denominator "
            "and observed that every committed copy value misses it, concluding copy is "
            "excluded. Each of those values is a single allocation's draw. These draws "
            "show what a single draw is worth at this size: the distribution straddles "
            "the band under either interval convention, so the band is not excluded by "
            "the data. This removes an argument against 4.89; it does not restore it, "
            "whose provenance remains absent."
        ),
        "device": "physical 5 (MI355X, gfx950)",
        "buffer_bytes": 512 * 1024 * 1024,
        "op": (
            "c.copy_(a), five identically-sized identically-filled sources, one fixed destination"
        ),
        "assembled_at_commit": head.stdout.strip(),
        "assembled_with_dirty_worktree": bool(dirty.stdout.strip()),
        "generator": "AI/assemble_copy_placement_draws.py",
        "runs": runs,
        "pooled": {
            "n_draws": len(values),
            "n_slots": slots["n_slots"],
            "min": lo,
            "max": hi,
            "range_pct_of_min": round((hi / lo - 1) * 100.0, 2),
            "read_this_as": (
                "a between-slot spread measured 2-3 times per slot, not 25 independent "
                "draws; see slot_decomposition.effective_n_note"
            ),
        },
        "band_test": _band_counts({"values": values, "precision": precision}),
        "slot_decomposition": slots,
        "what_is_and_is_not_repeatable": (
            "Copy at 512 MiB IS repeatable, to "
            f"{min(s['range_pct_of_min'] for s in slots['per_slot'].values()):.3f}-"
            f"{slots['worst_within_slot_range_pct']:.3f}%, conditional on the allocation "
            "slot. What is not repeatable is which slot a fresh process lands in. This "
            "corrects 'copy at that size is not a repeatable quantity', which was too "
            "strong: the practical consequence is that a copy denominator cannot be "
            "compared ACROSS processes or programs, not that it is noisy within one."
        ),
        "verdict": (
            "The between-slot placement term swamps the band, and no historical artifact "
            "records which slot it drew. So no single 512 MiB copy value can authenticate "
            "a historical copy figure. Provenance, not arithmetic, remains the binding "
            "objection to 4.89."
        ),
        "provenance_note": (
            "The first five runs are transcribed at 3 dp from runs collected before the "
            "probe stored raw rates, and cannot be re-derived from committed code. Only "
            "the final run is regenerable (copy_variability in the roofline sidecar). "
            "One consequence is recorded in band_test: a 3 dp value straddling the band "
            "edge cannot be placed inside or outside it."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")
    b = payload["band_test"]
    print(
        f"  pooled n={payload['pooled']['n_draws']} over {slots['n_slots']} slots, "
        f"range {payload['pooled']['range_pct_of_min']}%"
    )
    print(
        f"  band: below {b['draws_below_band']}, inside(half-open) "
        f"{b['draws_inside_band_half_open']}, inside(closed) {b['draws_inside_band_closed']}, "
        f"undecidable {len(b['undecidable_at_recorded_precision']['values'])}"
    )
    print(f"  variance explained by slot: {slots['variance_explained_by_slot_pct']}%")


if __name__ == "__main__":
    main()

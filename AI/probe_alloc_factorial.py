"""Cross prior-allocation COUNT with prior-allocation BYTES, above the measurement's own live set.

The staircase probe (`AI/probe_copy_size_draws.py`) varies one thing and calls
it two: its prefix allocates `n` buffers of 512 MiB, so count and peak bytes
move together, `bytes = n * 512 MiB`. Every step it finds is a step in both at
once. Worse, all its prefixes (0..21 -> 0..10.5 GiB) sit BELOW the 12 GiB live
set the measurement itself allocates, so the process high-water is 12 GiB in
every condition -- peak bytes does not actually vary across the treatment at
all, and the axis was renamed `n_prior_512mib_allocs` for exactly that reason.

@Reviewer's standing objection, 5c2e0083 and f3ee3e18: nothing in that design
can attribute the steps to bytes rather than to count, and the artifact should
not use "partial factorial" or "separable" language while they move together.
This probe is the separation.

Design
------
Two factors crossed, four cells, plus a size axis:

    count in {24, 48}  x  buffer size in {512 MiB, 1 GiB}

    cell         count   each      prefix high-water
    lo_lo           24   512 MiB        12.0 GiB
    lo_hi           24   1024 MiB       24.0 GiB
    hi_lo           48   512 MiB        24.0 GiB
    hi_hi           48   1024 MiB       48.0 GiB

`lo_hi` and `hi_lo` are the cell pair that does the work: **the same 24 GiB of
peak prior bytes reached with 24 allocations or with 48**. If the effect is
about bytes they agree; if it is about count they differ. `lo_lo` and `hi_hi`
anchor the ends so a null between the diagonal cells cannot be read as the
whole grid being flat.

Every prefix here reaches at least 12 GiB, so unlike the staircase sweep the
process high-water genuinely varies across cells (12 / 24 / 24 / 48 GiB) rather
than being pinned by the measurement. That was the design flaw named in
`probe_copy_size_draws.py`'s own module comment and never fixed.

What this CANNOT do
-------------------
It does not explain why any level matters. It separates two candidate axes; it
does not identify a mechanism, and no ordering or factorial control can. The
measurement is imported from the roofline probe, so "copy rate across identical
buffers" has one definition in this tree.

Ordering
--------
One fresh process per row, and the (cell, repeat) visiting order is a single
seeded shuffle -- not blocked by cell, not blocked by repeat. Blocking either
one reintroduces exactly the confound the staircase work spent three iterations
removing: level as a function of collection position. The seed is fixed and
recorded so the order is a property of the artifact rather than of the run.

The anchored run
----------------
`--anchored` adds the two cells the first run's artifact declared as its
follow-up: count=0 and count=13, both at 512 MiB, all six cells in ONE shuffle
(24 processes). count=0 supplies what the four-cell grid could not -- internal
evidence that the prefix moves the rate AT ALL, without which a null across the
grid is ambiguous between "neither factor matters" and "the curve is flat above
21 and this probe never left the plateau". count=13 sits on the first staircase
step, inside the 6.5-10.5 GiB region the four-cell grid started above.

Both anchors deliberately break the four-cell grid's own precondition: at 0 and
6.5 GiB they sit BELOW the measurement's 12.0 GiB live set, so their process
high-water is pinned by the measurement rather than by the prefix. That is the
staircase sweep's limitation, reintroduced knowingly, and it bounds what the
anchors can say -- "does the prefix matter at all", not "here are two more
points on the bytes curve".

The six cells are re-collected rather than pooled with the existing sixteen
rows. Those are a different session, and the entire reason this design shuffles
is that collection position is a confound; rows from two sessions cannot be
interleaved into one shuffle after the fact.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_alloc_factorial.py OUT.json
      HIP_VISIBLE_DEVICES=<idle> python AI/probe_alloc_factorial.py OUT.json --anchored
Assembled by AI/assemble_alloc_factorial.py.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MIB = 1024 * 1024

# (label, count, buffer_mib). The two middle cells are the comparison; the outer
# two exist so a null on the diagonal is not mistaken for a flat grid.
CELLS = (
    ("lo_lo", 24, 512),
    ("lo_hi", 24, 1024),
    ("hi_lo", 48, 512),
    ("hi_hi", 48, 1024),
)

# The anchor cells declared in the first run's artifact, added under --anchored
# rather than appended to CELLS. Two reasons for the flag rather than an edit.
#
# The four-cell run stays reproducible: `python AI/probe_alloc_factorial.py out`
# still means in this revision exactly what it meant in the revision that
# produced alloc_factorial.json. Editing CELLS in place would leave the
# committed artifact describing a grid the file no longer defines, and the
# artifact's own manifest hashes this source -- so the mismatch would surface as
# a hash that no longer matches rather than as anything a reader could act on.
#
# And the two runs are NOT poolable. Different session, different process set,
# and the whole reason the first grid shuffles is that collection position is a
# confound; rows from two sessions cannot be interleaved after the fact. The
# anchored run therefore re-collects all six cells in ONE shuffle rather than
# reusing the existing sixteen rows, which costs 24 processes instead of 8 and
# is the only version that supports a within-run comparison.
ANCHOR_CELLS = (
    ("zero", 0, 512),
    ("step", 13, 512),
)
REPS = 4
SHUFFLE_SEED = 20260803

# A distinct seed for the anchored run. Reusing SHUFFLE_SEED would give the six
# cells an order deterministically related to the four-cell order, which is not
# wrong but makes "the two runs disagree" and "the two orders were correlated"
# harder to separate than they need to be.
ANCHOR_SHUFFLE_SEED = 20260805

# The measurement's own live set, in GiB, from _identical_buffer_spread: six
# 2 GiB buffers (one dst + five srcs) for the copy pattern. Recorded so the
# artifact can assert every prefix exceeded it rather than asking a reader to
# take it on faith.
MEASUREMENT_LIVE_GIB = 12.0


def _measure(count, buffer_mib):
    """Run the prefix, then the standard 2 GiB identical-buffer measurement.

    Imported, not reimplemented: `_identical_buffer_spread` is the same function
    the committed copy artifacts use, so this probe cannot drift from them.
    """
    import torch

    from AI.probe_copy_size_draws import _identical_buffer_spread

    torch.cuda.reset_peak_memory_stats()
    n = (buffer_mib * MIB) // 4
    live = [torch.empty(n, device="cuda", dtype=torch.float32).fill_(1.0) for _ in range(count)]
    prefix_hw = torch.cuda.max_memory_allocated()
    del live
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    spread = _identical_buffer_spread(2048)
    measure_hw = torch.cuda.max_memory_allocated()

    return {
        "prefix_high_water_bytes": prefix_hw,
        "prefix_high_water_GiB": round(prefix_hw / 2**30, 3),
        "measurement_high_water_GiB": round(measure_hw / 2**30, 3),
        "TBps_per_slot": spread["TBps_per_identical_buffer"],
        "rounds_us_per_slot": spread["rounds_us_per_identical_buffer"],
        # The whole returned dict, not a chosen projection of it. The first
        # draft of this function picked two fields out of `spread` -- and picked
        # them from a shape the function does not have, which is how the error
        # surfaced at all. Had I guessed the key names correctly it would have
        # run green while silently discarding `within_buffer_spread_pct` and
        # `spread_pct_of_min`, the two fields that separate a genuinely slow
        # slot from one bad round. That is the exact distinction @Reviewer's
        # 23a6f662 objection was about, re-lost one file over. Keeping the
        # source dict costs nothing and makes the projection above a
        # convenience rather than a decision about what will matter later.
        "spread": spread,
    }


def _worker(label, count, buffer_mib):
    r = _measure(count, buffer_mib)
    r.update({"cell": label, "count": count, "buffer_mib": buffer_mib})
    print(json.dumps(r))


def _sweep(out_path, anchored=False):
    cells = CELLS + ANCHOR_CELLS if anchored else CELLS
    seed = ANCHOR_SHUFFLE_SEED if anchored else SHUFFLE_SEED
    order = [(c, r) for c in cells for r in range(REPS)]
    random.Random(seed).shuffle(order)

    rows = []
    for seq, ((label, count, buffer_mib), rep) in enumerate(order):
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "-",
                "--cell",
                label,
                "--count",
                str(count),
                "--buffer-mib",
                str(buffer_mib),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO),
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"worker for {label} rep {rep} failed ({proc.returncode}): {proc.stderr[-2000:]}"
            )
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        row.update({"seq": seq, "rep": rep})
        rows.append(row)
        print(f"[{seq + 1}/{len(order)}] {label} rep{rep}", file=sys.stderr, flush=True)

    import torch

    payload = {
        "what": (
            "Prior-allocation COUNT crossed with prior-allocation BYTES, with every "
            "prefix above the 12 GiB live set the measurement itself allocates."
        ),
        "why": (
            "the staircase sweep varies count and bytes together (bytes = n * 512 MiB) "
            "and its prefixes all sit below the measurement's own high-water, so peak "
            "bytes does not vary across its treatment at all. lo_hi and hi_lo here "
            "reach the same 24 GiB of prior peak with 24 and 48 allocations."
        ),
        "device_name": torch.cuda.get_device_name(0),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "cells": [{"cell": c, "count": n, "buffer_mib": b} for c, n, b in cells],
        "reps_per_cell": REPS,
        "one_process_per_row": True,
        "shuffle_seed": seed,
        "why_shuffled": (
            "the (cell, repeat) order is one seeded shuffle, not blocked by cell or by "
            "repeat. Blocking either reintroduces level-as-a-function-of-collection-"
            "position, which is the confound the staircase work needed three iterations "
            "to remove."
        ),
        "anchored": anchored,
        "anchor_note": (
            (
                "six cells in ONE shuffle: the original four plus count=0 and count=13 "
                "at 512 MiB, pre-registered in alloc_factorial_assembled.json's "
                "anchor_limitation.declared_followup and in slot_structure's "
                "declared_followup_for_the_anchor_run before these numbers existed. "
                "count=0 supplies the missing internal evidence that the prefix moves the "
                "rate at all; count=13 sits on the first staircase step, inside the "
                "6.5-10.5 GiB region the four-cell grid started above. All six are "
                "re-collected here rather than pooled with the earlier sixteen rows, "
                "which are a different session and cannot be interleaved into one shuffle "
                "after the fact."
            )
            if anchored
            else "four-cell run; anchors not included"
        ),
        "measurement_live_set_GiB": MEASUREMENT_LIVE_GIB,
        "the_precondition_this_run_knowingly_breaks": (
            (
                "count=0 allocates nothing and count=13 reaches 6.5 GiB, so both sit "
                "BELOW the 12.0 GiB live set the measurement itself allocates. The "
                "four-cell grid's every_prefix_above_measurement_live_set precondition is "
                "therefore false here BY DESIGN, not by oversight -- the anchors exist "
                "precisely to sample below where the original grid could not. Their "
                "process high-water is pinned by the measurement rather than by the "
                "prefix, which is the same limitation the staircase sweep had and the "
                "reason its bytes axis was renamed. Anchors are interpretable as 'does "
                "the prefix matter at all', NOT as additional points on the bytes curve."
            )
            if anchored
            else None
        ),
        "rows": rows,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cell", help=argparse.SUPPRESS)
    ap.add_argument("--count", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--buffer-mib", type=int, help=argparse.SUPPRESS)
    ap.add_argument(
        "--anchored",
        action="store_true",
        help="add the pre-registered count=0 and count=13 anchor cells (24 processes)",
    )
    args = ap.parse_args()

    if args.cell is not None:
        _worker(args.cell, args.count, args.buffer_mib)
        return
    _sweep(args.out, anchored=args.anchored)


if __name__ == "__main__":
    main()

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

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_alloc_factorial.py OUT.json
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
REPS = 4
SHUFFLE_SEED = 20260803

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


def _sweep(out_path):
    order = [(c, r) for c in CELLS for r in range(REPS)]
    random.Random(SHUFFLE_SEED).shuffle(order)

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
        "cells": [{"cell": c, "count": n, "buffer_mib": b} for c, n, b in CELLS],
        "reps_per_cell": REPS,
        "one_process_per_row": True,
        "shuffle_seed": SHUFFLE_SEED,
        "why_shuffled": (
            "the (cell, repeat) order is one seeded shuffle, not blocked by cell or by "
            "repeat. Blocking either reintroduces level-as-a-function-of-collection-"
            "position, which is the confound the staircase work needed three iterations "
            "to remove."
        ),
        "measurement_live_set_GiB": MEASUREMENT_LIVE_GIB,
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
    args = ap.parse_args()

    if args.cell is not None:
        _worker(args.cell, args.count, args.buffer_mib)
        return
    _sweep(args.out)


if __name__ == "__main__":
    main()

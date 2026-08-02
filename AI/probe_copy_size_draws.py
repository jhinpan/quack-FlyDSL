"""Measure what actually moves the copy rate: slot, process, or program.

`AI/flydsl_rmsnorm_notes.md:869` says the three-pattern roofline table that
produced `4.89`, `6.09`, `6.84` was "probing three patterns over **2 GiB**
buffers". Two problems followed from that line, and this probe exists because I
had missed both while arguing at length about that `4.89`:

  * The confound I used to retire @Autotune's band-exclusion argument -- the
    19.06% pooled range, the 10-slot decomposition, the band straddle -- was
    measured entirely at **512 MiB**. The placement effect is strongly
    size-dependent, so a confound magnitude measured at 512 MiB is not evidence
    about a number measured at 2 GiB. It has to be measured at the size the
    claim is about.

  * `AI/probe_rmsnorm_roofline.py` runs its three `roofline_probes` at
    `EVICT_BYTES` = **512 MiB**, and prints 4.95 / 6.00 / 6.89 against that
    table's 4.89 / 6.09 / 6.84. Close enough to read as a reproduction of it.
    It is not one -- it is a different buffer size, agreeing to within the
    confound. A check passing for a reason other than the one it documents.

The three axes, which the earlier work conflated:

  slot     -- which of several identically-sized buffers within one process.
              Dominant: 98.7-99.9% of the sum of squares at both sizes.
  process  -- re-running the *same* program, byte for byte. Nearly nothing: a
              given slot reproduces across processes to under 1% at both sizes.
  program  -- changing the allocator's peak high-water mark, i.e. editing the
              harness. This moves the number, and it is the axis that was
              previously mislabelled as "process".

That last distinction is why this probe takes `--peak-live-512mib`. The
published field `denominator_stability_across_processes` reported copy at 13.35%
with `n_processes: 3`; those three runs straddled edits to `_copy_variability`,
which allocates and frees buffers *before* the copy probe runs. Holding the
generator fixed, copy reproduces to 0.63% over six processes. The 13.35% was a
spread across generator versions wearing a process label -- and the fix for
hand-typed constants is what introduced it.

What moves it is *peak simultaneously-live bytes*, not churn: 0, 6 and 11 live
512 MiB buffers give indistinguishable results, then 13 and 17 each shift the
2 GiB slot values to a different reproducible set, and coming back down restores
the old ones. A staircase in the high-water mark, reversible. This is why
"allocation history does nothing" and "editing the harness moved it 13%" are
both true and were never in conflict: the history rows never changed the peak.

The measurement is imported from the roofline probe rather than reimplemented,
so there is one definition of "copy rate across identical buffers" in the tree
and it cannot drift from the one that produced the committed sidecar.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_copy_size_draws.py OUT.json [--prefix NAME]
Collected in bulk by AI/collect_copy_axes.sh; assembled by AI/assemble_copy_axes.py.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from AI.probe_rmsnorm_roofline import (
    _bench,
    _identical_buffer_spread,
    _summarise,
)

SIZES_MIB = (512, 2048)

# Allocation prefixes, expressed as the peak number of simultaneously-live
# 512 MiB buffers -- then all freed and `empty_cache()`d before anything is
# measured. Peak, not total: churn at a low water mark does nothing, while
# crossing certain thresholds shifts the 2 GiB slot values to a different,
# reproducible set. A staircase, and reversible -- coming back down restores the
# earlier values.
#
# This is the mechanism behind what was published as a cross-process spread.
# `_copy_variability` in the current generator peaks at 5x2048 MiB = 20 units;
# at 7de8da4 the size sweep did not exist and it peaked below the first step.
# Prefix 20 reproduces the current generator's 2 GiB slot means and prefix 0
# reproduces the old one's, so the "13.35% across 3 processes" was the
# allocator's high-water mark moving under an edit to the harness.
#
# The default set brackets the two generator versions. `--sweep-steps` walks a
# fine grid instead, so the step *locations* are a measured artifact field
# rather than a remembered pair of integers: the first draft of the notes
# asserted steps "at 13 and again at 17" from an exploratory script in /tmp,
# which is a number living only in prose -- the defect this whole line of work
# is about.
PREFIX_PEAK_LIVE_512MIB = (0, 6, 14, 20)
SWEEP_STEPS = tuple(range(22))
SWEEP_REPS = 2


def _run_prefix(peak):
    if not peak:
        return
    live = [
        torch.empty((512 * 1024 * 1024) // 4, device="cuda", dtype=torch.float32).fill_(1.0)
        for _ in range(peak)
    ]
    del live
    torch.cuda.empty_cache()


def _pattern_rates(mib):
    """The three roofline patterns at one buffer size, each against its traffic.

    Mirrors `_probes()` in the roofline generator, which hard-codes 512 MiB via
    EVICT_BYTES. Here the size is a parameter, which is the point: the table
    these rates get compared against was taken at 2 GiB.

    Five source slots per pattern, not one. `two_read_one_write` looked like a
    stable size discriminator at 0.31% until it was measured this way -- across
    slots it spreads 4.28% at 512 MiB, which is the same magnitude as the
    size gap it was being used to resolve. Measuring one slot and calling the
    result the instrument's error bar is how that nearly got published.
    """
    n = (mib * 1024 * 1024) // 4
    buf = n * 4
    dst = torch.empty(n, device="cuda", dtype=torch.float32)
    srcs_a = [torch.empty(n, device="cuda", dtype=torch.float32).fill_(1.0) for _ in range(5)]
    srcs_b = [torch.empty(n, device="cuda", dtype=torch.float32).fill_(2.0) for _ in range(5)]

    # Every closure binds its tensors as default arguments, and nothing in this
    # scope deletes a name a closure reads. The first draft did both -- built the
    # lambdas over `c`/`srcs_a`/`srcs_b` and then `del`d them -- which ruff
    # flagged F821 and was right to. It produced correct numbers because `slots`
    # runs eagerly, so the names were still bound when it mattered. That is
    # precisely the shape `_identical_buffer_spread` was already refactored to
    # remove, and it came straight back the moment I wrote a similar helper:
    # correct by evaluation order, with nothing in the code saying so.
    # Keeps every round, not just the derived rate. @Reviewer's objection to the
    # previous artifact was that "full precision" meant one number per buffer
    # while the seven timing rounds behind it were thrown away -- so a reader
    # cannot tell a slot that is genuinely slower from one that caught a single
    # bad round, which is exactly the distinction the slot argument rests on.
    # `_summarise` already computes them; discarding them was free to avoid.
    def slots(make_call, nbytes):
        out = []
        for i in range(5):
            s = _summarise(_bench(make_call(i)), nbytes)
            out.append(
                {
                    "TBps_at_min": s["TBps_at_min"],
                    "samples_us": s["samples_us"],
                    "spread_pct_of_min": s["spread_pct_of_min"],
                    "bytes_moved": s["bytes_moved"],
                }
            )
        return out

    return {
        "copy": slots(lambda i: lambda d=dst, s=srcs_a[i]: d.copy_(s), 2 * buf),
        "two_read_one_write": slots(
            lambda i: lambda a=srcs_a[i], b=srcs_b[i], d=dst: torch.add(a, b, out=d), 3 * buf
        ),
        "write": slots(lambda i: lambda d=dst, v=float(i + 1): d.fill_(v), buf),
        "buffer_bytes": buf,
    }


def _sweep_steps(out_path):
    """Locate the staircase steps: one FRESH process per peak level.

    The first version of this walked the grid inside a single process, which was
    invalid and produced a confidently wrong result: a ~10% shift at every
    single step including 0 -> 1, alternating between two slot patterns, i.e. no
    threshold at all. The reason is that the measurement is not passive. Calling
    `_identical_buffer_spread` allocates and frees six 2 GiB buffers, which
    re-rolls the placement on its own -- repeating the identical call eight times
    in one process, changing nothing else, gives a different slot pattern every
    time. So the sweep confounded "the peak changed" with "one more alloc/free
    cycle happened", and the second effect swamped the first.

    That is worth keeping in the file rather than quietly deleting, because the
    broken design looked *more* rigorous: walking up and back down to test
    reversibility, in one process to control for process-level variation. It
    controlled away the wrong thing. Reversibility now has to be read from
    repeats at the same level across processes, which is weaker evidence, and
    the file says so instead of implying a within-process A-B-A.

    Each level is measured `SWEEP_REPS` times in separate processes, so a step
    is only a step if it exceeds the spread of the repeats.
    """
    import sys as _sys

    rows = []
    for peak in SWEEP_STEPS:
        for rep in range(SWEEP_REPS):
            proc = subprocess.run(
                [_sys.executable, __file__, "-", "--peak-live-512mib-any", str(peak)],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ},
            )
            row = json.loads(proc.stdout)
            rows.append(
                {
                    "peak_live_512mib": peak,
                    "rep": rep,
                    "draws_2GiB": row["draws_2GiB"],
                    "rounds_us_2GiB": row["rounds_us_2GiB"],
                }
            )
    out_path.write_text(
        json.dumps(
            {
                "device_name": torch.cuda.get_device_name(0),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                "what": (
                    "2 GiB slot values by allocator peak high-water mark, one fresh "
                    "process per level per repeat"
                ),
                "one_process_per_row": True,
                "reps_per_level": SWEEP_REPS,
                "why_not_within_process": (
                    "the measurement itself allocates and frees 2 GiB buffers, which "
                    "re-rolls placement; a within-process sweep measures that instead of "
                    "the peak and reports a spurious step at every level"
                ),
                "rows": rows,
            }
        )
        + "\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument(
        "--peak-live-512mib",
        type=int,
        default=0,
        choices=PREFIX_PEAK_LIVE_512MIB,
        help="peak simultaneously-live 512 MiB buffers, all freed before measuring",
    )
    ap.add_argument(
        "--sweep-steps",
        action="store_true",
        help="one fresh process per peak level over a fine grid, to locate the steps",
    )
    ap.add_argument(
        "--peak-live-512mib-any",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # worker mode for --sweep-steps; prints one row to stdout
    )
    args = ap.parse_args()

    if args.peak_live_512mib_any is not None:
        _run_prefix(args.peak_live_512mib_any)
        block = _identical_buffer_spread(2048)
        # Rounds, not just the derived rates: a "step" in the staircase is only a
        # step if it clears the round-to-round noise of a single draw, and the
        # sweep cannot be read for that unless it carries the rounds along.
        print(
            json.dumps(
                {
                    "draws_2GiB": block["TBps_per_identical_buffer"],
                    "rounds_us_2GiB": block["rounds_us_per_identical_buffer"],
                }
            )
        )
        return

    if args.sweep_steps:
        return _sweep_steps(Path(args.out))

    _run_prefix(args.peak_live_512mib)
    payload = {
        "device_name": torch.cuda.get_device_name(0),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "peak_live_512mib": args.peak_live_512mib,
        "identical_buffers_by_size": {},
        "roofline_patterns_by_size": {},
    }
    for mib in SIZES_MIB:
        payload["identical_buffers_by_size"][f"{mib}MiB"] = _identical_buffer_spread(mib)
        torch.cuda.empty_cache()
        payload["roofline_patterns_by_size"][f"{mib}MiB"] = _pattern_rates(mib)
    Path(args.out).write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()

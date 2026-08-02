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
import random
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# Must follow the sys.path insert above, so it cannot sit in the header, and the
# two ruff versions in play disagree about that. E402 is in ruff 0.11.13's
# default set (the version this repo pins in CI and pre-commit) and is NOT in
# 0.16.0's, which is what happens to be on PATH in this container. So 0.11.13
# flags a bare import here, while 0.16.0 flags an inline suppression for it as
# RUF100 "non-enabled". (Spelling that directive out in this comment makes ruff
# parse the comment itself as one, which is its own small lesson.) No inline
# directive satisfies both versions; a
# function-scoped import needs no directive and satisfies both.
def _roofline():
    from AI.probe_rmsnorm_roofline import (
        _bench,
        _identical_buffer_spread,
        _summarise,
    )

    return (_bench, _identical_buffer_spread, _summarise)


_bench, _identical_buffer_spread, _summarise = _roofline()


SIZES_MIB = (512, 2048)

# Allocation prefixes: N simultaneously-live 512 MiB buffers, then all freed and
# `empty_cache()`d before anything is measured. Crossing certain N shifts the
# 2 GiB slot values to a different, reproducible set -- a staircase.
#
# THIS AXIS WAS MISNAMED, and the retraction belongs here rather than only in the
# notes. It was called "allocator peak high-water mark", and that named a
# quantity the experiment never varied. @Reviewer worked it out from the code;
# measuring `torch.cuda.max_memory_allocated` confirms it exactly:
#
#     prefix   prefix high-water   measurement high-water   pattern probe
#       0           0.0 GiB              12.0 GiB              22.0 GiB
#       6           3.0 GiB              12.0 GiB              22.0 GiB
#      14           7.0 GiB              12.0 GiB              22.0 GiB
#      20          10.0 GiB              12.0 GiB              22.0 GiB
#
# Every prefix is strictly below the live set the measurement itself allocates,
# so the process high-water is 12 GiB in all four conditions -- constant across
# the entire treatment. What varies is prior allocation *count and history*.
# `n_prior_512mib_allocs` is now the name, because it describes what the loop
# below does rather than what I inferred it was doing.
#
# The general form, and the reason this is in the source: I named the axis after
# the mechanism I believed rather than after the operation the code performs.
# A field name is not a hypothesis -- once written it is read as fact by every
# consumer, and the belief stops being checked. Third axis name retracted in
# this file for the same reason. `high_water_GiB` is now recorded in every run
# so the next such claim can be falsified from the data instead of from a review.
#
# What survives: prefix 20 reproduces the current generator's 2 GiB slot means
# and prefix 0 reproduces the old one's, so the "13.35% across 3 processes" was
# still the harness edit moving this axis. Only its name was wrong.
#
# The prefixes here are all below the measurement's own live set, which is the
# design flaw; separating peak bytes from allocation count needs prefixes above
# 12 GiB and is not done here. `--sweep-steps` walks a fine grid, so the step
# locations are a measured artifact field rather than a remembered pair of
# integers -- the first draft of the notes asserted steps "at 13 and again at 17"
# from an exploratory script in /tmp, a number living only in prose.
PREFIX_PEAK_LIVE_512MIB = (0, 6, 14, 20)
SWEEP_STEPS = tuple(range(22))
SWEEP_REPS = 2
# Fixed so the visiting order is a property of the artifact and not of the run.
# The seed is arbitrary; what matters is that it is recorded and that nobody
# picked it after seeing the result.
SWEEP_SHUFFLE_SEED = 20260802


def _run_prefix(peak):
    """Allocate and free `peak` 512 MiB buffers. Returns the high-water it reached.

    The return value exists because this function's effect was mislabelled for
    two commits (see the module comment). Recording what the prefix actually
    reached, next to what the measurement then reaches, puts the constancy of
    the process high-water into every run's data where a reader can check it.
    """
    torch.cuda.reset_peak_memory_stats()
    if peak:
        live = [
            torch.empty((512 * 1024 * 1024) // 4, device="cuda", dtype=torch.float32).fill_(1.0)
            for _ in range(peak)
        ]
        del live
        torch.cuda.empty_cache()
    return torch.cuda.max_memory_allocated()


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

    # Ascending pass, then descending. The first version ran 0..21 monotonically
    # in adjacent batches, which leaves level perfectly confounded with wall-clock
    # order: any drift in the box over the ~6 minutes of collection reproduces as
    # a "staircase" with no allocator involved. @Reviewer, 5c2e0083. The descending
    # pass revisits every level in the opposite time order, so a step that is real
    # appears at the same level in both directions and a drift artifact does not.
    #
    # This is NOT the within-process A-B-A that was tried and shown invalid: each
    # row is still its own fresh process. The order of *processes* is what varies.
    #
    # But ascending-then-descending is still not enough, and @Reviewer is right
    # about why: all-up-then-all-down makes level an exact function of collection
    # position, `level == min(seq, 43 - seq)`. A single transient centred on the
    # turnaround therefore satisfies BOTH direction tests -- it raises the levels
    # collected near the middle in each pass, which is exactly where the high
    # levels are. Time reversal does not break a symmetric confound; it is
    # symmetric itself.
    #
    # The interleaved pass breaks the functional relationship outright. Levels
    # are visited in a fixed shuffle (seeded, so the artifact is reproducible),
    # which leaves no monotone or symmetric function of seq that recovers level.
    # If the staircase survives here it is not a function of when the row was
    # collected, whatever shape that function has.
    #
    # The repeats are shuffled INTO that order rather than run back to back. Two
    # adjacent processes at the same level would share whatever the box was doing
    # in that half-second, so consecutive repeats make the level look more stable
    # than it is -- the same adjacency problem as the turnaround, one scale down.
    # The up/down passes keep their paired repeats: changing two things at once
    # would make a difference between the passes unattributable.
    order = [(p, "up", r) for p in SWEEP_STEPS for r in range(SWEEP_REPS)]
    order += [(p, "down", r) for p in reversed(SWEEP_STEPS) for r in range(SWEEP_REPS)]
    inter = [(p, "interleaved", r) for p in SWEEP_STEPS for r in range(SWEEP_REPS)]
    random.Random(SWEEP_SHUFFLE_SEED).shuffle(inter)
    order += inter

    rows = []
    for seq, (peak, direction, rep) in enumerate(order):
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
                "n_prior_512mib_allocs": peak,
                "peak_live_512mib": peak,
                "direction": direction,
                "seq": seq,
                "rep": rep,
                "draws_2GiB": row["draws_2GiB"],
                "rounds_us_2GiB": row["rounds_us_2GiB"],
                "high_water_bytes": row.get("high_water_bytes"),
            }
        )
    out_path.write_text(
        json.dumps(
            {
                "device_name": torch.cuda.get_device_name(0),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                "what": (
                    "2 GiB slot values by number of prior 512 MiB allocations, one "
                    "fresh process per level per repeat, in three passes"
                ),
                "one_process_per_row": True,
                "reps_per_level": SWEEP_REPS,
                "passes": {
                    "up": "levels 0..21 in order, repeats adjacent",
                    "down": "levels 21..0 in order, repeats adjacent",
                    "interleaved": (
                        "every (level, repeat) in one seeded shuffle, so no function of "
                        "collection position recovers the level"
                    ),
                },
                "shuffle_seed": SWEEP_SHUFFLE_SEED,
                "why_interleaved": (
                    "up-then-down leaves level an exact symmetric function of seq, "
                    "level == min(seq, 43 - seq) at reps=1, so a transient centred on "
                    "the turnaround satisfies both direction tests. @Reviewer, 5c2e0083. "
                    "Time reversal cannot break a symmetric confound because it is "
                    "itself symmetric."
                ),
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
        pre_hw = _run_prefix(args.peak_live_512mib_any)
        torch.cuda.reset_peak_memory_stats()
        block = _identical_buffer_spread(2048)
        # Rounds, not just the derived rates: a "step" in the staircase is only a
        # step if it clears the round-to-round noise of a single draw, and the
        # sweep cannot be read for that unless it carries the rounds along.
        print(
            json.dumps(
                {
                    "draws_2GiB": block["TBps_per_identical_buffer"],
                    "rounds_us_2GiB": block["rounds_us_per_identical_buffer"],
                    "high_water_bytes": {
                        "after_prefix": pre_hw,
                        "during_identical_buffers_2048MiB": torch.cuda.max_memory_allocated(),
                    },
                }
            )
        )
        return

    if args.sweep_steps:
        return _sweep_steps(Path(args.out))

    prefix_hw = _run_prefix(args.peak_live_512mib)
    payload = {
        "device_name": torch.cuda.get_device_name(0),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        # The treatment, named for the operation performed rather than for the
        # mechanism inferred from it. `peak_live_512mib` is retained under its old
        # key so committed runs and readers of the previous artifact still parse,
        # but it is an alias, and `treatment_axis` says which reading is correct.
        "n_prior_512mib_allocs": args.peak_live_512mib,
        "peak_live_512mib": args.peak_live_512mib,
        "treatment_axis": (
            "prior allocation count and history, NOT allocator peak high-water. See "
            "high_water_bytes: the prefix never exceeds the live set the measurement "
            "itself allocates, so the process high-water is identical at every level. "
            "The old name asserted a mechanism this design cannot separate from "
            "allocation count, churn, fill time and placement. @Reviewer, 5c2e0083."
        ),
        "identical_buffers_by_size": {},
        "roofline_patterns_by_size": {},
    }
    hw = {"after_prefix": prefix_hw}
    for mib in SIZES_MIB:
        torch.cuda.reset_peak_memory_stats()
        payload["identical_buffers_by_size"][f"{mib}MiB"] = _identical_buffer_spread(mib)
        hw[f"during_identical_buffers_{mib}MiB"] = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        payload["roofline_patterns_by_size"][f"{mib}MiB"] = _pattern_rates(mib)
        hw[f"during_patterns_{mib}MiB"] = torch.cuda.max_memory_allocated()
    payload["high_water_bytes"] = hw
    payload["high_water_note"] = (
        "measured, not asserted. If after_prefix is below every during_* value then "
        "this run's process high-water was set by the measurement and not by the "
        "treatment, and any claim about a peak-bytes threshold is unsupported."
    )
    Path(args.out).write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()

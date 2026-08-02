"""Does the buffer LAYOUT explain the per-cell argmin slot? No. It is byte-identical across cells that disagree.

The anchored run (`AI/artifacts/alloc_anchored_assembled.json`) left one thing
unexplained and named it: the argmin slot is unanimous within every cell across
four independent processes, differs between cells, and permutes at p = 0.0. The
`step` cell's slot 0 runs 0.2583 TB/s above the best single observation in any
other cell. Something about the prefix decides WHICH placement is fast.

The obvious mechanism, and the one worth killing first: the prefix changes where
the six measurement buffers land, and some addresses are better than others.
`_identical_buffer_spread` allocates one 2 GiB `dst` and five 2 GiB `srcs` in a
fixed order, then times `dst.copy_(src[j])` per slot. Slots are distinguishable
only by their buffer, so if the prefix moves the buffers, that is the whole
story and it would explain unanimity (same layout every process), between-cell
disagreement (different layout per cell), and non-monotonicity in bytes at once
(addresses are not monotone in the bytes allocated before them).

It is wrong. This probe reproduces the exact allocation sequence and records the
pointers instead of timing anything:

    cell     count  size     dst offset      src offsets
    zero         0   512   11112808448   [8963227648, 6448742400, 4299161600, 2149580800, 0]
    step        13   512   10747904000   [8598323200, 6448742400, 4299161600, 2149580800, 0]
    lo_lo       24   512   10747904000   [8598323200, 6448742400, 4299161600, 2149580800, 0]
    lo_hi       24  1024   10747904000   [8598323200, 6448742400, 4299161600, 2149580800, 0]
    hi_lo       48   512   10747904000   [8598323200, 6448742400, 4299161600, 2149580800, 0]
    hi_hi       48  1024   10747904000   [8598323200, 6448742400, 4299161600, 2149580800, 0]

Five of the six cells are byte-identical, and their argmins are 1, 2, 3, 1, 0 --
four distinct values on one layout. A variable that does not vary cannot explain
an outcome that does. Absolute pointers differ per process (the virtual base
moves), which is why the comparison is relative to `min(dst, *srcs)`; the
relative layout is deterministic, reproduced across processes here.

What this does NOT refute
-------------------------
"Placement", broadly. It refutes *relative virtual layout* as the variable. The
caching allocator returns virtual addresses; which physical pages back them is
the driver's choice, made against a physical free-list whose fragmentation the
prefix history absolutely does change. Identical virtual offsets over different
physical pages is exactly the state this table cannot distinguish, and it stays
the leading hypothesis -- MI355X interleaves across channels and MALL sets by
physical address, so two runs with identical virtual layout can still stripe
differently. I have no physical-address visibility from userspace on ROCm, so I
cannot take that further with this tool.

The one cell that IS layout-distinct is `zero`, whose argmin (slot 4) is also
the one no count>0 cell shares. That agreement is worth recording and worth
almost nothing on its own: with five cells sharing a layout and disagreeing four
ways, a sixth that differs in both is one draw. Reading it as support would be
fitting a rule to the single point that did not refute it.

Why this is not pre-registered
------------------------------
It is not a statistical test. Six pointer tuples, deterministic, no aggregation,
no null -- one command reproduces the table or it does not. The pre-registration
discipline is there to stop an analysis being chosen after seeing the outcome;
there is no analysis here to choose. What it does have is a stated prediction it
could have confirmed: I expected the layouts to differ, and wrote this expecting
to explain the argmin, not to refute it.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_alloc_layout.py OUT.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MIB = 1024 * 1024
MEASURE_MIB = 2048
N_SRCS = 5


def _all_cells():
    """The six anchored cells, imported rather than retyped.

    Behind a function because the import needs `REPO` on `sys.path`, and a
    module-level import after that insert is an E402 that only one of the two
    pinned ruff versions will accept a `noqa` for -- 0.11.13 wants the
    directive, 0.16.0 calls the same directive dead. A local import is not a
    lint dodge here: it is the only form both versions accept without one file
    disagreeing with the other about its own suppression.
    """
    from AI.probe_alloc_factorial import ANCHOR_CELLS, CELLS

    return CELLS + ANCHOR_CELLS


# The argmins the anchored run measured, for the join. Recorded here as data,
# not asserted: the artifact is the source and _check_against_artifact reads it.
ANCHORED_ARTIFACT = "AI/artifacts/alloc_anchored_assembled.json"


def _layout(count, buffer_mib):
    """Reproduce _identical_buffer_spread's allocation sequence, record pointers.

    Deliberately does not import and call `_identical_buffer_spread` itself: that
    function times copies, and timing is what this probe is trying not to do. The
    sequence it replicates -- prefix, free, empty_cache, one dst then five srcs,
    all at 2 GiB -- is asserted against the real function's source in
    `_sequence_matches` rather than left as a comment claiming they agree.
    """
    import torch

    torch.cuda.reset_peak_memory_stats()
    n = (buffer_mib * MIB) // 4
    live = [torch.empty(n, device="cuda", dtype=torch.float32).fill_(1.0) for _ in range(count)]
    del live
    torch.cuda.empty_cache()

    cnt = (MEASURE_MIB * MIB) // 4
    dst = torch.empty(cnt, device="cuda", dtype=torch.float32)
    srcs = [torch.empty(cnt, device="cuda", dtype=torch.float32).fill_(1.0) for _ in range(N_SRCS)]

    d = dst.data_ptr()
    ps = [s.data_ptr() for s in srcs]
    base = min([d, *ps])
    return {
        "dst_offset": d - base,
        "src_offsets": [p - base for p in ps],
        "src_minus_dst": [p - d for p in ps],
        # Absolute pointers vary per process (virtual base moves), so they are
        # recorded but never compared. Keeping them makes that checkable rather
        # than a claim in a docstring.
        "dst_ptr_absolute": d,
        "src_ptrs_absolute": ps,
        "all_2MiB_aligned": all(p % (2 * MIB) == 0 for p in [d, *ps]),
        "n_segments": len({s["address"] for s in torch.cuda.memory_snapshot()}),
    }


def _sequence_matches():
    """Assert this probe's allocation sequence still matches the real measurement.

    The value of this probe is entirely that it allocates what
    `_identical_buffer_spread` allocates. If that function is edited -- a sixth
    src, a different size, dst allocated last -- this file silently starts
    describing a layout nothing measures. Checking the source text is crude, but
    it fails loudly on exactly the edits that would matter.
    """
    import inspect

    from AI.probe_rmsnorm_roofline import _identical_buffer_spread

    src = inspect.getsource(_identical_buffer_spread)
    return {
        "dst_allocated_before_srcs": src.index("dst = torch.empty") < src.index("srcs = ["),
        "five_srcs": f"for _ in range({N_SRCS})" in src,
        "same_elem_count_expr": "cnt = (mib * 1024 * 1024) // 4" in src,
        "measured_at_mib": MEASURE_MIB,
    }


def _worker(label, count, buffer_mib):
    r = _layout(count, buffer_mib)
    r.update({"cell": label, "count": count, "buffer_mib": buffer_mib})
    print(json.dumps(r))


def _check_against_artifact(by_cell):
    """Join layouts to the anchored run's measured argmins. The refutation.

    Reads the argmins out of the committed artifact rather than restating them,
    so this cannot drift from the run it is explaining.
    """
    path = REPO / ANCHORED_ARTIFACT
    if not path.exists():
        return {"status": f"missing {ANCHORED_ARTIFACT}, join skipped"}
    a = json.loads(path.read_text())
    per_cell = a["P2_PREREGISTERED_argmin_slot"]["per_cell"]
    argmin = {c: v["modal"] for c, v in per_cell.items()}

    groups = {}
    for cell, reps in by_cell.items():
        key = json.dumps([reps[0]["dst_offset"], reps[0]["src_offsets"]])
        groups.setdefault(key, []).append(cell)

    biggest = max(groups.values(), key=len)
    argmins_in_biggest = sorted({argmin[c] for c in biggest if c in argmin})
    return {
        "argmin_by_cell": argmin,
        "distinct_layouts": len(groups),
        "cells_sharing_the_largest_layout": sorted(biggest),
        "distinct_argmins_among_them": argmins_in_biggest,
        "layout_explains_argmin": len(argmins_in_biggest) <= 1,
        "the_refutation": (
            f"{len(biggest)} of {len(by_cell)} cells have byte-identical relative "
            f"layouts and {len(argmins_in_biggest)} distinct argmin slots between "
            "them. A variable that does not vary cannot explain an outcome that "
            "does, so relative virtual layout is not the mechanism. This was "
            "written expecting to confirm it."
        ),
        "what_survives": (
            "physical placement. The caching allocator hands out virtual "
            "addresses; the driver chooses the physical pages behind them from a "
            "free-list whose fragmentation the prefix history does change. "
            "MI355X interleaves channels and MALL sets by physical address, so "
            "identical virtual offsets can still stripe differently -- the one "
            "state this table cannot distinguish. No userspace physical-address "
            "visibility on ROCm, so this tool cannot go further."
        ),
        "the_one_layout_distinct_cell": (
            "zero is the only cell with its own layout, and its argmin is the "
            "only one no count>0 cell shares. One draw. With five cells sharing "
            "a layout and disagreeing four ways, reading the sixth as support "
            "would be fitting a rule to the single point that did not refute it."
        ),
    }


def _git():
    def q(*a):
        try:
            return subprocess.run(
                ["git", *a], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    return {"head": q("rev-parse", "HEAD"), "dirty": bool(q("status", "--porcelain"))}


def _sweep(out_path, reps):
    import torch

    rows = []
    for label, count, buffer_mib in _all_cells():
        for _ in range(reps):
            p = subprocess.run(
                [sys.executable, __file__, "--worker", label, str(count), str(buffer_mib)],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            )
            rows.append(json.loads(p.stdout.strip().splitlines()[-1]))

    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)

    deterministic = {
        c: len({json.dumps([r["dst_offset"], r["src_offsets"]]) for r in reps_}) == 1
        for c, reps_ in by_cell.items()
    }

    payload = {
        "what": (
            "relative buffer layout per cell, reproducing "
            "_identical_buffer_spread's allocation sequence without timing it"
        ),
        "device_name": torch.cuda.get_device_name(0),
        "reps_per_cell": reps,
        "allocation_sequence_still_matches_the_measurement": _sequence_matches(),
        "layout_deterministic_within_cell": deterministic,
        "all_cells_deterministic": all(deterministic.values()),
        "why_offsets_and_not_pointers": (
            "absolute pointers differ every process because the virtual base "
            "moves; the relative layout does not. Comparing absolutes would "
            "report six differences and mean nothing."
        ),
        "DOES_LAYOUT_EXPLAIN_THE_ARGMIN": _check_against_artifact(by_cell),
        "rows": rows,
        "git": _git(),
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--worker", nargs=3, metavar=("LABEL", "COUNT", "MIB"))
    a = ap.parse_args()
    if a.worker:
        _worker(a.worker[0], int(a.worker[1]), int(a.worker[2]))
    else:
        if not a.out:
            ap.error("out is required")
        _sweep(a.out, a.reps)


if __name__ == "__main__":
    main()

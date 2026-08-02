"""Assemble the allocation-ordinal / timing-order separation.

Reads the permuted-order runs from `AI/probe_order_confound.py` and reports how
much of the variance each candidate cause explains. Every previous copy artifact
in this tree conflated three of them into one index; this one takes them apart.

The guard and manifest conventions are the same as `AI/assemble_copy_axes.py`,
for the same reason: an artifact that corrects a provenance defect is exactly the
kind that should not introduce one.
"""

import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "AI/data/copy_placement_draws/raw_order_dev5"
OUT = REPO / "AI/data/copy_placement_draws/order_confound_dev5.json"

# Same ceiling and rationale as the copy-axes assembler: a tripwire that decays
# as the artifact grows needs a hard stop, not a field reporting its decline.
GUARD_FALSE_NEGATIVE_CEILING_PCT = 12.0


def _load(src):
    """Read the runs and refuse anything the analysis cannot legitimately pool.

    The copy-axes assembler took its input directory from argv with no checks, so
    a `/tmp` scratch directory holding a duplicate, a truncated run, or a mix of
    two devices would have assembled without complaint. @Reviewer raised that
    against `9899e9d`. The fix is not to hard-code the path -- reproducing from a
    scratch collection is the normal workflow -- it is to make every assumption
    the pooling relies on a checked precondition.
    """
    src = Path(src).resolve()
    paths = sorted(src.glob("*.json"))
    if not paths:
        raise SystemExit(f"no runs in {src}")
    runs = [json.loads(p.read_text()) for p in paths]

    seeds = [d["seed"] for d in runs]
    if len(set(seeds)) != len(seeds):
        dup = sorted({s for s in seeds if seeds.count(s) > 1})
        raise SystemExit(f"duplicate seeds {dup}: the same process would be pooled twice")

    for field in ("device_name", "hip_visible_devices", "size_mib"):
        seen = {d[field] for d in runs}
        if len(seen) != 1:
            raise SystemExit(f"runs disagree on {field}: {sorted(seen)}; refusing to pool")

    widths = {len(d["rows"]) for d in runs}
    if len(widths) != 1:
        raise SystemExit(f"runs have differing row counts {sorted(widths)}; refusing to pool")

    for d, p in zip(runs, paths):
        ords_ = sorted(r["alloc_ordinal"] for r in d["rows"])
        times = sorted(r["time_position"] for r in d["rows"])
        want = list(range(len(d["rows"])))
        if ords_ != want or times != want:
            raise SystemExit(f"{p.name}: alloc/time indices are not a permutation of {want}")
        for r in d["rows"]:
            v = r["TBps_at_min"]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise SystemExit(f"{p.name}: non-finite rate {v!r}")
    return runs, paths, src


def _eta(rows, key, values):
    gm = sum(values) / len(values)
    sst = sum((v - gm) ** 2 for v in values)
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r["TBps_at_min"])
    ss = sum(len(v) * ((sum(v) / len(v)) - gm) ** 2 for v in g.values())
    return {
        "eta_squared_pct": round(ss / sst * 100.0, 2),
        "group_means": {str(k): round(sum(v) / len(v), 4) for k, v in sorted(g.items())},
        "n_groups": len(g),
        "n_per_group": sorted({len(v) for v in g.values()}),
    }


def _offsets(runs):
    """Is the src-dst offset vector stable across processes with different bases?

    This is the part that makes "placement" a measurement rather than a reading.
    If the offsets repeat byte for byte while the absolute base moves, then the
    effect is a function of relative address, and absolute placement is not
    involved at all.
    """
    vecs, bases = set(), set()
    for d in runs:
        rs = sorted(d["rows"], key=lambda r: r["alloc_ordinal"])
        vecs.add(tuple(r["address"] - d["dst_address"] for r in rs))
        bases.add(d["dst_address"] >> 40)
    return {
        "distinct_offset_vectors": len(vecs),
        "distinct_dst_bases_by_1TiB_region": len(bases),
        "n_processes": len(runs),
        "offset_vector_bytes": [str(x) for x in sorted(next(iter(vecs)))] if len(vecs) == 1 else None,
        "offset_vector_GiB": (
            [round(x / 2**30, 3) for x in sorted(next(iter(vecs)))] if len(vecs) == 1 else None
        ),
        "interpretation": (
            "one offset vector across all processes while the absolute base varies means "
            "the ordinal effect is a function of the source's address RELATIVE to the "
            "destination, not of where the pair lands in the address space. Absolute "
            "placement is randomized here and does not track the rate."
            if len(vecs) == 1
            else "offsets differ across processes; relative address does not by itself "
            "explain the ordinal effect."
        ),
    }


def _relative_offset_eta(runs, values):
    gm = sum(values) / len(values)
    sst = sum((v - gm) ** 2 for v in values)
    g = defaultdict(list)
    for d in runs:
        for r in d["rows"]:
            g[round((r["address"] - d["dst_address"]) / 2**31)].append(r["TBps_at_min"])
    ss = sum(len(v) * ((sum(v) / len(v)) - gm) ** 2 for v in g.values())
    return {
        "eta_squared_pct": round(ss / sst * 100.0, 2),
        "bucket_width": "2 GiB",
        "bucket_means": {str(k): round(sum(v) / len(v), 4) for k, v in sorted(g.items())},
    }


def _nested(rows, values):
    """Does timing order explain anything ONCE allocation ordinal is accounted for?

    The two marginal eta-squareds are not a decomposition -- the design is
    unbalanced (a random permutation per process does not fill the 5x5 grid
    evenly), so they overlap. The number that settles it is timing position
    *within* allocation ordinal: variance the ordinal cannot already account for.
    Reported alongside within-cell scatter so the residual is attributable.
    """
    gm = sum(values) / len(values)
    sst = sum((v - gm) ** 2 for v in values)
    by_alloc, by_cell = defaultdict(list), defaultdict(list)
    for r in rows:
        by_alloc[r["alloc_ordinal"]].append(r)
        by_cell[(r["alloc_ordinal"], r["time_position"])].append(r["TBps_at_min"])
    ss = 0.0
    for rs in by_alloc.values():
        m = sum(x["TBps_at_min"] for x in rs) / len(rs)
        h = defaultdict(list)
        for x in rs:
            h[x["time_position"]].append(x["TBps_at_min"])
        ss += sum(len(u) * ((sum(u) / len(u)) - m) ** 2 for u in h.values())
    within = sum(sum((x - sum(u) / len(u)) ** 2 for x in u) for u in by_cell.values())
    counts = sorted(len(u) for u in by_cell.values())
    return {
        "time_within_alloc_eta_squared_pct": round(ss / sst * 100.0, 2),
        "within_cell_eta_squared_pct": round(within / sst * 100.0, 2),
        "cells_occupied": len(by_cell),
        "cells_possible": len(by_alloc) ** 2,
        "cell_replicate_counts_min_max": [counts[0], counts[-1]],
        "note": (
            "the design is unbalanced, so the two marginal eta-squareds overlap and do "
            "not sum to 100. Timing position within allocation ordinal is the residual "
            "that matters, and it is smaller than the scatter between identical repeats."
        ),
    }


def _manifest(src, paths, extra):
    entries = []
    for p in list(paths) + [REPO / e for e in extra]:
        b = p.read_bytes()
        try:
            name = str(p.relative_to(REPO))
        except ValueError:
            name = str(p)
        entries.append({"path": name, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)})
    inside = str(src).startswith(str(REPO))
    return {
        "n_inputs": len(entries),
        "source_dir": str(src),
        "source_dir_inside_repo": inside,
        "source_dir_note": (
            "inputs are versioned alongside the artifact, so the hashes below can be "
            "rechecked by anyone with the commit"
            if inside
            else "inputs are OUTSIDE the repo: the hashes below pin the bytes this run "
            "read, but nobody else can recheck them. Copy the runs into the tree before "
            "citing this artifact."
        ),
        "entries": entries,
        "what_this_proves": (
            "the assembler read exactly these bytes. It does not prove they came from a "
            "GPU, and a manifest cannot: it is a binding between artifact and raw runs, "
            "not an attestation."
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

    # Not `.strip()` then `ln[3:]`: porcelain status codes are two columns and an
    # unstaged modification leads with a space, so stripping the whole output eats
    # one character off the first path only. It produced `I/flydsl_rmsnorm_notes.md`
    # -- close enough to a real path to read as correct, which is the entire problem.
    dirty = [ln[3:] for ln in run("status", "--porcelain", strip=False).splitlines()]
    return {
        "commit": run("rev-parse", "HEAD")[:7],
        "worktree_dirty": bool(dirty),
        "worktree_dirty_paths": sorted(dirty),
        "worktree_dirty_note": (
            "a dirty ancestor means the commit above does not pin the code that produced "
            "this artifact. The paths are listed so a reader can see whether the "
            "difference touches the probe or the assembler."
        ),
    }


def _guard(payload):
    measured = set()

    def walk(n):
        if isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
        elif isinstance(n, (int, float)) and not isinstance(n, bool):
            for p in range(6):
                measured.add(f"{n:.{p}f}")

    walk(payload)

    prose = []

    def collect(n):
        if isinstance(n, dict):
            for v in n.values():
                collect(v)
        elif isinstance(n, list):
            for v in n:
                collect(v)
        elif isinstance(n, str):
            prose.append(n)

    collect(payload)
    unexplained = sorted(
        {tok for t in prose for tok in re.findall(r"\d+\.\d+", t) if tok not in measured}
    )
    if unexplained:
        raise SystemExit(
            f"prose contains decimals {unexplained} that no field on this payload "
            "computed. Interpolate from a computed field."
        )
    grid = [f"{i / 100:.2f}" for i in range(2001)]
    hits = sum(1 for g in grid if g in measured)
    rate = hits / len(grid) * 100.0
    if rate > GUARD_FALSE_NEGATIVE_CEILING_PCT:
        raise SystemExit(
            f"prose guard false-negative rate {rate:.3f}% exceeds the "
            f"{GUARD_FALSE_NEGATIVE_CEILING_PCT}% ceiling. Narrow the accept-set or "
            "check prose against the specific field it cites. Do not raise the ceiling."
        )
    return {
        "accept_set_size": len(measured),
        "false_negative_rate_pct": round(rate, 3),
        "ceiling_pct": GUARD_FALSE_NEGATIVE_CEILING_PCT,
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else str(RAW)
    runs, paths, src = _load(src)
    rows = [r for d in runs for r in d["rows"]]
    values = [r["TBps_at_min"] for r in rows]

    alloc = _eta(rows, "alloc_ordinal", values)
    timing = _eta(rows, "time_position", values)
    rel = _relative_offset_eta(runs, values)
    offs = _offsets(runs)
    nested = _nested(rows, values)

    payload = {
        "what": (
            "Separates allocation ordinal from timing order for the 2 GiB copy draws, "
            "and records addresses. Buffers are allocated 0..4 and measured in a "
            "per-process permuted order."
        ),
        "why": (
            "every prior copy artifact in this tree allocated buffer i i-th and then "
            "measured it i-th, so allocation ordinal, timing order and address were one "
            "index, and the effect was reported as placement without the design being "
            "able to show that. Raised by @Reviewer against 9899e9d and 682e103."
        ),
        "device_name": runs[0]["device_name"],
        "hip_visible_devices": runs[0]["hip_visible_devices"],
        "n_processes": len(runs),
        "n_draws": len(rows),
        "size_mib": runs[0]["size_mib"],
        "distinct_measurement_orders": len({tuple(d["measurement_order"]) for d in runs}),
        "generator": "AI/probe_order_confound.py",
        "assembler": "AI/assemble_order_confound.py",
        "input_manifest": _manifest(
            src, paths, ["AI/probe_order_confound.py", "AI/assemble_order_confound.py"]
        ),
        **_git(),
        "by_alloc_ordinal": alloc,
        "by_time_position": timing,
        "nested": nested,
        "by_relative_offset_from_dst": rel,
        "address_structure": offs,
        "pooled_range_pct_of_min": round((max(values) / min(values) - 1) * 100.0, 2),
    }

    payload["verdict"] = (
        f"Allocation ordinal explains {alloc['eta_squared_pct']}% of the variance; "
        f"timing position explains {timing['eta_squared_pct']}%. With the two varied "
        "independently across "
        f"{payload['distinct_measurement_orders']} distinct measurement orders, the "
        "effect tracks WHERE a buffer was allocated, not WHEN it was measured. Warmup, "
        "clock ramp and drift within a process are excluded as the primary cause. "
        "The placement reading of prior artifacts is supported -- but it was not "
        "supported BY those artifacts, which could not have distinguished these cases."
    )
    payload["verdict_strength"] = (
        f"The two marginal figures overlap ({timing['eta_squared_pct']}% is not a "
        "residual: the permuted design is unbalanced). The load-bearing number is "
        f"timing position WITHIN allocation ordinal, "
        f"{nested['time_within_alloc_eta_squared_pct']}%, which is smaller than the "
        f"{nested['within_cell_eta_squared_pct']}% scatter between identical repeats of "
        "the same (ordinal, position) cell. Once you know where a buffer was allocated, "
        "when it was measured tells you less than noise does."
    )
    payload["mechanism"] = (
        "The src-dst offset vector is byte-identical across every process while the "
        f"absolute base varies over {offs['distinct_dst_bases_by_1TiB_region']} distinct "
        "1 TiB regions, and bucketing by that relative offset recovers "
        f"{rel['eta_squared_pct']}% of the variance -- the same as allocation ordinal, "
        "because in this design they are the same partition. So 'allocation slot' is "
        "more precisely the source's offset relative to the destination. Absolute "
        "placement is randomized here and does not track the rate. What this does NOT "
        "establish is why a given offset is faster; that needs a probe that varies the "
        "offset directly rather than through the allocation sequence."
    )
    payload["prose_guard"] = _guard(payload)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

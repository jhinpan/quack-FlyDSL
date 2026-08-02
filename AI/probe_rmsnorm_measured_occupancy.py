"""Measure occupancy, and reconcile rocprof's VGPR_Count with the artifact's.

`AI/data/rmsnorm_fwd_vgpr_by_n.json` carries a *computed* occupancy bound:
`min(floor(512 / vgpr_alloc), 8)`. @Reviewer's blocker was that a computed
bound coinciding with the bandwidth cliff does not establish residency or
causality -- the arithmetic would produce that step whether or not the hardware
ever ran at it. This probe measures the thing the arithmetic predicts, with
rocprofv3's `MeanOccupancyPerActiveCU`, and reports both side by side.

Two traps had to be cleared first, and both are checked here rather than
assumed.

**rocprof's VGPR column is a lossy function of the artifact's, not a second
unit system.** At N=49152 the artifact says 230 and rocprof says 116. The exact
relation is

    rocprof_vgpr == vgpr_alloc_wave64 / 2

on 10 of 10 widths -- half the *granule-8 rounded allocation*, not half the
count. It is asserted on every row as a toolchain regression guard. Nothing in
this file reads rocprof's register columns for any purpose.

This took three wrong answers to get right, and the sequence is the point.

1. I wrote the relation as `roundup(ceil(v / 2), 4)` and explained it as wave64
   architectural VGPRs versus 32-lane physical register-file entries -- a unit
   difference, neither source wrong. I supported it with "10 of 10 widths, five
   held out".
2. @Reviewer rejected the units story and proposed an incomplete
   ROCProfiler-SDK decode of gfx950 code-object fields. I checked AGPRs (a
   conversion scales them, a decoding gap drops them), found rocprof reports
   `Accum_VGPR_Count = 0` against artifact 8 and 44, and conceded his
   hypothesis was "better supported".
3. Both of those still treated this as an open empirical question with evidence
   on each side. @Autotune pointed out it never was: `roundup(ceil(v/2), 4)`
   **is identically `ceil(v/8)*4`** for every integer v. My formula could not
   have failed on any width. "10 of 10, five held out" was reporting an
   identity as a confirmed prediction -- a check that could not have come out
   otherwise, which is the same defect as every other one in this file's
   history, arriving this time inside the correction to the previous one.

@Reviewer's mechanism is the true one: ROCProfiler-SDK 1.1.0 has no gfx950
accumulator decoder, so gfx950 falls through to `(PGM_RSRC1+1)*4` with
`Accum_VGPR_Count = 0`, while LLVM encodes the total with granule 8. That
composition produces exactly the observed numbers.

The tell that settles it without reading any ROCProfiler source is in this
file's own rows: **the map is not injective.** Artifact 226 and 230 are two
different allocations and rocprof reports 116 for both. A unit conversion is
order-preserving and invertible; quantization is neither. That collision is now
asserted rather than left to be noticed. And `512 // (2 * rocprof)` recovers the
correct bound on all ten rows for the same reason -- `2 * rocprof` *is* the
allocation, so the "cross-check" was re-deriving a number the artifact already
stated.

One consequence for a nearby claim: I wrote that vgpr+agpr = 272/344 at
57344/65536 "still gives bound 1". @Reviewer flagged that amdhsa `.vgpr_count`
already includes AGPRs, so that was double-counting, and these rows confirm it
independently -- at N=65536 rocprof reports 152 = `ceil(300/8)*4`, where a
separate 44 AGPRs would have made the encoded total 344 and the reading 172.
The bound was computed from `vgpr_count` alone throughout, which is the correct
total, so nothing downstream moves.

**Occupancy is not register-bound unless enough waves exist to be bound.** At
m=1024 and N=49152 the grid supplies only 4 waves/SIMD, so any register limit
of 4 or more is invisible and a measurement there would "confirm" whatever it
was compared against. Every row records `grid_supply_waves_per_simd`, and
`register_bound_is_binding` says whether the register limit is actually the
smallest of the three constraints. Only rows where it is can discriminate.

That flag was wrong in the first version of this probe, in the same way the
thing it exists to catch is wrong: it compared the *already-capped* bound
against the other constraints, which is trivially true whenever the hardware
cap binds, so it marked the N=4096 and N=8192 rows (register limits 12 and 8
against a cap of 8) as register-bound when they are cap-bound. It now tests the
uncapped `waves_per_simd_register_limited`, and `limiting_constraint` names
which of registers / hardware cap / grid supply is actually smallest. The cliff
rows (N >= 16384) were never affected -- their register limits are 5, 3, 2, 1,
all well under both other constraints -- but a guard that says True when it
should say False is worth less than no guard, and @Reviewer caught it.

What the measurement decides. On the three widths where the two register
sources predict *different* occupancies (16384, 32768, 49152 at m=16384, where
grid supply is 8 and not binding), measured occupancy is 3.96 / 2.64 / 1.97
against artifact predictions 5 / 3 / 2 and rocprof-unit predictions 8 / 6 / 4.
The artifact's numbers are the ones the hardware behaves like.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_measured_occupancy.py
Writes AI/data/rmsnorm_fwd_measured_occupancy.json.

The script re-executes itself under rocprofv3 with `--inner`; that child does
the dispatches and this parent parses the counter CSV. rocprofv3 needs
`--output-format csv` -- without it the default output is a `.db` holding
`rocpd_kernel_dispatch` and no counter table at all.
"""

import csv
import hashlib
import json
import os
import pickle
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_CACHE_DIR = os.environ.setdefault(
    "FLYDSL_RUNTIME_CACHE_DIR", tempfile.mkdtemp(prefix="flydsl-occ-probe-")
)

import quack.rmsnorm_flydsl as flydsl_rmsnorm  # noqa: E402  (must follow the env + path setup)
from quack.flydsl import rmsnorm_config  # noqa: E402  (must follow the sys.path insert)

# Two sweeps at *fixed, stated* m. m matters: it sets how many waves exist, and
# a small m makes the register limit unobservable. The earlier reading of
# "~6.1 waves/SIMD" was taken at one m and compared against a bound derived at
# another, which is exactly the mistake this file is written to avoid.
#
# DISCRIMINATING: m large enough that grid supply is not binding, at widths
# where the artifact and rocprof unit systems predict different occupancies.
DISCRIMINATING_M = 16384
DISCRIMINATING_NS = [4096, 8192, 16384, 32768, 49152]

# BOUNDARY: the bandwidth cliff's own m, across the 49152 -> 57344 edge.
BOUNDARY_M = 4096
BOUNDARY_NS = [32768, 40960, 49152, 57344, 65536]

# The wave64 -> rocprof VGPR relation was established on widths split into a fit
# set and a held-out set, and the prose said so -- but two of the held-out
# widths (2048, 24576) appeared in no sweep, so the claim could not be audited
# from the committed artifact alone. @Reviewer caught that. The split is now
# declared here and every width in it is measured and emitted below, so the
# claim is checkable without taking my word for the history.
VGPR_RELATION_FIT_NS = [1024, 4096, 8192, 49152, 57344]
VGPR_RELATION_HELDOUT_NS = [2048, 16384, 24576, 32768, 40960]
VGPR_RELATION_M = 1024  # only registers are read here; occupancy is not claimed at this m

REPEATS = 3  # dispatches per shape; the CSV row count is asserted against this

VGPRS_PER_SIMD = 512
ALLOC_GRANULARITY_WAVE64 = 8
ROCPROF_VGPR_GRANULARITY = 4
WAVE_SIZE = 64
SIMDS_PER_CU = 4

COUNTER = "MeanOccupancyPerActiveCU"

# Bandwidth percentages at m=4096, read out of AI/data/rmsnorm_fwd_width_cliff.json
# rather than transcribed. They are carried here only so the occupancy and bandwidth
# columns can be read against each other; this probe does not measure them.
#
# They used to be three hand-copied literals, {49152: 77.5, 57344: 37.8, 65536: 39.2},
# and @Reviewer found that they were not co-measured: 77.5 and 37.8 are the cliff's
# GRAPH numbers while 39.2 is its EAGER one. Nothing downstream used them
# quantitatively, but a three-entry table silently mixing two timing regimes is
# exactly the kind of number that gets picked up later as if it were one series.
# Reading them from the source file with the regime named makes the mix impossible.
BANDWIDTH_REGIME = "eager_share_of_ceiling_pct"
CLIFF_SIDECAR = "AI/data/rmsnorm_fwd_width_cliff.json"
BANDWIDTH_NS = (49152, 57344, 65536)


def _bandwidth_pct_at_m4096():
    """Read the cliff sidecar's eager bandwidth column, and say which column it is."""
    raw = json.loads((REPO / CLIFF_SIDECAR).read_text())
    sweep = raw["width_sweep_m_fixed"]
    if sweep["m"] != BOUNDARY_M:
        raise SystemExit(
            f"the cliff sidecar's width sweep is at m={sweep['m']}, not {BOUNDARY_M}; "
            "the bandwidth column would not be comparable to these occupancy rows"
        )
    rows = {r["n"]: r for r in sweep["rows"]}
    missing = [n for n in BANDWIDTH_NS if n not in rows]
    if missing:
        raise SystemExit(f"cliff sidecar has no m={BOUNDARY_M} row for N={missing}")
    return {n: round(rows[n][BANDWIDTH_REGIME], 2) for n in BANDWIDTH_NS}


def _hw_max_waves_per_simd():
    values = set()
    for props in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        for line in props.read_text().splitlines():
            if line.startswith("max_waves_per_simd "):
                value = int(line.split()[1])
                if value:
                    values.add(value)
    if len(values) != 1:
        raise SystemExit(f"could not read a single max_waves_per_simd from KFD: {values or 'none'}")
    return values.pop()


def _num_cus():
    props = torch.cuda.get_device_properties(0)
    return props.multi_processor_count


def _raise_max_n():
    """Raise both MAX_N bindings, asserting they agreed first. Returns the original."""
    shipped = rmsnorm_config.MAX_N
    assert flydsl_rmsnorm.MAX_N == shipped, "the two MAX_N bindings already disagree"
    rmsnorm_config.MAX_N = 1 << 20
    flydsl_rmsnorm.MAX_N = 1 << 20
    return shipped


def _restore_max_n(shipped):
    rmsnorm_config.MAX_N = shipped
    flydsl_rmsnorm.MAX_N = shipped


def _inner(m, ns):
    """Child process: dispatch REPEATS forwards per width, nothing else."""
    torch.manual_seed(0)
    shipped = _raise_max_n()
    try:
        for n in ns:
            x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            for _ in range(REPEATS):
                flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
            torch.cuda.synchronize()
            del x, weight
            torch.cuda.empty_cache()
    finally:
        _restore_max_n(shipped)


def _artifact_registers(m, ns):
    """Compile each width in a fresh child process and read gpu.kernel_metadata.

    Two levels of caching have to be defeated, not one. Pointing
    FLYDSL_RUNTIME_CACHE_DIR at an empty directory defeats the on-disk cache,
    but flydsl also memoizes within a process: the second sweep asks for widths
    the first sweep already compiled, no pickle is written, and the row would
    report a stale kernel's registers. So this runs in a subprocess -- the
    `--regs` mode below -- and each row still asserts that something was
    actually compiled. Both guards are needed; either alone lets a hit through.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--regs",
            str(m),
            ",".join(str(n) for n in ns),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": tempfile.mkdtemp(prefix="flydsl-occ-art-")},
    )
    if proc.returncode != 0:
        raise SystemExit(f"register extraction failed:\n{proc.stdout}\n{proc.stderr}")
    return {int(k): v for k, v in json.loads(proc.stdout.strip().splitlines()[-1]).items()}


def _regs_inner(m, ns):
    """Child: compile each width into the fresh cache and print the metadata as JSON."""
    cache = Path(os.environ["FLYDSL_RUNTIME_CACHE_DIR"])
    torch.manual_seed(0)
    shipped = _raise_max_n()
    out = {}
    try:
        for n in ns:
            before = set(cache.rglob("*.pkl"))
            x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
            torch.cuda.synchronize()
            new = sorted(set(cache.rglob("*.pkl")) - before)
            if not new:
                raise SystemExit(
                    f"N={n}: nothing was compiled, so this row would report a stale "
                    "kernel's registers. The runtime cache was not empty."
                )
            best = None
            for path in new:
                ir = str(pickle.load(path.open("rb")).ir)
                meta = {}
                for field in ("vgpr_count", "sgpr_count", "agpr_count"):
                    match = re.search(rf"\b{field} = (\d+) : i64", ir)
                    if match is None:
                        raise SystemExit(f"{field} not present in the artifact metadata")
                    meta[field] = int(match.group(1))
                if best is None or meta["vgpr_count"] > best["vgpr_count"]:
                    best = meta
            out[n] = best
            del x, weight
            torch.cuda.empty_cache()
    finally:
        _restore_max_n(shipped)
    print(json.dumps(out))


def _vgpr_alloc(artifact_vgpr):
    """Granule-8 wave64 allocation. This is the number the hardware budgets."""
    return max(
        ALLOC_GRANULARITY_WAVE64,
        -(-artifact_vgpr // ALLOC_GRANULARITY_WAVE64) * ALLOC_GRANULARITY_WAVE64,
    )


def _predict_rocprof_vgpr(artifact_vgpr):
    """rocprof_vgpr == vgpr_alloc / 2 -- half the granule-8 rounded allocation.

    This used to be written `roundup(ceil(v / 2), 4)` and described as a
    relation confirmed on ten widths with five held out. Both parts were
    wrong in the same way. `roundup(ceil(v/2), 4)` is identically equal to
    `ceil(v/8)*4` for every integer v, so it could not have failed on any
    width, and 'held-out widths confirmed it' was reporting an identity as an
    empirical result -- the check could not have come out otherwise.
    @Autotune verified the identity by exhaustion and @Reviewer independently
    traced the mechanism: ROCProfiler-SDK 1.1.0 has no gfx950 accumulator
    decoder, so it falls through to (PGM_RSRC1+1)*4 against LLVM's granule-8
    encoding of the total.

    Written this way the content is visible: rocprof reports half the
    ALLOCATION, not half the count -- which is why 226 and 230 both come back
    as 116, and why the column carries strictly less information than the
    artifact it is being checked against.
    """
    return _vgpr_alloc(artifact_vgpr) // 2


def _run_rocprof(m, ns, tag):
    """Re-exec this file under rocprofv3 and return the parsed counter rows.

    Kernel dispatches come back in Dispatch_Id order, REPEATS per width, so the
    widths are recovered positionally. VGPR_Count is *not* used as the key --
    two widths can share one rounded value (40960 and 49152 both report 116)
    and keying on it silently merges them.
    """
    outdir = Path(tempfile.mkdtemp(prefix=f"rocprof-{tag}-"))
    cmd = [
        "rocprofv3",
        "--pmc",
        COUNTER,
        "-d",
        str(outdir),
        "-o",
        tag,
        "--output-format",
        "csv",  # without this the output is a .db with no counter table
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--inner",
        str(m),
        ",".join(str(n) for n in ns),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"rocprofv3 failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    csv_path = outdir / f"{tag}_counter_collection.csv"
    if not csv_path.exists():
        raise SystemExit(f"no counter CSV at {csv_path}; rocprofv3 emitted no counter table")

    rows = [
        r for r in csv.DictReader(csv_path.open()) if r["Kernel_Name"].startswith("rmsnorm_kernel")
    ]
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    expected = REPEATS * len(ns)
    if len(rows) != expected:
        raise SystemExit(
            f"expected {expected} rmsnorm dispatches ({REPEATS} x {len(ns)} widths), "
            f"got {len(rows)}; the positional width recovery would be wrong"
        )
    return rows


def _sweep(m, ns, tag, hw_cap, num_cus):
    bandwidth = _bandwidth_pct_at_m4096()
    artifacts = _artifact_registers(m, ns)
    rows = _run_rocprof(m, ns, tag)
    out = []
    for i, n in enumerate(ns):
        chunk = rows[i * REPEATS : (i + 1) * REPEATS]
        rocprof_vgprs = {int(c["VGPR_Count"]) for c in chunk}
        if len(rocprof_vgprs) != 1:
            raise SystemExit(f"N={n}: rocprof reported differing VGPR_Count across repeats")
        rocprof_vgpr = rocprof_vgprs.pop()
        # Recorded because it discriminates between the two explanations of the
        # VGPR relation: a unit conversion would scale AGPRs too, a decoding gap
        # drops them. See vgpr_relation_hypotheses in the payload.
        rocprof_agpr = int(chunk[0].get("Accum_VGPR_Count", -1))
        workgroup = int(chunk[0]["Workgroup_Size"])
        grid = int(chunk[0]["Grid_Size"])

        art = artifacts[n]
        predicted = _predict_rocprof_vgpr(art["vgpr_count"])
        if predicted != rocprof_vgpr:
            raise SystemExit(
                f"N={n}: rocprof's VGPR column is no longer half the granule-8 "
                f"allocation. artifact {art['vgpr_count']} allocates "
                f"{_vgpr_alloc(art['vgpr_count'])}, predicting {predicted}; rocprof says "
                f"{rocprof_vgpr}. This guard exists to catch a toolchain change, not to "
                "support a claim -- nothing here reads rocprof's register columns."
            )

        alloc = _vgpr_alloc(art["vgpr_count"])
        register_limited = VGPRS_PER_SIMD // alloc
        # Grid_Size is reported in work-items, not workgroups.
        total_waves = grid / WAVE_SIZE
        grid_supply = total_waves / (num_cus * SIMDS_PER_CU)
        bound = min(register_limited, hw_cap)
        measured = statistics.median(float(c["Counter_Value"]) for c in chunk)

        out.append(
            {
                "n": n,
                "m": m,
                "grid_size_workitems": grid,
                "workgroup_size": workgroup,
                "artifact_vgpr_count": art["vgpr_count"],
                "artifact_sgpr_count": art["sgpr_count"],
                "artifact_agpr_count": art["agpr_count"],
                "rocprof_vgpr_count": rocprof_vgpr,
                "rocprof_accum_vgpr_count": rocprof_agpr,
                "rocprof_vgpr_predicted_from_artifact": predicted,
                "vgpr_alloc_wave64": alloc,
                "waves_per_simd_register_limited": register_limited,
                "occupancy_upper_bound_waves_per_simd": bound,
                "grid_supply_waves_per_simd": round(grid_supply, 3),
                # Compare the UNCAPPED register limit against the other two
                # constraints. Testing the already-capped `bound` instead makes
                # this trivially true whenever the hardware cap binds, which
                # mislabelled the N=4096 and N=8192 rows as register-bound when
                # they are cap-bound. @Reviewer caught it.
                "register_bound_is_binding": (
                    register_limited < hw_cap and register_limited <= grid_supply
                ),
                "limiting_constraint": (
                    "registers"
                    if register_limited < min(hw_cap, grid_supply)
                    else ("hardware_cap" if hw_cap <= grid_supply else "grid_supply")
                ),
                "measured_waves_per_simd": round(measured, 4),
                "measured_over_bound": round(measured / bound, 4),
                "dispatches": len(chunk),
                # Every sample, unrounded, plus the dispatch ids they came from.
                # The sidecar used to publish only the rounded median, which
                # meant a reader could not see the spread behind it or check
                # the positional width recovery. @Reviewer called that
                # fail-open: an aggregate that cannot be recomputed from
                # anything in the file is a claim, not data.
                "counter_samples": [float(c["Counter_Value"]) for c in chunk],
                "dispatch_ids": [int(c["Dispatch_Id"]) for c in chunk],
                "bandwidth_pct_of_ceiling_at_m4096": bandwidth.get(n) if m == BOUNDARY_M else None,
                "bandwidth_regime": BANDWIDTH_REGIME
                if (m == BOUNDARY_M and n in bandwidth)
                else None,
            }
        )
    return out


def _vgpr_relation_audit():
    """Emit the widths behind the rocprof/artifact VGPR relation.

    This used to be a fit/held-out split, on the theory that predicting
    unmeasured widths was evidence for the relation. It is not, because the
    relation is an identity -- it holds on every width, fitted or not, and a
    held-out set cannot discriminate. The set labels are kept only because
    the earlier notes and commits refer to them.

    What the rows DO show, and what the split obscured, is that the map is
    not injective: two different artifact counts (226 and 230) return the
    same rocprof value, because rocprof reports half the granule-8 ALLOCATION
    and the rounding is where they merge. That is a property no unit
    conversion has, and it is visible here without leaving the file.
    """
    all_ns = sorted(set(VGPR_RELATION_FIT_NS) | set(VGPR_RELATION_HELDOUT_NS))
    artifacts = _artifact_registers(VGPR_RELATION_M, all_ns)
    rows = _run_rocprof(VGPR_RELATION_M, all_ns, "vgprrel")
    out = []
    for i, n in enumerate(all_ns):
        chunk = rows[i * REPEATS : (i + 1) * REPEATS]
        rocprof_vgprs = {int(c["VGPR_Count"]) for c in chunk}
        if len(rocprof_vgprs) != 1:
            raise SystemExit(f"N={n}: rocprof reported differing VGPR_Count across repeats")
        rocprof_vgpr = rocprof_vgprs.pop()
        art_vgpr = artifacts[n]["vgpr_count"]
        predicted = _predict_rocprof_vgpr(art_vgpr)
        if predicted != rocprof_vgpr:
            raise SystemExit(
                f"N={n}: the wave64->rocprof VGPR relation broke "
                f"(artifact {art_vgpr} predicts {predicted}, rocprof says {rocprof_vgpr})"
            )
        out.append(
            {
                "n": n,
                "m": VGPR_RELATION_M,
                "set": "fit" if n in VGPR_RELATION_FIT_NS else "held_out",
                "artifact_vgpr_count": art_vgpr,
                "vgpr_alloc_wave64": _vgpr_alloc(art_vgpr),
                "rocprof_vgpr_count": rocprof_vgpr,
                "predicted_from_alloc": predicted,
                "agrees": predicted == rocprof_vgpr,
            }
        )
    # The non-injectivity is the finding, so it is asserted rather than left
    # for a reader to notice. If a future toolchain makes this column
    # injective, the "lossy function of the artifact" reading needs revisiting
    # and the probe should say so rather than carry stale prose.
    collisions = {}
    for row in out:
        collisions.setdefault(row["rocprof_vgpr_count"], set()).add(row["artifact_vgpr_count"])
    if not any(len(v) > 1 for v in collisions.values()):
        raise SystemExit(
            "no two artifact VGPR counts collided in rocprof's column on these widths. "
            "The claim that rocprof reports a lossy (post-rounding) view rests on that "
            "collision; re-derive it before publishing."
        )
    return out


def _agreement_summary(discriminating, boundary):
    """Worst |measured/bound - 1| per group, computed rather than transcribed.

    The prose in the notes quoted these by hand and drifted: a paragraph
    correcting two over-stated agreement figures was itself quoting 23.9% and
    2.45% from a superseded run while the table directly beneath it, and the
    shipped rows, said 21.6% and 2.07%. That is the same transcription fault
    @Reviewer found in the bandwidth constants -- a hand-copied number cannot
    be checked against anything, and it silently ages every time the probe is
    re-run. Groups are by limiting constraint and by m, because that is the
    split the reading actually depends on: agreement is tight where the
    constraint binds hard and loose where it does not.
    """
    groups = {}
    for tag, rows in (("m16384", discriminating), ("m4096", boundary)):
        for r in rows:
            if r["limiting_constraint"] == "registers":
                key = f"register_bound_{tag}"
            else:
                key = f"{r['limiting_constraint']}_rows"
            dev = abs(r["measured_over_bound"] - 1.0) * 100.0
            prev = groups.get(key)
            if prev is None or dev > prev["worst_deviation_pct"]:
                groups[key] = {
                    "worst_deviation_pct": round(dev, 2),
                    "at_n": r["n"],
                    "at_m": r["m"],
                    "measured_over_bound": r["measured_over_bound"],
                }
    cliff = [r for r in boundary if r["occupancy_upper_bound_waves_per_simd"] == 1]
    if cliff:
        groups["cliff_rows_bound_1"] = {
            "worst_deviation_pct": round(
                max(abs(r["measured_over_bound"] - 1.0) * 100.0 for r in cliff), 2
            ),
            "at_n": [r["n"] for r in cliff],
            "at_m": BOUNDARY_M,
            "measured_over_bound": [r["measured_over_bound"] for r in cliff],
        }
    return {
        "what": "worst |measured/bound - 1| in each group, so the notes can cite a field "
        "instead of a transcribed number",
        "groups": groups,
    }


def _sha(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()[:16]


def _rocprof_version():
    """Which profiler decoded the code object. See vgpr_relation_resolution."""
    out = subprocess.run(
        ["rocprofv3", "--version"], capture_output=True, text=True, check=False
    ).stdout
    fields = {}
    for line in out.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    version = fields.get("version")
    if not version:
        raise SystemExit(
            "could not read a version out of `rocprofv3 --version`; the register columns "
            "in this sidecar are only interpretable against a known profiler build"
        )
    return {"version": version, "git_revision": fields.get("git_revision")}


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    if not shutil_which("rocprofv3"):
        raise SystemExit("rocprofv3 not on PATH")
    hw_cap = _hw_max_waves_per_simd()
    num_cus = _num_cus()

    # Provenance must fail closed. Run from an export without .git, these two
    # commands fail and the old code recorded commit="" with
    # worktree_dirty=false -- which reads as "clean checkout, commit unknown"
    # when the truth is "no version control at all". @Reviewer found that: a
    # field whose failure mode is indistinguishable from its success mode is
    # worse than an absent field, because it is quotable.
    head_run = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty_run = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head_run.returncode or dirty_run.returncode or not head_run.stdout.strip():
        raise SystemExit(
            f"cannot read git provenance for {REPO} (rev-parse rc="
            f"{head_run.returncode}, status rc={dirty_run.returncode}). Refusing to write "
            "a sidecar whose commit field would be empty and whose dirty flag would read "
            "clean by default."
        )
    head = head_run.stdout.strip()
    dirty = dirty_run.stdout.strip()

    print(f"discriminating sweep, m={DISCRIMINATING_M} ...", flush=True)
    discriminating = _sweep(DISCRIMINATING_M, DISCRIMINATING_NS, "discriminating", hw_cap, num_cus)
    print(f"boundary sweep, m={BOUNDARY_M} ...", flush=True)
    boundary = _sweep(BOUNDARY_M, BOUNDARY_NS, "boundary", hw_cap, num_cus)
    print(f"vgpr relation audit, m={VGPR_RELATION_M} ...", flush=True)
    vgpr_relation = _vgpr_relation_audit()

    agreement = _agreement_summary(discriminating, boundary)

    payload = {
        "what": "measured occupancy (rocprofv3 MeanOccupancyPerActiveCU) against the "
        "computed register bound, and the reconciliation of rocprof's VGPR_Count "
        "with the compiled artifact's",
        "generator": "AI/probe_rmsnorm_measured_occupancy.py",
        "note": (
            "gfx950 MI355X, bf16 fwd, has_weight only. This closes the half of @Reviewer's "
            "blocker that the computed 2->1 VGPR-capacity step was arithmetic rather than "
            "an observation: the step is now measured. Two guards run on every row and "
            "abort the probe rather than publish -- the rocprof/allocation VGPR relation, "
            "and the dispatch count behind the positional width recovery. MAX_N was raised "
            "in-process for the probe only; the shipped cap is unchanged."
        ),
        "device": torch.cuda.get_device_name(),
        "device_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", None)),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "host": platform.node(),
        "commit": head,
        "worktree_dirty": bool(dirty),
        # The register interpretation depends on which profiler decoded the
        # code object -- see vgpr_relation_resolution, where the whole
        # relation turns out to be a property of ROCProfiler-SDK 1.1.0's
        # missing gfx950 accumulator decoder. A sidecar that reports register
        # numbers without saying which profiler produced them cannot be
        # re-read later. @Reviewer asked for this.
        "rocprofv3_version": _rocprof_version(),
        "counter_note": (
            "Every row carries counter_samples (all REPEATS values, unrounded) and the "
            "dispatch_ids they came from, so the published median is recomputable and the "
            "positional width recovery is checkable. Earlier versions published only the "
            "rounded median."
        ),
        "shipped_max_n": rmsnorm_config.MAX_N,
        "source_sha256_16": {
            "quack/rmsnorm_flydsl.py": _sha("quack/rmsnorm_flydsl.py"),
            "quack/flydsl/rmsnorm_config.py": _sha("quack/flydsl/rmsnorm_config.py"),
            "AI/probe_rmsnorm_measured_occupancy.py": _sha(
                "AI/probe_rmsnorm_measured_occupancy.py"
            ),
        },
        "counter": COUNTER,
        "rocprofv3_invocation": (
            "rocprofv3 --pmc MeanOccupancyPerActiveCU -d <dir> -o <tag> --output-format csv "
            "-- python AI/probe_rmsnorm_measured_occupancy.py --inner <m> <n,n,...>. The "
            "--output-format csv is load-bearing: the default output is a .db with "
            "rocpd_kernel_dispatch and no counter table."
        ),
        "repeats_per_shape": REPEATS,
        "aggregation": "median across repeats",
        "vgprs_per_simd": VGPRS_PER_SIMD,
        "alloc_granularity_wave64": ALLOC_GRANULARITY_WAVE64,
        "max_waves_per_simd_hw": hw_cap,
        "compute_units": num_cus,
        "simds_per_cu": SIMDS_PER_CU,
        "vgpr_relation": (
            "SETTLED, and not in favour of what this field used to say. rocprof's VGPR "
            "column is not a second unit system for the artifact's; it is a LOSSY FUNCTION "
            "of it: rocprof_vgpr == vgpr_alloc_wave64 / 2 exactly, on 10 of 10 widths. It "
            "reports half of the GRANULE-8 ROUNDED allocation, so it carries strictly less "
            "information than the artifact -- see vgpr_relation_resolution. All register "
            "arithmetic here uses the artifact numbers, and nothing in this file's "
            "conclusions ever depended on rocprof's column."
        ),
        "vgpr_relation_resolution": {
            "verdict": (
                "roundup(ceil(v/2),4) is an IDENTITY, not a fitted law: it equals "
                "ceil(v/8)*4 for every integer v, which @Autotune verified by exhaustion. "
                "So '10 of 10 widths, five held out' was never evidence for anything -- an "
                "identity holds at every point, and the fit/held-out split has no "
                "discriminating power over it. @Reviewer had already located the mechanism "
                "in ROCProfiler-SDK 1.1.0: no gfx950 accumulator decoder, so gfx950 falls "
                "through to (PGM_RSRC1+1)*4 with Accum_VGPR_Count = 0, while LLVM encodes "
                "the total VGPR count with granule 8. That composition IS the formula."
            ),
            "not_a_unit_system": (
                "@Autotune's tell, checkable from this file: the map is not injective. "
                "artifact 226 and 230 are different allocations and rocprof reports 116 for "
                "both; artifact 20 and 20 -> 12. A unit conversion is order-preserving and "
                "invertible. Quantization is neither. The exact statement is "
                "rocprof == vgpr_alloc_wave64 / 2, i.e. rocprof sees the allocation AFTER "
                "granule-8 rounding, which is precisely where 226 and 230 become the same "
                "number. 512 // (2 * rocprof) recovers the correct bound on all ten rows "
                "for the same reason: 2 * rocprof IS the allocation."
            ),
            "my_error": (
                "I published a units explanation, then when @Reviewer challenged it I "
                "reported AGPRs as discriminating evidence and said his hypothesis was "
                "'better supported'. Both of those were still treating this as an open "
                "empirical question between two stories. It was not: one side was an "
                "algebraic identity, which I could have checked in one line at any point "
                "and did not, because the relation held on every width I tried and I read "
                "'always true' as strong evidence instead of asking what would make it "
                "unfalsifiable."
            ),
            "agpr_double_counting": (
                "CORRECTED. I wrote that vgpr+agpr = 272/344 at 57344/65536 'still gives "
                "bound 1', treating the artifact's vgpr_count and agpr_count as additive. "
                "@Reviewer says amdhsa .vgpr_count already includes AGPRs, so that was "
                "double-counting. The rows here confirm it independently: at N=65536, "
                "ceil(300/8)*4 = 152 is what rocprof reports, while ceil((300+44)/8)*4 = "
                "172 is not. If AGPRs were separate the encoded total would be 344 and "
                "rocprof would read 172. The occupancy bound is unaffected -- it was "
                "computed from vgpr_count alone throughout, which is the correct total."
            ),
            "blast_radius": (
                "None of the occupancy conclusions move. MeanOccupancyPerActiveCU is a "
                "counter, not a register decode; the computed bound comes from the artifact "
                "(MLIR gpu.kernel_metadata, agreeing with the msgpack amdhsa ELF note); and "
                "rocprof's VGPR_Count entered only as a cross-check. That cross-check is "
                "now known to be vacuous -- it re-derives the allocation the artifact "
                "already stated -- so it is retained as a toolchain regression guard and "
                "nothing else."
            ),
        },
        "binding_note": (
            "A register bound can only be observed where it is the smaller constraint. "
            "grid_supply_waves_per_simd = (grid_size / 64) / (CUs * 4) is how many waves the "
            "launch geometry supplies; register_bound_is_binding says whether the register "
            "limit is at or below it. Rows where it is False cannot discriminate between "
            "candidate bounds, and the m=1024 reading that prompted this probe was one of "
            "those. Measured occupancy is also an average over active CUs and over the "
            "kernel's life, so it sits slightly under a bound it is genuinely at."
        ),
        "vgpr_relation_audit": vgpr_relation,
        "vgpr_relation_audit_note": (
            "Ten widths with artifact count, granule-8 allocation, rocprof count and the "
            "prediction between them. The fit/held-out labels are vestigial: they were "
            "added when the relation was believed to be an empirical law, and a held-out "
            "set cannot test an identity -- see vgpr_relation_resolution. They are kept "
            "only because earlier commits and notes refer to them. What these rows DO "
            "show is the collision -- artifact 226 and 230 both return rocprof 116 -- "
            "which is asserted by the probe, and which is what rules out a unit "
            "conversion. Trajectory of this field, since it is the point: it first "
            "claimed a units explanation, then a held-out-prediction confirmation of it, "
            "then that a rival hypothesis was better supported. All three were "
            "unnecessary. One line of algebra was available throughout."
        ),
        "agreement_with_bound": agreement,
        "discriminating_sweep": discriminating,
        "boundary_sweep": boundary,
    }

    out_path = REPO / "AI/data/rmsnorm_fwd_measured_occupancy.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


def shutil_which(name):
    from shutil import which

    return which(name)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--inner":
        _inner(int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")])
    elif len(sys.argv) > 1 and sys.argv[1] == "--regs":
        _regs_inner(int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")])
    else:
        main()

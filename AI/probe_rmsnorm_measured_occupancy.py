"""Measure occupancy, and reconcile rocprof's VGPR_Count with the artifact's.

`AI/data/rmsnorm_fwd_vgpr_by_n.json` carries a *computed* occupancy bound:
`min(floor(512 / vgpr_alloc), 8)`. @Reviewer's blocker was that a computed
bound coinciding with the bandwidth cliff does not establish residency or
causality -- the arithmetic would produce that step whether or not the hardware
ever ran at it. This probe measures the thing the arithmetic predicts, with
rocprofv3's `MeanOccupancyPerActiveCU`, and reports both side by side.

Two traps had to be cleared first, and both are checked here rather than
assumed.

**rocprofv3 and the artifact disagree about VGPR_Count, and neither is wrong.**
At N=49152 the artifact says 230 and rocprof says 116. The relation is exact:

    rocprof_vgpr = roundup(ceil(artifact_vgpr / 2), 4)

on 10 of 10 widths, five of which (2048/16384/24576/32768/40960) were held out
-- the relation was fitted on the other five and predicted these before they
were measured. The halving is wave64 architectural VGPRs versus 32-lane
physical register-file entries; the granule-4 rounding is the allocation unit
in those units. So the two sources describe the same kernel in different units,
and the sidecar needs the units named, not the values changed. This probe
asserts the relation on every row: if a future toolchain breaks it, the probe
fails instead of quietly publishing two incompatible numbers.

All ten widths are emitted under `vgpr_relation_audit` with their fit/held-out
labels. The first version of this file named the held-out widths only in prose,
and two of them appeared in no sweep at all, so a reader could not check the
claim against the artifact -- the same failure as quoting a test count without
the invocation. What the JSON can establish is that the relation holds on all
ten and which set each width was in; that the held-out predictions were made
*before* those widths were measured rests on the commit history, and the
payload says so rather than implying the data proves it.

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

# Published bandwidth percentages at m=4096 from AI/data/rmsnorm_fwd_width_cliff.json,
# carried here only so the two columns can be read against each other. Not measured
# by this probe.
BANDWIDTH_PCT_AT_M4096 = {49152: 77.5, 57344: 37.8, 65536: 39.2}


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


def _predict_rocprof_vgpr(artifact_vgpr):
    """rocprof_vgpr = roundup(ceil(artifact_vgpr / 2), 4). See the module docstring."""
    halved = -(-artifact_vgpr // 2)
    return -(-halved // ROCPROF_VGPR_GRANULARITY) * ROCPROF_VGPR_GRANULARITY


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
    artifacts = _artifact_registers(m, ns)
    rows = _run_rocprof(m, ns, tag)
    out = []
    for i, n in enumerate(ns):
        chunk = rows[i * REPEATS : (i + 1) * REPEATS]
        rocprof_vgprs = {int(c["VGPR_Count"]) for c in chunk}
        if len(rocprof_vgprs) != 1:
            raise SystemExit(f"N={n}: rocprof reported differing VGPR_Count across repeats")
        rocprof_vgpr = rocprof_vgprs.pop()
        workgroup = int(chunk[0]["Workgroup_Size"])
        grid = int(chunk[0]["Grid_Size"])

        art = artifacts[n]
        predicted = _predict_rocprof_vgpr(art["vgpr_count"])
        if predicted != rocprof_vgpr:
            raise SystemExit(
                f"N={n}: the wave64->rocprof VGPR relation broke. artifact "
                f"{art['vgpr_count']} predicts {predicted}, rocprof says {rocprof_vgpr}. "
                "The two sources are no longer the same kernel in different units; "
                "do not publish either until this is understood."
            )

        alloc = max(
            ALLOC_GRANULARITY_WAVE64,
            -(-art["vgpr_count"] // ALLOC_GRANULARITY_WAVE64) * ALLOC_GRANULARITY_WAVE64,
        )
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
                "bandwidth_pct_of_ceiling_at_m4096": BANDWIDTH_PCT_AT_M4096.get(n),
            }
        )
    return out


def _vgpr_relation_audit():
    """Emit the fit/held-out split behind the wave64 -> rocprof VGPR relation.

    The relation is `rocprof = roundup(ceil(artifact / 2), 4)`. It was derived
    on the fit widths and then checked against the held-out ones. Prose alone
    cannot establish that ordering after the fact, so what this records is the
    weaker but *auditable* claim: the relation holds on every width in both
    sets, and which set each width belongs to is declared in the source rather
    than asserted in a paragraph.
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
                "rocprof_vgpr_count": rocprof_vgpr,
                "predicted_from_artifact": predicted,
                "agrees": predicted == rocprof_vgpr,
            }
        )
    return out


def _sha(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()[:16]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    if not shutil_which("rocprofv3"):
        raise SystemExit("rocprofv3 not on PATH")
    hw_cap = _hw_max_waves_per_simd()
    num_cus = _num_cus()

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    print(f"discriminating sweep, m={DISCRIMINATING_M} ...", flush=True)
    discriminating = _sweep(DISCRIMINATING_M, DISCRIMINATING_NS, "discriminating", hw_cap, num_cus)
    print(f"boundary sweep, m={BOUNDARY_M} ...", flush=True)
    boundary = _sweep(BOUNDARY_M, BOUNDARY_NS, "boundary", hw_cap, num_cus)
    print(f"vgpr relation audit, m={VGPR_RELATION_M} ...", flush=True)
    vgpr_relation = _vgpr_relation_audit()

    payload = {
        "what": "measured occupancy (rocprofv3 MeanOccupancyPerActiveCU) against the "
        "computed register bound, and the reconciliation of rocprof's VGPR_Count "
        "with the compiled artifact's",
        "generator": "AI/probe_rmsnorm_measured_occupancy.py",
        "note": (
            "gfx950 MI355X, bf16 fwd, has_weight only. This closes the half of @Reviewer's "
            "blocker that the computed 2->1 VGPR-capacity step was arithmetic rather than "
            "an observation: the step is now measured. Two guards run on every row and "
            "abort the probe rather than publish -- the wave64->rocprof VGPR relation, and "
            "the dispatch count behind the positional width recovery. MAX_N was raised "
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
        "vgpr_units_note": (
            "rocprof's VGPR_Count and the artifact's vgpr_count are the same kernel in "
            "different units: rocprof_vgpr = roundup(ceil(artifact_vgpr / 2), 4), exact on "
            "10 of 10 widths. Five of those (2048/16384/24576/32768/40960) were held out -- "
            "the relation was fitted on the other five and predicted these before they were "
            "measured. The factor of 2 is wave64 architectural VGPRs vs 32-lane physical "
            "register-file entries. Neither source was wrong; the sidecar was missing the "
            "units. All register arithmetic here uses the artifact (wave64) numbers."
        ),
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
            "Every width in the fit and held-out sets, with its artifact count, rocprof "
            "count and the prediction between them, so the relation is checkable from this "
            "file alone. An earlier version named held-out widths in prose that appeared in "
            "no sweep, which made the claim unauditable; @Reviewer caught it. Note what "
            "this does and does not establish: it shows the relation holds on all ten "
            "widths and which set each was in, but the committed artifact cannot by itself "
            "prove the held-out predictions were made before those widths were measured. "
            "That ordering rests on the commit history, not on this JSON."
        ),
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

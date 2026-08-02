"""Move occupancy at fixed N and see whether bandwidth follows.

The measured-occupancy sidecar established that occupancy halves at the same
width where achieved bandwidth halves. Two coincident edges are not causality,
and both @Reviewer and my own notes said so. This is the intervention that was
owed: hold N fixed, force the register allocator across the boundary, and watch
the bandwidth.

The lever is `--amdgpu-waves-per-eu`, which flydsl's ROCm backend accepts as a
`waves_per_eu` compile hint (`flydsl/compiler/backends/rocm.py`). It pressures
the allocator to fit more waves per EU, which at N=57344 drops the allocation
from 264 VGPRs (bound 1) to 256 (bound 2) -- across the boundary, at a width
that has not changed.

**The lever is not clean, and the honest reading depends on saying how.** It
buys occupancy with spills: 0 at the default, 8 at waves_per_eu=2, then 97 and
137 at 3 and 4. So this is not a single-variable intervention and cannot be
reported as one. What makes it evidential anyway is the shape:

  * The first step goes the predicted direction *while paying a spill cost that
    should push the other way*. Occupancy 1.00 -> 1.94, bandwidth 38.1% ->
    52.8%. A confound that works against your hypothesis and loses is worth
    more than one you had to argue away.
  * The later steps reverse, and they track spills rather than occupancy:
    occupancy keeps climbing (2.80, 3.67) while bandwidth falls back (43.1%,
    34.4%) as spills go 97, 137. So "more occupancy is always faster" is *not*
    what this shows, and the probe does not claim it.

The control is the load-bearing part. At N=49152 the kernel is already at 2
waves/SIMD, so the same hint has nothing to move: allocation stays at 232 and
bandwidth is 76.8% vs 76.7%, unchanged. The flag therefore does not make
kernels generically faster -- it moves bandwidth only where it moves occupancy.
Without this row the experiment would not distinguish "occupancy drives
bandwidth" from "this compiler flag is good for you".

What this still does not establish. Occupancy and spill count both move with
the hint, so the first step is a two-variable change read in the direction
where the variables disagree. A cleaner lever would change occupancy at
constant spills; I do not have one. The claim supported here is directional --
at the cliff, restoring occupancy restores a substantial part of the lost
bandwidth -- not quantitative, and not "occupancy is the whole story": even at
the best hint, 52.8% is well short of the 76.8% the kernel reaches one width
below.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_occupancy_intervention.py
Writes AI/data/rmsnorm_fwd_occupancy_intervention.json.
"""

import contextlib
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

os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", tempfile.mkdtemp(prefix="flydsl-interv-probe-"))

from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402  (after path setup)

import quack.rmsnorm_flydsl as flydsl_rmsnorm  # noqa: E402  (must follow the env + path setup)
from quack.flydsl import rmsnorm_config  # noqa: E402  (must follow the sys.path insert)

M = 4096  # the bandwidth cliff's own m, so the percentages are comparable to it

# TREATMENT: past the cliff, where the default allocation is over the boundary.
TREATMENT_N = 57344
# CONTROL: before the cliff, where the default is already at 2 waves and the
# hint has nothing to move. Without this the probe cannot tell "occupancy
# drives bandwidth" from "this flag is good for you".
CONTROL_N = 49152

HINTS = ["none", 2, 3, 4]
CONTROL_HINTS = ["none", 2]

WARMUP = 20
INNER = 20
REPEATS = 10

# From AI/data/rmsnorm_fwd_width_cliff.json: two_read_one_write, not copy.
CEILING_TBS = 6.075

VGPRS_PER_SIMD = 512
ALLOC_GRANULARITY_WAVE64 = 8
COUNTER = "MeanOccupancyPerActiveCU"

_METADATA_FIELDS = (
    "vgpr_count",
    "sgpr_count",
    "agpr_count",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "private_segment_fixed_size",
)


def _hint_ctx(hint):
    if hint == "none":
        return contextlib.nullcontext()
    return CompilationContext.compile_hints({"waves_per_eu": int(hint)})


def _raise_max_n():
    shipped = rmsnorm_config.MAX_N
    assert flydsl_rmsnorm.MAX_N == shipped, "the two MAX_N bindings already disagree"
    rmsnorm_config.MAX_N = 1 << 20
    flydsl_rmsnorm.MAX_N = 1 << 20
    return shipped


def _restore_max_n(shipped):
    rmsnorm_config.MAX_N = shipped
    flydsl_rmsnorm.MAX_N = shipped


def _regs_inner(n, hint):
    """Child: compile under the hint into a fresh cache, print the metadata.

    Runs in a subprocess because flydsl memoizes compilation in-process as well
    as on disk; a shared process across hints would report the first hint's
    kernel for all of them.
    """
    cache = Path(os.environ["FLYDSL_RUNTIME_CACHE_DIR"])
    torch.manual_seed(0)
    shipped = _raise_max_n()
    try:
        x = torch.randn((M, n), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        before = set(cache.rglob("*.pkl"))
        with _hint_ctx(hint):
            flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
        torch.cuda.synchronize()
        new = sorted(set(cache.rglob("*.pkl")) - before)
        if not new:
            raise SystemExit(
                f"N={n} hint={hint}: nothing compiled, so this row would report a "
                "different hint's kernel."
            )
        best = None
        for path in new:
            ir = str(pickle.load(path.open("rb")).ir)
            meta = {}
            for field in _METADATA_FIELDS:
                match = re.search(rf"\b{field} = (\d+) : i64", ir)
                if match is None:
                    raise SystemExit(f"{field} not present in the artifact metadata")
                meta[field] = int(match.group(1))
            if best is None or meta["vgpr_count"] > best["vgpr_count"]:
                best = meta
    finally:
        _restore_max_n(shipped)
    print(json.dumps(best))


def _bw_inner(n, hint):
    """Child: time the kernel under the hint, print the best per-call us."""
    torch.manual_seed(0)
    shipped = _raise_max_n()
    try:
        x = torch.randn((M, n), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        with _hint_ctx(hint):
            for _ in range(WARMUP):
                flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(REPEATS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(INNER):
                    flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
                end.record()
                torch.cuda.synchronize()
                best = min(best, start.elapsed_time(end) * 1000.0 / INNER)
    finally:
        _restore_max_n(shipped)
    print(json.dumps({"us": best}))


def _occ_inner(n, hint):
    """Child: a handful of dispatches under the hint, for rocprof to count."""
    torch.manual_seed(0)
    shipped = _raise_max_n()
    try:
        x = torch.randn((M, n), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        with _hint_ctx(hint):
            for _ in range(3):
                flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
            torch.cuda.synchronize()
    finally:
        _restore_max_n(shipped)


def _child(mode, n, hint, env_extra=None):
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, str(n), str(hint)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{mode} N={n} hint={hint} failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _occupancy(n, hint):
    outdir = Path(tempfile.mkdtemp(prefix="rocprof-interv-"))
    cmd = [
        "rocprofv3",
        "--pmc",
        COUNTER,
        "-d",
        str(outdir),
        "-o",
        "r",
        "--output-format",
        "csv",  # the default .db carries no counter table
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--occ",
        str(n),
        str(hint),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"rocprofv3 failed:\n{proc.stdout}\n{proc.stderr}")
    csv_path = outdir / "r_counter_collection.csv"
    rows = [
        r for r in csv.DictReader(csv_path.open()) if r["Kernel_Name"].startswith("rmsnorm_kernel")
    ]
    if not rows:
        raise SystemExit(f"N={n} hint={hint}: no rmsnorm dispatches in the counter CSV")
    return statistics.median(float(r["Counter_Value"]) for r in rows), len(rows)


def _row(n, hint):
    regs = _child(
        "--regs",
        n,
        hint,
        {"FLYDSL_RUNTIME_CACHE_DIR": tempfile.mkdtemp(prefix="flydsl-interv-art-")},
    )
    timing = _child("--bw", n, hint)
    occ, dispatches = _occupancy(n, hint)

    alloc = max(
        ALLOC_GRANULARITY_WAVE64,
        -(-regs["vgpr_count"] // ALLOC_GRANULARITY_WAVE64) * ALLOC_GRANULARITY_WAVE64,
    )
    moved_bytes = 2 * M * n * 2  # one read + one write, bf16; weight is negligible
    tbs = moved_bytes / (timing["us"] * 1e-6) / 1e12
    return {
        "n": n,
        "m": M,
        "waves_per_eu_hint": hint,
        **regs,
        "vgpr_alloc_wave64": alloc,
        "waves_per_simd_register_limited": VGPRS_PER_SIMD // alloc,
        "measured_waves_per_simd": round(occ, 4),
        "occupancy_dispatches": dispatches,
        "us_per_call": round(timing["us"], 3),
        "tbs": round(tbs, 4),
        "pct_of_ceiling": round(100.0 * tbs / CEILING_TBS, 2),
    }


def _sha(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()[:16]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    from shutil import which

    if not which("rocprofv3"):
        raise SystemExit("rocprofv3 not on PATH")

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    treatment = []
    for hint in HINTS:
        print(f"treatment N={TREATMENT_N} hint={hint} ...", flush=True)
        treatment.append(_row(TREATMENT_N, hint))
    control = []
    for hint in CONTROL_HINTS:
        print(f"control N={CONTROL_N} hint={hint} ...", flush=True)
        control.append(_row(CONTROL_N, hint))

    # The control only controls if the hint really had nothing to move there.
    base, hinted = control[0], control[1]
    if base["vgpr_alloc_wave64"] != hinted["vgpr_alloc_wave64"]:
        raise SystemExit(
            "the control width's allocation moved under the hint; it is no longer a "
            "control and the treatment cannot be read against it"
        )

    payload = {
        "what": "does bandwidth follow occupancy at fixed N? forcing the register "
        "allocator across the boundary with --amdgpu-waves-per-eu, plus a control "
        "width where the hint has nothing to move",
        "generator": "AI/probe_rmsnorm_occupancy_intervention.py",
        "note": (
            "gfx950 MI355X, bf16 fwd, has_weight only, m=4096 (the bandwidth cliff's own m). "
            "This is the intervention the measured-occupancy sidecar said was still owed. "
            "The lever is NOT clean: it buys occupancy with spills (0 / 8 / 97 / 137 VGPR "
            "spills at hints none/2/3/4), so this is a two-variable change and is reported "
            "as directional evidence, not as a quantitative result. MAX_N was raised "
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
            "AI/probe_rmsnorm_occupancy_intervention.py": _sha(
                "AI/probe_rmsnorm_occupancy_intervention.py"
            ),
        },
        "lever": (
            "waves_per_eu compile hint -> --amdgpu-waves-per-eu, via "
            "flydsl.compiler.kernel_function.CompilationContext.compile_hints; consumed in "
            "flydsl/compiler/backends/rocm.py"
        ),
        "ceiling_tbs": CEILING_TBS,
        "ceiling_source": (
            "AI/data/rmsnorm_fwd_width_cliff.json ceiling_probe, two_read_one_write -- "
            "not a copy kernel, which on this part reads MALL-inflated"
        ),
        "bytes_model": "2 * m * n * 2 (one read + one write, bf16); weight is negligible",
        "timing": f"cuda events, best of {REPEATS} x {INNER} inner calls after {WARMUP} warmup",
        "confound_note": (
            "occupancy and spill count both move with the hint, so the first step is a "
            "two-variable change. It is read in the direction where the two variables "
            "disagree: bandwidth improves 38.1 -> 52.8 pct despite the spill count going "
            "0 -> 8, i.e. the confound pushes against the hypothesis and loses. The later "
            "steps reverse and track spills (97, 137) while occupancy keeps rising, so "
            "'more occupancy is always faster' is NOT supported and is not claimed."
        ),
        "control_note": (
            "At N=49152 the default allocation is already at 2 waves/SIMD, so the hint has "
            "nothing to move: the probe asserts vgpr_alloc is unchanged, and bandwidth is "
            "flat. This is what separates 'occupancy drives bandwidth' from 'this compiler "
            "flag is generically good'."
        ),
        "still_open": (
            "Directional only. A clean lever would change occupancy at constant spills; "
            "none is available here. Restoring occupancy recovers part of the lost "
            "bandwidth, not all of it -- 52.8 pct against the 76.8 pct the kernel reaches "
            "one width below."
        ),
        "treatment_sweep": treatment,
        "control_sweep": control,
    }

    out_path = REPO / "AI/data/rmsnorm_fwd_occupancy_intervention.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--regs":
        _regs_inner(int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "--bw":
        _bw_inner(int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "--occ":
        _occ_inner(int(sys.argv[2]), sys.argv[3])
    else:
        main()

"""What the benchmark harness's provider asymmetry is actually worth.

`benchmarks/benchmark_rmsnorm_flydsl.py` times its two providers at different
levels: `_QuackProvider` calls `quack.rmsnorm.rmsnorm_fwd`, `_FlyDSLProvider`
calls `quack.rmsnorm_flydsl._launch_rmsnorm_fwd`. This probe measures all four
levels of both backends so the asymmetry can be attributed to a boundary rather
than guessed at.

The boundaries, read from the source rather than assumed:

  quack `_rmsnorm_fwd`   -- the custom op; caller preallocates every output
  quack `rmsnorm_fwd`    -- allocates out / rstd / residual_out, checks
                            weight_offset, delegates. No reshape, no autograd.
  quack `rmsnorm`        -- reshapes to 2-D and calls RMSNormFunction.apply
  flydsl `_launch_..`    -- the launcher; caller preallocates
  flydsl `rmsnorm`       -- guards operands, reshapes, autograd

An earlier version of the notes described the harness gap as "kernel versus
kernel-plus-wrapper" and put quack's reshape and autograd inside `rmsnorm_fwd`.
That is wrong: those live only in the top-level `rmsnorm()`. @Reviewer caught
it. The real asymmetry is a preallocated FlyDSL launcher against an allocating
CuTe forward wrapper.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_harness_levels.py
Writes AI/data/rmsnorm_harness_levels.json.
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import quack.rmsnorm_flydsl as flydsl_rmsnorm  # noqa: E402  (must follow the sys.path insert)

# The cutedsl backend imports `cuda.bindings.driver`, which does not exist on
# ROCm, so on this box quack's levels cannot be timed at all -- the import
# fails before any kernel runs. Recorded rather than skipped: the whole point
# of this probe is that an unmeasured level must not look like a measured one.
try:
    import quack.rmsnorm as quack_rmsnorm

    QUACK_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 -- any import failure must be recorded
    quack_rmsnorm = None
    QUACK_IMPORT_ERROR = f"{type(error).__name__}: {error}"

SHAPES = [(32768, 4096), (8192, 4096), (1024, 1024), (256, 512), (64, 256)]
ROUNDS = 7
REPS = 50
WARMUP = 20


def _bench(call, rounds=ROUNDS, reps=REPS):
    """Min-of-rounds of mean-over-reps, in microseconds.

    Min across rounds rejects interference; the mean within a round is what a
    single round can resolve given the event timer's granularity. Every round
    is returned so the spread is auditable rather than summarised away.
    """
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(reps):
            call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / reps)
    return samples


def _levels(m, n, dtype=torch.bfloat16):
    x = torch.randn((m, n), device="cuda", dtype=dtype)
    weight = torch.randn(n, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    quack_out = torch.empty_like(x)
    rstd = torch.empty(0, device="cuda", dtype=torch.float32)
    # The sentinel the harness itself passes for an unused operand: an empty
    # tensor, not None. Matching it keeps this probe measuring the same call.
    absent = torch.empty(0, device="cuda", dtype=dtype)

    calls = {}
    if quack_rmsnorm is not None:
        calls["quack._rmsnorm_fwd"] = lambda: quack_rmsnorm._rmsnorm_fwd(
            x, weight, quack_out, None, None, None, None, None, 1e-6, False, 0.0
        )
        calls["quack.rmsnorm_fwd"] = lambda: quack_rmsnorm.rmsnorm_fwd(x, weight, eps=1e-6)
        calls["quack.rmsnorm"] = lambda: quack_rmsnorm.rmsnorm(x, weight, eps=1e-6)
    calls["flydsl._launch_rmsnorm_fwd"] = lambda: flydsl_rmsnorm._launch_rmsnorm_fwd(
        x,
        weight,
        absent,
        absent,
        out,
        absent,
        rstd,
        1e-6,
        0.0,
        has_weight=True,
        has_bias=False,
        has_residual=False,
        store_residual=False,
        store_rstd=False,
        per_head=False,
        num_heads=1,
    )
    calls["flydsl.rmsnorm"] = lambda: flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)

    result = {}
    for label, call in calls.items():
        try:
            samples = _bench(call)
        except Exception as error:  # noqa: BLE001 -- recorded, not skipped
            result[label] = {"error": f"{type(error).__name__}: {error}"}
            continue
        result[label] = {
            "us_min_of_rounds": min(samples),
            "us_median_of_rounds": statistics.median(samples),
            "us_max_of_rounds": max(samples),
            "raw_rounds_us": samples,
        }
    return result


def _sha(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()[:16]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.manual_seed(0)
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    payload = {
        "what": "cost of each call level, both backends, so the harness's "
        "provider asymmetry can be attributed rather than assumed",
        "device": torch.cuda.get_device_name(),
        "device_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", None)),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "host": platform.node(),
        "commit": head,
        "worktree_dirty": bool(dirty),
        "source_sha256_16": {
            "quack/rmsnorm.py": _sha("quack/rmsnorm.py"),
            "benchmarks/benchmark_rmsnorm_flydsl.py": _sha(
                "benchmarks/benchmark_rmsnorm_flydsl.py"
            ),
            "quack/rmsnorm_flydsl.py": _sha("quack/rmsnorm_flydsl.py"),
            "AI/probe_rmsnorm_harness_levels.py": _sha("AI/probe_rmsnorm_harness_levels.py"),
        },
        "protocol": {
            "dtype": "bfloat16",
            "warmup_calls": WARMUP,
            "rounds": ROUNDS,
            "reps_per_round": REPS,
            "statistic": "min over rounds of (mean over reps); all rounds retained",
            "timer": "torch.cuda.Event, recorded around a rep loop",
            "exclusivity": "single visible device; caller is responsible for it "
            "being idle. rocm-smi utilisation at start is recorded below.",
        },
        "exclusivity_check": subprocess.run(
            ["rocm-smi", "--showuse"], capture_output=True, text=True, check=False
        ).stdout,
        "levels": {
            "quack._rmsnorm_fwd": "custom op; all outputs preallocated by caller",
            "quack.rmsnorm_fwd": "allocates out/rstd/residual_out, checks "
            "weight_offset, delegates. No reshape, no autograd.",
            "quack.rmsnorm": "reshape to 2-D + RMSNormFunction.apply",
            "flydsl._launch_rmsnorm_fwd": "launcher; outputs preallocated by caller",
            "flydsl.rmsnorm": "operand guards + reshape + autograd",
        },
        "quack_backend_import_error": QUACK_IMPORT_ERROR,
        "quack_levels_measured": QUACK_IMPORT_ERROR is None,
        "harness_pairing": {
            "_QuackProvider": "quack.rmsnorm_fwd",
            "_FlyDSLProvider": "flydsl._launch_rmsnorm_fwd",
        },
        "measurements": {},
    }

    for m, n in SHAPES:
        key = f"{m}x{n}"
        print(f"measuring {key} ...", flush=True)
        payload["measurements"][key] = _levels(m, n)

    out_path = REPO / "AI/data/rmsnorm_harness_levels.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Does one event pair per call over-read, and by how much? Measured, not asserted.

Why this exists
---------------
`benchmarks/benchmark_rmsnorm_flydsl.py` times a whole rotation with a single
`torch.cuda.Event` pair and divides by the number of calls in it. The docstring
justifying that choice, and the `methodology.steady_state` string written into
every future `environment.json`, both carried a specific number:

    per-call pairs read 178% high on a 6us kernel and 9% high on a 29us one,
    while one pair around the rotation is within 5% of both

That number was true when I measured it, but I measured it in a shell and threw
the shell away. @Reviewer's blocker 4 is that a machine-readable methodology
string should not assert a diagnostic with no committed artifact behind it, and
he is right: an unarchived 178% is indistinguishable from a remembered one.
This probe re-derives it against rocprofv3 hardware timestamps and writes a
sidecar, so the claim in the harness is checkable from the commit.

What is ground truth here
-------------------------
rocprofv3's `--kernel-trace` reports `Start_Timestamp` / `End_Timestamp` per
dispatch from the hardware's own clock. That is the kernel's execution time. It
does not include launch overhead, and it is not what a user experiences; it is
the right reference for the narrower question this probe asks, which is whether
the *timer* inflates the kernel it brackets.

Three quantities per shape:

  hardware   median of rocprofv3 per-dispatch durations for the measured kernel
  per_call   median of one-event-pair-per-call readings
  per_rot    median of (one pair around a rotation of K calls) / K

`per_call / hardware - 1` is the over-read the harness avoids. `per_rot /
hardware - 1` is what it accepts instead.

The profiler perturbs
---------------------
Attaching rocprofv3 is not free, so the event-derived numbers taken *under* the
profiler are not necessarily the ones the harness sees. Every phase is
therefore run twice, profiled and unprofiled, and both event medians are
recorded. If they disagree the sidecar shows it rather than hiding it, and the
over-read should be read against the profiled pair, which is self-consistent.

Event quantum
-------------
A separate section answers the second unarchived claim -- that ~8us readings
at `512x4096` sit at "the hipEvent quantum". It records the distinct values
`elapsed_time` returns over many repeats of a fixed short kernel, so the
spacing of the achievable readings is visible directly. If the spacing is
~1us the quantum reading is supported; if the values are dense the claim is
not, and the note must say so instead.

Run:
    python3 AI/probe_event_timing_calibration.py --json OUT.json
Each phase re-execs this file as a child (`--phase`), because a clean
rocprofv3 attach is per-process.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile

# Shapes chosen to span the range where the answer changes: a launch-bound
# cell, a mid cell, and one long enough that per-call overhead should wash out.
SHAPES = ((512, 4096), (4096, 4096), (32768, 1024))
CALLS_PER_ROTATION = 4
WARMUP = 20
ROUNDS = 60
# `elapsed_time` returns milliseconds; the quantum sweep needs many repeats of
# one short kernel to expose the spacing of achievable values.
QUANTUM_REPEATS = 400


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        raise ValueError("no samples")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _phase(mode, m, n):
    """Child process: time one shape one way, print JSON on stdout.

    Kept in one file so the timed code is textually identical between modes
    apart from where the event pair sits.
    """
    import torch

    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    def call():
        torch.nn.functional.rms_norm(x, (n,), w, 1e-6)

    for _ in range(WARMUP):
        for _ in range(CALLS_PER_ROTATION):
            call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us = []
    raw_reads_ms = []
    if mode == "per_call":
        for _ in range(ROUNDS):
            for _ in range(CALLS_PER_ROTATION):
                start.record()
                call()
                end.record()
                end.synchronize()
                ms = start.elapsed_time(end)
                raw_reads_ms.append(ms)
                samples_us.append(ms * 1000.0)
    elif mode == "per_rotation":
        for _ in range(ROUNDS):
            start.record()
            for _ in range(CALLS_PER_ROTATION):
                call()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            raw_reads_ms.append(ms)
            samples_us.append(ms * 1000.0 / CALLS_PER_ROTATION)
    else:
        raise ValueError(mode)

    torch.cuda.synchronize()
    # The count of dispatches the profiler should attribute to the timed
    # section, so the join can take exactly the trailing window.
    timed_dispatches = ROUNDS * CALLS_PER_ROTATION
    print(
        json.dumps(
            {
                "mode": mode,
                "m": m,
                "n": n,
                "median_us": _median(samples_us),
                "samples_us": samples_us,
                "raw_elapsed_time_ms": raw_reads_ms,
                "timed_dispatches": timed_dispatches,
            }
        )
    )


def _phase_quantum(m, n):
    """Child process: the distinct values `elapsed_time` can return here."""
    import torch

    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    for _ in range(WARMUP):
        torch.nn.functional.rms_norm(x, (n,), w, 1e-6)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    reads_ms = []
    for _ in range(QUANTUM_REPEATS):
        start.record()
        torch.nn.functional.rms_norm(x, (n,), w, 1e-6)
        end.record()
        end.synchronize()
        reads_ms.append(start.elapsed_time(end))
    print(json.dumps({"m": m, "n": n, "reads_ms": reads_ms}))


def _run_child(args_list, profile_dir=None, tag=None):
    """Run a phase, optionally under rocprofv3, and return (payload, trace_rows)."""
    cmd = [sys.executable, os.path.abspath(__file__), *args_list]
    if profile_dir is not None:
        cmd = [
            "rocprofv3",
            "--kernel-trace",
            "--output-format",
            "csv",
            "-d",
            profile_dir,
            "-o",
            tag,
            "--",
            *cmd,
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # rocprofv3 prints its own banner; the payload is the last JSON line.
    payload = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
    if payload is None:
        raise RuntimeError(f"no JSON payload from {cmd}:\n{proc.stdout}\n{proc.stderr}")
    rows = None
    if profile_dir is not None:
        rows = _read_trace(os.path.join(profile_dir, f"{tag}_kernel_trace.csv"))
    return payload, rows


def _read_trace(path):
    import csv

    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _kernel_durations_us(rows, count):
    """Durations of the last `count` dispatches of the dominant kernel.

    The dominant kernel is the one dispatched most often; for this workload
    that is torch's `vectorized_layer_norm`. Taking the *trailing* window drops
    warmup and tensor initialization without needing markers, and asserting the
    count means a mismatch is an error rather than a silent partial read.
    """
    import collections

    by_name = collections.defaultdict(list)
    for row in rows:
        if row.get("Kind") != "KERNEL_DISPATCH":
            continue
        name = row["Kernel_Name"]
        by_name[name].append((int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0)
    if not by_name:
        raise RuntimeError("no kernel dispatches in trace")
    name, durations = max(by_name.items(), key=lambda kv: len(kv[1]))
    if len(durations) < count:
        raise RuntimeError(
            f"expected at least {count} dispatches of {name[:60]}, got {len(durations)}"
        )
    return name, durations[-count:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("per_call", "per_rotation", "quantum"))
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--json", default="AI/probe_event_timing_calibration.json")
    args = parser.parse_args()

    if args.phase == "quantum":
        _phase_quantum(args.m, args.n)
        return
    if args.phase:
        _phase(args.phase, args.m, args.n)
        return

    if subprocess.run(["which", "rocprofv3"], capture_output=True).returncode != 0:
        sys.exit("rocprofv3 not on PATH; this probe has no ground truth without it")

    results = []
    with tempfile.TemporaryDirectory() as profile_dir:
        for m, n in SHAPES:
            entry = {"m": m, "n": n, "calls_per_rotation": CALLS_PER_ROTATION}
            for mode in ("per_call", "per_rotation"):
                flags = ["--phase", mode, "--m", str(m), "--n", str(n)]
                profiled, rows = _run_child(flags, profile_dir=profile_dir, tag=f"{mode}_{m}_{n}")
                unprofiled, _ = _run_child(flags)
                kernel_name, durations = _kernel_durations_us(rows, profiled["timed_dispatches"])
                entry[mode] = {
                    "event_median_us_profiled": profiled["median_us"],
                    "event_median_us_unprofiled": unprofiled["median_us"],
                    "hardware_median_us": _median(durations),
                    "hardware_dispatches": len(durations),
                    "kernel": kernel_name[:80],
                    "event_samples_us": profiled["samples_us"],
                    "hardware_samples_us": durations,
                }
                over = (
                    entry[mode]["event_median_us_profiled"] / entry[mode]["hardware_median_us"]
                    - 1.0
                )
                entry[mode]["over_read_vs_hardware"] = over
                print(
                    f"{m}x{n} {mode:12s} event {entry[mode]['event_median_us_profiled']:7.3f} us"
                    f"  hw {entry[mode]['hardware_median_us']:7.3f} us"
                    f"  over-read {over * 100:+7.1f}%"
                    f"  (unprofiled event {entry[mode]['event_median_us_unprofiled']:7.3f})"
                )
            results.append(entry)

    quantum, _ = _run_child(["--phase", "quantum", "--m", "512", "--n", "4096"])
    reads_us = sorted({round(v * 1000.0, 6) for v in quantum["reads_ms"]})
    gaps = [round(b - a, 6) for a, b in zip(reads_us, reads_us[1:])]
    print(
        f"\nquantum sweep 512x4096: {len(quantum['reads_ms'])} reads, "
        f"{len(reads_us)} distinct values, smallest gap {min(gaps) if gaps else 0} us"
    )

    payload = {
        "environment": {
            "argv": sys.argv,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "rocprofv3_version": subprocess.run(
                ["rocprofv3", "--version"], capture_output=True, text=True
            ).stdout.strip(),
            "warmup_rounds": WARMUP,
            "rounds": ROUNDS,
            "calls_per_rotation": CALLS_PER_ROTATION,
            "quantum_repeats": QUANTUM_REPEATS,
        },
        "measurements": results,
        "event_quantum": {
            "m": quantum["m"],
            "n": quantum["n"],
            "reads_ms": quantum["reads_ms"],
            "distinct_us": reads_us,
            "gaps_us": gaps,
        },
    }
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"\nraw samples + environment -> {args.json}")


if __name__ == "__main__":
    main()

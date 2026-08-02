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

Checkable is not authenticated
------------------------------
The first sidecar was checkable -- @Reviewer recomputed from it and refuted a
figure I had quoted, which is what it was written for -- but it recorded
nothing that pinned it to a script, a commit, a torch build or a card. So the
arithmetic inside it could be verified while the question "is this a
measurement of the tree in front of me?" stayed open. `_provenance()` closes
that: script hash, source commit with a dirty flag, torch/HIP versions, and the
GPU's KFD `unique_id` and BDF rather than an ordinal a visibility mask
renumbers.

Two further gaps in the same vein. The reduction from a rocprofv3 trace to one
median discarded the trace, so a selection that dropped dispatches it should
not have would look identical to a clean one -- `_kernel_durations_us` now
returns a census (rows seen, dispatches per kernel name, how many leading ones
the trailing window dropped) and each trace's SHA-256 is recorded, with
`--keep-traces DIR` to retain the CSVs those hashes name. And every phase ran
once, which cannot show its own reproducibility: `--repeats` (default 5) runs
each phase as independent processes and stores every repeat, with the spread
across them. The first repeat stays at the top level under its existing field
names, so `over_read_vs_hardware` still means one process's profiled pair.

That last one was not cosmetic. With repeats, the `512x4096` per-call over-read
turned out to move 131-157% run to run, so the `+138%` this file's consumers
quoted was one draw printed to a precision the measurement does not have. Two
repeats were not enough to see it; the first two-run check put that spread at
3.7pp. Hence the default of 5.

Run:
    python3 AI/probe_event_timing_calibration.py --json OUT.json
    python3 AI/probe_event_timing_calibration.py --repeats 3 --keep-traces /tmp/tr
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


def _provenance():
    """What this sidecar needs to be re-derivable, not merely re-readable.

    @Reviewer's point about the first version: it was *checkable* -- he
    recomputed from it and refuted a figure I had quoted, which is exactly what
    it was written for -- but it was not *authenticated*. It recorded hostname,
    platform, argv and rocprofv3's version, and nothing that pins it to a
    script, a commit, a torch build or a card. So a reader could verify the
    arithmetic inside the file and still not know whether the file describes
    the code in front of them.

    That distinction matters here more than it usually would, because this
    sidecar's whole job is to be the thing a claim is checked against. An
    unpinned reference artifact has the same defect as an unarchived number,
    one step removed: it looks like evidence about this tree and may be
    evidence about a different one.

    Each field is a specific thing that could have differed:

      script_sha256   the probe that ran, not the probe now on disk
      source_commit   with `dirty`, since a hash on a modified tree names code
                      that was not executed
      torch/hip       the runtime, which decides what an event pair costs
      gpu             name, arch, and the KFD `unique_id` -- the physical card,
                      not an ordinal that a visibility mask renumbers

    Best-effort: a missing git or an older torch degrades a field to None
    rather than failing the run, and None is the honest record of "not
    determined here". What it must never do is fabricate.
    """
    import hashlib

    here = os.path.abspath(__file__)
    root = os.path.dirname(os.path.dirname(here))
    provenance = {"script_path": os.path.relpath(here, root)}

    try:
        with open(here, "rb") as handle:
            provenance["script_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        provenance["script_sha256"] = None

    def _git(*args):
        try:
            done = subprocess.run(
                ["git", "-C", root, *args], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    provenance["source_commit"] = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    provenance["source_dirty"] = None if status is None else bool(status)

    try:
        import torch

        provenance["torch_version"] = getattr(torch, "__version__", None)
        provenance["torch_hip"] = getattr(torch.version, "hip", None)
        props = torch.cuda.get_device_properties(0)
        raw = getattr(getattr(props, "uuid", None), "bytes", None)
        text = None
        if raw is not None:
            try:
                text = bytes(raw).decode("ascii")
            except (UnicodeDecodeError, TypeError, ValueError):
                text = None
        domain = getattr(props, "pci_domain_id", None)
        bus = getattr(props, "pci_bus_id", None)
        device = getattr(props, "pci_device_id", None)
        provenance["gpu"] = {
            "name": getattr(props, "name", None),
            "arch": getattr(props, "gcnArchName", None),
            # The same field the harness matches KFD nodes on. An ordinal
            # identifies a card only relative to whatever mask was set.
            "uuid": text,
            "bdf": (
                None if None in (domain, bus, device) else f"{domain:04x}:{bus:02x}:{device:02x}"
            ),
        }
    except Exception:  # noqa: BLE001 - provenance must never fail the run
        provenance.setdefault("torch_version", None)
        provenance.setdefault("torch_hip", None)
        provenance.setdefault("gpu", None)
    return provenance


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


def _run_child(args_list, profile_dir=None, tag=None, keep_traces=None):
    """Run a phase, optionally under rocprofv3, and return (payload, rows, trace)."""
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
    rows, trace = None, None
    if profile_dir is not None:
        path = os.path.join(profile_dir, f"{tag}_kernel_trace.csv")
        rows = _read_trace(path)
        trace = _trace_identity(path, keep_traces, tag)
    return payload, rows, trace


def _trace_identity(path, keep_traces, tag):
    """Name the trace the durations came out of.

    A hash of a file this run then deletes proves nothing by itself, and saying
    otherwise would be the same overclaim this whole probe exists to correct.
    What it is good for is *linkage*: with `--keep-traces` the retained CSV can
    be hashed and matched to this sidecar, and two runs' traces can be compared
    without either being trusted. The thing that makes the reduction auditable
    on its own is the census in `_kernel_durations_us`, not this.
    """
    import hashlib

    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    kept = None
    if keep_traces:
        import shutil

        os.makedirs(keep_traces, exist_ok=True)
        kept = os.path.join(keep_traces, f"{tag}_kernel_trace.csv")
        shutil.copyfile(path, kept)
    return {"sha256": digest.hexdigest(), "bytes": size, "kept_at": kept}


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

    Returns a census alongside the window, because this function is where the
    trace stops being auditable: everything downstream sees `count` numbers and
    cannot tell whether they came from a clean trace or from a selection that
    quietly dropped something it should not have. The census records what was
    seen and what was discarded -- how many dispatches each kernel name got, so
    a second kernel appearing is visible rather than silently losing the max();
    and how many leading dispatches the trailing window dropped, so "warmup"
    stays a number a reader can check rather than an assurance.
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
    census = {
        "trace_rows": len(rows),
        "dispatch_rows": sum(len(v) for v in by_name.values()),
        "dispatches_by_kernel": {k[:80]: len(v) for k, v in sorted(by_name.items())},
        "selected_kernel_dispatches": len(durations),
        "window_requested": count,
        "leading_dispatches_dropped": len(durations) - count,
    }
    return name, durations[-count:], census


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("per_call", "per_rotation", "quantum"))
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--json", default="AI/probe_event_timing_calibration.json")
    # A single run of a median cannot show its own reproducibility: one number
    # is consistent with a stable measurement and with a lucky one.
    #
    # Default 5 rather than 2, decided by running it. At 2 repeats the
    # `512x4096` per-call spread read 3.7pp and looked tight; at 5 it read
    # 24.6pp, because one run in five lands near 158% while the others cluster
    # around 136%. Two draws can miss a tail that changes what the number
    # means, so the default is the smallest count that showed the tail here.
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--keep-traces",
        default=None,
        help="directory to copy each rocprofv3 CSV into, so the hashes in the "
        "sidecar can be checked against the files they were taken from",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

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
                reps = []
                for rep in range(args.repeats):
                    tag = f"{mode}_{m}_{n}_r{rep}"
                    profiled, rows, trace = _run_child(
                        flags, profile_dir=profile_dir, tag=tag, keep_traces=args.keep_traces
                    )
                    unprofiled, _, _ = _run_child(flags)
                    kernel_name, durations, census = _kernel_durations_us(
                        rows, profiled["timed_dispatches"]
                    )
                    rep_entry = {
                        "event_median_us_profiled": profiled["median_us"],
                        "event_median_us_unprofiled": unprofiled["median_us"],
                        "hardware_median_us": _median(durations),
                        "hardware_dispatches": len(durations),
                        "kernel": kernel_name[:80],
                        "event_samples_us": profiled["samples_us"],
                        "hardware_samples_us": durations,
                        "trace": trace,
                        "trace_census": census,
                    }
                    rep_entry["over_read_vs_hardware"] = (
                        rep_entry["event_median_us_profiled"] / rep_entry["hardware_median_us"]
                        - 1.0
                    )
                    reps.append(rep_entry)

                # The first repeat stays at the top level under the names it has
                # always had, so the field every consumer already reads keeps
                # meaning one process's profiled pair rather than silently
                # becoming an average across processes. The rest sit beside it.
                entry[mode] = dict(reps[0])
                entry[mode]["repeats"] = reps
                spread = [r["over_read_vs_hardware"] for r in reps]
                entry[mode]["over_read_repeat_spread"] = (
                    None if len(spread) < 2 else max(spread) - min(spread)
                )
                # The range is what a prose citation should quote. Quoting
                # `over_read_vs_hardware` alone reports one draw to three
                # significant figures from a quantity whose repeats here span
                # 24.6pp at the launch-bound shape -- true of that run and
                # misleading as a property of the machine.
                entry[mode]["over_read_range"] = [min(spread), max(spread)]
                entry[mode]["over_read_median_across_repeats"] = _median(spread)
                over = entry[mode]["over_read_vs_hardware"]
                extra = ""
                if len(spread) > 1:
                    extra = (
                        f"  [{len(spread)} runs, spread {(max(spread) - min(spread)) * 100:.1f}pp]"
                    )
                print(
                    f"{m}x{n} {mode:12s} event {entry[mode]['event_median_us_profiled']:7.3f} us"
                    f"  hw {entry[mode]['hardware_median_us']:7.3f} us"
                    f"  over-read {over * 100:+7.1f}%"
                    f"  (unprofiled event {entry[mode]['event_median_us_unprofiled']:7.3f}){extra}"
                )
            results.append(entry)

    quantum, _, _ = _run_child(["--phase", "quantum", "--m", "512", "--n", "4096"])
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
            "repeats_per_phase": args.repeats,
            **_provenance(),
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

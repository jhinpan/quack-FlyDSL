# Copyright (c) 2026, Tri Dao.

"""Run and archive the complete reproducible FlyDSL RMSNorm benchmark.

This is the one-command entry point for the 11-shape, three-provider forward
and backward matrix. It keeps profiling and controlled timing in one process
so expensive autotune winners are reused, records the exact checkout and
toolchain, and fails closed on a dirty tree, wrong import, wrong GPU, failed
correctness gate, or noisy bandwidth canary.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

PROVIDERS = ["flydsl", "flydsl_tuned", "torch_compile"]
EXPECTED_SHAPES = 11


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
    ).strip()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@contextmanager
def _benchmark_environment(cache_dir: Path, config_dir: Path):
    names = (
        "TORCH_COMPILE_DYNAMIC",
        "FLYDSL_AUTOTUNE",
        "FLYDSL_AUTOTUNE_CACHE_DIR",
        "FLYDSL_AUTOTUNE_CONFIG_DIR",
    )
    caller_environment = {name: os.environ.get(name) for name in names}
    try:
        os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")
        os.environ["FLYDSL_AUTOTUNE"] = "1"
        os.environ["FLYDSL_AUTOTUNE_CACHE_DIR"] = str(cache_dir)
        os.environ["FLYDSL_AUTOTUNE_CONFIG_DIR"] = str(config_dir)
        yield
    finally:
        for name, value in caller_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _controlled_winner_reuse(benchmark):
    try:
        tune_once_then_disable = benchmark._tune_once_then_disable
    except AttributeError as exc:
        raise RuntimeError("benchmark module does not expose its controlled tuning hook") from exc

    def reuse_without_retuning(call):
        previous = os.environ.get("FLYDSL_AUTOTUNE")
        os.environ["FLYDSL_AUTOTUNE"] = "0"
        try:
            output = call()
        except BaseException:
            if previous is None:
                os.environ.pop("FLYDSL_AUTOTUNE", None)
            else:
                os.environ["FLYDSL_AUTOTUNE"] = previous
            raise
        return output, previous

    # Preserve the standalone benchmark's tune-once default. This reproducer
    # owns its process and has already profiled every controlled shape.
    benchmark._tune_once_then_disable = reuse_without_retuning
    try:
        yield
    finally:
        benchmark._tune_once_then_disable = tune_once_then_disable


def _aggregate(profile, controlled) -> dict:
    result = {}
    for operation in ("fwd", "bwd"):
        device_rows = profile[operation]
        public_rows = controlled[operation]
        torch_device = (
            device_rows[device_rows["provider"] == "torch_compile"]
            .set_index(["M", "N"])
            .sort_index()
        )
        torch_public = (
            public_rows[public_rows["provider"] == "torch_compile"]
            .set_index(["M", "N"])
            .sort_index()
        )
        operation_result = {}
        for provider in ("flydsl", "flydsl_tuned"):
            ours_device = (
                device_rows[device_rows["provider"] == provider].set_index(["M", "N"]).sort_index()
            )
            ours_public = (
                public_rows[public_rows["provider"] == provider].set_index(["M", "N"]).sort_index()
            )
            device_speedups = (torch_device["device_us"] / ours_device["device_us"]).tolist()
            public_speedups = (
                torch_public["public_median_us"] / ours_public["public_median_us"]
            ).tolist()
            device_intervals = (
                ours_device["device_p90_us"] < torch_device["device_p10_us"]
            ).tolist()
            public_intervals = (
                ours_public["public_p90_us"] < torch_public["public_p10_us"]
            ).tolist()
            operation_result[provider] = {
                "device_geomean_speedup_vs_torch": math.exp(
                    sum(math.log(value) for value in device_speedups) / len(device_speedups)
                ),
                "device_wins": sum(value > 1.0 for value in device_speedups),
                "device_strict_gate_passes": sum(
                    value >= 1.02 and interval
                    for value, interval in zip(device_speedups, device_intervals)
                ),
                "public_geomean_speedup_vs_torch": math.exp(
                    sum(math.log(value) for value in public_speedups) / len(public_speedups)
                ),
                "public_wins": sum(value > 1.0 for value in public_speedups),
                "public_strict_gate_passes": sum(
                    value >= 1.02 and interval
                    for value, interval in zip(public_speedups, public_intervals)
                ),
            }
        result[operation] = operation_result
    return result


def _validate_frame(frame, name: str, expected_rows: int) -> None:
    if len(frame) != expected_rows:
        raise RuntimeError(f"{name}: expected {expected_rows} rows, found {len(frame)}")
    if set(frame["provider"]) != set(PROVIDERS):
        raise RuntimeError(f"{name}: incomplete providers: {set(frame['provider'])}")
    if frame.isnull().any().any():
        raise RuntimeError(f"{name}: output contains null values")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-arch", default="gfx950")
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--profile-rounds", type=int, default=5)
    parser.add_argument("--controlled-rounds", type=int, default=12)
    parser.add_argument("--forward-calls-per-sample", type=int, default=4)
    parser.add_argument("--backward-calls-per-sample", type=int, default=2)
    parser.add_argument("--rotation-buffers", type=int, default=2)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--probe-mib", type=int, default=512)
    parser.add_argument("--probe-samples", type=int, default=20)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow an uncommitted checkout; the diff is still recorded.",
    )
    return parser.parse_args()


def _run_benchmark_phases(benchmark, args, output_dir: Path) -> dict:
    if os.environ.get("FLYDSL_AUTOTUNE") != "1":
        raise RuntimeError("profile phase requires forced FlyDSL tuning")

    print(
        "PROFILE tuning mode active: forced tuning into output-local cache "
        f"{os.environ['FLYDSL_AUTOTUNE_CACHE_DIR']}"
    )
    frames = {"profile": {}, "controlled": {}}
    frames["profile"]["fwd"] = benchmark.run_profiled_shapes(
        benchmark.MN_PAIRS,
        backward=False,
        dtype_name="bfloat16",
        weight_mode="float32",
        providers=PROVIDERS,
        repeats=args.profile_repeats,
        rounds=args.profile_rounds,
        save_path=str(output_dir),
    )
    frames["profile"]["bwd"] = benchmark.run_profiled_shapes(
        benchmark.MN_PAIRS,
        backward=True,
        dtype_name="bfloat16",
        weight_mode="float32",
        providers=PROVIDERS,
        repeats=args.profile_repeats,
        rounds=args.profile_rounds,
        save_path=str(output_dir),
    )

    os.environ["FLYDSL_AUTOTUNE"] = "0"
    print(
        "REUSE mode active: controlled forward/backward use exact in-process "
        "flydsl_tuned winners with forced tuning disabled"
    )
    common_controlled = {
        "dtype_name": "bfloat16",
        "weight_mode": "float32",
        "rounds": args.controlled_rounds,
        "rotation_buffers": args.rotation_buffers,
        "settle_seconds": args.settle_seconds,
        "probe_mib": args.probe_mib,
        "probe_samples": args.probe_samples,
        "providers": PROVIDERS,
        "save_path": str(output_dir),
    }
    with _controlled_winner_reuse(benchmark):
        frames["controlled"]["fwd"] = benchmark.run_controlled_shapes(
            benchmark.MN_PAIRS,
            backward=False,
            calls_per_sample=args.forward_calls_per_sample,
            **common_controlled,
        )
        frames["controlled"]["bwd"] = benchmark.run_controlled_shapes(
            benchmark.MN_PAIRS,
            backward=True,
            calls_per_sample=args.backward_calls_per_sample,
            **common_controlled,
        )
    return frames


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    config_dir = output_dir / "config"
    cache_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    dirty = _git(repo_root, "status", "--porcelain")
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "refusing to benchmark a dirty checkout; commit it or pass --allow-dirty"
        )

    with _benchmark_environment(cache_dir, config_dir):
        _run_reproducer(args, repo_root, output_dir, dirty)

def _run_reproducer(args, repo_root: Path, output_dir: Path, dirty: str) -> None:
    import torch

    flydsl_distribution_version = importlib.metadata.version("flydsl")
    flydsl_spec = importlib.util.find_spec("flydsl")
    if flydsl_spec is None or flydsl_spec.origin is None:
        raise RuntimeError("unable to resolve the FlyDSL module origin")
    flydsl_module_origin = str(Path(flydsl_spec.origin).resolve())

    benchmark = importlib.import_module("benchmark_rmsnorm_flydsl")
    rmsnorm_module = importlib.import_module("quack.rmsnorm_flydsl")
    imported_path = Path(rmsnorm_module.__file__).resolve()
    if repo_root not in imported_path.parents:
        raise RuntimeError(
            f"wrong quack import: {imported_path}; expected checkout under {repo_root}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one idle GPU, for example HIP_VISIBLE_DEVICES=6")
    properties = torch.cuda.get_device_properties(0)
    arch = properties.gcnArchName.split(":", 1)[0]
    if arch != args.expected_arch:
        raise RuntimeError(f"expected {args.expected_arch}, found {arch}")

    environment_path = output_dir / "environment.json"
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    environment = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "repo_root": str(repo_root),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "git_dirty": bool(dirty),
        "git_diff": dirty,
        "imports": {
            "quack_rmsnorm_flydsl": str(imported_path),
            "flydsl_module_origin": flydsl_module_origin,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "flydsl_distribution": flydsl_distribution_version,
        },
        "gpu": {
            "name": properties.name,
            "architecture": arch,
            "index_within_visible_devices": 0,
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        },
        "contract": {
            "shapes": [list(shape) for shape in benchmark.MN_PAIRS],
            "providers": PROVIDERS,
            "correctness_gate": "required",
            "profile_repeats": args.profile_repeats,
            "profile_rounds": args.profile_rounds,
            "controlled_rounds": args.controlled_rounds,
            "rotation_buffers": args.rotation_buffers,
            "contention_canary_quiet_range": [0.95, 1.05],
            "tuned_winner_policy": {
                "profile": "forced tuning into output-local caches",
                "controlled": "forced tuning disabled; in-process winners reused",
            },
        },
    }
    _write_json(environment_path, environment)
    (output_dir / "reproduce-command.txt").write_text(
        f"cd {shlex.quote(str(repo_root))}\n"
        f"HIP_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=$PWD {command}\n"
    )

    frames = {"profile": {}, "controlled": {}}
    log_path = output_dir / "run.log"
    try:
        with log_path.open("w", buffering=1) as log:
            tee_out = _Tee(sys.stdout, log)
            tee_err = _Tee(sys.stderr, log)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                print(f"PIN git={environment['git_commit']} arch={arch}")
                print(
                    "PIN flydsl "
                    f"distribution={flydsl_distribution_version} "
                    f"module_origin={flydsl_module_origin}"
                )
                frames = _run_benchmark_phases(benchmark, args, output_dir)

        expected_rows = EXPECTED_SHAPES * len(PROVIDERS)
        for kind, operations in frames.items():
            for operation, frame in operations.items():
                _validate_frame(frame, f"{kind}-{operation}", expected_rows)
        summary = _aggregate(frames["profile"], frames["controlled"])
        _write_json(output_dir / "summary.json", summary)
        environment.update(
            {
                "status": "passed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result_rows": {
                    kind: {operation: len(frame) for operation, frame in operations.items()}
                    for kind, operations in frames.items()
                },
                "summary": summary,
            }
        )
        _write_json(environment_path, environment)
        print(f"PASS complete reproducible sweep: {output_dir}")
    except BaseException as exc:
        environment.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(environment_path, environment)
        raise


if __name__ == "__main__":
    main()

"""Profile RMSNorm backward stages and enforce before/after performance gates.

The production launcher emits the persistent partial kernel followed by the
FlyDSL parameter reducer. ROCm profiler events expose each kernel separately,
so this probe can gate both stages without changing the production API. It
also measures the equivalent PyTorch fp32 sum and output cast as a reference.

Run once before a cleanup and once after it:

    python AI/probe_rmsnorm_flydsl_backward_components.py --output /tmp/before.json
    python AI/probe_rmsnorm_flydsl_backward_components.py \
        --baseline /tmp/before.json --output /tmp/after.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from quack import rmsnorm_flydsl

DEFAULT_ROWS = (1, 64, 512, 4096, 32768)
DEFAULT_COLS = (2048, 8192)
DEFAULT_WEIGHT_DTYPES = ("bfloat16", "float32")
GATED_METRICS = ("partial_device_us", "reduce_device_us")


def _profile_device_us(
    call,
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {}
    for _ in range(rounds):
        with profile(activities=[ProfilerActivity.CUDA]) as profiler:
            for _ in range(iterations):
                call()
            torch.cuda.synchronize()

        components: dict[str, float] = {}
        for event in profiler.key_averages():
            if event.self_device_time_total <= 0:
                continue
            if "rmsnorm_bwd_partial_kernel" in event.key:
                name = "partial"
            elif "rmsnorm_bwd_dweight_reduce_kernel" in event.key:
                name = "reduce"
            elif "reduce_kernel" in event.key:
                name = "torch_sum"
            elif "copy_kernel" in event.key:
                name = "torch_cast"
            else:
                continue
            components[name] = components.get(name, 0.0) + (
                event.self_device_time_total / event.count
            )
        for name, value in components.items():
            samples.setdefault(name, []).append(value)
    return {name: statistics.median(values) for name, values in samples.items()}


def _synchronized_us(call, *, iterations: int, rounds: int) -> float:
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6 / iterations)
    return statistics.median(samples)


def _measure_cell(
    m: int,
    n: int,
    weight_dtype_name: str,
    *,
    profile_iterations: int,
    profile_rounds: int,
    rounds: int,
) -> dict[str, int | float | str]:
    device = torch.device("cuda", 0)
    weight_dtype = getattr(torch, weight_dtype_name)
    x = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    dout = torch.randn_like(x)
    weight = torch.randn(n, device=device, dtype=weight_dtype)
    rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6).contiguous()
    num_programs = rmsnorm_flydsl._select_rmsnorm_bwd_programs(
        m,
        n,
        "bf16",
        device,
    )
    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)
    partial = torch.randn(num_programs, n, device=device, dtype=torch.float32)

    def full_backward() -> None:
        rmsnorm_flydsl._launch_rmsnorm_bwd(x, weight, dout, rstd, dx, dweight)

    def torch_sum() -> torch.Tensor:
        return partial.sum(dim=0)

    def torch_finalize() -> torch.Tensor:
        return partial.sum(dim=0).to(weight_dtype)

    flydsl_events = _profile_device_us(
        full_backward,
        warmup=20,
        iterations=profile_iterations,
        rounds=profile_rounds,
    )
    sum_events = _profile_device_us(
        torch_sum,
        warmup=20,
        iterations=profile_iterations,
        rounds=profile_rounds,
    )
    finalize_events = _profile_device_us(
        torch_finalize,
        warmup=20,
        iterations=profile_iterations,
        rounds=profile_rounds,
    )
    missing = {"partial", "reduce"} - flydsl_events.keys()
    if missing:
        raise RuntimeError(f"profiler did not report FlyDSL components: {sorted(missing)}")

    full_iterations = 100 if m <= 4096 else 20
    return {
        "m": m,
        "n": n,
        "weight_dtype": weight_dtype_name,
        "num_programs": num_programs,
        "partial_device_us": flydsl_events["partial"],
        "reduce_device_us": flydsl_events["reduce"],
        "torch_sum_device_us": sum_events["torch_sum"],
        "torch_cast_device_us": finalize_events.get("torch_cast", 0.0),
        "full_us": _synchronized_us(
            full_backward,
            iterations=full_iterations,
            rounds=rounds,
        ),
    }


def _row_key(row: dict) -> tuple[int, int, str]:
    return int(row["m"]), int(row["n"]), str(row["weight_dtype"])


def _check_baseline(
    rows: list[dict],
    baseline_rows: list[dict],
    *,
    max_regression_percent: float,
    max_regression_us: float,
) -> list[str]:
    baseline_by_key = {_row_key(row): row for row in baseline_rows}
    failures = []
    for row in rows:
        key = _row_key(row)
        if key not in baseline_by_key:
            failures.append(f"{key}: missing from baseline")
            continue
        baseline = baseline_by_key[key]
        for metric in GATED_METRICS:
            before = float(baseline[metric])
            after = float(row[metric])
            tolerance = max(before * max_regression_percent / 100.0, max_regression_us)
            if after > before + tolerance:
                failures.append(
                    f"{key} {metric}: {before:.3f} -> {after:.3f} us "
                    f"(limit {before + tolerance:.3f})"
                )
    return failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=int, nargs="+", default=DEFAULT_COLS)
    parser.add_argument(
        "--weight-dtypes",
        nargs="+",
        choices=DEFAULT_WEIGHT_DTYPES,
        default=DEFAULT_WEIGHT_DTYPES,
    )
    parser.add_argument("--profile-iterations", type=int, default=100)
    parser.add_argument("--profile-rounds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression-percent", type=float, default=2.0)
    parser.add_argument("--max-regression-us", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    rows = []
    for weight_dtype in args.weight_dtypes:
        for n in args.cols:
            for m in args.rows:
                row = _measure_cell(
                    m,
                    n,
                    weight_dtype,
                    profile_iterations=args.profile_iterations,
                    profile_rounds=args.profile_rounds,
                    rounds=args.rounds,
                )
                rows.append(row)
                print(
                    f"{weight_dtype:>8} M={m:<7} N={n:<5} P={row['num_programs']:<4} "
                    f"partial={row['partial_device_us']:.2f}us "
                    f"reduce={row['reduce_device_us']:.2f}us "
                    f"torch={row['torch_sum_device_us'] + row['torch_cast_device_us']:.2f}us "
                    f"full={row['full_us']:.2f}us"
                )
                torch.cuda.empty_cache()

    artifact = {
        "schema_version": 1,
        "gpu": {
            "name": properties.name,
            "arch": properties.gcnArchName,
            "compute_units": properties.multi_processor_count,
        },
        "torch": torch.__version__,
        "rows": rows,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        failures = _check_baseline(
            rows,
            baseline["rows"],
            max_regression_percent=args.max_regression_percent,
            max_regression_us=args.max_regression_us,
        )
        if failures:
            raise SystemExit(
                "performance regressions:\n" + "\n".join(f"  {item}" for item in failures)
            )
        print("performance gate passed")


if __name__ == "__main__":
    main()

"""Controlled reproduction for the PR7 wide-row RMSNorm headline cells."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

import quack.rmsnorm_flydsl as rmsnorm_impl
from quack.rmsnorm_flydsl import rmsnorm

EPS = 1e-6


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_f32 = x.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + EPS)
    return (x_f32 * rstd * weight.float()).to(x.dtype)


def _backward_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_f32 = x.float()
    weight_f32 = weight.float()
    dout_f32 = dout.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + EPS)
    weighted = dout_f32 * weight_f32
    correction = (weighted * x_f32).mean(dim=-1, keepdim=True)
    dx = rstd * (weighted - x_f32 * rstd.square() * correction)
    dweight = (dout_f32 * x_f32 * rstd).sum(dim=0)
    return dx.to(x.dtype), dweight.to(weight.dtype)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "p10_us": _percentile(values, 0.1),
        "p90_us": _percentile(values, 0.9),
        "min_us": min(values),
        "max_us": max(values),
    }


def _time_batch(call, items: list, calls: int, offset: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for index in range(calls):
        output = call(items[(offset + index) % len(items)])
    end.record()
    end.synchronize()
    if output is None:
        raise AssertionError("timed batch executed no calls")
    return start.elapsed_time(end) * 1000.0 / calls


def _bandwidth_probe(
    probe_mib: int,
    *,
    warmup_rounds: int,
    sample_rounds: int,
    calls_per_sample: int,
) -> dict:
    num_bytes = probe_mib * 1024**2
    elements = num_bytes // 2
    left = torch.empty(elements, device="cuda", dtype=torch.bfloat16).fill_(1)
    right = torch.empty_like(left).fill_(2)
    out = torch.empty_like(left)
    probes = {
        "copy": (lambda: out.copy_(left), 2 * num_bytes),
        "two_read_one_write": (lambda: torch.add(left, right, out=out), 3 * num_bytes),
        "write": (lambda: out.zero_(), num_bytes),
    }
    results = {}
    for name, (call, logical_bytes) in probes.items():
        for _ in range(warmup_rounds):
            call()
        torch.cuda.synchronize()
        samples = []
        for _ in range(sample_rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(calls_per_sample):
                call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / calls_per_sample)
        median_us = statistics.median(samples)
        results[name] = {
            **_summary(samples),
            "logical_gbps": logical_bytes / (median_us * 1e-6) / 1e9,
        }
    best = max(results, key=lambda name: results[name]["logical_gbps"])
    return {
        "probe_mib": probe_mib,
        "calls_per_sample": calls_per_sample,
        "best_probe": best,
        "median_gbps": results[best]["logical_gbps"],
        "probes": results,
    }


def _settle(calls: dict, items: list, seconds: float) -> int:
    deadline = time.monotonic() + seconds
    count = 0
    names = list(calls)
    while time.monotonic() < deadline:
        for name in names if count % 2 == 0 else reversed(names):
            calls[name](items[count % len(items)])
            count += 1
        if count % 16 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return count


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_provenance() -> dict:
    repository = Path(__file__).resolve().parents[1]
    diff_scope = ("pyproject.toml", "quack", "benchmarks", "tests")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--", *diff_scope],
            check=True,
            capture_output=True,
            cwd=repository,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "git_dirty_paths": [],
            "git_diff_sha256": None,
            "git_diff_scope": list(diff_scope),
        }
    return {
        "git_commit": commit,
        "git_dirty_paths": [line[3:] for line in status.splitlines() if line],
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_diff_scope": list(diff_scope),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=262144)
    parser.add_argument("--operation", choices=("fwd", "bwd"), default="fwd")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--calls-per-sample", type=int, default=4)
    parser.add_argument("--rotation-buffers", type=int, default=2)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--probe-mib", type=int, default=512)
    parser.add_argument("--probe-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("isolate exactly one GPU with HIP_VISIBLE_DEVICES")
    properties = torch.cuda.get_device_properties(0)
    arch = properties.gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"expected gfx950, found {arch}")

    torch.manual_seed(0)
    free_before, total = torch.cuda.mem_get_info()
    inputs = [
        (torch.randn((args.m, args.n), device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_(
            args.operation == "bwd"
        )
        for _ in range(args.rotation_buffers)
    ]
    weight = (1.0 + torch.randn(args.n, device="cuda", dtype=torch.float32) * 0.1).requires_grad_(
        args.operation == "bwd"
    )

    torch._dynamo.reset()
    compiled_reference = torch.compile(_reference, dynamic=False)
    items = list(range(args.rotation_buffers))
    if args.operation == "fwd":
        calls = {
            "flydsl": lambda index: rmsnorm(inputs[index], weight, eps=EPS),
            "torch_compile": lambda index: compiled_reference(inputs[index], weight),
        }
        expected = (_reference(inputs[0], weight),)
    else:
        douts = [torch.randn_like(x) * 0.1 for x in inputs]
        flydsl_outputs = [rmsnorm(x, weight, eps=EPS) for x in inputs]
        torch_outputs = [compiled_reference(x, weight) for x in inputs]
        calls = {
            "flydsl": lambda index: torch.autograd.grad(
                flydsl_outputs[index],
                (inputs[index], weight),
                douts[index],
                retain_graph=True,
            ),
            "torch_compile": lambda index: torch.autograd.grad(
                torch_outputs[index],
                (inputs[index], weight),
                douts[index],
                retain_graph=True,
            ),
        }
        expected = _backward_reference(
            inputs[0].detach(),
            weight.detach(),
            douts[0],
        )

    outputs = {name: call(items[0]) for name, call in calls.items()}
    torch.cuda.synchronize()
    for name, output in outputs.items():
        actual_tensors = output if isinstance(output, tuple) else (output,)
        for actual, wanted in zip(actual_tensors, expected):
            tolerance = 2e-2 if args.operation == "fwd" else 3e-2
            torch.testing.assert_close(actual, wanted, rtol=tolerance, atol=tolerance)
        print(f"PASS correctness {name}", flush=True)
    del outputs, expected
    gc.collect()
    torch.cuda.empty_cache()

    settled_calls = _settle(calls, items, args.settle_seconds)
    opening = _bandwidth_probe(
        args.probe_mib,
        warmup_rounds=3,
        sample_rounds=args.probe_samples,
        calls_per_sample=8,
    )

    samples = {name: [] for name in calls}
    names = list(calls)
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for provider in order:
            samples[provider].append(
                _time_batch(
                    calls[provider],
                    items,
                    args.calls_per_sample,
                    round_index * args.calls_per_sample,
                )
            )

    closing = _bandwidth_probe(
        args.probe_mib,
        warmup_rounds=3,
        sample_rounds=args.probe_samples,
        calls_per_sample=8,
    )
    summaries = {name: _summary(values) for name, values in samples.items()}
    speedup = summaries["torch_compile"]["median_us"] / summaries["flydsl"]["median_us"]
    canary_ratio = closing["median_gbps"] / opening["median_gbps"]
    result = {
        "status": "passed",
        "operation": args.operation,
        "shape": [args.m, args.n],
        "dtype": "bfloat16",
        "weight_dtype": "float32",
        "rounds": args.rounds,
        "calls_per_sample": args.calls_per_sample,
        "rotation_buffers": args.rotation_buffers,
        "settle_seconds": args.settle_seconds,
        "settled_calls": settled_calls,
        "providers": summaries,
        "samples_us": samples,
        "torch_over_flydsl": speedup,
        "opening_bandwidth": opening,
        "closing_bandwidth": closing,
        "contention_canary": {
            "closing_over_opening": canary_ratio,
            "quiet": 0.95 <= canary_ratio <= 1.05,
        },
        "environment": {
            "hostname": platform.node(),
            **_git_provenance(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "flydsl": _package_version("flydsl"),
            "triton": _package_version("triton"),
            "gpu": properties.name,
            "arch": arch,
            "total_memory_bytes": total,
            "free_memory_bytes_at_start": free_before,
            "visibility": {
                name: os.environ.get(name)
                for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            },
            "quack_module": rmsnorm_impl.__file__,
            "script": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

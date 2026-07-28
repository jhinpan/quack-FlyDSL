# Copyright (c) 2026, Tri Dao.

"""Provider-lazy plain RMSNorm benchmark for the FlyDSL ROCm backend."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence
import warnings


COMPACT_SHAPES = (
    (1, 4096),
    (256, 4096),
    (512, 4096),
    (4096, 3000),
    (4096, 4096),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
)
DTYPE_WEIGHT_MODES = (
    ("float16", "same"),
    ("float16", "float32"),
    ("bfloat16", "same"),
    ("bfloat16", "float32"),
    ("float32", "same"),
)
OPERATIONS = ("fwd", "bwd")
PROVIDERS = ("flydsl", "quack", "torch")
RESULT_FIELDS = (
    "schema_version",
    "provider",
    "provider_detail",
    "operation",
    "m",
    "n",
    "activation_dtype",
    "weight_dtype",
    "weight_mode",
    "eps",
    "cold_compile_ms",
    "cold_compile_reused",
    "median_us",
    "p10_us",
    "p90_us",
    "logical_bytes",
    "logical_gbps",
    "peak_bw_gbps",
    "peak_bw_pct",
    "rotation_buffers",
    "rotation_working_set_bytes",
    "l2_target_bytes",
    "l2_eviction_between_calls",
    "timed_samples",
)

_ITEMSIZES = {"float16": 2, "bfloat16": 2, "float32": 4}


@dataclass(frozen=True)
class MatrixCell:
    m: int
    n: int
    activation_dtype: str
    weight_mode: str
    operation: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.m, self.n

    @property
    def weight_dtype(self) -> str:
        return self.activation_dtype if self.weight_mode == "same" else "float32"


@dataclass
class PreparedCase:
    calls: list[Callable[[], None]]
    reset: Callable[[], None]
    outputs: Callable[[], tuple[Any, ...]]
    provider_detail: str
    cold_compile_ms: float | None
    cold_compile_reused: bool


def build_matrix(
    shapes: Sequence[tuple[int, int]] = COMPACT_SHAPES,
    dtype_weight_modes: Sequence[tuple[str, str]] = DTYPE_WEIGHT_MODES,
    operations: Sequence[str] = OPERATIONS,
) -> list[MatrixCell]:
    """Build the distinct v1 matrix without duplicating fp32/fp32."""
    return [
        MatrixCell(m, n, activation_dtype, weight_mode, operation)
        for m, n in shapes
        for activation_dtype, weight_mode in dtype_weight_modes
        for operation in operations
    ]


def logical_bytes(
    operation: str,
    m: int,
    n: int,
    activation_itemsize: int,
    weight_itemsize: int,
) -> int:
    """Return provider-independent logical I/O bytes for plain RMSNorm."""
    activation_bytes = m * n * activation_itemsize
    weight_bytes = n * weight_itemsize
    if operation == "fwd":
        return 2 * activation_bytes + weight_bytes
    if operation == "bwd":
        # Read x, dout, weight, and fp32 rstd; write dx and dweight.
        return 3 * activation_bytes + 2 * weight_bytes + m * 4
    raise ValueError(f"unsupported operation: {operation}")


def write_artifacts(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    environment: dict[str, Any],
) -> tuple[Path, Path]:
    """Write the stable CSV result contract and environment JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    environment_path = output_dir / "environment.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, environment_path


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_us(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_us": _percentile(values, 0.5),
        "p10_us": _percentile(values, 0.1),
        "p90_us": _percentile(values, 0.9),
    }


def _rotation_count(
    bytes_per_set: int,
    target_bytes: int,
    free_bytes: int,
    *,
    min_buffers: int,
    max_buffers: int,
) -> int:
    by_target = max(1, math.ceil(target_bytes / max(1, bytes_per_set)))
    by_memory = max(1, int(free_bytes * 0.2) // max(1, bytes_per_set))
    desired = max(min_buffers, by_target)
    return max(1, min(max_buffers, by_memory, desired))


class _L2Evictor:
    def __init__(self, torch: Any, target_bytes: int):
        self.source = torch.empty(target_bytes, device="cuda", dtype=torch.uint8)
        self.destination = torch.empty_like(self.source)
        self.source.fill_(1)
        self.destination.zero_()
        torch.cuda.synchronize()

    def __call__(self) -> None:
        self.destination.copy_(self.source)


def _time_rotating_calls(
    torch: Any,
    prepared: PreparedCase,
    *,
    warmup_rounds: int,
    sample_rounds: int,
    evictor: _L2Evictor | None,
) -> list[float]:
    """Time a whole rotation with one event pair, not one pair per call.

    A hipEvent record carries barrier semantics, so bracketing every launch
    charges two pipeline drains to each kernel. Checked against rocprofv3's
    hardware timestamps on gfx950: per-call pairs read 178% high on a 6us
    kernel and 9% high on a 29us one, while one pair around the rotation is
    within 5% of both.

    The evictor runs outside the window. Keeping the operands out of L2 is the
    rotation's job -- ``_rotation_count`` sizes it against the L2 target for
    exactly that reason -- and the evictor only covers the case where memory
    capped the rotation short of it.
    """
    for _ in range(warmup_rounds):
        prepared.reset()
        if evictor is not None:
            evictor()
        for call in prepared.calls:
            call()
    torch.cuda.synchronize()

    calls_per_round = len(prepared.calls)
    samples_us = []
    for _ in range(sample_rounds):
        prepared.reset()
        if evictor is not None:
            evictor()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for call in prepared.calls:
            call()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / calls_per_round)
    return samples_us


def _measure_achievable_bandwidth(
    torch: Any,
    *,
    probe_bytes: int,
    warmup_rounds: int,
    sample_rounds: int,
) -> dict[str, Any]:
    """Best sustained bandwidth over several access patterns.

    A same-device copy alone understates the memory system badly enough that
    the RMSNorm forward exceeds it, which makes the resulting percentage
    useless as a ceiling. Probe a copy, a two-read one-write elementwise, and
    a pure write, and report the best; the elementwise probe is the closest
    match to what a normalization kernel actually does.
    """
    elements = probe_bytes // 2
    left = torch.empty(elements, device="cuda", dtype=torch.bfloat16).fill_(1)
    right = torch.empty_like(left).fill_(2)
    out = torch.empty_like(left)

    probes = {
        "copy": (lambda: out.copy_(left), 2 * probe_bytes),
        "two_read_one_write": (lambda: torch.add(left, right, out=out), 3 * probe_bytes),
        "write": (lambda: out.zero_(), probe_bytes),
    }
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    results = {}
    for name, (call, logical) in probes.items():
        for _ in range(warmup_rounds):
            call()
        torch.cuda.synchronize()
        samples_us = []
        for _ in range(sample_rounds):
            start.record()
            call()
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000.0)
        median_us = _percentile(samples_us, 0.5)
        results[name] = {
            "median_us": round(median_us, 6),
            "gbps": round(logical / (median_us * 1e-6) / 1e9, 6),
        }
    best = max(results, key=lambda name: results[name]["gbps"])
    return {
        "probe_bytes": probe_bytes,
        "probes": results,
        "best_probe": best,
        "median_gbps": results[best]["gbps"],
        "median_us": results[best]["median_us"],
        "samples": sample_rounds,
    }


def _torch_dtype(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _make_inputs(torch: Any, cell: MatrixCell, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    activation_dtype = _torch_dtype(torch, cell.activation_dtype)
    weight_dtype = _torch_dtype(torch, cell.weight_dtype)
    x = torch.randn((cell.m, cell.n), device="cuda", dtype=activation_dtype) * 0.5
    weight = 1.0 + torch.randn(cell.n, device="cuda", dtype=weight_dtype) * 0.1
    dout = torch.randn_like(x) * 0.1 if cell.operation == "bwd" else None
    return {"x": x, "weight": weight, "dout": dout}


def _reference(torch: Any, cell: MatrixCell, inputs: dict[str, Any], eps: float) -> tuple:
    with torch.no_grad():
        x = inputs["x"]
        weight = inputs["weight"]
        x_f32 = x.float()
        rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
        if cell.operation == "fwd":
            return ((x_f32 * rstd * weight.float()).to(x.dtype),)

        dout = inputs["dout"]
        dout_f32 = dout.float()
        weighted_dout = dout_f32 * weight.float()
        correction = (weighted_dout * x_f32).mean(dim=-1, keepdim=True)
        dx = (rstd * (weighted_dout - x_f32 * rstd.square() * correction)).to(x.dtype)
        dweight = (dout_f32 * x_f32 * rstd).sum(dim=0).to(weight.dtype)
        return dx, dweight, rstd.flatten()


def _assert_correct(torch: Any, cell: MatrixCell, actual: tuple, expected: tuple) -> None:
    if cell.operation == "fwd":
        tolerances = (2e-4, 2e-5) if cell.activation_dtype == "float32" else (2e-2, 2e-2)
    else:
        tolerances = (5e-3, 5e-3) if cell.activation_dtype == "float32" else (3e-2, 3e-2)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=tolerances[0],
            atol=tolerances[1],
        )


def _cold_launch(
    torch: Any,
    call: Callable[[], None],
    key: tuple,
    compile_timings: dict[tuple, float],
) -> tuple[float, bool]:
    reused = key in compile_timings
    torch.cuda.synchronize()
    started = time.perf_counter()
    call()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not reused:
        compile_timings[key] = elapsed_ms
    return compile_timings[key], reused


class _FlyDSLProvider:
    def __init__(self, torch: Any):
        self.torch = torch
        self.impl = importlib.import_module("quack.rmsnorm_flydsl")
        self.compile_timings: dict[tuple, float] = {}

    def prepare(
        self,
        cell: MatrixCell,
        inputs: dict[str, Any],
        reference: tuple,
        *,
        eps: float,
        rotation_buffers: int,
    ) -> PreparedCase:
        torch = self.torch
        if cell.operation == "fwd":
            tensor_sets = []
            calls = []
            for _ in range(rotation_buffers):
                x = inputs["x"].clone()
                weight = inputs["weight"].clone()
                out = torch.empty_like(x)
                rstd = torch.empty(0, device=x.device, dtype=torch.float32)
                tensor_sets.append((out,))

                def call(x=x, weight=weight, out=out, rstd=rstd):
                    self.impl._launch_rmsnorm_fwd(
                        x,
                        weight,
                        out,
                        rstd,
                        eps,
                        False,
                    )

                calls.append(call)

            compile_key = (
                "fwd",
                cell.n,
                cell.activation_dtype,
                cell.weight_dtype,
                eps,
            )
            cold_ms, reused = _cold_launch(
                torch,
                calls[0],
                compile_key,
                self.compile_timings,
            )
            _assert_correct(torch, cell, tensor_sets[0], reference)
            return PreparedCase(
                calls=calls,
                reset=lambda: None,
                outputs=lambda: tensor_sets[0],
                provider_detail="FlyDSL low-level forward",
                cold_compile_ms=cold_ms,
                cold_compile_reused=reused,
            )

        dtype_str = self.impl._dtype_to_str(inputs["x"].dtype)
        path, selected_programs = self.impl._select_rmsnorm_bwd_config(
            cell.m,
            cell.n,
            dtype_str,
            inputs["x"].device,
        )
        num_programs = selected_programs if path == "two_stage" else 0
        tensor_sets = []
        calls = []
        for _ in range(rotation_buffers):
            x = inputs["x"].clone()
            weight = inputs["weight"].clone()
            dout = inputs["dout"].clone()
            rstd = reference[2].clone()
            dx = torch.empty_like(x)
            if num_programs:
                raw_dweight = torch.empty_like(weight)
                partial = torch.empty(
                    num_programs * cell.n,
                    device=x.device,
                    dtype=torch.float32,
                )
            else:
                raw_dweight = torch.zeros(cell.n, device=x.device, dtype=torch.float32)
                partial = torch.empty(0, device=x.device, dtype=torch.float32)
            converted_dweight = [None]
            tensor_sets.append((dx, converted_dweight))

            def call(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                dx=dx,
                raw_dweight=raw_dweight,
                partial=partial,
                converted_dweight=converted_dweight,
            ):
                # The atomic path accumulates into dweight, so zeroing it is
                # part of the operation and has to be inside the timed region.
                if not num_programs:
                    raw_dweight.zero_()
                self.impl._launch_rmsnorm_bwd(
                    x,
                    weight,
                    dout,
                    rstd,
                    dx,
                    raw_dweight,
                    partial,
                    num_programs,
                )
                converted_dweight[0] = raw_dweight.to(weight.dtype)

            calls.append(call)

        def reset() -> None:
            return

        compile_key = (
            "bwd",
            path,
            cell.n,
            cell.activation_dtype,
            cell.weight_dtype,
            num_programs,
        )
        cold_ms, reused = _cold_launch(
            torch,
            calls[0],
            compile_key,
            self.compile_timings,
        )
        actual = tensor_sets[0][0], tensor_sets[0][1][0]
        _assert_correct(torch, cell, actual, reference[:2])
        return PreparedCase(
            calls=calls,
            reset=reset,
            outputs=lambda: (tensor_sets[0][0], tensor_sets[0][1][0]),
            provider_detail=f"FlyDSL low-level backward ({path})",
            cold_compile_ms=cold_ms,
            cold_compile_reused=reused,
        )


def _torch_rms_norm(torch: Any, x: Any, weight: Any, eps: float) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mismatch dtype between input and weight.*",
            category=UserWarning,
        )
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


class _TorchProvider:
    def __init__(self, torch: Any):
        self.torch = torch

    def prepare(
        self,
        cell: MatrixCell,
        inputs: dict[str, Any],
        reference: tuple,
        *,
        eps: float,
        rotation_buffers: int,
    ) -> PreparedCase:
        torch = self.torch
        tensor_sets = []
        calls = []
        if cell.operation == "fwd":
            for _ in range(rotation_buffers):
                x = inputs["x"].clone()
                weight = inputs["weight"].clone()
                output = [None]
                tensor_sets.append(output)

                def call(x=x, weight=weight, output=output):
                    output[0] = _torch_rms_norm(torch, x, weight, eps)

                calls.append(call)
            calls[0]()
            torch.cuda.synchronize()
            _assert_correct(torch, cell, (tensor_sets[0][0],), reference)
            return PreparedCase(
                calls=calls,
                reset=lambda: None,
                outputs=lambda: (tensor_sets[0][0],),
                provider_detail="torch.nn.functional.rms_norm",
                cold_compile_ms=None,
                cold_compile_reused=False,
            )

        for _ in range(rotation_buffers):
            x = inputs["x"].clone().requires_grad_(True)
            weight = inputs["weight"].clone().requires_grad_(True)
            dout = inputs["dout"].clone()
            output = _torch_rms_norm(torch, x, weight, eps)
            gradients = [None]
            tensor_sets.append(gradients)

            def call(
                x=x,
                weight=weight,
                dout=dout,
                output=output,
                gradients=gradients,
            ):
                gradients[0] = torch.autograd.grad(
                    output,
                    (x, weight),
                    dout,
                    retain_graph=True,
                )

            calls.append(call)
        torch.cuda.synchronize()
        calls[0]()
        torch.cuda.synchronize()
        _assert_correct(torch, cell, tensor_sets[0][0], reference[:2])
        return PreparedCase(
            calls=calls,
            reset=lambda: None,
            outputs=lambda: tensor_sets[0][0],
            provider_detail="torch.nn.functional.rms_norm autograd",
            cold_compile_ms=None,
            cold_compile_reused=False,
        )


class _QuackProvider:
    """Quack's own CuTe RMSNorm, so a CUDA box can be measured the same way.

    Calls the same low-level ``rmsnorm_fwd`` / ``rmsnorm_bwd`` entry points
    that ``benchmarks/benchmark_rmsnorm.py`` times, which is also the level
    the FlyDSL provider measures.
    """

    def __init__(self, torch: Any):
        self.torch = torch
        module = importlib.import_module("quack.rmsnorm")
        self._fwd = module.rmsnorm_fwd
        self._bwd = module.rmsnorm_bwd

    def prepare(
        self,
        cell: MatrixCell,
        inputs: dict[str, Any],
        reference: tuple,
        *,
        eps: float,
        rotation_buffers: int,
    ) -> PreparedCase:
        torch = self.torch
        tensor_sets = []
        calls = []
        if cell.operation == "fwd":
            for _ in range(rotation_buffers):
                x = inputs["x"].clone()
                weight = inputs["weight"].clone()
                slot = [None]
                tensor_sets.append(slot)

                def call(x=x, weight=weight, slot=slot):
                    slot[0] = self._fwd(x, weight, eps=eps)[0]

                calls.append(call)
            cold_start = time.perf_counter()
            calls[0]()
            torch.cuda.synchronize()
            cold_compile_ms = (time.perf_counter() - cold_start) * 1000.0
            _assert_correct(torch, cell, (tensor_sets[0][0],), reference)
            return PreparedCase(
                calls=calls,
                reset=lambda: None,
                outputs=lambda: (tensor_sets[0][0],),
                provider_detail="quack.rmsnorm.rmsnorm_fwd (CuTe)",
                cold_compile_ms=cold_compile_ms,
                cold_compile_reused=False,
            )

        for _ in range(rotation_buffers):
            x = inputs["x"].clone()
            weight = inputs["weight"].clone()
            dout = inputs["dout"].clone()
            # rmsnorm_fwd returns (out, residual_out, rstd).
            rstd = self._fwd(x, weight, eps=eps, store_rstd=True)[2]
            slot = [None]
            tensor_sets.append(slot)

            def call(x=x, weight=weight, dout=dout, rstd=rstd, slot=slot):
                dx, dweight, _, _ = self._bwd(x, weight, dout, rstd)
                slot[0] = (dx, dweight)

            calls.append(call)
        torch.cuda.synchronize()
        cold_start = time.perf_counter()
        calls[0]()
        torch.cuda.synchronize()
        cold_compile_ms = (time.perf_counter() - cold_start) * 1000.0
        _assert_correct(torch, cell, tensor_sets[0][0], reference[:2])
        return PreparedCase(
            calls=calls,
            reset=lambda: None,
            outputs=lambda: tensor_sets[0][0],
            provider_detail="quack.rmsnorm.rmsnorm_bwd (CuTe)",
            cold_compile_ms=cold_compile_ms,
            cold_compile_reused=False,
        )


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        m_text, n_text = value.lower().split("x", 1)
        shape = int(m_text), int(n_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must use MxN, for example 512x4096") from error
    if shape[0] < 1 or not 1 <= shape[1] <= 8192:
        raise argparse.ArgumentTypeError("shape requires M >= 1 and 1 <= N <= 8192")
    return shape


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmark_artifacts") / f"rmsnorm_flydsl_gfx950_{timestamp}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _device_arch(torch: Any) -> str:
    """Architecture string for the visible device, on either vendor.

    A CUDA build also exposes ``gcnArchName``, where it holds the marketing
    name, so the build is the discriminator rather than the attribute.
    """
    properties = torch.cuda.get_device_properties(0)
    if torch.version.hip is not None:
        return properties.gcnArchName.split(":", 1)[0]
    return f"sm_{properties.major}{properties.minor}"


def _environment(torch: Any, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 2,
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        # Every provider is checked against the fp32 reference before it is
        # timed, and a mismatch aborts the sweep. Recorded once here rather
        # than as a per-row column that could only ever say "passed".
        "correctness_gate": "required",
        "runtime_scope": f"{properties.name} / {_device_arch(torch)}",
        "comparison_scope": (
            "same-device providers. RMSNorm is memory bound, so a cross-vendor "
            "comparison of microseconds mostly reports the HBM bandwidth ratio; "
            "compare peak_bw_pct instead"
        ),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "git_commit": _git_commit(),
        "artifact_directory": str(output_dir.resolve()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "flydsl": _package_version("flydsl"),
            "quack": _package_version("quack-kernels") or _package_version("quack"),
        },
        "gpu": {
            "visible_index": 0,
            "visible_count": torch.cuda.device_count(),
            "name": properties.name,
            "arch": _device_arch(torch),
            "compute_units": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "l2_cache_bytes_reported": properties.L2_cache_size,
        },
        "visibility": {
            name: os.environ.get(name)
            for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        },
        "architecture_overrides": {
            name: os.environ.get(name) for name in ("ARCH", "FLYDSL_GPU_ARCH")
        },
        "host": {
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "matrix": {
            "shapes": [list(shape) for shape in args.shapes],
            "dtype_weight_modes": [list(mode) for mode in args.dtype_weight_modes],
            "operations": args.operations,
            "providers": args.providers,
            "eps": args.eps,
        },
        "methodology": {
            "steady_state": (
                "per-call torch.cuda device events after provider warmup; FlyDSL first-launch "
                "JIT is synchronized, recorded separately, and excluded"
            ),
            "cache": (
                "round-robin cloned tensor sets; when their logical working set is below "
                "the L2 target, a device copy evicts cache between individually timed calls"
            ),
            "logical_bytes": {
                "fwd": "read x + weight; write y",
                "bwd": "read x + dout + weight + fp32 rstd; write dx + dweight",
            },
            "peak_bandwidth": (
                "best of a same-device copy, a two-read one-write elementwise, and "
                "a pure write; a copy alone understates the memory system"
            ),
            "warmup_rounds": args.warmup_rounds,
            "sample_rounds": args.sample_rounds,
            "max_rotation_buffers": args.max_rotation_buffers,
            "l2_target_ratio": args.l2_target_ratio,
        },
        "result_schema": list(RESULT_FIELDS),
    }


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=None,
        help="Default: the vendor's own kernels plus torch (flydsl on ROCm, quack on CUDA)",
    )
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument(
        "--activation-dtypes",
        nargs="+",
        choices=tuple(_ITEMSIZES),
        default=list(_ITEMSIZES),
    )
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        choices=("same", "float32"),
        default=["same", "float32"],
    )
    parser.add_argument(
        "--shape",
        dest="shapes",
        action="append",
        type=_parse_shape,
        help="Restrict to an MxN shape; repeat for multiple shapes",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    # One sample per round now that a round is timed as a whole, so this is
    # also the sample count the percentiles are drawn from.
    parser.add_argument("--sample-rounds", type=int, default=40)
    parser.add_argument("--copy-mib", type=int, default=512)
    parser.add_argument("--copy-samples", type=int, default=30)
    parser.add_argument("--min-rotation-buffers", type=int, default=2)
    parser.add_argument("--max-rotation-buffers", type=int, default=4)
    parser.add_argument("--l2-target-ratio", type=int, default=3)
    parser.add_argument("--expected-arch", default="gfx950")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _validate_runtime(torch: Any, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU is visible")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark requires exactly one visible GPU; isolate it with HIP/ROCR visibility"
        )
    if args.providers is None:
        args.providers = ["flydsl" if torch.version.hip is not None else "quack", "torch"]
    actual_arch = _device_arch(torch)
    if actual_arch != args.expected_arch:
        raise RuntimeError(f"expected {args.expected_arch}, found {actual_arch}")
    if "flydsl" in args.providers and torch.version.hip is None:
        raise RuntimeError("the FlyDSL provider requires a ROCm PyTorch build")
    if "quack" in args.providers and torch.version.hip is not None:
        raise RuntimeError("the Quack CuTe provider requires a CUDA PyTorch build")
    if not math.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("--eps must be finite and positive")
    positive_values = {
        "--warmup-rounds": args.warmup_rounds,
        "--sample-rounds": args.sample_rounds,
        "--copy-mib": args.copy_mib,
        "--copy-samples": args.copy_samples,
        "--min-rotation-buffers": args.min_rotation_buffers,
        "--max-rotation-buffers": args.max_rotation_buffers,
        "--l2-target-ratio": args.l2_target_ratio,
    }
    for name, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if args.min_rotation_buffers > args.max_rotation_buffers:
        raise ValueError("--min-rotation-buffers cannot exceed --max-rotation-buffers")


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _run(
    torch: Any,
    args: argparse.Namespace,
    environment: dict[str, Any],
) -> list[dict[str, Any]]:
    properties = torch.cuda.get_device_properties(0)
    l2_target_bytes = properties.L2_cache_size * args.l2_target_ratio
    peak_bw = _measure_achievable_bandwidth(
        torch,
        probe_bytes=args.copy_mib * 1024**2,
        warmup_rounds=args.warmup_rounds,
        sample_rounds=args.copy_samples,
    )
    environment["achievable_bandwidth"] = peak_bw
    torch.cuda.empty_cache()
    evictor = _L2Evictor(torch, l2_target_bytes)

    providers = {}
    for name in args.providers:
        factory = {
            "flydsl": _FlyDSLProvider,
            "quack": _QuackProvider,
            "torch": _TorchProvider,
        }[name]
        providers[name] = factory(torch)

    cells = build_matrix(
        args.shapes,
        args.dtype_weight_modes,
        args.operations,
    )
    rows = []
    for cell_index, cell in enumerate(cells):
        inputs = _make_inputs(torch, cell, args.seed + cell_index)
        reference = _reference(torch, cell, inputs, args.eps)
        byte_count = logical_bytes(
            cell.operation,
            cell.m,
            cell.n,
            _ITEMSIZES[cell.activation_dtype],
            _ITEMSIZES[cell.weight_dtype],
        )
        for provider_name in args.providers:
            free_bytes, _ = torch.cuda.mem_get_info()
            rotation_buffers = _rotation_count(
                byte_count,
                l2_target_bytes,
                free_bytes,
                min_buffers=args.min_rotation_buffers,
                max_buffers=args.max_rotation_buffers,
            )
            prepared = providers[provider_name].prepare(
                cell,
                inputs,
                reference,
                eps=args.eps,
                rotation_buffers=rotation_buffers,
            )
            rotation_working_set_bytes = rotation_buffers * byte_count
            use_evictor = rotation_working_set_bytes < l2_target_bytes
            samples_us = _time_rotating_calls(
                torch,
                prepared,
                warmup_rounds=args.warmup_rounds,
                sample_rounds=args.sample_rounds,
                evictor=evictor if use_evictor else None,
            )
            stats = _summarize_us(samples_us)
            logical_gbps = byte_count / stats["median_us"] / 1000.0
            peak_bw_pct = logical_gbps / peak_bw["median_gbps"] * 100.0
            row = {
                "schema_version": 2,
                "provider": provider_name,
                "provider_detail": prepared.provider_detail,
                "operation": cell.operation,
                "m": cell.m,
                "n": cell.n,
                "activation_dtype": cell.activation_dtype,
                "weight_dtype": cell.weight_dtype,
                "weight_mode": cell.weight_mode,
                "eps": args.eps,
                "cold_compile_ms": _round(prepared.cold_compile_ms),
                "cold_compile_reused": prepared.cold_compile_reused,
                "median_us": _round(stats["median_us"]),
                "p10_us": _round(stats["p10_us"]),
                "p90_us": _round(stats["p90_us"]),
                "logical_bytes": byte_count,
                "logical_gbps": _round(logical_gbps),
                "peak_bw_gbps": _round(peak_bw["median_gbps"]),
                "peak_bw_pct": _round(peak_bw_pct),
                "rotation_buffers": rotation_buffers,
                "rotation_working_set_bytes": rotation_working_set_bytes,
                "l2_target_bytes": l2_target_bytes,
                "l2_eviction_between_calls": use_evictor,
                "timed_samples": len(samples_us),
            }
            rows.append(row)
            print(
                f"PASS {provider_name:6s} {cell.operation} "
                f"M={cell.m:<5d} N={cell.n:<4d} "
                f"{cell.activation_dtype}/{cell.weight_dtype}: "
                f"{stats['median_us']:.3f} us, {logical_gbps:.1f} GB/s, "
                f"{peak_bw_pct:.1f}% of peak BW",
                flush=True,
            )
            del prepared
        del inputs, reference
        torch.cuda.empty_cache()
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    args.shapes = args.shapes or list(COMPACT_SHAPES)
    args.dtype_weight_modes = [
        mode
        for mode in DTYPE_WEIGHT_MODES
        if mode[0] in args.activation_dtypes and mode[1] in args.weight_modes
    ]
    if not args.dtype_weight_modes:
        parser.error("dtype and weight-mode filters select no supported combinations")
    args.output_dir = args.output_dir or _default_output_dir()

    torch = importlib.import_module("torch")
    _validate_runtime(torch, args)
    environment = _environment(torch, args, args.output_dir)
    rows: list[dict[str, Any]] = []
    try:
        rows = _run(torch, args, environment)
    except Exception as error:
        environment["status"] = "failed"
        environment["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        csv_path, environment_path = write_artifacts(args.output_dir, rows, environment)
        print(f"Partial artifacts: {args.output_dir.resolve()}")
        print(f"CSV: {csv_path.resolve()}")
        print(f"Environment: {environment_path.resolve()}")
        raise

    # Contention canary: re-probe the memory system the sweep was normalised
    # against. A shared node can pick up a co-tenant partway through, and every
    # peak_bw_pct in the CSV is then measured against a ceiling that no longer
    # holds. Cheaper to record the drift than to discover it later.
    closing_bw = _measure_achievable_bandwidth(
        torch,
        probe_bytes=args.copy_mib * 1024**2,
        warmup_rounds=args.warmup_rounds,
        sample_rounds=args.copy_samples,
    )
    opening_gbps = environment["achievable_bandwidth"]["median_gbps"]
    closing_gbps = closing_bw["median_gbps"]
    drift = closing_gbps / opening_gbps
    environment["contention_canary"] = {
        "opening_gbps": _round(opening_gbps),
        "closing_gbps": _round(closing_gbps),
        "closing_over_opening": _round(drift),
        "quiet": 0.9 <= drift <= 1.1,
    }
    if not 0.9 <= drift <= 1.1:
        print(
            f"\nWARNING: achievable bandwidth moved {drift:.2f}x during the sweep "
            f"({opening_gbps:.0f} -> {closing_gbps:.0f} GB/s). "
            "The node was not quiet; treat these numbers as indicative only.",
            file=sys.stderr,
        )

    environment["status"] = "passed"
    environment["result_rows"] = len(rows)
    environment["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    csv_path, environment_path = write_artifacts(args.output_dir, rows, environment)
    print(f"Artifacts: {args.output_dir.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Environment: {environment_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

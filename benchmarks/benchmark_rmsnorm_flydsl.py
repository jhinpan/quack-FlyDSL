# Copyright (c) 2026, Tri Dao.

"""Benchmark rmsnorm fwd / bwd for the FlyDSL ROCm backend.

The CuTe backend's benchmark is benchmarks/benchmark_rmsnorm.py; this is the
same harness and shape ladder pointed at the FlyDSL backend.

The default perf-report sweep is for quick iteration. ``--controlled`` runs a
single shape with steady-state warmup, alternating provider order, batched
event timing, and an opening/closing bandwidth canary. ``--controlled-all``
applies that contract to the full shape ladder and writes a joinable CSV.
"""

import argparse
import gc
import json
import os
import statistics
import time

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
import torch._functorch.config as _functorch_config
from triton.testing import Benchmark, do_bench, perf_report

from quack.bench.bench_utils import run_and_print
from quack.rmsnorm_flydsl import rmsnorm, rmsnorm_autotuned

# Inductor's donated-buffer optimization is incompatible with retain_graph=True
# (used so we benchmark only bwd, not fwd+bwd). Disable it for the torch.compile
# bwd path. Must be set before torch.compile builds the bwd graph.
_functorch_config.donated_buffer = False


# Keep this list exactly aligned with benchmarks/benchmark_rmsnorm.py so ROCm
# and CUDA benchmark reports have the same rows.
MN_PAIRS = [
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
    (32768, 65536),
    (16384, 131072),
    (8192, 262144),
]

# A compact anti-overfit ladder: several row-count regimes without repeating
# the canonical M=32768-heavy search matrix. Run it with analytical providers
# unless explicitly investigating tuner generalization.
GENERALIZATION_PAIRS = [
    (63, 256),
    (511, 1024),
    (4097, 2048),
    (8191, 4096),
    (16385, 8192),
]

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

EPS = 1e-6


def rmsnorm_ref(x, w, eps=EPS):
    x_f32 = x.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_f32 * rstd * w.float()).to(x.dtype)


def rmsnorm_fused_ref(x, w, bias, residual, eps=EPS):
    """Residual add, norm, weight and bias, with the pre-norm sum also returned.

    The fused path is what the backend claims parity on, so it needs a cell of
    its own rather than only a plain one.
    """
    added = x.float() + residual.float()
    rstd = torch.rsqrt(added.square().mean(dim=-1, keepdim=True) + eps)
    out = (added * rstd * w.float() + bias.float()).to(x.dtype)
    return out, added.to(x.dtype)


def _result(num_bytes: int, ms: float) -> dict:
    gbps = num_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _bench(fn, **kwargs) -> float:
    return do_bench(fn, warmup=10, rep=100, **kwargs)


PROVIDER_NAMES = {
    "flydsl": "flydsl",
    "flydsl_tuned": "flydsl autotuned",
    "torch_compile": "torch.compile",
}


def _providers(names):
    return [(name, PROVIDER_NAMES[name]) for name in names]


def _compiled_ref(features: str = "plain"):
    """A torch.compile of the reference that is independent of earlier cells.

    Dynamo state carries across cells in one process: without the reset, the
    last shape of a sweep measures 9x slower than the same shape measured
    alone, reproducibly. The reset costs a recompile per cell and makes the
    sweep agree with per-cell runs.
    """
    torch._dynamo.reset()
    ref = rmsnorm_ref if features == "plain" else rmsnorm_fused_ref
    return torch.compile(ref, dynamic=False)


def _weight_dtype(dtype_name: str, weight_mode: str) -> torch.dtype:
    return DTYPE_MAP[dtype_name] if weight_mode == "same" else DTYPE_MAP[weight_mode]


def _mem_bytes(op: str, M: int, N: int, x: torch.Tensor, w: torch.Tensor, features: str) -> int:
    """Provider-independent logical I/O bytes.

    Logical rather than achieved: the backward's per-program dweight workspace
    is not counted. Both providers use the same formula, so the ratio between
    the columns is unaffected.
    """
    activation_bytes = M * N * x.dtype.itemsize
    weight_bytes = N * w.dtype.itemsize
    if op == "fwd":
        if features == "fused":
            # Read x and residual, write out and residual_out, read weight+bias.
            return 4 * activation_bytes + 2 * weight_bytes
        return 2 * activation_bytes + weight_bytes
    # Read x, dout, weight and the fp32 rstd; write dx and dweight.
    return 3 * activation_bytes + 2 * weight_bytes + M * 4


def _gate(op: str, dtype_name: str, actual, expected) -> None:
    """Check a provider against an fp32 reference before timing it.

    A wrong kernel is often a fast one, so nothing gets timed until it has
    matched. The tolerances are the activation dtype's own rounding.
    """
    if op == "fwd":
        rtol, atol = (2e-4, 2e-5) if dtype_name == "float32" else (2e-2, 2e-2)
    else:
        rtol, atol = (5e-3, 5e-3) if dtype_name == "float32" else (3e-2, 3e-2)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)


def _bwd_reference(x, w, dy, eps):
    x_f32, w_f32, dy_f32 = x.float(), w.float(), dy.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    weighted = dy_f32 * w_f32
    correction = (weighted * x_f32).mean(dim=-1, keepdim=True)
    dx = (rstd * (weighted - x_f32 * rstd.square() * correction)).to(x.dtype)
    dw = (dy_f32 * x_f32 * rstd).sum(dim=0).to(w.dtype)
    return dx, dw


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_us(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
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


def _bandwidth_probe(probe_mib: int, sample_rounds: int) -> dict:
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
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        samples = []
        for _ in range(sample_rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(8):
                call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / 8)
        median_us = statistics.median(samples)
        results[name] = logical_bytes / (median_us * 1e-6) / 1e9
    best = max(results, key=results.get)
    return {"best_probe": best, "gbps": results[best], "probes": results}


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


def _tune_once_then_disable(call):
    previous = os.environ.get("FLYDSL_AUTOTUNE")
    os.environ["FLYDSL_AUTOTUNE"] = "1"
    try:
        output = call()
    except Exception:
        if previous is None:
            os.environ.pop("FLYDSL_AUTOTUNE", None)
        else:
            os.environ["FLYDSL_AUTOTUNE"] = previous
        raise
    os.environ["FLYDSL_AUTOTUNE"] = "0"
    return output, previous


def _restore_autotune_env(previous) -> None:
    if previous is None:
        os.environ.pop("FLYDSL_AUTOTUNE", None)
    else:
        os.environ["FLYDSL_AUTOTUNE"] = previous


def _write_joined_contract(save_path: str, operation: str) -> None:
    """Join device/public intervals once both sweeps exist and apply the gate."""
    import pandas as pd

    profile_path = os.path.join(save_path, f"rmsnorm-flydsl-profile-{operation}.csv")
    controlled_path = os.path.join(save_path, f"rmsnorm-flydsl-controlled-{operation}.csv")
    if not (os.path.exists(profile_path) and os.path.exists(controlled_path)):
        return
    profile = pd.read_csv(profile_path)
    controlled = pd.read_csv(controlled_path)
    joined = controlled.merge(
        profile,
        on=["M", "N", "provider"],
        how="inner",
        validate="one_to_one",
    )
    joined["host_gap_us"] = (joined["public_median_us"] - joined["device_us"]).clip(lower=0)
    torch_rows = (
        joined[joined["provider"] == "torch_compile"].set_index(["M", "N"]).add_prefix("torch_")
    )
    contract_path = os.path.join(save_path, f"rmsnorm-flydsl-contract-{operation}.csv")
    if torch_rows.empty:
        joined.to_csv(contract_path, index=False)
        return
    for metric in (
        "device_us",
        "device_p10_us",
        "public_median_us",
        "public_p10_us",
    ):
        joined[f"torch_{metric}"] = [
            torch_rows.loc[(M, N), f"torch_{metric}"] for M, N in zip(joined["M"], joined["N"])
        ]
    joined["device_speedup_vs_torch"] = joined["torch_device_us"] / joined["device_us"]
    joined["public_speedup_vs_torch"] = (
        joined["torch_public_median_us"] / joined["public_median_us"]
    )
    joined["device_interval_win"] = joined["device_p90_us"] < joined["torch_device_p10_us"]
    joined["public_interval_win"] = joined["public_p90_us"] < joined["torch_public_p10_us"]
    joined["device_gate_pass"] = (joined["device_speedup_vs_torch"] >= 1.02) & joined[
        "device_interval_win"
    ]
    joined["public_gate_pass"] = (joined["public_speedup_vs_torch"] >= 1.02) & joined[
        "public_interval_win"
    ]
    joined.to_csv(contract_path, index=False)


def run_controlled(
    M: int,
    N: int,
    *,
    backward: bool,
    dtype_name: str,
    weight_mode: str,
    rounds: int,
    calls_per_sample: int,
    rotation_buffers: int,
    settle_seconds: float,
    probe_mib: int,
    probe_samples: int,
    providers: list[str],
) -> dict:
    """Run one correctness-gated, order-balanced steady-state comparison."""
    if torch.cuda.device_count() != 1:
        raise RuntimeError("isolate exactly one GPU with HIP_VISIBLE_DEVICES")
    properties = torch.cuda.get_device_properties(0)
    arch = properties.gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"expected gfx950, found {arch}")

    dtype = DTYPE_MAP[dtype_name]
    param_dtype = _weight_dtype(dtype_name, weight_mode)
    inputs = [
        (torch.randn((M, N), device="cuda", dtype=dtype) * 0.5).requires_grad_(backward)
        for _ in range(rotation_buffers)
    ]
    weight = (1.0 + torch.randn(N, device="cuda", dtype=param_dtype) * 0.1).requires_grad_(backward)
    items = list(range(rotation_buffers))
    calls = {}

    if not backward:
        if "flydsl" in providers:
            calls["flydsl"] = lambda index: rmsnorm(inputs[index], weight, eps=EPS)
        if "flydsl_tuned" in providers:
            calls["flydsl_tuned"] = lambda index: rmsnorm_autotuned(
                inputs[index],
                weight,
                eps=EPS,
            )
        if "torch_compile" in providers:
            compiled = _compiled_ref()
            calls["torch_compile"] = lambda index: compiled(inputs[index], weight, eps=EPS)
        expected = (rmsnorm_ref(inputs[0], weight, EPS),)
    else:
        douts = [torch.randn_like(x) * 0.1 for x in inputs]
        if "flydsl" in providers:
            flydsl_outputs = [rmsnorm(x, weight, eps=EPS) for x in inputs]
            calls["flydsl"] = lambda index: torch.autograd.grad(
                flydsl_outputs[index],
                (inputs[index], weight),
                douts[index],
                retain_graph=True,
            )
        if "flydsl_tuned" in providers:
            tuned_outputs = [rmsnorm_autotuned(x, weight, eps=EPS) for x in inputs]
            calls["flydsl_tuned"] = lambda index: torch.autograd.grad(
                tuned_outputs[index],
                (inputs[index], weight),
                douts[index],
                retain_graph=True,
            )
        if "torch_compile" in providers:
            compiled = _compiled_ref()
            torch_outputs = [compiled(x, weight, eps=EPS) for x in inputs]
            calls["torch_compile"] = lambda index: torch.autograd.grad(
                torch_outputs[index],
                (inputs[index], weight),
                douts[index],
                retain_graph=True,
            )
        expected = _bwd_reference(inputs[0].detach(), weight.detach(), douts[0], EPS)

    previous_autotune_env = None
    if "flydsl_tuned" in calls:
        _, previous_autotune_env = _tune_once_then_disable(lambda: calls["flydsl_tuned"](items[0]))

    for name, call in calls.items():
        output = call(items[0])
        actual = output if isinstance(output, tuple) else (output,)
        _gate("bwd" if backward else "fwd", dtype_name, actual, expected)
        print(f"PASS correctness {name}", flush=True)
    del output, expected
    gc.collect()
    torch.cuda.empty_cache()

    settled_calls = _settle(calls, items, settle_seconds)
    opening = _bandwidth_probe(probe_mib, probe_samples)
    torch.cuda.empty_cache()

    samples = {name: [] for name in calls}
    names = list(calls)
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for provider in order:
            samples[provider].append(
                _time_batch(
                    calls[provider],
                    items,
                    calls_per_sample,
                    round_index * calls_per_sample,
                )
            )

    closing = _bandwidth_probe(probe_mib, probe_samples)
    summaries = {name: _summarize_us(values) for name, values in samples.items()}
    canary = closing["gbps"] / opening["gbps"]
    quiet = 0.95 <= canary <= 1.05
    op = "bwd" if backward else "fwd"
    print(f"controlled {op} M={M} N={N} {dtype_name}/{param_dtype}:")
    for name, summary in summaries.items():
        print(
            f"  {name:13s} median={summary['median']:.3f}us "
            f"p10={summary['p10']:.3f}us p90={summary['p90']:.3f}us"
        )
    if "flydsl" in summaries:
        baseline = summaries["flydsl"]["median"]
        for name in ("flydsl_tuned", "torch_compile"):
            if name in summaries:
                print(f"  {name}/flydsl={summaries[name]['median'] / baseline:.3f}x")
    print(f"  settled with {settled_calls} provider calls")
    print(f"  BW {opening['gbps']:.1f}->{closing['gbps']:.1f} GB/s ({canary:.3f}x, quiet={quiet})")
    if not quiet:
        if "flydsl_tuned" in calls:
            _restore_autotune_env(previous_autotune_env)
        raise RuntimeError("bandwidth contention canary moved outside 0.95-1.05")
    if "flydsl_tuned" in calls:
        _restore_autotune_env(previous_autotune_env)
    return {
        "operation": op,
        "summaries": summaries,
        "opening_bandwidth": opening,
        "closing_bandwidth": closing,
        "contention_canary": canary,
        "settled_calls": settled_calls,
    }


def run_controlled_shapes(
    shapes,
    *,
    backward: bool,
    dtype_name: str,
    weight_mode: str,
    rounds: int,
    calls_per_sample: int,
    rotation_buffers: int,
    settle_seconds: float,
    probe_mib: int,
    probe_samples: int,
    providers: list[str],
    save_path: str | None,
):
    """Apply the controlled contract to every shape and persist raw intervals."""
    import pandas as pd

    rows = []
    for M, N in shapes:
        result = run_controlled(
            M,
            N,
            backward=backward,
            dtype_name=dtype_name,
            weight_mode=weight_mode,
            rounds=rounds,
            calls_per_sample=calls_per_sample,
            rotation_buffers=rotation_buffers,
            settle_seconds=settle_seconds,
            probe_mib=probe_mib,
            probe_samples=probe_samples,
            providers=providers,
        )
        opening = result["opening_bandwidth"]
        closing = result["closing_bandwidth"]
        for provider, summary in result["summaries"].items():
            rows.append(
                {
                    "operation": result["operation"],
                    "M": M,
                    "N": N,
                    "provider": provider,
                    "public_median_us": round(summary["median"], 3),
                    "public_p10_us": round(summary["p10"], 3),
                    "public_p90_us": round(summary["p90"], 3),
                    "contention_canary": round(result["contention_canary"], 6),
                    "opening_bandwidth_gbps": round(opening["gbps"], 3),
                    "closing_bandwidth_gbps": round(closing["gbps"], 3),
                    "opening_best_probe": opening["best_probe"],
                    "closing_best_probe": closing["best_probe"],
                    "timed_samples": rounds,
                    "calls_per_sample": calls_per_sample,
                    "rotation_buffers": rotation_buffers,
                }
            )
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        op = "bwd" if backward else "fwd"
        frame.to_csv(
            os.path.join(save_path, f"rmsnorm-flydsl-controlled-{op}.csv"),
            index=False,
        )
        _write_joined_contract(save_path, op)
    return frame


def _profile_device_call(call, repeats: int, rounds: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    device_samples = []
    launch_samples = []
    component_samples: dict[str, list[float]] = {}
    for _ in range(rounds):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(repeats):
                call()
            torch.cuda.synchronize()
        device_events = [
            event
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and not event.name.startswith("## Call CompiledFxGraph")
        ]
        device_samples.append(sum(event.device_time_total for event in device_events) / repeats)
        launch_samples.append(len(device_events) / repeats)
        components = {}
        for event in device_events:
            components[event.name] = components.get(event.name, 0.0) + (
                event.device_time_total / repeats
            )
        for name, value in components.items():
            component_samples.setdefault(name, []).append(value)

    summary = _summarize_us(device_samples)
    return {
        "device_us": summary["median"],
        "device_p10_us": summary["p10"],
        "device_p90_us": summary["p90"],
        "gpu_events_per_call": statistics.median(launch_samples),
        "components": {
            name: round(statistics.median(values), 3)
            for name, values in sorted(component_samples.items())
        },
    }


def run_profiled_shapes(
    shapes,
    *,
    backward: bool,
    dtype_name: str,
    weight_mode: str,
    providers: list[str],
    repeats: int,
    rounds: int,
    save_path: str | None,
):
    """Profile raw GPU events after every provider is warm and tuning is complete."""
    import pandas as pd

    rows = []
    dtype = DTYPE_MAP[dtype_name]
    param_dtype = _weight_dtype(dtype_name, weight_mode)
    for M, N in shapes:
        x = torch.randn((M, N), device="cuda", dtype=dtype, requires_grad=backward)
        weight = torch.randn(N, device="cuda", dtype=param_dtype, requires_grad=backward)
        calls = {}
        if not backward:
            if "flydsl" in providers:
                calls["flydsl"] = lambda x=x, weight=weight: rmsnorm(x, weight, eps=EPS)
            if "flydsl_tuned" in providers:
                calls["flydsl_tuned"] = lambda x=x, weight=weight: rmsnorm_autotuned(
                    x,
                    weight,
                    eps=EPS,
                )
            if "torch_compile" in providers:
                compiled = _compiled_ref()
                calls["torch_compile"] = lambda compiled=compiled, x=x, weight=weight: compiled(
                    x, weight, eps=EPS
                )
            expected = (rmsnorm_ref(x, weight, EPS),)
        else:
            dout = torch.randn_like(x)
            if "flydsl" in providers:
                flydsl_output = rmsnorm(x, weight, eps=EPS)
                calls["flydsl"] = (
                    lambda flydsl_output=flydsl_output, x=x, weight=weight, dout=dout: (
                        torch.autograd.grad(
                            flydsl_output,
                            (x, weight),
                            dout,
                            retain_graph=True,
                        )
                    )
                )
            if "flydsl_tuned" in providers:
                tuned_output = rmsnorm_autotuned(x, weight, eps=EPS)
                calls["flydsl_tuned"] = (
                    lambda tuned_output=tuned_output, x=x, weight=weight, dout=dout: (
                        torch.autograd.grad(
                            tuned_output,
                            (x, weight),
                            dout,
                            retain_graph=True,
                        )
                    )
                )
            if "torch_compile" in providers:
                compiled = _compiled_ref()
                torch_output = compiled(x, weight, eps=EPS)
                calls["torch_compile"] = (
                    lambda torch_output=torch_output, x=x, weight=weight, dout=dout: (
                        torch.autograd.grad(
                            torch_output,
                            (x, weight),
                            dout,
                            retain_graph=True,
                        )
                    )
                )
            expected = _bwd_reference(x.detach(), weight.detach(), dout, EPS)

        previous_autotune_env = None
        if "flydsl_tuned" in calls:
            _, previous_autotune_env = _tune_once_then_disable(calls["flydsl_tuned"])
        for name, call in calls.items():
            actual = call()
            actual_tensors = actual if isinstance(actual, tuple) else (actual,)
            _gate("bwd" if backward else "fwd", dtype_name, actual_tensors, expected)
            profile_summary = _profile_device_call(call, repeats, rounds)
            rows.append(
                {
                    "M": M,
                    "N": N,
                    "provider": name,
                    "device_us": round(profile_summary["device_us"], 3),
                    "device_p10_us": round(profile_summary["device_p10_us"], 3),
                    "device_p90_us": round(profile_summary["device_p90_us"], 3),
                    "gpu_events_per_call": round(
                        profile_summary["gpu_events_per_call"],
                        2,
                    ),
                    "components_json": json.dumps(
                        profile_summary["components"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "profile_rounds": rounds,
                    "calls_per_round": repeats,
                }
            )
            print(
                f"PASS profile {name:14s} M={M:<5d} N={N:<6d} "
                f"{profile_summary['device_us']:.3f}us "
                f"[{profile_summary['device_p10_us']:.3f},"
                f" {profile_summary['device_p90_us']:.3f}], "
                f"{profile_summary['gpu_events_per_call']:.2f} GPU events/call",
                flush=True,
            )
        if "flydsl_tuned" in calls:
            _restore_autotune_env(previous_autotune_env)
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    times = frame.pivot(index=["M", "N"], columns="provider", values="device_us")
    launches = frame.pivot(
        index=["M", "N"],
        columns="provider",
        values="gpu_events_per_call",
    )
    print("\nDevice time per call (us):")
    print(times.to_string())
    print("\nGPU events per call:")
    print(launches.to_string())
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        op = "bwd" if backward else "fwd"
        frame.to_csv(
            os.path.join(save_path, f"rmsnorm-flydsl-profile-{op}.csv"),
            index=False,
        )
        _write_joined_contract(save_path, op)
    return frame


def make_benchmark(
    op: str,
    dtype_name: str,
    weight_mode: str,
    features: str,
    providers,
    x_vals=None,
):
    line_vals, line_names = zip(*_providers(providers))
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"rmsnorm-flydsl-{op}-{features}-{dtype_name}-w-{weight_mode}",
        args={"dtype_name": dtype_name, "weight_mode": weight_mode, "features": features},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def rmsnorm_fwd_runner(M, N, provider, dtype_name, weight_mode, features):
    dtype = DTYPE_MAP[dtype_name]
    param_dtype = _weight_dtype(dtype_name, weight_mode)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=param_dtype)

    if features == "plain":
        if provider == "flydsl":
            fn = lambda: rmsnorm(x, w, eps=EPS)
        elif provider == "flydsl_tuned":
            fn = lambda: rmsnorm_autotuned(x, w, eps=EPS)
        elif provider == "torch_compile":
            compiled = _compiled_ref()
            fn = lambda: compiled(x, w, eps=EPS)
        else:
            raise ValueError(provider)
        expected = (rmsnorm_ref(x, w, EPS),)
    else:
        bias = torch.randn(N, device="cuda", dtype=param_dtype)
        residual = torch.randn(M, N, device="cuda", dtype=dtype)
        if provider == "flydsl":
            fn = lambda: rmsnorm(x, w, bias=bias, residual=residual, eps=EPS, prenorm=True)
        elif provider == "flydsl_tuned":
            fn = lambda: rmsnorm_autotuned(
                x,
                w,
                bias=bias,
                residual=residual,
                eps=EPS,
                prenorm=True,
            )
        elif provider == "torch_compile":
            compiled = _compiled_ref("fused")
            fn = lambda: compiled(x, w, bias, residual, EPS)
        else:
            raise ValueError(provider)
        expected = rmsnorm_fused_ref(x, w, bias, residual, EPS)

    if provider == "flydsl_tuned":
        actual, previous = _tune_once_then_disable(fn)
        try:
            _gate("fwd", dtype_name, actual if isinstance(actual, tuple) else (actual,), expected)
            ms = _bench(fn)
        finally:
            _restore_autotune_env(previous)
    else:
        actual = fn()
        _gate("fwd", dtype_name, actual if isinstance(actual, tuple) else (actual,), expected)
        ms = _bench(fn)
    return _result(_mem_bytes("fwd", M, N, x, w, features), ms)


def rmsnorm_bwd_runner(M, N, provider, dtype_name, weight_mode, features):
    if features != "plain":
        raise ValueError("the fused cell is forward-only")
    dtype = DTYPE_MAP[dtype_name]
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(
        N, device="cuda", dtype=_weight_dtype(dtype_name, weight_mode), requires_grad=True
    )
    dy = torch.randn(M, N, device="cuda", dtype=dtype)

    if provider == "flydsl":
        forward = rmsnorm
    elif provider == "flydsl_tuned":
        forward = rmsnorm_autotuned
    elif provider == "torch_compile":
        forward = _compiled_ref()
    else:
        raise ValueError(provider)

    # Both providers are timed through autograd on the public entry point, so
    # the two columns include the same wrapper. The FlyDSL backend exposes only
    # rmsnorm(), with no low-level bwd entry to time instead.
    y = forward(x, w, eps=EPS)
    fn = lambda: torch.autograd.grad(y, [x, w], grad_outputs=dy, retain_graph=True)

    expected = _bwd_reference(x.detach(), w.detach(), dy, EPS)
    if provider == "flydsl_tuned":
        actual, previous = _tune_once_then_disable(fn)
        try:
            _gate("bwd", dtype_name, actual, expected)
            ms = _bench(fn)
        finally:
            _restore_autotune_env(previous)
    else:
        _gate("bwd", dtype_name, fn(), expected)
        ms = _bench(fn)
    return _result(_mem_bytes("bwd", M, N, x, w, features), ms)


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlyDSL rmsnorm fwd / bwd")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument(
        "--weight_dtype",
        default="same",
        choices=["same", *DTYPE_MAP],
        help="Weight dtype; 'same' follows --dtype",
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--features",
        default="plain",
        choices=["plain", "fused"],
        help="'fused' adds bias, residual and prenorm (forward only)",
    )
    parser.add_argument("--M", type=int, default=None, help="Bench a single M (requires --N)")
    parser.add_argument("--N", type=int, default=None, help="Bench a single N (requires --M)")
    parser.add_argument(
        "--generalization",
        action="store_true",
        help="Use the compact off-ladder M/N matrix instead of the canonical shapes",
    )
    parser.add_argument("--save_path", default=None)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=tuple(PROVIDER_NAMES),
        default=["flydsl", "torch_compile"],
    )
    parser.add_argument(
        "--controlled",
        action="store_true",
        help="Run one steady-state, order-balanced comparison instead of perf_report",
    )
    parser.add_argument(
        "--controlled-all",
        action="store_true",
        help="Run the controlled comparison for every benchmark shape",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use torch.profiler GPU timestamps after warmup and tuning",
    )
    parser.add_argument("--profile_repeats", type=int, default=10)
    parser.add_argument("--profile_rounds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--calls_per_sample", type=int, default=None)
    parser.add_argument("--rotation_buffers", type=int, default=2)
    parser.add_argument("--settle_seconds", type=float, default=3.0)
    parser.add_argument("--probe_mib", type=int, default=512)
    parser.add_argument("--probe_samples", type=int, default=20)
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    if args.generalization and args.M is not None:
        parser.error("--generalization cannot be combined with --M/--N")
    if args.backward and args.features != "plain":
        parser.error("--features fused is forward only")
    if "flydsl_tuned" in args.providers and os.environ.get("FLYDSL_AUTOTUNE") != "1":
        parser.error("flydsl_tuned requires FLYDSL_AUTOTUNE=1")
    selected_modes = int(args.controlled) + int(args.controlled_all) + int(args.profile)
    if selected_modes > 1:
        parser.error("--controlled, --controlled-all and --profile are mutually exclusive")
    if args.profile_repeats <= 0 or args.profile_rounds <= 0:
        parser.error("--profile_repeats and --profile_rounds must be positive")
    if args.controlled or args.controlled_all:
        if args.features != "plain":
            parser.error("controlled modes support only --features plain")
        if args.controlled_all and args.M is not None:
            parser.error("--controlled-all uses a shape ladder; omit --M/--N")
        positive = {
            "--rounds": args.rounds,
            "--rotation_buffers": args.rotation_buffers,
            "--settle_seconds": args.settle_seconds,
            "--probe_mib": args.probe_mib,
            "--probe_samples": args.probe_samples,
        }
        for name, value in positive.items():
            if value <= 0:
                parser.error(f"{name} must be positive")
        M, N = (8192, 262144) if args.M is None else (args.M, args.N)
        calls_per_sample = (
            args.calls_per_sample
            if args.calls_per_sample is not None
            else (2 if args.backward else 4)
        )
        if calls_per_sample <= 0:
            parser.error("--calls_per_sample must be positive")
        torch.manual_seed(0)
        kwargs = {
            "backward": args.backward,
            "dtype_name": args.dtype,
            "weight_mode": args.weight_dtype,
            "rounds": args.rounds,
            "calls_per_sample": calls_per_sample,
            "rotation_buffers": args.rotation_buffers,
            "settle_seconds": args.settle_seconds,
            "probe_mib": args.probe_mib,
            "probe_samples": args.probe_samples,
            "providers": args.providers,
        }
        if args.controlled_all:
            run_controlled_shapes(
                GENERALIZATION_PAIRS if args.generalization else MN_PAIRS,
                save_path=args.save_path,
                **kwargs,
            )
        else:
            run_controlled(M, N, **kwargs)
        return
    x_vals = (
        [(args.M, args.N)]
        if args.M is not None
        else (GENERALIZATION_PAIRS if args.generalization else None)
    )

    torch.manual_seed(0)

    op = "bwd" if args.backward else "fwd"
    if args.profile:
        run_profiled_shapes(
            x_vals if x_vals is not None else MN_PAIRS,
            backward=args.backward,
            dtype_name=args.dtype,
            weight_mode=args.weight_dtype,
            providers=args.providers,
            repeats=args.profile_repeats,
            rounds=args.profile_rounds,
            save_path=args.save_path,
        )
        return
    runner = rmsnorm_bwd_runner if args.backward else rmsnorm_fwd_runner
    bench = perf_report(
        make_benchmark(
            op,
            args.dtype,
            args.weight_dtype,
            args.features,
            args.providers,
            x_vals,
        )
    )(runner)

    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()

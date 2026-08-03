# Copyright (c) 2026, Tri Dao.

"""Benchmark rmsnorm fwd / bwd for the FlyDSL ROCm backend.

The CuTe backend's benchmark is benchmarks/benchmark_rmsnorm.py; this is the
same harness and shape ladder pointed at the FlyDSL backend.

The default perf-report sweep is for quick iteration. ``--controlled`` runs a
single shape with steady-state warmup, alternating provider order, batched
event timing, and an opening/closing bandwidth canary.
"""

import argparse
import gc
import os
import statistics
import time

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
import torch._functorch.config as _functorch_config
from triton.testing import Benchmark, do_bench, perf_report

from quack.bench.bench_utils import run_and_print
from quack.rmsnorm_flydsl import rmsnorm

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


def _providers():
    return [("flydsl", "flydsl"), ("torch_compile", "torch.compile")]


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
    compiled = _compiled_ref()
    items = list(range(rotation_buffers))

    if not backward:
        calls = {
            "flydsl": lambda index: rmsnorm(inputs[index], weight, eps=EPS),
            "torch_compile": lambda index: compiled(inputs[index], weight, eps=EPS),
        }
        expected = (rmsnorm_ref(inputs[0], weight, EPS),)
    else:
        douts = [torch.randn_like(x) * 0.1 for x in inputs]
        flydsl_outputs = [rmsnorm(x, weight, eps=EPS) for x in inputs]
        torch_outputs = [compiled(x, weight, eps=EPS) for x in inputs]
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
        expected = _bwd_reference(inputs[0].detach(), weight.detach(), douts[0], EPS)

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
    speedup = summaries["torch_compile"]["median"] / summaries["flydsl"]["median"]
    canary = closing["gbps"] / opening["gbps"]
    quiet = 0.95 <= canary <= 1.05
    op = "bwd" if backward else "fwd"
    print(f"controlled {op} M={M} N={N} {dtype_name}/{param_dtype}:")
    for name, summary in summaries.items():
        print(
            f"  {name:13s} median={summary['median']:.3f}us "
            f"p10={summary['p10']:.3f}us p90={summary['p90']:.3f}us"
        )
    print(f"  torch/FlyDSL={speedup:.3f}x after {settled_calls} settling calls")
    print(f"  BW {opening['gbps']:.1f}->{closing['gbps']:.1f} GB/s ({canary:.3f}x, quiet={quiet})")
    if not quiet:
        raise RuntimeError("bandwidth contention canary moved outside 0.95-1.05")
    return {
        "operation": op,
        "flydsl_us": summaries["flydsl"]["median"],
        "torch_compile_us": summaries["torch_compile"]["median"],
        "torch_over_flydsl": speedup,
        "contention_canary": canary,
    }


def make_benchmark(op: str, dtype_name: str, weight_mode: str, features: str, x_vals=None):
    line_vals, line_names = zip(*_providers())
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
        else:
            compiled = _compiled_ref()
            fn = lambda: compiled(x, w, eps=EPS)
        expected = (rmsnorm_ref(x, w, EPS),)
    else:
        bias = torch.randn(N, device="cuda", dtype=param_dtype)
        residual = torch.randn(M, N, device="cuda", dtype=dtype)
        if provider == "flydsl":
            fn = lambda: rmsnorm(x, w, bias=bias, residual=residual, eps=EPS, prenorm=True)
        else:
            compiled = _compiled_ref("fused")
            fn = lambda: compiled(x, w, bias, residual, EPS)
        expected = rmsnorm_fused_ref(x, w, bias, residual, EPS)

    if provider not in ("flydsl", "torch_compile"):
        raise ValueError(provider)

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
    elif provider == "torch_compile":
        forward = _compiled_ref()
    else:
        raise ValueError(provider)

    # Both providers are timed through autograd on the public entry point, so
    # the two columns include the same wrapper. The FlyDSL backend exposes only
    # rmsnorm(), with no low-level bwd entry to time instead.
    y = forward(x, w, eps=EPS)
    fn = lambda: torch.autograd.grad(y, [x, w], grad_outputs=dy, retain_graph=True)

    _gate("bwd", dtype_name, fn(), _bwd_reference(x.detach(), w.detach(), dy, EPS))
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
    parser.add_argument("--save_path", default=None)
    parser.add_argument(
        "--controlled",
        action="store_true",
        help="Run one steady-state, order-balanced comparison instead of perf_report",
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--calls_per_sample", type=int, default=None)
    parser.add_argument("--rotation_buffers", type=int, default=2)
    parser.add_argument("--settle_seconds", type=float, default=3.0)
    parser.add_argument("--probe_mib", type=int, default=512)
    parser.add_argument("--probe_samples", type=int, default=20)
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    if args.backward and args.features != "plain":
        parser.error("--features fused is forward only")
    if args.controlled:
        if args.features != "plain":
            parser.error("--controlled supports only --features plain")
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
        run_controlled(
            M,
            N,
            backward=args.backward,
            dtype_name=args.dtype,
            weight_mode=args.weight_dtype,
            rounds=args.rounds,
            calls_per_sample=calls_per_sample,
            rotation_buffers=args.rotation_buffers,
            settle_seconds=args.settle_seconds,
            probe_mib=args.probe_mib,
            probe_samples=args.probe_samples,
        )
        return
    x_vals = [(args.M, args.N)] if args.M is not None else None

    torch.manual_seed(0)

    op = "bwd" if args.backward else "fwd"
    runner = rmsnorm_bwd_runner if args.backward else rmsnorm_fwd_runner
    bench = perf_report(make_benchmark(op, args.dtype, args.weight_dtype, args.features, x_vals))(
        runner
    )

    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()

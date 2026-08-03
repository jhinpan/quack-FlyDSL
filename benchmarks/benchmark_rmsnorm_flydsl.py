# Copyright (c) 2026, Tri Dao.

"""Benchmark rmsnorm fwd / bwd for the FlyDSL ROCm backend.

The CuTe backend's benchmark is benchmarks/benchmark_rmsnorm.py; this is the
same harness and shape ladder pointed at the FlyDSL backend.

This perf-report sweep is for quick iteration. Use
``benchmarks/repro_pr7_wide.py`` for the wide-row headline: it adds
steady-state warmup, alternating provider order, raw samples, provenance, and
an opening/closing bandwidth canary.
"""

import argparse
import os

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
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    if args.backward and args.features != "plain":
        parser.error("--features fused is forward only")
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

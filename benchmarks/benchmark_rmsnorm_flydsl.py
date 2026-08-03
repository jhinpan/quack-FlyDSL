# Copyright (c) 2026, Tri Dao.

"""Benchmark rmsnorm fwd / bwd for the FlyDSL ROCm backend.

The CuTe backend's benchmark is benchmarks/benchmark_rmsnorm.py; this is the
same harness and shape ladder pointed at the FlyDSL backend.
"""

import argparse
import os

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch  # noqa: E402
import torch._functorch.config as _functorch_config  # noqa: E402
from triton.testing import Benchmark, do_bench, perf_report  # noqa: E402

from quack.bench.bench_utils import run_and_print  # noqa: E402
from quack.rmsnorm_flydsl import rmsnorm  # noqa: E402

# Inductor's donated-buffer optimization is incompatible with retain_graph=True
# (used so we benchmark only bwd, not fwd+bwd). Disable it for the torch.compile
# bwd path. Must be set before torch.compile builds the bwd graph.
_functorch_config.donated_buffer = False


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


def _result(num_bytes: int, ms: float) -> dict:
    gbps = num_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _bench(fn, **kwargs) -> float:
    return do_bench(fn, warmup=10, rep=100, **kwargs)


def _providers():
    return [("flydsl", "flydsl"), ("torch_compile", "torch.compile")]


def _compiled_ref():
    """A torch.compile of the reference that is independent of earlier cells.

    Dynamo state carries across cells in one process: without the reset, the
    last shape of a sweep measures 9x slower than the same shape measured
    alone, reproducibly. The reset costs a recompile per cell and makes the
    sweep agree with per-cell runs.
    """
    torch._dynamo.reset()
    return torch.compile(rmsnorm_ref, dynamic=False)


def _weight_dtype(dtype_name: str, weight_mode: str) -> torch.dtype:
    return DTYPE_MAP[dtype_name] if weight_mode == "same" else DTYPE_MAP[weight_mode]


def _mem_bytes(op: str, M: int, N: int, x: torch.Tensor, w: torch.Tensor) -> int:
    """Provider-independent logical I/O bytes for plain RMSNorm."""
    activation_bytes = M * N * x.dtype.itemsize
    weight_bytes = N * w.dtype.itemsize
    if op == "fwd":
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


def make_fwd_benchmark(dtype_name: str, weight_mode: str, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_providers())
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"rmsnorm-flydsl-fwd-{dtype_name}-w-{weight_mode}",
        args={"dtype_name": dtype_name, "weight_mode": weight_mode},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def make_bwd_benchmark(dtype_name: str, weight_mode: str, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_providers())
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"rmsnorm-flydsl-bwd-{dtype_name}-w-{weight_mode}",
        args={"dtype_name": dtype_name, "weight_mode": weight_mode},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def rmsnorm_fwd_runner(M, N, provider, dtype_name, weight_mode):
    dtype = DTYPE_MAP[dtype_name]
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=_weight_dtype(dtype_name, weight_mode))

    if provider == "flydsl":
        fn = lambda: rmsnorm(x, w, eps=EPS)
    elif provider == "torch_compile":
        compiled = _compiled_ref()
        fn = lambda: compiled(x, w, eps=EPS)
    else:
        raise ValueError(provider)

    _gate("fwd", dtype_name, (fn(),), (rmsnorm_ref(x, w, EPS),))
    ms = _bench(fn)
    return _result(_mem_bytes("fwd", M, N, x, w), ms)


def rmsnorm_bwd_runner(M, N, provider, dtype_name, weight_mode):
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
    ms = _bench(fn, grad_to_none=(x, w))
    return _result(_mem_bytes("bwd", M, N, x, w), ms)


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
    parser.add_argument("--M", type=int, default=None, help="Bench a single M (requires --N)")
    parser.add_argument("--N", type=int, default=None, help="Bench a single N (requires --M)")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    x_vals = [(args.M, args.N)] if args.M is not None else None

    torch.manual_seed(0)

    if args.backward:
        bench = perf_report(make_bwd_benchmark(args.dtype, args.weight_dtype, x_vals))(
            rmsnorm_bwd_runner
        )
    else:
        bench = perf_report(make_fwd_benchmark(args.dtype, args.weight_dtype, x_vals))(
            rmsnorm_fwd_runner
        )

    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()

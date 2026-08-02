"""Compare Quack's analytical-heuristic RMSNorm against its autotuned path.

The H100/H200 sweep in benchmark_rmsnorm_flydsl.py calls rmsnorm_fwd /
rmsnorm_bwd, which pick a launch config from the hand-written analytical
ladder. This probe times those same entry points against rmsnorm_fwd_tuned /
rmsnorm_bwd_tuned, whose @autotune decorator searches the exhaustive config
space, so the headroom left on the table is measurable rather than assumed.

Both sides do identical work: the tuned forward allocates its own output the
way rmsnorm_fwd does, and the tuned backward finishes with the same
dw_partial.sum(dim=0) that rmsnorm_bwd performs internally.
"""

import argparse
import json
import statistics
import time

import torch

from quack.rmsnorm import (
    get_sm_count,
    rmsnorm_bwd,
    rmsnorm_bwd_tuned,
    rmsnorm_fwd,
    rmsnorm_fwd_tuned,
)
from quack.rmsnorm_config import RmsNormBwdConfig, RmsNormFwdConfig

EPS = 1e-6


def bench(call, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples[len(samples) // 10]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", dest="shapes", action="append", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    props = torch.cuda.get_device_properties(0)
    arch_major = props.major
    rows = []

    for text in args.shapes:
        m_text, n_text = text.lower().split("x", 1)
        M, N = int(m_text), int(n_text)
        x = torch.randn(M, N, device=device, dtype=dtype)
        weight = torch.randn(N, device=device, dtype=dtype)
        dout = torch.randn(M, N, device=device, dtype=dtype)
        rstd = rmsnorm_fwd(x, weight, eps=EPS, store_rstd=True)[2]
        sm_count = get_sm_count(N, x.device)

        def fwd_analytical():
            rmsnorm_fwd(x, weight, eps=EPS)

        def fwd_tuned():
            out = torch.empty_like(x)
            rmsnorm_fwd_tuned(x, weight, out, eps=EPS)

        def bwd_analytical():
            rmsnorm_bwd(x, weight, dout, rstd)

        def bwd_tuned():
            dx = torch.empty_like(x)
            dw_partial = torch.empty((sm_count, N), device=device, dtype=torch.float32)
            rmsnorm_bwd_tuned(x, weight, dout, rstd, dx, dw_partial, has_dw_partial=True)
            dw_partial.sum(dim=0).to(weight.dtype)

        for op, analytical_call, tuned_call, tuned_fn, analytical_config in (
            (
                "fwd",
                fwd_analytical,
                fwd_tuned,
                rmsnorm_fwd_tuned,
                RmsNormFwdConfig.from_analytical_heuristic(N, 16, arch_major=arch_major),
            ),
            (
                "bwd",
                bwd_analytical,
                bwd_tuned,
                rmsnorm_bwd_tuned,
                RmsNormBwdConfig.from_analytical_heuristic(N, 16, 16, arch_major=arch_major),
            ),
        ):
            tune_start = time.perf_counter()
            tuned_call()
            torch.cuda.synchronize()
            tune_seconds = time.perf_counter() - tune_start

            analytical_us, _ = bench(analytical_call, args.warmup, args.iters)
            tuned_us, _ = bench(tuned_call, args.warmup, args.iters)
            winner = getattr(tuned_fn, "best_config", None)

            row = {
                "shape": f"{M}x{N}",
                "operation": op,
                "analytical_us": round(analytical_us, 3),
                "tuned_us": round(tuned_us, 3),
                "tuned_over_analytical": round(tuned_us / analytical_us, 4),
                "search_seconds": round(tune_seconds, 1),
                "analytical_config": str(analytical_config),
                "tuned_config": str(winner),
            }
            rows.append(row)
            print(
                f"{op} {M}x{N}: analytical {analytical_us:8.2f} us | "
                f"tuned {tuned_us:8.2f} us | tuned/analytical {row['tuned_over_analytical']:.4f} | "
                f"search {tune_seconds:.1f}s",
                flush=True,
            )
            print(f"    analytical config: {analytical_config}", flush=True)
            print(f"    tuned config:      {winner}", flush=True)

    payload = {
        "gpu": props.name,
        "arch": f"sm_{props.major}{props.minor}",
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

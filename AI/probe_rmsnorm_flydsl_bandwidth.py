"""Probe: FlyDSL RMSNorm against torch.compile, and the geometry behind it.

Backs the numbers in flydsl_rmsnorm_notes.md. The compiler is the baseline
that matters: it fuses this shape of work close to the roofline, and it is
what a caller falls back to when the backend refuses a combination.

Each shape prints the geometry both paths pick for it, because that is what
explains the number: how wide a vector is, how many threads cover a row, how
many passes they take, and how many rows share a block.

The short and coprime rows at the bottom are the interesting ones. A 128-wide
row is the QK-norm case. A row whose length shares no factor with a 128-bit
access cannot vectorize at all, and is where the two paths disagree.
"""

import statistics

import torch
import triton

from quack.flydsl.rmsnorm_config import (
    RmsNormRowConfig,
    batch_feature_rows,
    multi_row_block_rows,
)
from quack.rmsnorm_flydsl import rmsnorm


def bench_pair(left, right, rounds=7):
    """Time two variants against each other, interleaved, and take medians.

    Two things bite here. ``do_bench`` sizes its repeat count from a first
    call, so a first call that also runs the FlyDSL build skews the whole
    sample -- hence the untimed call before the loop. Separately, about one
    sample in twenty on this machine comes back an order of magnitude slow.

    Measuring one variant to completion and then the other cannot survive
    either problem: AGENTS.md records that sequential rounds on a shared node
    drift 2-3x with clocks and co-tenants, so whichever variant ran second is
    not comparable to the one that ran first. Alternating them cancels the
    drift, and a median over the rounds discards the occasional slow sample
    without taking the luckiest one, which is what a minimum would do.
    """
    left()
    right()
    torch.cuda.synchronize()
    left_samples, right_samples = [], []
    for index in range(rounds):
        order = (
            ((left, left_samples), (right, right_samples))
            if index % 2
            else ((right, right_samples), (left, left_samples))
        )
        for fn, samples in order:
            samples.append(triton.testing.do_bench(fn, warmup=25, rep=100))
    return statistics.median(left_samples), statistics.median(right_samples)


def tbs(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e12


def geometry(n, dtype_width, batched):
    config = (
        RmsNormRowConfig.for_lane_group(n, dtype_width)
        if batched
        else RmsNormRowConfig.from_analytical_heuristic(n, dtype_width)
    )
    rows = multi_row_block_rows(config.num_threads) if batched else 1
    return f"vec{config.vecsize} x{config.num_tiles} {config.num_threads}thr {rows}row/blk"


def eager_plain(x, w):
    v = x.float()
    return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)


def eager_offset(x, w):
    v = x.float()
    normalized = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
    return (normalized * (w.float() + 1.0)).to(x.dtype)


def eager_bias(x, w, b):
    v = x.float()
    normalized = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
    return (normalized * w.float() + b.float()).to(x.dtype)


def eager_residual(x, w, r):
    v = x.float() + r.float()
    out = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)
    return out, v.to(x.dtype)


compiled = {
    "plain": torch.compile(eager_plain, dynamic=False),
    "weight_offset=1": torch.compile(eager_offset, dynamic=False),
    "+bias": torch.compile(eager_bias, dynamic=False),
    "+residual+prenorm": torch.compile(eager_residual, dynamic=False),
}

SHAPES = [
    (8192, 4096),
    (32768, 2048),
    (524288, 64),
    (262144, 128),
    (131072, 256),
    (131072, 257),
    (16384, 2047),
]

for m, n in SHAPES:
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    row = m * n * x.element_size()

    print(f"\n=== {m}x{n} bf16 ===")
    print(f"  geometry {geometry(n, 16, batch_feature_rows(n, 16))}")
    print(f"{'case':>19} {'FlyDSL':>19} {'torch.compile':>19} {'speedup':>8}")
    print("-" * 68)

    cases = [
        ("plain", lambda: rmsnorm(x, w), (x, w), 2 * row),
        ("weight_offset=1", lambda: rmsnorm(x, w, weight_offset=1.0), (x, w), 2 * row),
        ("+bias", lambda: rmsnorm(x, w, b), (x, w, b), 2 * row),
        (
            "+residual+prenorm",
            lambda: rmsnorm(x, w, residual=r, prenorm=True),
            (x, w, r),
            4 * row,
        ),
    ]
    for label, fly, args, nbytes in cases:
        reference = compiled[label]
        flydsl, torch_compile = bench_pair(fly, lambda: reference(*args))
        print(
            f"{label:>19} {flydsl:8.4f}ms {tbs(nbytes, flydsl):5.2f}TB/s "
            f"{torch_compile:8.4f}ms {tbs(nbytes, torch_compile):5.2f}TB/s "
            f"{torch_compile / flydsl:7.2f}x"
        )

torch.cuda.synchronize()

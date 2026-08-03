# Copyright (c) 2026, Tri Dao.

"""Native FlyDSL autotuning for the RMSNorm forward direct-JIT entry."""

import statistics

import torch
from flydsl.autotune import Config, autotune

from .rmsnorm_config import MAX_NUM_THREADS, RmsNormRowConfig, batch_short_rows
from .rmsnorm_kernel import rmsnorm_direct


RMSNORM_AUTOTUNE_SCHEMA_VERSION = 2
_WAVES_PER_EU = (None, 1, 2, 4)


def _row_candidates(n: int, dtype_width: int) -> list[int]:
    heuristic = (
        RmsNormRowConfig.for_lane_group(n, dtype_width)
        if batch_short_rows(n, dtype_width)
        else RmsNormRowConfig.from_analytical_heuristic(n, dtype_width)
    )
    ceiling = 64 if batch_short_rows(n, dtype_width) else MAX_NUM_THREADS
    candidates = {heuristic.num_threads}
    candidates.update((heuristic.num_threads // 2, heuristic.num_threads * 2))
    if not batch_short_rows(n, dtype_width):
        candidates.update((64, 128, 256))

    legal = []
    for threads in sorted(candidates):
        if threads < 1 or threads > ceiling or threads & (threads - 1):
            continue
        # Wide candidates use the builder's gmem-reload path, so their total
        # row assignment no longer has to fit in registers.
        RmsNormRowConfig.with_num_threads(n, dtype_width, threads)
        legal.append(threads)
    return legal


def rmsnorm_search_configs(*args, **kwargs) -> list[Config]:
    """Return legal, deduplicated row-width and occupancy candidates."""
    n = int(kwargs["n"])
    dtype_width = 32 if kwargs["input_dtype_str"] == "f32" else 16
    configs = []
    seen = set()
    for threads in _row_candidates(n, dtype_width):
        for waves_per_eu in _WAVES_PER_EU:
            identity = (threads, waves_per_eu)
            if identity in seen:
                continue
            seen.add(identity)
            configs.append(Config(threads_per_row=threads, waves_per_eu=waves_per_eu))
    return configs


def rmsnorm_default_config(*args, **kwargs) -> Config:
    """Preserve the existing analytical geometry when tuning is not forced."""
    n = int(kwargs["n"])
    dtype_width = 32 if kwargs["input_dtype_str"] == "f32" else 16
    config = (
        RmsNormRowConfig.for_lane_group(n, dtype_width)
        if batch_short_rows(n, dtype_width)
        else RmsNormRowConfig.from_analytical_heuristic(n, dtype_width)
    )
    return Config(threads_per_row=config.num_threads)


def batched_event_bench(fn, warmup=1, rep=7):
    """Time batches of launches so event overhead cannot dominate tiny kernels."""
    batch_size = 100
    fn()  # Compile outside the timed region.
    torch.cuda.synchronize()
    for _ in range(warmup):
        for _ in range(batch_size):
            fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch_size):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / batch_size)
    return statistics.median(times)


_RMSNORM_AUTOTUNE_KEY = [
    "m",
    "n",
    "input_dtype_str",
    "output_dtype_str",
    "weight_dtype_str",
    "bias_dtype_str",
    "residual_dtype_str",
    "residual_out_dtype_str",
    "has_weight",
    "has_bias",
    "has_residual",
    "store_residual",
    "store_rstd",
    "per_head",
    "num_heads",
    "arch",
    "schema_version",
]

_rmsnorm_fwd_tuner = autotune(
    configs=rmsnorm_search_configs,
    key=_RMSNORM_AUTOTUNE_KEY,
    warmup=1,
    rep=7,
    do_bench=batched_event_bench,
    default=rmsnorm_default_config,
    artifact_name="quack_rmsnorm_fwd",
)(rmsnorm_direct)


__all__ = [
    "RMSNORM_AUTOTUNE_SCHEMA_VERSION",
    "_rmsnorm_fwd_tuner",
    "batched_event_bench",
    "rmsnorm_default_config",
    "rmsnorm_search_configs",
]

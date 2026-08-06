# Copyright (c) 2026, Tri Dao.

"""Native FlyDSL autotuning for the RMSNorm forward direct-JIT entry."""

import importlib
import os
import statistics
import threading
import types
import warnings
from contextlib import nullcontext
from dataclasses import dataclass

import flydsl.compiler as flyc
import torch
from flydsl.autotune import (
    Autotuner,
    Config,
    _device_fingerprint,
    _env_fingerprint,
    _toolchain_fingerprint,
    _tuning_enabled,
)
from flydsl.compiler import CompiledFunction
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import Constexpr
from flydsl.utils import env

from .rmsnorm_common import FLYDSL_BUILD_LOCK
from .rmsnorm_config import (
    MAX_TUNED_NUM_THREADS,
    RmsNormRowConfig,
    batch_short_rows,
)
from .rmsnorm_kernel import rmsnorm_direct

RMSNORM_AUTOTUNE_SCHEMA_VERSION = 5
_WAVES_PER_EU = (None, 1, 2, 4)
_PERSISTENT_FWD_CONFIGS = {
    256: (32, 8, 9, None, False, False, None),
    512: (64, 1, 56, None, False, False, None),
    1024: (64, 2, 64, 3, True, True, 7),
    4096: (512, 1, 56, None, False, False, None),
    8192: (512, 1, 56, None, False, False, None),
}
_L2_TARGET_RATIO = 3
_L2_TIMED_CALLS = 200
_L2_MAX_BUFFER_SETS = _L2_TIMED_CALLS
_L2_WARMUP_TARGET_MS = 200.0
_L2_REPLAY_SAMPLES = 3
_L2_MAX_EXTRA_BYTES = 4 * 1024**3
_L2_EVICTION_CHUNK_BYTES = 256 * 1024**2
_ROCM_TRITON_CACHE_BYTES = 256 * 1024**2
_CORRECTNESS_ROWS = 16
_AUTOTUNE_BENCH_LOCK = threading.Lock()
_FAST_CONTEXT_ENV_VARS = (
    "FLYDSL_COMPILE_BACKEND",
    "FLYDSL_COMPILE_LLVM_DIR",
    "FLYDSL_COMPILE_OPT_LEVEL",
    "FLYDSL_DEBUG_ENABLE_DEBUG_INFO",
    "FLYDSL_EXTRA_SOURCE_DIRS",
    "FLYDSL_GPU_ARCH",
    "FLYDSL_RUNTIME_KIND",
    "FLYDSL_AUTOTUNE_CONFIG_DIR",
    "ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
    "COMPILE_ONLY",
)
_FAST_CONTEXT_ENV_VARS_BYTES = tuple(os.fsencode(name) for name in _FAST_CONTEXT_ENV_VARS)


def _row_candidates(n: int, dtype_width: int) -> list[int]:
    heuristic = (
        RmsNormRowConfig.for_lane_group(n, dtype_width)
        if batch_short_rows(n, dtype_width)
        else RmsNormRowConfig.from_analytical_heuristic(n, dtype_width)
    )
    ceiling = 64 if batch_short_rows(n, dtype_width) else MAX_TUNED_NUM_THREADS
    candidates = {heuristic.num_threads}
    candidates.update((heuristic.num_threads // 2, heuristic.num_threads * 2))
    if not batch_short_rows(n, dtype_width):
        # 512 is above what the heuristic may pick: it wins only when the row
        # is wide and there are many of them, and only the tuner sees M.
        candidates.update((64, 128, 256, 512))
        # At N=32768, 1024 lanes bring the per-thread fragment back to the
        # existing 32-element register budget, removing the epilogue reload.
        # Neighboring rows either already fit at 512 or still stream, so keep
        # the full-workgroup candidate target-only until separately measured.
        if n == 32768:
            candidates.add(1024)

    legal = []
    for threads in sorted(candidates):
        if threads < 1 or threads > ceiling or threads & (threads - 1):
            continue
        config = RmsNormRowConfig.with_num_threads(
            n,
            dtype_width,
            threads,
            max_num_threads=ceiling,
        )
        # Wide candidates use the builder's gmem-reload path, so their total
        # row assignment no longer has to fit in registers. Still reject
        # blocks wider than the row's vector count because their idle lanes
        # can fool the tuner into selecting a much slower configuration.
        if config.num_vecs >= threads:
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
    persistent = _PERSISTENT_FWD_CONFIGS.get(n)
    if (
        persistent is not None
        and len(args) >= 8
        and int(args[7]) == 32768
        and kwargs.get("input_dtype_str") == "bf16"
        and kwargs.get("output_dtype_str") == "bf16"
        and kwargs.get("weight_dtype_str") == "f32"
        and kwargs.get("has_weight", False)
        and not kwargs.get("has_bias", False)
        and not kwargs.get("has_residual", False)
        and not kwargs.get("store_residual", False)
        and not kwargs.get("store_rstd", False)
        and not kwargs.get("per_head", False)
        and kwargs.get("num_heads", 1) == 1
    ):
        (
            threads,
            row_groups_per_block,
            programs_per_cu,
            output_cache_modifier,
            persistent_single_pass,
            packed_flat_rows,
            waves_per_eu,
        ) = persistent
        if packed_flat_rows and args[0].stride(0) != n:
            return configs
        num_cus = torch.cuda.get_device_properties(args[0].device).multi_processor_count
        available_blocks = (int(args[7]) + row_groups_per_block - 1) // row_groups_per_block
        candidate = {
            "threads_per_row": threads,
            "row_groups_per_block": row_groups_per_block,
            "persistent_programs": min(available_blocks, num_cus * programs_per_cu),
        }
        if output_cache_modifier is not None:
            candidate["output_cache_modifier"] = output_cache_modifier
        if persistent_single_pass:
            candidate["persistent_single_pass"] = True
        if packed_flat_rows:
            candidate["packed_flat_rows"] = True
        if waves_per_eu is not None:
            candidate["waves_per_eu"] = waves_per_eu
        configs.append(Config(**candidate))
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


@dataclass
class _CacheAuthority:
    """Cache capacity used to size the cold-address working set."""

    cache_bytes: int
    source: str
    seed_buffer: torch.Tensor | None = None


@dataclass
class _ReferenceSamples:
    row_indices: torch.Tensor
    output: torch.Tensor
    residual_out: torch.Tensor | None
    rstd: torch.Tensor | None


@dataclass
class _L2RotationPlan:
    """One search call's reusable tensor-address and eviction working set."""

    arg_sets: list[tuple]
    kwarg_sets: list[dict]
    cache_bytes: int
    read_bytes_per_set: int
    eviction_buffers: list[torch.Tensor]
    cache_source: str
    bench_stream: torch.cuda.Stream
    expected: _ReferenceSamples
    n_timed_calls: int
    per_call_eviction: bool = False
    fallback_reason: str | None = None
    fallback_warning_emitted: bool = False

    @property
    def rotation_bytes(self) -> int:
        return len(self.arg_sets) * self.read_bytes_per_set

    @property
    def eviction_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.eviction_buffers)

    @property
    def working_set_bytes(self) -> int:
        return self.rotation_bytes + self.eviction_bytes


def _tensor_storage_span_bytes(tensor: torch.Tensor) -> int:
    """Bytes allocated by an exact-stride clone with storage offset zero."""
    if tensor.numel() == 0:
        return 0
    span = 1
    for size, stride in zip(tensor.shape, tensor.stride()):
        span += (size - 1) * stride
    return span * tensor.element_size()


def _clone_tensor_arguments(args, kwargs):
    """Clone tensors to new addresses while preserving metadata and object aliases."""
    memo = {}

    def clone(value):
        if not isinstance(value, torch.Tensor):
            return value
        identity = id(value)
        result = memo.get(identity)
        if result is not None:
            return result
        if value.layout != torch.strided:
            raise TypeError(f"L2-cold RMSNorm tuning requires strided tensors, got {value.layout}")
        result = torch.empty_strided(
            tuple(value.shape),
            tuple(value.stride()),
            dtype=value.dtype,
            device=value.device,
        )
        if value.numel():
            result.copy_(value)
        memo[identity] = result
        return result

    return tuple(clone(value) for value in args), {
        key: clone(value) for key, value in kwargs.items()
    }


def _named_call_arguments(args, kwargs):
    names = (
        "input_tensor",
        "weight_tensor",
        "bias_tensor",
        "residual_tensor",
        "output_tensor",
        "residual_out_tensor",
        "rstd_tensor",
        "m",
        "eps",
        "weight_offset",
    )
    values = dict(zip(names, args))
    values.update(kwargs)
    return values


def _unique_tensor_bytes(tensors, *, storage_span: bool) -> int:
    seen = set()
    total = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if storage_span:
            total += _tensor_storage_span_bytes(tensor)
        else:
            total += tensor.numel() * tensor.element_size()
    return total


def _rmsnorm_cache_read_bytes(args, kwargs) -> int:
    """Cache pressure from one launch's reads, excluding write-only outputs.

    Global stores are not a reliable eviction authority: they may bypass or
    allocate differently in MALL. Sizing rotation from reads only is
    conservative and prevents a write-heavy kernel from claiming a cold pool
    that is only half as large as its reported footprint.
    """
    values = _named_call_arguments(args, kwargs)
    names = ["input_tensor"]
    if values["has_weight"]:
        names.append("weight_tensor")
    if values["has_bias"]:
        names.append("bias_tensor")
    if values["has_residual"]:
        names.append("residual_tensor")
    return _unique_tensor_bytes((values[name] for name in names), storage_span=False)


def _all_tensor_storage_bytes(args, kwargs) -> int:
    return _unique_tensor_bytes(
        (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)),
        storage_span=True,
    )


def _cache_authority(device: torch.device) -> _CacheAuthority:
    """Return a conservative whole-device cache target without import-time bench deps.

    CUDA uses the device-reported L2. ROCm first asks Triton's active driver for
    the buffer it uses to evict cache before benchmarks. Current ROCm Triton
    returns 256 MiB, which covers gfx950's MALL even though torch reports only
    the roughly 4 MiB-per-XCD L2. The import and allocation happen only during
    a forced search; normal ``import quack.rmsnorm_flydsl`` stays independent of
    Triton and all benchmark extras.
    """
    props = torch.cuda.get_device_properties(device)
    reported = max(0, int(getattr(props, "L2_cache_size", 0) or 0))
    if torch.version.hip is None:
        if reported <= 0:
            raise RuntimeError("CUDA device did not report a usable L2 cache size")
        return _CacheAuthority(reported, "torch.cuda device L2")

    seed = None
    triton_bytes = 0
    try:
        driver = importlib.import_module("triton.runtime").driver
        with torch.cuda.device(device):
            seed = driver.active.get_empty_cache_for_benchmark()
        if isinstance(seed, torch.Tensor) and seed.device == device:
            triton_bytes = seed.numel() * seed.element_size()
        else:
            seed = None
    except Exception:  # noqa: BLE001 - an optional third-party probe must fail closed
        # Triton is not a runtime dependency of this opt-in backend. gfx950 is
        # the only supported ROCm target, so its benchmark driver's 256 MiB
        # policy remains the conservative fallback when the probe is absent.
        seed = None

    cache_bytes = max(reported, triton_bytes, _ROCM_TRITON_CACHE_BYTES)
    source = (
        f"Triton ROCm benchmark eviction buffer ({triton_bytes // 1024**2} MiB)"
        if triton_bytes
        else "ROCm conservative benchmark target (256 MiB)"
    )
    return _CacheAuthority(cache_bytes, source, seed)


def _reference_samples(args, kwargs, max_rows: int = _CORRECTNESS_ROWS) -> _ReferenceSamples:
    """Compute an independent fp32 RMSNorm reference on deterministic rows."""
    values = _named_call_arguments(args, kwargs)
    x = values["input_tensor"]
    n = int(values["n"])
    x_rows = x.reshape(-1, n)
    total_rows = x_rows.shape[0]
    sample_count = min(max_rows, total_rows)
    row_indices = (
        torch.linspace(
            0,
            total_rows - 1,
            sample_count,
            device=x.device,
            dtype=torch.float64,
        )
        .round()
        .to(torch.int64)
    )
    row_indices = torch.unique(row_indices, sorted=True)

    with torch.no_grad():
        source = x_rows.index_select(0, row_indices).float()
        if values["has_residual"]:
            residual = values["residual_tensor"].reshape(-1, n)
            source = source + residual.index_select(0, row_indices).float()
        rstd = torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + float(values["eps"]))
        normalized = source * rstd
        if values["has_weight"]:
            weight = values["weight_tensor"]
            if values["per_head"]:
                heads = row_indices.remainder(int(values["num_heads"]))
                weight = weight.index_select(0, heads)
            normalized = normalized * (weight.float() + float(values["weight_offset"]))
        if values["has_bias"]:
            bias = values["bias_tensor"]
            if values["per_head"]:
                heads = row_indices.remainder(int(values["num_heads"]))
                bias = bias.index_select(0, heads)
            normalized = normalized + bias.float()

        output = normalized.to(values["output_tensor"].dtype)
        residual_out = (
            source.to(values["residual_out_tensor"].dtype) if values["store_residual"] else None
        )
        expected_rstd = (
            rstd.squeeze(-1).to(values["rstd_tensor"].dtype) if values["store_rstd"] else None
        )
    return _ReferenceSamples(row_indices, output, residual_out, expected_rstd)


def _allocate_eviction_buffers(
    required_bytes: int,
    authority: _CacheAuthority,
    device: torch.device,
) -> list[torch.Tensor]:
    buffers = []
    try:
        remaining = required_bytes
        if authority.seed_buffer is not None:
            buffers.append(authority.seed_buffer)
            remaining -= authority.seed_buffer.numel() * authority.seed_buffer.element_size()
        while remaining > 0:
            chunk = min(remaining, _L2_EVICTION_CHUNK_BYTES)
            buffers.append(torch.empty(chunk, device=device, dtype=torch.uint8))
            remaining -= chunk
        for buffer in buffers:
            buffer.zero_()
        torch.cuda.synchronize(device)
        return buffers
    except (RuntimeError, MemoryError):
        buffers.clear()
        raise


def _validate_rotation_addresses(plan: _L2RotationPlan):
    """Require every active tensor name to use a different address in every set."""
    active_names = ["input_tensor", "output_tensor"]
    first = _named_call_arguments(plan.arg_sets[0], plan.kwarg_sets[0])
    if first["has_weight"]:
        active_names.append("weight_tensor")
    if first["has_bias"]:
        active_names.append("bias_tensor")
    if first["has_residual"]:
        active_names.append("residual_tensor")
    if first["store_residual"]:
        active_names.append("residual_out_tensor")
    if first["store_rstd"]:
        active_names.append("rstd_tensor")

    calls = [
        _named_call_arguments(args, kwargs) for args, kwargs in zip(plan.arg_sets, plan.kwarg_sets)
    ]
    for name in active_names:
        tensors = [call[name] for call in calls]
        pointers = [tensor.data_ptr() for tensor in tensors if tensor.numel()]
        if len(pointers) != len(set(pointers)):
            raise RuntimeError(f"L2-cold rotation reused the same {name} address across sets")


def _build_l2_rotation_plan(
    args,
    kwargs,
    *,
    n_timed_calls: int = _L2_TIMED_CALLS,
    target_ratio: int = _L2_TARGET_RATIO,
    read_bytes_fn=None,
    clone_fn=None,
    reference_fn=None,
    validate_addresses_fn=None,
) -> _L2RotationPlan:
    """Clone one call into a bounded, whole-cache-sized round-robin plan."""
    read_bytes_fn = read_bytes_fn or _rmsnorm_cache_read_bytes
    clone_fn = clone_fn or _clone_tensor_arguments
    reference_fn = reference_fn or _reference_samples
    validate_addresses_fn = validate_addresses_fn or _validate_rotation_addresses
    device = next(
        (value.device for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)),
        None,
    )
    if device is None or device.type != "cuda":
        raise RuntimeError("L2-cold RMSNorm tuning requires CUDA/ROCm tensor arguments")
    authority = _cache_authority(device)
    cache_bytes = authority.cache_bytes
    read_bytes = read_bytes_fn(args, kwargs)
    clone_bytes = _all_tensor_storage_bytes(args, kwargs)
    if read_bytes <= 0 or clone_bytes <= 0:
        raise RuntimeError("L2-cold RMSNorm tuning found no tensor working set")

    # If a set is reused inside the graph, at least target_ratio caches of
    # *other* addresses are touched in between. At the 200-set cap every timed
    # call uses a never-before-timed set, after a full eviction pass.
    n_sets = min(
        n_timed_calls,
        _L2_MAX_BUFFER_SETS,
        max(2, (target_ratio * cache_bytes + read_bytes - 1) // read_bytes + 1),
    )
    with torch.cuda.device(device):
        free_bytes, _ = torch.cuda.mem_get_info(device)
    memory_budget = min(_L2_MAX_EXTRA_BYTES, int(free_bytes * 0.25))

    def estimated_extra(count):
        rotation_bytes = count * read_bytes
        eviction_bytes = max(cache_bytes, target_ratio * cache_bytes - rotation_bytes)
        return (count - 1) * clone_bytes + eviction_bytes

    budget_degraded = estimated_extra(n_sets) > memory_budget
    if budget_degraded:
        clone_budget = max(0, memory_budget - cache_bytes)
        n_sets = max(2, min(n_sets, 1 + clone_budget // clone_bytes))
    arg_sets = [tuple(args)]
    kwarg_sets = [dict(kwargs)]
    fallback_reason = (
        f"full {_L2_TARGET_RATIO}x working set exceeded the memory budget"
        if budget_degraded
        else None
    )
    clone_failure = None
    try:
        for _ in range(1, n_sets):
            cloned_args, cloned_kwargs = clone_fn(args, kwargs)
            arg_sets.append(cloned_args)
            kwarg_sets.append(cloned_kwargs)
    except (RuntimeError, MemoryError) as error:
        clone_failure = type(error).__name__
        arg_sets = [tuple(args)]
        kwarg_sets = [dict(kwargs)]
        try:
            del cloned_args, cloned_kwargs
        except UnboundLocalError:
            pass
    if clone_failure is not None:
        torch.cuda.empty_cache()
        try:
            cloned_args, cloned_kwargs = clone_fn(args, kwargs)
        except (RuntimeError, MemoryError) as clone_error:
            raise RuntimeError(
                "Unable to allocate even two RMSNorm address sets for L2-cold autotuning"
            ) from clone_error
        arg_sets.append(cloned_args)
        kwarg_sets.append(cloned_kwargs)
        fallback_reason = f"clone allocation was reduced after {clone_failure}"

    rotation_bytes = len(arg_sets) * read_bytes
    eviction_required = (
        cache_bytes
        if budget_degraded
        else max(cache_bytes, target_ratio * cache_bytes - rotation_bytes)
    )
    try:
        eviction_buffers = _allocate_eviction_buffers(eviction_required, authority, device)
    except (RuntimeError, MemoryError) as error:
        authority.seed_buffer = None
        torch.cuda.empty_cache()
        try:
            # Explicitly degraded but still cold: one whole-cache eviction plus
            # at least two different RMSNorm address sets. The timed fallback
            # evicts before each call if the reduced rotation can otherwise reuse.
            eviction_buffers = _allocate_eviction_buffers(cache_bytes, authority, device)
        except (RuntimeError, MemoryError) as eviction_error:
            raise RuntimeError(
                f"Unable to allocate a {cache_bytes / 1024**2:.0f} MiB cache-eviction "
                "buffer for L2-cold autotuning"
            ) from eviction_error
        fallback_reason = (
            f"{fallback_reason + '; ' if fallback_reason else ''}"
            f"eviction working set was reduced after {type(error).__name__}"
        )

    per_call_eviction = (
        len(arg_sets) < n_timed_calls and (len(arg_sets) - 1) * read_bytes <= cache_bytes
    )
    if per_call_eviction:
        fallback_reason = (
            f"{fallback_reason + '; ' if fallback_reason else ''}"
            "rotation alone cannot evict one whole cache"
        )

    current_stream = torch.cuda.current_stream(device)
    bench_stream = torch.cuda.Stream(device=device)
    bench_stream.wait_stream(current_stream)
    with torch.cuda.stream(bench_stream):
        expected = reference_fn(args, kwargs)
    bench_stream.synchronize()

    plan = _L2RotationPlan(
        arg_sets=arg_sets,
        kwarg_sets=kwarg_sets,
        cache_bytes=cache_bytes,
        read_bytes_per_set=read_bytes,
        eviction_buffers=eviction_buffers,
        cache_source=authority.source,
        bench_stream=bench_stream,
        expected=expected,
        n_timed_calls=n_timed_calls,
        per_call_eviction=per_call_eviction,
        fallback_reason=fallback_reason,
    )
    if plan.working_set_bytes <= cache_bytes:
        raise RuntimeError(
            "L2-cold rotation working set does not exceed the cache target: "
            f"{plan.working_set_bytes} <= {cache_bytes}"
        )
    validate_addresses_fn(plan)
    return plan


def _with_runtime_stream(positional_sets, stream: torch.cuda.Stream):
    raw_stream = stream.cuda_stream
    return [tuple(positional[:-1]) + (raw_stream,) for positional in positional_sets]


def _clear_cache(plan: _L2RotationPlan):
    for buffer in plan.eviction_buffers:
        buffer.zero_()


def _candidate_correctness_gate(compiled, positional_sets, plan: _L2RotationPlan):
    """Run and numerically validate every timed address set before benchmarking."""
    if not isinstance(compiled, CompiledFunction):
        raise TypeError("candidate timing requires flydsl.compiler.CompiledFunction")
    indices = plan.expected.row_indices
    for set_index, (args, positional) in enumerate(zip(plan.arg_sets, positional_sets)):
        compiled(*positional)
        values = _named_call_arguments(args, plan.kwarg_sets[set_index])
        n = int(values["n"])
        try:
            actual = values["output_tensor"].reshape(-1, n).index_select(0, indices)
            tolerance = (
                {"rtol": 2e-4, "atol": 2e-5}
                if actual.dtype == torch.float32
                else {"rtol": 2e-2, "atol": 2e-2}
            )
            torch.testing.assert_close(actual, plan.expected.output, **tolerance)
            if plan.expected.residual_out is not None:
                residual_out = values["residual_out_tensor"].reshape(-1, n).index_select(0, indices)
                torch.testing.assert_close(
                    residual_out,
                    plan.expected.residual_out,
                    **(
                        {"rtol": 2e-4, "atol": 2e-5}
                        if residual_out.dtype == torch.float32
                        else {"rtol": 2e-2, "atol": 2e-2}
                    ),
                )
            if plan.expected.rstd is not None:
                rstd = values["rstd_tensor"].reshape(-1).index_select(0, indices)
                torch.testing.assert_close(
                    rstd,
                    plan.expected.rstd,
                    rtol=2e-4,
                    atol=2e-5,
                )
        except AssertionError as error:
            raise AssertionError(
                f"RMSNorm autotune candidate failed correctness on clone set {set_index}: {error}"
            ) from error
    plan.bench_stream.synchronize()


def _event_l2_rotate_bench(
    compiled,
    positional_sets,
    plan: _L2RotationPlan,
    *,
    warmup_target_ms: float,
    replay_samples: int,
):
    """Graph-free fallback that preserves cold addresses and excludes eviction time."""
    stream = plan.bench_stream
    with torch.cuda.stream(stream):
        probe_start = torch.cuda.Event(enable_timing=True)
        probe_end = torch.cuda.Event(enable_timing=True)
        probe_start.record()
        compiled(*positional_sets[0])
        probe_end.record()
        probe_end.synchronize()
        estimate_ms = max(probe_start.elapsed_time(probe_end), 1e-3)
        n_warmup = max(50, int(warmup_target_ms / estimate_ms))
        for i in range(n_warmup):
            compiled(*positional_sets[i % len(positional_sets)])
        stream.synchronize()

        if plan.per_call_eviction:
            samples = []
            for _ in range(replay_samples):
                starts = []
                ends = []
                for i in range(plan.n_timed_calls):
                    _clear_cache(plan)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    compiled(*positional_sets[i % len(positional_sets)])
                    end.record()
                    starts.append(start)
                    ends.append(end)
                ends[-1].synchronize()
                samples.append(
                    statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))
                )
            return statistics.median(samples)

        samples = []
        for _ in range(replay_samples):
            _clear_cache(plan)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i in range(plan.n_timed_calls):
                compiled(*positional_sets[i % len(positional_sets)])
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / plan.n_timed_calls)
    return statistics.median(samples)


def l2_cold_bench(
    compiled,
    positional_sets,
    plan: _L2RotationPlan,
    *,
    warmup_target_ms: float = _L2_WARMUP_TARGET_MS,
    replay_samples: int = _L2_REPLAY_SAMPLES,
):
    """Measure ms/call with a ~200-launch L2-cold multi-address CUDA/HIP graph.

    Clone creation, numerical gates, graph capture, warmup, and the explicit
    cache-eviction writes all occur outside timed event pairs. If graph capture
    is unsupported or fails (including OOM), the event fallback keeps the same
    address rotation. A memory-degraded plan evicts before every individually
    timed call instead of silently reverting to a cache-hot loop.
    """
    if not isinstance(compiled, CompiledFunction):
        raise TypeError("candidate timing requires flydsl.compiler.CompiledFunction")
    if plan.fallback_reason and not plan.per_call_eviction and not plan.fallback_warning_emitted:
        warnings.warn(
            f"RMSNorm L2-cold rotation is using a degraded allocation: {plan.fallback_reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        plan.fallback_warning_emitted = True
    if plan.per_call_eviction:
        if not plan.fallback_warning_emitted:
            warnings.warn(
                "CUDA/HIP graph disabled for RMSNorm autotuning because the bounded "
                f"rotation needs per-call cache eviction ({plan.fallback_reason})",
                RuntimeWarning,
                stacklevel=2,
            )
            plan.fallback_warning_emitted = True
        return _event_l2_rotate_bench(
            compiled,
            positional_sets,
            plan,
            warmup_target_ms=warmup_target_ms,
            replay_samples=replay_samples,
        )

    stream = plan.bench_stream
    with torch.cuda.stream(stream):
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for i in range(plan.n_timed_calls):
                    compiled(*positional_sets[i % len(positional_sets)])

            # Size graph replays from measured GPU work, not Python launch cost.
            warm_start = torch.cuda.Event(enable_timing=True)
            warm_end = torch.cuda.Event(enable_timing=True)
            warm_start.record()
            graph.replay()
            warm_end.record()
            warm_end.synchronize()
            graph_ms = max(warm_start.elapsed_time(warm_end), 1e-3)
            warm_replays = max(1, int(warmup_target_ms / graph_ms + 0.999))
            for _ in range(warm_replays - 1):
                graph.replay()
            stream.synchronize()

            samples = []
            for _ in range(replay_samples):
                _clear_cache(plan)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / plan.n_timed_calls)
            return statistics.median(samples)
        except (RuntimeError, MemoryError, NotImplementedError) as error:
            warnings.warn(
                "CUDA/HIP graph capture failed for RMSNorm autotuning; using "
                "event-timed L2-cold multi-address fallback "
                f"({type(error).__name__}: {error})",
                RuntimeWarning,
                stacklevel=2,
            )

    # A fresh stream avoids reusing a stream whose failed capture may carry
    # backend-specific error state. The data is already materialized and the
    # graph context synchronizes before capture, so no producer dependency remains.
    device = plan.arg_sets[0][0].device
    torch.cuda.synchronize(device)
    fallback_stream = torch.cuda.Stream(device=device)
    plan.bench_stream = fallback_stream
    fallback_positionals = _with_runtime_stream(positional_sets, fallback_stream)
    return _event_l2_rotate_bench(
        compiled,
        fallback_positionals,
        plan,
        warmup_target_ms=warmup_target_ms,
        replay_samples=replay_samples,
    )


def _typed_identity(value):
    """Return a hashable identity that does not collapse values across types."""
    value_type = (type(value).__module__, type(value).__qualname__)
    if isinstance(value, dict):
        body = tuple(
            sorted(
                ((_typed_identity(key), _typed_identity(item)) for key, item in value.items()),
                key=repr,
            )
        )
    elif isinstance(value, (tuple, list)):
        body = tuple(_typed_identity(item) for item in value)
    elif isinstance(value, (set, frozenset)):
        body = tuple(sorted((_typed_identity(item) for item in value), key=repr))
    else:
        body = repr(value)
    return value_type, body


def _clone_python_function(function, annotations):
    """Copy a JIT's pristine Python function before FlyDSL rewrites its AST."""
    clone = types.FunctionType(
        function.__code__,
        function.__globals__,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    clone.__annotations__ = dict(annotations)
    clone.__dict__.update(function.__dict__)
    clone.__doc__ = function.__doc__
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__module__ = function.__module__
    clone.__qualname__ = function.__qualname__
    return clone


class RmsNormAutotuner(Autotuner):
    """FlyDSL autotuner whose candidates and winners use CompiledFunction.

    FlyDSL's generic :class:`Autotuner` caches only the winning ``Config``.
    Every candidate repetition and winner cache hit consequently re-enters the
    full ``JitFunction`` dispatcher. RMSNorm needs the same config persistence
    semantics but keeps a second, process-local cache of compiled callables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled_cache = {}
        self._compiled_lookup = {}
        self._device_jit_functions = {}
        self._hot_cache = {}
        self._active_call = threading.local()
        self._toolchain_key = None
        self._fast_context_snapshot = None
        self._fast_context_generation = 0

    def __call__(self, *args, **kwargs):
        hot_key = self._hot_key(args, kwargs)
        tuning = _tuning_enabled()
        if not tuning and not env.compile.compile_only:
            hot_entry = self._hot_cache.get(hot_key)
            if hot_entry is not None:
                config, compiled, constexpr_suffix = hot_entry
                if len(args) == 10 and constexpr_suffix is not None:
                    stream = kwargs.get("stream", self._signature.parameters["stream"].default)
                    return compiled(*(args + constexpr_suffix + (stream,)))
                return compiled(*self._positional_arguments(config, args, kwargs))

        self._active_call.hot_key = hot_key
        self._active_call.rotation_plan = None
        try:
            # CUDA/HIP permits only one graph capture in a process at a time.
            # Non-forced cache/default calls remain fully concurrent.
            bench_context = _AUTOTUNE_BENCH_LOCK if tuning else nullcontext()
            with bench_context:
                return super().__call__(*args, **kwargs)
        finally:
            self._active_call.value = None
            self._active_call.hot_key = None
            self._active_call.rotation_plan = None

    def fast_context_token(self):
        """Cheap invalidation token for an adapter-level resolved-winner cache."""
        data = getattr(os.environ, "_data", None)
        if data is None:
            environment = tuple(os.environ.get(name, "") for name in _FAST_CONTEXT_ENV_VARS)
        else:
            environment = tuple(data.get(name, b"") for name in _FAST_CONTEXT_ENV_VARS_BYTES)
        persistent_hints = getattr(self.fn, "compile_hints", {})
        snapshot = (
            environment,
            _typed_identity(persistent_hints) if persistent_hints else (),
        )
        if snapshot != self._fast_context_snapshot:
            self._fast_context_snapshot = snapshot
            self._fast_context_generation += 1
        return self._fast_context_generation

    def resolved_fast_entry(self, args, kwargs):
        """Return the process-hot callable after a normal tuner call resolved it."""
        hot_entry = self._hot_cache.get(self._hot_key(args, kwargs))
        if hot_entry is None:
            return None
        config, compiled, constexpr_suffix = hot_entry
        return config, compiled, constexpr_suffix

    @staticmethod
    def _process_context_key():
        """Environment axes that must partition winners and loaded callables."""
        return (
            ("runtime_kind", os.environ.get("FLYDSL_RUNTIME_KIND", "")),
            ("artifact_dir", os.environ.get("FLYDSL_AUTOTUNE_CONFIG_DIR", "")),
        )

    def _contextual_decision_key(self, args, kwargs):
        # FlyDSL serializes this tuple through JSON. Keep the added context as a
        # string so a disk round-trip cannot turn a nested tuple into an
        # unhashable list.
        context = str(("_process_context_", self._process_context_key()))
        return (*super()._make_key(args, kwargs), context)

    def _make_key(self, args, kwargs):
        key = self._contextual_decision_key(args, kwargs)
        # Autotuner.__call__ passes these same tuple/dict objects to _bench_one
        # and _run_config. Keep only their ids so the thread-local does not retain
        # the caller's (potentially multi-GiB) tensors after an exception.
        self._active_call.value = (id(args), id(kwargs), key)
        return key

    def _decision_key(self, args, kwargs):
        active = getattr(self._active_call, "value", None)
        if active is not None and active[:2] == (id(args), id(kwargs)):
            return active[2]
        return self._contextual_decision_key(args, kwargs)

    def _device_key(self, args, kwargs):
        device = self._call_device(args, kwargs)
        if device is None:
            raise RuntimeError("RMSNorm fast autotuning requires a CUDA/ROCm tensor argument")
        index = device.index if device.index is not None else torch.cuda.current_device()
        return device.type, index

    def _hot_key(self, args, kwargs):
        if self._toolchain_key is None:
            self._toolchain_key = _toolchain_fingerprint()

        if len(args) == 10:
            explicit = (args[7],) + tuple(kwargs[name] for name in self.key[1:])
        else:
            return self._generic_hot_key(args, kwargs)

        # The public adapter canonicalizes every operand to a torch Tensor with a
        # layout-dynamic, unit inner stride. The explicit dtype/feature/shape axes
        # above therefore determine the FlyDSL ABI; the full compiled cache still
        # records exact tensor metadata and validates this alias on its first miss.
        effective_hints = self.fn._effective_compile_hints()
        return (
            ("explicit", explicit),
            ("device", args[0].device.type, args[0].device.index),
            ("env", _env_fingerprint()),
            ("toolchain", self._toolchain_key),
            ("device_fingerprint", os.environ.get("FLYDSL_GPU_ARCH", "")),
            ("compile_hints", _typed_identity(effective_hints)),
            ("process_context", self._process_context_key()),
        )

    def _generic_hot_key(self, args, kwargs):
        sig_args = dict(zip(self.arg_names, args))
        sig_args.update(kwargs)
        explicit = tuple((name, _typed_identity(sig_args.get(name))) for name in self.key)
        tensors = tuple(
            (name, self._tensor_abi(value))
            for name, value in sig_args.items()
            if isinstance(value, torch.Tensor)
        )
        runtime_types = tuple(
            (name, type(sig_args[name]).__module__, type(sig_args[name]).__qualname__)
            for name in ("m", "eps", "weight_offset", "stream")
            if name in sig_args
        )
        effective_hints = self.fn._effective_compile_hints()
        return (
            ("explicit", explicit),
            ("tensors", tensors),
            ("runtime_types", runtime_types),
            ("device", self._device_key(args, kwargs)),
            ("env", _env_fingerprint()),
            ("toolchain", self._toolchain_key),
            ("device_fingerprint", _device_fingerprint()),
            ("compile_hints", _typed_identity(effective_hints)),
            ("process_context", self._process_context_key()),
        )

    @staticmethod
    def _tensor_abi(value):
        index = value.device.index
        if index is None:
            index = torch.cuda.current_device()
        return (
            type(value).__module__,
            type(value).__qualname__,
            value.device.type,
            index,
            str(value.dtype),
            str(value.layout),
            tuple(value.shape),
            tuple(value.stride()),
        )

    def _abi_key(self, bound_arguments):
        identity = []
        for name, value in bound_arguments.items():
            parameter = self._signature.parameters[name]
            if isinstance(value, torch.Tensor):
                item = ("tensor", self._tensor_abi(value))
            elif Constexpr.is_constexpr_annotation(parameter.annotation):
                item = ("constexpr", _typed_identity(value))
            else:
                # Runtime scalar values and stream pointers must remain dynamic.
                # Their Python type determines the CallState packing ABI.
                item = ("runtime", type(value).__module__, type(value).__qualname__)
            identity.append((name, item))
        return tuple(identity)

    def _prepare_call(self, config, args, kwargs):
        merged = dict(kwargs)
        merged.update(config.all_kwargs())
        bound = self._signature.bind(*args, **merged)
        bound.apply_defaults()
        positional = tuple(bound.arguments.values())
        device_key = self._device_key(args, merged)
        fast_key = (
            ("decision", self._decision_key(args, kwargs)),
            ("device", device_key),
            ("config", _typed_identity(config.to_dict())),
            ("compiler_opts", _typed_identity(config.compiler_opts())),
            ("abi", self._abi_key(bound.arguments)),
        )
        return merged, positional, device_key, fast_key

    def _lookup_key(self, config, args, kwargs):
        return (
            ("decision", self._decision_key(args, kwargs)),
            ("device", self._device_key(args, kwargs)),
            ("config", _typed_identity(config.to_dict())),
        )

    def _positional_arguments(self, config, args, kwargs):
        values = dict(zip(self.arg_names, args))
        values.update(kwargs)
        values.update(config.all_kwargs())
        return tuple(
            values.get(name, parameter.default)
            for name, parameter in self._signature.parameters.items()
        )

    def _device_jit_function(self, device_key):
        jit_function = self._device_jit_functions.get(device_key)
        if jit_function is None:
            original = getattr(self.fn, "_original_func", self.fn.func)
            jit_function = flyc.jit(
                _clone_python_function(original, annotations=self.fn.func.__annotations__)
            )
            self._device_jit_functions[device_key] = jit_function
        # Persistent hints belong to the original decorated function. Copy them
        # onto this device-local JIT before a miss; Config hints remain scoped by
        # the thread-local CompilationContext below.
        jit_function.compile_hints = dict(getattr(self.fn, "compile_hints", {}))
        return jit_function

    def _compiled_callable(self, config, args, kwargs):
        lookup_key = self._lookup_key(config, args, kwargs)
        compiled = self._compiled_lookup.get(lookup_key)
        if compiled is not None:
            return compiled, self._positional_arguments(config, args, kwargs), False

        _merged, positional, device_key, fast_key = self._prepare_call(config, args, kwargs)
        compiled = self._compiled_cache.get(fast_key)
        if compiled is not None:
            self._compiled_lookup[lookup_key] = compiled
            return compiled, positional, False

        # FlyDSL compilation and the two process-local caches are protected by
        # the same lock as QuACK's other RMSNorm builders. The second lookup is
        # what makes concurrent first calls compile exactly once.
        with FLYDSL_BUILD_LOCK:
            compiled = self._compiled_lookup.get(lookup_key)
            if compiled is not None:
                return compiled, self._positional_arguments(config, args, kwargs), False

            compiled = self._compiled_cache.get(fast_key)
            if compiled is not None:
                self._compiled_lookup[lookup_key] = compiled
                return compiled, positional, False

            jit_function = self._device_jit_function(device_key)
            compiler_opts = config.compiler_opts()
            hints_context = (
                CompilationContext.compile_hints(compiler_opts) if compiler_opts else nullcontext()
            )
            with torch.cuda.device(device_key[1]), hints_context:
                # Do not use flyc.compile[hints]: FlyDSL 0.3 implements that
                # spelling by mutating JitFunction.compile_hints persistently.
                compiled = flyc.compile(jit_function, *positional)
            self._compiled_cache[fast_key] = compiled
            self._compiled_lookup[lookup_key] = compiled
            return compiled, positional, True

    def _bench_one(self, config, args, kwargs):
        """Compile one candidate untimed, then benchmark only its fast callable."""
        merged = dict(kwargs)
        merged.update(config.all_kwargs())
        with self._stream_context(args, merged):
            snapshot = self._snapshot_tensors(args, merged)
            try:
                compiled, positional, _ = self._compiled_callable(config, args, kwargs)

                # Keep support for tests and callers that deliberately inject a
                # custom benchmark function. The production objective below is
                # identified by function identity, so it cannot silently fall
                # back to the old same-address event loop.
                if self._do_bench is not l2_cold_bench:

                    def kernel_call():
                        self._restore_tensors(snapshot)
                        self._reset_tensors(args, merged)
                        if config.pre_hook:
                            config.pre_hook(merged)
                        if self.pre_hook:
                            self.pre_hook(merged)
                        compiled(*positional)
                        if self.post_hook:
                            self.post_hook(merged)

                    return self._do_bench(kernel_call, warmup=self.warmup, rep=self.rep)

                if (
                    config.pre_hook is not None
                    or self.pre_hook is not None
                    or self.post_hook is not None
                    or self.reset_to_zero
                    or self.restore_value
                ):
                    raise RuntimeError(
                        "L2-cold RMSNorm timing does not support benchmark hooks or "
                        "in-place restore/reset semantics"
                    )

                plan = getattr(self._active_call, "rotation_plan", None)
                if plan is None:
                    plan = _build_l2_rotation_plan(args, kwargs)
                    self._active_call.rotation_plan = plan
                positional_sets = [
                    self._positional_arguments(config, set_args, set_kwargs)
                    for set_args, set_kwargs in zip(plan.arg_sets, plan.kwarg_sets)
                ]
                positional_sets = _with_runtime_stream(positional_sets, plan.bench_stream)
                with torch.cuda.stream(plan.bench_stream):
                    _candidate_correctness_gate(compiled, positional_sets, plan)
                    return self._do_bench(
                        compiled,
                        positional_sets,
                        plan,
                        warmup_target_ms=float(self.warmup),
                        replay_samples=int(self.rep),
                    )

            finally:
                if snapshot:
                    self._restore_tensors(snapshot)

    def _run_config(self, config, args, kwargs):
        """Run a selected config without re-entering JitFunction dispatch."""
        merged = dict(kwargs)
        merged.update(config.all_kwargs())
        setup_context = self._stream_context(args, merged) if self.reset_to_zero else nullcontext()
        with setup_context:
            self._reset_tensors(args, merged)
            compiled, positional, compiled_now = self._compiled_callable(config, args, kwargs)
            hot_key = getattr(self._active_call, "hot_key", None)
            if hot_key is not None:
                config_snapshot = Config.from_dict(config.to_dict())
                constexpr_suffix = positional[10:-1] if len(args) == 10 else None
                self._hot_cache[hot_key] = (
                    config_snapshot,
                    compiled,
                    constexpr_suffix,
                )
            if env.compile.compile_only:
                return None
            # flyc.compile performs the first launch while producing the callable.
            if compiled_now:
                return None
            return compiled(*positional)


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

_rmsnorm_fwd_tuner = RmsNormAutotuner(
    fn=rmsnorm_direct,
    configs=rmsnorm_search_configs,
    key=_RMSNORM_AUTOTUNE_KEY,
    warmup=_L2_WARMUP_TARGET_MS,
    rep=_L2_REPLAY_SAMPLES,
    do_bench_fn=l2_cold_bench,
    default=rmsnorm_default_config,
    artifact_name="quack_rmsnorm_fwd",
)


__all__ = [
    "RMSNORM_AUTOTUNE_SCHEMA_VERSION",
    "RmsNormAutotuner",
    "_rmsnorm_fwd_tuner",
    "l2_cold_bench",
    "rmsnorm_default_config",
    "rmsnorm_search_configs",
]

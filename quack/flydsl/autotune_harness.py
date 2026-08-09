# Copyright (c) 2026, Tri Dao.

"""Shared L2-cold autotuning infrastructure for FlyDSL kernels."""

import importlib
import os
import statistics
import threading
import types
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

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

_L2_TARGET_RATIO = 3
_L2_TIMED_CALLS = 200
_L2_MAX_BUFFER_SETS = _L2_TIMED_CALLS
_L2_WARMUP_TARGET_MS = 200.0
_L2_REPLAY_SAMPLES = 3
_L2_MAX_EXTRA_BYTES = 4 * 1024**3
_L2_EVICTION_CHUNK_BYTES = 256 * 1024**2
_ROCM_TRITON_CACHE_BYTES = 256 * 1024**2
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


@dataclass
class _CacheAuthority:
    """Cache capacity used to size a cold-address working set."""

    cache_bytes: int
    source: str
    seed_buffer: torch.Tensor | None = None


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
    expected: Any
    n_timed_calls: int
    per_call_eviction: bool = False
    fallback_reason: str | None = None
    fallback_warning_emitted: bool = False
    benchmark_name: str = "FlyDSL"

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
            raise TypeError(f"L2-cold tuning requires strided tensors, got {value.layout}")
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


def _all_tensor_storage_bytes(args, kwargs) -> int:
    return _unique_tensor_bytes(
        (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)),
        storage_span=True,
    )


def _cuda_device(args, kwargs, benchmark_name: str) -> torch.device:
    device = next(
        (value.device for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)),
        None,
    )
    if device is None or device.type != "cuda":
        raise RuntimeError(f"L2-cold {benchmark_name} tuning requires CUDA/ROCm tensor arguments")
    return device


def _cache_authority(device: torch.device) -> _CacheAuthority:
    """Return a conservative whole-device cache target.

    CUDA uses the device-reported L2. ROCm first asks Triton's active driver for
    the buffer it uses to evict cache before benchmarks. The optional import and
    allocation happen only during a forced search.
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
        # Triton is not a runtime dependency of an opt-in FlyDSL backend. gfx950
        # is the supported ROCm target, so its benchmark driver's 256 MiB policy
        # remains the conservative fallback when the probe is absent.
        seed = None

    cache_bytes = max(reported, triton_bytes, _ROCM_TRITON_CACHE_BYTES)
    source = (
        f"Triton ROCm benchmark eviction buffer ({triton_bytes // 1024**2} MiB)"
        if triton_bytes
        else "ROCm conservative benchmark target (256 MiB)"
    )
    return _CacheAuthority(cache_bytes, source, seed)


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
    cache_authority_fn=None,
    allocate_eviction_buffers_fn=None,
    benchmark_name: str = "FlyDSL",
) -> _L2RotationPlan:
    """Clone one call into a bounded, whole-cache-sized round-robin plan.

    ``read_bytes_fn`` is deliberately required: only a kernel-specific caller
    knows which tensor operands are read rather than write-only.
    """
    if read_bytes_fn is None:
        raise TypeError("read_bytes_fn is required to size an L2-cold rotation")
    clone_fn = clone_fn or _clone_tensor_arguments
    cache_authority_fn = cache_authority_fn or _cache_authority
    allocate_eviction_buffers_fn = allocate_eviction_buffers_fn or _allocate_eviction_buffers
    device = _cuda_device(args, kwargs, benchmark_name)
    authority = cache_authority_fn(device)
    cache_bytes = authority.cache_bytes
    read_bytes = read_bytes_fn(args, kwargs)
    clone_bytes = _all_tensor_storage_bytes(args, kwargs)
    if read_bytes <= 0 or clone_bytes <= 0:
        raise RuntimeError(f"L2-cold {benchmark_name} tuning found no tensor working set")

    # If a set is reused inside the graph, at least target_ratio caches of
    # other addresses are touched in between. At the 200-set cap every timed
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
        f"full {target_ratio}x working set exceeded the memory budget" if budget_degraded else None
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
                f"Unable to allocate even two {benchmark_name} address sets for L2-cold autotuning"
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
        eviction_buffers = allocate_eviction_buffers_fn(eviction_required, authority, device)
    except (RuntimeError, MemoryError) as error:
        authority.seed_buffer = None
        torch.cuda.empty_cache()
        try:
            # Explicitly degraded but still cold: one whole-cache eviction plus
            # at least two different address sets.
            eviction_buffers = allocate_eviction_buffers_fn(cache_bytes, authority, device)
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
        expected = reference_fn(args, kwargs) if reference_fn is not None else None
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
        benchmark_name=benchmark_name,
    )
    if plan.working_set_bytes <= cache_bytes:
        raise RuntimeError(
            "L2-cold rotation working set does not exceed the cache target: "
            f"{plan.working_set_bytes} <= {cache_bytes}"
        )
    if validate_addresses_fn is not None:
        validate_addresses_fn(plan)
    return plan


def _with_runtime_stream(positional_sets, stream: torch.cuda.Stream):
    raw_stream = stream.cuda_stream
    return [tuple(positional[:-1]) + (raw_stream,) for positional in positional_sets]


def _clear_cache(plan: _L2RotationPlan):
    for buffer in plan.eviction_buffers:
        buffer.zero_()


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
    """Measure milliseconds per call with an L2-cold multi-address CUDA/HIP graph."""
    if not isinstance(compiled, CompiledFunction):
        raise TypeError("candidate timing requires flydsl.compiler.CompiledFunction")
    benchmark_name = plan.benchmark_name
    if plan.fallback_reason and not plan.per_call_eviction and not plan.fallback_warning_emitted:
        warnings.warn(
            f"{benchmark_name} L2-cold rotation is using a degraded allocation: "
            f"{plan.fallback_reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        plan.fallback_warning_emitted = True
    if plan.per_call_eviction:
        if not plan.fallback_warning_emitted:
            warnings.warn(
                f"CUDA/HIP graph disabled for {benchmark_name} autotuning because the "
                f"bounded rotation needs per-call cache eviction ({plan.fallback_reason})",
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
                f"CUDA/HIP graph capture failed for {benchmark_name} autotuning; using "
                "event-timed L2-cold multi-address fallback "
                f"({type(error).__name__}: {error})",
                RuntimeWarning,
                stacklevel=2,
            )

    # A fresh stream avoids reusing a stream whose failed capture may carry
    # backend-specific error state.
    device = _cuda_device(plan.arg_sets[0], plan.kwarg_sets[0], benchmark_name)
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


class FlydslL2Autotuner(Autotuner):
    """FlyDSL autotuner with process-local compiled and hot-callable caches."""

    benchmark_name = "FlyDSL"

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
                return self._call_hot_entry(hot_entry, args, kwargs)

        self._active_call.hot_key = hot_key
        self._active_call.rotation_plan = None
        try:
            # CUDA/HIP permits only one graph capture in a process at a time.
            # Non-forced cache/default calls remain fully concurrent.
            bench_context = _AUTOTUNE_BENCH_LOCK if tuning else nullcontext()
            with bench_context:
                if tuning:
                    return self._forced_tune(*args, **kwargs)
                return super().__call__(*args, **kwargs)
        finally:
            self._active_call.value = None
            self._active_call.hot_key = None
            self._active_call.rotation_plan = None

    def _forced_tune(self, *args, **kwargs):
        key = self._make_key(args, kwargs)
        artifact = self._artifact_ref(args, kwargs, required=True)
        config, best_time = self._tune_configs(args, kwargs)
        print(f"[autotune] best: {config} ({best_time:.3f} ms)")
        if artifact is not None:
            self._emit_artifact(config, artifact, args, kwargs)
        self.cache[key] = config
        self._save_disk_cache()
        return self._run_config(config, args, kwargs)

    def _tune_configs(self, args, kwargs):
        configs = self.configs(*args, **kwargs) if callable(self.configs) else self.configs
        configs = self._prune(configs, args, kwargs)
        return self._select_result(self._benchmark_configs(configs, args, kwargs), args, kwargs)

    def _benchmark_configs(self, configs, args, kwargs):
        print(f"[autotune] tuning {len(configs)} configs...")
        results = []
        for index, config in enumerate(configs):
            try:
                elapsed = self._bench_one(config, args, kwargs)
                results.append((config, elapsed))
                print(f"  [{index + 1}/{len(configs)}] {config} -> {elapsed:.3f} ms")
            except Exception as error:  # noqa: BLE001 - one bad candidate must not abort a search
                print(f"  [{index + 1}/{len(configs)}] {config} -> FAILED: {error}")
        if not results:
            raise RuntimeError("All autotune configs failed")
        return results

    def _select_result(self, results, args, kwargs):
        return min(results, key=lambda item: item[1])

    def _call_hot_entry(self, hot_entry, args, kwargs):
        config, compiled, _payload = hot_entry
        return compiled(*self._positional_arguments(config, args, kwargs))

    def _hot_entry_payload(self, args, positional):
        return None

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
        return self._hot_cache.get(self._hot_key(args, kwargs))

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
        # Keep only object ids so the thread-local does not retain large tensors.
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
            raise RuntimeError(
                f"{self.benchmark_name} fast autotuning requires a CUDA/ROCm tensor argument"
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        return device.type, index

    def _hot_key(self, args, kwargs):
        return self._generic_hot_key(args, kwargs)

    def _generic_hot_key(self, args, kwargs):
        if self._toolchain_key is None:
            self._toolchain_key = _toolchain_fingerprint()
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
                # Runtime scalar values and stream pointers remain dynamic.
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
        # Persistent hints belong to the original decorated function. Config
        # hints remain scoped by the thread-local CompilationContext below.
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

        # Compilation and both process-local caches share the process build
        # lock. The second lookup makes concurrent first calls compile once.
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
                # flyc.compile[hints] mutates JitFunction.compile_hints in
                # FlyDSL 0.3, so keep Config hints scoped to this compilation.
                compiled = flyc.compile(jit_function, *positional)
            self._compiled_cache[fast_key] = compiled
            self._compiled_lookup[lookup_key] = compiled
            return compiled, positional, True

    def _bench_one(self, config, args, kwargs):
        """Benchmark one candidate; kernel-specific subclasses must override."""
        raise NotImplementedError

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
                self._hot_cache[hot_key] = (
                    config_snapshot,
                    compiled,
                    self._hot_entry_payload(args, positional),
                )
            if env.compile.compile_only:
                return None
            # flyc.compile performs the first launch while producing the callable.
            if compiled_now:
                return None
            return compiled(*positional)


__all__ = ["FlydslL2Autotuner", "l2_cold_bench"]

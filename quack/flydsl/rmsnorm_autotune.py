# Copyright (c) 2026, Tri Dao.

"""Native FlyDSL autotuning for the RMSNorm forward direct-JIT entry."""

import os
import statistics
from dataclasses import dataclass

import torch
from flydsl.autotune import (
    Config,
    _env_fingerprint,
    _toolchain_fingerprint,
)
from flydsl.compiler import CompiledFunction

from . import autotune_harness as _autotune_harness
from .autotune_harness import FlydslL2Autotuner, l2_cold_bench
from .rmsnorm_config import (
    MAX_TUNED_NUM_THREADS,
    REGISTER_CACHE_ELEMS,
    WAVE_SIZE,
    RmsNormRowConfig,
    batch_short_rows,
    multi_row_block_rows,
)
from .rmsnorm_kernel import rmsnorm_direct

# Compatibility aliases for callers that imported the original private helpers.
_allocate_eviction_buffers = _autotune_harness._allocate_eviction_buffers
_all_tensor_storage_bytes = _autotune_harness._all_tensor_storage_bytes
_AUTOTUNE_BENCH_LOCK = _autotune_harness._AUTOTUNE_BENCH_LOCK
_build_generic_l2_rotation_plan = _autotune_harness._build_l2_rotation_plan
_cache_authority = _autotune_harness._cache_authority
_CacheAuthority = _autotune_harness._CacheAuthority
_clear_cache = _autotune_harness._clear_cache
_clone_python_function = _autotune_harness._clone_python_function
_clone_tensor_arguments = _autotune_harness._clone_tensor_arguments
_event_l2_rotate_bench = _autotune_harness._event_l2_rotate_bench
_FAST_CONTEXT_ENV_VARS = _autotune_harness._FAST_CONTEXT_ENV_VARS
_FAST_CONTEXT_ENV_VARS_BYTES = _autotune_harness._FAST_CONTEXT_ENV_VARS_BYTES
FLYDSL_BUILD_LOCK = _autotune_harness.FLYDSL_BUILD_LOCK
_L2_EVICTION_CHUNK_BYTES = _autotune_harness._L2_EVICTION_CHUNK_BYTES
_L2_MAX_BUFFER_SETS = _autotune_harness._L2_MAX_BUFFER_SETS
_L2_MAX_EXTRA_BYTES = _autotune_harness._L2_MAX_EXTRA_BYTES
_L2_REPLAY_SAMPLES = _autotune_harness._L2_REPLAY_SAMPLES
_L2_TARGET_RATIO = _autotune_harness._L2_TARGET_RATIO
_L2_TIMED_CALLS = _autotune_harness._L2_TIMED_CALLS
_L2_WARMUP_TARGET_MS = _autotune_harness._L2_WARMUP_TARGET_MS
_L2RotationPlan = _autotune_harness._L2RotationPlan
_ROCM_TRITON_CACHE_BYTES = _autotune_harness._ROCM_TRITON_CACHE_BYTES
_tensor_storage_span_bytes = _autotune_harness._tensor_storage_span_bytes
_typed_identity = _autotune_harness._typed_identity
_unique_tensor_bytes = _autotune_harness._unique_tensor_bytes
_with_runtime_stream = _autotune_harness._with_runtime_stream

RMSNORM_AUTOTUNE_SCHEMA_VERSION = 7
_WAVES_PER_EU = (None, 4)
_EXHAUSTIVE_WAVES_PER_EU = (None, 1, 2, 4)
_EXHAUSTIVE_WAVES_ENV = "QUACK_RMSNORM_EXHAUSTIVE_WAVES"
_CACHE_POLICY_CANDIDATES = ((2, 2),)
_CORRECTNESS_ROWS = 16
_RERANK_RELATIVE_BAND = 0.10
_RERANK_ABSOLUTE_BAND_MS = 0.005
_RERANK_FINAL_TIE_BAND = 0.02
_RERANK_CALLS = 64
_RERANK_SAMPLES = 3


def _waves_per_eu_candidates():
    value = os.environ.get(_EXHAUSTIVE_WAVES_ENV, "").strip().lower()
    return _EXHAUSTIVE_WAVES_PER_EU if value in {"1", "true", "yes", "on"} else _WAVES_PER_EU


def _persistent_program_candidates(m: int, num_cus: int) -> list[int]:
    programs = set()
    for divisor in (4, 2):
        target = (m + divisor - 1) // divisor
        rounded = ((target + num_cus - 1) // num_cus) * num_cus
        programs.add(min(m, max(num_cus, rounded)))
    return sorted(programs)


def _row_candidates(n: int, dtype_width: int) -> list[int]:
    heuristic = (
        RmsNormRowConfig.for_lane_group(n, dtype_width)
        if batch_short_rows(n, dtype_width)
        else RmsNormRowConfig.from_register_budget(n, dtype_width)
    )
    ceiling = 64 if batch_short_rows(n, dtype_width) else MAX_TUNED_NUM_THREADS
    candidates = {heuristic.num_threads}
    candidates.update((heuristic.num_threads // 2, heuristic.num_threads * 2))
    if not batch_short_rows(n, dtype_width):
        # 512 is above what the heuristic may pick: it wins only when the row
        # is wide and there are many of them, and only the tuner sees M.
        candidates.update((64, 128, 256, 512))
        # Add the first wider workgroup that restores the bounded register
        # fragment. This is a resource rule, not a benchmark-shape lookup.
        for threads in (1024,):
            config = RmsNormRowConfig.with_num_threads(
                n,
                dtype_width,
                threads,
                max_num_threads=MAX_TUNED_NUM_THREADS,
            )
            if config.elems_per_thread <= REGISTER_CACHE_ELEMS:
                candidates.add(threads)
                break

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

    def append(**options):
        identity = tuple(sorted(options.items()))
        if identity not in seen:
            seen.add(identity)
            configs.append(Config(**options))

    threads_candidates = _row_candidates(n, dtype_width)
    for threads in threads_candidates:
        for waves_per_eu in _waves_per_eu_candidates():
            append(threads_per_row=threads, waves_per_eu=waves_per_eu)

    plain_bf16 = (
        len(args) >= 8
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
    )
    if not plain_bf16:
        return configs

    # Cache-policy exploration is explicit and shape-independent. Keep one
    # variant per policy at the analytical width to bound compile/search cost.
    default_threads = rmsnorm_default_config(*args, **kwargs).kwargs["threads_per_row"]
    for input_modifier, output_modifier in _CACHE_POLICY_CANDIDATES:
        append(
            threads_per_row=default_threads,
            input_cache_modifier=input_modifier,
            output_cache_modifier=output_modifier,
        )

    if args[0].stride(0) != n:
        return configs

    m = int(args[7])
    num_cus = torch.cuda.get_device_properties(args[0].device).multi_processor_count
    persistent_threads = max(
        (
            threads
            for threads in threads_candidates
            if threads <= WAVE_SIZE
            and RmsNormRowConfig.with_num_threads(
                n,
                dtype_width,
                threads,
                max_num_threads=MAX_TUNED_NUM_THREADS,
            ).reload_from
            != "gmem"
        ),
        default=None,
    )
    for threads in threads_candidates:
        if threads != persistent_threads:
            continue
        row = RmsNormRowConfig.with_num_threads(
            n,
            dtype_width,
            threads,
            max_num_threads=MAX_TUNED_NUM_THREADS,
        )
        if threads > WAVE_SIZE or row.reload_from == "gmem":
            continue
        row_groups_per_block = multi_row_block_rows(threads)
        if row_groups_per_block <= 1:
            continue
        available_blocks = (m + row_groups_per_block - 1) // row_groups_per_block
        if available_blocks < num_cus:
            continue
        append(
            threads_per_row=threads,
            row_groups_per_block=row_groups_per_block,
            persistent_programs=available_blocks,
            output_cache_modifier=3,
            persistent_single_pass=True,
            packed_flat_rows=True,
        )
        for persistent_programs in _persistent_program_candidates(m, num_cus):
            append(
                threads_per_row=threads,
                row_groups_per_block=1,
                persistent_programs=persistent_programs,
            )
    return configs


def rmsnorm_default_config(*args, **kwargs) -> Config:
    """Preserve the existing analytical geometry when tuning is not forced."""
    n = int(kwargs["n"])
    dtype_width = 32 if kwargs["input_dtype_str"] == "f32" else 16
    config = (
        RmsNormRowConfig.for_lane_group(n, dtype_width)
        if batch_short_rows(n, dtype_width)
        else RmsNormRowConfig.from_register_budget(n, dtype_width)
    )
    return Config(threads_per_row=config.num_threads)


@dataclass
class _ReferenceSamples:
    row_indices: torch.Tensor
    output: torch.Tensor
    residual_out: torch.Tensor | None
    rstd: torch.Tensor | None


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
    """Build the shared rotation plan with RMSNorm-specific accounting and checks."""
    return _build_generic_l2_rotation_plan(
        args,
        kwargs,
        n_timed_calls=n_timed_calls,
        target_ratio=target_ratio,
        read_bytes_fn=read_bytes_fn or _rmsnorm_cache_read_bytes,
        clone_fn=clone_fn or _clone_tensor_arguments,
        reference_fn=reference_fn or _reference_samples,
        validate_addresses_fn=validate_addresses_fn or _validate_rotation_addresses,
        cache_authority_fn=_cache_authority,
        allocate_eviction_buffers_fn=_allocate_eviction_buffers,
        benchmark_name="RMSNorm",
    )


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


class RmsNormAutotuner(FlydslL2Autotuner):
    """Forward RMSNorm specialization of the shared FlyDSL L2 harness."""

    benchmark_name = "RMSNorm"

    @staticmethod
    def _process_context_key():
        return (
            *FlydslL2Autotuner._process_context_key(),
            ("exhaustive_waves", os.environ.get(_EXHAUSTIVE_WAVES_ENV, "")),
        )

    def _call_hot_entry(self, hot_entry, args, kwargs):
        _config, compiled, constexpr_suffix = hot_entry
        if len(args) == 10 and constexpr_suffix is not None:
            stream = kwargs.get("stream", self._signature.parameters["stream"].default)
            return compiled(*(args + constexpr_suffix + (stream,)))
        return super()._call_hot_entry(hot_entry, args, kwargs)

    def _hot_entry_payload(self, args, positional):
        return positional[10:-1] if len(args) == 10 else None

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

    def _inference_sibling(self, args, kwargs):
        if not kwargs.get("store_rstd", False):
            return None
        inference_kwargs = dict(kwargs)
        inference_kwargs["store_rstd"] = False
        key = self._contextual_decision_key(args, inference_kwargs)
        return self.cache.get(key)

    def _tune_configs(self, args, kwargs):
        sibling = self._inference_sibling(args, kwargs)
        if sibling is None:
            return super()._tune_configs(args, kwargs)
        candidate = Config(
            threads_per_row=sibling.kwargs["threads_per_row"],
            waves_per_eu=sibling.waves_per_eu,
        )
        results = self._benchmark_configs([candidate], args, kwargs)
        return self._select_result(results, args, kwargs)

    @staticmethod
    def _config_identity(config):
        return _typed_identity(config.to_dict())

    def _deployment_time(self, config, args, kwargs, plan):
        compiled, _positional, _ = self._compiled_callable(config, args, kwargs)
        arg_sets = plan.arg_sets[:2]
        kwarg_sets = plan.kwarg_sets[:2]
        positional_sets = [
            self._positional_arguments(config, set_args, set_kwargs)
            for set_args, set_kwargs in zip(arg_sets, kwarg_sets)
        ]
        positional_sets = _with_runtime_stream(positional_sets, plan.bench_stream)
        with torch.cuda.stream(plan.bench_stream):
            for index in range(8):
                compiled(*positional_sets[index % len(positional_sets)])
            plan.bench_stream.synchronize()
            samples = []
            for sample in range(_RERANK_SAMPLES):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for index in range(_RERANK_CALLS):
                    compiled(*positional_sets[(sample + index) % len(positional_sets)])
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / _RERANK_CALLS)
        return statistics.median(samples)

    def _rerank_result(self, results, selected, args, kwargs):
        plan = getattr(self._active_call, "rotation_plan", None)
        if plan is None or self._do_bench is not l2_cold_bench:
            return selected
        cold_best = selected[1]
        threshold = cold_best + max(
            cold_best * _RERANK_RELATIVE_BAND,
            _RERANK_ABSOLUTE_BAND_MS,
        )
        incumbent = rmsnorm_default_config(*args, **kwargs)
        incumbent_identity = self._config_identity(incumbent)
        candidates = [
            config
            for config, elapsed in results
            if elapsed <= threshold or self._config_identity(config) == incumbent_identity
        ]
        reranked = [
            (config, self._deployment_time(config, args, kwargs, plan)) for config in candidates
        ]
        for config, elapsed in reranked:
            print(f"  [rerank] {config} -> {elapsed:.3f} ms")
        best_time = min(elapsed for _config, elapsed in reranked)
        final = [
            (config, elapsed)
            for config, elapsed in reranked
            if elapsed <= best_time * (1.0 + _RERANK_FINAL_TIE_BAND)
        ]
        persistent = [
            (config, elapsed)
            for config, elapsed in final
            if config.kwargs.get("persistent_programs")
        ]
        if persistent:
            return min(
                persistent,
                key=lambda item: (item[1], self._config_identity(item[0])),
            )
        for config, elapsed in final:
            if self._config_identity(config) == incumbent_identity:
                return config, elapsed
        return min(final, key=lambda item: (item[1], self._config_identity(item[0])))

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

# Copyright (c) 2026, Tri Dao.

"""Native FlyDSL autotuning for the deterministic staged RMSNorm backward."""

import os
from dataclasses import dataclass

import torch
from flydsl.autotune import Config, _env_fingerprint, _toolchain_fingerprint

from .rmsnorm_autotune import (
    RmsNormAutotuner,
    _build_l2_rotation_plan,
    _clone_tensor_arguments,
    _typed_identity,
    _with_runtime_stream,
    l2_cold_bench,
)
from .rmsnorm_bwd_kernel import (
    PARAMETER_REDUCE_THREADS,
    TWO_STAGE_MAX_NUM_THREADS,
    rmsnorm_bwd_direct,
    rmsnorm_bwd_parameter_reduce_cols,
    rmsnorm_bwd_two_stage_config,
)
from .rmsnorm_common import dtype_to_elem_bits
from .rmsnorm_config import WAVE_SIZE, RmsNormRowConfig, next_power_of_two

RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION = 2
_MAX_WORKSPACE_BYTES = 4 * 1024**3
_CORRECTNESS_ROWS = 16
_SMALL_N_MAX = 1024
_SMALL_N_PROGRAMS_PER_CU = (2, 4, 6, 8, 12)


def _call_values(args, kwargs):
    names = (
        "source_tensor",
        "weight_tensor",
        "dy_tensor",
        "dresidual_out_tensor",
        "rstd_tensor",
        "correction_tensor",
        "dx_tensor",
        "dresidual_tensor",
        "dweight_tensor",
        "dbias_tensor",
        "workspace_tensor",
        "workspace_flat",
        "m",
        "weight_offset",
    )
    values = dict(zip(names, args))
    values.update(kwargs)
    return values


def _threads_candidates(n: int, dtype_width: int) -> list[int]:
    heuristic = RmsNormRowConfig.from_analytical_heuristic(
        n,
        dtype_width,
        TWO_STAGE_MAX_NUM_THREADS,
    )
    candidates = {
        heuristic.num_threads,
        heuristic.num_threads // 2,
        heuristic.num_threads * 2,
        64,
        128,
        256,
        512,
    }
    legal = []
    for threads in sorted(candidates):
        if threads < WAVE_SIZE or threads > TWO_STAGE_MAX_NUM_THREADS or threads & (threads - 1):
            continue
        config = RmsNormRowConfig.with_num_threads(
            n,
            dtype_width,
            threads,
            max_num_threads=TWO_STAGE_MAX_NUM_THREADS,
        )
        if threads == WAVE_SIZE or config.num_vecs >= threads:
            legal.append(threads)
    return legal


def _programs_for_threads(values, threads: int) -> int:
    m = int(values["m"])
    device = values["source_tensor"].device
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    programs = num_cus if m < 2048 else (3 * num_cus) // 2
    programs *= TWO_STAGE_MAX_NUM_THREADS // threads
    if int(values["n"]) <= _SMALL_N_MAX:
        programs = min(programs, _SMALL_N_PROGRAMS_PER_CU[-1] * num_cus)
    programs = min(next_power_of_two(m), programs)
    if values["per_head"]:
        programs = max(1, next_power_of_two(programs // int(values["num_heads"])))
    return programs


def _program_candidates(values, threads: int) -> list[int]:
    base = _programs_for_threads(values, threads)
    if int(values["n"]) > _SMALL_N_MAX:
        return sorted(
            {
                max(1, base // 2),
                base,
                min(next_power_of_two(int(values["m"])), base * 2),
            }
        )

    device = values["source_tensor"].device
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    row_bound = next_power_of_two(int(values["m"]))
    candidates = {base}
    for multiplier in _SMALL_N_PROGRAMS_PER_CU:
        programs = min(row_bound, multiplier * num_cus)
        if values["per_head"]:
            programs = max(
                1,
                next_power_of_two(programs // int(values["num_heads"])),
            )
        candidates.add(programs)
    return sorted(candidates)


def _parameter_reduce_cols_candidates(values, num_programs: int) -> list[int]:
    parameter_numel = int(values["num_heads"]) * int(values["n"])
    device = values["source_tensor"].device
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    heuristic = rmsnorm_bwd_parameter_reduce_cols(
        parameter_numel,
        num_programs,
        target_blocks=num_cus,
    )
    candidates = {
        max(1, heuristic // 2),
        heuristic,
        min(PARAMETER_REDUCE_THREADS, heuristic * 2),
    }
    return sorted(candidates)


def _workspace_bytes(values, num_programs: int) -> int:
    rows = (
        num_programs
        * int(values["num_heads"])
        * (int(values["compute_dweight"]) + int(values["compute_dbias"]))
    )
    return max(1, rows) * int(values["n"]) * 4


def rmsnorm_bwd_search_configs(*args, **kwargs) -> list[Config]:
    values = _call_values(args, kwargs)
    n = int(values["n"])
    dtype_width = dtype_to_elem_bits(values["source_dtype_str"])
    device = values["source_tensor"].device
    with torch.cuda.device(device):
        free_bytes, _ = torch.cuda.mem_get_info(device)
    budget = min(_MAX_WORKSPACE_BYTES, int(free_bytes * 0.25))
    configs = []
    seen = set()
    for threads in _threads_candidates(n, dtype_width):
        for num_programs in _program_candidates(values, threads):
            if _workspace_bytes(values, num_programs) > budget:
                continue
            for parameter_reduce_cols in _parameter_reduce_cols_candidates(
                values,
                num_programs,
            ):
                identity = (threads, num_programs, parameter_reduce_cols)
                if identity in seen:
                    continue
                seen.add(identity)
                configs.append(
                    Config(
                        threads_per_row=threads,
                        num_programs=num_programs,
                        parameter_reduce_cols=parameter_reduce_cols,
                    )
                )
    if not configs:
        raise RuntimeError("RMSNorm backward autotuning found no workspace-legal configs")
    return configs


def rmsnorm_bwd_default_config(*args, **kwargs) -> Config:
    values = _call_values(args, kwargs)
    row = rmsnorm_bwd_two_stage_config(
        int(values["n"]),
        values["source_dtype_str"],
    )
    num_programs = _programs_for_threads(values, row.num_threads)
    device = values["source_tensor"].device
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    return Config(
        threads_per_row=row.num_threads,
        num_programs=num_programs,
        parameter_reduce_cols=rmsnorm_bwd_parameter_reduce_cols(
            int(values["num_heads"]) * int(values["n"]),
            num_programs,
            target_blocks=num_cus,
        ),
    )


@dataclass
class _BackwardReference:
    row_indices: torch.Tensor
    dx: torch.Tensor | None
    dresidual: torch.Tensor | None
    dweight: torch.Tensor | None
    dbias: torch.Tensor | None


def _backward_reference(args, kwargs) -> _BackwardReference:
    values = _call_values(args, kwargs)
    n = int(values["n"])
    num_heads = int(values["num_heads"])
    source_rows = values["source_tensor"].reshape(-1, n)
    dy_rows = values["dy_tensor"].reshape(-1, n)
    rstd = values["rstd_tensor"].reshape(-1)
    total_rows = source_rows.shape[0]
    sample_count = min(_CORRECTNESS_ROWS, total_rows)
    row_indices = (
        torch.linspace(
            0,
            total_rows - 1,
            sample_count,
            device=source_rows.device,
            dtype=torch.float64,
        )
        .round()
        .to(torch.int64)
    )
    row_indices = torch.unique(row_indices, sorted=True)

    with torch.no_grad():
        source = source_rows.index_select(0, row_indices).float()
        dy = dy_rows.index_select(0, row_indices).float()
        selected_rstd = rstd.index_select(0, row_indices).float().unsqueeze(-1)
        if values["has_weight"]:
            weight = values["weight_tensor"]
            if values["per_head"]:
                heads = row_indices.remainder(num_heads)
                weight = weight.index_select(0, heads)
            weighted_dy = dy * (weight.float() + float(values["weight_offset"]))
        else:
            weighted_dy = dy
        source_hat = source * selected_rstd
        correction = (source_hat * weighted_dy).mean(dim=-1, keepdim=True)
        total = (weighted_dy - source_hat * correction) * selected_rstd
        if values["has_dresidual_out"]:
            dresidual_out = values["dresidual_out_tensor"].reshape(-1, n)
            total = total + dresidual_out.index_select(0, row_indices).float()
        dx = total.to(values["dx_tensor"].dtype) if values["store_dx"] else None
        dresidual = (
            total.to(values["dresidual_tensor"].dtype) if values["store_dresidual"] else None
        )

        parameter_shape = (num_heads, n) if values["per_head"] else (n,)
        dweight = (
            torch.zeros(parameter_shape, device=source.device, dtype=torch.float32)
            if values["compute_dweight"]
            else None
        )
        dbias = (
            torch.zeros(parameter_shape, device=source.device, dtype=torch.float32)
            if values["compute_dbias"]
            else None
        )
        chunk_rows = max(1, min(64, 256 * 1024**2 // max(1, n * 12)))
        for start in range(0, total_rows, chunk_rows):
            stop = min(total_rows, start + chunk_rows)
            source_chunk = source_rows[start:stop].float()
            dy_chunk = dy_rows[start:stop].float()
            rstd_chunk = rstd[start:stop].float().unsqueeze(-1)
            if values["compute_dweight"]:
                contribution = dy_chunk * source_chunk * rstd_chunk
                if values["per_head"]:
                    heads = torch.arange(start, stop, device=source.device).remainder(num_heads)
                    dweight.index_add_(0, heads, contribution)
                else:
                    dweight.add_(contribution.sum(dim=0))
            if values["compute_dbias"]:
                if values["per_head"]:
                    heads = torch.arange(start, stop, device=source.device).remainder(num_heads)
                    dbias.index_add_(0, heads, dy_chunk)
                else:
                    dbias.add_(dy_chunk.sum(dim=0))
        if dweight is not None:
            dweight = dweight.to(values["dweight_tensor"].dtype)
        if dbias is not None:
            dbias = dbias.to(values["dbias_tensor"].dtype)
    return _BackwardReference(row_indices, dx, dresidual, dweight, dbias)


def _cache_read_bytes(args, kwargs) -> int:
    values = _call_values(args, kwargs)
    names = ["source_tensor", "dy_tensor", "rstd_tensor"]
    if values["has_weight"]:
        names.append("weight_tensor")
    if values["has_dresidual_out"]:
        names.append("dresidual_out_tensor")
    seen = set()
    total = 0
    for name in names:
        tensor = values[name]
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def _clone_backward_arguments(args, kwargs):
    cloned_args, cloned_kwargs = _clone_tensor_arguments(args, kwargs)
    mutable = list(cloned_args)
    mutable[11] = mutable[10].view(-1)
    return tuple(mutable), cloned_kwargs


def _validate_addresses(plan):
    active_indices = [0, 2, 4, 6, 8, 10]
    values = _call_values(plan.arg_sets[0], plan.kwarg_sets[0])
    if values["has_weight"]:
        active_indices.append(1)
    if values["has_dresidual_out"]:
        active_indices.append(3)
    if values["store_dresidual"]:
        active_indices.append(7)
    if values["compute_dbias"]:
        active_indices.append(9)
    if values["correction_tensor"] is not values["rstd_tensor"]:
        active_indices.append(5)
    for index in active_indices:
        pointers = [
            call_args[index].data_ptr() for call_args in plan.arg_sets if call_args[index].numel()
        ]
        if len(pointers) != len(set(pointers)):
            raise RuntimeError(f"L2-cold backward rotation reused tensor argument {index}")


def _candidate_correctness_gate(compiled, positional_sets, plan):
    expected = plan.expected
    n = int(_call_values(plan.arg_sets[0], plan.kwarg_sets[0])["n"])
    for set_index, (args, positional) in enumerate(zip(plan.arg_sets, positional_sets)):
        compiled(*positional)
        values = _call_values(args, plan.kwarg_sets[set_index])
        try:
            if expected.dx is not None:
                actual = values["dx_tensor"].reshape(-1, n).index_select(0, expected.row_indices)
                torch.testing.assert_close(actual, expected.dx, rtol=3e-2, atol=3e-2)
            if expected.dresidual is not None:
                actual = (
                    values["dresidual_tensor"].reshape(-1, n).index_select(0, expected.row_indices)
                )
                torch.testing.assert_close(
                    actual,
                    expected.dresidual,
                    rtol=3e-2,
                    atol=3e-2,
                )
            if expected.dweight is not None:
                torch.testing.assert_close(
                    values["dweight_tensor"],
                    expected.dweight,
                    rtol=3e-2,
                    atol=3e-2,
                )
            if expected.dbias is not None:
                torch.testing.assert_close(
                    values["dbias_tensor"],
                    expected.dbias,
                    rtol=3e-2,
                    atol=3e-2,
                )
        except AssertionError as error:
            raise AssertionError(
                "RMSNorm backward autotune candidate failed correctness on "
                f"clone set {set_index}: {error}"
            ) from error
    plan.bench_stream.synchronize()


class RmsNormBwdAutotuner(RmsNormAutotuner):
    def __call__(self, *args, **kwargs):
        self._active_call.bwd_expected = None
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._active_call.bwd_expected = None

    def _hot_key(self, args, kwargs):
        if self._toolchain_key is None:
            self._toolchain_key = _toolchain_fingerprint()
        explicit = (args[12],) + tuple(kwargs[name] for name in self.key[1:])
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

    def _candidate_arguments(self, config, args, kwargs):
        values = _call_values(args, kwargs)
        mutable = list(args)
        num_programs = int(config.kwargs["num_programs"])
        workspace_rows = (
            num_programs
            * int(values["num_heads"])
            * (int(values["compute_dweight"]) + int(values["compute_dbias"]))
        )
        workspace = torch.empty(
            (workspace_rows, int(values["n"])) if workspace_rows else (1, int(values["n"])),
            device=values["source_tensor"].device,
            dtype=torch.float32,
        )
        row = RmsNormRowConfig.with_num_threads(
            int(values["n"]),
            dtype_to_elem_bits(values["source_dtype_str"]),
            int(config.kwargs["threads_per_row"]),
            max_num_threads=TWO_STAGE_MAX_NUM_THREADS,
        )
        correction = (
            torch.empty(
                int(values["m"]) * int(values["num_heads"]),
                device=values["source_tensor"].device,
                dtype=torch.float32,
            )
            if row.reload_from == "gmem" and values["compute_input_grad"]
            else values["rstd_tensor"]
        )
        mutable[5] = correction
        mutable[10] = workspace
        mutable[11] = workspace.view(-1)
        return tuple(mutable)

    def _prepare_call(self, config, args, kwargs):
        candidate_args = self._candidate_arguments(config, args, kwargs)
        merged = dict(kwargs)
        merged.update(config.all_kwargs())
        bound = self._signature.bind(*candidate_args, **merged)
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

    def _positional_arguments(self, config, args, kwargs):
        candidate_args = self._candidate_arguments(config, args, kwargs)
        values = dict(zip(self.arg_names, candidate_args))
        values.update(kwargs)
        values.update(config.all_kwargs())
        return tuple(
            values.get(name, parameter.default)
            for name, parameter in self._signature.parameters.items()
        )

    def _bench_one(self, config, args, kwargs):
        candidate_args = self._candidate_arguments(config, args, kwargs)
        compiled, positional, _ = self._compiled_callable(config, args, kwargs)
        if self._do_bench is not l2_cold_bench:

            def kernel_call():
                compiled(*positional)

            return self._do_bench(kernel_call, warmup=self.warmup, rep=self.rep)
        expected = getattr(self._active_call, "bwd_expected", None)
        if expected is None:
            expected = _backward_reference(args, kwargs)
            self._active_call.bwd_expected = expected
        plan = _build_l2_rotation_plan(
            candidate_args,
            kwargs,
            read_bytes_fn=_cache_read_bytes,
            clone_fn=_clone_backward_arguments,
            reference_fn=lambda _args, _kwargs: expected,
            validate_addresses_fn=_validate_addresses,
        )
        positional_sets = [
            self._positional_arguments(config, set_args, set_kwargs)
            for set_args, set_kwargs in zip(plan.arg_sets, plan.kwarg_sets)
        ]
        positional_sets = _with_runtime_stream(positional_sets, plan.bench_stream)
        with torch.cuda.stream(plan.bench_stream):
            _candidate_correctness_gate(compiled, positional_sets, plan)
            return l2_cold_bench(
                compiled,
                positional_sets,
                plan,
                warmup_target_ms=float(self.warmup),
                replay_samples=int(self.rep),
            )


_RMSNORM_BWD_AUTOTUNE_KEY = [
    "m",
    "n",
    "source_dtype_str",
    "dy_dtype_str",
    "dx_dtype_str",
    "dresidual_dtype_str",
    "dresidual_out_dtype_str",
    "weight_dtype_str",
    "dbias_dtype_str",
    "has_weight",
    "has_bias",
    "compute_dweight",
    "compute_dbias",
    "compute_input_grad",
    "store_dx",
    "store_dresidual",
    "has_residual",
    "has_dresidual_out",
    "per_head",
    "num_heads",
    "arch",
    "schema_version",
]

_rmsnorm_bwd_tuner = RmsNormBwdAutotuner(
    fn=rmsnorm_bwd_direct,
    configs=rmsnorm_bwd_search_configs,
    key=_RMSNORM_BWD_AUTOTUNE_KEY,
    warmup=200.0,
    rep=3,
    do_bench_fn=l2_cold_bench,
    default=rmsnorm_bwd_default_config,
    artifact_name="quack_rmsnorm_bwd",
)

__all__ = [
    "RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION",
    "RmsNormBwdAutotuner",
    "_rmsnorm_bwd_tuner",
    "rmsnorm_bwd_default_config",
    "rmsnorm_bwd_search_configs",
]

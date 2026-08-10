# Copyright (c) 2026, Tri Dao.

"""Host-side launch and build policy for the FlyDSL RMSNorm backend."""

import os

import torch

from .rmsnorm_autotune import (
    RMSNORM_AUTOTUNE_SCHEMA_VERSION,
    _rmsnorm_fwd_tuner,
)
from .rmsnorm_bwd_autotune import (
    RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION,
    _rmsnorm_bwd_tuner,
)
from .rmsnorm_bwd_kernel import (
    TWO_STAGE_MAX_NUM_THREADS,
    build_rmsnorm_bwd_two_stage_module,
    rmsnorm_bwd_two_stage_config,
)
from .rmsnorm_common import FLYDSL_BUILD_LOCK, run_compiled
from .rmsnorm_config import next_power_of_two
from .rmsnorm_kernel import build_rmsnorm_module
from .rmsnorm_preflight import (
    _VALIDATED_INPUT_LAST,
    _validate_arch,
    _validated_autotune_arch,
)

_FWD_CACHE: dict[tuple, object] = {}
_BWD_CACHE: dict[tuple, object] = {}
_FWD_AUTOTUNED_FAST_CACHE: dict[tuple, tuple] = {}
_BWD_AUTOTUNED_FAST_CACHE: dict[tuple, tuple] = {}
_FWD_AUTOTUNED_LAST: list[tuple[tuple, tuple] | None] = [None]
_FWD_PUBLIC_GENERIC_LAST: list[object | None] = [None]
_FWD_PUBLIC_RESOLVED_LAST: list[tuple[object, tuple, tuple] | None] = [None]
_BWD_CU_COUNT_CACHE: dict[torch.device, int] = {}


def _dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "f32"
    raise TypeError(f"unsupported dtype: {dtype}")


def _current_raw_stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _env_flag_enabled(name: str) -> bool:
    data = getattr(os.environ, "_data", None)
    if data is not None:
        value = data.get(os.fsencode(name), b"").strip().lower()
        return value in {b"1", b"true", b"yes", b"on"}
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_cached(cache: dict, key: tuple, device: torch.device, build):
    """Build a launcher at most once, even if several threads race here.

    This is also the only place the architecture is validated, and the
    validated architecture is what the builder specializes on, so the kernels
    can never disagree with the target FlyDSL will compile for.
    """
    with FLYDSL_BUILD_LOCK:
        launcher = cache.get(key)
        if launcher is None:
            launcher = build(_validate_arch(device))
            cache[key] = launcher
    return launcher


def _select_rmsnorm_bwd_programs(
    m: int,
    n: int,
    dtype_str: str,
    device: torch.device,
) -> int:
    """Size the persistent grid for the staged weight-gradient reduction.

    There is one reduction path at every row count. A small-row atomic variant
    existed and was removed: fp32 atomics need a zeroed accumulator and a cast
    back to the weight dtype, which is two torch launches to save one FlyDSL
    launch, and its scalar kernel lost on device time as well. See
    AI/flydsl_rmsnorm_notes.md.
    """
    num_cus = _BWD_CU_COUNT_CACHE.get(device)
    if num_cus is None:
        num_cus = torch.cuda.get_device_properties(device).multi_processor_count
        if not torch.compiler.is_compiling():
            _BWD_CU_COUNT_CACHE[device] = num_cus
    config = rmsnorm_bwd_two_stage_config(n, dtype_str)
    num_programs = num_cus if m < 2048 else (3 * num_cus) // 2
    # A row narrower than the widest block gets a narrower block, so launch
    # proportionally more of them to keep the same threads resident.
    num_programs *= TWO_STAGE_MAX_NUM_THREADS // config.num_threads
    # num_programs is baked into the kernel as the grid size, the row-loop
    # stride and the reduce bound, so it must not track m: `min(m, ...)` would
    # compile and retain a separate kernel for every batch size. Rounding to a
    # power of two bounds that at one entry per octave. A block with no row of
    # its own just contributes a zeroed partial.
    return min(next_power_of_two(m), num_programs)


def _launch_rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    weight_offset: float,
    *,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    per_head: bool,
    num_heads: int,
) -> None:
    m, n = x.shape[0], x.shape[-1]
    dtype_str = _dtype_to_str(x.dtype)
    output_dtype_str = _dtype_to_str(out.dtype)
    weight_dtype_str = _dtype_to_str(weight.dtype)
    bias_dtype_str = _dtype_to_str(bias.dtype)
    residual_dtype_str = _dtype_to_str(residual.dtype)
    residual_out_dtype_str = _dtype_to_str(residual_out.dtype)
    apply_weight_offset = weight_offset != 0.0

    with torch.cuda.device(x.device):
        key = (
            x.device.index,
            n,
            dtype_str,
            output_dtype_str,
            weight_dtype_str,
            bias_dtype_str,
            residual_dtype_str,
            residual_out_dtype_str,
            has_weight,
            has_bias,
            has_residual,
            store_residual,
            store_rstd,
            per_head,
            num_heads,
            apply_weight_offset,
        )
        launcher = _FWD_CACHE.get(key)
        if launcher is None:

            def build_forward(arch):
                built = build_rmsnorm_module(
                    n,
                    dtype_str,
                    output_dtype_str,
                    weight_dtype_str=weight_dtype_str,
                    bias_dtype_str=bias_dtype_str,
                    residual_dtype_str=residual_dtype_str,
                    residual_out_dtype_str=residual_out_dtype_str,
                    has_weight=has_weight,
                    has_bias=has_bias,
                    has_residual=has_residual,
                    store_residual=store_residual,
                    store_rstd=store_rstd,
                    per_head=per_head,
                    num_heads=num_heads,
                    arch=arch,
                    apply_weight_offset=apply_weight_offset,
                )
                return built

            launcher = _build_cached(
                _FWD_CACHE,
                key,
                x.device,
                build_forward,
            )
        run_compiled(
            launcher,
            x,
            weight,
            bias,
            residual,
            out,
            residual_out,
            rstd,
            m,
            eps,
            weight_offset,
            _current_raw_stream(x.device),
        )


def _is_generic_forward_config(config) -> bool:
    return config.waves_per_eu is None and set(config.kwargs) == {"threads_per_row"}


def _public_runtime_guard():
    return (
        os.environ.get("FLYDSL_COMPILE_BACKEND", ""),
        os.environ.get("FLYDSL_RUNTIME_KIND", ""),
        os.environ.get("FLYDSL_GPU_ARCH", ""),
        os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""),
        os.environ.get("ARCH", ""),
        tuple(sorted(getattr(_rmsnorm_fwd_tuner.fn, "compile_hints", {}).items())),
    )


def _forward_public_key(
    x,
    weight,
    bias,
    residual,
    out,
    residual_out,
    *,
    has_weight,
    has_bias,
    has_residual,
    store_residual,
    store_rstd,
    per_head,
    num_heads,
):
    return (
        x.device.index,
        x.shape[0],
        x.shape[-1],
        x.dtype,
        out.dtype,
        weight.dtype,
        bias.dtype,
        residual.dtype,
        residual_out.dtype,
        has_weight,
        has_bias,
        has_residual,
        store_residual,
        store_rstd,
        per_head,
        num_heads,
    )


def _launch_resolved_fwd_entry(
    entry,
    x,
    weight,
    bias,
    residual,
    out,
    residual_out,
    rstd,
    m,
    eps,
    weight_offset,
    *,
    has_weight,
    has_bias,
    has_residual,
    store_residual,
    store_rstd,
    per_head,
    num_heads,
) -> None:
    config, compiled, constexpr_suffix = entry
    _FWD_PUBLIC_RESOLVED_LAST[0] = (
        _VALIDATED_INPUT_LAST[0],
        _public_runtime_guard(),
        entry,
    )
    if _is_generic_forward_config(config):
        _FWD_PUBLIC_GENERIC_LAST[0] = _VALIDATED_INPUT_LAST[0]
        _launch_rmsnorm_fwd(
            x,
            weight,
            bias,
            residual,
            out,
            residual_out,
            rstd,
            eps,
            weight_offset,
            has_weight=has_weight,
            has_bias=has_bias,
            has_residual=has_residual,
            store_residual=store_residual,
            store_rstd=store_rstd,
            per_head=per_head,
            num_heads=num_heads,
        )
        return
    if _FWD_PUBLIC_GENERIC_LAST[0] is _VALIDATED_INPUT_LAST[0]:
        _FWD_PUBLIC_GENERIC_LAST[0] = None
    compiled(
        *(
            (x, weight, bias, residual, out, residual_out, rstd, m, eps, weight_offset)
            + constexpr_suffix
            + (_current_raw_stream(x.device),)
        )
    )


def _launch_rmsnorm_fwd_autotuned(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    weight_offset: float,
    *,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    per_head: bool,
    num_heads: int,
) -> None:
    """Launch through the tuner's ABI/config/device-aware CompiledFunction cache."""
    m, n = x.shape[0], x.shape[-1]
    forced = _env_flag_enabled("FLYDSL_AUTOTUNE")
    compile_only = _env_flag_enabled("COMPILE_ONLY")
    context_token = _rmsnorm_fwd_tuner.fast_context_token()
    public_key = _forward_public_key(
        x,
        weight,
        bias,
        residual,
        out,
        residual_out,
        has_weight=has_weight,
        has_bias=has_bias,
        has_residual=has_residual,
        store_residual=store_residual,
        store_rstd=store_rstd,
        per_head=per_head,
        num_heads=num_heads,
    )
    last_key = (context_token, *public_key)
    last_entry = _FWD_AUTOTUNED_LAST[0]
    if not forced and not compile_only and last_entry is not None and last_entry[0] == last_key:
        with torch.cuda.device(x.device):
            _launch_resolved_fwd_entry(
                last_entry[1],
                x,
                weight,
                bias,
                residual,
                out,
                residual_out,
                rstd,
                m,
                eps,
                weight_offset,
                has_weight=has_weight,
                has_bias=has_bias,
                has_residual=has_residual,
                store_residual=store_residual,
                store_rstd=store_rstd,
                per_head=per_head,
                num_heads=num_heads,
            )
        return

    input_dtype_str = _dtype_to_str(x.dtype)
    output_dtype_str = _dtype_to_str(out.dtype)
    weight_dtype_str = _dtype_to_str(weight.dtype)
    bias_dtype_str = _dtype_to_str(bias.dtype)
    residual_dtype_str = _dtype_to_str(residual.dtype)
    residual_out_dtype_str = _dtype_to_str(residual_out.dtype)
    fast_key = (
        context_token,
        x.device.index,
        m,
        n,
        input_dtype_str,
        output_dtype_str,
        weight_dtype_str,
        bias_dtype_str,
        residual_dtype_str,
        residual_out_dtype_str,
        has_weight,
        has_bias,
        has_residual,
        store_residual,
        store_rstd,
        per_head,
        num_heads,
    )
    with torch.cuda.device(x.device):
        if not forced and not compile_only:
            fast_entry = _FWD_AUTOTUNED_FAST_CACHE.get(fast_key)
            if fast_entry is not None:
                _FWD_AUTOTUNED_LAST[0] = (last_key, fast_entry)
                _launch_resolved_fwd_entry(
                    fast_entry,
                    x,
                    weight,
                    bias,
                    residual,
                    out,
                    residual_out,
                    rstd,
                    m,
                    eps,
                    weight_offset,
                    has_weight=has_weight,
                    has_bias=has_bias,
                    has_residual=has_residual,
                    store_residual=store_residual,
                    store_rstd=store_rstd,
                    per_head=per_head,
                    num_heads=num_heads,
                )
                return

        arch = _validated_autotune_arch(x.device)
        args = (
            x,
            weight,
            bias,
            residual,
            out,
            residual_out,
            rstd,
            m,
            eps,
            weight_offset,
        )
        kwargs = {
            "n": n,
            "input_dtype_str": input_dtype_str,
            "output_dtype_str": output_dtype_str,
            "weight_dtype_str": weight_dtype_str,
            "bias_dtype_str": bias_dtype_str,
            "residual_dtype_str": residual_dtype_str,
            "residual_out_dtype_str": residual_out_dtype_str,
            "has_weight": has_weight,
            "has_bias": has_bias,
            "has_residual": has_residual,
            "store_residual": store_residual,
            "store_rstd": store_rstd,
            "per_head": per_head,
            "num_heads": num_heads,
            "arch": arch,
            "schema_version": RMSNORM_AUTOTUNE_SCHEMA_VERSION,
            "stream": _current_raw_stream(x.device),
        }
        _rmsnorm_fwd_tuner(
            *args,
            **kwargs,
        )
        resolved = _rmsnorm_fwd_tuner.resolved_fast_entry(args, kwargs)
        if resolved is not None:
            _FWD_AUTOTUNED_FAST_CACHE[fast_key] = resolved
            _FWD_AUTOTUNED_LAST[0] = (last_key, resolved)
            _FWD_PUBLIC_RESOLVED_LAST[0] = (
                _VALIDATED_INPUT_LAST[0],
                _public_runtime_guard(),
                resolved,
            )
            if _is_generic_forward_config(resolved[0]):
                _FWD_PUBLIC_GENERIC_LAST[0] = _VALIDATED_INPUT_LAST[0]
            elif _FWD_PUBLIC_GENERIC_LAST[0] is _VALIDATED_INPUT_LAST[0]:
                _FWD_PUBLIC_GENERIC_LAST[0] = None


def _launch_rmsnorm_bwd(
    source: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    dresidual_out: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
    dresidual: torch.Tensor,
    dweight: torch.Tensor,
    dbias: torch.Tensor,
    weight_offset: float,
    *,
    has_weight: bool,
    has_bias: bool,
    compute_dweight: bool,
    compute_dbias: bool,
    compute_input_grad: bool,
    store_dx: bool,
    store_dresidual: bool,
    has_residual: bool,
    has_dresidual_out: bool,
    per_head: bool,
    num_heads: int,
) -> None:
    m, n = source.shape[0], source.shape[-1]
    source_dtype_str = _dtype_to_str(source.dtype)
    dy_dtype_str = _dtype_to_str(dout.dtype)
    dx_dtype_str = _dtype_to_str(dx.dtype)
    dresidual_dtype_str = _dtype_to_str(dresidual.dtype)
    dresidual_out_dtype_str = _dtype_to_str(dresidual_out.dtype)
    weight_dtype_str = _dtype_to_str(weight.dtype)
    dbias_dtype_str = _dtype_to_str(dbias.dtype)

    num_programs = _select_rmsnorm_bwd_programs(m, n, source_dtype_str, source.device)
    if per_head:
        # The staged grid is num_programs * num_heads, and the workspace has a
        # row per block, so the CU-derived count has to be divided by the head
        # count or a per-head launch oversubscribes and allocates num_heads
        # times the workspace it needs. Each program just walks more rows.
        num_programs = max(1, next_power_of_two(num_programs // num_heads))
    workspace_rows = num_programs * num_heads * (int(compute_dweight) + int(compute_dbias))
    workspace = torch.empty(
        (workspace_rows, n) if workspace_rows else (1, n),
        device=source.device,
        dtype=torch.float32,
    )
    wide_bwd = rmsnorm_bwd_two_stage_config(n, source_dtype_str).reload_from == "gmem"
    correction = (
        torch.empty(m * num_heads, device=source.device, dtype=torch.float32)
        if wide_bwd and compute_input_grad
        else rstd
    )
    with torch.cuda.device(source.device):
        key = (
            source.device.index,
            n,
            source_dtype_str,
            dy_dtype_str,
            dx_dtype_str,
            dresidual_dtype_str,
            dresidual_out_dtype_str,
            weight_dtype_str,
            dbias_dtype_str,
            has_weight,
            has_bias,
            compute_dweight,
            compute_dbias,
            compute_input_grad,
            store_dx,
            store_dresidual,
            has_residual,
            has_dresidual_out,
            per_head,
            num_heads,
            num_programs,
        )
        launcher = _BWD_CACHE.get(key)
        if launcher is None:
            launcher = _build_cached(
                _BWD_CACHE,
                key,
                source.device,
                lambda arch: build_rmsnorm_bwd_two_stage_module(
                    n,
                    source_dtype_str,
                    dy_dtype_str,
                    dx_dtype_str,
                    dresidual_dtype_str,
                    dresidual_out_dtype_str,
                    num_programs,
                    weight_dtype_str=weight_dtype_str,
                    dbias_dtype_str=dbias_dtype_str,
                    has_weight=has_weight,
                    has_bias=has_bias,
                    compute_dweight=compute_dweight,
                    compute_dbias=compute_dbias,
                    compute_input_grad=compute_input_grad,
                    store_dx=store_dx,
                    store_dresidual=store_dresidual,
                    has_residual=has_residual,
                    has_dresidual_out=has_dresidual_out,
                    per_head=per_head,
                    num_heads=num_heads,
                    arch=arch,
                ),
            )
        run_compiled(
            launcher,
            source,
            weight,
            dout,
            dresidual_out,
            rstd,
            correction,
            dx,
            dresidual,
            dweight,
            dbias,
            workspace,
            # The partial kernel writes the workspace a row at a time and the
            # reduce reads all of it, so it takes the same memory flat: one
            # descriptor it can hoist out of its accumulation loop.
            workspace.view(-1),
            m,
            weight_offset,
            _current_raw_stream(source.device),
        )


def _launch_rmsnorm_bwd_autotuned(
    source: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    dresidual_out: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
    dresidual: torch.Tensor,
    dweight: torch.Tensor,
    dbias: torch.Tensor,
    weight_offset: float,
    *,
    has_weight: bool,
    has_bias: bool,
    compute_dweight: bool,
    compute_dbias: bool,
    compute_input_grad: bool,
    store_dx: bool,
    store_dresidual: bool,
    has_residual: bool,
    has_dresidual_out: bool,
    per_head: bool,
    num_heads: int,
) -> None:
    """Launch through the resolved backward winner without tuner redispatch."""
    m, n = source.shape[0], source.shape[-1]
    source_dtype_str = _dtype_to_str(source.dtype)
    dy_dtype_str = _dtype_to_str(dout.dtype)
    dx_dtype_str = _dtype_to_str(dx.dtype)
    dresidual_dtype_str = _dtype_to_str(dresidual.dtype)
    dresidual_out_dtype_str = _dtype_to_str(dresidual_out.dtype)
    weight_dtype_str = _dtype_to_str(weight.dtype)
    dbias_dtype_str = _dtype_to_str(dbias.dtype)
    fast_key = (
        _rmsnorm_bwd_tuner.fast_context_token(),
        source.device.index,
        m,
        n,
        source_dtype_str,
        dy_dtype_str,
        dx_dtype_str,
        dresidual_dtype_str,
        dresidual_out_dtype_str,
        weight_dtype_str,
        dbias_dtype_str,
        has_weight,
        has_bias,
        compute_dweight,
        compute_dbias,
        compute_input_grad,
        store_dx,
        store_dresidual,
        has_residual,
        has_dresidual_out,
        per_head,
        num_heads,
    )
    # The tuner replaces these two entries with its candidate workspace before
    # either a tuning or resolved launch. A view keeps this adapter independent
    # of the facade's eager placeholder cache without allocating device storage.
    workspace = rstd[:0]
    args = (
        source,
        weight,
        dout,
        dresidual_out,
        rstd,
        rstd,
        dx,
        dresidual,
        dweight,
        dbias,
        workspace,
        workspace.view(-1),
        m,
        weight_offset,
    )
    kwargs = {
        "n": n,
        "source_dtype_str": source_dtype_str,
        "dy_dtype_str": dy_dtype_str,
        "dx_dtype_str": dx_dtype_str,
        "dresidual_dtype_str": dresidual_dtype_str,
        "dresidual_out_dtype_str": dresidual_out_dtype_str,
        "weight_dtype_str": weight_dtype_str,
        "dbias_dtype_str": dbias_dtype_str,
        "has_weight": has_weight,
        "has_bias": has_bias,
        "compute_dweight": compute_dweight,
        "compute_dbias": compute_dbias,
        "compute_input_grad": compute_input_grad,
        "store_dx": store_dx,
        "store_dresidual": store_dresidual,
        "has_residual": has_residual,
        "has_dresidual_out": has_dresidual_out,
        "per_head": per_head,
        "num_heads": num_heads,
    }
    forced = _env_flag_enabled("FLYDSL_AUTOTUNE")
    compile_only = _env_flag_enabled("COMPILE_ONLY")
    with torch.cuda.device(source.device):
        if not forced and not compile_only:
            fast_entry = _BWD_AUTOTUNED_FAST_CACHE.get(fast_key)
            if fast_entry is not None:
                config, compiled, constexpr_suffix = fast_entry
                runtime_args = _rmsnorm_bwd_tuner._candidate_arguments(config, args, kwargs)
                compiled(*(runtime_args + constexpr_suffix + (_current_raw_stream(source.device),)))
                return

        kwargs.update(
            arch=_validated_autotune_arch(source.device),
            schema_version=RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION,
            stream=_current_raw_stream(source.device),
        )
        _rmsnorm_bwd_tuner(*args, **kwargs)
        resolved = _rmsnorm_bwd_tuner.resolved_fast_entry(args, kwargs)
        if resolved is not None:
            _BWD_AUTOTUNED_FAST_CACHE[fast_key] = resolved

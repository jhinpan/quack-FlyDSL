# Copyright (c) 2026, Tri Dao.

"""ROCm/FlyDSL RMSNorm backend.

On ROCm, the package-level ``quack.rmsnorm`` export resolves lazily to this
module's ``rmsnorm`` function. Importing ``quack`` alone does not import FlyDSL,
and this backend does not alter Quack's existing CUDA/CuTe dispatch. The
``rmsnorm_autotuned`` entry point remains FlyDSL-specific.
"""

import torch

from quack.flydsl import rmsnorm_launch as _rmsnorm_launch
from quack.flydsl import rmsnorm_preflight as _rmsnorm_preflight
from quack.flydsl.rmsnorm_common import EPS

__all__ = ["rmsnorm", "rmsnorm_autotuned"]

_EAGER_EMPTY_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

# Preserve the facade's existing private lookup surface. Mutable caches are
# aliases, not copies, so callers clearing them still affect the owning module.
_SUPPORTED_DTYPES = _rmsnorm_preflight._SUPPORTED_DTYPES
_MAX_ROWS = _rmsnorm_preflight._MAX_ROWS
MAX_N = _rmsnorm_preflight.MAX_N
N_ALIGNMENT = _rmsnorm_preflight.N_ALIGNMENT
_VALIDATED_INPUT_LAST = _rmsnorm_preflight._VALIDATED_INPUT_LAST
_validate_inputs = _rmsnorm_preflight._validate_inputs
_validation_tensor_metadata = _rmsnorm_preflight._validation_tensor_metadata
_validated_inputs = _rmsnorm_preflight._validated_inputs

_packed_rows = _rmsnorm_preflight._packed_rows
_unambiguous_layout = _rmsnorm_preflight._unambiguous_layout
_rows_are_disjoint_and_packed = _rmsnorm_preflight._rows_are_disjoint_and_packed

_SUPPORTED_ARCHES = _rmsnorm_preflight._SUPPORTED_ARCHES
_DEVICE_ARCH_CACHE = _rmsnorm_preflight._DEVICE_ARCH_CACHE
_AUTOTUNE_ARCH_CACHE = _rmsnorm_preflight._AUTOTUNE_ARCH_CACHE
_AUTOTUNE_TARGET_ENV_VARS = _rmsnorm_preflight._AUTOTUNE_TARGET_ENV_VARS
_normalize_arch = _rmsnorm_preflight._normalize_arch
_flydsl_compile_target = _rmsnorm_preflight._flydsl_compile_target
_flydsl_runtime_arch = _rmsnorm_preflight._flydsl_runtime_arch
_validate_arch = _rmsnorm_preflight._validate_arch
_validated_autotune_arch = _rmsnorm_preflight._validated_autotune_arch

RMSNORM_AUTOTUNE_SCHEMA_VERSION = _rmsnorm_launch.RMSNORM_AUTOTUNE_SCHEMA_VERSION
RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION = _rmsnorm_launch.RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION
TWO_STAGE_MAX_NUM_THREADS = _rmsnorm_launch.TWO_STAGE_MAX_NUM_THREADS
FLYDSL_BUILD_LOCK = _rmsnorm_launch.FLYDSL_BUILD_LOCK
next_power_of_two = _rmsnorm_launch.next_power_of_two
run_compiled = _rmsnorm_launch.run_compiled
build_rmsnorm_module = _rmsnorm_launch.build_rmsnorm_module
build_rmsnorm_bwd_two_stage_module = _rmsnorm_launch.build_rmsnorm_bwd_two_stage_module
rmsnorm_bwd_two_stage_config = _rmsnorm_launch.rmsnorm_bwd_two_stage_config
_rmsnorm_fwd_tuner = _rmsnorm_launch._rmsnorm_fwd_tuner
_rmsnorm_bwd_tuner = _rmsnorm_launch._rmsnorm_bwd_tuner
_FWD_CACHE = _rmsnorm_launch._FWD_CACHE
_BWD_CACHE = _rmsnorm_launch._BWD_CACHE
_FWD_AUTOTUNED_FAST_CACHE = _rmsnorm_launch._FWD_AUTOTUNED_FAST_CACHE
_BWD_AUTOTUNED_FAST_CACHE = _rmsnorm_launch._BWD_AUTOTUNED_FAST_CACHE
_FWD_AUTOTUNED_LAST = _rmsnorm_launch._FWD_AUTOTUNED_LAST
_FWD_PUBLIC_GENERIC_LAST = _rmsnorm_launch._FWD_PUBLIC_GENERIC_LAST
_FWD_PUBLIC_RESOLVED_LAST = _rmsnorm_launch._FWD_PUBLIC_RESOLVED_LAST
_BWD_CU_COUNT_CACHE = _rmsnorm_launch._BWD_CU_COUNT_CACHE
_dtype_to_str = _rmsnorm_launch._dtype_to_str
_current_raw_stream = _rmsnorm_launch._current_raw_stream
_env_flag_enabled = _rmsnorm_launch._env_flag_enabled
_build_cached = _rmsnorm_launch._build_cached
_select_rmsnorm_bwd_programs = _rmsnorm_launch._select_rmsnorm_bwd_programs
_launch_rmsnorm_fwd = _rmsnorm_launch._launch_rmsnorm_fwd
_is_generic_forward_config = _rmsnorm_launch._is_generic_forward_config
_public_runtime_guard = _rmsnorm_launch._public_runtime_guard
_forward_public_key = _rmsnorm_launch._forward_public_key
_launch_resolved_fwd_entry = _rmsnorm_launch._launch_resolved_fwd_entry
_launch_rmsnorm_fwd_autotuned = _rmsnorm_launch._launch_rmsnorm_fwd_autotuned
_launch_rmsnorm_bwd = _rmsnorm_launch._launch_rmsnorm_bwd
_launch_rmsnorm_bwd_autotuned = _rmsnorm_launch._launch_rmsnorm_bwd_autotuned


def _eager_empty(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Reuse an internal zero-sized placeholder for eager inference.

    A zero-sized tensor owns no device storage, and callers never observe these
    placeholders. Sharing one per device and dtype therefore removes allocator
    dispatch without creating stream dependencies or cross-device pointers.
    """
    key = (device, dtype)
    result = _EAGER_EMPTY_CACHE.get(key)
    if result is None:
        result = torch.empty(0, device=device, dtype=dtype)
        _EAGER_EMPTY_CACHE[key] = result
    return result


def _dispatch(custom_op, eager_launch, *args, **kwargs) -> None:
    """Use the opaque custom op only while torch is tracing."""
    target = custom_op if torch.compiler.is_compiling() else eager_launch
    target(*args, **kwargs)


def _noop_fake(*args, **kwargs) -> None:
    """Mutation-only custom ops have no fake-tensor work to perform."""


@torch.library.custom_op(
    "quack::_rmsnorm_flydsl_fwd",
    mutates_args=("out", "residual_out", "rstd"),
    device_types="cuda",
    schema=(
        "(Tensor x, Tensor weight, Tensor bias, Tensor residual, "
        "Tensor(a4!) out, Tensor(a5!) residual_out, Tensor(a6!) rstd, "
        "float eps, float weight_offset, bool has_weight, bool has_bias, "
        "bool has_residual, bool store_residual, bool store_rstd, "
        "bool per_head, int num_heads) -> ()"
    ),
)
def _rmsnorm_flydsl_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    weight_offset: float,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    per_head: bool,
    num_heads: int,
) -> None:
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


_rmsnorm_flydsl_fwd_op.register_fake(_noop_fake)


@torch.library.custom_op(
    "quack::_rmsnorm_flydsl_fwd_autotuned",
    mutates_args=("out", "residual_out", "rstd"),
    device_types="cuda",
    schema=(
        "(Tensor x, Tensor weight, Tensor bias, Tensor residual, "
        "Tensor(a4!) out, Tensor(a5!) residual_out, Tensor(a6!) rstd, "
        "float eps, float weight_offset, bool has_weight, bool has_bias, "
        "bool has_residual, bool store_residual, bool store_rstd, "
        "bool per_head, int num_heads) -> ()"
    ),
)
def _rmsnorm_flydsl_fwd_autotuned_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    weight_offset: float,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    per_head: bool,
    num_heads: int,
) -> None:
    _launch_rmsnorm_fwd_autotuned(
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


_rmsnorm_flydsl_fwd_autotuned_op.register_fake(_noop_fake)


@torch.library.custom_op(
    "quack::_rmsnorm_flydsl_bwd",
    mutates_args=("dx", "dresidual", "dweight", "dbias"),
    device_types="cuda",
    schema=(
        "(Tensor source, Tensor weight, Tensor dout, Tensor dresidual_out, "
        "Tensor rstd, Tensor(a5!) dx, Tensor(a6!) dresidual, "
        "Tensor(a7!) dweight, Tensor(a8!) dbias, float weight_offset, "
        "bool has_weight, bool has_bias, bool compute_dweight, "
        "bool compute_dbias, bool compute_input_grad, bool store_dx, "
        "bool store_dresidual, bool has_residual, bool has_dresidual_out, "
        "bool per_head, int num_heads) -> ()"
    ),
)
def _rmsnorm_flydsl_bwd_op(
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
    _launch_rmsnorm_bwd(
        source,
        weight,
        dout,
        dresidual_out,
        rstd,
        dx,
        dresidual,
        dweight,
        dbias,
        weight_offset,
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
    )


_rmsnorm_flydsl_bwd_op.register_fake(_noop_fake)


@torch.library.custom_op(
    "quack::_rmsnorm_flydsl_bwd_autotuned",
    mutates_args=("dx", "dresidual", "dweight", "dbias"),
    device_types="cuda",
    schema=(
        "(Tensor source, Tensor weight, Tensor dout, Tensor dresidual_out, "
        "Tensor rstd, Tensor(a5!) dx, Tensor(a6!) dresidual, "
        "Tensor(a7!) dweight, Tensor(a8!) dbias, float weight_offset, "
        "bool has_weight, bool has_bias, bool compute_dweight, "
        "bool compute_dbias, bool compute_input_grad, bool store_dx, "
        "bool store_dresidual, bool has_residual, bool has_dresidual_out, "
        "bool per_head, int num_heads) -> ()"
    ),
)
def _rmsnorm_flydsl_bwd_autotuned_op(
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
    _launch_rmsnorm_bwd_autotuned(
        source,
        weight,
        dout,
        dresidual_out,
        rstd,
        dx,
        dresidual,
        dweight,
        dbias,
        weight_offset,
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
    )


_rmsnorm_flydsl_bwd_autotuned_op.register_fake(_noop_fake)


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        residual: torch.Tensor,
        eps: float,
        out_dtype: torch.dtype,
        residual_out_dtype: torch.dtype,
        prenorm: bool,
        weight_offset: float,
        has_weight: bool,
        has_bias: bool,
        has_residual: bool,
        per_head: bool,
        num_heads: int,
        autotuned: bool,
    ):
        programs = x.shape[0] * num_heads
        needs_grad = any(ctx.needs_input_grad[:4])
        # residual_out is only ever read by the caller under prenorm, or by
        # backward as the saved source of a fused residual. Materializing it
        # otherwise costs a full extra tensor write for nothing.
        store_residual = prenorm or (needs_grad and has_residual)
        out = torch.empty_like(x, dtype=out_dtype)
        residual_out = (
            torch.empty_like(x, dtype=residual_out_dtype)
            if store_residual
            else torch.empty(0, device=x.device, dtype=residual_out_dtype)
        )
        rstd = torch.empty(
            programs if needs_grad else 0,
            device=x.device,
            dtype=torch.float32,
        )
        _dispatch(
            _rmsnorm_flydsl_fwd_autotuned_op if autotuned else _rmsnorm_flydsl_fwd_op,
            _launch_rmsnorm_fwd_autotuned if autotuned else _launch_rmsnorm_fwd,
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
            store_rstd=needs_grad,
            per_head=per_head,
            num_heads=num_heads,
        )
        if needs_grad:
            source = residual_out if has_residual else x
            ctx.save_for_backward(source, weight, bias, rstd)
            ctx.x_dtype = x.dtype
            ctx.residual_dtype = residual.dtype
            ctx.has_weight = has_weight
            ctx.has_bias = has_bias
            ctx.has_residual = has_residual
            ctx.per_head = per_head
            ctx.num_heads = num_heads
            ctx.prenorm = prenorm
            ctx.autotuned = autotuned
            ctx.weight_offset = weight_offset
            ctx.x_needs_grad = ctx.needs_input_grad[0]
            ctx.weight_needs_grad = has_weight and ctx.needs_input_grad[1]
            ctx.bias_needs_grad = has_bias and ctx.needs_input_grad[2]
            ctx.residual_needs_grad = has_residual and ctx.needs_input_grad[3]
        if prenorm:
            return out, residual_out
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args):
        source, weight, bias, rstd = ctx.saved_tensors
        dout = _packed_rows(dout)
        has_dresidual_out = ctx.prenorm
        if has_dresidual_out:
            dresidual_out = _packed_rows(args[0])
        else:
            dresidual_out = source

        store_dx = ctx.x_needs_grad
        store_dresidual = ctx.residual_needs_grad
        compute_input_grad = store_dx or store_dresidual
        dx = (
            torch.empty_like(source, dtype=ctx.x_dtype)
            if store_dx
            else torch.empty(0, device=source.device, dtype=ctx.x_dtype)
        )
        dresidual = (
            torch.empty_like(source, dtype=ctx.residual_dtype)
            if store_dresidual
            else torch.empty(0, device=source.device, dtype=ctx.residual_dtype)
        )
        parameter_shape = (ctx.num_heads, source.shape[-1]) if ctx.per_head else (source.shape[-1],)
        # The parameter reduce kernel writes every element it is asked for in
        # the parameter's own dtype, so these need neither zeroing nor a cast
        # on the way out; a torch.zeros here was a memset launch per call, and
        # one for the unused placeholder as well.
        dweight = torch.empty(
            parameter_shape if ctx.weight_needs_grad else (1,),
            device=source.device,
            dtype=weight.dtype,
        )
        dbias = torch.empty(
            parameter_shape if ctx.bias_needs_grad else (1,),
            device=source.device,
            dtype=bias.dtype if ctx.bias_needs_grad else weight.dtype,
        )
        _dispatch(
            _rmsnorm_flydsl_bwd_autotuned_op if ctx.autotuned else _rmsnorm_flydsl_bwd_op,
            _launch_rmsnorm_bwd_autotuned if ctx.autotuned else _launch_rmsnorm_bwd,
            source,
            weight,
            dout,
            dresidual_out,
            rstd,
            dx,
            dresidual,
            dweight,
            dbias,
            ctx.weight_offset,
            has_weight=ctx.has_weight,
            has_bias=ctx.has_bias,
            compute_dweight=ctx.weight_needs_grad,
            compute_dbias=ctx.bias_needs_grad,
            compute_input_grad=compute_input_grad,
            store_dx=store_dx,
            store_dresidual=store_dresidual,
            has_residual=ctx.has_residual,
            has_dresidual_out=has_dresidual_out,
            per_head=ctx.per_head,
            num_heads=ctx.num_heads,
        )

        return (
            dx if ctx.x_needs_grad else None,
            dweight if ctx.weight_needs_grad else None,
            dbias if ctx.bias_needs_grad else None,
            dresidual if ctx.residual_needs_grad else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _rmsnorm_impl(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    eps: float = EPS,
    prenorm: bool = False,
    weight_offset: float = 0.0,
    *,
    autotuned: bool,
) -> torch.Tensor:
    """Apply RMSNorm over the last dimension using the FlyDSL backend."""
    m, n, num_heads, per_head, eps, weight_offset = _validated_inputs(
        x,
        weight,
        bias,
        residual,
        out_dtype,
        residual_dtype,
        eps,
        prenorm,
        weight_offset,
    )
    output_dtype = x.dtype if out_dtype is None else out_dtype
    residual_out_dtype = (
        residual_dtype
        if residual_dtype is not None
        else (residual.dtype if residual is not None else x.dtype)
    )
    if m == 0:
        added = x.float()
        if residual is not None:
            added = added + residual.float()
        normalized = added * torch.rsqrt(added.square().mean(dim=-1, keepdim=True) + eps)
        if weight is not None:
            normalized = normalized * (weight.float() + weight_offset)
        if bias is not None:
            normalized = normalized + bias.float()
        out = normalized.to(output_dtype)
        if prenorm:
            return out, added.to(residual_out_dtype)
        return out

    plain_eager_inference = (
        not torch.compiler.is_compiling()
        and x.ndim == 2
        and x.stride() == (n, 1)
        and weight is not None
        and weight.ndim == 1
        and weight.stride() == (1,)
        and bias is None
        and residual is None
        and out_dtype is None
        and residual_dtype is None
        and not prenorm
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
    )
    if plain_eager_inference:
        absent = _eager_empty(x.device, x.dtype)
        residual_out = _eager_empty(x.device, x.dtype)
        rstd = _eager_empty(x.device, torch.float32)
        out = torch.empty_like(x)
        resolved_public = _FWD_PUBLIC_RESOLVED_LAST[0]
        if (
            autotuned
            and not _env_flag_enabled("FLYDSL_AUTOTUNE")
            and resolved_public is not None
            and resolved_public[0] is _VALIDATED_INPUT_LAST[0]
            and resolved_public[1] == _public_runtime_guard()
        ):
            with torch.cuda.device(x.device):
                if _is_generic_forward_config(resolved_public[2][0]):
                    _launch_rmsnorm_fwd(
                        x,
                        weight,
                        absent,
                        x,
                        out,
                        residual_out,
                        rstd,
                        eps,
                        weight_offset,
                        has_weight=True,
                        has_bias=False,
                        has_residual=False,
                        store_residual=False,
                        store_rstd=False,
                        per_head=False,
                        num_heads=1,
                    )
                else:
                    _launch_resolved_fwd_entry(
                        resolved_public[2],
                        x,
                        weight,
                        absent,
                        x,
                        out,
                        residual_out,
                        rstd,
                        m,
                        eps,
                        weight_offset,
                        has_weight=True,
                        has_bias=False,
                        has_residual=False,
                        store_residual=False,
                        store_rstd=False,
                        per_head=False,
                        num_heads=1,
                    )
            return out
        use_generic_winner = (
            autotuned
            and not _env_flag_enabled("FLYDSL_AUTOTUNE")
            and _FWD_PUBLIC_GENERIC_LAST[0] is _VALIDATED_INPUT_LAST[0]
        )
        launch = (
            _launch_rmsnorm_fwd
            if not autotuned or use_generic_winner
            else _launch_rmsnorm_fwd_autotuned
        )
        launch(
            x,
            weight,
            absent,
            x,
            out,
            residual_out,
            rstd,
            eps,
            weight_offset,
            has_weight=True,
            has_bias=False,
            has_residual=False,
            store_residual=False,
            store_rstd=False,
            per_head=False,
            num_heads=1,
        )
        return out

    last_shape = (num_heads, n) if per_head else (n,)
    x_flat = _packed_rows(x.reshape(-1, *last_shape))

    needs_grad = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in (x, weight, bias, residual)
    )
    eager_no_grad = not torch.compiler.is_compiling() and not needs_grad
    # An absent weight or bias still has to be passed, because the custom op
    # schema is fixed. The kernels build no descriptor for it, so eager
    # inference can safely share an internal zero-sized tensor. Keep the
    # custom-op/autograd path's placeholder call-local.
    absent = (
        _eager_empty(x.device, x.dtype)
        if eager_no_grad
        else torch.empty(0, device=x.device, dtype=x.dtype)
    )
    weight_arg = _packed_rows(weight) if weight is not None else absent
    bias_arg = _packed_rows(bias) if bias is not None else absent
    residual_arg = (
        _packed_rows(residual.reshape(-1, *last_shape)) if residual is not None else x_flat
    )
    if eager_no_grad:
        store_residual = prenorm
        out = torch.empty_like(x_flat, dtype=output_dtype)
        residual_out = (
            torch.empty_like(x_flat, dtype=residual_out_dtype)
            if store_residual
            else _eager_empty(x.device, residual_out_dtype)
        )
        rstd = _eager_empty(x.device, torch.float32)
        use_generic_winner = (
            autotuned
            and not _env_flag_enabled("FLYDSL_AUTOTUNE")
            and _FWD_PUBLIC_GENERIC_LAST[0] is _VALIDATED_INPUT_LAST[0]
        )
        launch = (
            _launch_rmsnorm_fwd
            if not autotuned or use_generic_winner
            else _launch_rmsnorm_fwd_autotuned
        )
        launch(
            x_flat,
            weight_arg,
            bias_arg,
            residual_arg,
            out,
            residual_out,
            rstd,
            eps,
            weight_offset,
            has_weight=weight is not None,
            has_bias=bias is not None,
            has_residual=residual is not None,
            store_residual=store_residual,
            store_rstd=False,
            per_head=per_head,
            num_heads=num_heads,
        )
        out = out.reshape(x.shape)
        if prenorm:
            return out, residual_out.reshape(x.shape)
        return out

    result = _RMSNormFunction.apply(
        x_flat,
        weight_arg,
        bias_arg,
        residual_arg,
        eps,
        output_dtype,
        residual_out_dtype,
        prenorm,
        weight_offset,
        weight is not None,
        bias is not None,
        residual is not None,
        per_head,
        num_heads,
        autotuned,
    )
    if isinstance(result, tuple):
        return tuple(tensor.reshape(x.shape) for tensor in result)
    return result.reshape(x.shape)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    eps: float = EPS,
    prenorm: bool = False,
    weight_offset: float = 0.0,
) -> torch.Tensor:
    """Apply RMSNorm with the stable analytical forward heuristic."""
    return _rmsnorm_impl(
        x,
        weight,
        bias,
        residual,
        out_dtype,
        residual_dtype,
        eps,
        prenorm,
        weight_offset,
        autotuned=False,
    )


def rmsnorm_autotuned(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    eps: float = EPS,
    prenorm: bool = False,
    weight_offset: float = 0.0,
) -> torch.Tensor:
    """Apply RMSNorm through FlyDSL's native forward autotuner."""
    return _rmsnorm_impl(
        x,
        weight,
        bias,
        residual,
        out_dtype,
        residual_dtype,
        eps,
        prenorm,
        weight_offset,
        autotuned=True,
    )

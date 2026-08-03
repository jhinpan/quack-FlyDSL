# Copyright (c) 2026, Tri Dao.

"""Explicit ROCm/FlyDSL RMSNorm backend.

This module is intentionally opt-in. Importing ``quack`` does not import
FlyDSL, and this backend does not alter Quack's existing CUDA/CuTe dispatch.
"""

import math
import numbers

import torch

from quack.flydsl.rmsnorm_bwd_kernel import (
    TWO_STAGE_MAX_NUM_THREADS,
    build_rmsnorm_bwd_two_stage_module,
    rmsnorm_bwd_two_stage_config,
)
from quack.flydsl.rmsnorm_autotune import (
    RMSNORM_AUTOTUNE_SCHEMA_VERSION,
    _rmsnorm_fwd_tuner,
)
from quack.flydsl.rmsnorm_common import EPS, FLYDSL_BUILD_LOCK, run_compiled
from quack.flydsl.rmsnorm_config import MAX_N, N_ALIGNMENT, next_power_of_two
from quack.flydsl.rmsnorm_kernel import build_rmsnorm_module

__all__ = ["rmsnorm", "rmsnorm_autotuned"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# gfx950 is the only architecture this backend has been built and run on.
# gfx942 is wave64 and should work, but until it executes on real hardware it
# is not claimed here.
_SUPPORTED_ARCHES = frozenset({"gfx950"})
# Row counts cross the Int32 kernel ABI here; reject before FlyDSL's argument
# packing raises a struct.error from inside the dispatch.
_MAX_ROWS = 2**31 - 1
_FWD_CACHE: dict[tuple, object] = {}
_BWD_CACHE: dict[tuple, object] = {}
_BWD_CU_COUNT_CACHE: dict[torch.device, int] = {}
_DEVICE_ARCH_CACHE: dict[int, str] = {}


def _dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "f32"
    raise TypeError(f"unsupported dtype: {dtype}")


def _normalize_arch(arch: str) -> str:
    arch = arch.split(":", 1)[0]
    if arch.startswith("gfx"):
        return arch
    parts = arch.split(".")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return f"gfx{parts[0]}{parts[1]}{parts[2]}"
    return arch


def _flydsl_compile_target() -> tuple[str, str]:
    """Ask FlyDSL what it will actually generate code for."""
    from flydsl.compiler.backends import get_backend

    target = get_backend().target
    return target.backend, _normalize_arch(target.arch)


def _validate_arch(device: torch.device) -> str:
    """Resolve and validate the architecture behind ``device``.

    The device query is memoized because a device index cannot change identity
    within a process. FlyDSL's compile target is *not* memoized: it is driven
    by the environment and can change under a long-lived process, so every
    build rechecks it. Both only run on the build path, never on a launch.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    actual = _DEVICE_ARCH_CACHE.get(index)
    if actual is None:
        actual = _normalize_arch(torch.cuda.get_device_properties(index).gcnArchName)
        if actual not in _SUPPORTED_ARCHES:
            raise ValueError(
                f"FlyDSL RMSNorm supports {', '.join(sorted(_SUPPORTED_ARCHES))}; "
                f"cuda:{index} is {actual}"
            )
        _DEVICE_ARCH_CACHE[index] = actual

    backend, compile_arch = _flydsl_compile_target()
    if backend != "rocm":
        raise RuntimeError(
            f"FlyDSL RMSNorm requires FlyDSL's ROCm backend, but it is targeting {backend!r}"
        )
    if compile_arch != actual:
        raise ValueError(
            "FlyDSL RMSNorm does not support mixed architectures: "
            f"cuda:{index} is {actual}, but FlyDSL compiles for {compile_arch}. "
            "Set ARCH and FLYDSL_GPU_ARCH to the device architecture."
        )
    return actual


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    residual_dtype: torch.dtype | None,
    eps: float,
    prenorm: bool,
    weight_offset: float,
) -> tuple[int, int, int, bool, float, float]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    for name, tensor in (("weight", weight), ("bias", bias), ("residual", residual)):
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor).__name__}")

    parameter_ranks = {tensor.ndim for tensor in (weight, bias) if tensor is not None}
    if not parameter_ranks.issubset({1, 2}):
        raise ValueError("weight and bias must be 1-D or 2-D")
    if len(parameter_ranks) > 1:
        raise ValueError("weight and bias must use the same rank")
    per_head = parameter_ranks == {2}
    if per_head:
        if x.ndim < 2:
            raise ValueError("per-head RMSNorm requires an input with at least two dimensions")
        num_heads, n = x.shape[-2:]
        parameter_shape = (num_heads, n)
    else:
        num_heads, n = 1, x.shape[-1]
        parameter_shape = (n,)

    if not 1 <= n <= MAX_N:
        raise ValueError(f"x normalized dimension must be between 1 and {MAX_N}, got {n}")
    if n % N_ALIGNMENT:
        # Every row start is then naturally aligned and every operand reaches
        # full vector width, which is what lets the kernels treat the vector
        # size as a property of the dtype. Refusing is deliberate: serving such
        # a row with a narrower access is a path nothing measures and nothing
        # in practice reaches, so it would only distort what gets optimized.
        raise ValueError(
            f"x normalized dimension must be a multiple of {N_ALIGNMENT}, got {n}; "
            "use quack.rmsnorm for row lengths this backend does not accept"
        )
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is not None and tuple(tensor.shape) != parameter_shape:
            raise ValueError(f"{name} shape must be {parameter_shape}, got {tuple(tensor.shape)}")
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape must match x, got {tuple(residual.shape)}/{tuple(x.shape)}"
        )

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be float16, bfloat16, or float32, got {x.dtype}")
    for name, tensor in (("weight", weight), ("bias", bias), ("residual", residual)):
        if tensor is not None and tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"{name} dtype must be float16, bfloat16, or float32, got {tensor.dtype}"
            )
    for name, dtype in (("out_dtype", out_dtype), ("residual_dtype", residual_dtype)):
        if dtype is not None and dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must be float16, bfloat16, or float32, got {dtype}")

    if torch.version.hip is None or x.device.type != "cuda":
        raise ValueError(f"x must be on a ROCm device, got {x.device}")
    for name, tensor in (("weight", weight), ("bias", bias), ("residual", residual)):
        if tensor is not None and tensor.device != x.device:
            raise ValueError(
                f"x and {name} must be on the same device, got {x.device}/{tensor.device}"
            )

    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError(f"eps must be a real number, got {type(eps).__name__}")
    eps = float(eps)
    if not 0.0 < eps < math.inf:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if not isinstance(prenorm, bool):
        raise TypeError(f"prenorm must be a bool, got {type(prenorm).__name__}")
    if isinstance(weight_offset, bool) or not isinstance(weight_offset, numbers.Real):
        raise TypeError(f"weight_offset must be a real number, got {type(weight_offset).__name__}")
    weight_offset = float(weight_offset)
    if not -math.inf < weight_offset < math.inf:
        raise ValueError(f"weight_offset must be finite, got {weight_offset}")
    if weight is None and weight_offset != 0.0:
        raise ValueError("weight_offset requires an explicit weight")

    m = x.numel() // (num_heads * n)
    if m > _MAX_ROWS:
        raise ValueError(f"x has {m} rows, but the kernels address at most {_MAX_ROWS}")
    return m, n, num_heads, per_head, eps, weight_offset


def _current_raw_stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


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


def _dispatch(custom_op, eager_launch, *args, **kwargs) -> None:
    """Use the opaque custom op only while torch is tracing."""
    target = custom_op if torch.compiler.is_compiling() else eager_launch
    target(*args, **kwargs)


def _noop_fake(*args, **kwargs) -> None:
    """Mutation-only custom ops have no fake-tensor work to perform."""


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
        )
        launcher = _FWD_CACHE.get(key)
        if launcher is None:
            launcher = _build_cached(
                _FWD_CACHE,
                key,
                x.device,
                lambda arch: build_rmsnorm_module(
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
                ),
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
    """Launch through FlyDSL's native tuner without consulting ``_FWD_CACHE``."""
    m, n = x.shape[0], x.shape[-1]
    arch = _validate_arch(x.device)
    with torch.cuda.device(x.device):
        _rmsnorm_fwd_tuner(
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
            n=n,
            input_dtype_str=_dtype_to_str(x.dtype),
            output_dtype_str=_dtype_to_str(out.dtype),
            weight_dtype_str=_dtype_to_str(weight.dtype),
            bias_dtype_str=_dtype_to_str(bias.dtype),
            residual_dtype_str=_dtype_to_str(residual.dtype),
            residual_out_dtype_str=_dtype_to_str(residual_out.dtype),
            has_weight=has_weight,
            has_bias=has_bias,
            has_residual=has_residual,
            store_residual=store_residual,
            store_rstd=store_rstd,
            per_head=per_head,
            num_heads=num_heads,
            arch=arch,
            schema_version=RMSNORM_AUTOTUNE_SCHEMA_VERSION,
            stream=_current_raw_stream(x.device),
        )


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
            _rmsnorm_flydsl_bwd_op,
            _launch_rmsnorm_bwd,
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


def _packed_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Copy only when the row is not already contiguous along the last axis.

    Every operand reaches the kernels as a row-scoped buffer descriptor built
    by ``fx.slice(tensor, (row, None))``, so what the kernels require is that
    each row be contiguous -- not that the whole tensor be. A row-padded view
    (``full[:, :n]``, stride ``(pitch, 1)``) already satisfies that, and the
    descriptor is sized to ``n`` elements, so the padding is never addressed.
    Verified bit-identical to the packed result at pitches ``n+{0,1,2,3,4,8,
    256,512}`` for 2-D and per-head inputs.

    ``.contiguous()`` unconditionally was costing a full extra read+write of
    the activation on those views: roughly half the call was repacking. The
    packed baseline at 32768x4096 bf16 is 89.3-89.9 us, which against the
    ``two_read_one_write`` probe on the same card (6.139 TB/s) is 97.9% of
    achievable. The before/after pair that used to be quoted here (183.2 us /
    93.3 us) is withdrawn -- 183.2 does not reproduce and could not be
    reconstructed -- so the size of the saving is deliberately not restated
    until it is re-measured.

    The residual cost on a padded view is *not* claimed to be zero, and the
    explanation once given for it ("more DRAM pages per row") is withdrawn as
    well: it was never isolated, and a plausible mechanism is not a measured
    one.

    A note on how these were measured, because the first version of this
    docstring quoted numbers that were wrong. Timing ran on ``cuda:6`` while
    the process default device was 0, so the ``torch.cuda.Event`` pair bound to
    device 0 at its first ``record()`` and bracketed nothing: it read 26.9 us
    for a call that actually takes 89.9 us. That implied a bandwidth above the
    roofline, which is the measurement announcing its own invalidity, and the
    number was quoted for a whole commit before the check caught it.

    A unit last stride is necessary but **not sufficient**, and the first
    version of this helper tested only that. ``storage.unfold(0, n, 1)`` gives
    stride ``(1, 1)``: every row is contiguous, ``stride(-1) == 1`` holds, and
    consecutive rows *overlap in storage*. FlyDSL builds the ABI from the first
    unit-stride axis, so such a tensor is not merely computed wrong -- the
    launcher built for it is cached under ``_FWD_CACHE``'s key, which carries
    no layout term, and the **next ordinary contiguous call reuses it and is
    silently wrong too**, until the cache is cleared. Measured at 64x256 bf16:
    the overlapping call reads max abs error 11.75, and the plain call right
    after it reads 11.79 where it should read 0. @Reviewer found this.

    So the predicate also requires ``stride(0) >= size(-1)`` -- rows that do not
    tread on each other. That admits exactly the row-padded views this helper
    exists for (``(pitch, 1)`` with ``pitch >= n``) and rejects the overlapping
    ones, which is the distinction the buffer descriptor actually needs. For
    dimensions above 2 the same must hold at every level, so the general test is
    that each stride be at least the extent of everything inside it.

    ``quack.rmsnorm._ensure_contiguous`` admits the overlapping view too, so
    upstream shares the hole -- but upstream never claims a row-only contract:
    it copies every non-``stride(-1)==1`` input and its correctness does not
    rest on the weaker property. This helper's does, so the obligation to close
    it is here. The predicate remains a strict subset of upstream's: everything
    upstream copies, this copies. Under ``torch.compile`` the copy stays
    unconditional because dynamo cannot inspect strides on fake tensors; that is
    upstream's reasoning and it applies here unchanged.

    One layout defeats ``.contiguous()`` entirely, and @Reviewer found it after
    the disjointness fix above. ``torch.randn((64, 1)).t()`` has shape ``(1,
    64)`` and stride ``(1, 1)``, and torch reports ``is_contiguous() == True``
    -- correctly, since with one row there is nothing to be discontiguous with.
    So ``.contiguous()`` returns *the same tensor*, and a guard that copies
    cannot fix it. The problem is not the data: it is that
    ``mark_layout_dynamic`` picks the **first** unit-stride axis, which for
    stride ``(1, 1)`` is axis 0 rather than the row axis, so the launcher is
    built against the wrong leading dimension. Since ``_FWD_CACHE``'s key holds
    no layout term, that launcher then serves the next ordinary ``(4, 64)``
    call, which reads 6.57 where it should read ~0.016 -- the same poisoning as
    the unfold case, reached by a different door. It reproduces under
    ``torch.compile``; in eager the singleton happens to survive, but the
    ambiguity is identical and is not worth relying on.

    ``quack/blockscaled/utils.py`` already names this exact trap -- "size-1
    dims carry an arbitrary (often 1) stride ... and must not shadow the real
    contiguous dim" -- and solves it by passing an explicit ``leading_dim``.
    That hook is not available here, because FlyDSL builds the descriptor
    itself from the tensor. So the fix is to hand it a tensor whose first
    unit-stride axis *is* the row axis, which is what ``_unambiguous_layout``
    does.

    One correction to how I described this to the team: I said the two layouts
    were indistinguishable to FlyDSL, so a layout term in the cache key could
    not have helped either. That is false, and @Autotune measured it::

        MemRefSpec(16, [1, 64], [1, 1]).mark_layout_dynamic()
            -> get_cache_signature() == (2, False, (-1, -1), (1, -1))
        MemRefSpec(16, [4, 64], [64, 1]).mark_layout_dynamic()
            -> get_cache_signature() == (2, False, (-1, -1), (-1, 1))

    The signatures differ, so an ABI-signature term in the key *would* separate
    these two entries. Canonicalizing first is still the right primary fix --
    it makes the reuse correct rather than merely rarer, and it keeps one
    launcher per shape -- but "a key term could not have worked" was an
    argument for it that does not hold, and it should not be repeated.
    """
    tensor = _unambiguous_layout(tensor)
    if torch.compiler.is_compiling():
        return tensor.contiguous()
    return tensor if _rows_are_disjoint_and_packed(tensor) else tensor.contiguous()


def _unambiguous_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the first unit-stride axis is the row axis.

    FlyDSL selects the leading dimension as ``next(i for i in range(ndim) if
    stride[i] == 1)``. A size-1 axis before the row carrying stride 1 wins that
    search and silently redefines the ABI. Copying does not help: such a tensor
    is already ``is_contiguous()``, so ``.contiguous()`` is the identity.

    Relabelling the offending axes is enough and costs nothing. For a size-1
    axis the stride is unobservable -- there is no second element to step to --
    so any value describes the same memory, and any non-unit value moves it out
    of the way of the search.

    Dropping the axis and putting it back is what does the relabelling.
    ``unsqueeze`` derives the reinserted stride as ``size * stride`` of the axis
    within. What that guarantees is only that the result is **non-unit** --
    not that it is above 1. It can be 0: ``(1,1,8)`` stride ``(1,0,1)``
    relabels to ``(0,0,1)``, values preserved, and the predicate below then
    rejects the zero stride and copies. Non-unit is all this helper needs,
    because the offending axis no longer carries the row's unit stride and the
    descriptor search cannot mistake it for the row. An earlier version of this
    paragraph said "above 1"; @Reviewer supplied the broadcast counterexample.

    It is *not* in general "at least the extent of everything inside it" --
    an earlier version claimed that too, and @Reviewer produced the
    counterexample. ``(1,2,2,8)`` stride ``(1,8,100,1)`` relabels to stride
    ``(16,8,100,1)``, and 16 is well under the inner occupied span of 116 --
    max reachable offset ``8*1 + 100*1 + 1*7 = 115``, plus one. (I first wrote
    108, having dropped the ``8*(2-1)`` term; the inequality holds either way,
    but the number in a worked counterexample should be the right one.) That
    layout has no alias and is row-packed, so ``_packed_rows`` keeps it and the
    result is correct -- the stronger nested-span property is neither true nor
    required here. Likewise ``(1,2,8)`` stride ``(1,1,2)`` relabels to
    ``(2,1,2)``, which still has a non-row unit-stride axis; the predicate
    below then rejects it and copies. The contract is the pair: this helper
    removes the ambiguous unit stride on the offending singleton, and
    ``_rows_are_disjoint_and_packed`` plus the copy close everything else. This
    helper alone guarantees nothing about arbitrary exotic layouts.

    An earlier version called ``as_strided(..., tensor.storage_offset())``
    instead. That is correct eagerly and breaks under
    ``fullgraph=True``: ``storage_offset()`` returns a
    Python scalar, Dynamo cannot keep a non-Tensor from a ``torch.*`` op, and
    both static and dynamic compiles raised ``Unsupported`` on exactly the
    singleton this function exists to handle. @Reviewer caught it -- the test I
    had written omitted ``fullgraph=True`` and graph-broke around the helper,
    which hid it. ``squeeze``/``unsqueeze`` carry the offset implicitly, so a
    view into the middle of a storage survives with its offset intact.
    """
    row = tensor.dim() - 1
    offenders = [
        axis for axis in range(row) if tensor.shape[axis] == 1 and tensor.stride(axis) == 1
    ]
    if not offenders:
        return tensor
    for axis in reversed(offenders):
        tensor = tensor.squeeze(axis)
    for axis in offenders:
        tensor = tensor.unsqueeze(axis)
    return tensor


def _rows_are_disjoint_and_packed(tensor: torch.Tensor) -> bool:
    """Whether the row is contiguous *and* no two elements share storage.

    Two independent requirements, and an earlier draft of this dropped the
    first. ``stride(-1) == 1`` is upstream's test and it stays, unweakened: the
    descriptor reads a row as consecutive elements, and it also keeps this
    predicate a strict subset of ``_ensure_contiguous``'s. Sorting the axes by
    stride and checking only disjointness would accept a transposed view --
    every element distinct, but the row scattered -- which upstream copies and
    which produces a wrong answer here.

    Given that, walk the remaining axes outwards from the row: each stride must
    be at least the extent spanned by everything inside it. Equality is a packed
    axis, a strict inequality is padding between rows, and both are fine because
    the descriptor is sized to the row and never addresses the gap. A stride
    *smaller* than the enclosed extent means two positions alias -- overlapping
    rows, or a zero-stride broadcast -- which is what this rejects.

    The span of an axis is ``stride * (size - 1) + inner``, not ``stride *
    size``: the last block along the axis occupies only ``inner`` elements, and
    any padding after it is beyond the tensor. An earlier version used the
    latter and so counted trailing padding as occupied, rejecting layouts that
    do not alias -- @Reviewer's example is shape ``(2, 2, 64)`` stride ``(144,
    80, 1)``, whose true span is ``80 * 1 + 64 = 144``, exactly the outer
    stride, but which the old arithmetic scored as 160 and copied. That was
    conservative rather than wrong, and it cost a needless copy on padded
    per-head inputs.

    Even corrected, this remains **sufficient rather than equivalent** to "no
    two elements share storage". It tests a nested-containment property, which
    every layout the kernels can address satisfies, but a sufficiently exotic
    stride set could be alias-free and still be rejected. Rejection costs a
    copy, never correctness, so the asymmetry is the safe way round -- but the
    predicate should not be described as deciding aliasing in general.
    """
    if tensor.stride(-1) != 1:
        return False
    span = tensor.shape[-1]
    for stride, size in sorted(
        zip(tensor.stride()[:-1], tensor.shape[:-1]), key=lambda axis: axis[0]
    ):
        if stride < span:
            return False
        span = stride * (size - 1) + span
    return True


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
    m, n, num_heads, per_head, eps, weight_offset = _validate_inputs(
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

    last_shape = (num_heads, n) if per_head else (n,)
    parameter_shape = (num_heads, n) if per_head else (n,)
    x_flat = _packed_rows(x.reshape(-1, *last_shape))

    # An absent weight or bias still has to be passed, because the custom op
    # schema is fixed. The kernels build no descriptor for it, so an empty
    # tensor is enough and keeps the allocation off every call.
    absent = torch.empty(0, device=x.device, dtype=x.dtype)
    weight_arg = _packed_rows(weight) if weight is not None else absent
    bias_arg = _packed_rows(bias) if bias is not None else absent
    residual_arg = (
        _packed_rows(residual.reshape(-1, *last_shape)) if residual is not None else x_flat
    )
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

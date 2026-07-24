# Copyright (c) 2026, Tri Dao.

"""Explicit ROCm/FlyDSL plain RMSNorm backend.

This module is intentionally opt-in. Importing ``quack`` does not import
FlyDSL, and this backend does not alter Quack's existing CUDA/CuTe dispatch.
"""

import math
import numbers
import os

import torch

from quack._flydsl.kernel_utils import run_compiled
from quack._flydsl.rmsnorm_bwd_kernel import (
    build_rmsnorm_bwd_module,
    build_rmsnorm_bwd_two_stage_module,
    is_rmsnorm_bwd_two_stage_vec_config,
)
from quack._flydsl.rmsnorm_common import EPS
from quack._flydsl.rmsnorm_kernel import build_rmsnorm_module


__all__ = ["rmsnorm"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FWD_CACHE: dict[tuple, object] = {}
_BWD_CACHE: dict[tuple, object] = {}
_BWD_CU_COUNT_CACHE: dict[torch.device, int] = {}
_BWD_TWO_STAGE_MIN_ROWS = 512
_BWD_TWO_STAGE_MAX_N = 8192


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


def _validate_arch(device: torch.device) -> str:
    properties = torch.cuda.get_device_properties(device)
    actual = _normalize_arch(properties.gcnArchName)
    flydsl_override = os.environ.get("FLYDSL_GPU_ARCH")
    arch_override = os.environ.get("ARCH")
    if flydsl_override is not None:
        requested = _normalize_arch(flydsl_override)
        if arch_override is not None and _normalize_arch(arch_override) != requested:
            raise ValueError("ARCH and FLYDSL_GPU_ARCH must select the same architecture")
        if requested != actual:
            raise ValueError(
                "FlyDSL RMSNorm does not support mixed architectures: "
                f"device is {actual}, FLYDSL_GPU_ARCH selects {requested}"
            )
    return actual


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[int, int, float]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D, got shape {tuple(weight.shape)}")

    n = x.shape[-1]
    if weight.shape[0] != n:
        raise ValueError(f"x last dimension ({n}) must equal weight length ({weight.shape[0]})")
    if not 1 <= n <= 8192:
        raise ValueError(f"x last dimension must be between 1 and 8192, got {n}")

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be float16, bfloat16, or float32, got {x.dtype}")
    valid_weight_dtype = weight.dtype == x.dtype or (
        x.dtype in (torch.float16, torch.bfloat16) and weight.dtype == torch.float32
    )
    if not valid_weight_dtype:
        raise TypeError(
            "weight dtype must match x dtype, except float16/bfloat16 x may use "
            f"float32 weight; got {x.dtype}/{weight.dtype}"
        )

    if torch.version.hip is None or x.device.type != "cuda":
        raise ValueError(f"x must be on a ROCm device, got {x.device}")
    if weight.device != x.device:
        raise ValueError(f"x and weight must be on the same device, got {x.device}/{weight.device}")

    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError(f"eps must be a real number, got {type(eps).__name__}")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")

    m = x.numel() // n
    return m, n, eps


def _current_raw_stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _select_rmsnorm_bwd_config(
    m: int,
    n: int,
    dtype_str: str,
    device: torch.device,
) -> tuple[str, int | None]:
    if m >= _BWD_TWO_STAGE_MIN_ROWS and n <= _BWD_TWO_STAGE_MAX_N:
        num_cus = _BWD_CU_COUNT_CACHE.get(device)
        if num_cus is None:
            num_cus = torch.cuda.get_device_properties(device).multi_processor_count
            if not torch.compiler.is_compiling():
                _BWD_CU_COUNT_CACHE[device] = num_cus
        if is_rmsnorm_bwd_two_stage_vec_config(n, dtype_str):
            num_programs = num_cus if m < 2048 else (3 * num_cus) // 2
        else:
            num_programs = num_cus if m < 1024 else 2 * num_cus
        return "two_stage", min(m, num_programs)
    return "atomic", None


def _launch_rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    store_rstd: bool,
) -> None:
    m, n = x.shape
    dtype_str = _dtype_to_str(x.dtype)
    weight_dtype_str = _dtype_to_str(weight.dtype)

    with torch.cuda.device(x.device):
        arch = _validate_arch(x.device)
        key = (
            x.device.index,
            arch,
            n,
            dtype_str,
            weight_dtype_str,
            store_rstd,
            eps,
        )
        launcher = _FWD_CACHE.get(key)
        if launcher is None:
            launcher = build_rmsnorm_module(
                n,
                dtype_str,
                store_rstd=store_rstd,
                eps=eps,
                weight_dtype_str=weight_dtype_str,
            )
            _FWD_CACHE[key] = launcher
        stream = _current_raw_stream(x.device)
        if store_rstd:
            run_compiled(launcher, x, weight, out, rstd, m, stream)
        else:
            run_compiled(launcher, x, weight, out, m, stream)


@torch.library.custom_op(
    "quack::_rmsnorm_flydsl_fwd",
    mutates_args=("out", "rstd"),
    device_types="cuda",
    schema=(
        "(Tensor x, Tensor weight, Tensor(a2!) out, Tensor(a3!) rstd, "
        "float eps, bool store_rstd) -> ()"
    ),
)
def _rmsnorm_flydsl_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    store_rstd: bool,
) -> None:
    _launch_rmsnorm_fwd(x, weight, out, rstd, eps, store_rstd)


@_rmsnorm_flydsl_fwd_op.register_fake
def _rmsnorm_flydsl_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    store_rstd: bool,
) -> None:
    return None


def _dispatch_rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    store_rstd: bool,
) -> None:
    if torch.compiler.is_compiling():
        _rmsnorm_flydsl_fwd_op(
            x,
            weight,
            out,
            rstd,
            eps,
            store_rstd,
        )
    else:
        _launch_rmsnorm_fwd(
            x,
            weight,
            out,
            rstd,
            eps,
            store_rstd,
        )


def _rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    store_rstd: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    m = x.shape[0]
    out = torch.empty_like(x)
    rstd = torch.empty(
        (m if store_rstd else 0,),
        device=x.device,
        dtype=torch.float32,
    )
    _dispatch_rmsnorm_fwd(
        x,
        weight,
        out,
        rstd,
        eps,
        store_rstd,
    )
    return out, rstd if store_rstd else None


def _launch_rmsnorm_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    partial: torch.Tensor,
    num_programs: int,
) -> None:
    m, n = x.shape
    dtype_str = _dtype_to_str(x.dtype)
    weight_dtype_str = _dtype_to_str(weight.dtype)
    path = "two_stage" if num_programs > 0 else "atomic"

    with torch.cuda.device(x.device):
        arch = _validate_arch(x.device)
        key = (
            path,
            x.device.index,
            arch,
            n,
            dtype_str,
            weight_dtype_str,
            num_programs,
        )
        launcher = _BWD_CACHE.get(key)
        stream = _current_raw_stream(x.device)
        if path == "two_stage":
            if launcher is None:
                launcher = build_rmsnorm_bwd_two_stage_module(
                    n,
                    dtype_str,
                    num_programs,
                    weight_dtype_str=weight_dtype_str,
                )
                _BWD_CACHE[key] = launcher
            run_compiled(
                launcher,
                x,
                weight,
                dout,
                rstd,
                dx,
                dweight,
                partial,
                m,
                stream,
            )
            return

        if launcher is None:
            launcher = build_rmsnorm_bwd_module(
                n,
                dtype_str,
                weight_dtype_str=weight_dtype_str,
            )
            _BWD_CACHE[key] = launcher
        run_compiled(
            launcher,
            x,
            weight,
            dout,
            rstd,
            dx,
            dweight,
            m,
            stream,
        )


@torch.library.custom_op(
    "quack::_rmsnorm_flydsl_bwd",
    mutates_args=("dx", "dweight", "partial"),
    device_types="cuda",
    schema=(
        "(Tensor x, Tensor weight, Tensor dout, Tensor rstd, Tensor(a4!) dx, "
        "Tensor(a5!) dweight, Tensor(a6!) partial, int num_programs) -> ()"
    ),
)
def _rmsnorm_flydsl_bwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    partial: torch.Tensor,
    num_programs: int,
) -> None:
    _launch_rmsnorm_bwd(
        x,
        weight,
        dout,
        rstd,
        dx,
        dweight,
        partial,
        num_programs,
    )


@_rmsnorm_flydsl_bwd_op.register_fake
def _rmsnorm_flydsl_bwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    partial: torch.Tensor,
    num_programs: int,
) -> None:
    return None


def _dispatch_rmsnorm_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    rstd: torch.Tensor,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    partial: torch.Tensor,
    num_programs: int,
) -> None:
    if torch.compiler.is_compiling():
        _rmsnorm_flydsl_bwd_op(
            x,
            weight,
            dout,
            rstd,
            dx,
            dweight,
            partial,
            num_programs,
        )
    else:
        _launch_rmsnorm_bwd(
            x,
            weight,
            dout,
            rstd,
            dx,
            dweight,
            partial,
            num_programs,
        )


def _rmsnorm_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    dtype_str = _dtype_to_str(x.dtype)
    path, selected_programs = _select_rmsnorm_bwd_config(
        m,
        n,
        dtype_str,
        x.device,
    )
    num_programs = selected_programs if path == "two_stage" else 0
    dx = torch.empty_like(x)
    if num_programs:
        dweight = torch.empty_like(weight)
        partial = torch.empty(
            (num_programs * n,),
            device=x.device,
            dtype=torch.float32,
        )
    else:
        dweight = torch.zeros((n,), device=x.device, dtype=torch.float32)
        partial = torch.empty((0,), device=x.device, dtype=torch.float32)

    _dispatch_rmsnorm_bwd(
        x,
        weight,
        dout,
        rstd,
        dx,
        dweight,
        partial,
        num_programs,
    )
    return dx, dweight.to(weight.dtype)


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        needs_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        out, rstd = _rmsnorm_fwd(
            x,
            weight,
            eps,
            store_rstd=needs_grad,
        )
        if needs_grad:
            ctx.save_for_backward(x, weight, rstd)
            ctx.x_needs_grad = ctx.needs_input_grad[0]
            ctx.weight_needs_grad = ctx.needs_input_grad[1]
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        x, weight, rstd = ctx.saved_tensors
        dx, dweight = _rmsnorm_bwd(
            x,
            weight,
            dout.contiguous(),
            rstd,
        )
        return (
            dx if ctx.x_needs_grad else None,
            dweight if ctx.weight_needs_grad else None,
            None,
        )


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Apply plain RMSNorm over the last dimension using the FlyDSL backend."""
    m, n, eps = _validate_inputs(x, weight, eps)
    if m == 0:
        return x * weight.to(x.dtype)

    x_flat = x.reshape(-1, n).contiguous()
    weight_contiguous = weight.contiguous()
    out_flat = _RMSNormFunction.apply(
        x_flat,
        weight_contiguous,
        eps,
    )
    return out_flat.reshape(x.shape)

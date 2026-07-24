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
from quack._flydsl.rmsnorm_common import EPS
from quack._flydsl.rmsnorm_kernel import build_rmsnorm_module


__all__ = ["rmsnorm"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FWD_CACHE: dict[tuple, object] = {}


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


def _rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    store_rstd: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    m, n = x.shape
    out = torch.empty_like(x)
    rstd = torch.empty((m,), device=x.device, dtype=torch.float32) if store_rstd else None
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
    return out, rstd


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Apply plain RMSNorm over the last dimension using the FlyDSL backend."""
    m, n, eps = _validate_inputs(x, weight, eps)
    if m == 0:
        return torch.empty_like(x)

    x_flat = x.reshape(-1, n).contiguous()
    weight_contiguous = weight.contiguous()
    out_flat, _ = _rmsnorm_fwd(
        x_flat,
        weight_contiguous,
        eps,
        store_rstd=False,
    )
    return out_flat.reshape(x.shape)

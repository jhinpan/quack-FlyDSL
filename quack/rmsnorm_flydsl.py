# Copyright (c) 2026, Tri Dao.

"""Explicit ROCm/FlyDSL plain RMSNorm backend.

This module is intentionally opt-in. Importing ``quack`` does not import
FlyDSL, and this backend does not alter Quack's existing CUDA/CuTe dispatch.
"""

import math
import numbers

import torch

from quack._flydsl.kernel_utils import FLYDSL_BUILD_LOCK, run_compiled
from quack._flydsl.rmsnorm_bwd_kernel import (
    build_rmsnorm_bwd_module,
    build_rmsnorm_bwd_two_stage_module,
    is_rmsnorm_bwd_two_stage_vec_config,
)
from quack._flydsl.rmsnorm_common import EPS
from quack._flydsl.rmsnorm_kernel import build_rmsnorm_module


__all__ = ["rmsnorm"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_ARCHES = frozenset({"gfx942", "gfx950"})
_FWD_CACHE: dict[tuple, object] = {}
_BWD_CACHE: dict[tuple, object] = {}
_BWD_CU_COUNT_CACHE: dict[torch.device, int] = {}
_DEVICE_ARCH_CACHE: dict[int, str] = {}
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


def _reject_unsupported_features(**kwargs) -> None:
    """Reject the parts of the upstream RMSNorm contract this backend lacks.

    Keeping the list in one place means an unsupported call names the missing
    feature instead of failing somewhere deeper with a shape or dtype error.
    """
    unsupported = (
        ("bias", None, "a bias"),
        ("residual", None, "residual fusion"),
        ("out_dtype", None, "out_dtype"),
        ("residual_dtype", None, "residual_dtype"),
        ("prenorm", False, "prenorm outputs"),
        ("weight_offset", 0.0, "weight_offset"),
    )
    for name, default, description in unsupported:
        if kwargs[name] is not default and kwargs[name] != default:
            raise NotImplementedError(
                f"the FlyDSL RMSNorm backend does not support {description} ({name})"
            )
    if kwargs["weight"] is None:
        raise NotImplementedError("the FlyDSL RMSNorm backend requires an explicit weight")


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
        raise NotImplementedError(
            "the FlyDSL RMSNorm backend does not support per-head weights; "
            f"weight must be 1-D, got shape {tuple(weight.shape)}"
        )

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
    # Spelled as a comparison chain rather than math.isfinite so it still
    # traces when Dynamo hands us a symbolic float under dynamic=True. It
    # rejects NaN, both infinities and non-positive values just the same.
    if not 0.0 < eps < math.inf:
        raise ValueError(f"eps must be finite and positive, got {eps}")

    m = x.numel() // n
    return m, n, eps


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


def _select_rmsnorm_bwd_config(
    m: int,
    n: int,
    dtype_str: str,
    device: torch.device,
) -> tuple[str, int | None]:
    """Pick the weight-gradient reduction.

    The atomic path accumulates dweight with unordered fp32 atomics, so it is
    not run-to-run reproducible. The staged path uses a fixed reduction tree,
    and n is capped well below _BWD_TWO_STAGE_MAX_N by the public API, so it
    is always available when reproducibility is asked for.
    """
    deterministic = torch.are_deterministic_algorithms_enabled()
    if n <= _BWD_TWO_STAGE_MAX_N and (deterministic or m >= _BWD_TWO_STAGE_MIN_ROWS):
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
        key = (
            x.device.index,
            n,
            dtype_str,
            weight_dtype_str,
            store_rstd,
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
                    store_rstd=store_rstd,
                    weight_dtype_str=weight_dtype_str,
                    arch=arch,
                ),
            )
        stream = _current_raw_stream(x.device)
        if store_rstd:
            run_compiled(launcher, x, weight, out, rstd, m, eps, stream)
        else:
            run_compiled(launcher, x, weight, out, m, eps, stream)


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
        key = (
            path,
            x.device.index,
            n,
            dtype_str,
            weight_dtype_str,
            num_programs,
        )
        launcher = _BWD_CACHE.get(key)
        stream = _current_raw_stream(x.device)
        if path == "two_stage":
            if launcher is None:
                launcher = _build_cached(
                    _BWD_CACHE,
                    key,
                    x.device,
                    lambda arch: build_rmsnorm_bwd_two_stage_module(
                        n,
                        dtype_str,
                        num_programs,
                        weight_dtype_str=weight_dtype_str,
                        arch=arch,
                    ),
                )
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
            launcher = _build_cached(
                _BWD_CACHE,
                key,
                x.device,
                lambda arch: build_rmsnorm_bwd_module(
                    n,
                    dtype_str,
                    weight_dtype_str=weight_dtype_str,
                    arch=arch,
                ),
            )
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
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    eps: float = EPS,
    prenorm: bool = False,
    weight_offset: float = 0.0,
) -> torch.Tensor:
    """Apply plain RMSNorm over the last dimension using the FlyDSL backend.

    The signature mirrors :func:`quack.rmsnorm` so this backend can stand in
    for it. Everything beyond plain weighted RMSNorm is rejected by name
    rather than silently ignored.
    """
    _reject_unsupported_features(
        weight=weight,
        bias=bias,
        residual=residual,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        prenorm=prenorm,
        weight_offset=weight_offset,
    )
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

# Copyright (c) 2026, Tri Dao.

"""Explicit ROCm/FlyDSL RMSNorm backend.

This module is intentionally opt-in. Importing ``quack`` does not import
FlyDSL, and this backend does not alter Quack's existing CUDA/CuTe dispatch.
"""

import math
import numbers

import torch

from quack.flydsl.kernel_utils import FLYDSL_BUILD_LOCK, run_compiled
from quack.flydsl.rmsnorm_bwd_kernel import (
    build_rmsnorm_feature_bwd_atomic_module,
    build_rmsnorm_feature_bwd_two_stage_module,
    build_rmsnorm_bwd_module,
    build_rmsnorm_bwd_two_stage_module,
    TWO_STAGE_MAX_NUM_THREADS,
    rmsnorm_bwd_two_stage_config,
)
from quack.flydsl.rmsnorm_common import EPS
from quack.flydsl.rmsnorm_kernel import build_rmsnorm_feature_module, build_rmsnorm_module


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


def _validate_feature_inputs(
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

    if not 1 <= n <= 8192:
        raise ValueError(f"x normalized dimension must be between 1 and 8192, got {n}")
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
        config = rmsnorm_bwd_two_stage_config(n, dtype_str)
        if config.vectorized:
            num_programs = num_cus if m < 2048 else (3 * num_cus) // 2
        else:
            num_programs = num_cus if m < 1024 else 2 * num_cus
        # A row narrower than the widest block gets a narrower block, so launch
        # proportionally more of them to keep the same threads resident.
        num_programs *= TWO_STAGE_MAX_NUM_THREADS // config.num_threads
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


def _launch_rmsnorm_feature_fwd(
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
            "feature",
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
                lambda arch: build_rmsnorm_feature_module(
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


@torch.library.custom_op(
    "quack::_rmsnorm_flydsl_feature_fwd",
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
def _rmsnorm_flydsl_feature_fwd_op(
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
    _launch_rmsnorm_feature_fwd(
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


@_rmsnorm_flydsl_feature_fwd_op.register_fake
def _rmsnorm_flydsl_feature_fwd_fake(
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
    return None


def _dispatch_rmsnorm_feature_fwd(*args, **kwargs) -> None:
    if torch.compiler.is_compiling():
        _rmsnorm_flydsl_feature_fwd_op(*args, **kwargs)
    else:
        _launch_rmsnorm_feature_fwd(*args, **kwargs)


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


def _launch_rmsnorm_feature_bwd(
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

    path, selected_programs = _select_rmsnorm_bwd_config(
        m,
        n,
        source_dtype_str,
        source.device,
    )
    has_parameter_grads = compute_dweight or compute_dbias
    num_programs = selected_programs if path == "two_stage" and has_parameter_grads else 0
    path = "two_stage" if num_programs else "atomic"
    workspace_rows = num_programs * num_heads * (int(compute_dweight) + int(compute_dbias))
    workspace = torch.empty(
        (workspace_rows, n) if workspace_rows else (1, n),
        device=source.device,
        dtype=torch.float32,
    )
    with torch.cuda.device(source.device):
        key = (
            f"feature-{path}",
            source.device.index,
            n,
            source_dtype_str,
            dy_dtype_str,
            dx_dtype_str,
            dresidual_dtype_str,
            dresidual_out_dtype_str,
            weight_dtype_str,
            has_weight,
            has_bias,
            compute_dweight,
            compute_dbias,
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
                lambda arch: (
                    build_rmsnorm_feature_bwd_two_stage_module(
                        n,
                        source_dtype_str,
                        dy_dtype_str,
                        dx_dtype_str,
                        dresidual_dtype_str,
                        dresidual_out_dtype_str,
                        num_programs,
                        weight_dtype_str=weight_dtype_str,
                        has_weight=has_weight,
                        has_bias=has_bias,
                        compute_dweight=compute_dweight,
                        compute_dbias=compute_dbias,
                        has_residual=has_residual,
                        has_dresidual_out=has_dresidual_out,
                        per_head=per_head,
                        num_heads=num_heads,
                        arch=arch,
                    )
                    if path == "two_stage"
                    else build_rmsnorm_feature_bwd_atomic_module(
                        n,
                        source_dtype_str,
                        dy_dtype_str,
                        dx_dtype_str,
                        dresidual_dtype_str,
                        dresidual_out_dtype_str,
                        weight_dtype_str=weight_dtype_str,
                        has_weight=has_weight,
                        has_bias=has_bias,
                        compute_dweight=compute_dweight,
                        compute_dbias=compute_dbias,
                        has_residual=has_residual,
                        has_dresidual_out=has_dresidual_out,
                        per_head=per_head,
                        num_heads=num_heads,
                        arch=arch,
                    )
                ),
            )
        stream = _current_raw_stream(source.device)
        if path == "two_stage":
            run_compiled(
                launcher,
                source,
                weight,
                dout,
                dresidual_out,
                rstd,
                dx,
                dresidual,
                dweight,
                dbias,
                workspace,
                m,
                weight_offset,
                stream,
            )
        else:
            run_compiled(
                launcher,
                source,
                weight,
                dout,
                dresidual_out,
                rstd,
                dx,
                dresidual,
                dweight,
                dbias,
                m,
                weight_offset,
                stream,
            )


@torch.library.custom_op(
    "quack::_rmsnorm_flydsl_feature_bwd",
    mutates_args=("dx", "dresidual", "dweight", "dbias"),
    device_types="cuda",
    schema=(
        "(Tensor source, Tensor weight, Tensor dout, Tensor dresidual_out, "
        "Tensor rstd, Tensor(a5!) dx, Tensor(a6!) dresidual, "
        "Tensor(a7!) dweight, Tensor(a8!) dbias, float weight_offset, "
        "bool has_weight, bool has_bias, bool compute_dweight, "
        "bool compute_dbias, bool has_residual, "
        "bool has_dresidual_out, bool per_head, int num_heads) -> ()"
    ),
)
def _rmsnorm_flydsl_feature_bwd_op(
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
    has_residual: bool,
    has_dresidual_out: bool,
    per_head: bool,
    num_heads: int,
) -> None:
    _launch_rmsnorm_feature_bwd(
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
        has_residual=has_residual,
        has_dresidual_out=has_dresidual_out,
        per_head=per_head,
        num_heads=num_heads,
    )


@_rmsnorm_flydsl_feature_bwd_op.register_fake
def _rmsnorm_flydsl_feature_bwd_fake(
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
    has_residual: bool,
    has_dresidual_out: bool,
    per_head: bool,
    num_heads: int,
) -> None:
    return None


def _dispatch_rmsnorm_feature_bwd(*args, **kwargs) -> None:
    if torch.compiler.is_compiling():
        _rmsnorm_flydsl_feature_bwd_op(*args, **kwargs)
    else:
        _launch_rmsnorm_feature_bwd(*args, **kwargs)


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


class _RMSNormFeatureFunction(torch.autograd.Function):
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
    ):
        programs = x.shape[0] * num_heads
        needs_grad = any(ctx.needs_input_grad[:4])
        store_residual = has_residual or prenorm or residual_out_dtype != x.dtype
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
        _dispatch_rmsnorm_feature_fwd(
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
            ctx.residual_out_dtype = residual_out.dtype
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
        dout = dout.contiguous()
        has_dresidual_out = ctx.prenorm
        if has_dresidual_out:
            dresidual_out = args[0].contiguous()
        else:
            dresidual_out = source

        dx = torch.empty_like(source, dtype=ctx.x_dtype)
        dresidual = (
            torch.empty_like(source, dtype=ctx.residual_dtype)
            if ctx.has_residual
            else torch.empty(0, device=source.device, dtype=ctx.residual_dtype)
        )
        parameter_shape = (ctx.num_heads, source.shape[-1]) if ctx.per_head else (source.shape[-1],)
        dweight_f32 = torch.zeros(
            parameter_shape if ctx.weight_needs_grad else (1,),
            device=source.device,
            dtype=torch.float32,
        )
        dbias_f32 = torch.zeros(
            parameter_shape if ctx.bias_needs_grad else (1,),
            device=source.device,
            dtype=torch.float32,
        )
        _dispatch_rmsnorm_feature_bwd(
            source,
            weight,
            dout,
            dresidual_out,
            rstd,
            dx,
            dresidual,
            dweight_f32,
            dbias_f32,
            ctx.weight_offset,
            has_weight=ctx.has_weight,
            has_bias=ctx.has_bias,
            compute_dweight=ctx.weight_needs_grad,
            compute_dbias=ctx.bias_needs_grad,
            has_residual=ctx.has_residual,
            has_dresidual_out=has_dresidual_out,
            per_head=ctx.per_head,
            num_heads=ctx.num_heads,
        )

        dweight = (
            dweight_f32.reshape(parameter_shape).to(weight.dtype) if ctx.weight_needs_grad else None
        )
        dbias = dbias_f32.reshape(parameter_shape).to(bias.dtype) if ctx.bias_needs_grad else None
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
    """Apply RMSNorm over the last dimension using the FlyDSL backend."""
    m, n, num_heads, per_head, eps, weight_offset = _validate_feature_inputs(
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
    x_flat = x.reshape(-1, *last_shape).contiguous()

    plain_optimized = (
        weight is not None
        and bias is None
        and residual is None
        and not per_head
        and output_dtype == x.dtype
        and residual_dtype is None
        and not prenorm
        and weight_offset == 0.0
        and (
            weight.dtype == x.dtype
            or (x.dtype in (torch.float16, torch.bfloat16) and weight.dtype == torch.float32)
        )
    )
    if plain_optimized:
        out_flat = _RMSNormFunction.apply(
            x_flat,
            weight.contiguous(),
            eps,
        )
        return out_flat.reshape(x.shape)

    weight_arg = (
        weight.contiguous()
        if weight is not None
        else torch.empty(parameter_shape, device=x.device, dtype=x.dtype)
    )
    bias_arg = (
        bias.contiguous()
        if bias is not None
        else torch.empty(parameter_shape, device=x.device, dtype=x.dtype)
    )
    residual_arg = (
        residual.reshape(-1, *last_shape).contiguous() if residual is not None else x_flat
    )
    result = _RMSNormFeatureFunction.apply(
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
    )
    if isinstance(result, tuple):
        return tuple(tensor.reshape(x.shape) for tensor in result)
    return result.reshape(x.shape)

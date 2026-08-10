# Copyright (c) 2026, Tri Dao.

"""FlyDSL-free-at-import preflight policy for the FlyDSL RMSNorm backend."""

import math
import numbers
import os

import torch

from .rmsnorm_config import MAX_N, N_ALIGNMENT

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# gfx950 is the only architecture this backend has been built and run on.
# gfx942 is wave64 and should work, but until it executes on real hardware it
# is not claimed here.
_SUPPORTED_ARCHES = frozenset({"gfx950"})
# Row counts cross the Int32 kernel ABI here; reject before FlyDSL's argument
# packing raises a struct.error from inside the dispatch.
_MAX_ROWS = 2**31 - 1
_DEVICE_ARCH_CACHE: dict[int, str] = {}
_AUTOTUNE_ARCH_CACHE: dict[tuple, str] = {}
_VALIDATED_INPUT_LAST: list[tuple[tuple, tuple] | None] = [None]
_AUTOTUNE_TARGET_ENV_VARS = (
    "FLYDSL_COMPILE_BACKEND",
    "ARCH",
    "FLYDSL_GPU_ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
)


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
    from flydsl.compiler import get_backend

    target = get_backend().target
    return target.backend, _normalize_arch(target.arch)


def _flydsl_runtime_arch() -> str:
    """Ask FlyDSL which architecture its runtime helpers assume."""
    from flydsl.runtime.device import get_rocm_arch

    return _normalize_arch(get_rocm_arch())


def _validate_arch(device: torch.device) -> str:
    """Resolve and validate the architecture behind ``device``.

    The device query is memoized because a device index cannot change identity
    within a process. FlyDSL's compile target and runtime-helper architecture
    are *not* memoized here: both are environment-driven and can change under a
    long-lived process, so every build rechecks them. All queries run only on
    the build path, never on a launch.
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
    runtime_arch = _flydsl_runtime_arch()
    if runtime_arch != actual:
        raise ValueError(
            "FlyDSL RMSNorm does not support mixed architectures: "
            f"cuda:{index} is {actual}, but FlyDSL runtime helpers use {runtime_arch}. "
            "Set FLYDSL_GPU_ARCH or HSA_OVERRIDE_GFX_VERSION to the device architecture."
        )
    return actual


def _validated_autotune_arch(device: torch.device) -> str:
    """Revalidate the compile target only when its controlling environment changes."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index,) + tuple(os.environ.get(name, "") for name in _AUTOTUNE_TARGET_ENV_VARS)
    arch = _AUTOTUNE_ARCH_CACHE.get(key)
    if arch is None:
        arch = _validate_arch(device)
        _AUTOTUNE_ARCH_CACHE[key] = arch
    return arch


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

    # Both limits name torch as the fallback, not quack.rmsnorm: the CuTe
    # backend imports cuda.bindings and so cannot be imported on ROCm, which is
    # the only platform that reaches this code.
    if not 1 <= n <= MAX_N:
        raise ValueError(
            f"x normalized dimension must be between 1 and {MAX_N}, got {n}; "
            "use torch.nn.functional.rms_norm for rows this backend does not accept"
        )
    if n % N_ALIGNMENT:
        # Every row start is then naturally aligned and every operand reaches
        # full vector width, which is what lets the kernels treat the vector
        # size as a property of the dtype. Refusing is deliberate: serving such
        # a row with a narrower access is a path nothing measures and nothing
        # in practice reaches, so it would only distort what gets optimized.
        raise ValueError(
            f"x normalized dimension must be a multiple of {N_ALIGNMENT}, got {n}; "
            "use torch.nn.functional.rms_norm for rows this backend does not accept"
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


def _validation_tensor_metadata(tensor):
    if type(tensor) is not torch.Tensor:
        return (type(tensor),)
    return (
        torch.Tensor,
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tensor.layout,
    )


def _validated_inputs(
    x,
    weight,
    bias,
    residual,
    out_dtype,
    residual_dtype,
    eps,
    prenorm,
    weight_offset,
):
    if torch.compiler.is_compiling():
        return _validate_inputs(
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
    if (
        isinstance(eps, bool)
        or not isinstance(eps, numbers.Real)
        or not isinstance(prenorm, bool)
        or isinstance(weight_offset, bool)
        or not isinstance(weight_offset, numbers.Real)
    ):
        return _validate_inputs(
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
    key = (
        _validation_tensor_metadata(x),
        _validation_tensor_metadata(weight),
        _validation_tensor_metadata(bias),
        _validation_tensor_metadata(residual),
        out_dtype,
        residual_dtype,
        type(eps),
        eps,
        type(prenorm),
        prenorm,
        type(weight_offset),
        weight_offset,
    )
    cached = _VALIDATED_INPUT_LAST[0]
    if cached is not None and cached[0] == key:
        return cached[1]
    result = _validate_inputs(
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
    _VALIDATED_INPUT_LAST[0] = (key, result)
    return result


def _packed_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Copy only when a row is not already contiguous along its last axis.

    Every operand reaches the kernels as a row-scoped buffer descriptor, so the
    requirement is that each row be contiguous, not that the whole tensor be. A
    row-padded view (``full[:, :n]``, stride ``(pitch, 1)``) already satisfies
    it and the descriptor is sized to ``n``, so the padding is never addressed;
    copying such a view would cost a full extra read and write of the
    activation for nothing.

    Under ``torch.compile`` the copy is unconditional: the predicate below
    inspects strides, which are not available on a symbolic tensor.

    Canonicalising here rather than adding a layout term to the forward
    launcher cache is deliberate. Two views with the same shape and dtype but
    different layouts must not share a launcher, and making them share one
    canonical layout is a stronger fix than making them miss the cache.
    """
    tensor = _unambiguous_layout(tensor)
    if torch.compiler.is_compiling():
        return tensor.contiguous()
    return tensor if _rows_are_disjoint_and_packed(tensor) else tensor.contiguous()


def _unambiguous_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the first unit-stride axis is the row axis.

    FlyDSL picks the leading dimension as ``next(i for i in range(ndim) if
    stride[i] == 1)``. A size-1 axis ahead of the row that also carries stride 1
    wins that search and silently redefines the ABI. ``.contiguous()`` cannot
    fix it: such a tensor already reports ``is_contiguous()``, so the copy is
    the identity.

    Relabelling is enough and costs nothing. A size-1 axis has no observable
    stride -- there is no second element to step to -- so squeezing it out and
    putting it back rewrites the stride to a non-unit value describing the same
    memory. The reinserted stride can be 0, which is fine here: the axis no
    longer holds the row's unit stride, and the packing predicate rejects zero
    strides separately.
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
    """Whether every row is packed and no two rows share storage.

    Row-scoped descriptors are safe exactly when each row is contiguous and
    distinct rows do not overlap. A broadcast or reversed view satisfies
    neither, and both are silently wrong rather than loud, so they are copied.
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

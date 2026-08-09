# Copyright (c) 2026, Tri Dao.

"""Pure input validation for the FlyDSL RMSNorm backend."""

import math
import numbers

import torch

from .rmsnorm_config import MAX_N, N_ALIGNMENT

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Row counts cross the Int32 kernel ABI here; reject before FlyDSL's argument
# packing raises a struct.error from inside the dispatch.
_MAX_ROWS = 2**31 - 1


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

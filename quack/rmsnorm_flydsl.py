# Copyright (c) 2026, Tri Dao.

"""ROCm/FlyDSL RMSNorm backend.

On ROCm, the package-level ``quack.rmsnorm`` export resolves lazily to this
module's ``rmsnorm`` function. Importing ``quack`` alone does not import FlyDSL,
and this backend does not alter Quack's existing CUDA/CuTe dispatch. The
``rmsnorm_autotuned`` entry point remains FlyDSL-specific.
"""

import math
import numbers
import os

import torch

from quack.flydsl.rmsnorm_autotune import (
    RMSNORM_AUTOTUNE_SCHEMA_VERSION,
    _rmsnorm_fwd_tuner,
)
from quack.flydsl.rmsnorm_bwd_autotune import (
    RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION,
    _rmsnorm_bwd_tuner,
)
from quack.flydsl.rmsnorm_bwd_kernel import (
    TWO_STAGE_MAX_NUM_THREADS,
    build_rmsnorm_bwd_two_stage_module,
    rmsnorm_bwd_two_stage_config,
)
from quack.flydsl.rmsnorm_common import EPS, FLYDSL_BUILD_LOCK, run_compiled
from quack.flydsl.rmsnorm_config import (
    MAX_N,
    N_ALIGNMENT,
    next_power_of_two,
)
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
_FWD_AUTOTUNED_FAST_CACHE: dict[tuple, tuple] = {}
_BWD_AUTOTUNED_FAST_CACHE: dict[tuple, tuple] = {}
_FWD_AUTOTUNED_LAST: list[tuple[tuple, tuple] | None] = [None]
_FWD_PUBLIC_GENERIC_LAST: list[object | None] = [None]
_FWD_PUBLIC_RESOLVED_LAST: list[tuple[object, tuple, tuple] | None] = [None]
_VALIDATED_INPUT_LAST: list[tuple[tuple, tuple] | None] = [None]
_BWD_CU_COUNT_CACHE: dict[torch.device, int] = {}
_DEVICE_ARCH_CACHE: dict[int, str] = {}
_AUTOTUNE_ARCH_CACHE: dict[tuple, str] = {}
_EAGER_EMPTY_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_AUTOTUNE_TARGET_ENV_VARS = (
    "FLYDSL_COMPILE_BACKEND",
    "ARCH",
    "FLYDSL_GPU_ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
)


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


def _current_raw_stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


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
    workspace = _eager_empty(source.device, torch.float32)
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

    Canonicalising here rather than adding a layout term to ``_FWD_CACHE`` is
    deliberate. Two views with the same shape and dtype but different layouts
    must not share a launcher, and making them share one canonical layout is a
    stronger fix than making them miss the cache.
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

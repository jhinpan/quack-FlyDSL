# Copyright (c) 2026, Tri Dao.

"""Architecture resolution and validation for FlyDSL RMSNorm builds."""

import os

import torch

# gfx950 is the only architecture this backend has been built and run on.
# gfx942 is wave64 and should work, but until it executes on real hardware it
# is not claimed here.
_SUPPORTED_ARCHES = frozenset({"gfx950"})
_DEVICE_ARCH_CACHE: dict[int, str] = {}
_AUTOTUNE_ARCH_CACHE: dict[tuple, str] = {}
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

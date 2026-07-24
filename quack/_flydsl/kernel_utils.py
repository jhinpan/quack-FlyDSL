# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Minimal FlyDSL helpers required by the vendored RMSNorm kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch


def dtype_to_elem_type(dtype_str: str):
    """Map a supported RMSNorm dtype string to its FlyDSL type."""
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r}")


def get_warp_size(arch=None) -> int:
    """Return the wavefront size for the selected ROCm architecture."""
    arch = get_rocm_arch() if arch is None else arch
    return 32 if is_rdna_arch(arch) else 64


def run_compiled(executable, *args) -> None:
    """Compile-and-run once, then dispatch through the cached callable."""
    compiled = getattr(executable, "_cf", None)
    if compiled is None:
        compiled = flyc.compile(executable, *args)
        executable._cf = compiled
    else:
        compiled(*args)

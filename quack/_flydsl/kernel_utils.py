# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Minimal FlyDSL helpers required by the vendored RMSNorm kernels."""

import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch


# Serializes every FlyDSL trace and codegen in the process. Compilation is
# rare and already tens of milliseconds, so one lock costs nothing and keeps
# concurrent first calls out of the compiler's global state.
FLYDSL_BUILD_LOCK = threading.RLock()


def dtype_to_elem_type(dtype_str: str):
    """Map a supported RMSNorm dtype string to its FlyDSL type."""
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r}")


def dtype_to_elem_bits(dtype_str: str) -> int:
    """Storage width of one element, the basis for every vector width."""
    if dtype_str == "f32":
        return 32
    if dtype_str in ("f16", "bf16"):
        return 16
    raise ValueError(f"unsupported dtype: {dtype_str!r}")


def get_warp_size(arch=None) -> int:
    """Return the wavefront size for the selected ROCm architecture."""
    arch = get_rocm_arch() if arch is None else arch
    return 32 if is_rdna_arch(arch) else 64


def atomic_add(
    destination,
    offset,
    value,
    *,
    dtype_bytes: int = 4,
):
    """Atomically add a scalar into a global-memory tensor element."""
    pointer_type = ir.Type.parse("!llvm.ptr<1>")
    base_pointer = _fly.extract_aligned_pointer_as_index(pointer_type, destination)
    base_pointer = _llvm.PtrToIntOp(T.i64, base_pointer).result
    byte_offset = arith.index_cast(T.i64, fx.Index(offset) * fx.Index(dtype_bytes))
    pointer = _llvm.AddOp(
        base_pointer,
        byte_offset,
        _llvm.IntegerOverflowFlags(0),
    ).result
    pointer = _llvm.IntToPtrOp(pointer_type, pointer).result
    pointer = pointer._value if const_expr(hasattr(pointer, "_value")) else pointer

    raw_value = value.ir_value() if const_expr(hasattr(value, "ir_value")) else value
    return _llvm.AtomicRMWOp(
        _llvm.AtomicBinOp.fadd,
        pointer,
        raw_value,
        _llvm.AtomicOrdering.monotonic,
        syncscope="agent",
        alignment=dtype_bytes,
    ).result


def run_compiled(executable, *args) -> None:
    """Compile-and-run once, then dispatch through the cached callable."""
    compiled = getattr(executable, "_cf", None)
    if compiled is not None:
        compiled(*args)
        return
    with FLYDSL_BUILD_LOCK:
        if getattr(executable, "_cf", None) is None:
            # flyc.compile performs the first launch as well as the codegen.
            executable._cf = flyc.compile(executable, *args)
            return
    executable._cf(*args)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Shared device-side helpers for the plain RMSNorm kernels."""

import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import full

from .kernel_utils import get_warp_size
from .rmsnorm_config import ACCESS_BITS


EPS = 1e-6
BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()

_BUFFER_COPY_OPS = {
    8: fx.rocdl.BufferCopy8b,
    16: fx.rocdl.BufferCopy16b,
    32: fx.rocdl.BufferCopy32b,
    64: fx.rocdl.BufferCopy64b,
    128: fx.rocdl.BufferCopy128b,
}


def buffer_copy_atom(access_bits: int, elem_bits: int):
    """Copy atom that moves ``access_bits`` at a time.

    The width follows from the vector size the config picked, so a row that is
    not a whole number of 128-bit vectors uses the widest access that does
    divide it rather than dropping to scalar.
    """
    try:
        copy_op = _BUFFER_COPY_OPS[access_bits]
    except KeyError:
        raise ValueError(f"no buffer copy for a {access_bits}-bit access") from None
    return fx.make_copy_atom(copy_op(), elem_bits)


def weight_access_plan(vecsize: int, weight_dtype_width: int) -> tuple[int, int]:
    """Split ``vecsize`` weights into whole accesses of at most 128 bits.

    Returns the number of accesses and the elements each one carries. Only an
    FP32 weight paired with a full 16-bit activation vector needs more than one.
    """
    accesses = max(1, (vecsize * weight_dtype_width) // ACCESS_BITS)
    return accesses, vecsize // accesses


def assert_arch_matches_reductions(arch: str) -> None:
    """Fail loudly if a target's wavefront differs from the baked-in one.

    The block reductions unroll over ``WARP_SIZE``, which is resolved once at
    import time. Every architecture this backend supports is wave64, so this
    only fires if the supported set grows without the reductions following.
    """
    target_warp_size = get_warp_size(arch)
    if target_warp_size != WARP_SIZE:
        raise RuntimeError(
            f"FlyDSL RMSNorm reductions are built for a wavefront of {WARP_SIZE}, "
            f"but {arch} has {target_warp_size}"
        )


def row_buffer(tensor, row, elem_bits: int, n: int):
    """Wrap a single row of ``tensor`` in its own buffer descriptor.

    A buffer descriptor addresses at most 4 GiB, so wrapping the whole tensor
    and then slicing a row would silently wrap around on any operand larger
    than that. Slicing first keeps every descriptor one row wide, which also
    turns the hardware bounds check into a real per-row guard.
    """
    return fx.rocdl.make_buffer_tensor(
        fx.slice(tensor, (row, None)),
        num_records_bytes=n * (elem_bits // 8),
    )


def row_head_buffer(tensor, row, head, elem_bits: int, n: int):
    """Wrap one ``(row, head)`` slice of a per-head tensor."""
    return fx.rocdl.make_buffer_tensor(
        fx.slice(tensor, (row, head, None)),
        num_records_bytes=n * (elem_bits // 8),
    )


def make_reduction_storage(red_slots: int):
    """One fp32 slot per wave, for the block half of the reduction.

    The reduction itself stays inline in each kernel: FlyDSL rewrites the AST
    of the decorated function only, so a shared helper containing
    ``if lane == 0`` would be traced as a plain Python conditional and raise.
    """

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    register = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, register)
    return fx.memref_load_vec(register)[0]


def store_scalar(copy_atom, elem_dtype, store_dtype, divided_tensor, index, value):
    register = fx.make_rmem_tensor(1, elem_dtype)
    tensor = full(1, store_dtype(value), store_dtype)
    fx.memref_store_vec(tensor, register)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, register, view)


def load_vec(copy_atom, vec_width, elem_dtype, divided_tensor, index):
    register = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(divided_tensor, (None, index)), register)
    return fx.memref_load_vec(register)


def load_weight_vec(
    copy_atom,
    weight_elem_dtype,
    weight_dtype_width,
    divided_tensor,
    index,
    vecsize,
):
    """Load ``vecsize`` weights as fp32 using whole accesses.

    An FP32 weight paired with a full 16-bit activation vector needs two
    accesses to cover it; every other combination needs exactly one.
    """
    accesses, per_access = weight_access_plan(vecsize, weight_dtype_width)
    if const_expr(accesses <= 1):
        return load_vec(
            copy_atom,
            vecsize,
            weight_elem_dtype,
            divided_tensor,
            index,
        ).to(fx.Float32)
    elements = []
    for part in range(accesses):
        chunk = load_vec(
            copy_atom,
            per_access,
            weight_elem_dtype,
            divided_tensor,
            index * accesses + part,
        )
        elements.extend(chunk[lane] for lane in range(per_access))
    return fx.Vector.from_elements(elements, fx.Float32)


def store_vec(copy_atom, vec_width, elem_dtype, value, divided_tensor, index):
    register = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(value, register)
    fx.copy_atom_call(copy_atom, register, fx.slice(divided_tensor, (None, index)))


def to_elem_scalar(dtype_str: str, elem_dtype, value):
    if const_expr(dtype_str == "f32"):
        return value
    return value.to(elem_dtype)


def to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, value, vec_width: int):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return value.to(elem_dtype)
        # Round to nearest even by hand, then pack pairs of results into one
        # 32-bit lane each. Pre-gfx95x has no packed convert to do this.
        bits = value.bitcast(fx.Uint32)
        upper = bits >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        rounded = value.bitcast(fx.Uint32) + bias
        bf16_bits = rounded >> 16
        even = bf16_bits.shuffle(bf16_bits, list(range(0, vec_width, 2)))
        odd = bf16_bits.shuffle(bf16_bits, list(range(1, vec_width, 2)))
        return (even | (odd << 16)).bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return value
    return value.to(elem_dtype)


def resolve_rmsnorm_weight_dtype(
    dtype_str: str,
    weight_dtype_str: str | None = None,
) -> str:
    """Resolve the v1 activation/weight dtype contract."""
    weight_dtype_str = dtype_str if weight_dtype_str is None else weight_dtype_str
    supported = dtype_str in ("f16", "bf16", "f32") and (
        weight_dtype_str == dtype_str
        or (dtype_str in ("f16", "bf16") and weight_dtype_str == "f32")
    )
    if not supported:
        raise ValueError(
            "RMSNorm supports matching activation/weight dtypes or "
            f"FP16/BF16 activations with FP32 weights, got {dtype_str}/{weight_dtype_str}"
        )
    return weight_dtype_str

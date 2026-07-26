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
from .rmsnorm_tiling import VECTOR_BITS


EPS = 1e-6
BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()


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


def make_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]
        s_red2: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def make_single_reduction_storage(red_slots: int):
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
    weight_elem_bits,
    divided_tensor,
    index,
    vec_width,
):
    """Load ``vec_width`` weights as fp32 using whole 128-bit accesses.

    An FP32 weight paired with 16-bit activations needs two accesses to cover
    one activation vector; every other combination needs exactly one.
    """
    per_access = weight_vec_width(weight_elem_bits)
    accesses = vec_width // per_access
    if const_expr(accesses <= 1):
        return load_vec(
            copy_atom,
            vec_width,
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


def to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, value):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return value.to(elem_dtype)
        bits = value.bitcast(fx.Uint32)
        upper = bits >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        rounded = value.bitcast(fx.Uint32) + bias
        bf16_bits = rounded >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
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


def weight_vec_width(weight_elem_bits: int) -> int:
    """Weight elements carried by one 128-bit access."""
    return VECTOR_BITS // weight_elem_bits

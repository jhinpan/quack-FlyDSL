# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Host and device helpers shared by the RMSNorm kernels."""

import math
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import full
from flydsl.runtime.device import is_rdna_arch

from .rmsnorm_config import ACCESS_BITS, WAVE_SIZE

EPS = 1e-6

# The block reductions unroll over the wavefront at trace time, and the launch
# geometry sizes lane groups against the same number. Taken from the config
# module rather than queried from the environment, so the two cannot disagree
# and neither can drift from the target a build specializes for.
WARP_SIZE = WAVE_SIZE

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


def has_hw_bf16_convert(arch: str) -> bool:
    """Whether the target converts fp32 to bf16 in hardware.

    gfx95x has the packed convert; earlier parts round to nearest even in
    software instead.
    """
    return str(arch).startswith("gfx95")


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


_BUFFER_COPY_OPS = {
    8: fx.rocdl.BufferCopy8b,
    16: fx.rocdl.BufferCopy16b,
    32: fx.rocdl.BufferCopy32b,
    64: fx.rocdl.BufferCopy64b,
    128: fx.rocdl.BufferCopy128b,
}


def buffer_copy_atom(access_bits: int, elem_bits: int, cache_modifier: int = 0):
    """Copy atom that moves ``access_bits`` at a time.

    The width follows from the vector size the config picked, so a row that is
    not a whole number of 128-bit vectors uses the widest access that does
    divide it rather than dropping to scalar. ``cache_modifier=2`` selects the
    CDNA non-temporal form; zero keeps the ordinary cached access.
    """
    try:
        copy_op = _BUFFER_COPY_OPS[access_bits]
    except KeyError:
        raise ValueError(f"no buffer copy for a {access_bits}-bit access") from None
    return fx.make_copy_atom(copy_op(cache_modifier), elem_bits)


def vector_access_plan(vecsize: int, dtype_width: int) -> tuple[int, int]:
    """Split ``vecsize`` elements into whole accesses of at most 128 bits.

    Returns the number of accesses and the elements each one carries. The
    vector width is chosen from the activation dtype, so only an FP32 operand
    paired with a full 16-bit activation vector needs more than one access.
    """
    accesses = max(1, (vecsize * dtype_width) // ACCESS_BITS)
    return accesses, vecsize // accesses


def require_wave64(arch: str) -> None:
    """Reject a build target the reductions cannot serve."""
    if is_rdna_arch(arch):
        raise ValueError(f"FlyDSL RMSNorm reductions require a wave64 target, but {arch} is wave32")


def row_buffer(tensor, row, elem_bits: int, n: int, valid=None):
    """Wrap a single row of ``tensor`` in its own buffer descriptor.

    A buffer descriptor addresses at most 4 GiB, so wrapping the whole tensor
    and then slicing a row would silently wrap around on any operand larger
    than that. Slicing first keeps every descriptor one row wide, which also
    turns the hardware bounds check into a real per-row guard.

    ``valid`` says whether ``row`` is a real row. A block that batches several
    rows can run past the last one, and sizing that group's descriptor to zero
    bytes makes the hardware drop its loads and stores, which is cheaper than
    predicating each of them.
    """
    return fx.rocdl.make_buffer_tensor(
        fx.slice(tensor, (row, None)),
        num_records_bytes=_row_records(elem_bits, n, valid),
    )


def row_head_buffer(tensor, row, head, elem_bits: int, n: int, valid=None):
    """Wrap one ``(row, head)`` slice of a per-head tensor."""
    return fx.rocdl.make_buffer_tensor(
        fx.slice(tensor, (row, head, None)),
        num_records_bytes=_row_records(elem_bits, n, valid),
    )


def _row_records(elem_bits: int, n: int, valid):
    records = n * (elem_bits // 8)
    if valid is None:
        return records
    return valid.select(fx.Int32(records), fx.Int32(0))


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


def _dpp_shuffle_xor(value, offset: int):
    """Exchange an fp32 value within each 16-lane DPP row."""
    raw = value.ir_value()
    result_type = raw.type
    # Row shifts need complementary bank masks to implement XOR rather than a
    # one-way shift. Quad permutations cover the final two butterfly stages.
    if offset == 8:
        peer = fx.rocdl.update_dpp(result_type, raw, raw, 0x118, 0xF, 0xC, False)
        peer = fx.rocdl.update_dpp(result_type, peer, raw, 0x108, 0xF, 0x3, False)
    elif offset == 4:
        peer = fx.rocdl.update_dpp(result_type, raw, raw, 0x114, 0xF, 0xA, False)
        peer = fx.rocdl.update_dpp(result_type, peer, raw, 0x104, 0xF, 0x5, False)
    elif offset == 2:
        peer = fx.rocdl.update_dpp(result_type, raw, raw, 0x4E, 0xF, 0xF, False)
    elif offset == 1:
        peer = fx.rocdl.update_dpp(result_type, raw, raw, 0xB1, 0xF, 0xF, False)
    else:
        raise ValueError(f"unsupported DPP XOR offset: {offset}")
    return fx.Float32(peer)


def _ds_swizzle_xor(value, offset: int):
    """Exchange the two 16-lane halves of each 32-lane group."""
    bits = value.bitcast(fx.Uint32)
    # AMD's SWAP encoding is ``group_size << 10 | 0x1f``.
    peer = fx.rocdl.ds_swizzle(
        bits.ir_value().type,
        bits.ir_value(),
        fx.Int32((offset << 10) | 0x1F).ir_value(),
    )
    return fx.Uint32(peer).bitcast(fx.Float32)


def shuffle_reduce_add(value, lanes: int, shuffle_width, fast_math):
    """Add ``value`` across a compile-time-sized lane group."""
    result = value
    for shift_exp in range_constexpr(int(math.log2(lanes))):
        offset = lanes // (2 << shift_exp)
        if lanes in (32, 64) and offset <= 8:
            peer = _dpp_shuffle_xor(result, offset)
        elif lanes in (32, 64) and offset == 16:
            peer = _ds_swizzle_xor(result, offset)
        else:
            peer = result.shuffle_xor(offset, shuffle_width)
        result = result.addf(peer, fastmath=fast_math)
    return result


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


def load_dtype_vec(
    copy_atom,
    elem_dtype,
    dtype_width,
    divided_tensor,
    index,
    vecsize,
):
    """Load ``vecsize`` elements of any supported dtype as fp32.

    An FP32 operand paired with a full 16-bit activation vector needs two
    accesses to cover it; every other combination needs exactly one.
    """
    accesses, per_access = vector_access_plan(vecsize, dtype_width)
    if const_expr(accesses <= 1):
        return load_vec(
            copy_atom,
            vecsize,
            elem_dtype,
            divided_tensor,
            index,
        ).to(fx.Float32)
    elements = []
    for part in range(accesses):
        chunk = load_vec(
            copy_atom,
            per_access,
            elem_dtype,
            divided_tensor,
            index * accesses + part,
        )
        elements.extend(chunk[lane] for lane in range(per_access))
    return fx.Vector.from_elements(elements, fx.Float32)


def store_dtype_vec(
    copy_atom,
    elem_dtype,
    dtype_width,
    value,
    divided_tensor,
    index,
    vecsize,
):
    """Store a ``vecsize``-wide vector using the same whole-access split.

    Mirrors :func:`load_dtype_vec`. Only a 32-bit destination under a full
    16-bit activation vector splits; a 16-bit destination is always one
    access, so the software BF16 packing in :func:`to_elem_vec` never has to
    survive being sliced here.
    """
    accesses, per_access = vector_access_plan(vecsize, dtype_width)
    if const_expr(accesses <= 1):
        store_vec(copy_atom, vecsize, elem_dtype, value, divided_tensor, index)
        return
    for part in range(accesses):
        lanes = list(range(part * per_access, (part + 1) * per_access))
        store_vec(
            copy_atom,
            per_access,
            elem_dtype,
            value.shuffle(value, lanes),
            divided_tensor,
            index * accesses + part,
        )


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


def to_store_dtype(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, value, vecsize: int):
    """Narrow an fp32 value while preserving software BF16 vector packing."""
    if const_expr(vecsize > 1):
        return to_elem_vec(dtype_str, elem_dtype, use_hw_cvt_bf16, value, vecsize)
    return to_elem_scalar(dtype_str, elem_dtype, value)

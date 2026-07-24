# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Plain RMSNorm atomic and two-stage backward kernel builders."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch

from .kernel_utils import atomic_add, dtype_to_elem_type
from .rmsnorm_common import (
    BLOCK_THREADS,
    VEC_WIDTH,
    WARP_SIZE,
    load_scalar,
    load_vec,
    load_weight_vec,
    make_single_reduction_storage,
    resolve_rmsnorm_weight_dtype,
    store_scalar,
    store_vec,
    to_elem_vec,
    weight_vec_width,
)


DWEIGHT_REDUCE_COLS = 64
DWEIGHT_REDUCE_ROW_LANES = 4
DWEIGHT_REDUCE_THREADS = DWEIGHT_REDUCE_COLS * DWEIGHT_REDUCE_ROW_LANES
TWO_STAGE_PARTIAL_THREADS = 512


def is_rmsnorm_bwd_two_stage_vec_config(n: int, dtype_str: str) -> bool:
    """Return whether the staged kernel can use full-width vec8 column I/O."""
    return (
        dtype_str in ("f16", "bf16")
        and n >= TWO_STAGE_PARTIAL_THREADS * VEC_WIDTH
        and n % VEC_WIDTH == 0
    )


def build_rmsnorm_bwd_module(
    n: int,
    dtype_str: str,
    weight_dtype_str: str | None = None,
):
    """Build the one-block-per-row backward with fp32 weight atomics."""
    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    red_slots = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    shared_storage = make_single_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_bwd_kernel(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        dy: fx.Tensor,
        rstd_tensor: fx.Tensor,
        dx: fx.Tensor,
        dweight: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fast_math = arith.FastMathFlags.fast
        zero = fx.Float32(0.0)

        storage = fx.SharedAllocator().allocate(shared_storage).peek()
        s_red = storage.s_red.view(fx.make_layout(red_slots, 1))

        def wave_reduce_add(value):
            result = value
            for shift_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                offset = WARP_SIZE // (2 << shift_exp)
                peer = result.shuffle_xor(offset, WARP_SIZE)
                result = result.addf(peer, fastmath=fast_math)
            return result

        def block_reduce_add(value):
            if const_expr(red_slots == 1):
                return wave_reduce_add(value)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            reduced = wave_reduce_add(value)
            if lane == 0:
                fx.memref_store(reduced, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < red_slots
                safe_lane = in_range.select(lane, 0)
                wave_value = in_range.select(fx.memref_load(s_red, safe_lane), zero)
                wave_value = wave_reduce_add(wave_value)
                if lane == 0:
                    fx.memref_store(wave_value, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        input_buffer = fx.rocdl.make_buffer_tensor(input_tensor)
        gamma_buffer = fx.rocdl.make_buffer_tensor(gamma)
        dy_buffer = fx.rocdl.make_buffer_tensor(dy)
        rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
        dx_buffer = fx.rocdl.make_buffer_tensor(dx)

        input_div = fx.logical_divide(
            fx.slice(input_buffer, (row, None)),
            fx.make_layout(1, 1),
        )
        gamma_div = fx.logical_divide(gamma_buffer, fx.make_layout(1, 1))
        dy_div = fx.logical_divide(
            fx.slice(dy_buffer, (row, None)),
            fx.make_layout(1, 1),
        )
        rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(
            fx.slice(dx_buffer, (row, None)),
            fx.make_layout(1, 1),
        )

        copy_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        gamma_copy_atom = fx.make_copy_atom(
            (fx.rocdl.BufferCopy16b() if weight_elem_bits <= 16 else fx.rocdl.BufferCopy32b()),
            weight_elem_bits,
        )
        f32_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        rstd = load_scalar(f32_copy_atom, fx.Float32, rstd_div, row)

        thread_acc = zero
        for base in range_constexpr(0, n, BLOCK_THREADS):
            index = tid + base
            is_valid = index < n
            safe_index = is_valid.select(index, 0)
            x_elem = load_scalar(copy_atom, elem_dtype, input_div, safe_index)
            dy_elem = load_scalar(copy_atom, elem_dtype, dy_div, safe_index)
            gamma_elem = load_scalar(
                gamma_copy_atom,
                weight_elem_dtype,
                gamma_div,
                safe_index,
            )
            x_value = x_elem if dtype_str == "f32" else x_elem.to(fx.Float32)
            dy_value = dy_elem if dtype_str == "f32" else dy_elem.to(fx.Float32)
            gamma_value = gamma_elem if weight_dtype_str == "f32" else gamma_elem.to(fx.Float32)
            x_hat = x_value * rstd
            thread_acc = thread_acc + is_valid.select(
                x_hat * dy_value * gamma_value,
                zero,
            )

        correction = block_reduce_add(thread_acc) / float(n)
        for base in range_constexpr(0, n, BLOCK_THREADS):
            index = tid + base
            if index < n:
                x_elem = load_scalar(copy_atom, elem_dtype, input_div, index)
                dy_elem = load_scalar(copy_atom, elem_dtype, dy_div, index)
                gamma_elem = load_scalar(
                    gamma_copy_atom,
                    weight_elem_dtype,
                    gamma_div,
                    index,
                )
                x_value = x_elem if dtype_str == "f32" else x_elem.to(fx.Float32)
                dy_value = dy_elem if dtype_str == "f32" else dy_elem.to(fx.Float32)
                gamma_value = gamma_elem if weight_dtype_str == "f32" else gamma_elem.to(fx.Float32)
                x_hat = x_value * rstd
                weighted_dy = dy_value * gamma_value
                dx_value = (weighted_dy - x_hat * correction) * rstd
                dx_elem = dx_value if dtype_str == "f32" else dx_value.to(elem_dtype)
                store_scalar(
                    copy_atom,
                    elem_dtype,
                    elem_dtype,
                    dx_div,
                    index,
                    dx_elem,
                )
                atomic_add(dweight, index, dy_value * x_hat, dtype_bytes=4)

    @flyc.jit
    def launch_rmsnorm_bwd(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        dy: fx.Tensor,
        rstd_tensor: fx.Tensor,
        dx: fx.Tensor,
        dweight: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_bwd_kernel(
            input_tensor,
            gamma,
            dy,
            rstd_tensor,
            dx,
            dweight,
        )
        launcher.launch(
            grid=(m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_bwd


def build_rmsnorm_bwd_two_stage_module(
    n: int,
    dtype_str: str,
    num_programs: int,
    weight_dtype_str: str | None = None,
):
    """Build the persistent backward and deterministic weight finalizer."""
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")

    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    red_slots = max(
        1,
        (TWO_STAGE_PARTIAL_THREADS + WARP_SIZE - 1) // WARP_SIZE,
    )
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    use_vec = is_rmsnorm_bwd_two_stage_vec_config(n, dtype_str)
    io_width = VEC_WIDTH if use_vec else 1
    weight_io_width = weight_vec_width(weight_dtype_str) if use_vec else 1
    num_io_tiles = (n + io_width - 1) // io_width
    num_io_iters = (num_io_tiles + TWO_STAGE_PARTIAL_THREADS - 1) // TWO_STAGE_PARTIAL_THREADS
    partial_acc_size = num_io_iters * io_width
    arch = get_rocm_arch() if use_vec else ""
    use_hw_cvt_bf16 = arch == "gfx950" or str(arch).startswith("gfx95")
    shared_storage = make_single_reduction_storage(red_slots)
    dweight_reduce_storage = make_single_reduction_storage(DWEIGHT_REDUCE_THREADS)

    @flyc.kernel(known_block_size=[TWO_STAGE_PARTIAL_THREADS, 1, 1])
    def rmsnorm_bwd_partial_kernel(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        dy: fx.Tensor,
        rstd_tensor: fx.Tensor,
        dx: fx.Tensor,
        dweight_partial: fx.Tensor,
        m: fx.Int32,
    ):
        block = fx.block_idx.x
        tid = fx.thread_idx.x
        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fast_math = arith.FastMathFlags.fast
        zero = fx.Float32(0.0)

        storage = fx.SharedAllocator().allocate(shared_storage).peek()
        s_red = storage.s_red.view(fx.make_layout(red_slots, 1))

        def wave_reduce_add(value):
            result = value
            for shift_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                offset = WARP_SIZE // (2 << shift_exp)
                peer = result.shuffle_xor(offset, WARP_SIZE)
                result = result.addf(peer, fastmath=fast_math)
            return result

        def block_reduce_add(value):
            if const_expr(red_slots == 1):
                return wave_reduce_add(value)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            reduced = wave_reduce_add(value)
            if lane == 0:
                fx.memref_store(reduced, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < red_slots
                safe_lane = in_range.select(lane, 0)
                wave_value = in_range.select(fx.memref_load(s_red, safe_lane), zero)
                wave_value = wave_reduce_add(wave_value)
                if lane == 0:
                    fx.memref_store(wave_value, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        input_buffer = fx.rocdl.make_buffer_tensor(input_tensor)
        gamma_buffer = fx.rocdl.make_buffer_tensor(gamma)
        dy_buffer = fx.rocdl.make_buffer_tensor(dy)
        rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
        dx_buffer = fx.rocdl.make_buffer_tensor(dx)
        partial_buffer = fx.rocdl.make_buffer_tensor(dweight_partial)
        rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
        partial_div = fx.logical_divide(partial_buffer, fx.make_layout(1, 1))

        copy_atom = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy128b()
                if use_vec
                else (fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b())
            ),
            elem_bits,
        )
        gamma_copy_atom = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy128b()
                if use_vec
                else (
                    fx.rocdl.BufferCopy16b() if weight_elem_bits <= 16 else fx.rocdl.BufferCopy32b()
                )
            ),
            weight_elem_bits,
        )
        f32_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        gamma_div = fx.logical_divide(
            gamma_buffer,
            fx.make_layout(weight_io_width, 1),
        )

        gamma_local = []
        if const_expr(use_vec):
            for tile_i in range_constexpr(num_io_iters):
                io_index = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_index < num_io_tiles
                safe_index = is_valid.select(io_index, 0)
                gamma_local.append(
                    load_weight_vec(
                        gamma_copy_atom,
                        weight_dtype_str,
                        weight_elem_dtype,
                        gamma_div,
                        safe_index,
                    )
                )

        accumulated_dweight = fx.Vector.filled(
            partial_acc_size,
            0.0,
            fx.Float32,
        )
        for row in range(fx.Int32(block), m, num_programs):
            input_div = fx.logical_divide(
                fx.slice(input_buffer, (row, None)),
                fx.make_layout(io_width, 1),
            )
            dy_div = fx.logical_divide(
                fx.slice(dy_buffer, (row, None)),
                fx.make_layout(io_width, 1),
            )
            dx_div = fx.logical_divide(
                fx.slice(dx_buffer, (row, None)),
                fx.make_layout(io_width, 1),
            )
            rstd = load_scalar(f32_copy_atom, fx.Float32, rstd_div, row)

            thread_acc = zero
            input_local = []
            dy_local = []
            for tile_i in range_constexpr(num_io_iters):
                io_index = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_index < num_io_tiles
                safe_index = is_valid.select(io_index, 0)
                if const_expr(use_vec):
                    x_elem = load_vec(
                        copy_atom,
                        io_width,
                        elem_dtype,
                        input_div,
                        safe_index,
                    )
                    dy_elem = load_vec(
                        copy_atom,
                        io_width,
                        elem_dtype,
                        dy_div,
                        safe_index,
                    )
                    input_local.append(x_elem)
                    dy_local.append(dy_elem)
                    gamma_elem = gamma_local[tile_i]
                else:
                    x_elem = load_scalar(
                        copy_atom,
                        elem_dtype,
                        input_div,
                        safe_index,
                    )
                    dy_elem = load_scalar(
                        copy_atom,
                        elem_dtype,
                        dy_div,
                        safe_index,
                    )
                    gamma_elem = load_scalar(
                        gamma_copy_atom,
                        weight_elem_dtype,
                        gamma_div,
                        safe_index,
                    )

                x_value = x_elem if dtype_str == "f32" else x_elem.to(fx.Float32)
                dy_value = dy_elem if dtype_str == "f32" else dy_elem.to(fx.Float32)
                gamma_value = (
                    gamma_elem
                    if use_vec or weight_dtype_str == "f32"
                    else gamma_elem.to(fx.Float32)
                )
                product = x_value * rstd * dy_value * gamma_value
                if const_expr(use_vec):
                    product = product.reduce(ReductionOp.ADD, fastmath=fast_math)
                thread_acc = thread_acc + is_valid.select(product, zero)

            correction = block_reduce_add(thread_acc) / float(n)
            row_dweight = []
            for tile_i in range_constexpr(num_io_iters):
                io_index = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_index < num_io_tiles
                safe_index = is_valid.select(io_index, 0)
                if const_expr(use_vec):
                    x_elem = input_local[tile_i]
                    dy_elem = dy_local[tile_i]
                    gamma_elem = gamma_local[tile_i]
                else:
                    x_elem = load_scalar(
                        copy_atom,
                        elem_dtype,
                        input_div,
                        safe_index,
                    )
                    dy_elem = load_scalar(
                        copy_atom,
                        elem_dtype,
                        dy_div,
                        safe_index,
                    )
                    gamma_elem = load_scalar(
                        gamma_copy_atom,
                        weight_elem_dtype,
                        gamma_div,
                        safe_index,
                    )

                x_value = x_elem if dtype_str == "f32" else x_elem.to(fx.Float32)
                dy_value = dy_elem if dtype_str == "f32" else dy_elem.to(fx.Float32)
                gamma_value = (
                    gamma_elem
                    if use_vec or weight_dtype_str == "f32"
                    else gamma_elem.to(fx.Float32)
                )
                x_hat = x_value * rstd
                dx_value = (dy_value * gamma_value - x_hat * correction) * rstd
                if io_index < num_io_tiles:
                    if const_expr(use_vec):
                        dx_elem = to_elem_vec(
                            dtype_str,
                            elem_dtype,
                            use_hw_cvt_bf16,
                            dx_value,
                        )
                        store_vec(
                            copy_atom,
                            io_width,
                            elem_dtype,
                            dx_elem,
                            dx_div,
                            io_index,
                        )
                    else:
                        dx_elem = dx_value if dtype_str == "f32" else dx_value.to(elem_dtype)
                        store_scalar(
                            copy_atom,
                            elem_dtype,
                            elem_dtype,
                            dx_div,
                            io_index,
                            dx_elem,
                        )

                dweight_value = dy_value * x_hat
                if const_expr(use_vec):
                    for lane in range_constexpr(io_width):
                        row_dweight.append(is_valid.select(dweight_value[lane], zero))
                else:
                    row_dweight.append(is_valid.select(dweight_value, zero))

            accumulated_dweight = accumulated_dweight + fx.Vector.from_elements(
                row_dweight,
                fx.Float32,
            )
            gpu.barrier()

        for tile_i in range_constexpr(num_io_iters):
            io_index = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
            if io_index < num_io_tiles:
                for lane in range_constexpr(io_width):
                    column = io_index * io_width + lane
                    partial_index = block * n + column
                    store_scalar(
                        f32_copy_atom,
                        fx.Float32,
                        fx.Float32,
                        partial_div,
                        partial_index,
                        accumulated_dweight[tile_i * io_width + lane],
                    )

    @flyc.kernel
    def rmsnorm_bwd_dweight_reduce_kernel(
        dweight_partial: fx.Tensor,
        dweight: fx.Tensor,
    ):
        block = fx.block_idx.x
        tid = fx.thread_idx.x
        column_lane = tid % DWEIGHT_REDUCE_COLS
        partial_lane = tid // DWEIGHT_REDUCE_COLS
        column = block * DWEIGHT_REDUCE_COLS + column_lane
        is_valid = column < n
        safe_column = is_valid.select(column, 0)

        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        partial_buffer = fx.rocdl.make_buffer_tensor(dweight_partial)
        dweight_buffer = fx.rocdl.make_buffer_tensor(dweight)
        partial_div = fx.logical_divide(partial_buffer, fx.make_layout(1, 1))
        dweight_div = fx.logical_divide(dweight_buffer, fx.make_layout(1, 1))
        f32_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        weight_copy_atom = fx.make_copy_atom(
            (fx.rocdl.BufferCopy16b() if weight_elem_bits <= 16 else fx.rocdl.BufferCopy32b()),
            weight_elem_bits,
        )

        storage = fx.SharedAllocator().allocate(dweight_reduce_storage).peek()
        shared_partial = storage.s_red.view(fx.make_layout(DWEIGHT_REDUCE_THREADS, 1))

        accumulator = fx.Float32(0.0)
        for partial_base in range(
            0,
            num_programs,
            DWEIGHT_REDUCE_ROW_LANES,
        ):
            partial_row = partial_base + partial_lane
            partial_valid = partial_row < num_programs
            safe_row = partial_valid.select(partial_row, 0)
            partial_index = safe_row * n + safe_column
            value = load_scalar(
                f32_copy_atom,
                fx.Float32,
                partial_div,
                partial_index,
            )
            accumulator = accumulator + partial_valid.select(
                value,
                fx.Float32(0.0),
            )
        fx.memref_store(accumulator, shared_partial, tid)
        gpu.barrier()

        if partial_lane == 0:
            if column < n:
                total = fx.Float32(0.0)
                for lane in range_constexpr(DWEIGHT_REDUCE_ROW_LANES):
                    total = total + fx.memref_load(
                        shared_partial,
                        lane * DWEIGHT_REDUCE_COLS + column_lane,
                    )
                output = total if weight_dtype_str == "f32" else total.to(weight_elem_dtype)
                store_scalar(
                    weight_copy_atom,
                    weight_elem_dtype,
                    weight_elem_dtype,
                    dweight_div,
                    column,
                    output,
                )

    reduce_grid = (n + DWEIGHT_REDUCE_COLS - 1) // DWEIGHT_REDUCE_COLS

    @flyc.jit
    def launch_rmsnorm_bwd_two_stage(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        dy: fx.Tensor,
        rstd_tensor: fx.Tensor,
        dx: fx.Tensor,
        dweight: fx.Tensor,
        dweight_partial: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        partial_launcher = rmsnorm_bwd_partial_kernel(
            input_tensor,
            gamma,
            dy,
            rstd_tensor,
            dx,
            dweight_partial,
            m,
        )
        partial_launcher.launch(
            grid=(num_programs, 1, 1),
            block=(TWO_STAGE_PARTIAL_THREADS, 1, 1),
            stream=stream,
        )
        reduce_launcher = rmsnorm_bwd_dweight_reduce_kernel(
            dweight_partial,
            dweight,
        )
        reduce_launcher.launch(
            grid=(reduce_grid, 1, 1),
            block=(DWEIGHT_REDUCE_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_bwd_two_stage

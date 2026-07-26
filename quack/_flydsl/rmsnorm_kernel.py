# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Plain RMSNorm forward kernel builders."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch

from .kernel_utils import dtype_to_elem_bits, dtype_to_elem_type
from .rmsnorm_common import (
    EPS,
    WARP_SIZE,
    load_scalar,
    load_vec,
    load_weight_vec,
    make_reduction_storage,
    resolve_rmsnorm_weight_dtype,
    store_scalar,
    store_vec,
    to_elem_scalar,
    to_elem_vec,
    weight_vec_width,
)
from .rmsnorm_tiling import select_row_tiling, use_multi_row_kernel


def build_rmsnorm_module(
    n: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    weight_dtype_str: str | None = None,
):
    """Build a plain RMSNorm launcher specialized by hidden size and dtypes."""
    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    elem_bits = dtype_to_elem_bits(dtype_str)
    if use_multi_row_kernel(n, elem_bits):
        return _build_rmsnorm_small_n_module(
            n,
            dtype_str,
            store_rstd,
            eps,
            weight_dtype_str,
        )

    arch = get_rocm_arch()
    use_hw_cvt_bf16 = arch == "gfx950" or str(arch).startswith("gfx95")
    tiling = select_row_tiling(n, elem_bits)
    block_threads = tiling.block_threads
    red_slots = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)
    weight_elem_bits = dtype_to_elem_bits(weight_dtype_str)
    shared_storage = make_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_kernel(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        rstd_tensor: fx.Tensor,
        output: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fast_math = arith.FastMathFlags.fast

        storage = fx.SharedAllocator().allocate(shared_storage).peek()
        s_red = storage.s_red.view(fx.make_layout(red_slots, 1))
        s_red2 = storage.s_red2.view(fx.make_layout(red_slots, 1))

        if const_expr(store_rstd):
            rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
            rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
            rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(value):
            result = value
            for shift_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                offset = WARP_SIZE // (2 << shift_exp)
                peer = result.shuffle_xor(offset, WARP_SIZE)
                result = result.addf(peer, fastmath=fast_math)
            return result

        def block_reduce_add(value):
            reduced, _ = block_reduce_add2(value, fx.Float32(0.0))
            return reduced

        def block_reduce_add2(value0, value1):
            if const_expr(red_slots == 1):
                return wave_reduce_add(value0), wave_reduce_add(value1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            wave0 = wave_reduce_add(value0)
            wave1 = wave_reduce_add(value1)
            if lane == 0:
                fx.memref_store(wave0, s_red, wave)
                fx.memref_store(wave1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < red_slots
                lane_safe = in_range.select(lane, 0)
                reduced0 = in_range.select(fx.memref_load(s_red, lane_safe), 0.0)
                reduced1 = in_range.select(fx.memref_load(s_red2, lane_safe), 0.0)
                reduced0 = wave_reduce_add(reduced0)
                reduced1 = wave_reduce_add(reduced1)
                if lane == 0:
                    fx.memref_store(reduced0, s_red, 0)
                    fx.memref_store(reduced1, s_red2, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        input_buffer = fx.rocdl.make_buffer_tensor(input_tensor)
        output_buffer = fx.rocdl.make_buffer_tensor(output)
        gamma_buffer = fx.rocdl.make_buffer_tensor(gamma)
        row_input = fx.slice(input_buffer, (row, None))
        row_output = fx.slice(output_buffer, (row, None))

        if const_expr(tiling.vectorized):
            vec_width = tiling.vec_width
            num_vecs = tiling.num_vecs
            last_tile = tiling.num_tiles - 1
            input_div = fx.logical_divide(row_input, fx.make_layout(vec_width, 1))
            output_div = fx.logical_divide(row_output, fx.make_layout(vec_width, 1))
            gamma_div = fx.logical_divide(
                gamma_buffer,
                fx.make_layout(weight_vec_width(weight_elem_bits), 1),
            )
            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            gamma_copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(),
                weight_elem_bits,
            )

            thread_sumsq = fx.Float32(0.0)
            input_local = []
            for tile_i in range_constexpr(tiling.num_tiles):
                # Only the final tile can run off the end of the row.
                partial = tiling.needs_predicate and tile_i == last_tile
                index = tid + tile_i * block_threads
                if const_expr(partial):
                    in_row = index < num_vecs
                    index = in_row.select(index, 0)
                vector = load_vec(copy_atom, vec_width, elem_dtype, input_div, index)
                input_local.append(vector)
                values = vector.to(fx.Float32)
                contribution = (values * values).reduce(
                    ReductionOp.ADD,
                    fastmath=fast_math,
                )
                if const_expr(partial):
                    contribution = in_row.select(contribution, fx.Float32(0.0))
                thread_sumsq = thread_sumsq + contribution

            _, sum_sq = block_reduce_add2(fx.Float32(0.0), thread_sumsq)
            rrms = fmath.rsqrt(sum_sq / float(n) + eps, fastmath=fast_math)

            if const_expr(store_rstd):
                if tid == 0:
                    store_scalar(
                        rstd_copy_atom,
                        fx.Float32,
                        fx.Float32,
                        rstd_div,
                        row,
                        rrms,
                    )

            for tile_i in range_constexpr(tiling.num_tiles):
                partial = tiling.needs_predicate and tile_i == last_tile
                index = tid + tile_i * block_threads
                safe_index = index
                if const_expr(partial):
                    in_row = index < num_vecs
                    safe_index = in_row.select(index, 0)
                weights = load_weight_vec(
                    gamma_copy_atom,
                    weight_elem_dtype,
                    weight_elem_bits,
                    gamma_div,
                    safe_index,
                    vec_width,
                )
                values = input_local[tile_i].to(fx.Float32)
                result = to_elem_vec(
                    dtype_str,
                    elem_dtype,
                    use_hw_cvt_bf16,
                    values * rrms * weights,
                )
                if const_expr(partial):
                    if in_row:
                        store_vec(copy_atom, vec_width, elem_dtype, result, output_div, index)
                else:
                    store_vec(copy_atom, vec_width, elem_dtype, result, output_div, index)
        else:
            copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            gamma_copy_atom = fx.make_copy_atom(
                (fx.rocdl.BufferCopy16b() if weight_elem_bits <= 16 else fx.rocdl.BufferCopy32b()),
                weight_elem_bits,
            )
            input_div = fx.logical_divide(row_input, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(gamma_buffer, fx.make_layout(1, 1))
            output_div = fx.logical_divide(row_output, fx.make_layout(1, 1))

            thread_sumsq = fx.Float32(0.0)
            for base in range_constexpr(0, n, block_threads):
                index = tid + base
                is_valid = index < n
                safe_index = is_valid.select(index, 0)
                value_elem = load_scalar(copy_atom, elem_dtype, input_div, safe_index)
                value = value_elem if dtype_str == "f32" else value_elem.to(fx.Float32)
                thread_sumsq = thread_sumsq + is_valid.select(
                    value * value,
                    fx.Float32(0.0),
                )

            sum_sq = block_reduce_add(thread_sumsq)
            rrms = fmath.rsqrt(sum_sq / float(n) + eps, fastmath=fast_math)

            if const_expr(store_rstd):
                if tid == 0:
                    store_scalar(
                        rstd_copy_atom,
                        fx.Float32,
                        fx.Float32,
                        rstd_div,
                        row,
                        rrms,
                    )

            for base in range_constexpr(0, n, block_threads):
                index = tid + base
                if index < n:
                    value_elem = load_scalar(copy_atom, elem_dtype, input_div, index)
                    weight_elem = load_scalar(
                        gamma_copy_atom,
                        weight_elem_dtype,
                        gamma_div,
                        index,
                    )
                    value = value_elem if dtype_str == "f32" else value_elem.to(fx.Float32)
                    weight = (
                        weight_elem if weight_dtype_str == "f32" else weight_elem.to(fx.Float32)
                    )
                    result = to_elem_scalar(
                        dtype_str,
                        elem_dtype,
                        value * rrms * weight,
                    )
                    store_scalar(
                        copy_atom,
                        elem_dtype,
                        elem_dtype,
                        output_div,
                        index,
                        result,
                    )

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm(
            input_tensor: fx.Tensor,
            gamma: fx.Tensor,
            output: fx.Tensor,
            rstd_tensor: fx.Tensor,
            m: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_kernel(input_tensor, gamma, rstd_tensor, output)
            launcher.launch(
                grid=(m, 1, 1),
                block=(block_threads, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm

    @flyc.jit
    def launch_rmsnorm(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        output: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_kernel(input_tensor, gamma, gamma, output)
        launcher.launch(
            grid=(m, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _build_rmsnorm_small_n_module(
    n: int,
    dtype_str: str,
    store_rstd: bool,
    eps: float,
    weight_dtype_str: str,
):
    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    block_n = 1 << (n - 1).bit_length()
    block_m = max(min(16384 // block_n, 32), 8)
    threads_per_row = min(WARP_SIZE, 1024 // block_m)
    block_threads = block_m * threads_per_row
    elem_bits = dtype_to_elem_bits(dtype_str)
    weight_elem_bits = dtype_to_elem_bits(weight_dtype_str)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rmsnorm_small_n_kernel(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        rstd_tensor: fx.Tensor,
        output: fx.Tensor,
        m: fx.Int32,
    ):
        block = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % threads_per_row
        row_local = tid // threads_per_row
        row = block * fx.Int32(block_m) + row_local

        if row < m:
            elem_dtype = dtype_to_elem_type(dtype_str)
            weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
            fast_math = arith.FastMathFlags.fast

            input_buffer = fx.rocdl.make_buffer_tensor(input_tensor)
            gamma_buffer = fx.rocdl.make_buffer_tensor(gamma)
            output_buffer = fx.rocdl.make_buffer_tensor(output)
            if const_expr(store_rstd):
                rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
                rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
                rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

            input_div = fx.logical_divide(
                fx.slice(input_buffer, (row, None)),
                fx.make_layout(1, 1),
            )
            gamma_div = fx.logical_divide(gamma_buffer, fx.make_layout(1, 1))
            output_div = fx.logical_divide(
                fx.slice(output_buffer, (row, None)),
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

            def group_reduce_add(value):
                result = value
                for shift_exp in range_constexpr(int(math.log2(threads_per_row))):
                    offset = threads_per_row // (2 << shift_exp)
                    peer = result.shuffle_xor(offset, fx.Int32(threads_per_row))
                    result = result.addf(peer, fastmath=fast_math)
                return result

            thread_sumsq = fx.Float32(0.0)
            for base in range_constexpr(0, block_n, threads_per_row):
                index = lane + base
                is_valid = index < n
                safe_index = is_valid.select(index, 0)
                value_elem = load_scalar(copy_atom, elem_dtype, input_div, safe_index)
                value = value_elem if dtype_str == "f32" else value_elem.to(fx.Float32)
                thread_sumsq = thread_sumsq + is_valid.select(
                    value * value,
                    fx.Float32(0.0),
                )

            rrms = fmath.rsqrt(
                group_reduce_add(thread_sumsq) / float(n) + eps,
                fastmath=fast_math,
            )
            if const_expr(store_rstd):
                if lane == 0:
                    store_scalar(
                        rstd_copy_atom,
                        fx.Float32,
                        fx.Float32,
                        rstd_div,
                        row,
                        rrms,
                    )

            for base in range_constexpr(0, block_n, threads_per_row):
                index = lane + base
                if index < n:
                    value_elem = load_scalar(copy_atom, elem_dtype, input_div, index)
                    weight_elem = load_scalar(
                        gamma_copy_atom,
                        weight_elem_dtype,
                        gamma_div,
                        index,
                    )
                    value = value_elem if dtype_str == "f32" else value_elem.to(fx.Float32)
                    weight = (
                        weight_elem if weight_dtype_str == "f32" else weight_elem.to(fx.Float32)
                    )
                    result = to_elem_scalar(
                        dtype_str,
                        elem_dtype,
                        value * rrms * weight,
                    )
                    store_scalar(
                        copy_atom,
                        elem_dtype,
                        elem_dtype,
                        output_div,
                        index,
                        result,
                    )

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm_small_n(
            input_tensor: fx.Tensor,
            gamma: fx.Tensor,
            output: fx.Tensor,
            rstd_tensor: fx.Tensor,
            m: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_small_n_kernel(
                input_tensor,
                gamma,
                rstd_tensor,
                output,
                m,
            )
            launcher.launch(
                grid=((m + fx.Int32(block_m - 1)) // fx.Int32(block_m), 1, 1),
                block=(block_threads, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_small_n

    @flyc.jit
    def launch_rmsnorm_small_n(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        output: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_small_n_kernel(input_tensor, gamma, gamma, output, m)
        launcher.launch(
            grid=((m + fx.Int32(block_m - 1)) // fx.Int32(block_m), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_small_n

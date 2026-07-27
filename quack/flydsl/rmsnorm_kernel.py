# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Optimized plain and feature-complete RMSNorm forward builders."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch

from .kernel_utils import dtype_to_elem_bits, dtype_to_elem_type, has_hw_bf16_convert
from .rmsnorm_common import (
    WARP_SIZE,
    assert_arch_matches_reductions,
    buffer_copy_atom,
    load_scalar,
    load_vec,
    load_weight_vec,
    make_reduction_storage,
    resolve_rmsnorm_weight_dtype,
    row_buffer,
    row_head_buffer,
    store_scalar,
    store_vec,
    to_elem_scalar,
    to_elem_vec,
    weight_access_plan,
)
from .rmsnorm_config import RmsNormRowConfig, use_multi_row_kernel


def build_rmsnorm_module(
    n: int,
    dtype_str: str,
    store_rstd: bool = False,
    weight_dtype_str: str | None = None,
    arch: str | None = None,
):
    """Build a plain RMSNorm launcher specialized by hidden size and dtypes.

    ``arch`` is the architecture the caller has already validated FlyDSL will
    compile for. It defaults to autodetection, but the adapter always passes
    the validated value so kernel codegen cannot disagree with the target.
    """
    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    arch = get_rocm_arch() if arch is None else arch
    assert_arch_matches_reductions(arch)
    elem_bits = dtype_to_elem_bits(dtype_str)
    if use_multi_row_kernel(n, elem_bits):
        return _build_rmsnorm_small_n_module(
            n,
            dtype_str,
            store_rstd,
            weight_dtype_str,
        )

    use_hw_cvt_bf16 = has_hw_bf16_convert(arch)
    config = RmsNormRowConfig.from_analytical_heuristic(n, elem_bits)
    block_threads = config.num_threads
    red_slots = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)
    weight_elem_bits = dtype_to_elem_bits(weight_dtype_str)
    shared_storage = make_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_kernel(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        rstd_tensor: fx.Tensor,
        output: fx.Tensor,
        eps: fx.Float32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fast_math = arith.FastMathFlags.fast

        storage = fx.SharedAllocator().allocate(shared_storage).peek()
        s_red = storage.s_red.view(fx.make_layout(red_slots, 1))

        if const_expr(store_rstd):
            rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
            rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
            rstd_copy_atom = buffer_copy_atom(32, 32)

        def wave_reduce_add(value):
            result = value
            for shift_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                offset = WARP_SIZE // (2 << shift_exp)
                peer = result.shuffle_xor(offset, WARP_SIZE)
                result = result.addf(peer, fastmath=fast_math)
            return result

        # Inline rather than shared: FlyDSL rewrites the AST of the decorated
        # kernel only, so a helper holding `if lane == 0` would be traced as a
        # plain Python conditional and fail.
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
                partial = in_range.select(fx.memref_load(s_red, safe_lane), fx.Float32(0.0))
                partial = wave_reduce_add(partial)
                if lane == 0:
                    fx.memref_store(partial, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        row_input = row_buffer(input_tensor, row, elem_bits, n)
        row_output = row_buffer(output, row, elem_bits, n)
        gamma_buffer = fx.rocdl.make_buffer_tensor(gamma)

        if const_expr(config.vectorized):
            vecsize = config.vecsize
            num_vecs = config.num_vecs
            last_tile = config.num_tiles - 1
            weight_accesses, weight_per_access = weight_access_plan(vecsize, weight_elem_bits)
            input_div = fx.logical_divide(row_input, fx.make_layout(vecsize, 1))
            output_div = fx.logical_divide(row_output, fx.make_layout(vecsize, 1))
            gamma_div = fx.logical_divide(gamma_buffer, fx.make_layout(weight_per_access, 1))
            copy_atom = buffer_copy_atom(config.access_bits, elem_bits)
            gamma_copy_atom = buffer_copy_atom(
                weight_per_access * weight_elem_bits,
                weight_elem_bits,
            )

            thread_sumsq = fx.Float32(0.0)
            input_local = []
            for tile_i in range_constexpr(config.num_tiles):
                # Only the final tile can run off the end of the row.
                partial = config.needs_predicate and tile_i == last_tile
                index = tid + tile_i * block_threads
                if const_expr(partial):
                    in_row = index < num_vecs
                    index = in_row.select(index, 0)
                vector = load_vec(copy_atom, vecsize, elem_dtype, input_div, index)
                input_local.append(vector)
                values = vector.to(fx.Float32)
                contribution = (values * values).reduce(
                    ReductionOp.ADD,
                    fastmath=fast_math,
                )
                if const_expr(partial):
                    contribution = in_row.select(contribution, fx.Float32(0.0))
                thread_sumsq = thread_sumsq + contribution

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

            for tile_i in range_constexpr(config.num_tiles):
                partial = config.needs_predicate and tile_i == last_tile
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
                    vecsize,
                )
                values = input_local[tile_i].to(fx.Float32)
                result = to_elem_vec(
                    dtype_str,
                    elem_dtype,
                    use_hw_cvt_bf16,
                    values * rrms * weights,
                    vecsize,
                )
                if const_expr(partial):
                    if in_row:
                        store_vec(copy_atom, vecsize, elem_dtype, result, output_div, index)
                else:
                    store_vec(copy_atom, vecsize, elem_dtype, result, output_div, index)
        else:
            copy_atom = buffer_copy_atom(elem_bits, elem_bits)
            gamma_copy_atom = buffer_copy_atom(weight_elem_bits, weight_elem_bits)
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
            eps: fx.Float32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_kernel(input_tensor, gamma, rstd_tensor, output, eps)
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
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_kernel(input_tensor, gamma, gamma, output, eps)
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
        eps: fx.Float32,
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

            gamma_buffer = fx.rocdl.make_buffer_tensor(gamma)
            if const_expr(store_rstd):
                rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
                rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
                rstd_copy_atom = buffer_copy_atom(32, 32)

            input_div = fx.logical_divide(
                row_buffer(input_tensor, row, elem_bits, n),
                fx.make_layout(1, 1),
            )
            gamma_div = fx.logical_divide(gamma_buffer, fx.make_layout(1, 1))
            output_div = fx.logical_divide(
                row_buffer(output, row, elem_bits, n),
                fx.make_layout(1, 1),
            )
            copy_atom = buffer_copy_atom(elem_bits, elem_bits)
            gamma_copy_atom = buffer_copy_atom(weight_elem_bits, weight_elem_bits)

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
            eps: fx.Float32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_small_n_kernel(
                input_tensor,
                gamma,
                rstd_tensor,
                output,
                m,
                eps,
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
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_small_n_kernel(input_tensor, gamma, gamma, output, m, eps)
        launcher.launch(
            grid=((m + fx.Int32(block_m - 1)) // fx.Int32(block_m), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_small_n


def build_rmsnorm_feature_module(
    n: int,
    input_dtype_str: str,
    output_dtype_str: str,
    *,
    weight_dtype_str: str,
    bias_dtype_str: str,
    residual_dtype_str: str,
    residual_out_dtype_str: str,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    per_head: bool,
    num_heads: int,
    arch: str | None = None,
):
    """Build the feature-complete scalar RMSNorm forward path.

    The optimized plain-weighted builders above remain unchanged. This path
    owns optional affine inputs, fused residual addition, independent output
    dtypes, and per-head parameter addressing.
    """
    arch = get_rocm_arch() if arch is None else arch
    assert_arch_matches_reductions(arch)
    input_bits = dtype_to_elem_bits(input_dtype_str)
    output_bits = dtype_to_elem_bits(output_dtype_str)
    weight_bits = dtype_to_elem_bits(weight_dtype_str)
    bias_bits = dtype_to_elem_bits(bias_dtype_str)
    residual_bits = dtype_to_elem_bits(residual_dtype_str)
    residual_out_bits = dtype_to_elem_bits(residual_out_dtype_str)
    config = RmsNormRowConfig.from_analytical_heuristic(n, input_bits)
    block_threads = config.num_threads
    red_slots = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)
    shared_storage = make_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_feature_kernel(
        input_tensor: fx.Tensor,
        weight_tensor: fx.Tensor,
        bias_tensor: fx.Tensor,
        residual_tensor: fx.Tensor,
        output_tensor: fx.Tensor,
        residual_out_tensor: fx.Tensor,
        rstd_tensor: fx.Tensor,
        eps: fx.Float32,
        weight_offset: fx.Float32,
    ):
        program = fx.block_idx.x
        tid = fx.thread_idx.x
        row = program // fx.Int32(num_heads) if per_head else program
        head = program % fx.Int32(num_heads) if per_head else fx.Int32(0)

        input_dtype = dtype_to_elem_type(input_dtype_str)
        output_dtype = dtype_to_elem_type(output_dtype_str)
        weight_dtype = dtype_to_elem_type(weight_dtype_str)
        bias_dtype = dtype_to_elem_type(bias_dtype_str)
        residual_dtype = dtype_to_elem_type(residual_dtype_str)
        residual_out_dtype = dtype_to_elem_type(residual_out_dtype_str)
        fast_math = arith.FastMathFlags.fast

        storage = fx.SharedAllocator().allocate(shared_storage).peek()
        reduction = storage.s_red.view(fx.make_layout(red_slots, 1))

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
                fx.memref_store(reduced, reduction, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < red_slots
                safe_lane = in_range.select(lane, 0)
                partial = in_range.select(
                    fx.memref_load(reduction, safe_lane),
                    fx.Float32(0.0),
                )
                partial = wave_reduce_add(partial)
                if lane == 0:
                    fx.memref_store(partial, reduction, 0)
            gpu.barrier()
            return fx.memref_load(reduction, 0)

        input_row = (
            row_head_buffer(input_tensor, row, head, input_bits, n)
            if per_head
            else row_buffer(input_tensor, row, input_bits, n)
        )
        output_row = (
            row_head_buffer(output_tensor, row, head, output_bits, n)
            if per_head
            else row_buffer(output_tensor, row, output_bits, n)
        )
        input_div = fx.logical_divide(input_row, fx.make_layout(1, 1))
        output_div = fx.logical_divide(output_row, fx.make_layout(1, 1))
        if const_expr(has_residual):
            residual_row = (
                row_head_buffer(residual_tensor, row, head, residual_bits, n)
                if per_head
                else row_buffer(residual_tensor, row, residual_bits, n)
            )
            residual_div = fx.logical_divide(residual_row, fx.make_layout(1, 1))
        if const_expr(store_residual):
            residual_out_row = (
                row_head_buffer(
                    residual_out_tensor,
                    row,
                    head,
                    residual_out_bits,
                    n,
                )
                if per_head
                else row_buffer(
                    residual_out_tensor,
                    row,
                    residual_out_bits,
                    n,
                )
            )
            residual_out_div = fx.logical_divide(
                residual_out_row,
                fx.make_layout(1, 1),
            )

        weight_row = (
            row_buffer(weight_tensor, head, weight_bits, n)
            if per_head
            else fx.rocdl.make_buffer_tensor(weight_tensor)
        )
        bias_row = (
            row_buffer(bias_tensor, head, bias_bits, n)
            if per_head
            else fx.rocdl.make_buffer_tensor(bias_tensor)
        )
        weight_div = fx.logical_divide(weight_row, fx.make_layout(1, 1))
        bias_div = fx.logical_divide(bias_row, fx.make_layout(1, 1))

        if const_expr(store_rstd):
            rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
            rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))

        input_copy = buffer_copy_atom(input_bits, input_bits)
        output_copy = buffer_copy_atom(output_bits, output_bits)
        weight_copy = buffer_copy_atom(weight_bits, weight_bits)
        bias_copy = buffer_copy_atom(bias_bits, bias_bits)
        if const_expr(has_residual):
            residual_copy = buffer_copy_atom(residual_bits, residual_bits)
        if const_expr(store_residual):
            residual_out_copy = buffer_copy_atom(
                residual_out_bits,
                residual_out_bits,
            )
        if const_expr(store_rstd):
            f32_copy = buffer_copy_atom(32, 32)

        thread_sumsq = fx.Float32(0.0)
        for base in range_constexpr(0, n, block_threads):
            index = tid + base
            valid = index < n
            safe_index = valid.select(index, 0)
            input_elem = load_scalar(input_copy, input_dtype, input_div, safe_index)
            value = input_elem if input_dtype_str == "f32" else input_elem.to(fx.Float32)
            if const_expr(has_residual):
                residual_elem = load_scalar(
                    residual_copy,
                    residual_dtype,
                    residual_div,
                    safe_index,
                )
                residual_value = (
                    residual_elem if residual_dtype_str == "f32" else residual_elem.to(fx.Float32)
                )
                value = value + residual_value
            if const_expr(store_residual):
                if index < n:
                    residual_out_value = to_elem_scalar(
                        residual_out_dtype_str,
                        residual_out_dtype,
                        value,
                    )
                    store_scalar(
                        residual_out_copy,
                        residual_out_dtype,
                        residual_out_dtype,
                        residual_out_div,
                        index,
                        residual_out_value,
                    )
            thread_sumsq = thread_sumsq + valid.select(
                value * value,
                fx.Float32(0.0),
            )

        sum_sq = block_reduce_add(thread_sumsq)
        rrms = fmath.rsqrt(sum_sq / float(n) + eps, fastmath=fast_math)
        if const_expr(store_rstd):
            if tid == 0:
                store_scalar(
                    f32_copy,
                    fx.Float32,
                    fx.Float32,
                    rstd_div,
                    program,
                    rrms,
                )

        for base in range_constexpr(0, n, block_threads):
            index = tid + base
            if index < n:
                input_elem = load_scalar(input_copy, input_dtype, input_div, index)
                value = input_elem if input_dtype_str == "f32" else input_elem.to(fx.Float32)
                if const_expr(has_residual):
                    residual_elem = load_scalar(
                        residual_copy,
                        residual_dtype,
                        residual_div,
                        index,
                    )
                    residual_value = (
                        residual_elem
                        if residual_dtype_str == "f32"
                        else residual_elem.to(fx.Float32)
                    )
                    value = value + residual_value
                result = value * rrms
                if const_expr(has_weight):
                    weight_elem = load_scalar(
                        weight_copy,
                        weight_dtype,
                        weight_div,
                        index,
                    )
                    weight_value = (
                        weight_elem if weight_dtype_str == "f32" else weight_elem.to(fx.Float32)
                    )
                    result = result * (weight_value + weight_offset)
                if const_expr(has_bias):
                    bias_elem = load_scalar(
                        bias_copy,
                        bias_dtype,
                        bias_div,
                        index,
                    )
                    bias_value = bias_elem if bias_dtype_str == "f32" else bias_elem.to(fx.Float32)
                    result = result + bias_value
                output_value = to_elem_scalar(
                    output_dtype_str,
                    output_dtype,
                    result,
                )
                store_scalar(
                    output_copy,
                    output_dtype,
                    output_dtype,
                    output_div,
                    index,
                    output_value,
                )

    @flyc.jit
    def launch_rmsnorm_feature(
        input_tensor: fx.Tensor,
        weight_tensor: fx.Tensor,
        bias_tensor: fx.Tensor,
        residual_tensor: fx.Tensor,
        output_tensor: fx.Tensor,
        residual_out_tensor: fx.Tensor,
        rstd_tensor: fx.Tensor,
        m: fx.Int32,
        eps: fx.Float32,
        weight_offset: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        rmsnorm_feature_kernel(
            input_tensor,
            weight_tensor,
            bias_tensor,
            residual_tensor,
            output_tensor,
            residual_out_tensor,
            rstd_tensor,
            eps,
            weight_offset,
        ).launch(
            grid=(m * fx.Int32(num_heads), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_feature

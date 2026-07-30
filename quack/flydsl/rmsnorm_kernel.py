# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Optimized plain and feature-complete RMSNorm forward builders."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch

from .rmsnorm_common import (
    WARP_SIZE,
    buffer_copy_atom,
    dtype_to_elem_bits,
    dtype_to_elem_type,
    has_hw_bf16_convert,
    load_dtype_vec,
    load_vec,
    make_reduction_storage,
    require_wave64,
    resolve_rmsnorm_weight_dtype,
    row_buffer,
    row_head_buffer,
    shuffle_reduce_add,
    store_dtype_vec,
    store_scalar,
    store_vec,
    to_store_dtype,
    vector_access_plan,
)
from .rmsnorm_config import (
    RmsNormRowConfig,
    batch_feature_rows,
    multi_row_block_rows,
    use_multi_row_kernel,
)


def build_rmsnorm_module(
    n: int,
    dtype_str: str,
    store_rstd: bool = False,
    weight_dtype_str: str | None = None,
    arch: str | None = None,
):
    """Build a plain RMSNorm launcher specialized by hidden size and dtypes.

    One kernel covers both geometries. A row wide enough to fill a block gets a
    block; a row too short to fill one gets a group of lanes and shares the
    block with its neighbours. One block per row is then just the case where the
    group is the block, which is why the reduction only reaches for LDS when the
    group spans more than one wavefront -- that can only happen when the row
    owns the block. ``vecsize == 1`` falls out of the same body, so a row that
    cannot vectorize is a narrower access rather than a second kernel.

    ``arch`` is the architecture the caller has already validated FlyDSL will
    compile for. It defaults to autodetection, but the adapter always passes
    the validated value so kernel codegen cannot disagree with the target.
    """
    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    arch = get_rocm_arch() if arch is None else arch
    require_wave64(arch)
    use_hw_cvt_bf16 = has_hw_bf16_convert(arch)
    elem_bits = dtype_to_elem_bits(dtype_str)
    weight_elem_bits = dtype_to_elem_bits(weight_dtype_str)
    batched = use_multi_row_kernel(n, elem_bits)
    config = (
        RmsNormRowConfig.for_lane_group(n, elem_bits)
        if batched
        else RmsNormRowConfig.from_analytical_heuristic(n, elem_bits)
    )
    threads_per_row = config.num_threads
    rows_per_block = multi_row_block_rows(threads_per_row) if batched else 1
    block_threads = rows_per_block * threads_per_row
    vecsize = config.vecsize
    num_vecs = config.num_vecs
    last_tile = config.num_tiles - 1
    # Lanes the row reduction shuffles over, and how many of those groups it has
    # to stitch together through LDS. A group wider than a wavefront only
    # happens when the row has a block to itself.
    reduce_lanes = min(threads_per_row, WARP_SIZE)
    red_slots = max(1, threads_per_row // WARP_SIZE)
    shared_storage = make_reduction_storage(red_slots)
    _, weight_per_access = vector_access_plan(vecsize, weight_elem_bits)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rmsnorm_kernel(
        input_tensor: fx.Tensor,
        gamma: fx.Tensor,
        rstd_tensor: fx.Tensor,
        output: fx.Tensor,
        m: fx.Int32,
        eps: fx.Float32,
    ):
        tid = fx.thread_idx.x
        if const_expr(rows_per_block > 1):
            lane = tid % threads_per_row
            row = fx.block_idx.x * fx.Int32(rows_per_block) + tid // threads_per_row
            # The grid rounds up to whole blocks, so the last one can hold groups
            # with no row of their own. They are skipped wholesale below.
            in_grid = row < m
        else:
            lane = tid
            row = fx.block_idx.x
            in_grid = None

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fast_math = arith.FastMathFlags.fast

        if const_expr(red_slots > 1):
            storage = fx.SharedAllocator().allocate(shared_storage).peek()
            reduction = storage.s_red.view(fx.make_layout(red_slots, 1))

        def group_reduce_add(value):
            """Sum across the lanes covering one row, within a wavefront."""
            return shuffle_reduce_add(
                value,
                reduce_lanes,
                fx.Int32(reduce_lanes),
                fast_math,
            )

        # Inline rather than shared: FlyDSL rewrites the AST of the decorated
        # kernel only, so a helper holding `if lane == 0` would be traced as a
        # plain Python conditional and fail.
        def row_reduce_add(value):
            if const_expr(red_slots == 1):
                return group_reduce_add(value)
            # More than one wavefront per row means the row owns the block, so
            # the slots are its own waves and the barrier is not shared.
            wave_lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            reduced = group_reduce_add(value)
            if wave_lane == 0:
                fx.memref_store(reduced, reduction, wave)
            gpu.barrier()
            if wave == 0:
                in_range = wave_lane < red_slots
                safe_lane = in_range.select(wave_lane, 0)
                partial = in_range.select(
                    fx.memref_load(reduction, safe_lane),
                    fx.Float32(0.0),
                )
                partial = group_reduce_add(partial)
                if wave_lane == 0:
                    fx.memref_store(partial, reduction, 0)
            gpu.barrier()
            return fx.memref_load(reduction, 0)

        def normalize_row():
            """One row: reduce its sum of squares, then scale and store it."""
            input_div = fx.logical_divide(
                row_buffer(input_tensor, row, elem_bits, n),
                fx.make_layout(vecsize, 1),
            )
            output_div = fx.logical_divide(
                row_buffer(output, row, elem_bits, n),
                fx.make_layout(vecsize, 1),
            )
            gamma_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(gamma),
                fx.make_layout(weight_per_access, 1),
            )
            copy_atom = buffer_copy_atom(config.access_bits, elem_bits)
            gamma_copy_atom = buffer_copy_atom(
                weight_per_access * weight_elem_bits,
                weight_elem_bits,
            )
            if const_expr(store_rstd):
                # One entry per row, so bound the descriptor to that rather than
                # leaving it wide open over the whole allocation.
                rstd_buffer = fx.rocdl.make_buffer_tensor(
                    rstd_tensor,
                    num_records_bytes=m * fx.Int32(4),
                )
                rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
                rstd_copy_atom = buffer_copy_atom(32, 32)

            # The row is held in registers between the two passes, in its own dtype,
            # so it is read from memory once and costs no more registers than it has
            # to.
            thread_sumsq = fx.Float32(0.0)
            row_values = []
            for tile_i in range_constexpr(config.num_tiles):
                # Only the final tile can run off the end of the row.
                partial = config.needs_predicate and tile_i == last_tile
                index = lane + tile_i * threads_per_row
                if const_expr(partial):
                    in_row = index < num_vecs
                    index = in_row.select(index, 0)
                vector = load_vec(copy_atom, vecsize, elem_dtype, input_div, index)
                row_values.append(vector)
                values = vector.to(fx.Float32)
                contribution = (values * values).reduce(ReductionOp.ADD, fastmath=fast_math)
                if const_expr(partial):
                    contribution = in_row.select(contribution, fx.Float32(0.0))
                thread_sumsq = thread_sumsq + contribution

            sum_sq = row_reduce_add(thread_sumsq)
            rrms = fmath.rsqrt(sum_sq / float(n) + eps, fastmath=fast_math)

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

            for tile_i in range_constexpr(config.num_tiles):
                partial = config.needs_predicate and tile_i == last_tile
                index = lane + tile_i * threads_per_row
                safe_index = index
                if const_expr(partial):
                    in_row = index < num_vecs
                    safe_index = in_row.select(index, 0)
                weights = load_dtype_vec(
                    gamma_copy_atom,
                    weight_elem_dtype,
                    weight_elem_bits,
                    gamma_div,
                    safe_index,
                    vecsize,
                )
                values = row_values[tile_i].to(fx.Float32)
                result = to_store_dtype(
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

        if const_expr(rows_per_block > 1):
            # One uniform branch around the whole row, rather than sizing the
            # spare group's descriptors to zero bytes the way the feature kernel
            # does. The row index is uniform across a group, so a group with no
            # row skips it entirely and the shuffles stay collective for every
            # group that has one.
            #
            # This is not interchangeable with the descriptor form: without the
            # branch the compiler predicates every access on its own, emitting 64
            # s_cbranch_execnz for a 2047-wide row against one here, and 1311
            # instructions against 476. That costs 1.6x on 16384x2047 and
            # 131072x257. The feature kernel is unaffected because
            # batch_feature_rows only batches rows a group covers in one pass, so
            # it never has a deep tile loop to guard.
            if in_grid:
                normalize_row()
        else:
            normalize_row()

    def grid_blocks(m):
        if const_expr(rows_per_block == 1):
            return m
        return (m + fx.Int32(rows_per_block - 1)) // fx.Int32(rows_per_block)

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
            launcher = rmsnorm_kernel(
                input_tensor,
                gamma,
                rstd_tensor,
                output,
                m,
                eps,
            )
            launcher.launch(
                grid=(grid_blocks(m), 1, 1),
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
        launcher = rmsnorm_kernel(input_tensor, gamma, gamma, output, m, eps)
        launcher.launch(
            grid=(grid_blocks(m), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


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
    """Build the feature-complete RMSNorm forward path.

    The optimized plain-weighted builders above remain unchanged. This path
    owns optional affine inputs, fused residual addition, independent output
    dtypes, and per-head parameter addressing.

    The vector width comes from the activation dtype, exactly as it does for
    the plain path, and every other operand covers that same span with whole
    accesses of its own width. A row that is not a whole number of vectors
    degrades to a narrower vector rather than to scalar, and ``vecsize == 1``
    falls out of the same code as the scalar case.

    A row is covered by a group of threads, and a block holds one or more such
    groups. A row long enough to fill a block gets a group that wide and a
    block to itself; a short row gets a narrower group and shares the block,
    which is what keeps a 128-element head from launching a block per row with
    three quarters of its lanes idle.
    """
    arch = get_rocm_arch() if arch is None else arch
    require_wave64(arch)
    use_hw_cvt_bf16 = has_hw_bf16_convert(arch)
    input_bits = dtype_to_elem_bits(input_dtype_str)
    output_bits = dtype_to_elem_bits(output_dtype_str)
    weight_bits = dtype_to_elem_bits(weight_dtype_str)
    bias_bits = dtype_to_elem_bits(bias_dtype_str)
    residual_bits = dtype_to_elem_bits(residual_dtype_str)
    residual_out_bits = dtype_to_elem_bits(residual_out_dtype_str)
    batched = batch_feature_rows(n, input_bits)
    config = (
        RmsNormRowConfig.for_lane_group(n, input_bits)
        if batched
        else RmsNormRowConfig.from_analytical_heuristic(n, input_bits)
    )
    threads_per_row = config.num_threads
    rows_per_block = multi_row_block_rows(threads_per_row) if batched else 1
    block_threads = rows_per_block * threads_per_row
    vecsize = config.vecsize
    num_vecs = config.num_vecs
    last_tile = config.num_tiles - 1
    # Lanes the row reduction shuffles over, and how many of those groups it
    # has to stitch together through LDS. A group wider than a wavefront only
    # happens when the row has a block to itself.
    reduce_lanes = min(threads_per_row, WARP_SIZE)
    red_slots = max(1, threads_per_row // WARP_SIZE)
    shared_storage = make_reduction_storage(red_slots)

    # Elements each access carries, per operand. Only a 32-bit operand under a
    # full 16-bit activation vector needs more than one access to cover the span.
    _, input_per_access = vector_access_plan(vecsize, input_bits)
    _, output_per_access = vector_access_plan(vecsize, output_bits)
    _, weight_per_access = vector_access_plan(vecsize, weight_bits)
    _, bias_per_access = vector_access_plan(vecsize, bias_bits)
    _, residual_per_access = vector_access_plan(vecsize, residual_bits)
    _, residual_out_per_access = vector_access_plan(vecsize, residual_out_bits)

    @flyc.kernel
    def rmsnorm_feature_kernel(
        input_tensor: fx.Tensor,
        weight_tensor: fx.Tensor,
        bias_tensor: fx.Tensor,
        residual_tensor: fx.Tensor,
        output_tensor: fx.Tensor,
        residual_out_tensor: fx.Tensor,
        rstd_tensor: fx.Tensor,
        num_programs: fx.Int32,
        eps: fx.Float32,
        weight_offset: fx.Float32,
    ):
        tid = fx.thread_idx.x
        if const_expr(rows_per_block > 1):
            lane = tid % threads_per_row
            program = fx.block_idx.x * fx.Int32(rows_per_block) + tid // threads_per_row
            # The grid rounds up to whole blocks, so the last one can hold
            # groups with no row of their own. Their descriptors are sized to
            # nothing below, which drops their accesses in hardware and leaves
            # them free to keep taking part in the reduction shuffle.
            in_grid = program < num_programs
        else:
            lane = tid
            program = fx.block_idx.x
            in_grid = None
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

        def group_reduce_add(value):
            """Sum across the lanes covering one row, within a wavefront."""
            return shuffle_reduce_add(
                value,
                reduce_lanes,
                fx.Int32(reduce_lanes),
                fast_math,
            )

        def row_reduce_add(value):
            if const_expr(red_slots == 1):
                return group_reduce_add(value)
            # More than one wavefront per row means the row owns the block, so
            # the slots are its own waves and the barrier is not shared.
            wave_lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            reduced = group_reduce_add(value)
            if wave_lane == 0:
                fx.memref_store(reduced, reduction, wave)
            gpu.barrier()
            if wave == 0:
                in_range = wave_lane < red_slots
                safe_lane = in_range.select(wave_lane, 0)
                partial = in_range.select(
                    fx.memref_load(reduction, safe_lane),
                    fx.Float32(0.0),
                )
                partial = group_reduce_add(partial)
                if wave_lane == 0:
                    fx.memref_store(partial, reduction, 0)
            gpu.barrier()
            return fx.memref_load(reduction, 0)

        def row_div(tensor, elem_bits, per_access):
            """One row (or one row/head slice) split into whole accesses."""
            buffer = (
                row_head_buffer(tensor, row, head, elem_bits, n, in_grid)
                if per_head
                else row_buffer(tensor, row, elem_bits, n, in_grid)
            )
            return fx.logical_divide(buffer, fx.make_layout(per_access, 1))

        def parameter_div(tensor, elem_bits, per_access):
            """A weight or bias, which is per-head at most, never per-row."""
            buffer = (
                row_buffer(tensor, head, elem_bits, n)
                if per_head
                else fx.rocdl.make_buffer_tensor(tensor)
            )
            return fx.logical_divide(buffer, fx.make_layout(per_access, 1))

        input_div = row_div(input_tensor, input_bits, input_per_access)
        output_div = row_div(output_tensor, output_bits, output_per_access)
        input_copy = buffer_copy_atom(input_per_access * input_bits, input_bits)
        output_copy = buffer_copy_atom(output_per_access * output_bits, output_bits)
        if const_expr(has_residual):
            residual_div = row_div(residual_tensor, residual_bits, residual_per_access)
            residual_copy = buffer_copy_atom(residual_per_access * residual_bits, residual_bits)
        if const_expr(store_residual):
            residual_out_div = row_div(
                residual_out_tensor,
                residual_out_bits,
                residual_out_per_access,
            )
            residual_out_copy = buffer_copy_atom(
                residual_out_per_access * residual_out_bits,
                residual_out_bits,
            )
        if const_expr(has_weight):
            weight_div = parameter_div(weight_tensor, weight_bits, weight_per_access)
            weight_copy = buffer_copy_atom(weight_per_access * weight_bits, weight_bits)
        if const_expr(has_bias):
            bias_div = parameter_div(bias_tensor, bias_bits, bias_per_access)
            bias_copy = buffer_copy_atom(bias_per_access * bias_bits, bias_bits)
        if const_expr(store_rstd):
            # Sized to the real programs rather than left wide open, so the
            # tail block's groups have nowhere to write either.
            rstd_buffer = fx.rocdl.make_buffer_tensor(
                rstd_tensor,
                num_records_bytes=num_programs * fx.Int32(4),
            )
            rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
            f32_copy = buffer_copy_atom(32, 32)

        # The normalized value is held in registers between the two passes, so
        # the row is read once. It is the residual sum when one is fused, which
        # is also what the second pass and residual_out both need.
        thread_sumsq = fx.Float32(0.0)
        row_values = []
        for tile_i in range_constexpr(config.num_tiles):
            # Only the final tile can run off the end of the row.
            partial = config.needs_predicate and tile_i == last_tile
            index = lane + tile_i * threads_per_row
            safe_index = index
            if const_expr(partial):
                in_row = index < num_vecs
                safe_index = in_row.select(index, 0)
            value = load_dtype_vec(
                input_copy,
                input_dtype,
                input_bits,
                input_div,
                safe_index,
                vecsize,
            )
            if const_expr(has_residual):
                value = value + load_dtype_vec(
                    residual_copy,
                    residual_dtype,
                    residual_bits,
                    residual_div,
                    safe_index,
                    vecsize,
                )
            if const_expr(store_residual):
                stored = to_store_dtype(
                    residual_out_dtype_str,
                    residual_out_dtype,
                    use_hw_cvt_bf16,
                    value,
                    vecsize,
                )
                if const_expr(partial):
                    if in_row:
                        store_dtype_vec(
                            residual_out_copy,
                            residual_out_dtype,
                            residual_out_bits,
                            stored,
                            residual_out_div,
                            index,
                            vecsize,
                        )
                else:
                    store_dtype_vec(
                        residual_out_copy,
                        residual_out_dtype,
                        residual_out_bits,
                        stored,
                        residual_out_div,
                        index,
                        vecsize,
                    )
            row_values.append(value)
            contribution = (value * value).reduce(ReductionOp.ADD, fastmath=fast_math)
            if const_expr(partial):
                contribution = in_row.select(contribution, fx.Float32(0.0))
            thread_sumsq = thread_sumsq + contribution

        sum_sq = row_reduce_add(thread_sumsq)
        rrms = fmath.rsqrt(sum_sq / float(n) + eps, fastmath=fast_math)
        if const_expr(store_rstd):
            if lane == 0:
                store_scalar(
                    f32_copy,
                    fx.Float32,
                    fx.Float32,
                    rstd_div,
                    program,
                    rrms,
                )

        for tile_i in range_constexpr(config.num_tiles):
            partial = config.needs_predicate and tile_i == last_tile
            index = lane + tile_i * threads_per_row
            safe_index = index
            if const_expr(partial):
                in_row = index < num_vecs
                safe_index = in_row.select(index, 0)
            result = row_values[tile_i] * rrms
            if const_expr(has_weight):
                weights = load_dtype_vec(
                    weight_copy,
                    weight_dtype,
                    weight_bits,
                    weight_div,
                    safe_index,
                    vecsize,
                )
                result = result * (weights + weight_offset)
            if const_expr(has_bias):
                result = result + load_dtype_vec(
                    bias_copy,
                    bias_dtype,
                    bias_bits,
                    bias_div,
                    safe_index,
                    vecsize,
                )
            output_value = to_store_dtype(
                output_dtype_str,
                output_dtype,
                use_hw_cvt_bf16,
                result,
                vecsize,
            )
            if const_expr(partial):
                if in_row:
                    store_dtype_vec(
                        output_copy,
                        output_dtype,
                        output_bits,
                        output_value,
                        output_div,
                        index,
                        vecsize,
                    )
            else:
                store_dtype_vec(
                    output_copy,
                    output_dtype,
                    output_bits,
                    output_value,
                    output_div,
                    index,
                    vecsize,
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
        num_programs = m * fx.Int32(num_heads)
        rmsnorm_feature_kernel(
            input_tensor,
            weight_tensor,
            bias_tensor,
            residual_tensor,
            output_tensor,
            residual_out_tensor,
            rstd_tensor,
            num_programs,
            eps,
            weight_offset,
        ).launch(
            grid=(
                (num_programs + fx.Int32(rows_per_block - 1)) // fx.Int32(rows_per_block),
                1,
                1,
            ),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_feature

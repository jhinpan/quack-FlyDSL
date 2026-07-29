# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Plain and feature-complete RMSNorm backward kernel builders.

Both are staged: a persistent kernel writes one partial parameter gradient per
block, then a second kernel reduces the partials. There is no atomic variant.
One existed for small row counts and was removed after measurement -- fp32
atomics force a zeroed accumulator and a cast back to the weight dtype, which
cost two torch launches to save one FlyDSL launch. See
AI/flydsl_rmsnorm_notes.md.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch

from .rmsnorm_common import (
    BLOCK_THREADS,
    WARP_SIZE,
    buffer_copy_atom,
    dtype_to_elem_bits,
    dtype_to_elem_type,
    has_hw_bf16_convert,
    load_dtype_vec,
    load_scalar,
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
    to_elem_vec,
    to_store_dtype,
    vector_access_plan,
)
from .rmsnorm_config import RmsNormRowConfig

DWEIGHT_REDUCE_COLS = 64
DWEIGHT_REDUCE_ROW_LANES = 4
DWEIGHT_REDUCE_THREADS = DWEIGHT_REDUCE_COLS * DWEIGHT_REDUCE_ROW_LANES

# The staged backward accepts a wider block than the forward: it is persistent,
# so a block also has to keep the machine busy across rows, not just cover one.
TWO_STAGE_MAX_NUM_THREADS = 512


def rmsnorm_bwd_two_stage_config(n: int, dtype_str: str) -> RmsNormRowConfig:
    """How the staged backward splits one row's columns across its block."""
    return RmsNormRowConfig.from_analytical_heuristic(
        n,
        dtype_to_elem_bits(dtype_str),
        TWO_STAGE_MAX_NUM_THREADS,
    )


def build_rmsnorm_bwd_two_stage_module(
    n: int,
    dtype_str: str,
    num_programs: int,
    weight_dtype_str: str | None = None,
    arch: str | None = None,
):
    """Build the persistent backward and deterministic weight finalizer."""
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")

    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    arch = get_rocm_arch() if arch is None else arch
    require_wave64(arch)
    config = rmsnorm_bwd_two_stage_config(n, dtype_str)
    partial_threads = config.num_threads
    red_slots = max(1, (partial_threads + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = dtype_to_elem_bits(dtype_str)
    weight_elem_bits = dtype_to_elem_bits(weight_dtype_str)
    io_width = config.vecsize
    use_vec = config.vectorized
    _, weight_io_width = vector_access_plan(io_width, weight_elem_bits)
    num_io_tiles = config.num_vecs
    num_io_iters = config.num_tiles
    partial_acc_size = num_io_iters * io_width
    use_hw_cvt_bf16 = has_hw_bf16_convert(arch) if use_vec else False
    shared_storage = make_reduction_storage(red_slots)
    dweight_reduce_storage = make_reduction_storage(DWEIGHT_REDUCE_THREADS)

    @flyc.kernel(known_block_size=[partial_threads, 1, 1])
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
            return shuffle_reduce_add(value, WARP_SIZE, WARP_SIZE, fast_math)

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

        gamma_buffer = fx.rocdl.make_buffer_tensor(gamma)
        rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
        partial_buffer = fx.rocdl.make_buffer_tensor(dweight_partial)
        rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
        partial_div = fx.logical_divide(partial_buffer, fx.make_layout(1, 1))

        copy_atom = buffer_copy_atom(config.access_bits, elem_bits)
        gamma_copy_atom = buffer_copy_atom(weight_io_width * weight_elem_bits, weight_elem_bits)
        f32_copy_atom = buffer_copy_atom(32, 32)
        gamma_div = fx.logical_divide(
            gamma_buffer,
            fx.make_layout(weight_io_width, 1),
        )

        gamma_local = []
        if const_expr(use_vec):
            for tile_i in range_constexpr(num_io_iters):
                io_index = tid + tile_i * partial_threads
                is_valid = io_index < num_io_tiles
                safe_index = is_valid.select(io_index, 0)
                gamma_local.append(
                    load_dtype_vec(
                        gamma_copy_atom,
                        weight_elem_dtype,
                        weight_elem_bits,
                        gamma_div,
                        safe_index,
                        io_width,
                    )
                )

        accumulated_dweight = fx.Vector.filled(
            partial_acc_size,
            0.0,
            fx.Float32,
        )
        for row in range(fx.Int32(block), m, num_programs):
            input_div = fx.logical_divide(
                row_buffer(input_tensor, row, elem_bits, n),
                fx.make_layout(io_width, 1),
            )
            dy_div = fx.logical_divide(
                row_buffer(dy, row, elem_bits, n),
                fx.make_layout(io_width, 1),
            )
            dx_div = fx.logical_divide(
                row_buffer(dx, row, elem_bits, n),
                fx.make_layout(io_width, 1),
            )
            rstd = load_scalar(f32_copy_atom, fx.Float32, rstd_div, row)

            thread_acc = zero
            input_local = []
            dy_local = []
            for tile_i in range_constexpr(num_io_iters):
                io_index = tid + tile_i * partial_threads
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
                io_index = tid + tile_i * partial_threads
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
                            io_width,
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
            io_index = tid + tile_i * partial_threads
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
        f32_copy_atom = buffer_copy_atom(32, 32)
        weight_copy_atom = buffer_copy_atom(weight_elem_bits, weight_elem_bits)

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
            block=(partial_threads, 1, 1),
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


def build_rmsnorm_feature_bwd_two_stage_module(
    n: int,
    source_dtype_str: str,
    dy_dtype_str: str,
    dx_dtype_str: str,
    dresidual_dtype_str: str,
    dresidual_out_dtype_str: str,
    num_programs: int,
    *,
    weight_dtype_str: str,
    has_weight: bool,
    has_bias: bool,
    compute_dweight: bool,
    compute_dbias: bool,
    has_residual: bool,
    has_dresidual_out: bool,
    per_head: bool,
    num_heads: int,
    arch: str | None = None,
):
    """Build deterministic persistent feature backward plus parameter reduce."""
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")
    arch = get_rocm_arch() if arch is None else arch
    require_wave64(arch)
    use_hw_cvt_bf16 = has_hw_bf16_convert(arch)
    source_bits = dtype_to_elem_bits(source_dtype_str)
    dy_bits = dtype_to_elem_bits(dy_dtype_str)
    dx_bits = dtype_to_elem_bits(dx_dtype_str)
    dresidual_bits = dtype_to_elem_bits(dresidual_dtype_str)
    dresidual_out_bits = dtype_to_elem_bits(dresidual_out_dtype_str)
    weight_bits = dtype_to_elem_bits(weight_dtype_str)
    config = rmsnorm_bwd_two_stage_config(n, source_dtype_str)
    block_threads = config.num_threads
    vecsize = config.vecsize
    num_vecs = config.num_vecs
    num_tiles = config.num_tiles
    values_per_thread = num_tiles * vecsize
    red_slots = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)
    shared_storage = make_reduction_storage(red_slots)
    parameter_numel = num_heads * n
    dweight_workspace_row_offset = 0
    dbias_workspace_row_offset = num_programs * num_heads if compute_dweight else 0

    _, source_per_access = vector_access_plan(vecsize, source_bits)
    _, dy_per_access = vector_access_plan(vecsize, dy_bits)
    _, dx_per_access = vector_access_plan(vecsize, dx_bits)
    _, dresidual_per_access = vector_access_plan(vecsize, dresidual_bits)
    _, dresidual_out_per_access = vector_access_plan(vecsize, dresidual_out_bits)
    _, weight_per_access = vector_access_plan(vecsize, weight_bits)
    _, workspace_per_access = vector_access_plan(vecsize, 32)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rmsnorm_feature_bwd_partial_kernel(
        source_tensor: fx.Tensor,
        weight_tensor: fx.Tensor,
        dy_tensor: fx.Tensor,
        dresidual_out_tensor: fx.Tensor,
        rstd_tensor: fx.Tensor,
        dx_tensor: fx.Tensor,
        dresidual_tensor: fx.Tensor,
        workspace_tensor: fx.Tensor,
        m: fx.Int32,
        weight_offset: fx.Float32,
    ):
        linear_program = fx.block_idx.x
        tid = fx.thread_idx.x
        program = linear_program // fx.Int32(num_heads) if per_head else linear_program
        head = linear_program % fx.Int32(num_heads) if per_head else fx.Int32(0)

        source_dtype = dtype_to_elem_type(source_dtype_str)
        dy_dtype = dtype_to_elem_type(dy_dtype_str)
        dx_dtype = dtype_to_elem_type(dx_dtype_str)
        dresidual_dtype = dtype_to_elem_type(dresidual_dtype_str)
        dresidual_out_dtype = dtype_to_elem_type(dresidual_out_dtype_str)
        weight_dtype = dtype_to_elem_type(weight_dtype_str)
        fast_math = arith.FastMathFlags.fast

        storage = fx.SharedAllocator().allocate(shared_storage).peek()
        reduction = storage.s_red.view(fx.make_layout(red_slots, 1))

        def wave_reduce_add(value):
            return shuffle_reduce_add(value, WARP_SIZE, WARP_SIZE, fast_math)

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

        def row_div(tensor, row_index, elem_bits, per_access):
            buffer = (
                row_head_buffer(tensor, row_index, head, elem_bits, n)
                if per_head
                else row_buffer(tensor, row_index, elem_bits, n)
            )
            return fx.logical_divide(buffer, fx.make_layout(per_access, 1))

        rstd_buffer = fx.rocdl.make_buffer_tensor(rstd_tensor)
        rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))

        source_copy = buffer_copy_atom(source_per_access * source_bits, source_bits)
        dy_copy = buffer_copy_atom(dy_per_access * dy_bits, dy_bits)
        dx_copy = buffer_copy_atom(dx_per_access * dx_bits, dx_bits)
        if const_expr(has_residual):
            dresidual_copy = buffer_copy_atom(
                dresidual_per_access * dresidual_bits,
                dresidual_bits,
            )
        if const_expr(has_dresidual_out):
            dresidual_out_copy = buffer_copy_atom(
                dresidual_out_per_access * dresidual_out_bits,
                dresidual_out_bits,
            )
        f32_copy = buffer_copy_atom(32, 32)
        workspace_copy = buffer_copy_atom(workspace_per_access * 32, 32)

        # The weight does not vary by row, so it is loaded once per block and
        # kept in registers with weight_offset already folded in.
        weight_local = []
        if const_expr(has_weight):
            weight_copy = buffer_copy_atom(weight_per_access * weight_bits, weight_bits)
            weight_buffer = (
                row_buffer(weight_tensor, head, weight_bits, n)
                if per_head
                else fx.rocdl.make_buffer_tensor(weight_tensor)
            )
            weight_div = fx.logical_divide(weight_buffer, fx.make_layout(weight_per_access, 1))
            for tile_i in range_constexpr(num_tiles):
                index = tid + tile_i * block_threads
                safe_index = (index < num_vecs).select(index, 0)
                weight_local.append(
                    load_dtype_vec(
                        weight_copy,
                        weight_dtype,
                        weight_bits,
                        weight_div,
                        safe_index,
                        vecsize,
                    )
                    + weight_offset
                )

        initial_dweight = fx.Vector.filled(
            values_per_thread,
            0.0,
            fx.Float32,
        )
        initial_dbias = fx.Vector.filled(
            values_per_thread,
            0.0,
            fx.Float32,
        )
        for row, state in range(
            fx.Int32(program),
            m,
            fx.Int32(num_programs),
            init=[initial_dweight, initial_dbias],
        ):
            accumulated_dweight = state[0]
            accumulated_dbias = state[1]
            source_div = row_div(source_tensor, row, source_bits, source_per_access)
            dy_div = row_div(dy_tensor, row, dy_bits, dy_per_access)
            dx_div = row_div(dx_tensor, row, dx_bits, dx_per_access)
            if const_expr(has_dresidual_out):
                dresidual_out_div = row_div(
                    dresidual_out_tensor,
                    row,
                    dresidual_out_bits,
                    dresidual_out_per_access,
                )
            if const_expr(has_residual):
                dresidual_div = row_div(
                    dresidual_tensor,
                    row,
                    dresidual_bits,
                    dresidual_per_access,
                )

            rstd_index = fx.Int32(row) * fx.Int32(num_heads) + head if per_head else row
            rstd = load_scalar(f32_copy, fx.Float32, rstd_div, rstd_index)

            # Hold the row in registers across the reduction so it is read once.
            thread_sum = fx.Float32(0.0)
            source_local = []
            dy_local = []
            for tile_i in range_constexpr(num_tiles):
                index = tid + tile_i * block_threads
                valid = index < num_vecs
                safe_index = valid.select(index, 0)
                source = load_dtype_vec(
                    source_copy,
                    source_dtype,
                    source_bits,
                    source_div,
                    safe_index,
                    vecsize,
                )
                dy = load_dtype_vec(dy_copy, dy_dtype, dy_bits, dy_div, safe_index, vecsize)
                source_local.append(source)
                dy_local.append(dy)
                wdy = dy * weight_local[tile_i] if has_weight else dy
                product = (source * rstd * wdy).reduce(ReductionOp.ADD, fastmath=fast_math)
                thread_sum = thread_sum + valid.select(product, fx.Float32(0.0))

            correction = block_reduce_add(thread_sum) / float(n)
            row_dweight = []
            row_dbias = []
            for tile_i in range_constexpr(num_tiles):
                index = tid + tile_i * block_threads
                valid = index < num_vecs
                safe_index = valid.select(index, 0)
                source = source_local[tile_i]
                dy = dy_local[tile_i]
                x_hat = source * rstd
                wdy = dy * weight_local[tile_i] if has_weight else dy
                total = (wdy - x_hat * correction) * rstd
                if const_expr(has_dresidual_out):
                    total = total + load_dtype_vec(
                        dresidual_out_copy,
                        dresidual_out_dtype,
                        dresidual_out_bits,
                        dresidual_out_div,
                        safe_index,
                        vecsize,
                    )
                if index < num_vecs:
                    store_dtype_vec(
                        dx_copy,
                        dx_dtype,
                        dx_bits,
                        to_store_dtype(
                            dx_dtype_str,
                            dx_dtype,
                            use_hw_cvt_bf16,
                            total,
                            vecsize,
                        ),
                        dx_div,
                        index,
                        vecsize,
                    )
                    if const_expr(has_residual):
                        store_dtype_vec(
                            dresidual_copy,
                            dresidual_dtype,
                            dresidual_bits,
                            to_store_dtype(
                                dresidual_dtype_str,
                                dresidual_dtype,
                                use_hw_cvt_bf16,
                                total,
                                vecsize,
                            ),
                            dresidual_div,
                            index,
                            vecsize,
                        )
                if const_expr(compute_dweight):
                    dweight_value = dy * x_hat
                    for lane in range_constexpr(vecsize):
                        row_dweight.append(valid.select(dweight_value[lane], fx.Float32(0.0)))
                if const_expr(compute_dbias):
                    for lane in range_constexpr(vecsize):
                        row_dbias.append(valid.select(dy[lane], fx.Float32(0.0)))

            next_dweight = accumulated_dweight
            next_dbias = accumulated_dbias
            if const_expr(compute_dweight):
                next_dweight = accumulated_dweight + fx.Vector.from_elements(
                    row_dweight,
                    fx.Float32,
                )
            if const_expr(compute_dbias):
                next_dbias = accumulated_dbias + fx.Vector.from_elements(
                    row_dbias,
                    fx.Float32,
                )
            gpu.barrier()
            results = yield [next_dweight, next_dbias]

        final_dweight = results[0]
        final_dbias = results[1]
        workspace_row = (
            fx.Int64(program) * fx.Int64(num_heads) + fx.Int64(head)
            if per_head
            else fx.Int64(program)
        )
        if const_expr(compute_dweight):
            dweight_workspace_div = fx.logical_divide(
                row_buffer(
                    workspace_tensor,
                    dweight_workspace_row_offset + workspace_row,
                    32,
                    n,
                ),
                fx.make_layout(workspace_per_access, 1),
            )
        if const_expr(compute_dbias):
            dbias_workspace_div = fx.logical_divide(
                row_buffer(
                    workspace_tensor,
                    dbias_workspace_row_offset + workspace_row,
                    32,
                    n,
                ),
                fx.make_layout(workspace_per_access, 1),
            )
        for tile_i in range_constexpr(num_tiles):
            index = tid + tile_i * block_threads
            lanes = list(range(tile_i * vecsize, (tile_i + 1) * vecsize))
            if index < num_vecs:
                if const_expr(compute_dweight):
                    store_dtype_vec(
                        workspace_copy,
                        fx.Float32,
                        32,
                        final_dweight.shuffle(final_dweight, lanes),
                        dweight_workspace_div,
                        index,
                        vecsize,
                    )
                if const_expr(compute_dbias):
                    store_dtype_vec(
                        workspace_copy,
                        fx.Float32,
                        32,
                        final_dbias.shuffle(final_dbias, lanes),
                        dbias_workspace_div,
                        index,
                        vecsize,
                    )

    @flyc.kernel
    def rmsnorm_feature_parameter_reduce_kernel(
        workspace_tensor: fx.Tensor,
        dweight_tensor: fx.Tensor,
        dbias_tensor: fx.Tensor,
    ):
        block = fx.block_idx.x
        tid = fx.thread_idx.x
        parameter_index = fx.Int64(block) * fx.Int64(BLOCK_THREADS) + fx.Int64(tid)
        valid = parameter_index < parameter_numel
        safe_index = valid.select(parameter_index, 0)
        parameter_head = safe_index // fx.Int64(n) if per_head else fx.Int64(0)
        parameter_column = safe_index % fx.Int64(n) if per_head else safe_index
        if const_expr(compute_dweight):
            dweight_buffer = (
                row_buffer(dweight_tensor, parameter_head, 32, n)
                if per_head
                else fx.rocdl.make_buffer_tensor(dweight_tensor)
            )
            dweight_div = fx.logical_divide(
                dweight_buffer,
                fx.make_layout(1, 1),
            )
        if const_expr(compute_dbias):
            dbias_buffer = (
                row_buffer(dbias_tensor, parameter_head, 32, n)
                if per_head
                else fx.rocdl.make_buffer_tensor(dbias_tensor)
            )
            dbias_div = fx.logical_divide(
                dbias_buffer,
                fx.make_layout(1, 1),
            )
        output_index = parameter_column if per_head else parameter_index
        f32_copy = buffer_copy_atom(32, 32)

        dweight_total = fx.Float32(0.0)
        dbias_total = fx.Float32(0.0)
        # A device loop, not range_constexpr: num_programs tracks the row count,
        # so unrolling it made codegen linear in the batch size (32s at 1536
        # against a flat 0.13s for the plain reduce next door).
        for partial_row in range(num_programs):
            workspace_row = (
                fx.Int64(partial_row) * fx.Int64(num_heads) + parameter_head
                if per_head
                else fx.Int64(partial_row)
            )
            if const_expr(compute_dweight):
                dweight_workspace_row = row_buffer(
                    workspace_tensor,
                    dweight_workspace_row_offset + workspace_row,
                    32,
                    n,
                )
                dweight_workspace_div = fx.logical_divide(
                    dweight_workspace_row,
                    fx.make_layout(1, 1),
                )
                dweight_total = dweight_total + load_scalar(
                    f32_copy,
                    fx.Float32,
                    dweight_workspace_div,
                    parameter_column,
                )
            if const_expr(compute_dbias):
                dbias_workspace_row = row_buffer(
                    workspace_tensor,
                    dbias_workspace_row_offset + workspace_row,
                    32,
                    n,
                )
                dbias_workspace_div = fx.logical_divide(
                    dbias_workspace_row,
                    fx.make_layout(1, 1),
                )
                dbias_total = dbias_total + load_scalar(
                    f32_copy,
                    fx.Float32,
                    dbias_workspace_div,
                    parameter_column,
                )
        if parameter_index < parameter_numel:
            if const_expr(compute_dweight):
                store_scalar(
                    f32_copy,
                    fx.Float32,
                    fx.Float32,
                    dweight_div,
                    output_index,
                    dweight_total,
                )
            if const_expr(compute_dbias):
                store_scalar(
                    f32_copy,
                    fx.Float32,
                    fx.Float32,
                    dbias_div,
                    output_index,
                    dbias_total,
                )

    reduce_grid = (parameter_numel + BLOCK_THREADS - 1) // BLOCK_THREADS
    reduces_parameters = compute_dweight or compute_dbias

    @flyc.jit
    def launch_rmsnorm_feature_bwd_two_stage(
        source_tensor: fx.Tensor,
        weight_tensor: fx.Tensor,
        dy_tensor: fx.Tensor,
        dresidual_out_tensor: fx.Tensor,
        rstd_tensor: fx.Tensor,
        dx_tensor: fx.Tensor,
        dresidual_tensor: fx.Tensor,
        dweight_tensor: fx.Tensor,
        dbias_tensor: fx.Tensor,
        workspace_tensor: fx.Tensor,
        m: fx.Int32,
        weight_offset: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        rmsnorm_feature_bwd_partial_kernel(
            source_tensor,
            weight_tensor,
            dy_tensor,
            dresidual_out_tensor,
            rstd_tensor,
            dx_tensor,
            dresidual_tensor,
            workspace_tensor,
            m,
            weight_offset,
        ).launch(
            grid=(num_programs * num_heads, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )
        # Both reduce branches compile out when no parameter gradient is
        # wanted, so the launch would cover parameter_numel doing nothing.
        if const_expr(reduces_parameters):
            rmsnorm_feature_parameter_reduce_kernel(
                workspace_tensor,
                dweight_tensor,
                dbias_tensor,
            ).launch(
                grid=(reduce_grid, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch_rmsnorm_feature_bwd_two_stage

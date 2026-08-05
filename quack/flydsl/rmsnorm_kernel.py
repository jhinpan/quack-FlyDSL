# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""RMSNorm forward builder."""

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
    row_buffer,
    row_head_buffer,
    shuffle_reduce_add,
    store_dtype_vec,
    store_scalar,
    to_store_dtype,
    vector_access_plan,
)
from .rmsnorm_config import (
    MAX_TUNED_NUM_THREADS,
    RmsNormRowConfig,
    batch_short_rows,
    multi_row_block_rows,
)


def build_rmsnorm_module(
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
    row_config: RmsNormRowConfig | None = None,
    apply_weight_offset: bool = True,
    persistent_rows: bool = False,
    persistent_programs: int = 0,
):
    """Build the RMSNorm forward, specialized by shape, dtypes and feature flags.

    One builder covers every combination the backend accepts: optional affine
    inputs, fused residual addition, independent output dtypes, and per-head
    parameter addressing. Each flag is compile-time, so a combination that is
    off costs nothing at run time.

    The vector width comes from the activation dtype, and every other operand
    covers that same span with whole accesses of its own width -- only a 32-bit
    operand under a full 16-bit activation vector needs more than one.

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
    batched = batch_short_rows(n, input_bits)
    config = row_config
    if config is None:
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
    reload_from_gmem = config.reload_from == "gmem"
    runtime_wide_loop = reload_from_gmem
    plain_bf16_f32 = (
        input_dtype_str == output_dtype_str == "bf16"
        and weight_dtype_str == "f32"
        and has_weight
        and not has_bias
        and not has_residual
        and not store_residual
        and not store_rstd
        and not per_head
    )
    non_temporal_input = plain_bf16_f32 and n in (4096, 8192)
    non_temporal_output = plain_bf16_f32 and n in (256, 512, 8192)
    wide_full_tiles = num_vecs // threads_per_row
    wide_tail_vecs = num_vecs % threads_per_row
    if persistent_rows:
        if persistent_programs <= 0:
            raise ValueError("persistent_programs must be positive in persistent-row mode")
        if (
            batched
            or input_dtype_str != "bf16"
            or output_dtype_str != "bf16"
            or weight_dtype_str != "f32"
            or not has_weight
            or has_bias
            or has_residual
            or store_residual
            or store_rstd
            or per_head
            or num_heads != 1
            or reload_from_gmem
            or config.needs_predicate
        ):
            raise ValueError(
                "persistent-row mode currently requires aligned BF16 input/output, "
                "FP32 weight, and plain inference feature flags"
            )
    # Lanes the row reduction shuffles over, and how many of those groups it
    # has to stitch together through LDS. A group wider than a wavefront only
    # happens when the row has a block to itself.
    reduce_lanes = min(threads_per_row, WARP_SIZE)
    red_slots = max(1, threads_per_row // WARP_SIZE)
    # Persistent rows alternate two tiny LDS reduction buffers. That lets the
    # next row start without racing waves that are still consuming the prior
    # result, and avoids a third block barrier per row.
    red_buffers = 2 if persistent_rows and red_slots > 1 else 1
    red_storage_slots = red_slots * red_buffers
    shared_storage = make_reduction_storage(red_storage_slots)

    # Elements each access carries, per operand. Only a 32-bit operand under a
    # full 16-bit activation vector needs more than one access to cover the span.
    _, input_per_access = vector_access_plan(vecsize, input_bits)
    _, output_per_access = vector_access_plan(vecsize, output_bits)
    _, weight_per_access = vector_access_plan(vecsize, weight_bits)
    _, bias_per_access = vector_access_plan(vecsize, bias_bits)
    _, residual_per_access = vector_access_plan(vecsize, residual_bits)
    _, residual_out_per_access = vector_access_plan(vecsize, residual_out_bits)

    # A block wider than a wavefront quad needs the launch bound declared, or
    # the AMDGPU backend keeps its 256-thread default and refuses the launch.
    @flyc.kernel(**({} if block_threads <= 256 else {"known_block_size": [block_threads, 1, 1]}))
    def rmsnorm_kernel(
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
        reduction = storage.s_red.view(fx.make_layout(red_storage_slots, 1))

        def group_reduce_add(value):
            """Sum across the lanes covering one row, within a wavefront."""
            return shuffle_reduce_add(
                value,
                reduce_lanes,
                fx.Int32(reduce_lanes),
                fast_math,
            )

        def row_reduce_add(value, red_buffer=0):
            if const_expr(red_slots == 1):
                return group_reduce_add(value)
            # More than one wavefront per row means the row owns the block, so
            # the slots are its own waves and the barrier is not shared.
            wave_lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            red_base = fx.Int32(red_buffer) * fx.Int32(red_slots)
            reduced = group_reduce_add(value)
            if wave_lane == 0:
                fx.memref_store(reduced, reduction, red_base + wave)
            gpu.barrier()
            if wave == 0:
                in_range = wave_lane < red_slots
                safe_lane = in_range.select(wave_lane, 0)
                partial = in_range.select(
                    fx.memref_load(reduction, red_base + safe_lane),
                    fx.Float32(0.0),
                )
                partial = group_reduce_add(partial)
                if wave_lane == 0:
                    fx.memref_store(partial, reduction, red_base)
            gpu.barrier()
            return fx.memref_load(reduction, red_base)

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
        input_copy = buffer_copy_atom(
            input_per_access * input_bits,
            input_bits,
            cache_modifier=2 if non_temporal_input else 0,
        )
        output_copy = buffer_copy_atom(
            output_per_access * output_bits,
            output_bits,
            cache_modifier=2 if non_temporal_output else 0,
        )
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

        if const_expr(persistent_rows):
            # This is the same generic feature kernel under a compile-time
            # launch mode, not a second kernel. The restricted host selector
            # above keeps unsupported feature combinations on the unchanged
            # one-row fallback.
            weight_local = []
            initial_input = []
            for tile_i in range_constexpr(config.num_tiles):
                index = lane + tile_i * threads_per_row
                persistent_weight = load_dtype_vec(
                    weight_copy,
                    weight_dtype,
                    weight_bits,
                    weight_div,
                    index,
                    vecsize,
                )
                if const_expr(apply_weight_offset):
                    persistent_weight = persistent_weight + weight_offset
                weight_local.append(persistent_weight)
                initial_input.append(
                    load_vec(
                        input_copy,
                        vecsize,
                        input_dtype,
                        input_div,
                        index,
                    )
                )

            for persistent_row, prefetched_input in range(
                fx.Int32(program),
                num_programs,
                fx.Int32(persistent_programs),
                init=initial_input,
            ):
                thread_sumsq = fx.Float32(0.0)
                for tile_i in range_constexpr(config.num_tiles):
                    value = prefetched_input[tile_i].to(fx.Float32)
                    thread_sumsq = thread_sumsq + (value * value).reduce(
                        ReductionOp.ADD,
                        fastmath=fast_math,
                    )

                sum_sq = row_reduce_add(
                    thread_sumsq,
                    (fx.Int32(persistent_row) // fx.Int32(persistent_programs))
                    % fx.Int32(red_buffers),
                )
                rrms = fmath.rsqrt(sum_sq / float(n) + eps, fastmath=fast_math)

                # Issue the following row after the reduction barriers and
                # before this row's epilogue. Carrying the fragment through
                # the loop SSA lets those loads overlap the multiply/store
                # work without a barrier forcing their vmcnt to zero.
                next_row = fx.Int32(persistent_row) + fx.Int32(persistent_programs)
                next_valid = next_row < num_programs
                next_input_div = fx.logical_divide(
                    row_buffer(
                        input_tensor,
                        next_row,
                        input_bits,
                        n,
                        next_valid,
                    ),
                    fx.make_layout(input_per_access, 1),
                )
                next_input = []
                for tile_i in range_constexpr(config.num_tiles):
                    index = lane + tile_i * threads_per_row
                    next_value = prefetched_input[tile_i]
                    if next_valid:
                        next_value = load_vec(
                            input_copy,
                            vecsize,
                            input_dtype,
                            next_input_div,
                            index,
                        )
                    next_input.append(next_value)

                persistent_output_div = fx.logical_divide(
                    row_buffer(output_tensor, persistent_row, output_bits, n),
                    fx.make_layout(output_per_access, 1),
                )
                for tile_i in range_constexpr(config.num_tiles):
                    index = lane + tile_i * threads_per_row
                    result = prefetched_input[tile_i].to(fx.Float32) * rrms * weight_local[tile_i]
                    store_dtype_vec(
                        output_copy,
                        output_dtype,
                        output_bits,
                        to_store_dtype(
                            output_dtype_str,
                            output_dtype,
                            use_hw_cvt_bf16,
                            result,
                            vecsize,
                        ),
                        persistent_output_div,
                        index,
                        vecsize,
                    )
                _persistent_results = yield next_input
            return

        # Narrow rows keep their values across the reduction. Wide rows reload
        # them for the epilogue: one extra global read is much cheaper than
        # spilling an unbounded row fragment to private memory.
        thread_sumsq = fx.Float32(0.0)
        row_values = []
        native_row_values = []
        if const_expr(runtime_wide_loop):
            # A device loop keeps code size and temporary VGPRs independent of N.
            for tile_i in range(wide_full_tiles):
                index = lane + tile_i * threads_per_row
                value = load_dtype_vec(
                    input_copy,
                    input_dtype,
                    input_bits,
                    input_div,
                    index,
                    vecsize,
                )
                if const_expr(has_residual):
                    value = value + load_dtype_vec(
                        residual_copy,
                        residual_dtype,
                        residual_bits,
                        residual_div,
                        index,
                        vecsize,
                    )
                if const_expr(store_residual):
                    store_dtype_vec(
                        residual_out_copy,
                        residual_out_dtype,
                        residual_out_bits,
                        to_store_dtype(
                            residual_out_dtype_str,
                            residual_out_dtype,
                            use_hw_cvt_bf16,
                            value,
                            vecsize,
                        ),
                        residual_out_div,
                        index,
                        vecsize,
                    )
                contribution = (value * value).reduce(ReductionOp.ADD, fastmath=fast_math)
                thread_sumsq = thread_sumsq + contribution
            if const_expr(wide_tail_vecs > 0):
                index = lane + wide_full_tiles * threads_per_row
                in_row = lane < wide_tail_vecs
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
                if const_expr(store_residual):  # noqa: SIM102 - compile-time guard
                    if in_row:
                        store_dtype_vec(
                            residual_out_copy,
                            residual_out_dtype,
                            residual_out_bits,
                            to_store_dtype(
                                residual_out_dtype_str,
                                residual_out_dtype,
                                use_hw_cvt_bf16,
                                value,
                                vecsize,
                            ),
                            residual_out_div,
                            index,
                            vecsize,
                        )
                contribution = (value * value).reduce(ReductionOp.ADD, fastmath=fast_math)
                thread_sumsq = thread_sumsq + in_row.select(
                    contribution,
                    fx.Float32(0.0),
                )
        else:
            for tile_i in range_constexpr(config.num_tiles):
                # Only the final tile can run off the end of the row.
                partial = config.needs_predicate and tile_i == last_tile
                index = lane + tile_i * threads_per_row
                safe_index = index
                if const_expr(partial):
                    in_row = index < num_vecs
                    safe_index = in_row.select(index, 0)
                native_value = load_vec(
                    input_copy,
                    vecsize,
                    input_dtype,
                    input_div,
                    safe_index,
                )
                value = native_value.to(fx.Float32)
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
                if const_expr(not reload_from_gmem):
                    if const_expr(has_residual):
                        row_values.append(value)
                    else:
                        # Keep the source in its storage dtype across the
                        # reduction. The feature builder still owns this path,
                        # but the no-residual specialization does not double
                        # BF16's live register footprint before FP32 math.
                        native_row_values.append(native_value)
                contribution = (value * value).reduce(
                    ReductionOp.ADD,
                    fastmath=fast_math,
                )
                if const_expr(partial):
                    contribution = in_row.select(contribution, fx.Float32(0.0))
                thread_sumsq = thread_sumsq + contribution

        sum_sq = row_reduce_add(thread_sumsq)
        rrms = fmath.rsqrt(sum_sq / float(n) + eps, fastmath=fast_math)
        # Keep the compile-time branch separate from the traced lane predicate.
        if const_expr(store_rstd):  # noqa: SIM102
            if lane == 0:
                store_scalar(
                    f32_copy,
                    fx.Float32,
                    fx.Float32,
                    rstd_div,
                    program,
                    rrms,
                )

        if const_expr(runtime_wide_loop):
            for tile_i in range(wide_full_tiles):
                index = lane + tile_i * threads_per_row
                value = load_dtype_vec(
                    input_copy,
                    input_dtype,
                    input_bits,
                    input_div,
                    index,
                    vecsize,
                )
                if const_expr(has_residual):
                    value = value + load_dtype_vec(
                        residual_copy,
                        residual_dtype,
                        residual_bits,
                        residual_div,
                        index,
                        vecsize,
                    )
                result = value * rrms
                if const_expr(has_weight):
                    weights = load_dtype_vec(
                        weight_copy,
                        weight_dtype,
                        weight_bits,
                        weight_div,
                        index,
                        vecsize,
                    )
                    if const_expr(apply_weight_offset):
                        weights = weights + weight_offset
                    result = result * weights
                if const_expr(has_bias):
                    result = result + load_dtype_vec(
                        bias_copy,
                        bias_dtype,
                        bias_bits,
                        bias_div,
                        index,
                        vecsize,
                    )
                store_dtype_vec(
                    output_copy,
                    output_dtype,
                    output_bits,
                    to_store_dtype(
                        output_dtype_str,
                        output_dtype,
                        use_hw_cvt_bf16,
                        result,
                        vecsize,
                    ),
                    output_div,
                    index,
                    vecsize,
                )
            if const_expr(wide_tail_vecs > 0):
                index = lane + wide_full_tiles * threads_per_row
                in_row = lane < wide_tail_vecs
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
                result = value * rrms
                if const_expr(has_weight):
                    weights = load_dtype_vec(
                        weight_copy,
                        weight_dtype,
                        weight_bits,
                        weight_div,
                        safe_index,
                        vecsize,
                    )
                    if const_expr(apply_weight_offset):
                        weights = weights + weight_offset
                    result = result * weights
                if const_expr(has_bias):
                    result = result + load_dtype_vec(
                        bias_copy,
                        bias_dtype,
                        bias_bits,
                        bias_div,
                        safe_index,
                        vecsize,
                    )
                if in_row:
                    store_dtype_vec(
                        output_copy,
                        output_dtype,
                        output_bits,
                        to_store_dtype(
                            output_dtype_str,
                            output_dtype,
                            use_hw_cvt_bf16,
                            result,
                            vecsize,
                        ),
                        output_div,
                        index,
                        vecsize,
                    )
        else:
            for tile_i in range_constexpr(config.num_tiles):
                partial = config.needs_predicate and tile_i == last_tile
                index = lane + tile_i * threads_per_row
                safe_index = index
                if const_expr(partial):
                    in_row = index < num_vecs
                    safe_index = in_row.select(index, 0)
                if const_expr(reload_from_gmem):
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
                else:
                    if const_expr(has_residual):
                        value = row_values[tile_i]
                    else:
                        value = native_row_values[tile_i].to(fx.Float32)
                result = value * rrms
                if const_expr(has_weight):
                    weights = load_dtype_vec(
                        weight_copy,
                        weight_dtype,
                        weight_bits,
                        weight_div,
                        safe_index,
                        vecsize,
                    )
                    if const_expr(apply_weight_offset):
                        weights = weights + weight_offset
                    result = result * weights
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

    # FlyDSL requires a typed stream default in the traced signature.
    @flyc.jit
    def launch_rmsnorm(
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
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        num_programs = m * fx.Int32(num_heads)
        rmsnorm_kernel(
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
                (
                    fx.Int32(persistent_programs)
                    if persistent_rows
                    else (num_programs + fx.Int32(rows_per_block - 1)) // fx.Int32(rows_per_block)
                ),
                1,
                1,
            ),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


@flyc.jit
def rmsnorm_direct(
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
    n: fx.Constexpr[int],
    input_dtype_str: fx.Constexpr[str],
    output_dtype_str: fx.Constexpr[str],
    weight_dtype_str: fx.Constexpr[str],
    bias_dtype_str: fx.Constexpr[str],
    residual_dtype_str: fx.Constexpr[str],
    residual_out_dtype_str: fx.Constexpr[str],
    has_weight: fx.Constexpr[bool],
    has_bias: fx.Constexpr[bool],
    has_residual: fx.Constexpr[bool],
    store_residual: fx.Constexpr[bool],
    store_rstd: fx.Constexpr[bool],
    per_head: fx.Constexpr[bool],
    num_heads: fx.Constexpr[int],
    arch: fx.Constexpr[str],
    schema_version: fx.Constexpr[int],
    threads_per_row: fx.Constexpr[int],
    persistent_programs: fx.Constexpr[int] = 0,
    stream: fx.Stream = fx.Stream(None),  # noqa: B008 - required by FlyDSL's traced ABI
):
    """Specialize the existing forward builder through autotunable Constexpr inputs."""
    row_config = RmsNormRowConfig.with_num_threads(
        n,
        dtype_to_elem_bits(input_dtype_str),
        threads_per_row,
        max_num_threads=MAX_TUNED_NUM_THREADS,
    )
    launch = build_rmsnorm_module(
        n,
        input_dtype_str,
        output_dtype_str,
        weight_dtype_str=weight_dtype_str,
        bias_dtype_str=bias_dtype_str,
        residual_dtype_str=residual_dtype_str,
        residual_out_dtype_str=residual_out_dtype_str,
        has_weight=has_weight,
        has_bias=has_bias,
        has_residual=has_residual,
        store_residual=store_residual,
        store_rstd=store_rstd,
        per_head=per_head,
        num_heads=num_heads,
        arch=arch,
        row_config=row_config,
        persistent_rows=persistent_programs > 0,
        persistent_programs=persistent_programs,
    )
    launch(
        input_tensor,
        weight_tensor,
        bias_tensor,
        residual_tensor,
        output_tensor,
        residual_out_tensor,
        rstd_tensor,
        m,
        eps,
        weight_offset,
        stream,
    )

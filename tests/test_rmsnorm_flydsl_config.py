# Copyright (c) 2026, Tri Dao.

"""Specification for how a FlyDSL RMSNorm row is split across a thread block.

Pure arithmetic with no FlyDSL, torch, or GPU dependency, so it runs
everywhere. The kernels consume :class:`RmsNormRowConfig` as the single source
of truth for vector size, block width, and tile-loop trip count.
"""

import math

import pytest

from quack.flydsl.rmsnorm_config import (
    ACCESS_BITS,
    MAX_NUM_THREADS,
    MIN_NUM_THREADS,
    WAVE_SIZE,
    RmsNormRowConfig,
    batch_feature_rows,
    multi_row_block_rows,
    use_multi_row_kernel,
)


DTYPE_WIDTHS = (16, 32)
HIDDEN_SIZES = (
    1,
    3,
    7,
    8,
    16,
    64,
    127,
    128,
    255,
    256,
    512,
    1000,
    1020,
    1024,
    2048,
    3000,
    3001,
    4092,
    4095,
    4096,
    6144,
    8191,
    8192,
)
# The persistent backward accepts a wider block than the one-block-per-row
# forward because it also has to keep the machine busy across rows.
STAGED_MAX_THREADS = 512


def config(N, dtype_width, max_num_threads=MAX_NUM_THREADS):
    return RmsNormRowConfig.from_analytical_heuristic(N, dtype_width, max_num_threads)


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
def test_a_whole_row_uses_the_full_128_bit_access(dtype_width):
    assert config(4096, dtype_width).access_bits == ACCESS_BITS


def test_fp32_rows_vectorize():
    """Regression: FP32 used to fall through to a scalar loop unconditionally."""
    assert config(4096, 32).vecsize == 4


@pytest.mark.parametrize(
    ("N", "dtype_width", "vecsize"),
    [
        (4096, 16, 8),
        (4092, 16, 4),
        (1020, 16, 4),
        (1018, 16, 2),
        (3001, 16, 1),
        (4096, 32, 4),
        (4094, 32, 2),
        (4095, 32, 1),
    ],
)
def test_vecsize_degrades_by_gcd_rather_than_collapsing_to_scalar(N, dtype_width, vecsize):
    """Matches quack.rmsnorm: gcd(N, 128 // dtype_width), not an all-or-nothing test."""
    assert config(N, dtype_width).vecsize == vecsize


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_every_row_gets_a_coverable_block(N, dtype_width):
    """The invariants a forward config must satisfy for any supported row."""
    c = config(N, dtype_width)

    assert c.vecsize == math.gcd(N, ACCESS_BITS // dtype_width)
    assert c.num_vecs * c.vecsize == N
    assert c.access_bits in (8, 16, 32, 64, 128)

    assert MIN_NUM_THREADS <= c.num_threads <= MAX_NUM_THREADS
    assert c.num_threads & (c.num_threads - 1) == 0

    assert c.num_tiles * c.num_threads * c.vecsize >= N
    assert (c.num_tiles - 1) * c.num_threads * c.vecsize < N
    assert c.needs_predicate is not (c.num_vecs == c.num_tiles * c.num_threads)

    # The forward keeps the whole row in registers between its two passes,
    # which is what caps N.
    assert c.elems_per_thread <= 32


def test_small_rows_do_not_reserve_a_whole_wide_block():
    """A 512-element bf16 row is one 64-thread tile, not a quarter-idle 256."""
    c = config(512, 16)
    assert (c.num_threads, c.num_tiles, c.needs_predicate) == (64, 1, False)


def test_wide_rows_saturate_the_block_and_add_tiles():
    c = config(8192, 16)
    assert (c.num_threads, c.num_tiles, c.needs_predicate) == (MAX_NUM_THREADS, 4, False)


@pytest.mark.parametrize(
    ("N", "dtype_width", "vecsize", "num_threads"),
    [
        (1024, 16, 8, 128),
        (2048, 16, 8, 256),
        (4096, 16, 8, 512),
        (8192, 16, 8, 512),
        (1024, 32, 4, 256),
        (2048, 32, 4, 512),
        (8192, 32, 4, 512),
    ],
)
def test_the_staged_block_shrinks_to_the_row(N, dtype_width, vecsize, num_threads):
    """Regression: a fixed 512-thread block forced short rows back to scalar."""
    c = config(N, dtype_width, STAGED_MAX_THREADS)
    assert (c.vecsize, c.num_threads) == (vecsize, num_threads)
    assert not c.needs_predicate


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_the_staged_block_stays_within_its_wider_ceiling(N, dtype_width):
    c = config(N, dtype_width, STAGED_MAX_THREADS)
    assert MIN_NUM_THREADS <= c.num_threads <= STAGED_MAX_THREADS
    assert c.num_tiles * c.num_threads * c.vecsize >= N


def test_unsupported_element_width_is_rejected():
    with pytest.raises(ValueError, match="element width"):
        config(4096, 24)


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", (1, 3, 7, 8, 16, 64, 127, 128, 255))
def test_tiny_rows_are_batched_several_to_a_block(N, dtype_width):
    """One block per row wastes a launch when the row cannot fill one wave."""
    assert use_multi_row_kernel(N, dtype_width)


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", (1024, 2048, 4096, 6144, 8192))
def test_rows_that_fill_a_block_use_the_vectorized_kernel(N, dtype_width):
    """Regression: 1024 and 2048 used to be forced onto the scalar kernel."""
    assert not use_multi_row_kernel(N, dtype_width)
    assert config(N, dtype_width).vectorized


@pytest.mark.parametrize("N", (3001, 4095, 8191))
def test_wide_unvectorizable_rows_prefer_one_block_per_row(N):
    """A long scalar row still beats batching it into a multi-row block."""
    assert not use_multi_row_kernel(N, 16)


def test_the_vectorized_crossover_is_one_minimum_block_of_vectors():
    """bf16 crosses over at 64 x 8 elements, fp32 at 64 x 4."""
    assert use_multi_row_kernel(504, 16)
    assert not use_multi_row_kernel(512, 16)
    assert use_multi_row_kernel(252, 32)
    assert not use_multi_row_kernel(256, 32)


# The multi-row kernel splits a row across a group of lanes rather than a whole
# block, so it asks the same factory for a different lane budget.

MULTI_ROW_CASES = [
    (N, width) for width in DTYPE_WIDTHS for N in HIDDEN_SIZES if use_multi_row_kernel(N, width)
]


def lane_group(N, dtype_width):
    return RmsNormRowConfig.for_lane_group(N, dtype_width)


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_the_lane_group_vectorizes_exactly_like_one_block_per_row(N, dtype_width):
    """Vector width is a property of the row, not of who covers it."""
    c = lane_group(N, dtype_width)
    assert (c.vecsize, c.num_vecs) == (
        config(N, dtype_width).vecsize,
        config(N, dtype_width).num_vecs,
    )


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_the_lane_group_fits_one_wavefront(N, dtype_width):
    """Wider than a wave would make the reduction need LDS and a barrier."""
    c = lane_group(N, dtype_width)
    assert 1 <= c.num_threads <= WAVE_SIZE
    assert c.num_threads & (c.num_threads - 1) == 0


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_the_lane_group_is_no_wider_than_the_row_needs(N, dtype_width):
    """Regression: a 64-lane floor left 128-element bf16 rows three-quarters idle.

    Rounding up to a power of two can still idle lanes, because the reduction
    shuffles over the group; what it must not do is round up past that.
    """
    c = lane_group(N, dtype_width)
    assert c.num_threads <= max(1, 1 << (c.num_vecs - 1).bit_length())


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_lane_group_tiles_cover_the_row_without_a_dead_pass(N, dtype_width):
    c = lane_group(N, dtype_width)
    assert c.num_tiles * c.num_threads * c.vecsize >= N
    assert (c.num_tiles - 1) * c.num_threads * c.vecsize < N


@pytest.mark.parametrize(("N", "dtype_width"), MULTI_ROW_CASES)
def test_a_multi_row_block_fills_up_but_never_overflows(N, dtype_width):
    c = lane_group(N, dtype_width)
    rows = multi_row_block_rows(c.num_threads)
    assert rows >= 1
    assert rows * c.num_threads <= MAX_NUM_THREADS
    assert (rows + 1) * c.num_threads > MAX_NUM_THREADS


@pytest.mark.parametrize(("N", "dtype_width"), MULTI_ROW_CASES)
def test_a_multi_row_thread_caches_a_bounded_slice_of_its_row(N, dtype_width):
    """The vectorized multi-row forward also holds the row between its passes."""
    assert lane_group(N, dtype_width).elems_per_thread <= 32


@pytest.mark.parametrize(
    ("N", "dtype_width", "vecsize", "threads_per_row", "block_rows"),
    [
        # One 128-bit access per lane, so the group shrinks with the row.
        (128, 16, 8, 16, 16),
        (256, 16, 8, 32, 8),
        (64, 16, 8, 8, 32),
        (8, 16, 8, 1, 256),
        (128, 32, 4, 32, 8),
        # gcd(257, 8) == 1 leaves no vector to widen, so the group saturates a
        # wave and loops instead.
        (257, 16, 1, 64, 4),
    ],
)
def test_the_lane_group_shrinks_to_the_row(N, dtype_width, vecsize, threads_per_row, block_rows):
    c = lane_group(N, dtype_width)
    assert (c.vecsize, c.num_threads) == (vecsize, threads_per_row)
    assert multi_row_block_rows(c.num_threads) == block_rows


def test_a_short_bf16_row_is_one_whole_access_per_lane():
    """Regression: 128 bf16 was 4 scalar 16-bit loads per lane across 32 lanes."""
    c = lane_group(128, 16)
    assert (c.access_bits, c.num_tiles, c.needs_predicate) == (ACCESS_BITS, 1, False)


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_lane_group_selection_is_deterministic(N, dtype_width):
    assert lane_group(N, dtype_width) == lane_group(N, dtype_width)


def test_the_wavefront_constant_is_the_block_width_floor():
    """One authority for 64: a partial wave would idle lanes all kernel long."""
    assert MIN_NUM_THREADS == WAVE_SIZE


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_the_feature_path_only_batches_rows_a_group_covers_in_one_pass(N, dtype_width):
    """Measured: batching the feature kernel stops paying once the group loops."""
    if batch_feature_rows(N, dtype_width):
        assert lane_group(N, dtype_width).num_tiles == 1


@pytest.mark.parametrize("dtype_width", DTYPE_WIDTHS)
@pytest.mark.parametrize("N", HIDDEN_SIZES)
def test_the_feature_path_never_batches_more_than_the_plain_path(N, dtype_width):
    if batch_feature_rows(N, dtype_width):
        assert use_multi_row_kernel(N, dtype_width)


@pytest.mark.parametrize(
    ("N", "dtype_width", "batched"),
    [
        # Vectorizable short rows: the group covers them outright.
        (128, 16, True),
        (256, 16, True),
        (8, 16, True),
        (128, 32, True),
        # gcd(N, 8) == 1 leaves nothing to widen, so a group of 64 lanes has to
        # loop and the row is better off with a block of its own.
        (255, 16, False),
        (127, 16, False),
        # Long enough that neither path batches.
        (512, 16, False),
        (4096, 16, False),
    ],
)
def test_the_feature_batching_crossover(N, dtype_width, batched):
    assert batch_feature_rows(N, dtype_width) is batched

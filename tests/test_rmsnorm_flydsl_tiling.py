# Copyright (c) 2026, Tri Dao.

"""Specification for how a RMSNorm row is split across a thread block.

This is pure arithmetic with no FlyDSL, torch, or GPU dependency, so it runs
everywhere. The kernels consume ``select_row_tiling`` as the single source of
truth for vector width, block size, and tile-loop trip count.
"""

import pytest

from quack._flydsl.rmsnorm_tiling import (
    MAX_BLOCK_THREADS,
    MIN_BLOCK_THREADS,
    VECTOR_BITS,
    select_row_tiling,
    use_multi_row_kernel,
)


ELEM_BITS = (16, 32)
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
    1024,
    2048,
    3000,
    3001,
    4095,
    4096,
    6144,
    8191,
    8192,
)


@pytest.mark.parametrize("elem_bits", ELEM_BITS)
def test_vector_width_fills_one_128_bit_access(elem_bits):
    """A vectorized copy always moves exactly 128 bits, whatever the dtype."""
    tiling = select_row_tiling(4096, elem_bits)
    assert tiling.vectorized
    assert tiling.vec_width * elem_bits == VECTOR_BITS


def test_fp32_rows_vectorize():
    """Regression: FP32 used to fall through to a scalar loop unconditionally."""
    tiling = select_row_tiling(4096, 32)
    assert tiling.vectorized
    assert tiling.vec_width == 4


@pytest.mark.parametrize("n", (3000, 6144, 1024, 512))
def test_rows_that_are_not_a_whole_number_of_tiles_still_vectorize(n):
    """Regression: vectorizing used to require n % (block_threads * 8) == 0."""
    tiling = select_row_tiling(n, 16)
    assert tiling.vectorized


@pytest.mark.parametrize("n", (1, 3, 7, 127, 255, 3001, 4095, 8191))
def test_rows_that_cannot_hold_a_whole_vector_stay_scalar(n):
    """A partial 128-bit access at the row end would touch a neighbouring row."""
    tiling = select_row_tiling(n, 16)
    assert not tiling.vectorized
    assert tiling.vec_width == 1
    assert tiling.num_vecs == n


@pytest.mark.parametrize("elem_bits", ELEM_BITS)
@pytest.mark.parametrize("n", HIDDEN_SIZES)
def test_tiling_covers_the_row_exactly_once(n, elem_bits):
    tiling = select_row_tiling(n, elem_bits)
    covered = tiling.num_tiles * tiling.block_threads * tiling.vec_width
    assert covered >= n
    # One fewer tile must not be enough, otherwise we are launching dead tiles.
    assert (tiling.num_tiles - 1) * tiling.block_threads * tiling.vec_width < n
    assert tiling.num_vecs * tiling.vec_width == n or not tiling.vectorized


@pytest.mark.parametrize("elem_bits", ELEM_BITS)
@pytest.mark.parametrize("n", HIDDEN_SIZES)
def test_block_size_stays_within_bounds_and_is_a_power_of_two(n, elem_bits):
    tiling = select_row_tiling(n, elem_bits)
    assert MIN_BLOCK_THREADS <= tiling.block_threads <= MAX_BLOCK_THREADS
    assert tiling.block_threads & (tiling.block_threads - 1) == 0


@pytest.mark.parametrize("elem_bits", ELEM_BITS)
@pytest.mark.parametrize("n", HIDDEN_SIZES)
def test_registers_cached_per_thread_stay_bounded(n, elem_bits):
    """The forward keeps the whole row in registers between its two passes."""
    tiling = select_row_tiling(n, elem_bits)
    assert tiling.elems_per_thread <= 32


@pytest.mark.parametrize("elem_bits", ELEM_BITS)
@pytest.mark.parametrize("n", HIDDEN_SIZES)
def test_predication_flag_matches_the_arithmetic(n, elem_bits):
    tiling = select_row_tiling(n, elem_bits)
    exact = tiling.num_vecs == tiling.num_tiles * tiling.block_threads
    assert tiling.needs_predicate is not exact


def test_small_rows_do_not_reserve_a_whole_wide_block():
    """A 512-element bf16 row is one 64-thread tile, not a quarter-idle 256."""
    tiling = select_row_tiling(512, 16)
    assert tiling.block_threads == 64
    assert tiling.num_tiles == 1
    assert not tiling.needs_predicate


def test_wide_rows_saturate_the_block_and_add_tiles():
    tiling = select_row_tiling(8192, 16)
    assert tiling.block_threads == MAX_BLOCK_THREADS
    assert tiling.num_tiles == 4
    assert not tiling.needs_predicate


@pytest.mark.parametrize("elem_bits", ELEM_BITS)
@pytest.mark.parametrize("n", HIDDEN_SIZES)
def test_selection_is_deterministic(n, elem_bits):
    assert select_row_tiling(n, elem_bits) == select_row_tiling(n, elem_bits)


def test_unsupported_element_width_is_rejected():
    with pytest.raises(ValueError, match="element width"):
        select_row_tiling(4096, 24)


@pytest.mark.parametrize("elem_bits", ELEM_BITS)
@pytest.mark.parametrize("n", (1, 3, 7, 8, 16, 64, 127, 128, 255))
def test_tiny_rows_are_batched_several_to_a_block(n, elem_bits):
    """One block per row wastes a launch when the row cannot fill one wave."""
    assert use_multi_row_kernel(n, elem_bits)


@pytest.mark.parametrize("elem_bits", ELEM_BITS)
@pytest.mark.parametrize("n", (1024, 2048, 4096, 6144, 8192))
def test_rows_that_fill_a_block_use_the_vectorized_kernel(n, elem_bits):
    """Regression: 1024 and 2048 used to be forced onto the scalar kernel."""
    assert not use_multi_row_kernel(n, elem_bits)
    assert select_row_tiling(n, elem_bits).vectorized


@pytest.mark.parametrize("n", (3001, 4095, 8191))
def test_wide_unvectorizable_rows_prefer_one_block_per_row(n):
    """A long scalar row still beats batching it into a multi-row block."""
    assert not use_multi_row_kernel(n, 16)


def test_the_vectorized_crossover_is_one_minimum_block_of_vectors():
    """bf16 crosses over at 64 x 8 elements, fp32 at 64 x 4."""
    assert use_multi_row_kernel(504, 16)
    assert not use_multi_row_kernel(512, 16)
    assert use_multi_row_kernel(252, 32)
    assert not use_multi_row_kernel(256, 32)

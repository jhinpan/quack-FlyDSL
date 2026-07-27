# Copyright (c) 2026, Tri Dao.

"""Launch configuration for the FlyDSL RMSNorm kernels.

Mirrors :mod:`quack.rmsnorm_config`: a frozen dataclass capturing the launch
knobs, plus factories that own the heuristic. The knobs are the vector size,
the width of the thread group covering a row, and the tile-loop trip count.
One factory hands a row a whole block; the other hands it a lane group so the
multi-row kernel can batch short rows. The atomic backward is scalar and sets
its own geometry.

Pure arithmetic: no FlyDSL, no torch.
"""

import math
from dataclasses import dataclass


ACCESS_BITS = 128
# Wavefront width of every architecture this backend supports.
# ``assert_arch_matches_reductions`` checks the real target against it.
WAVE_SIZE = 64
# A block never narrows below a full wavefront: a partial wave idles lanes for
# the whole kernel.
MIN_NUM_THREADS = WAVE_SIZE
MAX_NUM_THREADS = 256
SUPPORTED_DTYPE_WIDTHS = (16, 32)

# Longest scalar row still worth batching several-to-a-block rather than
# giving each row a block of its own.
SMALL_ROW_THRESHOLD = 2048


@dataclass(frozen=True, slots=True)
class RmsNormRowConfig:
    """How one row of ``N`` elements is covered by one thread block."""

    vecsize: int
    num_threads: int
    num_tiles: int
    num_vecs: int
    dtype_width: int

    @property
    def access_bits(self) -> int:
        """Width of one load or store, which selects the buffer copy atom."""
        return self.vecsize * self.dtype_width

    @property
    def vectorized(self) -> bool:
        return self.vecsize > 1

    @property
    def needs_predicate(self) -> bool:
        """Whether the last tile runs partially off the end of the row."""
        return self.num_vecs != self.num_tiles * self.num_threads

    @property
    def elems_per_thread(self) -> int:
        """Row elements each thread holds live between the two forward passes."""
        return self.num_tiles * self.vecsize

    @classmethod
    def from_analytical_heuristic(
        cls,
        N: int,
        dtype_width: int,
        max_num_threads: int = MAX_NUM_THREADS,
        min_num_threads: int = MIN_NUM_THREADS,
    ) -> "RmsNormRowConfig":
        """Pick the widest whole access and the narrowest block that covers ``N``.

        ``vecsize`` follows :mod:`quack.rmsnorm`: the greatest common divisor of
        the row length and a full 128-bit access, so a row that is not a whole
        number of wide vectors degrades to a narrower one rather than all the
        way to scalar. ``max_num_threads`` is wider for the persistent backward,
        whose block also has to keep the machine busy across rows.
        """
        if dtype_width not in SUPPORTED_DTYPE_WIDTHS:
            raise ValueError(f"unsupported element width: {dtype_width} bits")

        vecsize = math.gcd(N, ACCESS_BITS // dtype_width)
        num_vecs = N // vecsize
        num_threads = min(
            max(_next_power_of_two(num_vecs), min_num_threads),
            max_num_threads,
        )
        return cls(
            vecsize=vecsize,
            num_threads=num_threads,
            num_tiles=-(-num_vecs // num_threads),
            num_vecs=num_vecs,
            dtype_width=dtype_width,
        )

    @classmethod
    def for_lane_group(cls, N: int, dtype_width: int) -> "RmsNormRowConfig":
        """Cover a row with a group of lanes inside one wavefront.

        The multi-row kernel gives a row a slice of a block rather than all of
        it, so ``num_threads`` is the lane group and the block-width floor does
        not apply: holding a 128-element BF16 row to 64 lanes would leave 48 of
        them with nothing to load. Capping at the wavefront is what keeps the
        row reduction a bare shuffle, with no LDS and no barrier.
        """
        return cls.from_analytical_heuristic(
            N,
            dtype_width,
            max_num_threads=WAVE_SIZE,
            min_num_threads=1,
        )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length() if value > 1 else 1


def multi_row_block_rows(threads_per_row: int) -> int:
    """How many rows share one block, given the lane group a row occupies.

    Rows are packed until the block reaches the same width the one-block-per-row
    forward targets, so both kernels present the same block to the scheduler
    however short the row is.
    """
    return max(1, MAX_NUM_THREADS // threads_per_row)


def use_multi_row_kernel(
    N: int,
    dtype_width: int,
    small_row_threshold: int = SMALL_ROW_THRESHOLD,
) -> bool:
    """Whether to batch several rows per block instead of one block per row.

    Batching pays off only when a row is too short to keep one block busy. A
    row that fills at least the smallest block is always better served by the
    vectorized one-block-per-row kernel; a long row that cannot vectorize
    still prefers one block per row over a deep scalar loop per lane.
    """
    config = RmsNormRowConfig.from_analytical_heuristic(N, dtype_width)
    if config.vectorized:
        return config.num_vecs < MIN_NUM_THREADS
    return N <= small_row_threshold


def batch_feature_rows(N: int, dtype_width: int) -> bool:
    """Whether the feature forward should batch several rows into one block.

    Stricter than :func:`use_multi_row_kernel`, and measured rather than
    derived: batching pays for the feature kernel only while a lane group
    covers the row in a single pass. Once the group has to loop, giving the
    row a whole block is faster, because a wider group makes the same trip
    count in fewer passes. The plain kernel keeps its own crossover, which
    sits further out; the two disagree only on rows whose length shares no
    factor with a 128-bit access, where neither is close to the compiler.
    """
    return (
        use_multi_row_kernel(N, dtype_width)
        and RmsNormRowConfig.for_lane_group(N, dtype_width).num_tiles == 1
    )


__all__ = [
    "ACCESS_BITS",
    "MAX_NUM_THREADS",
    "MIN_NUM_THREADS",
    "SMALL_ROW_THRESHOLD",
    "SUPPORTED_DTYPE_WIDTHS",
    "WAVE_SIZE",
    "RmsNormRowConfig",
    "batch_feature_rows",
    "multi_row_block_rows",
    "use_multi_row_kernel",
]

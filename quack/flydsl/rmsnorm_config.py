# Copyright (c) 2026, Tri Dao.

"""Launch configuration for the FlyDSL RMSNorm kernels.

Mirrors :mod:`quack.rmsnorm_config`: a frozen dataclass capturing the launch
knobs, plus factories that own the heuristic. The knobs are the vector size,
the width of the thread group covering a row, and the tile-loop trip count.
One factory hands a row a whole block; the other hands it a lane group so the
multi-row kernel can batch short rows.

Pure arithmetic: no FlyDSL, no torch.
"""

import math
from dataclasses import dataclass


ACCESS_BITS = 128
# Wavefront width of every architecture this backend supports. The single
# authority: the reductions in rmsnorm_common unroll over it and the launch
# geometry here sizes lane groups against it, so the two cannot disagree.
# ``require_wave64`` rejects a build target that would not match.
WAVE_SIZE = 64
# A block never narrows below a full wavefront: a partial wave idles lanes for
# the whole kernel.
MIN_NUM_THREADS = WAVE_SIZE
MAX_NUM_THREADS = 256
SUPPORTED_DTYPE_WIDTHS = (16, 32)

# Widest row a block can hold live between the two forward passes. A thread
# keeps ``num_tiles * vecsize`` elements in registers, so this is the register
# budget expressed as a row length. Every other cap on N derives from it.
MAX_N = 8192

# A row must be a whole number of 128-bit accesses at the narrowest element the
# backend supports. That is what makes every row start naturally aligned and
# every operand reach full vector width, so the vector size is a property of
# the dtype alone and never of the row length. Hidden sizes are multiples of 64
# or 128 in practice -- 3584, 4608, 5120, 7168 all qualify -- and a row that is
# not is rejected rather than served by a narrower access.
N_ALIGNMENT = ACCESS_BITS // min(SUPPORTED_DTYPE_WIDTHS)


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

        ``vecsize`` is written as the gcd :mod:`quack.rmsnorm` uses, but the
        adapter only admits rows aligned to :data:`N_ALIGNMENT`, so it always
        comes back at full width for the dtype. Keeping the gcd rather than the
        constant means a row that somehow reached here unaligned narrows its
        access instead of reading past the row. ``max_num_threads`` is wider for
        the persistent backward, whose block also has to keep the machine busy
        across rows.
        """
        if dtype_width not in SUPPORTED_DTYPE_WIDTHS:
            raise ValueError(f"unsupported element width: {dtype_width} bits")

        vecsize = math.gcd(N, ACCESS_BITS // dtype_width)
        num_vecs = N // vecsize
        num_threads = min(
            max(next_power_of_two(num_vecs), min_num_threads),
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

    @classmethod
    def with_num_threads(
        cls,
        N: int,
        dtype_width: int,
        num_threads: int,
    ) -> "RmsNormRowConfig":
        """Build a legal row config with an explicitly selected lane count."""
        if dtype_width not in SUPPORTED_DTYPE_WIDTHS:
            raise ValueError(f"unsupported element width: {dtype_width} bits")
        if num_threads < 1 or num_threads > MAX_NUM_THREADS:
            raise ValueError(f"num_threads must be between 1 and {MAX_NUM_THREADS}")
        if num_threads & (num_threads - 1):
            raise ValueError("num_threads must be a power of two")
        vecsize = math.gcd(N, ACCESS_BITS // dtype_width)
        num_vecs = N // vecsize
        return cls(
            vecsize=vecsize,
            num_threads=num_threads,
            num_tiles=-(-num_vecs // num_threads),
            num_vecs=num_vecs,
            dtype_width=dtype_width,
        )


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length() if value > 1 else 1


def multi_row_block_rows(threads_per_row: int) -> int:
    """How many rows share one block, given the lane group a row occupies.

    Rows are packed until the block reaches the same width the one-block-per-row
    forward targets, so both kernels present the same block to the scheduler
    however short the row is.
    """
    return max(1, MAX_NUM_THREADS // threads_per_row)


def batch_short_rows(N: int, dtype_width: int) -> bool:
    """Whether the forward should batch several rows into one block.

    Batching pays only when a row is too short to keep a block busy on its own,
    and only while a lane group still covers the row in a single pass -- once
    the group has to loop, a whole block is faster, because a wider group makes
    the same trip count in fewer passes. On an aligned row those two conditions
    are the same one: a row of fewer vectors than the block-width floor gets a
    group rounded up to a power of two no narrower than the row, so its tile
    count is one by construction.
    """
    return RmsNormRowConfig.from_analytical_heuristic(N, dtype_width).num_vecs < MIN_NUM_THREADS


__all__ = [
    "MAX_N",
    "N_ALIGNMENT",
    "WAVE_SIZE",
    "RmsNormRowConfig",
    "batch_short_rows",
    "multi_row_block_rows",
    "next_power_of_two",
]

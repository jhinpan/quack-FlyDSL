# Copyright (c) 2026, Tri Dao.

"""Single source of truth for how a RMSNorm row maps onto a thread block.

:func:`select_row_tiling` decides vector width, block size and tile-loop trip
count. The one-block-per-row forward and the persistent backward both use it,
differing only in how wide a block they will accept. The atomic backward and
the multi-row small-N forward are scalar and set their own geometry.

Pure arithmetic: no FlyDSL, no torch.
"""

from dataclasses import dataclass


VECTOR_BITS = 128
MIN_BLOCK_THREADS = 64
MAX_BLOCK_THREADS = 256
SUPPORTED_ELEM_BITS = (16, 32)

# Longest scalar row still worth batching several-to-a-block rather than
# giving each row a block of its own.
SMALL_ROW_THRESHOLD = 2048


@dataclass(frozen=True, slots=True)
class RowTiling:
    """How one row of ``n`` elements is covered by one thread block."""

    vec_width: int
    block_threads: int
    num_tiles: int
    num_vecs: int

    @property
    def vectorized(self) -> bool:
        return self.vec_width > 1

    @property
    def needs_predicate(self) -> bool:
        """Whether the last tile runs partially off the end of the row."""
        return self.num_vecs != self.num_tiles * self.block_threads

    @property
    def elems_per_thread(self) -> int:
        """Row elements each thread holds live between the two forward passes."""
        return self.num_tiles * self.vec_width


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length() if value > 1 else 1


def select_row_tiling(
    n: int,
    elem_bits: int,
    *,
    max_block_threads: int = MAX_BLOCK_THREADS,
) -> RowTiling:
    """Choose the widest 128-bit access and the narrowest block that covers ``n``.

    A vectorized copy always moves ``VECTOR_BITS`` at a time, so the vector
    width follows from the element width rather than being fixed per dtype.
    Rows that are not a whole number of vectors stay scalar, because a partial
    128-bit access at the row end would spill into the neighbouring row.
    """
    if elem_bits not in SUPPORTED_ELEM_BITS:
        raise ValueError(f"unsupported element width: {elem_bits} bits")

    vec_width = VECTOR_BITS // elem_bits
    if n % vec_width:
        vec_width = 1
    num_vecs = n // vec_width

    block_threads = min(
        max(_next_power_of_two(num_vecs), MIN_BLOCK_THREADS),
        max_block_threads,
    )
    num_tiles = -(-num_vecs // block_threads)
    return RowTiling(
        vec_width=vec_width,
        block_threads=block_threads,
        num_tiles=num_tiles,
        num_vecs=num_vecs,
    )


def use_multi_row_kernel(
    n: int,
    elem_bits: int,
    *,
    small_row_threshold: int = SMALL_ROW_THRESHOLD,
) -> bool:
    """Whether to batch several rows per block instead of one block per row.

    Batching pays off only when a row is too short to keep one block busy. A
    row that fills at least the smallest block is always better served by the
    vectorized one-block-per-row kernel; a long row that cannot vectorize
    still prefers one block per row over a deep scalar loop per lane.
    """
    tiling = select_row_tiling(n, elem_bits)
    if tiling.vectorized:
        return tiling.num_vecs < MIN_BLOCK_THREADS
    return n <= small_row_threshold


__all__ = [
    "MAX_BLOCK_THREADS",
    "MIN_BLOCK_THREADS",
    "SMALL_ROW_THRESHOLD",
    "SUPPORTED_ELEM_BITS",
    "VECTOR_BITS",
    "RowTiling",
    "select_row_tiling",
    "use_multi_row_kernel",
]

# Copyright (c) 2026, Tri Dao.

"""Launch configuration for the FlyDSL RMSNorm kernels.

Mirrors :mod:`quack.rmsnorm_config`: a frozen dataclass capturing the launch
knobs, plus a factory that owns the heuristic. One block covers one row here,
so the knobs are the vector size, the block width, and the tile-loop trip
count. The atomic backward and the multi-row small-N forward are scalar and
set their own geometry.

Pure arithmetic: no FlyDSL, no torch.
"""

import math
from dataclasses import dataclass


ACCESS_BITS = 128
MIN_NUM_THREADS = 64
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
            max(_next_power_of_two(num_vecs), MIN_NUM_THREADS),
            max_num_threads,
        )
        return cls(
            vecsize=vecsize,
            num_threads=num_threads,
            num_tiles=-(-num_vecs // num_threads),
            num_vecs=num_vecs,
            dtype_width=dtype_width,
        )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length() if value > 1 else 1


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


__all__ = [
    "ACCESS_BITS",
    "MAX_NUM_THREADS",
    "MIN_NUM_THREADS",
    "SMALL_ROW_THRESHOLD",
    "SUPPORTED_DTYPE_WIDTHS",
    "RmsNormRowConfig",
    "use_multi_row_kernel",
]

# Copyright (c) 2026, Tri Dao.

"""FlyDSL sibling of :mod:`quack.copy_utils`, so a ROCm kernel body can be read
against its CuTe counterpart.

FlyDSL ships the CuTe copy stack -- ``TiledCopy`` -> ``get_slice`` ->
``partition_S``/``partition_D`` over a TV layout -- but three gaps keep a kernel
from being written the way :mod:`quack.copy_utils` lets a CuTe one be written:

* there is no ``local_tile`` (``flat_divide`` plus an index is the documented
  equivalent) and no identity-tensor constructor;
* ``fx.copy`` requires the atom to move exactly the partition's innermost
  vector, so one atom cannot serve operands of different element widths;
* a fragment's values arrive shaped ``((ATOM_V, REST_V), ...)``, which differs
  per width and blocks arithmetic between two operands of the same tile.

The last two are the same problem. CuTe avoids it by sizing ``vecsize`` off the
*widest* operand, which keeps ``ATOM_V == vecsize`` for everything; on gfx950
that costs 2-4% at mid N and 28% at N=262144, because a bf16 row then loads
64 bits per lane instead of the 128 the hardware allows. So each width keeps its
own atom here, and :class:`TiledCopy2d` hides the per-width bookkeeping behind
the CuTe method names.
"""

from typing import Optional, Sequence

import flydsl.expr as fx

from quack.flydsl_constants import MAX_ACCESS_BITS

__all__ = [
    "TiledCopy2d",
    "buffer_tensor",
    "copy",
    "expand",
    "load",
    "local_tile",
    "make_fragment",
    "predicate_k",
    "store",
    "thread_coords",
    "tile_of",
    "tiled_copy_2d",
]


def _as_tuple(unpacked) -> tuple:
    """``IntTuple.unpack`` hands back a bare scalar for a rank-1 profile."""
    return unpacked if isinstance(unpacked, tuple) else (unpacked,)


def _shape(tensor) -> tuple:
    return _as_tuple(tensor.shape.unpack())


def buffer_tensor(tensor):
    """Bind a global tensor to the descriptor a CDNA copy atom addresses."""
    if tensor is None:
        return None
    return fx.rocdl.make_buffer_tensor(tensor)


def expand(tensor, dim: int, size: int):
    """Insert a stride-0 mode of extent ``size`` at ``dim`` (CuTe's
    ``layout_utils.expand``).

    Broadcasts a per-column vector (weight, bias) down a tile's rows, or a
    per-row scalar (rstd) across its columns, so one tiler covers every operand.
    """
    if tensor is None:
        return None
    shape, stride = list(_shape(tensor)), list(_as_tuple(tensor.stride.unpack()))
    shape.insert(dim, size)
    stride.insert(dim, 0)
    return fx.make_view(fx.get_iter(tensor), fx.make_layout(tuple(shape), tuple(stride)))


def local_tile(tensor, tiler, coord):
    """CuTe's ``local_tile``: cut ``tensor`` into ``tiler`` blocks, keep ``coord``.

    FlyDSL has no such helper; ``flat_divide`` followed by an index is what its
    layout-algebra guide prescribes. ``None`` in ``coord`` keeps that block mode,
    which is how a caller iterates the N-blocks of one row.
    """
    if tensor is None:
        return None
    return fx.flat_divide(tensor, tiler)[(None,) * len(tiler) + tuple(coord)]


def tiled_copy_2d(
    widths: Sequence[int],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
) -> "TiledCopy2d":
    """One tile, partitioned once per operand element width in ``widths``.

    Every width shares the thread and value layouts, so ``tile_mn`` and the
    columns a thread owns are identical; only the atom differs, and with it the
    ``(ATOM_V, REST_V)`` split of the innermost mode.
    """
    assert num_threads % threads_per_row == 0
    thr_layout = fx.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row), order=(1, 0)
    )
    val_layout = fx.make_ordered_layout((1, num_copy_elems), order=(0, 1))
    tiled = {
        width: fx.make_tiled_copy_tv(
            fx.make_copy_atom(
                fx.rocdl.BufferCopy(_atom_elems(width, num_copy_elems) * width, 0), width
            ),
            thr_layout,
            val_layout,
        )
        for width in sorted(set(widths))
    }
    return TiledCopy2d(tiled, threads_per_row, num_threads, num_copy_elems)


def _atom_elems(width: int, num_copy_elems: int) -> int:
    """Elements of this width that fit in the widest MUBUF access (CuTe's ATOM_V)."""
    return min(num_copy_elems, MAX_ACCESS_BITS // width)


class TiledCopy2d:
    """A tile shared by operands of several element widths.

    Stands in for ``cute.TiledCopy`` where CuTe needs only one: the CuTe kernel
    partitions every operand with a single ``tiled_copy`` because its ``vecsize``
    was capped so one atom fits them all.
    """

    __slots__ = ("num_copy_elems", "num_threads", "threads_per_row", "tiled")

    def __init__(self, tiled: dict, threads_per_row: int, num_threads: int, num_copy_elems: int):
        self.tiled = tiled
        self.threads_per_row = threads_per_row
        self.num_threads = num_threads
        self.num_copy_elems = num_copy_elems

    @property
    def widths(self):
        return tuple(self.tiled)

    @property
    def cols_per_block(self) -> int:
        """Columns one pass of the thread layout covers, before any rest modes."""
        return self.threads_per_row * self.num_copy_elems

    @property
    def tile_mn(self):
        return next(iter(self.tiled.values())).tile_mn

    @property
    def layout_tv_tiled(self):
        return next(iter(self.tiled.values())).layout_tv_tiled

    def get_slice(self, thr_idx) -> "ThrCopy2d":
        return ThrCopy2d(self, {w: t.get_slice(thr_idx) for w, t in self.tiled.items()}, thr_idx)


class ThrCopy2d:
    """One thread's slice of a :class:`TiledCopy2d`.

    ``partition_S``/``partition_D`` pick the partitioner from the tensor's own
    element width, so call sites read exactly like the CuTe ones.
    """

    __slots__ = ("slices", "thr_idx", "tiled_copy")

    def __init__(self, tiled_copy: TiledCopy2d, slices: dict, thr_idx):
        self.tiled_copy = tiled_copy
        self.slices = slices
        self.thr_idx = thr_idx

    @property
    def widths(self):
        return self.tiled_copy.widths

    def partition_S(self, tensor, width: Optional[int] = None):
        if tensor is None:
            return None
        return self.slices[width or tensor.element_type.width].partition_S(tensor)

    def partition_D(self, tensor, width: Optional[int] = None):
        if tensor is None:
            return None
        return self.slices[width or tensor.element_type.width].partition_D(tensor)


def tile_of(tensor, tile):
    """Drop the trailing N-tile mode :func:`local_tile` was asked to keep.

    Rank 4 means the mode is still there; a fragment, already cut to one tile,
    is rank 3 and passes through.
    """
    if tensor is None or tile is None or len(_shape(tensor)) < 4:
        return tensor
    return tensor[None, None, None, tile]


def make_fragment(partitioned):
    """Register fragment for one tile of a partitioned global tensor."""
    if partitioned is None:
        return None
    return fx.make_fragment_like(tile_of(partitioned, 0))


def copy(src, dst, *, pred=None, tile=None) -> None:
    """Move one tile between gmem and registers, atom chosen from the operands.

    ``src.shape[0][0]`` is the atom's element count -- the ``ATOM_V`` the tiled
    copy already settled on -- which is exactly what
    :func:`quack.copy_utils.copy` reads on the CuTe side.

    A masked lane of a predicated load keeps whatever the fragment held before,
    where CuTe's ``cp.async`` zero-fills it, so the destination is cleared first:
    an out-of-bounds column has to contribute zero to the row's reduction rather
    than the previous tile's value.
    """
    src, dst = tile_of(src, tile), tile_of(dst, tile)
    width = src.element_type.width
    if pred is not None:
        pred = pred.at(tile, width)
        if dst.address_space == fx.AddressSpace.Register:
            dst.fill(0)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(_shape(src)[0][0] * width, 0), width)
    fx.copy(atom, src, dst, pred=pred)


def thread_coords(thr_copy: ThrCopy2d, block_row, tiler_mn) -> "ThreadCoords":
    """Where this thread's slice of the tile sits in the tensor.

    CuTe reads this off an identity tensor put through the same partitioner.
    FlyDSL cannot: it offsets a partitioned coordinate tensor by the thread's
    coordinate in the *thread* layout rather than by the element that thread
    starts at, so every column comes out divided by the value layout's extent.
    (FlyDSL's ``examples/01-vectorAdd.py`` predicates off that tensor and still
    passes, because its unmasked stores land on the wrapped-around next row
    carrying the value that row wanted.) The same numbers follow from the
    geometry, so they are derived rather than read back.
    """
    return ThreadCoords(thr_copy, block_row, tiler_mn)


class ThreadCoords:
    """The (row, column) this thread's tile slice starts at, and its atoms'.

    ``tiler_mn`` is the tile :func:`local_tile` cut, which may be several passes
    of the thread layout wide; the extra passes are the ``rest_n`` mode that
    ``partition_S`` leaves on the result.
    """

    __slots__ = ("col", "row", "tiled_copy", "tiler_mn")

    def __init__(self, thr_copy: ThrCopy2d, block_row, tiler_mn):
        tiled_copy = thr_copy.tiled_copy
        threads_per_row = fx.Int32(tiled_copy.threads_per_row)
        thr_idx = fx.Int32(thr_copy.thr_idx)
        self.tiled_copy = tiled_copy
        self.tiler_mn = tiler_mn
        self.row = fx.Int32(block_row) * fx.Int32(tiler_mn[0]) + thr_idx // threads_per_row
        self.col = (thr_idx % threads_per_row) * fx.Int32(tiled_copy.num_copy_elems)

    @property
    def rest_n(self) -> int:
        return self.tiler_mn[1] // self.tiled_copy.cols_per_block

    def atom_col(self, width: int, rest_v: int, rest_n: int, tile):
        """First column of the atom at ``(rest_v, rest_n)`` of tile ``tile``.

        A thread owns ``num_copy_elems`` contiguous columns per ``rest_n`` block,
        split into atoms of ``ATOM_V``; consecutive blocks are one pass of the
        thread layout apart.
        """
        tiled_copy = self.tiled_copy
        within = rest_n * tiled_copy.cols_per_block + rest_v * _atom_elems(
            width, tiled_copy.num_copy_elems
        )
        return self.col + fx.Int32(tile) * fx.Int32(self.tiler_mn[1]) + fx.Int32(within)


def predicate_k(coords: ThreadCoords, limit) -> "TilePredicate":
    """Column predicates for the tile this thread owns.

    Only the N ("k") dimension is predicated, as in the CuTe original; the M
    dimension is guarded by an ``if`` around the copy.
    """
    return TilePredicate(coords, limit)


class TilePredicate:
    """The N-dimension predicate for one tile of a row, at one operand width.

    CuTe returns a fragment here, because its tile spans the whole row and one
    set of bits covers it. A row walked in several tiles needs a set per tile,
    built on demand so only the tile being copied has its bits live; repeats
    within a tile are identical compares that the backend folds together.

    Per width for the same reason :class:`TiledCopy2d` is: a 32-bit atom spans
    half the columns a 16-bit one does, so it needs twice the bits over the same
    tile.
    """

    __slots__ = ("coords", "limit")

    def __init__(self, coords: ThreadCoords, limit):
        self.coords = coords
        self.limit = limit

    def at(self, tile, width: int):
        vec = self.coords.tiled_copy.num_copy_elems
        rest_v, rest_n = vec // _atom_elems(width, vec), self.coords.rest_n
        # Compact, unlike CuTe's stride-0 rest_m: FlyDSL reads the predicate back
        # as one vector, so a mode that aliases leaves it short of the atom count.
        tApA = fx.make_rmem_tensor(
            fx.make_ordered_layout((rest_v, 1, rest_n), order=(0, 1, 2)), fx.Boolean
        )
        for v in fx.range_constexpr(rest_v):
            for k in fx.range_constexpr(rest_n):
                tApA[v, 0, k] = self.coords.atom_col(width, v, k, tile) < fx.Int32(self.limit)
        return tApA


def load(fragment, dtype=None):
    """A fragment's values as one flat vector, optionally converted.

    The flattening is what lets two operands of the same tile meet in an
    expression: a 32-bit fragment arrives shaped ``((4, 2), ...)`` where a
    16-bit one is ``((8, 1), ...)``, and the reshape is pure metadata over the
    same ``vector<Nxty>``.
    """
    value = fragment.load().reshape(fx.size(fragment.shape).unpack())
    return value if dtype is None else value.to(dtype)


def store(fragment, value) -> None:
    """Write a flat vector back into a fragment, converting to its element type."""
    fragment.store(value.to(fragment.element_type).reshape(_shape(fragment)))

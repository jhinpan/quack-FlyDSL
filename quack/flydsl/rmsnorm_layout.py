# Copyright (c) 2026, Tri Dao.

"""Layout canonicalization for the FlyDSL RMSNorm tensor ABI."""

import torch


def _packed_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Copy only when a row is not already contiguous along its last axis.

    Every operand reaches the kernels as a row-scoped buffer descriptor, so the
    requirement is that each row be contiguous, not that the whole tensor be. A
    row-padded view (``full[:, :n]``, stride ``(pitch, 1)``) already satisfies
    it and the descriptor is sized to ``n``, so the padding is never addressed;
    copying such a view would cost a full extra read and write of the
    activation for nothing.

    Under ``torch.compile`` the copy is unconditional: the predicate below
    inspects strides, which are not available on a symbolic tensor.

    Canonicalising here rather than adding a layout term to the forward
    launcher cache is deliberate. Two views with the same shape and dtype but
    different layouts must not share a launcher, and making them share one
    canonical layout is a stronger fix than making them miss the cache.
    """
    tensor = _unambiguous_layout(tensor)
    if torch.compiler.is_compiling():
        return tensor.contiguous()
    return tensor if _rows_are_disjoint_and_packed(tensor) else tensor.contiguous()


def _unambiguous_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the first unit-stride axis is the row axis.

    FlyDSL picks the leading dimension as ``next(i for i in range(ndim) if
    stride[i] == 1)``. A size-1 axis ahead of the row that also carries stride 1
    wins that search and silently redefines the ABI. ``.contiguous()`` cannot
    fix it: such a tensor already reports ``is_contiguous()``, so the copy is
    the identity.

    Relabelling is enough and costs nothing. A size-1 axis has no observable
    stride -- there is no second element to step to -- so squeezing it out and
    putting it back rewrites the stride to a non-unit value describing the same
    memory. The reinserted stride can be 0, which is fine here: the axis no
    longer holds the row's unit stride, and the packing predicate rejects zero
    strides separately.
    """
    row = tensor.dim() - 1
    offenders = [
        axis for axis in range(row) if tensor.shape[axis] == 1 and tensor.stride(axis) == 1
    ]
    if not offenders:
        return tensor
    for axis in reversed(offenders):
        tensor = tensor.squeeze(axis)
    for axis in offenders:
        tensor = tensor.unsqueeze(axis)
    return tensor


def _rows_are_disjoint_and_packed(tensor: torch.Tensor) -> bool:
    """Whether every row is packed and no two rows share storage.

    Row-scoped descriptors are safe exactly when each row is contiguous and
    distinct rows do not overlap. A broadcast or reversed view satisfies
    neither, and both are silently wrong rather than loud, so they are copied.
    """
    if tensor.stride(-1) != 1:
        return False
    span = tensor.shape[-1]
    for stride, size in sorted(
        zip(tensor.stride()[:-1], tensor.shape[:-1]), key=lambda axis: axis[0]
    ):
        if stride < span:
            return False
        span = stride * (size - 1) + span
    return True

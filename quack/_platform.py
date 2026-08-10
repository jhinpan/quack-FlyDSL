# Copyright (c) 2026, Tri Dao.

"""Backend detection shared by the CuTe and FlyDSL entry points."""

import torch

__all__ = ["is_rocm"]


def is_rocm() -> bool:
    """True when CuTe is unavailable because torch is a ROCm build.

    Strictly this asks "is torch built for AMD", not "can cutlass be imported".
    The two coincide in practice: torch's CUDA and ROCm builds are mutually
    exclusive, and QuACK's CuTe kernels only target SM90/100/120. Kept as a
    single definition so the predicate can be refined in one place instead of
    across the package bootstrap, the pytest plugin, and the FlyDSL kernels.
    """
    return torch.version.hip is not None

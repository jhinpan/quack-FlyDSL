__version__ = "0.6.1"

import torch


if torch.version.hip is None:
    import os

    import quack.dsl as _quack_dsl  # noqa: F401

    if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
        from quack.dsl import cute_dsl_ptxas as _cute_dsl_ptxas

        # Patch before importing any modules that instantiate CuTeDSL. The patch
        # forces PTX dumping so the CUDA library loader can replace CUTLASS DSL's
        # embedded ptxas-library cubin with one assembled by system ptxas.
        _cute_dsl_ptxas.patch()

    # Pythonic CuTe tensor indexing (`:` / `...` sugar) is installed as a side effect
    # of importing `quack.dsl`, which imports `quack.dsl.cute_tensor_indexing` and
    # monkey-patches CuTe's tensor classes process-wide.
    from quack.rmsnorm import rmsnorm  # noqa: E402
    from quack.softmax import softmax  # noqa: E402
    from quack.cross_entropy import cross_entropy  # noqa: E402
    from quack.rounding import RoundingMode  # noqa: E402

    __all__ = [
        "rmsnorm",
        "softmax",
        "cross_entropy",
        "RoundingMode",
    ]
else:
    # The CuTe kernels need cutlass, which is CUDA-only. ROCm users reach the
    # FlyDSL backend explicitly through quack.rmsnorm_flydsl; exporting nothing
    # keeps `from quack import *` from re-exporting the torch imported above.
    __all__ = []

    _CUDA_ONLY = ("rmsnorm", "softmax", "cross_entropy", "RoundingMode")

    def __getattr__(name):
        # Without this, `from quack import rmsnorm` on ROCm falls through to
        # importing the quack.rmsnorm submodule and surfaces as
        # "No module named 'cuda'", which names neither the cause nor the fix.
        if name in _CUDA_ONLY:
            raise AttributeError(
                f"quack.{name} is a CuTe kernel and needs CUDA; this is a ROCm build. "
                "The FlyDSL RMSNorm backend is at quack.rmsnorm_flydsl."
            )
        raise AttributeError(f"module 'quack' has no attribute '{name}'")

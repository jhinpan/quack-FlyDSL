# ruff: noqa: I001, RUF022

import os

import torch

__version__ = "0.6.1"


if torch.version.hip is None:
    import quack.dsl as _quack_dsl  # noqa: F401

    if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
        from quack.dsl import cute_dsl_ptxas as _cute_dsl_ptxas

        # Patch before importing any modules that instantiate CuTeDSL. The patch
        # forces PTX dumping so the CUDA library loader can replace CUTLASS DSL's
        # embedded ptxas-library cubin with one assembled by system ptxas.
        _cute_dsl_ptxas.patch()

    # Pythonic CuTe tensor indexing (`:` / `...` sugar) is installed as a side effect
    # of importing `quack.dsl`, which imports `quack.dsl.cute_tensor_indexing` and
    # monkey-patches CuTe's tensor classes process-wide. The import and __all__
    # order is a tested part of this eager CUDA bootstrap.
    from quack.rmsnorm import rmsnorm
    from quack.softmax import softmax
    from quack.cross_entropy import cross_entropy
    from quack.rounding import RoundingMode

    __all__ = [
        "rmsnorm",
        "softmax",
        "cross_entropy",
        "RoundingMode",
    ]
else:
    # The CuTe kernels need cutlass, which is CUDA-only. Resolve RMSNorm lazily
    # so importing quack remains valid without the optional FlyDSL dependency.
    __all__ = ["rmsnorm"]

    _CUDA_ONLY = ("softmax", "cross_entropy", "RoundingMode")

    def __getattr__(name):
        if name == "rmsnorm":
            try:
                from quack.rmsnorm_flydsl import rmsnorm as _rmsnorm
            except ModuleNotFoundError as exc:
                if exc.name != "flydsl":
                    raise
                raise ModuleNotFoundError(
                    "quack.rmsnorm uses the FlyDSL backend on ROCm; install it with "
                    "\"pip install 'quack-kernels[flydsl]'\"",
                    name="flydsl",
                ) from exc
            # Return the backend function itself, rather than wrapping every
            # call, and cache it so its identity remains stable.
            globals()["rmsnorm"] = _rmsnorm
            return _rmsnorm
        if name in _CUDA_ONLY:
            raise AttributeError(
                f"quack.{name} is a CuTe kernel and needs CUDA; this is a ROCm build."
            )
        raise AttributeError(f"module 'quack' has no attribute '{name}'")

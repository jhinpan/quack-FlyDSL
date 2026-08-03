# Copyright (c) 2026, Tri Dao.

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_python(
    source: str, *, cwd: Path = ROOT, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run ``source`` in a fresh interpreter.

    ``check=False`` is for callers that treat the exit code as the result
    rather than as a precondition; they must inspect ``returncode``
    themselves. Every other caller keeps the default and gets the stdout and
    stderr of a failed child in the assertion message.
    """
    env = os.environ.copy()
    pythonpath = [str(ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def _is_rocm_build() -> bool:
    result = _run_python(
        """
        import torch

        print(torch.version.hip is not None)
        """
    )
    return result.stdout.strip() == "True"


def test_real_rocm_import_skips_cuda_bootstrap_without_initializing_context():
    if not _is_rocm_build():
        pytest.skip("requires a real ROCm PyTorch build")

    _run_python(
        """
        import importlib.abc
        import sys

        import torch


        class NoCutlassImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    raise AssertionError(f"unexpected CUDA dependency import: {fullname}")
                return None


        sys.meta_path.insert(0, NoCutlassImports())
        assert not torch.cuda.is_initialized()
        assert torch.version.hip is not None
        assert not torch.cuda.is_initialized()

        import quack

        assert quack.__version__
        assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
        assert not torch.cuda.is_initialized()
        """
    )


def test_flydsl_rmsnorm_import_does_not_load_benchmark_extras():
    if not _is_rocm_build():
        pytest.skip("requires a real ROCm PyTorch build")

    _run_python(
        """
        import sys

        import torch

        assert torch.version.hip is not None
        assert "pandas" not in sys.modules
        assert "tyro" not in sys.modules
        assert "triton" not in sys.modules

        import quack.rmsnorm_flydsl

        assert callable(quack.rmsnorm_flydsl.rmsnorm)
        assert "pandas" not in sys.modules
        assert "tyro" not in sys.modules
        assert "triton" not in sys.modules
        assert not any(
            name == "cutlass" or name.startswith("cutlass.") for name in sys.modules
        )
        """
    )


def test_simulated_cuda_preserves_eager_bootstrap_order_and_exports():
    _run_python(
        """
        import importlib.abc
        import importlib.util
        import os
        import sys

        import torch


        events = []
        exports = {}


        class PtxasPatch:
            @staticmethod
            def patch():
                events.append("patch_ptxas")


        class CudaModuleLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                name = module.__name__
                events.append(f"import_{name.removeprefix('quack.')}")
                if name == "quack.dsl":
                    module.cute_dsl_ptxas = PtxasPatch
                elif name == "quack.rounding":
                    module.RoundingMode = type("RoundingMode", (), {})
                    exports["RoundingMode"] = module.RoundingMode
                else:
                    export_name = name.removeprefix("quack.")

                    def exported(*args, **kwargs):
                        return export_name, args, kwargs

                    exported.__name__ = export_name
                    setattr(module, export_name, exported)
                    exports[export_name] = exported


        class CudaModuleFinder(importlib.abc.MetaPathFinder):
            names = {
                "quack.dsl",
                "quack.rmsnorm",
                "quack.softmax",
                "quack.cross_entropy",
                "quack.rounding",
            }

            def find_spec(self, fullname, path=None, target=None):
                if fullname in self.names:
                    return importlib.util.spec_from_loader(
                        fullname,
                        CudaModuleLoader(),
                        is_package=fullname == "quack.dsl",
                    )
                return None


        torch.version.hip = None
        os.environ["CUTE_DSL_PTXAS_PATH"] = "/fake/ptxas"
        sys.meta_path.insert(0, CudaModuleFinder())

        import quack

        assert events == [
            "import_dsl",
            "patch_ptxas",
            "import_rmsnorm",
            "import_softmax",
            "import_cross_entropy",
            "import_rounding",
        ]
        assert quack.__all__ == ["rmsnorm", "softmax", "cross_entropy", "RoundingMode"]
        assert quack.rmsnorm is exports["rmsnorm"]
        assert quack.softmax is exports["softmax"]
        assert quack.cross_entropy is exports["cross_entropy"]
        assert quack.RoundingMode is exports["RoundingMode"]
        """
    )


class CutedslGateStillCouplesFlydsl(AssertionError):
    """The one failure that counts as the expected xfail below.

    A distinct type, and ``raises=`` on the marker, because @Reviewer measured
    the hole in the first version: ``xfail(strict=True)`` with no ``raises``
    records ANY failure of the test as the same XFAIL. He replaced the child
    body with an unrelated ``RuntimeError`` and still got ``1 xfailed`` /
    ``4 passed, 1 xfailed`` -- so my claim that "the parent refuses any other
    exit code" was true of the parent's assertions and false of what pytest
    reported, which is the only place a reader looks. Reproduced here before
    fixing: same result.

    With ``raises=`` set to this type, the third-outcome guard raises a plain
    ``AssertionError``, which is no longer the expected exception and so
    surfaces as a real failure instead of hiding inside the xfail.
    """


@pytest.mark.xfail(
    strict=True,
    raises=CutedslGateStillCouplesFlydsl,
    reason=(
        "DESIRED BEHAVIOUR, NOT CURRENT BEHAVIOUR. FlyDSL has no cutlass "
        "dependency, so importing quack.rmsnorm_flydsl should survive a broken "
        "cutedsl chain. Today it does not: quack/__init__.py:6 runs the CuTe "
        "bootstrap unconditionally on CUDA. Written as a strict xfail per "
        "@Reviewer, so that repairing the import boundary turns this GREEN "
        "instead of red -- a plain green assertion on the broken behaviour "
        "would make the fix look like a regression. raises= is narrowed to "
        "CutedslGateStillCouplesFlydsl so that an unrelated failure cannot be "
        "absorbed as this expected one."
    ),
)
def test_simulated_cuda_flydsl_import_survives_a_broken_cutedsl_chain():
    """The gate at ``quack/__init__.py:6`` is one-directional, and this pins it.

    On ROCm the ``torch.version.hip is None`` gate is what keeps the CuTe
    bootstrap out of the way, so ``import quack.rmsnorm_flydsl`` works with no
    cutlass installed at all. On a CUDA box the same gate runs that bootstrap
    unconditionally -- so if the installed cutlass does not match what
    ``quack/pipeline.py`` imports (e.g. ``alloc_reserved_mbarrier``, absent
    before nvidia-cutlass-dsl 4.6.0), then ``import quack.rmsnorm_flydsl``
    fails inside ``quack/__init__.py`` before Python ever looks for the FlyDSL
    module. The FlyDSL backend has no cutlass dependency and is brought down
    by a package it does not use.

    This is a REAL failure mode, not a hypothetical: it is exactly what
    hyper00/hyper01's system interpreters do, both carrying cutlass 4.5.2.

    Why the test is written as a simulation rather than skipped on ROCm: I
    claimed in AI/flydsl_rmsnorm_notes.md that this direction was
    "structurally untestable on this hardware" because ROCm takes the other
    branch. That was wrong in the same way as several things above it -- the
    branch is chosen by ``torch.version.hip``, which the sibling test at
    :77 already overrides, and the failing cutedsl import is a meta_path
    loader away. Untestable on this hardware and untested on this hardware are
    different claims, and I had asserted the stronger one.

    The assertion is deliberately about the ERROR, not just the failure: a
    bare "it raises" would also pass if FlyDSL were merely missing. What is
    being pinned is that the cutedsl error propagates out of a FlyDSL import.

    Why the assertion states the DESIRED outcome and the marker carries the
    current one: @Reviewer's objection to the first version, and it is the
    same defect I had just audited in ``f747907``. A plain green test asserting
    that the import FAILS would make the obvious repair -- decoupling the
    FlyDSL path from the cutedsl bootstrap -- show up as a red test, i.e. the
    fix would look like the regression. Under ``xfail(strict=True)`` the file
    reads as "this SHOULD work", XFAILs today, and turns XPASS -> failure the
    moment the boundary is fixed, which is the signal that the successor
    landed rather than a wall in front of it.

    The ImportError branch is what currently runs, and it still checks the
    message rather than the bare fact of raising, because an xfail that
    triggers for the wrong reason is no better than a green one.
    """
    result = _run_python(
        """
        import importlib.abc
        import importlib.util
        import sys

        import torch


        MESSAGE = "cannot import name 'alloc_reserved_mbarrier' from 'cutlass.pipeline'"


        class BrokenCutedslLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise ImportError(MESSAGE)


        class BrokenCutedslFinder(importlib.abc.MetaPathFinder):
            # quack.dsl too: on a real CUDA box it is imported first and would
            # fail on the same missing cutlass, so intercepting only
            # quack.rmsnorm would let the run die with a less specific error
            # and the assertion below would pass for the wrong reason.
            names = {"quack.dsl", "quack.rmsnorm"}

            def find_spec(self, fullname, path=None, target=None):
                if fullname in self.names:
                    return importlib.util.spec_from_loader(fullname, BrokenCutedslLoader())
                return None


        torch.version.hip = None
        sys.meta_path.insert(0, BrokenCutedslFinder())

        # The desired behaviour: FlyDSL does not import cutlass, so a broken
        # cutedsl chain should be irrelevant to it. Today the package gate
        # makes it fatal. Exit 3 is reserved for "failed, and failed for
        # exactly the cutedsl reason" so the caller can tell that apart from
        # an unrelated breakage, which exits 1 with a traceback.
        try:
            import quack.rmsnorm_flydsl
        except ImportError as exc:
            assert MESSAGE in str(exc), f"wrong failure: {exc!r}"
            assert "quack.rmsnorm_flydsl" not in sys.modules
            raise SystemExit(3)

        # Callable, not merely present in sys.modules: a half-initialised
        # module object would satisfy the weaker check.
        assert callable(quack.rmsnorm_flydsl.rmsnorm)
        """,
        check=False,
    )

    # A plain AssertionError, deliberately NOT the expected-xfail type: an
    # unrelated breakage must surface as a real failure rather than be absorbed
    # as the expected one. This is the assertion @Reviewer's mutation escapes
    # through when raises= is absent.
    assert result.returncode in (0, 3), (
        "the simulation broke for a reason that is neither outcome it "
        f"distinguishes (rc={result.returncode}):\n" + result.stdout + result.stderr
    )
    # Today: 3, raised as the expected type -> XFAIL. When the import boundary
    # is fixed: 0, the test passes, and strict=True turns that pass into a
    # failure telling you to flip it.
    if result.returncode != 0:
        raise CutedslGateStillCouplesFlydsl(
            "import quack.rmsnorm_flydsl still dies inside the cutedsl "
            "bootstrap it does not depend on (quack/__init__.py:6)"
        )


def test_pytest_plugin_collects_on_rocm_without_cutlass(tmp_path):
    if not _is_rocm_build():
        pytest.skip("requires a real ROCm PyTorch build")

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(
        'pytest_plugins = ["quack.testing.pytest_plugin"]\n',
        encoding="utf-8",
    )
    (suite / "test_collection.py").write_text(
        textwrap.dedent(
            """
            import sys

            import torch


            assert torch.version.hip is not None
            assert not torch.cuda.is_initialized()
            assert not any(
                name == "cutlass" or name.startswith("cutlass.") for name in sys.modules
            )


            def test_collected():
                pass
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = [str(ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=suite,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 test collected" in output


def test_pytest_plugin_preserves_non_compile_outcomes_without_cutlass(tmp_path):
    if not _is_rocm_build():
        pytest.skip("requires a real ROCm PyTorch build")

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(
        'pytest_plugins = ["quack.testing.pytest_plugin"]\n',
        encoding="utf-8",
    )
    (suite / "test_outcomes.py").write_text(
        textwrap.dedent(
            """
            import pytest


            def test_failure():
                raise RuntimeError("expected ordinary failure")


            def test_skip():
                pytest.skip("expected ordinary skip")
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = [str(ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=suite,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "RuntimeError: expected ordinary failure" in output
    assert "1 failed, 1 skipped" in output
    assert "cutlass" not in output

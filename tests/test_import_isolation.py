# Copyright (c) 2026, Tri Dao.

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_python(
    source: str, *, cwd: Path = ROOT, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run ``source`` in a fresh interpreter."""
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
        check=False,
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
        assert quack.__all__ == ["rmsnorm"]
        assert "quack.rmsnorm_flydsl" not in sys.modules
        assert not any(name == "flydsl" or name.startswith("flydsl.") for name in sys.modules)
        assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
        assert not torch.cuda.is_initialized()
        """
    )


def test_simulated_rocm_import_skips_cuda_bootstrap_without_initializing_context():
    _run_python(
        """
        import importlib.abc
        import sys

        import torch


        class NoCudaBootstrap(importlib.abc.MetaPathFinder):
            blocked = {
                "quack.dsl",
                "quack.rmsnorm",
                "quack.softmax",
                "quack.cross_entropy",
                "quack.rounding",
            }

            def find_spec(self, fullname, path=None, target=None):
                if fullname in self.blocked or fullname == "cutlass" or fullname.startswith(
                    "cutlass."
                ):
                    raise AssertionError(f"unexpected CUDA bootstrap import: {fullname}")
                return None


        torch.version.hip = "simulated-rocm"
        sys.meta_path.insert(0, NoCudaBootstrap())
        assert not torch.cuda.is_initialized()

        import quack

        assert quack.__version__
        assert quack.__all__ == ["rmsnorm"]
        assert "quack.rmsnorm_flydsl" not in sys.modules
        assert not any(name in NoCudaBootstrap.blocked for name in sys.modules)
        assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
        assert not torch.cuda.is_initialized()
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


def _write_inactive_plugin_guard(suite: Path) -> None:
    (suite / "conftest.py").write_text(
        textwrap.dedent(
            """
            import importlib.abc
            import sys

            import torch


            class NoInactiveCudaImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "quack.cache.async_compile":
                        raise AssertionError(f"unexpected inactive CUDA import: {fullname}")
                    if fullname == "cutlass" or fullname.startswith("cutlass."):
                        raise AssertionError(f"unexpected CUDA dependency import: {fullname}")
                    return None


            sys.meta_path.insert(0, NoInactiveCudaImports())
            assert torch.version.hip is not None
            assert not torch.cuda.is_initialized()
            pytest_plugins = ["quack.testing.pytest_plugin"]


            def pytest_unconfigure(config):
                assert "quack.cache.async_compile" not in sys.modules
                assert not torch.cuda.is_initialized()
            """
        ),
        encoding="utf-8",
    )


def _run_pytest_suite(suite: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = [str(ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), *args, "-p", "no:cacheprovider"],
        cwd=suite,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_pytest_plugin_collects_and_tears_down_without_async_compile(tmp_path):
    if not _is_rocm_build():
        pytest.skip("requires a real ROCm PyTorch build")

    suite = tmp_path / "suite"
    suite.mkdir()
    _write_inactive_plugin_guard(suite)
    (suite / "test_collection.py").write_text(
        "def test_collected():\n    pass\n",
        encoding="utf-8",
    )

    result = _run_pytest_suite(suite, "--collect-only", "-q")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 test collected" in output
    assert "unexpected inactive CUDA import" not in output


def test_pytest_plugin_inactive_hooks_preserve_non_compile_outcomes(tmp_path):
    if not _is_rocm_build():
        pytest.skip("requires a real ROCm PyTorch build")

    suite = tmp_path / "suite"
    suite.mkdir()
    _write_inactive_plugin_guard(suite)
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

    result = _run_pytest_suite(suite, "-q")
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "RuntimeError: expected ordinary failure" in output
    assert "1 failed, 1 skipped" in output
    assert "unexpected inactive CUDA import" not in output

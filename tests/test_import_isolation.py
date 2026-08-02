# Copyright (c) 2026, Tri Dao.

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Prepended to every child. ``-I`` drops PYTHONPATH, so the tree under test has
# to be named explicitly -- which is the stronger arrangement anyway.
_PREAMBLE = f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"


def _run_python(
    source: str, *, cwd: Path = ROOT, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run ``source`` in a fresh interpreter, with a fresh package state.

    ``check=False`` is for callers that treat the exit code as the result
    rather than as a precondition; they must inspect ``returncode``
    themselves. Every other caller keeps the default and gets the stdout and
    stderr of a failed child in the assertion message.

    ``-I`` is what makes the child's *package* state fresh rather than merely
    its interpreter, and it is not optional. @Reviewer measured the difference:
    with a ``sitecustomize.py`` on the inherited ``PYTHONPATH`` containing
    nothing but ``import quack``, the simulation below reported
    ``XPASS(strict)`` against a completely unmodified ``quack/__init__.py`` --
    a green "the boundary is fixed" signal with no fix anywhere. A new
    interpreter is not a new package state. Since ``-I`` also ignores
    ``PYTHONPATH``, ROOT goes onto ``sys.path`` from inside the child, which
    additionally pins these tests to *this* tree rather than to whichever
    ``quack`` the ambient environment resolves first.
    """
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-I", "-c", _PREAMBLE + textwrap.dedent(source)],
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
        "@Reviewer, so that repairing the import boundary produces "
        "XPASS(strict) -- a red result whose meaning is FLIP THIS MARKER, "
        "not a regression. (The earlier wording here said the repair turns "
        "the test GREEN. The desired assertion does pass, but strict=True "
        "deliberately makes the pytest outcome a failure until the marker "
        "is removed; @Reviewer caught the reason contradicting the "
        "docstring below it.) raises= is narrowed to "
        "CutedslGateStillCouplesFlydsl, and the child now authenticates its "
        "own branch, so neither an unrelated exception nor an unrelated "
        "exit 3 can be absorbed as this expected one."
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
    :150 already overrides, and the failing cutedsl import is a meta_path
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
    reads as "this SHOULD work", XFAILs today, and turns into a failing
    XPASS(strict) the moment the boundary is fixed -- red, but red meaning
    "flip this marker", which is the signal that the successor landed rather
    than a wall in front of it.

    Three things this version does that its predecessor did not, each because
    @Reviewer measured the gap rather than reading the reason text:

    1. The child names its own branch and the parent checks that name. An exit
       code crossing a process boundary authenticates nothing: he replaced the
       child body with a bare ``SystemExit(3)``, which collided with the
       reserved code and was converted straight into the expected exception --
       ``1 xfailed``, pytest exit 0, for a reason this test does not describe.
    2. The failure travels the real chain. ``alloc_reserved_mbarrier`` is
       imported at exactly one place, ``quack/pipeline.py:13``, reached from
       ``quack/rmsnorm.py:24``. The predecessor fabricated that error at
       ``quack.dsl`` -- which imports ``cutlass.cute*`` and never
       ``cutlass.pipeline``, so it never requests the missing name -- and
       ``quack.dsl`` is imported first, so every run passed at a boundary the
       real chain does not fail at.
    3. Reach is recorded, not inferred. ``"quack.rmsnorm_flydsl" not in
       sys.modules`` was never evidence that Python failed to reach the target;
       the import machinery deletes a module whose body raised.
    """
    result = _run_python(
        """
        import importlib.abc
        import importlib.machinery
        import importlib.util
        import json
        import sys


        # Preload precondition, checked before anything else can mask it. Under
        # -I this should be impossible; asserting it anyway means a future
        # change that drops -I fails loudly rather than reporting a fixed
        # boundary that was never broken in the child.
        assert "quack" not in sys.modules, "quack was preloaded; the child is not hermetic"

        import torch


        MESSAGE = "cannot import name 'alloc_reserved_mbarrier' from 'cutlass.pipeline'"
        REACHED = []


        class WorkingDslLoader(importlib.abc.Loader):
            \"\"\"``quack.dsl`` SUCCEEDS, because in the real tree it does.

            The predecessor made this module raise the missing-symbol error
            too, on the stated grounds that it "would fail on the same missing
            cutlass". @Reviewer checked that against the tree and it is false:
            ``quack/dsl/`` imports ``cutlass.cute*`` and never
            ``cutlass.pipeline``, so it does not request the missing name. It
            was also imported first, which means the predecessor's ImportError
            always came from here -- the assertion passed on a boundary the
            real chain does not fail at. Verified again while writing this: the
            fabricated error was raised by ``quack.dsl`` on every run.

            It is stubbed rather than left real only because this box has no
            cutlass at all; what it stubs is a module that succeeds.
            \"\"\"

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                REACHED.append("quack.dsl")
                module.cute_op = lambda *a, **k: (lambda f: f)
                module.__path__ = []


        class RmsnormLoader(importlib.abc.Loader):
            \"\"\"Replays ``quack/rmsnorm.py``'s own import order to the failing edge.

            The real module reaches ``quack.pipeline`` at :24. Its earlier
            lines pull in ``cuda.bindings.driver`` (:7) and ``cutlass.cute``
            (:9-13), neither installed here, so those are what the stub stands
            in for -- not the edge under test, which is exercised below.
            \"\"\"

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                REACHED.append("quack.rmsnorm")
                import quack.pipeline  # noqa: F401


        class MissingSymbolLoader(importlib.abc.Loader):
            \"\"\"``quack/pipeline.py:13``, the one line this test is about.

            That line is ``from cutlass.pipeline import agent_sync,
            alloc_reserved_mbarrier``, and on cutlass 4.5.2 the second name is
            absent. ``name=`` is set so the exception carries the same metadata
            a real one would.
            \"\"\"

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                REACHED.append("quack.pipeline")
                raise ImportError(MESSAGE, name="cutlass.pipeline")


        class TargetRecordingLoader(importlib.abc.Loader):
            \"\"\"Stands in for quack.rmsnorm_flydsl and records that it was reached.

            Two things @Reviewer showed the previous version could not do.
            ``"quack.rmsnorm_flydsl" not in sys.modules`` proves nothing about
            reach -- the import machinery deletes a module whose body raised --
            so reach is recorded here on entry instead. And supplying
            ``rmsnorm`` from the stub decouples the success path from whether
            this host happens to have the optional FlyDSL package installed;
            the previous version's XPASS depended on it, which is why blocking
            flydsl turned the repaired boundary into an ordinary rc=1.
            \"\"\"

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                REACHED.append("quack.rmsnorm_flydsl")
                module.rmsnorm = lambda *a, **k: None


        class Finder(importlib.abc.MetaPathFinder):
            LOADERS = {
                "quack.dsl": WorkingDslLoader,
                "quack.rmsnorm": RmsnormLoader,
                "quack.pipeline": MissingSymbolLoader,
                "quack.rmsnorm_flydsl": TargetRecordingLoader,
            }

            def find_spec(self, fullname, path=None, target=None):
                loader = self.LOADERS.get(fullname)
                if loader is not None:
                    return importlib.util.spec_from_loader(fullname, loader())
                return None


        torch.version.hip = None
        sys.meta_path.insert(0, Finder())

        outcome = {"branch": None, "reached": None, "message_ok": None}
        try:
            import quack.rmsnorm_flydsl
        except ImportError as exc:
            outcome["branch"] = "cutedsl_gate_still_couples_flydsl"
            outcome["message_ok"] = MESSAGE in str(exc)
            outcome["reached"] = REACHED
            print("SENTINEL " + json.dumps(outcome))
            raise SystemExit(3)

        # Callable, not merely present in sys.modules: a half-initialised
        # module object would satisfy the weaker check.
        assert callable(quack.rmsnorm_flydsl.rmsnorm)
        outcome["branch"] = "flydsl_import_survived"
        outcome["reached"] = REACHED
        print("SENTINEL " + json.dumps(outcome))
        """,
        check=False,
    )

    # Every assertion below is a plain AssertionError, deliberately NOT the
    # expected-xfail type: an unrelated breakage must surface as a real failure
    # rather than be absorbed as the expected one.
    #
    # The exit code alone is not evidence of anything. @Reviewer's second
    # mutation replaced the child's ImportError with a bare SystemExit(3),
    # which collided with the reserved code and was converted straight into the
    # expected exception -- 1 xfailed, pytest exit 0, for a reason the test does
    # not describe. A cross-process exit status is not an authenticated source,
    # so the branch identifies itself in the child's own words and the parent
    # checks that instead.
    assert result.returncode in (0, 3), (
        "the simulation broke for a reason that is neither outcome it "
        f"distinguishes (rc={result.returncode}):\n" + result.stdout + result.stderr
    )
    sentinels = [ln for ln in result.stdout.splitlines() if ln.startswith("SENTINEL ")]
    assert len(sentinels) == 1, (
        "the child did not report exactly one outcome, so its exit code is "
        f"unattributable:\n{result.stdout}{result.stderr}"
    )
    outcome = json.loads(sentinels[0][len("SENTINEL ") :])

    if result.returncode == 3:
        assert outcome["branch"] == "cutedsl_gate_still_couples_flydsl", (
            f"exit 3 from an unexpected branch {outcome['branch']!r}"
        )
        assert outcome["message_ok"], (
            "the import failed, but not with the missing-symbol error this "
            f"test is about:\n{result.stdout}{result.stderr}"
        )
        assert outcome["reached"] == ["quack.dsl", "quack.rmsnorm", "quack.pipeline"], (
            "the failure did not travel the real chain (quack.dsl succeeds, "
            "then quack.rmsnorm:24 -> quack/pipeline.py:13): "
            f"reached {outcome['reached']!r}"
        )
        raise CutedslGateStillCouplesFlydsl(
            "import quack.rmsnorm_flydsl still dies inside the cutedsl "
            "bootstrap it does not depend on (quack/__init__.py:6)"
        )

    # rc=0: the boundary is repaired. strict=True turns this pass into a
    # failure telling you to flip the marker.
    assert outcome["branch"] == "flydsl_import_survived", (
        f"exit 0 from an unexpected branch {outcome['branch']!r}"
    )
    assert "quack.rmsnorm_flydsl" in outcome["reached"], (
        "the import succeeded without ever reaching the target module: "
        f"reached {outcome['reached']!r}"
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

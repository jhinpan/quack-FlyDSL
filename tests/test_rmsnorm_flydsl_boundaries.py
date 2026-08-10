# Copyright (c) 2026, Tri Dao.

"""Module-boundary coverage for the FlyDSL RMSNorm backend."""

import ast
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FLYDSL_ROOT = ROOT / "quack" / "flydsl"


def _imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = f"quack.flydsl.{module}"
            modules.add(module.rstrip("."))
    return modules


def test_backend_modules_depend_inward_without_importing_the_facade():
    modules = {
        name: _imported_modules(FLYDSL_ROOT / f"{name}.py")
        for name in ("rmsnorm_preflight", "rmsnorm_launch")
    }

    assert all("quack.rmsnorm_flydsl" not in imports for imports in modules.values())
    assert "quack.flydsl.rmsnorm_launch" not in modules["rmsnorm_preflight"]
    assert "quack.flydsl.rmsnorm_preflight" in modules["rmsnorm_launch"]


def test_host_boundaries_import_without_loading_flydsl_or_the_facade():
    if torch.version.hip is None:
        pytest.skip("ROCm package import is the lazy path under test")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.abc
                import sys


                class NoFlydsl(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "flydsl" or fullname.startswith("flydsl."):
                            raise AssertionError(f"unexpected FlyDSL import: {fullname}")
                        return None


                sys.meta_path.insert(0, NoFlydsl())
                from quack.flydsl import rmsnorm_preflight

                assert rmsnorm_preflight
                assert "quack.rmsnorm_flydsl" not in sys.modules
                assert "quack.flydsl.rmsnorm_launch" not in sys.modules
                assert not any(
                    name == "flydsl" or name.startswith("flydsl.") for name in sys.modules
                )
                """
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _backend_modules():
    if torch.version.hip is None:
        pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build")
    pytest.importorskip("flydsl.compiler")
    facade = importlib.import_module("quack.rmsnorm_flydsl")
    launch = importlib.import_module("quack.flydsl.rmsnorm_launch")
    preflight = importlib.import_module("quack.flydsl.rmsnorm_preflight")
    return facade, preflight, launch


def test_facade_reexports_backend_functions_and_mutable_caches_by_identity():
    facade, preflight, launch = _backend_modules()

    assert facade._validate_inputs is preflight._validate_inputs
    assert facade._validated_inputs is preflight._validated_inputs
    assert facade._packed_rows is preflight._packed_rows
    assert facade._validate_arch is preflight._validate_arch
    assert facade._launch_rmsnorm_fwd is launch._launch_rmsnorm_fwd
    assert facade._launch_resolved_fwd_entry is launch._launch_resolved_fwd_entry
    assert facade._launch_rmsnorm_bwd is launch._launch_rmsnorm_bwd
    assert launch._VALIDATED_INPUT_LAST is preflight._VALIDATED_INPUT_LAST

    for name in ("_DEVICE_ARCH_CACHE", "_AUTOTUNE_ARCH_CACHE", "_VALIDATED_INPUT_LAST"):
        assert getattr(facade, name) is getattr(preflight, name)
    for name in (
        "_FWD_CACHE",
        "_BWD_CACHE",
        "_FWD_AUTOTUNED_FAST_CACHE",
        "_BWD_AUTOTUNED_FAST_CACHE",
        "_FWD_AUTOTUNED_LAST",
        "_FWD_PUBLIC_GENERIC_LAST",
        "_FWD_PUBLIC_RESOLVED_LAST",
        "_BWD_CU_COUNT_CACHE",
    ):
        assert getattr(facade, name) is getattr(launch, name)


def test_build_cache_monkeypatches_the_launch_lookup_site(monkeypatch):
    _facade, _preflight, launch = _backend_modules()
    cache = {}
    device = object()
    built = object()
    calls = []

    monkeypatch.setattr(launch, "_validate_arch", lambda actual: "gfx-test")

    def build(arch):
        calls.append(arch)
        return built

    assert launch._build_cached(cache, ("key",), device, build) is built
    assert launch._build_cached(cache, ("key",), device, build) is built
    assert calls == ["gfx-test"]

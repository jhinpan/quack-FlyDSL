# Copyright (c) 2026, Tri Dao.

"""Host-side tests for shared FlyDSL RMSNorm helpers."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("flydsl.compiler")

from quack.flydsl import rmsnorm_common


class _EqualExecutable:
    """Distinct launchers that deliberately collide under equality and hashing."""

    def __eq__(self, other):
        return isinstance(other, _EqualExecutable)

    def __hash__(self):
        return 0


class _IndexedTensor:
    def __init__(self):
        self.values = {3: "loaded"}
        self.events = []

    def __getitem__(self, index):
        self.events.append(("load", index))
        return self.values[index]

    def __setitem__(self, index, value):
        self.events.append(("store", index, value))
        self.values[index] = value


@pytest.fixture(autouse=True)
def _empty_compiled_callable_cache():
    with rmsnorm_common._COMPILED_CALLABLES_LOCK:
        rmsnorm_common._COMPILED_CALLABLES.clear()
    yield
    with rmsnorm_common._COMPILED_CALLABLES_LOCK:
        rmsnorm_common._COMPILED_CALLABLES.clear()


def test_run_compiled_caches_by_identity_and_preserves_positional_abi(monkeypatch):
    first = _EqualExecutable()
    second = _EqualExecutable()
    events = []

    def compile_and_launch(executable, *args):
        owner = id(executable)
        events.append(("compile", owner, args))

        def compiled(*later_args):
            events.append(("call", owner, later_args))

        return compiled

    monkeypatch.setattr(rmsnorm_common.flyc, "compile", compile_and_launch)

    rmsnorm_common.run_compiled(first, "first-launch", 1)
    rmsnorm_common.run_compiled(second, "second-launch", 2)
    rmsnorm_common.run_compiled(first, "cached-launch", 3)

    assert events == [
        ("compile", id(first), ("first-launch", 1)),
        ("compile", id(second), ("second-launch", 2)),
        ("call", id(first), ("cached-launch", 3)),
    ]


def test_run_compiled_serializes_concurrent_first_call(monkeypatch):
    executable = _EqualExecutable()
    compile_started = threading.Event()
    release_compile = threading.Event()
    compile_args = []
    cached_calls = []

    def compile_and_launch(actual, *args):
        assert actual is executable
        compile_args.append(args)
        compile_started.set()
        assert release_compile.wait(timeout=5)

        def compiled(*later_args):
            cached_calls.append(later_args)

        return compiled

    monkeypatch.setattr(rmsnorm_common.flyc, "compile", compile_and_launch)

    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(rmsnorm_common.run_compiled, executable, "first")
        assert compile_started.wait(timeout=5)
        followers = [
            pool.submit(rmsnorm_common.run_compiled, executable, f"follower-{index}")
            for index in range(3)
        ]
        release_compile.set()
        first.result(timeout=5)
        for follower in followers:
            follower.result(timeout=5)

    assert compile_args == [("first",)]
    assert sorted(cached_calls) == [(f"follower-{index}",) for index in range(3)]


def test_buffer_copy_atom_uses_generic_width_api(monkeypatch):
    copy_op = object()
    copy_atom = object()
    calls = []

    def make_buffer_copy(bit_size, cache_modifier=0):
        calls.append(("buffer", bit_size, cache_modifier))
        return copy_op

    def make_copy_atom(actual_copy_op, elem_bits):
        calls.append(("atom", actual_copy_op, elem_bits))
        return copy_atom

    monkeypatch.setattr(rmsnorm_common.fx.rocdl, "BufferCopy", make_buffer_copy)
    monkeypatch.setattr(rmsnorm_common.fx, "make_copy_atom", make_copy_atom)

    assert rmsnorm_common.buffer_copy_atom(64, 16, cache_modifier=2) is copy_atom
    assert calls == [("buffer", 64, 2), ("atom", copy_op, 16)]


def test_buffer_copy_atom_rejects_unsupported_width():
    with pytest.raises(ValueError, match="24-bit"):
        rmsnorm_common.buffer_copy_atom(24, 16)


def test_scalar_helpers_use_direct_tensor_indexing():
    tensor = _IndexedTensor()

    assert rmsnorm_common.load_scalar(tensor, 3) == "loaded"
    rmsnorm_common.store_scalar(tensor, 4, "stored")

    assert tensor.values[4] == "stored"
    assert tensor.events == [("load", 3), ("store", 4, "stored")]

# Copyright (c) 2026, Tri Dao.

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_rmsnorm_flydsl.py"
SPEC = importlib.util.spec_from_file_location("benchmark_rmsnorm_flydsl_contract", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_default_matrix_covers_compact_shapes_and_distinct_dtype_weight_modes():
    cells = benchmark.build_matrix()

    assert {cell.shape for cell in cells} == set(benchmark.COMPACT_SHAPES)
    assert {(cell.activation_dtype, cell.weight_mode) for cell in cells} == {
        ("float16", "same"),
        ("float16", "float32"),
        ("bfloat16", "same"),
        ("bfloat16", "float32"),
        ("float32", "same"),
    }
    assert {cell.operation for cell in cells} == {"fwd", "bwd"}
    assert len(cells) == len(benchmark.COMPACT_SHAPES) * 5 * 2


def test_logical_byte_accounting_uses_the_public_fwd_bwd_contract():
    # M=2, N=4, two-byte activations, four-byte weights.
    # fwd: read x + weight and write y.
    assert benchmark.logical_bytes("fwd", 2, 4, 2, 4) == 48
    # bwd: read x/dy/weight/rstd, write dx/dweight.
    assert benchmark.logical_bytes("bwd", 2, 4, 2, 4) == 88


def test_result_contract_and_artifact_writers(tmp_path):
    row = {
        field: index if field not in {"provider", "operation"} else field
        for index, field in enumerate(benchmark.RESULT_FIELDS)
    }
    environment = {"schema_version": 2, "correctness_gate": "required"}

    csv_path, environment_path = benchmark.write_artifacts(tmp_path, [row], environment)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == benchmark.RESULT_FIELDS
        assert len(list(reader)) == 1
    assert json.loads(environment_path.read_text(encoding="utf-8")) == environment


def test_module_import_is_provider_lazy():
    script = (
        "import importlib.util, sys;"
        f"p={str(BENCHMARK_PATH)!r};"
        "s=importlib.util.spec_from_file_location('bench_lazy', p);"
        "m=importlib.util.module_from_spec(s);"
        "sys.modules[s.name]=m;"
        "s.loader.exec_module(m);"
        "assert 'torch' not in sys.modules;"
        "assert 'quack.rmsnorm_flydsl' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


# --- last-level cache resolution -------------------------------------------
#
# These cover the paths that decide whether the benchmark measures cache-cold
# or cache-warm. Before this block there were zero: the parser, the identity
# matching, the ambiguity refusal, the fail-closed policy and the gate
# arithmetic were all exercised only by running the benchmark on one host,
# where every one of them happens to take its happy path.


def _write_node(root, index, *, gfx=90000, domain=0, location=29952, unique_id=None, caches=()):
    node = root / str(index)
    (node / "caches").mkdir(parents=True)
    lines = [f"gfx_target_version {gfx}", f"domain {domain}", f"location_id {location}"]
    if unique_id is not None:
        lines.append(f"unique_id {unique_id}")
    (node / "properties").write_text("\n".join(lines) + "\n")
    for cache_index, (level, size_kb) in enumerate(caches):
        cache = node / "caches" / str(cache_index)
        cache.mkdir()
        (cache / "properties").write_text(f"level {level}\nsize {size_kb}\n")
    return node


def _properties(*, bus=0x75, device=0, uuid_text=b"a60c2956cd9dd4c5", l2=4 * 1024**2):
    uuid = types.SimpleNamespace(bytes=uuid_text) if uuid_text is not None else None
    return types.SimpleNamespace(
        L2_cache_size=l2, pci_domain_id=0, pci_bus_id=bus, pci_device_id=device, uuid=uuid
    )


def _hip_torch():
    return types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2"))


def test_torch_uuid_is_ascii_hex_not_a_uuid():
    # The 16 bytes are the text "a60c2956cd9dd4c5", which is KFD's unique_id
    # for the same card. Decoding them as a binary UUID gives a different
    # number and matches no node.
    assert benchmark._torch_unique_id(_properties()) == 0xA60C2956CD9DD4C5
    assert benchmark._torch_unique_id(_properties(uuid_text=b"not hex at all!!")) is None
    assert benchmark._torch_unique_id(_properties(uuid_text=None)) is None


def test_gate_margin_and_motivating_cell_is_not_evicted_by_the_gate():
    # Guards the claim that used to be wrong in the source comment. The cell
    # this whole investigation started from -- 32768x1024 fwd -- ends up with a
    # 512.008 MiB rotation, which is ABOVE 2 x 256 MiB, so the gate leaves the
    # evictor OFF. It is rescued by the larger rotation, not by the gate.
    llc = 256 * 1024**2
    per_set = benchmark.logical_bytes("fwd", 32768, 1024, 2, 2)
    buffers = benchmark._rotation_count(
        per_set, llc * 3, free_bytes=10**12, min_buffers=2, max_buffers=4
    )
    working_set = buffers * per_set
    assert buffers == 4
    assert working_set == 512 * 1024**2 + 8 * 1024
    # Calls the real predicate, not a copy of it.
    assert benchmark.evictor_is_needed(working_set, llc) is False
    # ...and it clears the threshold by only 8 KiB, so the margin is not what
    # saves this cell either way.
    assert working_set - benchmark.EVICTOR_LLC_MARGIN * llc == 8 * 1024


def test_gate_flips_exactly_the_cells_the_docs_claim():
    # 90 cells: the target change moves 53 rotation counts, the gate flips 37
    # false->true, and those are different sets. Conflating them is the error
    # @Reviewer caught in the commit message and the source comment.
    llc = 256 * 1024**2
    rotation_changed = gate_flipped = 0
    for cell in benchmark.build_matrix():
        act = benchmark._ITEMSIZES[cell.activation_dtype]
        weight = 4 if cell.weight_mode == "float32" else act
        per_set = benchmark.logical_bytes(cell.operation, cell.m, cell.n, act, weight)
        kwargs = {"free_bytes": 10**12, "min_buffers": 2, "max_buffers": 4}
        old_n = benchmark._rotation_count(per_set, 4 * 1024**2 * 3, **kwargs)
        new_n = benchmark._rotation_count(per_set, llc * 3, **kwargs)
        old_on = old_n * per_set < 4 * 1024**2 * 3  # the pre-fix predicate
        new_on = benchmark.evictor_is_needed(new_n * per_set, llc)
        rotation_changed += old_n != new_n
        gate_flipped += (not old_on) and new_on
    assert rotation_changed == 53
    assert gate_flipped == 37


def test_resolve_llc_fails_closed_on_gfx950_rather_than_measuring_cache_warm():
    # The whole point of the helper. Returning torch's 4 MiB L2 here would let
    # the run complete with every row silently measured against a resident
    # 256 MiB MALL -- the original defect, reintroduced quietly.
    torch = _hip_torch()
    args = argparse.Namespace(llc_bytes=None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            benchmark,
            "_last_level_cache_bytes",
            lambda *_: (4 * 1024**2, {"source": "torch_l2_fallback", "reason": "no_matching_node"}),
        )
        patch.setattr(benchmark, "_device_arch", lambda _: "gfx950")
        with pytest.raises(RuntimeError, match="known wrong"):
            benchmark._resolve_llc(torch, _properties(), args)

        # Another architecture has no known-wrong fallback, so it is allowed.
        patch.setattr(benchmark, "_device_arch", lambda _: "gfx942")
        value, provenance = benchmark._resolve_llc(torch, _properties(), args)
        assert value == 4 * 1024**2
        assert provenance["reason"] == "no_matching_node"


def test_explicit_override_bypasses_topology_and_is_recorded():
    args = argparse.Namespace(llc_bytes=123456)
    value, provenance = benchmark._resolve_llc(_hip_torch(), _properties(), args)
    assert value == 123456
    assert provenance == {"source": "explicit_override", "flag": "--llc-bytes"}


def test_matches_by_unique_id_even_when_the_pci_address_is_shared(tmp_path):
    # Reachable under CPX: eight logical devices behind one PCI address. Before
    # unique_id matching this returned the fallback for every one of them.
    root = tmp_path / "nodes"
    root.mkdir()
    _write_node(root, 0, unique_id=0xAAAA, caches=((1, 32), (2, 4096)))
    _write_node(root, 1, unique_id=0xA60C2956CD9DD4C5, caches=((2, 4096), (3, 262144)))
    _write_node(root, 2, unique_id=0xBBBB, caches=((2, 4096),))
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "unique_id"


def test_ambiguous_pci_address_without_unique_id_refuses_rather_than_guessing(tmp_path):
    # Same shared address, but no unique_id to separate them. Returning the
    # first match would make the answer depend on directory iteration order.
    root = tmp_path / "nodes"
    root.mkdir()
    _write_node(root, 0, caches=((2, 4096), (3, 262144)))
    _write_node(root, 1, caches=((2, 4096),))
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 4 * 1024**2
    assert provenance["reason"] == "ambiguous_pci_address"


def test_one_node_per_address_still_resolves_without_unique_id(tmp_path):
    # The over-refusal negative case: a guard written as "more than one node
    # anywhere -> refuse" would break every normal eight-card host.
    root = tmp_path / "nodes"
    root.mkdir()
    _write_node(root, 0, location=29952, caches=((2, 4096), (3, 262144)))
    _write_node(root, 1, location=30208, caches=((2, 4096), (3, 262144)))
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "pci_domain_bus_device"


def test_a_malformed_unrelated_node_is_skipped_not_fatal(tmp_path):
    # Regression: the parser used to let ValueError escape _run() and kill the
    # benchmark, and the offending node need not even be a candidate match.
    root = tmp_path / "nodes"
    root.mkdir()
    bad = root / "0"
    (bad / "caches").mkdir(parents=True)
    (bad / "properties").write_text("gfx_target_version not-a-number\ndomain 0\n")
    _write_node(root, 1, unique_id=0xA60C2956CD9DD4C5, caches=((2, 4096), (3, 262144)))
    value, _ = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2


def test_missing_topology_reports_a_reason_rather_than_a_bare_number(tmp_path):
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(), str(tmp_path / "absent")
    )
    assert value == 4 * 1024**2
    assert provenance == {"source": "torch_l2_fallback", "reason": "no_kfd_topology"}


def test_non_hip_build_uses_the_torch_value_without_touching_sysfs():
    torch = types.SimpleNamespace(version=types.SimpleNamespace(hip=None))
    value, provenance = benchmark._last_level_cache_bytes(torch, _properties(), "/nonexistent")
    assert value == 4 * 1024**2
    assert provenance["reason"] == "not_a_hip_build"

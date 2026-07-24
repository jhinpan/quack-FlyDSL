# Copyright (c) 2026, Tri Dao.

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


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
    environment = {"schema_version": 1, "correctness_gate": "required"}

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

# Copyright (c) 2026, Tri Dao.

"""Focused orchestration tests for the RMSNorm FlyDSL reproducer."""

import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "reproduce_rmsnorm_flydsl.py"


def _load_reproducer():
    spec = importlib.util.spec_from_file_location("reproduce_rmsnorm_flydsl_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproducer = _load_reproducer()


class _MiniBenchmark:
    MN_PAIRS = ((2, 8),)

    def __init__(self):
        self.events = []
        self.sweeps = 0
        self.winners = {}

    def _tune_once_then_disable(self, call):
        previous = os.environ.get("FLYDSL_AUTOTUNE")
        os.environ["FLYDSL_AUTOTUNE"] = "1"
        output = call()
        os.environ["FLYDSL_AUTOTUNE"] = "0"
        return output, previous

    @staticmethod
    def _restore_autotune_env(previous):
        if previous is None:
            os.environ.pop("FLYDSL_AUTOTUNE", None)
        else:
            os.environ["FLYDSL_AUTOTUNE"] = previous

    def run_profiled_shapes(self, shapes, *, backward, providers, **_kwargs):
        assert shapes == self.MN_PAIRS
        operation = "bwd" if backward else "fwd"
        cache_dir = Path(os.environ["FLYDSL_AUTOTUNE_CACHE_DIR"])
        config_dir = Path(os.environ["FLYDSL_AUTOTUNE_CONFIG_DIR"])
        forced_tuning = os.environ["FLYDSL_AUTOTUNE"]
        fresh_cache = not any(cache_dir.iterdir()) and not any(config_dir.iterdir())
        assert forced_tuning == "1"

        self.sweeps += 1
        winner = object()
        self.winners[operation] = winner
        (cache_dir / f"{operation}.compiled").write_text("compiled\n")
        (config_dir / f"{operation}.winner").write_text("winner\n")
        self.events.append(
            ("profile", operation, forced_tuning, tuple(providers), id(winner), fresh_cache)
        )
        return f"profile-{operation}"

    def run_controlled_shapes(self, shapes, *, backward, providers, **_kwargs):
        assert shapes == self.MN_PAIRS
        operation = "bwd" if backward else "fwd"

        def tuned_call():
            if os.environ["FLYDSL_AUTOTUNE"] == "1":
                self.sweeps += 1
                self.winners[operation] = object()
            return self.winners[operation]

        winner, previous = self._tune_once_then_disable(tuned_call)
        forced_tuning = os.environ["FLYDSL_AUTOTUNE"]
        self._restore_autotune_env(previous)
        self.events.append(
            ("controlled", operation, forced_tuning, tuple(providers), id(winner), False)
        )
        return f"controlled-{operation}"


def _summary(geomean=1.0, strict=1):
    result = {}
    for operation in ("fwd", "bwd"):
        result[operation] = {}
        for provider in ("flydsl", "flydsl_tuned"):
            result[operation][provider] = {}
            for scope in ("device", "public"):
                result[operation][provider][f"{scope}_geomean_speedup_vs_torch"] = geomean
                result[operation][provider][f"{scope}_strict_gate_passes"] = strict
    return result


def test_outcome_policy_keeps_the_strict_gate_and_adds_ties():
    assert reproducer._classify_outcome(1.03, True) == "WIN"
    assert reproducer._classify_outcome(1.03, False) == "TIE"
    assert reproducer._classify_outcome(0.99, True) == "TIE"
    assert reproducer._classify_outcome(0.97, True) == "LOSS"


def test_summary_comparison_records_drift_and_strict_stability(tmp_path):
    baseline = _summary(geomean=1.0, strict=3)
    current = _summary(geomean=1.01, strict=3)
    baseline_path = tmp_path / "summary.json"
    baseline_path.write_text(json.dumps(baseline))

    comparison = reproducer._compare_summary(current, baseline_path)

    assert comparison["strict_counts_unchanged"] is True
    assert comparison["geomean_drift_percent_range"] == pytest.approx([1.0, 1.0])


def test_generalization_matrix_covers_distinct_off_ladder_row_regimes():
    source = (ROOT / "benchmarks" / "benchmark_rmsnorm_flydsl.py").read_text()
    module = ast.parse(source)
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "GENERALIZATION_PAIRS"
            for target in node.targets
        )
    )
    pairs = ast.literal_eval(assignment.value)

    assert len(pairs) >= 5
    assert len({m for m, _n in pairs}) == len(pairs)
    assert all(m != 32768 for m, _n in pairs)


def test_profile_then_controlled_reuses_winners_without_retuning_and_restores_env(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "fresh-output"
    cache_dir = output_dir / "cache"
    config_dir = output_dir / "config"
    cache_dir.mkdir(parents=True)
    config_dir.mkdir()

    monkeypatch.setenv("TORCH_COMPILE_DYNAMIC", "caller-dynamic")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "caller-autotune")
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CACHE_DIR", raising=False)
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CONFIG_DIR", raising=False)

    args = SimpleNamespace(
        profile_repeats=1,
        profile_rounds=1,
        controlled_rounds=1,
        rotation_buffers=1,
        settle_seconds=0.0,
        probe_mib=1,
        probe_samples=1,
        forward_calls_per_sample=1,
        backward_calls_per_sample=1,
    )
    benchmark = _MiniBenchmark()
    original_tuning_hook = benchmark._tune_once_then_disable.__func__
    frames = None

    with (
        pytest.raises(RuntimeError, match="prove failure restoration"),
        reproducer._benchmark_environment(cache_dir, config_dir),
    ):
        frames = reproducer._run_benchmark_phases(benchmark, args, output_dir)
        assert os.environ["TORCH_COMPILE_DYNAMIC"] == "caller-dynamic"
        assert os.environ["FLYDSL_AUTOTUNE"] == "0"
        raise RuntimeError("prove failure restoration")

    assert os.environ["TORCH_COMPILE_DYNAMIC"] == "caller-dynamic"
    assert os.environ["FLYDSL_AUTOTUNE"] == "caller-autotune"
    assert "FLYDSL_AUTOTUNE_CACHE_DIR" not in os.environ
    assert "FLYDSL_AUTOTUNE_CONFIG_DIR" not in os.environ
    assert benchmark._tune_once_then_disable.__func__ is original_tuning_hook

    assert frames == {
        "profile": {"fwd": "profile-fwd", "bwd": "profile-bwd"},
        "controlled": {"fwd": "controlled-fwd", "bwd": "controlled-bwd"},
    }
    assert benchmark.sweeps == 2
    assert benchmark.events[0][-1] is True
    assert all("flydsl_tuned" in event[3] for event in benchmark.events)

    for operation in ("fwd", "bwd"):
        profile = next(event for event in benchmark.events if event[:2] == ("profile", operation))
        controlled = next(
            event for event in benchmark.events if event[:2] == ("controlled", operation)
        )
        assert profile[2] == "1"
        assert controlled[2] == "0"
        assert controlled[4] == profile[4]

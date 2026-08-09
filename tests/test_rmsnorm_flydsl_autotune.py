# Copyright (c) 2026, Tri Dao.

"""Focused contracts for the standalone RMSNorm forward autotuner."""

import inspect

import pytest
import torch

if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl.compiler")

import quack.flydsl.rmsnorm_autotune as autotune
from quack.flydsl import autotune_harness
from quack.flydsl.rmsnorm_config import (
    MAX_TUNED_NUM_THREADS,
    REGISTER_CACHE_ELEMS,
    RmsNormRowConfig,
    batch_short_rows,
)


def _direct_call(*, rows=4, n=512, has_bias=False):
    x = torch.randn((rows, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device=x.device, dtype=torch.float32)
    absent = torch.empty(0, device=x.device, dtype=torch.bfloat16)
    bias = torch.randn(n, device=x.device, dtype=torch.float32) if has_bias else absent
    output = torch.empty_like(x)
    rstd = torch.empty(0, device=x.device, dtype=torch.float32)
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0]
    args = (x, weight, bias, absent, output, absent, rstd, rows, 1e-6, 0.0)
    kwargs = {
        "n": n,
        "input_dtype_str": "bf16",
        "output_dtype_str": "bf16",
        "weight_dtype_str": "f32",
        "bias_dtype_str": "f32" if has_bias else "bf16",
        "residual_dtype_str": "bf16",
        "residual_out_dtype_str": "bf16",
        "has_weight": True,
        "has_bias": has_bias,
        "has_residual": False,
        "store_residual": False,
        "store_rstd": False,
        "per_head": False,
        "num_heads": 1,
        "arch": arch,
        "schema_version": autotune.RMSNORM_AUTOTUNE_SCHEMA_VERSION,
        "stream": torch.cuda.current_stream(x.device).cuda_stream,
    }
    return args, kwargs


def test_forward_autotuner_specializes_shared_harness_without_reverse_import():
    assert issubclass(autotune.RmsNormAutotuner, autotune_harness.FlydslL2Autotuner)
    assert "_compiled_callable" in autotune_harness.FlydslL2Autotuner.__dict__
    assert "_contextual_decision_key" in autotune_harness.FlydslL2Autotuner.__dict__
    assert "_compiled_callable" not in autotune.RmsNormAutotuner.__dict__
    assert "_contextual_decision_key" not in autotune.RmsNormAutotuner.__dict__
    assert "_bench_one" in autotune.RmsNormAutotuner.__dict__
    assert "rmsnorm_autotune" not in inspect.getsource(autotune_harness)


def test_moved_private_helpers_remain_compatible_reexports():
    for name in (
        "FLYDSL_BUILD_LOCK",
        "_CacheAuthority",
        "_L2RotationPlan",
        "_clone_tensor_arguments",
        "_typed_identity",
        "_with_runtime_stream",
    ):
        assert getattr(autotune, name) is getattr(autotune_harness, name)
    assert autotune.l2_cold_bench is autotune_harness.l2_cold_bench


@pytest.mark.parametrize(
    ("n", "dtype_name"),
    [(128, "bf16"), (512, "bf16"), (2048, "bf16"), (4096, "f32")],
)
def test_candidates_are_legal_unique_and_retain_the_default(n, dtype_name):
    dtype_width = 32 if dtype_name == "f32" else 16
    configs = autotune.rmsnorm_search_configs(n=n, input_dtype_str=dtype_name)
    default = autotune.rmsnorm_default_config(n=n, input_dtype_str=dtype_name)
    identities = [(config.kwargs["threads_per_row"], config.waves_per_eu) for config in configs]

    assert len(identities) == len(set(identities))
    assert (default.kwargs["threads_per_row"], None) in identities

    ceiling = 64 if batch_short_rows(n, dtype_width) else MAX_TUNED_NUM_THREADS
    for threads, _waves_per_eu in identities:
        assert 1 <= threads <= ceiling
        assert threads & (threads - 1) == 0
        row = RmsNormRowConfig.with_num_threads(
            n,
            dtype_width,
            threads,
            max_num_threads=ceiling,
        )
        assert row.num_vecs >= threads


def test_persistent_candidates_follow_geometry_for_off_ladder_rows():
    for rows in (4097, 8193):
        args, kwargs = _direct_call(rows=rows, n=1024)
        configs = autotune.rmsnorm_search_configs(*args, **kwargs)
        persistent = [config for config in configs if config.kwargs.get("packed_flat_rows")]

        assert persistent
        for config in persistent:
            row_groups = config.kwargs["row_groups_per_block"]
            assert row_groups > 1
            assert config.kwargs["output_cache_modifier"] == 3
            assert config.kwargs["persistent_single_pass"] is True
            assert config.kwargs["persistent_programs"] == (rows + row_groups - 1) // row_groups


def test_wide_candidate_uses_register_budget_not_exact_n():
    candidates = autotune._row_candidates(32768, 16)
    config = RmsNormRowConfig.with_num_threads(
        32768,
        16,
        1024,
        max_num_threads=MAX_TUNED_NUM_THREADS,
    )

    assert 1024 in candidates
    assert config.elems_per_thread <= REGISTER_CACHE_ELEMS
    wide = RmsNormRowConfig.from_register_budget(65536, 16)
    assert wide.num_threads == 1024
    assert wide.reload_from == "gmem"
    assert "n == 32768" not in inspect.getsource(autotune._row_candidates)


def test_cache_policy_candidates_are_explicit_and_shape_independent():
    args, kwargs = _direct_call(rows=4097, n=2048)
    configs = autotune.rmsnorm_search_configs(*args, **kwargs)
    policies = {
        (
            config.kwargs.get("input_cache_modifier", 0),
            config.kwargs.get("output_cache_modifier", 0),
        )
        for config in configs
    }

    assert autotune.RMSNORM_AUTOTUNE_SCHEMA_VERSION == 6
    assert {(0, 0), (2, 2)} <= policies
    source = inspect.getsource(autotune.rmsnorm_search_configs)
    assert "_PERSISTENT_FWD_CONFIGS" not in source
    assert "32768" not in source


def test_decision_key_partitions_shapes_and_features_but_not_runtime_eps():
    tuner = autotune._rmsnorm_fwd_tuner
    args, kwargs = _direct_call(rows=4, n=512)
    base = tuner._contextual_decision_key(args, kwargs)

    row_args, row_kwargs = _direct_call(rows=5, n=512)
    n_args, n_kwargs = _direct_call(rows=4, n=1024)
    bias_args, bias_kwargs = _direct_call(rows=4, n=512, has_bias=True)
    eps_args = args[:8] + (0.5,) + args[9:]

    assert (
        len(
            {
                base,
                tuner._contextual_decision_key(row_args, row_kwargs),
                tuner._contextual_decision_key(n_args, n_kwargs),
                tuner._contextual_decision_key(bias_args, bias_kwargs),
            }
        )
        == 4
    )
    assert tuner._contextual_decision_key(eps_args, kwargs) == base
    assert tuner.key[:2] == ["m", "n"]
    assert {"has_bias", "has_residual", "store_residual", "store_rstd"} <= set(tuner.key)


def test_selected_config_launch_passes_the_correctness_gate(tmp_path, monkeypatch):
    torch.manual_seed(0)
    tuner = autotune._rmsnorm_fwd_tuner
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path))
    tuner._compiled_cache.clear()
    tuner._compiled_lookup.clear()
    tuner._device_jit_functions.clear()
    tuner._hot_cache.clear()

    args, kwargs = _direct_call(rows=5, n=512)
    config = autotune.rmsnorm_default_config(*args, **kwargs)
    expected = autotune._reference_samples(args, kwargs)
    compiled, positional, _compiled_now = tuner._compiled_callable(config, args, kwargs)
    stream = torch.cuda.current_stream(args[0].device)
    plan = autotune._L2RotationPlan(
        arg_sets=[args],
        kwarg_sets=[kwargs],
        cache_bytes=1,
        read_bytes_per_set=1,
        eviction_buffers=[],
        cache_source="test",
        bench_stream=stream,
        expected=expected,
        n_timed_calls=1,
    )

    autotune._candidate_correctness_gate(
        compiled,
        autotune._with_runtime_stream([positional], stream),
        plan,
    )
    torch.testing.assert_close(args[4], expected.output, rtol=2e-2, atol=2e-2)

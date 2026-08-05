# Copyright (c) 2026, Tri Dao.

"""Focused contracts for the standalone RMSNorm forward autotuner."""

import pytest
import torch

if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl")

import quack.flydsl.rmsnorm_autotune as autotune
from quack.flydsl.rmsnorm_config import (
    MAX_TUNED_NUM_THREADS,
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


def test_n1024_candidate_matches_the_measured_persistent_identity():
    args, kwargs = _direct_call(rows=1, n=1024)
    args = args[:7] + (32768,) + args[8:]
    configs = autotune.rmsnorm_search_configs(*args, **kwargs)
    num_cus = torch.cuda.get_device_properties(args[0].device).multi_processor_count
    expected = {
        "threads_per_row": 64,
        "row_groups_per_block": 2,
        "persistent_programs": min(16384, num_cus * 64),
        "output_cache_modifier": 3,
        "persistent_single_pass": True,
        "packed_flat_rows": True,
    }

    matching = [
        config for config in configs if config.kwargs == expected and config.waves_per_eu == 7
    ]
    assert len(matching) == 1


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
    for cache_name in (
        "_compiled_cache",
        "_compiled_lookup",
        "_device_jit_functions",
        "_hot_cache",
    ):
        getattr(tuner, cache_name).clear()

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

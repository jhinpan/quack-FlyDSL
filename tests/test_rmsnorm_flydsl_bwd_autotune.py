# Copyright (c) 2026, Tri Dao.

"""Focused contracts for the standalone RMSNorm backward autotuner."""

import pytest
import torch
from flydsl.autotune import Config

if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl.compiler")

import quack.flydsl.rmsnorm_bwd_autotune as autotune
from quack.flydsl.rmsnorm_bwd_kernel import (
    PARAMETER_REDUCE_THREADS,
    TWO_STAGE_MAX_NUM_THREADS,
    rmsnorm_bwd_parameter_reduce_cols,
    rmsnorm_bwd_two_stage_config,
)
from quack.flydsl.rmsnorm_common import dtype_to_elem_bits
from quack.flydsl.rmsnorm_config import WAVE_SIZE, RmsNormRowConfig, next_power_of_two

EPS = 1e-6


def _direct_call(*, m=64, n=512):
    source = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device=source.device, dtype=torch.float32)
    dy = torch.randn_like(source)
    rstd = torch.rsqrt(source.float().square().mean(dim=-1) + EPS)
    dx = torch.empty_like(source)
    dweight = torch.empty_like(weight)
    absent = torch.empty(0, device=source.device, dtype=source.dtype)
    dbias = torch.empty(1, device=source.device, dtype=weight.dtype)
    workspace = torch.empty((1, n), device=source.device, dtype=torch.float32)
    args = (
        source,
        weight,
        dy,
        absent,
        rstd,
        rstd,
        dx,
        absent,
        dweight,
        dbias,
        workspace,
        workspace.view(-1),
        m,
        0.0,
    )
    arch = torch.cuda.get_device_properties(source.device).gcnArchName.split(":", 1)[0]
    kwargs = {
        "n": n,
        "source_dtype_str": "bf16",
        "dy_dtype_str": "bf16",
        "dx_dtype_str": "bf16",
        "dresidual_dtype_str": "bf16",
        "dresidual_out_dtype_str": "bf16",
        "weight_dtype_str": "f32",
        "dbias_dtype_str": "f32",
        "has_weight": True,
        "has_bias": False,
        "compute_dweight": True,
        "compute_dbias": False,
        "compute_input_grad": True,
        "store_dx": True,
        "store_dresidual": False,
        "has_residual": False,
        "has_dresidual_out": False,
        "per_head": False,
        "num_heads": 1,
        "arch": arch,
        "schema_version": autotune.RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION,
        "stream": torch.cuda.current_stream(source.device).cuda_stream,
    }
    return args, kwargs


def _identity(config):
    return tuple(
        config.kwargs[name] for name in ("threads_per_row", "num_programs", "parameter_reduce_cols")
    )


@pytest.mark.parametrize("n", [256, 1024, 8192])
def test_candidates_have_legal_axes_and_retain_the_default(n):
    args, kwargs = _direct_call(m=2048, n=n)
    configs = autotune.rmsnorm_bwd_search_configs(*args, **kwargs)
    identities = [_identity(config) for config in configs]

    assert len(identities) == len(set(identities))
    assert _identity(autotune.rmsnorm_bwd_default_config(*args, **kwargs)) in identities
    assert len({programs for _, programs, _ in identities}) >= 3
    num_cus = torch.cuda.get_device_properties(args[0].device).multi_processor_count
    for _threads, programs, cols in identities:
        assert cols == rmsnorm_bwd_parameter_reduce_cols(
            n,
            programs,
            target_blocks=num_cus,
        )
    reduce_configs = autotune._reduce_stage_configs(
        autotune._call_values(args, kwargs),
        autotune.rmsnorm_bwd_default_config(*args, **kwargs),
    )
    assert len({_identity(config)[2] for config in reduce_configs}) >= 2

    dtype_width = dtype_to_elem_bits(kwargs["source_dtype_str"])
    for threads, programs, cols in identities:
        assert WAVE_SIZE <= threads <= TWO_STAGE_MAX_NUM_THREADS
        assert threads & (threads - 1) == 0
        row = RmsNormRowConfig.with_num_threads(
            n,
            dtype_width,
            threads,
            max_num_threads=TWO_STAGE_MAX_NUM_THREADS,
        )
        assert threads == WAVE_SIZE or row.num_vecs >= threads
        assert 1 <= programs <= next_power_of_two(args[12])
        assert 1 <= cols <= PARAMETER_REDUCE_THREADS
        assert cols & (cols - 1) == 0
        assert PARAMETER_REDUCE_THREADS % cols == 0


def test_default_geometry_is_deterministic_and_analytical():
    args, kwargs = _direct_call(m=2048, n=8192)
    first = autotune.rmsnorm_bwd_default_config(*args, **kwargs)
    second = autotune.rmsnorm_bwd_default_config(*args, **kwargs)
    row = rmsnorm_bwd_two_stage_config(kwargs["n"], kwargs["source_dtype_str"])
    num_cus = torch.cuda.get_device_properties(args[0].device).multi_processor_count
    expected_programs = min(
        next_power_of_two(args[12]),
        (3 * num_cus) // 2 * (TWO_STAGE_MAX_NUM_THREADS // row.num_threads),
    )
    expected = (
        row.num_threads,
        expected_programs,
        rmsnorm_bwd_parameter_reduce_cols(
            kwargs["n"],
            expected_programs,
            target_blocks=num_cus,
        ),
    )

    assert autotune.RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION == 4
    assert _identity(first) == _identity(second) == expected


def test_tie_band_prefers_the_analytical_incumbent():
    args, kwargs = _direct_call(m=2048, n=1024)
    tuner = autotune._rmsnorm_bwd_tuner
    incumbent = autotune.rmsnorm_bwd_default_config(*args, **kwargs)
    alternative = autotune._reduce_stage_configs(
        autotune._call_values(args, kwargs),
        incumbent,
    )[0]
    if _identity(alternative) == _identity(incumbent):
        alternative = autotune._reduce_stage_configs(
            autotune._call_values(args, kwargs),
            incumbent,
        )[-1]

    selected, _elapsed = tuner._select_stable(
        [(alternative, 1.0), (incumbent, 1.019)],
        args,
        kwargs,
        incumbent,
    )

    assert _identity(selected) == _identity(incumbent)


def test_staged_canonical_search_stays_within_candidate_budget():
    total = 0
    for n in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
        args, kwargs = _direct_call(m=1, n=n)
        args = (*args[:12], 32768, *args[13:])
        row_grid = autotune.rmsnorm_bwd_search_configs(*args, **kwargs)
        reduce_count = max(
            len(
                autotune._reduce_stage_configs(
                    autotune._call_values(args, kwargs),
                    config,
                )
            )
            for config in row_grid
        )
        total += len(row_grid) + reduce_count - 1

    assert total <= 160


def test_rotation_plan_is_reused_across_reduce_column_candidates(monkeypatch):
    args, kwargs = _direct_call(m=19, n=760)
    tuner = autotune._rmsnorm_bwd_tuner
    base = autotune.rmsnorm_bwd_default_config(*args, **kwargs)
    alternative = Config(
        threads_per_row=base.kwargs["threads_per_row"],
        num_programs=base.kwargs["num_programs"],
        parameter_reduce_cols=max(1, base.kwargs["parameter_reduce_cols"] // 2),
    )
    built = []
    expected = object()
    plan = object()

    def build(*build_args, **build_kwargs):
        built.append((build_args, build_kwargs))
        return plan

    monkeypatch.setattr(autotune, "_build_l2_rotation_plan", build)
    tuner._active_call.bwd_plans = {}
    try:
        first = tuner._rotation_plan(base, args, kwargs, args, expected)
        second = tuner._rotation_plan(alternative, args, kwargs, args, expected)
    finally:
        tuner._active_call.bwd_plans = None

    assert first is second is plan
    assert len(built) == 1


def test_selected_default_config_launch_matches_reference():
    torch.manual_seed(0)
    args, kwargs = _direct_call(m=19, n=760)
    config = autotune.rmsnorm_bwd_default_config(*args, **kwargs)
    compiled, positional, _ = autotune._rmsnorm_bwd_tuner._compiled_callable(
        config,
        args,
        kwargs,
    )

    compiled(*positional)
    torch.cuda.synchronize(args[0].device)

    source = args[0].float().detach().requires_grad_(True)
    weight = args[1].float().detach().requires_grad_(True)
    output = source * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + EPS) * weight
    dx_ref, dweight_ref = torch.autograd.grad(output, (source, weight), args[2].float())
    torch.testing.assert_close(
        args[6],
        dx_ref.to(args[6].dtype),
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(args[8], dweight_ref, rtol=5e-3, atol=5e-3)


def test_resolved_fast_entry_exposes_backward_constexpr_suffix(monkeypatch):
    tuner = autotune._rmsnorm_bwd_tuner
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    tuner._hot_cache.clear()
    args, kwargs = _direct_call(m=19, n=760)

    tuner(*args, **kwargs)
    resolved = tuner.resolved_fast_entry(args, kwargs)

    assert resolved is not None
    config, _compiled, constexpr_suffix = resolved
    positional = tuner._positional_arguments(config, args, kwargs)
    assert constexpr_suffix == positional[len(args) : -1]

# Copyright (c) 2026, Tri Dao.

"""Direct correctness coverage for the FlyDSL RMSNorm forward builder."""

import inspect

import pytest
import torch

if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl.compiler")

from quack.flydsl.rmsnorm_common import run_compiled
from quack.flydsl.rmsnorm_config import MAX_TUNED_NUM_THREADS, RmsNormRowConfig
from quack.flydsl.rmsnorm_kernel import build_rmsnorm_module

EPS = 1e-6


def _build(n: int, **overrides):
    options = {
        "weight_dtype_str": "f32",
        "bias_dtype_str": "f32",
        "residual_dtype_str": "bf16",
        "residual_out_dtype_str": "bf16",
        "has_weight": True,
        "has_bias": False,
        "has_residual": False,
        "store_residual": False,
        "store_rstd": False,
        "per_head": False,
        "num_heads": 1,
    }
    options.update(overrides)
    return build_rmsnorm_module(n, "bf16", "bf16", **options)


def _run(
    launcher,
    x,
    weight,
    *,
    bias=None,
    residual=None,
    output=None,
    residual_out=None,
    rstd=None,
    weight_offset=0.0,
):
    absent = torch.empty(0, device=x.device, dtype=x.dtype)
    output = torch.empty_like(x) if output is None else output
    residual_out = absent if residual_out is None else residual_out
    rstd = torch.empty(0, device=x.device, dtype=torch.float32) if rstd is None else rstd
    run_compiled(
        launcher,
        x,
        weight,
        absent if bias is None else bias,
        absent if residual is None else residual,
        output,
        residual_out,
        rstd,
        x.shape[0],
        EPS,
        weight_offset,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return output, residual_out, rstd


def _reference(x, weight, *, bias=None, residual=None, weight_offset=0.0):
    value = x.float()
    if residual is not None:
        value = value + residual.float()
    rstd = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + EPS)
    output = value * rstd * (weight.float() + weight_offset)
    if bias is not None:
        output = output + bias.float()
    return output.to(torch.bfloat16), value.to(torch.bfloat16), rstd.squeeze(-1)


def _assert_bf16_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_default_cache_policy_contains_no_exact_row_lengths():
    source = inspect.getsource(build_rmsnorm_module)

    assert "non_temporal_input" not in source
    assert "non_temporal_output" not in source
    assert "n in (" not in source


def test_bf16_input_fp32_weight_matches_fp32_reference():
    torch.manual_seed(0)
    x = torch.randn((3, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device=x.device, dtype=torch.float32)

    output, _, _ = _run(_build(512), x, weight)

    expected, _, _ = _reference(x, weight)
    _assert_bf16_close(output, expected)


def test_optional_residual_bias_and_aux_outputs_match_reference():
    torch.manual_seed(1)
    m, n = 7, 120
    x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(n, device=x.device, dtype=torch.float32)
    bias = torch.randn(n, device=x.device, dtype=torch.float32)
    residual_out = torch.empty_like(x)
    rstd = torch.empty(m, device=x.device, dtype=torch.float32)

    launcher = _build(
        n,
        has_bias=True,
        has_residual=True,
        store_residual=True,
        store_rstd=True,
    )
    output, residual_out, rstd = _run(
        launcher,
        x,
        weight,
        bias=bias,
        residual=residual,
        residual_out=residual_out,
        rstd=rstd,
        weight_offset=1.0,
    )

    expected, expected_residual, expected_rstd = _reference(
        x,
        weight,
        bias=bias,
        residual=residual,
        weight_offset=1.0,
    )
    _assert_bf16_close(output, expected)
    _assert_bf16_close(residual_out, expected_residual)
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-5)


def test_packed_persistent_n1024_handles_an_odd_row_tail():
    torch.manual_seed(2)
    m, n = 67, 1024
    x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device=x.device, dtype=torch.float32)
    sentinel = 13.0
    output_storage = torch.full(
        (m * n + 16,),
        sentinel,
        device=x.device,
        dtype=torch.bfloat16,
    )
    output = output_storage[: m * n].view(m, n)

    launcher = _build(
        n,
        row_config=RmsNormRowConfig.with_num_threads(
            n,
            16,
            64,
            max_num_threads=MAX_TUNED_NUM_THREADS,
        ),
        row_groups_per_block=2,
        input_cache_modifier=2,
        output_cache_modifier=2,
        persistent_single_pass=True,
        packed_flat_rows=True,
        persistent_rows=True,
        persistent_programs=(m + 1) // 2,
    )
    output, _, _ = _run(launcher, x, weight, output=output)

    expected, _, _ = _reference(x, weight)
    _assert_bf16_close(output, expected)
    torch.testing.assert_close(
        output_storage[m * n :],
        torch.full_like(output_storage[m * n :], sentinel),
        rtol=0,
        atol=0,
    )


def test_regular_fallback_preserves_padded_row_guards():
    torch.manual_seed(3)
    m, n, pad = 43, 256, 5
    input_storage = torch.randn((m, n + pad), device="cuda", dtype=torch.bfloat16)
    x = input_storage[:, :n]
    weight = torch.randn(n, device=x.device, dtype=torch.float32)
    sentinel = 17.0
    output_storage = torch.full_like(input_storage, sentinel)
    output = output_storage[:, :n]

    output, _, _ = _run(_build(n), x, weight, output=output)

    expected, _, _ = _reference(x, weight)
    _assert_bf16_close(output, expected)
    torch.testing.assert_close(
        output_storage[:, n:],
        torch.full_like(output_storage[:, n:], sentinel),
        rtol=0,
        atol=0,
    )

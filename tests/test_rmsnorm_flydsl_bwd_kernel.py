# Copyright (c) 2026, Tri Dao.

"""Direct correctness coverage for the FlyDSL RMSNorm backward builder."""

import pytest
import torch

if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl.compiler")

from quack.flydsl.rmsnorm_bwd_kernel import (
    build_rmsnorm_bwd_two_stage_module,
    rmsnorm_bwd_two_stage_config,
)
from quack.flydsl.rmsnorm_common import run_compiled

EPS = 1e-6


def _build(n: int, num_programs: int):
    return build_rmsnorm_bwd_two_stage_module(
        n,
        "bf16",
        "bf16",
        "bf16",
        "bf16",
        "bf16",
        num_programs,
        weight_dtype_str="f32",
        dbias_dtype_str="f32",
        has_weight=True,
        has_bias=False,
        compute_dweight=True,
        compute_dbias=False,
        compute_input_grad=True,
        store_dx=True,
        store_dresidual=False,
        has_residual=False,
        has_dresidual_out=False,
        per_head=False,
        num_heads=1,
    )


def _run(launcher, x, weight, dout, num_programs):
    m, n = x.shape
    absent_bf16 = torch.empty(0, device=x.device, dtype=torch.bfloat16)
    absent_fp32 = torch.empty(0, device=x.device, dtype=torch.float32)
    rstd = torch.rsqrt(x.float().square().mean(dim=-1) + EPS)
    correction = (
        torch.empty_like(rstd)
        if rmsnorm_bwd_two_stage_config(n, "bf16").reload_from == "gmem"
        else rstd
    )
    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)
    workspace = torch.empty(
        (num_programs, n),
        device=x.device,
        dtype=torch.float32,
    )
    run_compiled(
        launcher,
        x,
        weight,
        dout,
        absent_bf16,
        rstd,
        correction,
        dx,
        absent_bf16,
        dweight,
        absent_fp32,
        workspace,
        workspace.view(-1),
        m,
        0.0,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return dx, dweight


def _reference(x, weight, dout):
    x_ref = x.float().detach().requires_grad_(True)
    weight_ref = weight.float().detach().requires_grad_(True)
    rstd = torch.rsqrt(x_ref.square().mean(dim=-1, keepdim=True) + EPS)
    output = x_ref * rstd * weight_ref
    return torch.autograd.grad(output, (x_ref, weight_ref), dout.float())


def _assert_matches_reference(actual, expected):
    dx, dweight = actual
    dx_ref, dweight_ref = expected
    torch.testing.assert_close(
        dx,
        dx_ref.to(dx.dtype),
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(dweight, dweight_ref, rtol=5e-3, atol=5e-3)


def test_two_stage_backward_matches_reference_and_repeats_exactly():
    torch.manual_seed(0)
    m, n, num_programs = 19, 760, 4
    x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device=x.device, dtype=torch.float32)
    dout = torch.randn_like(x)
    launcher = _build(n, num_programs)

    first = _run(launcher, x, weight, dout, num_programs)
    second = _run(launcher, x, weight, dout, num_programs)

    _assert_matches_reference(first, _reference(x, weight, dout))
    for repeated, original in zip(second, first):
        torch.testing.assert_close(repeated, original, rtol=0.0, atol=0.0)


def test_wide_staged_geometry_matches_fp32_reference():
    torch.manual_seed(1)
    m, n, num_programs = 3, 32768, 2
    assert rmsnorm_bwd_two_stage_config(n, "bf16").reload_from == "gmem"
    x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device=x.device, dtype=torch.float32)
    dout = torch.randn_like(x)

    actual = _run(_build(n, num_programs), x, weight, dout, num_programs)

    _assert_matches_reference(actual, _reference(x, weight, dout))

# Copyright (c) 2026, Tri Dao.

import ast
import inspect
import itertools
import math
import threading
from pathlib import Path

import pytest
import torch

if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl")

import quack
import quack.flydsl.rmsnorm_autotune as rmsnorm_autotune_impl
import quack.flydsl.rmsnorm_bwd_autotune as rmsnorm_bwd_autotune_impl
import quack.rmsnorm_flydsl as rmsnorm_flydsl_impl
from quack.flydsl.rmsnorm_autotune import rmsnorm_search_configs
from quack.flydsl.rmsnorm_bwd_autotune import (
    rmsnorm_bwd_default_config,
    rmsnorm_bwd_search_configs,
)
from quack.flydsl.rmsnorm_bwd_kernel import (
    PARAMETER_REDUCE_THREADS,
    rmsnorm_bwd_parameter_reduce_cols,
)
from quack.flydsl.rmsnorm_config import (
    MAX_TUNED_NUM_THREADS,
    RmsNormRowConfig,
    next_power_of_two,
)
from quack.rmsnorm_flydsl import rmsnorm, rmsnorm_autotuned


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_f32 * rstd * weight.float()).to(x.dtype)


def _full_reference(
    x,
    weight=None,
    bias=None,
    residual=None,
    *,
    eps=1e-6,
    weight_offset=0.0,
    out_dtype=None,
    residual_dtype=None,
):
    value = x.float()
    if residual is not None:
        value = value + residual.float()
    normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        normalized = normalized * (weight.float() + weight_offset)
    if bias is not None:
        normalized = normalized + bias.float()
    output = normalized.to(x.dtype if out_dtype is None else out_dtype)
    residual_out = value.to(
        residual_dtype
        if residual_dtype is not None
        else (residual.dtype if residual is not None else x.dtype)
    )
    return output, residual_out


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _reference_with_grads(x, weight, dout, eps):
    x_ref = x.detach().float().requires_grad_(True)
    weight_ref = weight.detach().float().requires_grad_(True)
    out_ref = _reference(x_ref, weight_ref, eps)
    dx_ref, dweight_ref = torch.autograd.grad(
        out_ref,
        (x_ref, weight_ref),
        dout.float(),
    )
    return out_ref.to(x.dtype), dx_ref.to(x.dtype), dweight_ref.to(weight.dtype)


def _assert_grad_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def _assert_fused_residual_grad_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare gradients that were recomputed from a rounded residual sum.

    A fused residual forward normalizes from the fp32 sum but saves that sum
    rounded to ``residual_dtype``, and backward recomputes ``x_hat`` from the
    rounded copy. Quack's own kernels do the same. The slack here absorbs that
    one rounding, so it applies whatever the gradient's own dtype is -- and it
    is why the tests without a residual use the tighter _assert_grad_close
    instead of borrowing this tolerance.
    """
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    ("shape", "dtype", "weight_dtype", "eps"),
    [
        ((3, 120), torch.float16, torch.float16, 1e-6),
        ((2, 1024), torch.bfloat16, torch.float32, 1e-5),
        # The profiled inference specializations: a 32-lane DPP reduction and
        # the non-temporal store/load policies used by the larger rows.
        ((4, 256), torch.bfloat16, torch.float32, 1e-6),
        ((4, 512), torch.bfloat16, torch.float32, 1e-6),
        ((4, 4096), torch.float16, torch.float16, 1e-6),
        ((2, 4096), torch.bfloat16, torch.float32, 1e-5),
        ((4, 8192), torch.bfloat16, torch.float32, 1e-6),
        ((3, 3584), torch.float16, torch.float32, 1e-6),
        ((2, 3584), torch.bfloat16, torch.bfloat16, 1e-5),
        ((2, 4096), torch.float32, torch.float32, 1e-6),
        # Vectorized with a predicated final tile: 375 vectors over 256 threads.
        ((3, 3000), torch.bfloat16, torch.bfloat16, 1e-6),
        ((2, 3000), torch.float16, torch.float32, 1e-6),
        # Predicated inside a single tile: 125 vectors over a 128-thread block.
        ((4, 1000), torch.bfloat16, torch.float32, 1e-6),
        # 128-bit FP32 loads, exact and predicated.
        ((2, 2048), torch.float32, torch.float32, 1e-6),
        ((2, 1000), torch.float32, torch.float32, 1e-6),
        # Rows short enough to share a block, so a lane group covers the row and
        # the reduction is a shuffle rather than a trip through LDS. One whole
        # access per lane: 16 lanes of 8 BF16, and the same row with an FP32
        # weight, which takes two accesses to cover one activation vector.
        ((4, 128), torch.bfloat16, torch.bfloat16, 1e-6),
        ((4, 128), torch.bfloat16, torch.float32, 1e-5),
        # A lane group narrower than its vectors: 3 vectors rounded up to 4 lanes.
        ((3, 24), torch.bfloat16, torch.bfloat16, 1e-6),
        # One vector for the whole row, so the group is a single lane and the
        # reduction has nothing to shuffle.
        ((5, 8), torch.bfloat16, torch.bfloat16, 1e-6),
        # Predicated inside a lane group: 62 FP32 vectors over 64 lanes.
        ((4, 248), torch.float32, torch.float32, 1e-6),
        # Weight dtype is validated on its own, not paired against the
        # activation, so every combination the validator admits is legal and
        # has to stay correct -- including the ones no caller is expected to
        # use.
        ((2, 4096), torch.bfloat16, torch.float16, 1e-6),
        ((2, 4096), torch.float16, torch.bfloat16, 1e-6),
        ((2, 4096), torch.float32, torch.bfloat16, 1e-6),
        ((2, 4096), torch.float32, torch.float16, 1e-6),
        # Wide rows use bounded-state runtime loops and reload from gmem after
        # the reduction rather than retaining the whole row in VGPRs.
        ((4, 16384), torch.bfloat16, torch.float32, 1e-6),
        ((4, 57344), torch.bfloat16, torch.float32, 1e-6),
        ((2, 262136), torch.bfloat16, torch.float32, 1e-6),
        ((4, 262144), torch.bfloat16, torch.float32, 1e-6),
        ((4, 32768), torch.float32, torch.float32, 1e-6),
        ((2, 262144), torch.float16, torch.bfloat16, 1e-6),
        ((2, 262144), torch.float32, torch.float32, 1e-6),
    ],
)
def test_forward_matches_fp32_reference(shape, dtype, weight_dtype, eps):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=weight_dtype)

    actual = rmsnorm(x, weight, eps=eps)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    _assert_close(actual, _reference(x, weight, eps))


def test_forward_flattens_noncontiguous_leading_dimensions():
    torch.manual_seed(1)
    n = 3584
    x = torch.randn((2, n, 3), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.randn(n * 2, device="cuda", dtype=torch.float32)[::2]
    assert not x.is_contiguous()
    assert not weight.is_contiguous()

    actual = rmsnorm(x, weight)

    assert actual.shape == x.shape
    _assert_close(actual, _reference(x, weight, 1e-6))


def test_forward_empty_m_returns_empty_without_launching():
    x = torch.empty((2, 0, 128), device="cuda", dtype=torch.float16)
    weight = torch.ones(128, device="cuda", dtype=torch.float16)

    actual = rmsnorm(x, weight)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.numel() == 0


@pytest.mark.parametrize(
    ("x", "weight", "error", "message"),
    [
        ("not-a-tensor", torch.ones(8), TypeError, "x must be a torch.Tensor"),
        (torch.ones(8), 5.0, TypeError, "weight must be a torch.Tensor or None"),
        (torch.tensor(1.0), torch.ones(1), ValueError, "at least one dimension"),
        (torch.ones(2, 8), torch.ones(7), ValueError, "weight shape"),
        (torch.empty(2, 0), torch.empty(0), ValueError, "between 1 and 262144"),
        (
            torch.ones(1, 262152),
            torch.ones(262152),
            ValueError,
            "between 1 and 262144",
        ),
        (
            torch.ones(2, 8, dtype=torch.float64),
            torch.ones(8, dtype=torch.float64),
            TypeError,
            "x dtype",
        ),
        (torch.ones(2, 8), torch.ones(8), ValueError, "ROCm device"),
    ],
)
def test_public_contract_rejects_unsupported_inputs(x, weight, error, message):
    with pytest.raises(error, match=message):
        rmsnorm(x, weight)


@pytest.mark.parametrize(
    ("eps", "error"),
    [
        ("1e-6", TypeError),
        (True, TypeError),
        (0.0, ValueError),
        (-1e-6, ValueError),
        (math.inf, ValueError),
        (math.nan, ValueError),
    ],
)
def test_public_contract_validates_eps(eps, error):
    x = torch.ones(2, 8, device="cuda", dtype=torch.float16)
    weight = torch.ones(8, device="cuda", dtype=torch.float16)
    with pytest.raises(error, match="eps"):
        rmsnorm(x, weight, eps=eps)


@pytest.mark.parametrize("n", [1, 3, 7, 127, 257, 1020, 3001])
def test_public_contract_rejects_rows_that_are_not_whole_accesses(n):
    """A row this backend cannot cover with whole 128-bit accesses is refused.

    Serving it with a narrower access is implementable -- quack.rmsnorm does
    exactly that, and this backend used to -- but no hidden size in practice
    reaches it, so the path could only ever be measured on shapes nobody runs.
    The error names the fallback rather than leaving the caller guessing.
    """
    x = torch.ones(2, n, device="cuda", dtype=torch.float16)
    weight = torch.ones(n, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="multiple of 8"):
        rmsnorm(x, weight)


def test_public_contract_rejects_mixed_devices():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two ROCm devices")
    x = torch.ones(2, 8, device="cuda:0", dtype=torch.float16)
    weight = torch.ones(8, device="cuda:1", dtype=torch.float16)
    with pytest.raises(ValueError, match="same device"):
        rmsnorm(x, weight)


@pytest.mark.parametrize(
    ("shape", "dtype", "weight_dtype"),
    [
        # Fewer rows than the persistent grid would like, so most blocks
        # contribute a zeroed partial.
        ((17, 520), torch.float16, torch.float16),
        # Two 32-lane row groups share one wave; the odd row count leaves the
        # final group empty and exercises its zero partial.
        ((65, 256), torch.bfloat16, torch.float32),
        ((5, 760), torch.float32, torch.float32),
        ((512, 4096), torch.bfloat16, torch.float32),
        ((512, 3584), torch.float16, torch.float16),
        # 128-bit FP32 column I/O.
        ((512, 2048), torch.float32, torch.float32),
        # Column count that is not a whole number of blocks.
        ((512, 3000), torch.bfloat16, torch.float32),
        # The wide path separates the row correction from bounded column
        # tiles, while retaining the same deterministic parameter reduction.
        ((8, 32768), torch.bfloat16, torch.float32),
        ((4, 57344), torch.bfloat16, torch.float32),
        ((2, 262136), torch.bfloat16, torch.float32),
        ((4, 262144), torch.bfloat16, torch.float32),
        ((4, 32768), torch.float32, torch.float32),
        ((2, 262144), torch.float16, torch.bfloat16),
        ((2, 262144), torch.float32, torch.float32),
    ],
)
def test_backward_matches_fp32_reference(shape, dtype, weight_dtype):
    torch.manual_seed(2)
    x = (torch.randn(shape, device="cuda", dtype=dtype) * 0.5).requires_grad_()
    weight = (
        1.0 + torch.randn(shape[-1], device="cuda", dtype=weight_dtype) * 0.1
    ).requires_grad_()
    dout = torch.randn(shape, device="cuda", dtype=dtype) * 0.1
    eps = 1e-6

    actual = rmsnorm(x, weight, eps=eps)
    actual.backward(dout)
    out_ref, dx_ref, dweight_ref = _reference_with_grads(x, weight, dout, eps)

    _assert_close(actual, out_ref)
    _assert_grad_close(x.grad, dx_ref)
    _assert_grad_close(weight.grad, dweight_ref)


@pytest.mark.parametrize("per_head", [False, True])
def test_wide_residual_prenorm_backward_matches_fp32_reference(per_head):
    torch.manual_seed(22)
    n = 32768
    shape = (2, 2, n) if per_head else (2, n)
    parameter_shape = (2, n) if per_head else (n,)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(parameter_shape, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(parameter_shape, device="cuda", dtype=torch.float32, requires_grad=True)

    actual, residual_out = rmsnorm(
        x,
        weight,
        bias=bias,
        residual=residual,
        out_dtype=torch.float16,
        residual_dtype=torch.float32,
        prenorm=True,
    )
    dout = torch.randn_like(actual)
    dresidual_out = torch.randn_like(residual_out)
    actual_grads = torch.autograd.grad(
        (actual, residual_out),
        (x, residual, weight, bias),
        grad_outputs=(dout, dresidual_out),
    )

    x_ref = x.detach().float().requires_grad_(True)
    residual_ref = residual.detach().float().requires_grad_(True)
    weight_ref = weight.detach().float().requires_grad_(True)
    bias_ref = bias.detach().float().requires_grad_(True)
    expected, expected_residual = _full_reference(
        x_ref,
        weight_ref,
        bias_ref,
        residual_ref,
        out_dtype=torch.float16,
        residual_dtype=torch.float32,
    )
    expected_grads = torch.autograd.grad(
        (expected, expected_residual),
        (x_ref, residual_ref, weight_ref, bias_ref),
        grad_outputs=(dout.float(), dresidual_out),
    )

    _assert_close(actual, expected)
    _assert_close(residual_out, expected_residual)
    for got, want in zip(actual_grads, expected_grads):
        _assert_fused_residual_grad_close(got, want.to(got.dtype))


def test_wide_parameter_only_backward_skips_input_gradient_path():
    torch.manual_seed(23)
    n = 32768
    x = torch.randn((8, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(n, device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn_like(x)

    actual = rmsnorm(x, weight, bias=bias)
    dweight, dbias = torch.autograd.grad(actual, (weight, bias), dout)

    weight_ref = weight.detach().float().requires_grad_(True)
    bias_ref = bias.detach().float().requires_grad_(True)
    expected, _ = _full_reference(x.float(), weight_ref, bias_ref)
    dweight_ref, dbias_ref = torch.autograd.grad(
        expected,
        (weight_ref, bias_ref),
        dout.float(),
    )
    _assert_grad_close(dweight, dweight_ref)
    _assert_grad_close(dbias, dbias_ref)


@pytest.mark.parametrize(
    ("requires_x", "requires_weight"),
    [(True, False), (False, True), (True, True)],
)
def test_autograd_respects_requested_gradients(requires_x, requires_weight):
    torch.manual_seed(3)
    n = 760
    x = (
        torch.randn((2, n, 3), device="cuda", dtype=torch.float16)
        .transpose(1, 2)
        .detach()
        .requires_grad_(requires_x)
    )
    weight = (
        torch.randn(n * 2, device="cuda", dtype=torch.float32)[::2]
        .detach()
        .requires_grad_(requires_weight)
    )
    dout = torch.randn_like(x)
    assert not x.is_contiguous()
    assert not weight.is_contiguous()

    actual = rmsnorm(x, weight)
    actual.backward(dout)
    _, dx_ref, dweight_ref = _reference_with_grads(x, weight, dout, 1e-6)

    if requires_x:
        _assert_grad_close(x.grad, dx_ref)
    else:
        assert x.grad is None
    if requires_weight:
        _assert_grad_close(weight.grad, dweight_ref)
    else:
        assert weight.grad is None


def test_empty_m_autograd_returns_empty_and_zero_weight_grad():
    x = torch.empty(
        (2, 0, 128),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.ones(
        128,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    out = rmsnorm(x, weight)
    out.sum().backward()

    assert out.shape == x.shape
    assert x.grad is not None and x.grad.numel() == 0
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize(
    ("has_weight", "has_bias", "weight_offset", "out_dtype"),
    [
        (False, False, 0.0, None),
        (True, True, 0.0, None),
        (True, False, 1.0, None),
        (True, True, 0.0, torch.float32),
    ],
)
def test_optional_affine_features_match_reference(
    use_compile,
    has_weight,
    has_bias,
    weight_offset,
    out_dtype,
):
    torch.manual_seed(11)
    x = torch.randn(
        (4, 760),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = (
        torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)
        if has_weight
        else None
    )
    bias = (
        torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)
        if has_bias
        else None
    )
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True) if weight is not None else None
    bias_ref = bias.detach().clone().requires_grad_(True) if bias is not None else None
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm

    actual = function(
        x,
        weight,
        bias=bias,
        weight_offset=weight_offset,
        out_dtype=out_dtype,
    )
    expected, _ = _full_reference(
        x_ref,
        weight_ref,
        bias_ref,
        weight_offset=weight_offset,
        out_dtype=out_dtype,
    )
    dout = torch.randn_like(actual)
    actual.backward(dout)
    expected.backward(dout)

    _assert_close(actual, expected)
    _assert_grad_close(x.grad, x_ref.grad)
    if weight is not None:
        _assert_grad_close(weight.grad, weight_ref.grad)
    if bias is not None:
        _assert_grad_close(bias.grad, bias_ref.grad)


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("prenorm", [False, True])
def test_residual_and_prenorm_match_reference(use_compile, prenorm):
    torch.manual_seed(12)
    x = torch.randn(
        (4, 760),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    residual_ref = residual.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm

    result = function(
        x,
        weight,
        residual=residual,
        prenorm=prenorm,
    )
    actual, residual_out = result if prenorm else (result, None)
    expected, residual_out_ref = _full_reference(
        x_ref,
        weight_ref,
        residual=residual_ref,
    )
    dout = torch.randn_like(actual)
    if prenorm:
        dresidual_out = torch.randn_like(residual_out)
        torch.autograd.backward((actual, residual_out), (dout, dresidual_out))
        torch.autograd.backward(
            (expected, residual_out_ref),
            (dout, dresidual_out),
        )
        _assert_close(residual_out, residual_out_ref)
    else:
        actual.backward(dout)
        expected.backward(dout)

    _assert_close(actual, expected)
    _assert_fused_residual_grad_close(x.grad, x_ref.grad)
    _assert_fused_residual_grad_close(residual.grad, residual_ref.grad)
    _assert_fused_residual_grad_close(weight.grad, weight_ref.grad)


@pytest.mark.parametrize("use_compile", [False, True])
def test_residual_dtype_override_preserves_fp32_sum(use_compile):
    torch.manual_seed(13)
    x = torch.randn((3, 760), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(760, device="cuda", dtype=torch.float32)
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm

    actual, residual_out = function(
        x,
        weight,
        residual=residual,
        residual_dtype=torch.float32,
        prenorm=True,
    )
    expected, residual_out_ref = _full_reference(
        x,
        weight,
        residual=residual,
        residual_dtype=torch.float32,
    )

    _assert_close(actual, expected)
    torch.testing.assert_close(residual_out, residual_out_ref, rtol=0.0, atol=0.0)


def test_prenorm_without_residual_propagates_second_output_gradient():
    x = torch.randn(
        (4, 760),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)

    _, residual_out = rmsnorm(x, weight, prenorm=True)
    residual_out.sum().backward()

    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0.0, atol=0.0)


def test_mixed_input_and_weight_dtypes_use_generic_path():
    x = torch.randn(
        (4, 760),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.randn(
        760,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    actual = rmsnorm(x, weight)
    expected, _ = _full_reference(x_ref, weight_ref)
    dout = torch.randn_like(actual)
    actual.backward(dout)
    expected.backward(dout)

    _assert_close(actual, expected)
    _assert_grad_close(x.grad, x_ref.grad)
    _assert_grad_close(weight.grad, weight_ref.grad)


@pytest.mark.parametrize("n", [760, 32768])
def test_selective_gradients_are_respected(n):
    x = torch.randn((4, n), device="cuda", dtype=torch.float16)
    weight = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(
        n,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    residual = torch.randn_like(x, requires_grad=True)

    rmsnorm(x, weight, bias=bias, residual=residual).sum().backward()

    assert x.grad is None
    assert weight.grad is None
    assert bias.grad is not None
    assert residual.grad is not None


def test_empty_batch_preserves_autograd_contract():
    x = torch.empty(
        (0, 4, 64),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        (4, 64),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.randn(
        (4, 64),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    residual = torch.empty_like(x, requires_grad=True)

    out, residual_out = rmsnorm(
        x,
        weight,
        bias=bias,
        residual=residual,
        prenorm=True,
    )
    (out.sum() + residual_out.sum()).backward()

    assert out.shape == x.shape
    assert residual_out.shape == x.shape
    assert x.grad is not None and x.grad.numel() == 0
    assert residual.grad is not None and residual.grad.numel() == 0
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
    torch.testing.assert_close(bias.grad, torch.zeros_like(bias))


@pytest.mark.parametrize("use_compile", [False, True])
def test_per_head_affine_residual_matches_reference(use_compile):
    torch.manual_seed(14)
    x = torch.randn(
        (2, 3, 4, 64),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn((4, 64), device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn((4, 64), device="cuda", dtype=torch.float32, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    residual_ref = residual.detach().clone().requires_grad_(True)
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm

    actual = function(x, weight, bias=bias, residual=residual)
    expected, _ = _full_reference(
        x_ref,
        weight_ref,
        bias_ref,
        residual_ref,
    )
    dout = torch.randn_like(actual)
    actual.backward(dout)
    expected.backward(dout)

    _assert_close(actual, expected)
    _assert_fused_residual_grad_close(x.grad, x_ref.grad)
    _assert_fused_residual_grad_close(weight.grad, weight_ref.grad)
    _assert_fused_residual_grad_close(bias.grad, bias_ref.grad)
    _assert_fused_residual_grad_close(residual.grad, residual_ref.grad)


def test_paired_short_row_backward_handles_generic_features():
    """The full generic feature set remains valid when two rows share a wave."""
    torch.manual_seed(28)
    shape = (17, 2, 256)
    parameter_shape = (2, 256)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        parameter_shape,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.randn(
        parameter_shape,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    residual = torch.randn_like(x, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    residual_ref = residual.detach().clone().requires_grad_(True)

    actual, residual_out = rmsnorm(
        x,
        weight,
        bias=bias,
        residual=residual,
        prenorm=True,
        weight_offset=1.0,
    )
    expected, residual_out_ref = _full_reference(
        x_ref,
        weight_ref,
        bias_ref,
        residual_ref,
        weight_offset=1.0,
    )
    dout = torch.randn_like(actual)
    dresidual_out = torch.randn_like(residual_out)
    actual_grads = torch.autograd.grad(
        (actual, residual_out),
        (x, residual, weight, bias),
        grad_outputs=(dout, dresidual_out),
    )
    expected_grads = torch.autograd.grad(
        (expected, residual_out_ref),
        (x_ref, residual_ref, weight_ref, bias_ref),
        grad_outputs=(dout, dresidual_out),
    )

    _assert_close(actual, expected)
    _assert_close(residual_out, residual_out_ref)
    for got, want in zip(actual_grads, expected_grads):
        _assert_fused_residual_grad_close(got, want)


def test_paired_raw_io_honors_row_pitch():
    """The contiguous fast access still follows a padded row's real stride."""
    torch.manual_seed(29)
    m, n, pitch_pad = 65, 256, 3
    full = torch.randn((m, n + pitch_pad), device="cuda", dtype=torch.bfloat16)
    x = full[:, :n].detach().requires_grad_(True)
    weight = torch.randn(n, device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn_like(x)
    assert not x.is_contiguous() and x.stride() == (n + pitch_pad, 1)
    assert rmsnorm_flydsl_impl._packed_rows(x).data_ptr() == x.data_ptr()

    actual = rmsnorm(x, weight)
    actual.backward(dout)
    expected, dx_expected, dweight_expected = _reference_with_grads(
        x,
        weight,
        dout,
        1e-6,
    )

    _assert_close(actual, expected)
    _assert_grad_close(x.grad, dx_expected)
    _assert_grad_close(weight.grad, dweight_expected)


def test_n256_target_pins_measured_forward_and_backward_geometries():
    """The public path uses both measured winners without tuner dispatch."""
    _clear_caches()
    torch.manual_seed(30)
    x = torch.randn(
        (32768, 256),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn_like(x)

    torch.autograd.grad(rmsnorm(x, weight), (x, weight), dout)
    inference_x = x.detach()
    inference_weight = weight.detach()
    _assert_close(
        rmsnorm(inference_x, inference_weight),
        _reference(inference_x, inference_weight, 1e-6),
    )

    assert {key[-2:] for key in rmsnorm_flydsl_impl._BWD_CACHE} == {(2048, 8)}
    expected_blocks = min(
        (x.shape[0] + 7) // 8,
        torch.cuda.get_device_properties(x.device).multi_processor_count * 9,
    )
    persistent_geometries = {
        key[-8:-5] for key in rmsnorm_flydsl_impl._FWD_CACHE if key[-6] is not None
    }
    assert persistent_geometries == {(32, 8, expected_blocks)}


def test_n1024_target_packs_two_rows_per_persistent_block():
    _clear_caches()
    torch.manual_seed(30)
    x = torch.randn((32768, 1024), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32)

    actual = rmsnorm(x, weight)
    row_indices = torch.tensor(
        [0, 1, 2, 63, 127, 1023, 16383, 32766, 32767],
        device=x.device,
    )
    _assert_close(
        actual.index_select(0, row_indices),
        _reference(x.index_select(0, row_indices), weight, 1e-6),
    )

    expected_blocks = min(
        (x.shape[0] + 1) // 2,
        torch.cuda.get_device_properties(x.device).multi_processor_count * 64,
    )
    assert {key[-8:-1] for key in rmsnorm_flydsl_impl._FWD_CACHE} == {
        (64, 2, expected_blocks, 3, True, True, 7)
    }

    _clear_caches()
    padded_storage = torch.randn(
        (32768, 1027),
        device="cuda",
        dtype=torch.bfloat16,
    )
    padded = padded_storage[:, :1024]
    padded_actual = rmsnorm(padded, weight)
    _assert_close(
        padded_actual.index_select(0, row_indices),
        _reference(padded.index_select(0, row_indices), weight, 1e-6),
    )
    assert {key[-8:-1] for key in rmsnorm_flydsl_impl._FWD_CACHE} == {
        (64, None, None, None, False, False, None)
    }


def test_compiled_rmsnorm_switches_from_plain_to_per_head():
    torch._dynamo.reset()
    compiled = torch.compile(rmsnorm, fullgraph=True)

    x = torch.randn(
        (4, 256),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
    compiled(x, weight).sum().backward()
    assert x.grad is not None and weight.grad is not None

    x_head = torch.randn(
        (2, 3, 4, 64),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight_head = torch.randn(
        (4, 64),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    bias_head = torch.randn(
        (4, 64),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    residual_head = torch.randn_like(x_head, requires_grad=True)
    compiled(
        x_head,
        weight_head,
        bias=bias_head,
        residual=residual_head,
    ).sum().backward()
    for tensor in (x_head, weight_head, bias_head, residual_head):
        assert tensor.grad is not None


def test_dynamic_compiled_backward_selects_at_runtime():
    @torch.compile(fullgraph=True, dynamic=True)
    def compiled(x, weight, bias):
        return rmsnorm(x, weight, bias=bias)

    x = torch.randn(
        (512, 760),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)

    compiled(x, weight, bias).sum().backward()

    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None


def test_backward_with_bias_matches_reference():
    torch.manual_seed(15)
    shape = (512, 760)
    x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn_like(x)

    actual = rmsnorm(x, weight, bias=bias)
    actual.backward(dout)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    expected, _ = _full_reference(x_ref, weight_ref, bias_ref)
    expected.backward(dout)

    _assert_close(actual, expected)
    _assert_grad_close(x.grad, x_ref.grad)
    _assert_grad_close(weight.grad, weight_ref.grad)
    _assert_grad_close(bias.grad, bias_ref.grad)


@pytest.mark.parametrize("requested", ["weight", "bias"])
def test_staged_per_head_backward_supports_selective_parameter_grads(requested):
    x = torch.randn(
        (512, 2, 72),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(
        (2, 72),
        device="cuda",
        dtype=torch.float32,
        requires_grad=requested == "weight",
    )
    bias = torch.randn(
        (2, 72),
        device="cuda",
        dtype=torch.float32,
        requires_grad=requested == "bias",
    )

    rmsnorm(x, weight, bias=bias).sum().backward()

    assert x.grad is not None
    assert (weight.grad is not None) == (requested == "weight")
    assert (bias.grad is not None) == (requested == "bias")


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("n", [760, 32768])
def test_backward_with_no_parameter_grads(per_head, n):
    """Frozen parameters used to be the atomic kernel's other job.

    Nothing reduces, so the persistent kernel covers the rows and the parameter
    reduce is not launched at all.
    """
    torch.manual_seed(11)
    shape = (64, 2, n) if per_head else (64, n)
    parameter_shape = (2, n) if per_head else (n,)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(parameter_shape, device="cuda", dtype=torch.float32)
    bias = torch.randn(parameter_shape, device="cuda", dtype=torch.float32)
    dout = torch.randn_like(x)

    rmsnorm(x, weight, bias=bias).backward(dout)

    assert weight.grad is None and bias.grad is None
    x_ref = x.detach().clone().requires_grad_(True)
    expected, _ = _full_reference(x_ref, weight, bias)
    expected.backward(dout)
    _assert_grad_close(x.grad, x_ref.grad)


def test_workspace_descriptors_are_row_scoped():
    source = inspect.getsource(rmsnorm_flydsl_impl.build_rmsnorm_bwd_two_stage_module)
    assert "make_buffer_tensor(workspace_tensor)" not in source
    assert source.count("row_buffer(") >= 4


@pytest.mark.parametrize("n", [256, 760, 32768])
def test_deterministic_backward_is_reproducible(n):
    torch.manual_seed(16)
    x = torch.randn((64, n), device="cuda", dtype=torch.float16)
    weight = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    dout = torch.randn_like(x)
    results = []

    torch.use_deterministic_algorithms(True)
    try:
        for _ in range(2):
            x_i = x.detach().clone().requires_grad_(True)
            weight_i = weight.detach().clone().requires_grad_(True)
            bias_i = bias.detach().clone().requires_grad_(True)
            rmsnorm(x_i, weight_i, bias=bias_i).backward(dout)
            results.append((x_i.grad, weight_i.grad, bias_i.grad))
    finally:
        torch.use_deterministic_algorithms(False)

    for first, second in zip(results[0], results[1]):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


class _FlyDSLOpCounter(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "_rmsnorm_flydsl_" in str(func):
            self.count += 1
        return func(*args, **(kwargs or {}))


class _AtenOpRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    """Every aten op a region dispatches, for asserting on what it does not do."""

    def __init__(self):
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))

    def matching(self, needle: str) -> list[str]:
        return [op for op in self.ops if needle in op]


def _clear_caches():
    rmsnorm_flydsl_impl._FWD_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CACHE.clear()
    rmsnorm_flydsl_impl._FWD_AUTOTUNED_FAST_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_AUTOTUNED_FAST_CACHE.clear()
    rmsnorm_flydsl_impl._EAGER_EMPTY_CACHE.clear()
    rmsnorm_flydsl_impl._FWD_CU_COUNT_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CU_COUNT_CACHE.clear()
    rmsnorm_flydsl_impl._DEVICE_ARCH_CACHE.clear()
    rmsnorm_flydsl_impl._AUTOTUNE_ARCH_CACHE.clear()
    rmsnorm_flydsl_impl._rmsnorm_fwd_tuner.cache.clear()
    rmsnorm_flydsl_impl._rmsnorm_fwd_tuner._artifact_cache.clear()
    rmsnorm_flydsl_impl._rmsnorm_fwd_tuner._compiled_cache.clear()
    rmsnorm_flydsl_impl._rmsnorm_fwd_tuner._compiled_lookup.clear()
    rmsnorm_flydsl_impl._rmsnorm_fwd_tuner._device_jit_functions.clear()
    rmsnorm_flydsl_impl._rmsnorm_fwd_tuner._hot_cache.clear()
    rmsnorm_flydsl_impl._rmsnorm_bwd_tuner.cache.clear()
    rmsnorm_flydsl_impl._rmsnorm_bwd_tuner._artifact_cache.clear()
    rmsnorm_flydsl_impl._rmsnorm_bwd_tuner._compiled_cache.clear()
    rmsnorm_flydsl_impl._rmsnorm_bwd_tuner._compiled_lookup.clear()
    rmsnorm_flydsl_impl._rmsnorm_bwd_tuner._device_jit_functions.clear()
    rmsnorm_flydsl_impl._rmsnorm_bwd_tuner._hot_cache.clear()


def test_custom_ops_are_unique_mutation_only_and_fake_safe():
    fwd = torch.ops.quack._rmsnorm_flydsl_fwd.default
    fwd_tuned = torch.ops.quack._rmsnorm_flydsl_fwd_autotuned.default
    bwd = torch.ops.quack._rmsnorm_flydsl_bwd.default
    bwd_tuned = torch.ops.quack._rmsnorm_flydsl_bwd_autotuned.default
    for op in (fwd, fwd_tuned, bwd, bwd_tuned):
        assert str(op._schema).endswith("-> ()")
    # The staged workspace is scratch the launcher allocates for itself; nothing
    # outside the op reads it, so it is not in the schema.
    assert "workspace" not in str(bwd._schema)
    assert "Tensor(a4!) out" in str(fwd._schema)
    assert "Tensor(a5!) residual_out" in str(fwd._schema)
    assert "Tensor(a8!) dbias" in str(bwd._schema)

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty((2, 64), device="cuda", dtype=torch.float16)
        weight = torch.empty(64, device="cuda", dtype=torch.float16)
        out = torch.empty_like(x)
        rstd = torch.empty(2, device="cuda", dtype=torch.float32)
        dout = torch.empty_like(x)
        dx = torch.empty_like(x)
        dweight = torch.empty_like(weight)

        bias = torch.empty_like(weight)
        residual = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        for op in (fwd, fwd_tuned):
            op(
                x,
                weight,
                bias,
                residual,
                out,
                residual_out,
                rstd,
                1e-6,
                0.0,
                True,
                True,
                True,
                True,
                True,
                False,
                1,
            )
        dresidual = torch.empty_like(x)
        dbias = torch.empty_like(weight, dtype=torch.float32)
        for op in (bwd, bwd_tuned):
            op(
                x,
                weight,
                dout,
                residual,
                rstd,
                dx,
                dresidual,
                dweight,
                dbias,
                0.0,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                1,
            )


def test_eager_fast_path_bypasses_custom_op_dispatch():
    _clear_caches()
    x = torch.randn(
        (4, 136),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(
        136,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    with _FlyDSLOpCounter() as counter:
        rmsnorm(x, weight).sum().backward()
    assert counter.count == 0
    assert not rmsnorm_flydsl_impl._EAGER_EMPTY_CACHE


def test_eager_no_grad_inputs_bypass_autograd_function(monkeypatch):
    x = torch.randn((8, 136), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(136, device="cuda", dtype=torch.float32)

    def forbid_apply(*args, **kwargs):
        raise AssertionError("inference entered autograd.Function")

    monkeypatch.setattr(rmsnorm_flydsl_impl._RMSNormFunction, "apply", forbid_apply)
    _assert_close(rmsnorm(x, weight), _reference(x, weight, 1e-6))


def test_eager_no_grad_reuses_internal_empty_placeholders(monkeypatch):
    _clear_caches()
    x = torch.randn((8, 136), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(136, device="cuda", dtype=torch.float32)
    seen = []
    original = rmsnorm_flydsl_impl._eager_empty

    def record(device, dtype):
        result = original(device, dtype)
        seen.append((device, dtype, result))
        return result

    monkeypatch.setattr(rmsnorm_flydsl_impl, "_eager_empty", record)
    first = rmsnorm(x, weight)
    second = rmsnorm(x.clone(), weight.clone())

    _assert_close(first, _reference(x, weight, 1e-6))
    _assert_close(second, _reference(x, weight, 1e-6))
    for dtype, expected_calls in ((torch.bfloat16, 4), (torch.float32, 2)):
        placeholders = [tensor for _, seen_dtype, tensor in seen if seen_dtype == dtype]
        assert len(placeholders) == expected_calls
        assert all(tensor is placeholders[0] for tensor in placeholders)
        assert placeholders[0].numel() == 0
        assert placeholders[0].device == x.device
    assert set(rmsnorm_flydsl_impl._EAGER_EMPTY_CACHE) == {
        (x.device, torch.bfloat16),
        (x.device, torch.float32),
    }


def test_fullgraph_forward_backward_cold_and_warm_cache():
    torch._dynamo.reset()
    _clear_caches()
    eps = 1e-5

    @torch.compile(fullgraph=True)
    def compiled_rmsnorm(x, weight):
        return rmsnorm(x, weight, eps=eps)

    torch.manual_seed(4)
    x = torch.randn(
        (8, 520),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(
        520,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    dout = torch.randn_like(x)
    actual = compiled_rmsnorm(x, weight)
    actual.backward(dout)
    expected, dx_expected, dw_expected = _reference_with_grads(
        x,
        weight,
        dout,
        eps,
    )
    _assert_close(actual, expected)
    _assert_grad_close(x.grad, dx_expected)
    _assert_grad_close(weight.grad, dw_expected)

    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1
    assert len(rmsnorm_flydsl_impl._BWD_CACHE) == 1
    fwd_launcher = next(iter(rmsnorm_flydsl_impl._FWD_CACHE.values()))
    bwd_launcher = next(iter(rmsnorm_flydsl_impl._BWD_CACHE.values()))
    fwd_compiled = fwd_launcher._cf
    bwd_compiled = bwd_launcher._cf

    x_warm = x.detach().clone().requires_grad_()
    weight_warm = weight.detach().clone().requires_grad_()
    warm = compiled_rmsnorm(x_warm, weight_warm)
    warm.backward(dout)
    _assert_close(warm, expected)
    _assert_grad_close(x_warm.grad, dx_expected)
    _assert_grad_close(weight_warm.grad, dw_expected)
    assert next(iter(rmsnorm_flydsl_impl._FWD_CACHE.values())) is fwd_launcher
    assert next(iter(rmsnorm_flydsl_impl._BWD_CACHE.values())) is bwd_launcher
    assert fwd_launcher._cf is fwd_compiled
    assert bwd_launcher._cf is bwd_compiled
    assert not rmsnorm_flydsl_impl._EAGER_EMPTY_CACHE


def test_fullgraph_two_stage_backward():
    torch._dynamo.reset()
    _clear_caches()

    @torch.compile(fullgraph=True)
    def compiled_rmsnorm(x, weight):
        return rmsnorm(x, weight)

    torch.manual_seed(6)
    x = (torch.randn((512, 4096), device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
    weight = (1.0 + torch.randn(4096, device="cuda", dtype=torch.float32) * 0.1).requires_grad_()
    dout = torch.randn_like(x) * 0.1

    actual = compiled_rmsnorm(x, weight)
    actual.backward(dout)
    expected, dx_expected, dw_expected = _reference_with_grads(
        x,
        weight,
        dout,
        1e-6,
    )

    _assert_close(actual, expected)
    _assert_grad_close(x.grad, dx_expected)
    _assert_grad_close(weight.grad, dw_expected)
    # One staged backward built, not one per graph the compiler produced.
    assert len(rmsnorm_flydsl_impl._BWD_CACHE) == 1


def test_forward_cache_identity_is_shape_and_dtype_only():
    """eps is a launch argument, so it must not multiply compiled kernels."""
    _clear_caches()

    def call(n, dtype, eps):
        x = torch.randn((2, n), device="cuda", dtype=dtype)
        weight = torch.randn(n, device="cuda", dtype=dtype)
        return rmsnorm(x, weight, eps=eps)

    call(136, torch.float16, 1e-6)
    first_launcher = next(iter(rmsnorm_flydsl_impl._FWD_CACHE.values()))
    call(136, torch.float16, 1e-6)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1
    assert next(iter(rmsnorm_flydsl_impl._FWD_CACHE.values())) is first_launcher

    for eps in (1e-5, 1e-4, 3e-7, 0.5):
        call(136, torch.float16, eps)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1

    call(144, torch.float16, 1e-6)
    call(136, torch.bfloat16, 1e-6)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 3


def _direct_autotune_call_args(
    *,
    rows=8,
    n=512,
    has_weight=True,
    has_bias=False,
    has_residual=False,
    store_residual=False,
    store_rstd=False,
):
    x = torch.randn((rows, n), device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(n, device=x.device, dtype=torch.float32)
        if has_weight
        else torch.empty(0, device=x.device, dtype=torch.bfloat16)
    )
    bias = (
        torch.randn(n, device=x.device, dtype=torch.float32)
        if has_bias
        else torch.empty(0, device=x.device, dtype=torch.bfloat16)
    )
    residual = torch.randn_like(x) if has_residual else x
    output = torch.empty_like(x)
    residual_out = (
        torch.empty_like(x)
        if store_residual
        else torch.empty(0, device=x.device, dtype=torch.bfloat16)
    )
    rstd = (
        torch.empty(rows, device=x.device, dtype=torch.float32)
        if store_rstd
        else torch.empty(0, device=x.device, dtype=torch.float32)
    )
    args = (x, weight, bias, residual, output, residual_out, rstd, rows, 1e-6, 0.0)
    kwargs = {
        "n": n,
        "has_weight": has_weight,
        "has_bias": has_bias,
        "has_residual": has_residual,
        "store_residual": store_residual,
        "store_rstd": store_rstd,
        "per_head": False,
        "num_heads": 1,
    }
    return args, kwargs


def test_autotune_schema_five_and_candidates_retain_the_heuristic():
    assert rmsnorm_autotune_impl.RMSNORM_AUTOTUNE_SCHEMA_VERSION == 5
    for n, dtype_name in ((128, "bf16"), (512, "bf16"), (2048, "bf16"), (4096, "f32")):
        default = rmsnorm_autotune_impl.rmsnorm_default_config(
            n=n,
            input_dtype_str=dtype_name,
        )
        candidates = rmsnorm_autotune_impl.rmsnorm_search_configs(
            n=n,
            input_dtype_str=dtype_name,
        )
        assert any(
            config.kwargs["threads_per_row"] == default.kwargs["threads_per_row"]
            and config.waves_per_eu is None
            for config in candidates
        )


@pytest.mark.parametrize(
    (
        "n",
        "threads",
        "blocks_per_cu",
        "rows_per_block",
        "output_cache_modifier",
        "persistent_single_pass",
        "packed_flat_rows",
        "waves_per_eu",
    ),
    [
        (256, 32, 9, 8, None, False, False, None),
        (512, 64, 56, 1, None, False, False, None),
        (1024, 64, 64, 2, 3, True, True, 7),
    ],
)
def test_autotune_offers_persistent_candidate_for_supported_plain_shape(
    n,
    threads,
    blocks_per_cu,
    rows_per_block,
    output_cache_modifier,
    persistent_single_pass,
    packed_flat_rows,
    waves_per_eu,
):
    args, kwargs = _direct_autotune_call_args(n=n)
    args = args[:7] + (32768,) + args[8:]
    kwargs.update(
        input_dtype_str="bf16",
        output_dtype_str="bf16",
        weight_dtype_str="f32",
    )

    candidates = rmsnorm_autotune_impl.rmsnorm_search_configs(*args, **kwargs)
    num_cus = torch.cuda.get_device_properties(args[0].device).multi_processor_count

    matching = [
        config
        for config in candidates
        if config.kwargs.get("persistent_programs")
        == min((32768 + rows_per_block - 1) // rows_per_block, num_cus * blocks_per_cu)
        and config.kwargs["threads_per_row"] == threads
    ]
    assert len(matching) == 1
    candidate = matching[0]
    assert candidate.kwargs["row_groups_per_block"] == rows_per_block
    assert candidate.kwargs.get("output_cache_modifier") == output_cache_modifier
    assert candidate.kwargs.get("persistent_single_pass", False) is persistent_single_pass
    assert candidate.kwargs.get("packed_flat_rows", False) is packed_flat_rows
    assert candidate.waves_per_eu == waves_per_eu


def test_l2_rotation_clones_preserve_metadata_aliases_and_distinct_addresses():
    base = torch.randn((8, 520), device="cuda", dtype=torch.bfloat16)
    x = base[:, :512]
    weight = torch.randn(512, device=x.device, dtype=torch.float32)
    absent = torch.empty(0, device=x.device, dtype=torch.bfloat16)
    output = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    rstd = torch.empty(0, device=x.device, dtype=torch.float32)
    args = (x, weight, absent, x, output, absent, rstd, 8, 1e-6, 0.0)
    kwargs = {"has_weight": True, "has_residual": False}

    first_args, first_kwargs = rmsnorm_autotune_impl._clone_tensor_arguments(args, kwargs)
    second_args, _ = rmsnorm_autotune_impl._clone_tensor_arguments(args, kwargs)

    assert first_args[0] is first_args[3], "the absent-residual alias to x was broken"
    assert first_args[2] is first_args[5], "shared empty placeholders were split"
    for original, clone in zip(args[:7], first_args[:7]):
        assert clone.shape == original.shape
        assert clone.dtype == original.dtype
        assert clone.stride() == original.stride()
        assert clone.device == original.device
    assert first_kwargs == kwargs
    for index in (0, 1, 4):
        pointers = {
            args[index].data_ptr(),
            first_args[index].data_ptr(),
            second_args[index].data_ptr(),
        }
        assert len(pointers) == 3
    assert first_args[2].numel() == first_args[5].numel() == 0


def test_l2_rotation_working_set_exceeds_cache_and_is_device_local(monkeypatch):
    args, kwargs = _direct_autotune_call_args()
    monkeypatch.setattr(
        rmsnorm_autotune_impl,
        "_cache_authority",
        lambda device: rmsnorm_autotune_impl._CacheAuthority(4096, "test"),
    )
    plan = rmsnorm_autotune_impl._build_l2_rotation_plan(
        args,
        kwargs,
        n_timed_calls=8,
    )

    assert len(plan.arg_sets) >= 2
    assert plan.working_set_bytes >= 3 * plan.cache_bytes
    assert plan.read_bytes_per_set == (
        args[0].numel() * args[0].element_size() + args[1].numel() * args[1].element_size()
    )
    assert plan.cache_source == "test"
    assert plan.bench_stream.device == args[0].device
    for arg_index in (0, 1, 4):
        pointers = [arg_set[arg_index].data_ptr() for arg_set in plan.arg_sets]
        assert len(pointers) == len(set(pointers))
    assert all(arg_set[3] is arg_set[0] for arg_set in plan.arg_sets)


def test_clone_oom_fallback_keeps_multiple_addresses_and_per_call_eviction(monkeypatch):
    args, kwargs = _direct_autotune_call_args(rows=1, n=64)
    monkeypatch.setattr(
        rmsnorm_autotune_impl,
        "_cache_authority",
        lambda device: rmsnorm_autotune_impl._CacheAuthority(4096, "test"),
    )
    real_clone = rmsnorm_autotune_impl._clone_tensor_arguments
    attempts = 0

    def fail_during_full_rotation(args, kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise torch.OutOfMemoryError("synthetic clone OOM")
        return real_clone(args, kwargs)

    monkeypatch.setattr(
        rmsnorm_autotune_impl,
        "_clone_tensor_arguments",
        fail_during_full_rotation,
    )
    plan = rmsnorm_autotune_impl._build_l2_rotation_plan(
        args,
        kwargs,
        n_timed_calls=8,
    )

    assert len(plan.arg_sets) == 2
    assert plan.arg_sets[0][0].data_ptr() != plan.arg_sets[1][0].data_ptr()
    assert plan.working_set_bytes > plan.cache_bytes
    assert plan.per_call_eviction
    assert "clone allocation was reduced" in plan.fallback_reason


def test_gfx950_cache_authority_uses_the_triton_benchmark_capacity():
    authority = rmsnorm_autotune_impl._cache_authority(torch.device("cuda", 0))
    assert authority.cache_bytes >= 256 * 1024**2
    assert "Triton ROCm benchmark eviction buffer" in authority.source
    assert authority.seed_buffer is not None
    assert authority.seed_buffer.numel() * authority.seed_buffer.element_size() >= 256 * 1024**2


def test_wide_autotune_candidates_all_use_the_reload_path():
    configs = rmsnorm_search_configs(n=262144, input_dtype_str="bf16")
    assert {config.kwargs["threads_per_row"] for config in configs} == {64, 128, 256, 512}
    for candidate in configs:
        row = RmsNormRowConfig.with_num_threads(
            262144,
            16,
            candidate.kwargs["threads_per_row"],
            max_num_threads=MAX_TUNED_NUM_THREADS,
        )
        assert row.reload_from == "gmem"


def test_n32768_autotune_alone_offers_a_single_read_1024_thread_config():
    target = rmsnorm_search_configs(n=32768, input_dtype_str="bf16")
    neighbors = {n: rmsnorm_search_configs(n=n, input_dtype_str="bf16") for n in (16384, 65536)}

    assert {config.kwargs["threads_per_row"] for config in target} == {
        64,
        128,
        256,
        512,
        1024,
    }
    assert all(
        1024 not in {config.kwargs["threads_per_row"] for config in configs}
        for configs in neighbors.values()
    )
    row = RmsNormRowConfig.with_num_threads(
        32768,
        16,
        1024,
        max_num_threads=MAX_TUNED_NUM_THREADS,
    )
    assert row.reload_from is None


def _direct_bwd_autotune_call_args(m=64, n=512):
    source = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device="cuda", dtype=torch.float32)
    dout = torch.randn_like(source)
    rstd = torch.rsqrt(source.float().square().mean(dim=-1) + 1e-6)
    dx = torch.empty_like(source)
    dweight = torch.empty_like(weight)
    absent = torch.empty(0, device=source.device, dtype=source.dtype)
    dbias = torch.empty(1, device=source.device, dtype=weight.dtype)
    workspace = torch.empty((1, n), device=source.device, dtype=torch.float32)
    args = (
        source,
        weight,
        dout,
        source,
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
        "arch": "gfx950",
        "schema_version": rmsnorm_bwd_autotune_impl.RMSNORM_BWD_AUTOTUNE_SCHEMA_VERSION,
        "stream": torch.cuda.current_stream().cuda_stream,
    }
    return args, kwargs


@pytest.mark.parametrize("n", [256, 512, 1024])
def test_backward_autotune_candidates_cover_row_and_grid_axes(n):
    args, kwargs = _direct_bwd_autotune_call_args(m=2048, n=n)
    configs = rmsnorm_bwd_search_configs(*args, **kwargs)
    default = rmsnorm_bwd_default_config(*args, **kwargs)
    identities = {
        (
            config.kwargs["threads_per_row"],
            config.kwargs["num_programs"],
            config.kwargs["parameter_reduce_cols"],
        )
        for config in configs
    }
    default_identity = (
        default.kwargs["threads_per_row"],
        default.kwargs["num_programs"],
        default.kwargs["parameter_reduce_cols"],
    )

    assert default_identity in identities
    assert all(threads >= 64 and threads & (threads - 1) == 0 for threads, _, _ in identities)
    assert all(
        1 <= cols <= PARAMETER_REDUCE_THREADS and cols & (cols - 1) == 0
        for _, _, cols in identities
    )
    assert min(programs for _, programs, _ in identities) < 1536
    assert len({programs for _, programs, _ in identities}) >= 3
    assert len({cols for _, _, cols in identities}) >= 2


@pytest.mark.parametrize(
    ("parameter_numel", "expected_cols"),
    [(256, 1), (512, 2), (1024, 4), (8192, 32), (32768, 128)],
)
def test_parameter_reduce_geometry_targets_one_grid_wave(parameter_numel, expected_cols):
    cols = rmsnorm_bwd_parameter_reduce_cols(
        parameter_numel,
        num_programs=1536,
        target_blocks=256,
    )

    assert cols == expected_cols
    assert (parameter_numel + cols - 1) // cols == 256


def test_parameter_reduce_geometry_does_not_outgrow_tiny_partial_counts():
    assert (
        rmsnorm_bwd_parameter_reduce_cols(
            parameter_numel=256,
            num_programs=1,
            target_blocks=256,
        )
        == PARAMETER_REDUCE_THREADS
    )


def test_autotuned_backward_default_matches_reference_without_using_plain_cache():
    _clear_caches()
    torch.manual_seed(24)
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn_like(x)

    output = rmsnorm_autotuned(x, weight)
    dx, dweight = torch.autograd.grad(output, (x, weight), dout)
    _, dx_ref, dweight_ref = _reference_with_grads(x, weight, dout, 1e-6)
    assert not rmsnorm_flydsl_impl._BWD_CACHE
    assert len(rmsnorm_flydsl_impl._rmsnorm_bwd_tuner._hot_cache) == 1
    x_plain = x.detach().clone().requires_grad_(True)
    weight_plain = weight.detach().clone().requires_grad_(True)
    plain = rmsnorm(x_plain, weight_plain)
    dx_plain, dweight_plain = torch.autograd.grad(
        plain,
        (x_plain, weight_plain),
        dout,
    )

    _assert_grad_close(dx, dx_ref)
    _assert_grad_close(dweight, dweight_ref)
    torch.testing.assert_close(dx, dx_plain, rtol=0.0, atol=0.0)
    torch.testing.assert_close(dweight, dweight_plain, rtol=0.0, atol=0.0)


def test_autotuned_backward_searches_once_then_reuses_pinned_winner(tmp_path, monkeypatch):
    from flydsl.autotune import Config

    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_bwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        tuner,
        "configs",
        [
            Config(threads_per_row=64, num_programs=32, parameter_reduce_cols=8),
            Config(threads_per_row=64, num_programs=64, parameter_reduce_cols=4),
        ],
    )
    completed = 0

    def bench_once(call, warmup, rep):
        nonlocal completed
        call()
        torch.cuda.synchronize()
        completed += 1
        return float(completed)

    monkeypatch.setattr(tuner, "_do_bench", bench_once)
    args, kwargs = _direct_bwd_autotune_call_args()

    tuner(*args, **kwargs)
    assert completed == 2
    assert len(tuner.cache) == len(tuner._hot_cache) == 1
    artifacts = list((tmp_path / "artifacts").glob("*.json"))
    assert len(artifacts) == 1
    first_dx = args[6].clone()
    first_dweight = args[8].clone()

    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    monkeypatch.setattr(
        tuner,
        "_do_bench",
        lambda *args, **kwargs: pytest.fail("hot winner unexpectedly benchmarked"),
    )
    tuner(*args, **kwargs)

    assert completed == 2
    torch.testing.assert_close(args[6], first_dx, rtol=0.0, atol=0.0)
    torch.testing.assert_close(args[8], first_dweight, rtol=0.0, atol=0.0)


def test_autotuned_backward_runs_on_non_default_stream():
    _clear_caches()
    torch.manual_seed(26)
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        output = rmsnorm_autotuned(x, weight)
        dx, dweight = torch.autograd.grad(output, (x, weight), dout)
    stream.synchronize()
    _, dx_ref, dweight_ref = _reference_with_grads(x, weight, dout, 1e-6)
    _assert_grad_close(dx, dx_ref)
    _assert_grad_close(dweight, dweight_ref)


def test_autotuned_backward_handles_per_head_residual_and_selective_grads():
    _clear_caches()
    torch.manual_seed(25)
    x = torch.randn((4, 2, 512), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn((2, 512), device="cuda", dtype=torch.float32)
    bias = torch.randn((2, 512), device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn_like(x)

    actual = rmsnorm_autotuned(x, weight, bias=bias, residual=residual)
    dresidual, dbias = torch.autograd.grad(actual, (residual, bias), dout)

    residual_ref = residual.detach().float().requires_grad_(True)
    bias_ref = bias.detach().float().requires_grad_(True)
    expected, _ = _full_reference(
        x.float(),
        weight.float(),
        bias_ref,
        residual_ref,
    )
    dresidual_ref, dbias_ref = torch.autograd.grad(
        expected,
        (residual_ref, bias_ref),
        dout.float(),
    )
    _assert_grad_close(dresidual, dresidual_ref.to(dresidual.dtype))
    _assert_grad_close(dbias, dbias_ref)


def test_autotuned_fullgraph_dynamic_backward_uses_tuned_custom_op():
    _clear_caches()
    torch._dynamo.reset()
    compiled = torch.compile(rmsnorm_autotuned, fullgraph=True, dynamic=True)
    weight = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    for rows in (8, 16):
        x = torch.randn(
            (rows, 512),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        dout = torch.randn_like(x)
        output = compiled(x, weight)
        dx, dweight = torch.autograd.grad(output, (x, weight), dout)
        _, dx_ref, dweight_ref = _reference_with_grads(x, weight, dout, 1e-6)
        _assert_grad_close(dx, dx_ref)
        _assert_grad_close(dweight, dweight_ref)


def test_autotuned_wide_forward_matches_reference(tmp_path, monkeypatch):
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    x = torch.randn((4, 32768), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(32768, device="cuda", dtype=torch.float32)
    _assert_close(rmsnorm_autotuned(x, weight), _reference(x, weight, 1e-6))


def test_autotuned_n32768_1024_thread_forward_matches_reference(tmp_path, monkeypatch):
    from flydsl.autotune import Config

    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(tuner, "configs", [Config(threads_per_row=1024)])

    def bench_once(call, warmup, rep):
        call()
        return 1.0

    monkeypatch.setattr(tuner, "_do_bench", bench_once)
    x = torch.randn((4, 32768), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(32768, device="cuda", dtype=torch.float32)
    _assert_close(rmsnorm_autotuned(x, weight), _reference(x, weight, 1e-6))
    assert next(iter(tuner.cache.values())).kwargs["threads_per_row"] == 1024


def test_default_and_autotuned_forward_use_independent_caches(tmp_path, monkeypatch):
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float32)

    expected = _reference(x, weight, 1e-6)
    _assert_close(rmsnorm(x, weight), expected)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1
    assert not tuner.cache

    _assert_close(rmsnorm_autotuned(x, weight), expected)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1
    assert not tuner.cache  # The default heuristic does not pretend to be a searched winner.


def test_a_warm_tuned_call_reuses_the_compiled_function(monkeypatch):
    # A process-hot call bypasses the tuner object and launches the resolved
    # CompiledFunction from the adapter's flat winner cache.
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    entries = 0
    real_run_config = tuner._run_config

    def counted(*args, **kwargs):
        nonlocal entries
        entries += 1
        return real_run_config(*args, **kwargs)

    monkeypatch.setattr(tuner, "_run_config", counted)

    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float32)
    expected = _reference(x, weight, 1e-6)

    _assert_close(rmsnorm_autotuned(x, weight), expected)
    assert entries == 1, "the first call has to resolve a winner"
    assert len(rmsnorm_flydsl_impl._FWD_AUTOTUNED_FAST_CACHE) == 1

    tuner_type = type(tuner)
    real_tuner_call = tuner_type.__call__

    def forbid_tuner_reentry(self, *args, **kwargs):
        if self is tuner:
            raise AssertionError("a warm adapter hit re-entered the tuner")
        return real_tuner_call(self, *args, **kwargs)

    monkeypatch.setattr(tuner_type, "__call__", forbid_tuner_reentry)

    for _ in range(5):
        _assert_close(rmsnorm_autotuned(x, weight), expected)
    assert entries == 1, f"a warm call selected a config {entries - 1} more time(s)"
    assert len(tuner._hot_cache) == 1
    assert len(tuner._compiled_cache) == 1


def test_resolved_winner_generation_tracks_environment_and_hints(monkeypatch):
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    first = tuner.fast_context_token()
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "generation-test")
    second = tuner.fast_context_token()
    assert second != first

    monkeypatch.setattr(tuner.fn, "compile_hints", {"waves_per_eu": 2})
    third = tuner.fast_context_token()
    assert third != second


def test_autotuned_forward_searches_all_candidates_then_hits_cache(tmp_path, monkeypatch):
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    completed = 0
    candidate_count = 0
    real_configs = tuner.configs

    def counted_configs(*args, **kwargs):
        nonlocal candidate_count
        configs = real_configs(*args, **kwargs)
        candidate_count = len(configs)
        return configs

    def bench_once(call, warmup, rep):
        nonlocal completed
        call()
        torch.cuda.synchronize()
        completed += 1
        return float(completed)

    monkeypatch.setattr(tuner, "configs", counted_configs)
    monkeypatch.setattr(tuner, "_do_bench", bench_once)
    x = torch.randn((16, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float32)

    first = rmsnorm_autotuned(x, weight, eps=1e-5)
    _assert_close(first, _reference(x, weight, 1e-5))
    assert completed == candidate_count > 1
    assert len(tuner.cache) == 1
    assert tuner.cache[next(iter(tuner.cache))].to_dict() in [
        config.to_dict() for config in real_configs(n=4096, input_dtype_str="bf16")
    ]
    artifacts = list((tmp_path / "artifacts").glob("*.json"))
    assert len(artifacts) == 1
    assert "eps" not in tuner.key and "weight_offset" not in tuner.key

    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    monkeypatch.setattr(
        tuner,
        "_do_bench",
        lambda *args, **kwargs: pytest.fail("cache hit unexpectedly benchmarked"),
    )
    cached = rmsnorm_autotuned(x, weight, eps=0.5, weight_offset=1.0)
    _assert_close(cached, _reference(x, weight + 1.0, 0.5))
    assert completed == candidate_count


def test_candidates_and_winner_cache_hits_use_compiled_functions(tmp_path, monkeypatch):
    """Candidate repetitions and a warm winner must never re-enter JitFunction."""
    from flydsl.autotune import Config

    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    config = Config(threads_per_row=128, waves_per_eu=2)
    monkeypatch.setattr(tuner, "configs", [config])

    jit_dispatches = 0
    benchmark_launches = 0
    jit_type = type(tuner.fn)
    real_jit_call = jit_type.__call__

    def counted_jit_call(self, *args, **kwargs):
        nonlocal jit_dispatches
        if self.func.__name__ == "rmsnorm_direct":
            jit_dispatches += 1
        return real_jit_call(self, *args, **kwargs)

    def repeat_fast_callable(call, warmup, rep):
        nonlocal benchmark_launches
        dispatches_before = jit_dispatches
        for _ in range(4):
            call()
            benchmark_launches += 1
        torch.cuda.synchronize()
        assert jit_dispatches == dispatches_before, "candidate timing re-entered JitFunction"
        return 1.0

    monkeypatch.setattr(jit_type, "__call__", counted_jit_call)
    monkeypatch.setattr(tuner, "_do_bench", repeat_fast_callable)
    original_hints = dict(tuner.fn.compile_hints)
    x = torch.randn((16, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device="cuda", dtype=torch.float32)

    first = rmsnorm_autotuned(x, weight)
    _assert_close(first, _reference(x, weight, 1e-6))
    assert benchmark_launches == 4
    assert jit_dispatches == 1, "one untimed flyc.compile call is expected per candidate"
    assert len(tuner.cache) == 1
    assert len(tuner._compiled_cache) == 1
    compiled = next(iter(tuner._compiled_cache.values()))
    assert tuner.fn.compile_hints == original_hints, "Config hints leaked onto the shared JIT"

    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    monkeypatch.setattr(
        tuner,
        "default",
        lambda *args, **kwargs: pytest.fail("warm winner unexpectedly used default config"),
    )

    def forbid_jit_dispatch(self, *args, **kwargs):
        if self.func.__name__ == "rmsnorm_direct":
            pytest.fail("warm winner cache hit re-entered JitFunction")
        return real_jit_call(self, *args, **kwargs)

    monkeypatch.setattr(jit_type, "__call__", forbid_jit_dispatch)
    second = rmsnorm_autotuned(x, weight, eps=0.5, weight_offset=1.0)
    _assert_close(second, _reference(x, weight + 1.0, 0.5))
    assert benchmark_launches == 4
    assert len(tuner._compiled_cache) == 1
    assert next(iter(tuner._compiled_cache.values())) is compiled


def test_production_cold_candidate_path_gates_every_clone_and_stays_on_fast_callable(
    tmp_path,
    monkeypatch,
):
    """The real graph objective, not only an injected bench, must bypass JitFunction."""
    from flydsl.autotune import Config

    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(tuner, "configs", [Config(threads_per_row=128)])
    monkeypatch.setattr(tuner, "warmup", 2.0)
    monkeypatch.setattr(tuner, "rep", 1)
    monkeypatch.setattr(
        rmsnorm_autotune_impl,
        "_cache_authority",
        lambda device: rmsnorm_autotune_impl._CacheAuthority(64 * 1024, "test"),
    )

    jit_dispatches = 0
    jit_type = type(tuner.fn)
    real_jit_call = jit_type.__call__

    def counted_jit_call(self, *args, **kwargs):
        nonlocal jit_dispatches
        if self.func.__name__ == "rmsnorm_direct":
            jit_dispatches += 1
        return real_jit_call(self, *args, **kwargs)

    gated = []
    real_gate = rmsnorm_autotune_impl._candidate_correctness_gate

    def checked_gate(compiled, positional_sets, plan):
        gated.append(plan)
        assert len(positional_sets) == len(plan.arg_sets) >= 2
        assert plan.working_set_bytes >= 3 * plan.cache_bytes
        return real_gate(compiled, positional_sets, plan)

    monkeypatch.setattr(jit_type, "__call__", counted_jit_call)
    monkeypatch.setattr(rmsnorm_autotune_impl, "_candidate_correctness_gate", checked_gate)

    x = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(512, device=x.device, dtype=torch.float32)
    bias = torch.randn(512, device=x.device, dtype=torch.float32)
    residual = torch.randn_like(x)
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream):
        actual, residual_out = rmsnorm_autotuned(
            x,
            weight,
            bias=bias,
            residual=residual,
            prenorm=True,
        )
    stream.synchronize()

    expected, expected_residual = _full_reference(x, weight, bias, residual)
    _assert_close(actual, expected)
    _assert_close(residual_out, expected_residual)
    assert jit_dispatches == 1, "only the untimed flyc.compile dispatch is permitted"
    assert len(gated) == 1
    assert gated[0].bench_stream.device == x.device


def test_graph_failure_fallback_remains_l2_cold_and_multi_address(tmp_path, monkeypatch):
    from flydsl.autotune import Config

    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(tuner, "configs", [Config(threads_per_row=128)])
    monkeypatch.setattr(tuner, "warmup", 0.1)
    monkeypatch.setattr(tuner, "rep", 1)
    monkeypatch.setattr(
        rmsnorm_autotune_impl,
        "_cache_authority",
        lambda device: rmsnorm_autotune_impl._CacheAuthority(64 * 1024, "test"),
    )

    def graph_failure():
        raise RuntimeError("graph unavailable in test")

    observed = []
    real_fallback = rmsnorm_autotune_impl._event_l2_rotate_bench

    def checked_fallback(compiled, positional_sets, plan, **kwargs):
        pointers = [positional[0].data_ptr() for positional in positional_sets]
        assert len(pointers) == len(set(pointers)) >= 2
        assert plan.working_set_bytes > plan.cache_bytes
        observed.append((len(pointers), plan.working_set_bytes))
        return real_fallback(compiled, positional_sets, plan, **kwargs)

    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph_failure)
    monkeypatch.setattr(rmsnorm_autotune_impl, "_event_l2_rotate_bench", checked_fallback)

    x = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device=x.device, dtype=torch.float32)
    with pytest.warns(RuntimeWarning, match="event-timed L2-cold multi-address fallback"):
        actual = rmsnorm_autotuned(x, weight)

    _assert_close(actual, _reference(x, weight, 1e-6))
    assert observed


def test_fast_callable_cache_partitions_config_and_tensor_abi(tmp_path, monkeypatch):
    """A Config or memref ABI change must produce a distinct fast callable."""
    from flydsl.autotune import Config

    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        tuner,
        "_do_bench",
        lambda call, warmup, rep: (call(), torch.cuda.synchronize(), 1.0)[-1],
    )
    x = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16)
    weight_f32 = torch.randn(512, device="cuda", dtype=torch.float32)

    monkeypatch.setattr(tuner, "configs", [Config(threads_per_row=64)])
    out_64 = rmsnorm_autotuned(x, weight_f32)
    _assert_close(out_64, _reference(x, weight_f32, 1e-6))
    callable_64 = next(iter(tuner._compiled_cache.values()))

    monkeypatch.setattr(tuner, "configs", [Config(threads_per_row=128)])
    out_128 = rmsnorm_autotuned(x, weight_f32)
    _assert_close(out_128, _reference(x, weight_f32, 1e-6))
    assert len(tuner._compiled_cache) == 2
    assert callable_64 not in tuple(tuner._compiled_cache.values())[1:]

    weight_bf16 = weight_f32.to(torch.bfloat16)
    out_bf16 = rmsnorm_autotuned(x, weight_bf16)
    _assert_close(out_bf16, _reference(x, weight_bf16, 1e-6))
    assert len(tuner._compiled_cache) == 3
    assert len({id(compiled) for compiled in tuner._compiled_cache.values()}) == 3


def test_autotuned_feature_outputs_on_non_default_stream(tmp_path, monkeypatch):
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    observed = []
    real_run_config = tuner._run_config

    def checked_run_config(config, args, kwargs):
        observed.append(
            (
                kwargs["has_bias"],
                kwargs["has_residual"],
                kwargs["store_residual"],
                kwargs["store_rstd"],
                kwargs["stream"],
            )
        )
        return real_run_config(config, args, kwargs)

    monkeypatch.setattr(tuner, "_run_config", checked_run_config)
    x = torch.randn((8, 760), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(x)
    weight = torch.randn(760, device="cuda", dtype=torch.float32)
    bias = torch.randn(760, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))

    with torch.cuda.stream(stream):
        actual, residual_out = rmsnorm_autotuned(
            x,
            weight,
            bias=bias,
            residual=residual,
            prenorm=True,
        )
    stream.synchronize()

    expected, expected_residual = _full_reference(x, weight, bias, residual)
    _assert_close(actual, expected)
    _assert_close(residual_out, expected_residual)

    # The second stream takes the process-hot callable path, where no Autotuner
    # stream context remains; the raw runtime stream argument must still win.
    second_stream = torch.cuda.Stream(device=x.device)
    second_stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(second_stream):
        hot_actual, hot_residual = rmsnorm_autotuned(
            x,
            weight,
            bias=bias,
            residual=residual,
            prenorm=True,
        )
    second_stream.synchronize()
    _assert_close(hot_actual, expected)
    _assert_close(hot_residual, expected_residual)
    assert observed == [(True, True, True, True, stream.cuda_stream)]


def test_a_runtime_eps_still_reaches_the_kernel():
    """A cached kernel must honour a new eps rather than the one it was built with."""
    _clear_caches()
    torch.manual_seed(11)
    x = torch.randn((4, 256), device="cuda", dtype=torch.float32)
    weight = torch.randn(256, device="cuda", dtype=torch.float32)

    for eps in (1e-6, 0.5, 8.0):
        _assert_close(rmsnorm(x, weight, eps=eps), _reference(x, weight, eps))
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1


def test_fullgraph_with_dynamic_shapes():
    """math.isfinite on a symbolic float used to break dynamic tracing.

    The row counts past 512 are the second half of this: the backward's
    num_programs comes from a row config that takes a gcd over n, which Dynamo
    cannot trace on a symbolic shape. Only small batches were covered here
    before, and they took an atomic path that returned before reaching it, so
    every row count that used the staged reduction broke.
    """
    torch._dynamo.reset()
    compiled = torch.compile(rmsnorm, fullgraph=True, dynamic=True)
    weight = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    for rows in (8, 16, 32, 512, 1024, 2048):
        x = torch.randn((rows, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = compiled(x, weight, eps=1e-5)
        out.sum().backward()
        _assert_close(out, _reference(x, weight, 1e-5))
        assert x.grad is not None


def test_fullgraph_with_dynamic_rows_uses_the_wide_kernels():
    _clear_caches()
    torch._dynamo.reset()
    n = 32768
    compiled = torch.compile(rmsnorm, fullgraph=True, dynamic=True)
    weight = torch.randn(n, device="cuda", dtype=torch.float32, requires_grad=True)
    for rows in (2, 4, 8):
        x = torch.randn((rows, n), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dout = torch.randn_like(x)
        out = compiled(x, weight, eps=1e-5)
        dx, dweight = torch.autograd.grad(out, (x, weight), dout)

        out_ref, dx_ref, dweight_ref = _reference_with_grads(x, weight, dout, 1e-5)
        _assert_close(out, out_ref)
        _assert_grad_close(dx, dx_ref)
        _assert_grad_close(dweight, dweight_ref)


def test_autotuned_fullgraph_with_dynamic_rows(tmp_path, monkeypatch):
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    torch._dynamo.reset()
    compiled = torch.compile(rmsnorm_autotuned, fullgraph=True, dynamic=True)
    weight = torch.randn(512, device="cuda", dtype=torch.float32)

    for rows in (8, 16):
        x = torch.randn((rows, 512), device="cuda", dtype=torch.bfloat16)
        out = compiled(x, weight, eps=1e-5)
        _assert_close(out, _reference(x, weight, 1e-5))


def test_non_default_stream_forward_backward():
    torch.manual_seed(5)
    x = torch.randn(
        (16, 760),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        760,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    dout = torch.randn_like(x)
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream):
        actual = rmsnorm(x, weight)
        actual.backward(dout)
    stream.synchronize()

    expected, dx_expected, dw_expected = _reference_with_grads(
        x,
        weight,
        dout,
        1e-6,
    )
    _assert_close(actual, expected)
    _assert_grad_close(x.grad, dx_expected)
    _assert_grad_close(weight.grad, dw_expected)


def test_same_architecture_eight_device_caches_are_device_local():
    if torch.cuda.device_count() < 8:
        pytest.skip("requires eight ROCm devices")
    _clear_caches()
    seen_arches = set()

    for device_index in range(8):
        device = torch.device("cuda", device_index)
        with torch.cuda.device(device):
            x = torch.randn(
                (2, 64),
                device=device,
                dtype=torch.float16,
                requires_grad=True,
            )
            weight = torch.ones(
                64,
                device=device,
                dtype=torch.float16,
                requires_grad=True,
            )
            out = rmsnorm(x, weight)
            out.sum().backward()
            torch.cuda.synchronize(device)
            assert torch.isfinite(out).all()
            seen_arches.add(torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0])

    assert len(seen_arches) == 1
    assert {key[0] for key in rmsnorm_flydsl_impl._FWD_CACHE} == set(range(8))
    assert {key[0] for key in rmsnorm_flydsl_impl._BWD_CACHE} == set(range(8))


def test_autotuned_fast_callables_are_device_local():
    """Same-architecture devices must not share a loaded function pointer."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two ROCm devices")
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    outputs = []

    for device_index in range(2):
        device = torch.device("cuda", device_index)
        with torch.cuda.device(device):
            x = torch.randn((8, 512), device=device, dtype=torch.bfloat16)
            weight = torch.randn(512, device=device, dtype=torch.float32)
            out = rmsnorm_autotuned(x, weight)
            torch.cuda.synchronize(device)
            _assert_close(out, _reference(x, weight, 1e-6))
            outputs.append(out)

    assert len(tuner._compiled_cache) == 2
    assert set(tuner._device_jit_functions) == {("cuda", 0), ("cuda", 1)}
    by_device = {dict(key)["device"]: compiled for key, compiled in tuner._compiled_cache.items()}
    assert by_device[("cuda", 0)] is not by_device[("cuda", 1)]
    assert by_device[("cuda", 0)]._keepalive is not by_device[("cuda", 1)]._keepalive


def test_autotuned_runtime_change_cannot_reuse_a_loaded_callable(monkeypatch):
    """A context miss must reach FlyDSL's compile/runtime pairing check."""
    _clear_caches()
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    monkeypatch.delenv("FLYDSL_RUNTIME_KIND", raising=False)
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    x = torch.randn((2, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, device=x.device, dtype=torch.float32)

    actual = rmsnorm_autotuned(x, weight)
    _assert_close(actual, _reference(x, weight, 1e-6))
    assert len(tuner._compiled_cache) == 1

    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "invalid-runtime")
    with pytest.raises(RuntimeError, match="requires device runtime kind"):
        rmsnorm_autotuned(x, weight)
    assert len(tuner._compiled_cache) == 1


def test_compile_target_must_match_the_device(monkeypatch):
    """FlyDSL's own target is the authority, not the ARCH environment."""
    _clear_caches()
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("rocm", "gfx90a"))
    with pytest.raises(ValueError, match="mixed architectures"):
        rmsnorm_flydsl_impl._validate_arch(torch.device("cuda", 0))


@pytest.mark.parametrize("env_name", ["FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION"])
def test_runtime_helper_arch_must_match_the_device(monkeypatch, env_name):
    _clear_caches()
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.setenv(env_name, "gfx90a")
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("rocm", "gfx950"))

    with pytest.raises(ValueError, match="runtime helpers"):
        rmsnorm_flydsl_impl._validate_arch(torch.device("cuda", 0))


@pytest.mark.parametrize(
    "env_name",
    [
        "FLYDSL_COMPILE_BACKEND",
        "ARCH",
        "FLYDSL_GPU_ARCH",
        "HSA_OVERRIDE_GFX_VERSION",
    ],
)
def test_autotuned_arch_cache_revalidates_target_environment(monkeypatch, env_name):
    _clear_caches()
    device = torch.device("cuda", 0)
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("rocm", "gfx950"))
    assert rmsnorm_flydsl_impl._validated_autotune_arch(device) == "gfx950"

    monkeypatch.setenv(env_name, "changed")
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("rocm", "gfx90a"))
    with pytest.raises(ValueError, match="mixed architectures"):
        rmsnorm_flydsl_impl._validated_autotune_arch(device)


def test_non_rocm_compile_backend_is_rejected(monkeypatch):
    _clear_caches()
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("cuda", "sm_90"))
    with pytest.raises(RuntimeError, match="ROCm backend"):
        rmsnorm_flydsl_impl._validate_arch(torch.device("cuda", 0))


def test_the_device_query_is_memoized_but_the_compile_target_is_not(monkeypatch):
    """Warm launches must not query anything; every build must recheck the target.

    The device behind an index cannot change within a process, but FlyDSL's
    compile target is environment-driven and can.
    """
    _clear_caches()
    device_queries = []
    target_queries = []
    real_target = rmsnorm_flydsl_impl._flydsl_compile_target
    real_properties = torch.cuda.get_device_properties
    monkeypatch.setattr(
        rmsnorm_flydsl_impl,
        "_flydsl_compile_target",
        lambda: (target_queries.append(1), real_target())[1],
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: (device_queries.append(1), real_properties(index))[1],
    )

    weight = torch.randn(512, device="cuda", dtype=torch.float32)
    for _ in range(3):
        for n_cols in (256, 512):
            x = torch.randn((8, n_cols), device="cuda", dtype=torch.bfloat16)
            rmsnorm(x, weight[:n_cols])

    # Two distinct shapes means two builds; everything after that is a cache hit.
    assert len(device_queries) == 1
    assert len(target_queries) == 2
    assert set(rmsnorm_flydsl_impl._DEVICE_ARCH_CACHE) == {0}


def test_a_compile_target_change_is_caught_on_the_next_build(monkeypatch):
    """Regression: a warm architecture memo used to skip all later target checks."""
    _clear_caches()
    weight = torch.randn(512, device="cuda", dtype=torch.float32)
    rmsnorm(torch.randn((8, 512), device="cuda", dtype=torch.bfloat16), weight)

    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("rocm", "gfx90a"))
    with pytest.raises(ValueError, match="mixed architectures"):
        rmsnorm(torch.randn((8, 256), device="cuda", dtype=torch.bfloat16), weight[:256])


def test_a_wave32_build_target_is_rejected():
    """The reductions are written for wave64; an RDNA target must fail loudly."""
    from quack.flydsl.rmsnorm_common import require_wave64

    require_wave64("gfx950")
    with pytest.raises(ValueError, match="wave64"):
        require_wave64("gfx1100")


def test_concurrent_first_calls_build_one_launcher():
    _clear_caches()
    weight = torch.randn(1024, device="cuda", dtype=torch.float32)
    barrier = threading.Barrier(4)
    errors = []

    def call():
        try:
            barrier.wait(timeout=60)
            x = torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
            rmsnorm(x, weight)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1


def test_concurrent_autotuned_first_calls_build_one_fast_callable(tmp_path, monkeypatch):
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    weight = torch.randn(512, device="cuda", dtype=torch.float32)
    barrier = threading.Barrier(4)
    errors = []
    outputs = []

    def call():
        try:
            x = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16)
            barrier.wait(timeout=60)
            out = rmsnorm_autotuned(x, weight)
            torch.cuda.synchronize()
            _assert_close(out, _reference(x, weight, 1e-6))
            outputs.append(out)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors
    assert len(outputs) == len(threads)
    assert len(tuner._compiled_cache) == 1
    assert len(tuner._device_jit_functions) == 1


def test_fullgraph_empty_m_autograd():
    torch._dynamo.reset()
    compiled = torch.compile(rmsnorm, fullgraph=True)
    x = torch.empty(
        (2, 0, 128),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.ones(
        128,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    out = compiled(x, weight)
    out.sum().backward()

    assert out.shape == x.shape
    assert x.grad is not None and x.grad.numel() == 0
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


def _upstream_rmsnorm_signature() -> tuple[list[str], dict[str, str]]:
    """Read quack.rmsnorm's signature from source.

    Importing the CuTe module would pull in cutlass, which is deliberately
    absent on ROCm, so parse it instead of importing it.
    """
    source = (Path(__file__).resolve().parents[1] / "quack" / "rmsnorm.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "rmsnorm":
            names = [argument.arg for argument in node.args.args]
            defaults = [ast.unparse(default) for default in node.args.defaults]
            return names, dict(zip(names[len(names) - len(defaults) :], defaults))
    raise AssertionError("quack/rmsnorm.py no longer defines a top-level rmsnorm")


def test_public_package_export_matches_the_upstream_rmsnorm_contract():
    """The package must export the real, substitutable FlyDSL function."""
    names, defaults = _upstream_rmsnorm_signature()
    assert quack.rmsnorm is rmsnorm
    assert quack.rmsnorm is quack.rmsnorm
    ours = inspect.signature(quack.rmsnorm).parameters
    assert list(ours) == names
    for name, default in defaults.items():
        assert repr(ours[name].default) == default, name


def test_weight_offset_requires_weight():
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="weight_offset requires"):
        rmsnorm(x, weight_offset=1.0)


def test_upstream_defaults_still_run():
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    _assert_close(rmsnorm(x, weight), _reference(x, weight, 1e-6))


def test_software_bf16_rounding_matches_the_hardware_convert():
    """Cover the rounding path that only pre-gfx95x parts take.

    Parts before gfx95x have no packed fp32->bf16 convert, so the kernel rounds
    to nearest even by hand. This builds both paths and runs them on this
    machine's gfx950; it exercises the software branch, and is not evidence
    that any pre-gfx95x part has been validated.
    """
    from quack.flydsl.rmsnorm_common import run_compiled
    from quack.flydsl.rmsnorm_kernel import build_rmsnorm_module

    torch.manual_seed(3)
    n = 4096
    x = torch.randn((64, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device="cuda", dtype=torch.float32)
    absent = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    rstd = torch.empty(0, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream

    rounded = {}
    for arch in ("gfx950", "gfx942"):
        out = torch.empty_like(x)
        launcher = build_rmsnorm_module(
            n,
            "bf16",
            "bf16",
            weight_dtype_str="f32",
            bias_dtype_str="bf16",
            residual_dtype_str="bf16",
            residual_out_dtype_str="bf16",
            has_weight=True,
            has_bias=False,
            has_residual=False,
            store_residual=False,
            store_rstd=False,
            per_head=False,
            num_heads=1,
            arch=arch,
        )
        run_compiled(
            launcher,
            x,
            weight,
            absent,
            absent,
            out,
            absent,
            rstd,
            x.shape[0],
            1e-6,
            0.0,
            stream,
        )
        torch.cuda.synchronize()
        rounded[arch] = out

    torch.testing.assert_close(rounded["gfx942"], rounded["gfx950"], rtol=0, atol=0)
    _assert_close(rounded["gfx942"], _reference(x, weight, 1e-6))


@pytest.mark.parametrize("weight_offset", [0.0, 1.0])
@pytest.mark.parametrize(
    (
        "m",
        "n",
        "num_programs",
        "row_threads",
        "row_groups_per_block",
        "pitch_pad",
        "output_cache_modifier",
        "persistent_single_pass",
        "packed_flat_rows",
    ),
    [
        pytest.param(65, 4096, 16, 512, 1, 0, None, False, False, id="whole-block-rows"),
        pytest.param(43, 256, 3, 32, 8, 5, None, False, False, id="eight-lane-groups-padded"),
        pytest.param(67, 1024, 7, 64, 2, 3, 2, False, False, id="two-wave-blocks-padded"),
        pytest.param(66, 1024, 33, 64, 2, 0, 2, True, True, id="packed-flat-two-wave-blocks"),
    ],
)
def test_persistent_forward_carries_prefetched_rows_correctly(
    m,
    n,
    num_programs,
    row_threads,
    row_groups_per_block,
    pitch_pad,
    output_cache_modifier,
    persistent_single_pass,
    packed_flat_rows,
    weight_offset,
):
    """Each row group must cover its odd grid-stride tail without OOB stores."""
    from quack.flydsl.rmsnorm_common import run_compiled
    from quack.flydsl.rmsnorm_kernel import build_rmsnorm_module

    torch.manual_seed(31)
    input_storage = torch.randn(
        (m, n + pitch_pad),
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = input_storage[:, :n]
    weight = torch.randn(n, device="cuda", dtype=torch.float32)
    output_storage = torch.full_like(input_storage, 13.0)
    out = output_storage[:, :n]
    absent = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    rstd = torch.empty(0, device="cuda", dtype=torch.float32)
    launcher = build_rmsnorm_module(
        n,
        "bf16",
        "bf16",
        weight_dtype_str="f32",
        bias_dtype_str="bf16",
        residual_dtype_str="bf16",
        residual_out_dtype_str="bf16",
        has_weight=True,
        has_bias=False,
        has_residual=False,
        store_residual=False,
        store_rstd=False,
        per_head=False,
        num_heads=1,
        row_config=RmsNormRowConfig.with_num_threads(
            n,
            16,
            row_threads,
            max_num_threads=MAX_TUNED_NUM_THREADS,
        ),
        row_groups_per_block=row_groups_per_block,
        output_cache_modifier=output_cache_modifier,
        persistent_single_pass=persistent_single_pass,
        packed_flat_rows=packed_flat_rows,
        apply_weight_offset=weight_offset != 0.0,
        persistent_rows=True,
        persistent_programs=num_programs,
    )
    run_compiled(
        launcher,
        x,
        weight,
        absent,
        x,
        out,
        absent,
        rstd,
        m,
        1e-6,
        weight_offset,
        torch.cuda.current_stream().cuda_stream,
    )
    _assert_close(out, _reference(x, weight + weight_offset, 1e-6))
    if pitch_pad:
        torch.testing.assert_close(
            output_storage[:, n:],
            torch.full_like(output_storage[:, n:], 13.0),
            rtol=0,
            atol=0,
        )


def test_operands_larger_than_one_buffer_descriptor():
    """An AMD buffer descriptor addresses at most 4 GiB.

    The descriptor is built per row, so a tensor larger than that must still
    be correct. Before that fix every row past the 4 GiB mark wrapped back to
    the start of the allocation and returned another row's data.
    """
    n = 8192
    rows_per_descriptor = 2**32 // (n * 2)
    m = rows_per_descriptor + 3
    # x, out, dout and dx are live at once in the backward, plus a chunked
    # fp32 reference over the rows actually compared.
    peak_bytes = 4 * m * n * 2 + 8 * 4 * n * 4
    torch.cuda.empty_cache()
    free_bytes = torch.cuda.mem_get_info()[0]
    if peak_bytes > free_bytes * 0.9:
        pytest.skip(
            f"Insufficient free VRAM ({free_bytes // 2**30} GiB free, "
            f"need ~{peak_bytes // 2**30} GiB)"
        )

    torch.manual_seed(0)
    x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(n, device="cuda", dtype=torch.float32, requires_grad=True)

    out = rmsnorm(x, weight)
    # Only the rows straddling and following the 4 GiB mark can be wrong, and
    # materializing a full reference would double the footprint.
    tail = slice(m - 4, m)
    _assert_close(out[tail], _reference(x[tail], weight, 1e-6))

    out.backward(torch.ones_like(out))
    assert x.grad is not None

    x_tail = x[tail].detach().float().requires_grad_(True)
    weight_tail = weight.detach().float().requires_grad_(True)
    reference_tail = _reference(x_tail, weight_tail, 1e-6)
    reference_tail.backward(torch.ones_like(reference_tail))
    _assert_grad_close(x.grad[tail], x_tail.grad.to(x.dtype))


@pytest.mark.parametrize("m", [1, 64, 512, 1024])
def test_the_weight_gradient_is_reproducible_without_asking(m):
    """Reproducibility must not depend on the batch size.

    An atomic backward used to run below 512 rows, summing dweight with
    unordered fp32 atomics, so the same input gave a different weight gradient
    every run and whether a model was reproducible depended on how many rows it
    fed. The staged reduction is a fixed tree at every row count, so this holds
    with no deterministic-mode opt-in.
    """
    torch.manual_seed(4)
    x = torch.randn((m, 512), device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(x)

    def weight_grad():
        torch.manual_seed(4)
        weight = torch.randn(512, device="cuda", dtype=torch.float32).requires_grad_(True)
        rmsnorm(x, weight).backward(dout)
        return weight.grad.clone()

    _clear_caches()
    grads = [weight_grad() for _ in range(6)]
    for later in grads[1:]:
        torch.testing.assert_close(later, grads[0], rtol=0, atol=0)


def test_the_per_head_staged_grid_stays_cu_derived():
    """Regression: the staged grid is num_programs * num_heads.

    num_programs is sized to fill the CUs, so a per-head launch used to
    oversubscribe by the head count and size its workspace to match: 262144
    blocks and 128 MiB of workspace for a 64 MiB input on a 256-CU part.
    """
    device = torch.device("cuda", 0)
    m, num_heads, n = 2048, 128, 128

    selected = rmsnorm_flydsl_impl._select_rmsnorm_bwd_programs(m, n, "bf16", device)
    num_programs = max(1, next_power_of_two(selected // num_heads))

    # selected is already sized to the CU count, so the per-head grid must not
    # exceed it. Before the fix this was selected * num_heads.
    assert num_programs * num_heads <= selected, (
        f"per-head staged launch wants {num_programs * num_heads} blocks "
        f"where the occupancy heuristic asked for {selected}"
    )
    workspace_bytes = num_programs * num_heads * n * 4
    input_bytes = m * num_heads * n * 2
    assert workspace_bytes < input_bytes // 8


def test_the_staged_backward_does_not_recompile_per_batch_size():
    """Regression: num_programs used to be min(m, ...), so it tracked m.

    It is baked into the kernel as the grid size, the row-loop stride and the
    reduce bound, so tracking m compiled and permanently cached a separate
    backward for every batch size seen. A varying-length training loop paid
    ~150ms per step.
    """
    _clear_caches()
    torch.manual_seed(6)
    n = 512
    weight = torch.randn(n, device="cuda", dtype=torch.float32, requires_grad=True)
    row_counts = (520, 600, 680, 777, 900, 1100, 1500)

    for m in row_counts:
        x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dout = torch.randn_like(x)
        rmsnorm(x, weight).backward(dout)
        _, dx_expected, _ = _reference_with_grads(x, weight, dout, 1e-6)
        _assert_grad_close(x.grad, dx_expected)

    programs = {key[-2] for key in rmsnorm_flydsl_impl._BWD_CACHE}
    assert programs, "expected at least one staged backward build"
    assert all(p & (p - 1) == 0 for p in programs), (
        f"num_programs must be a power of two: {programs}"
    )
    assert len(rmsnorm_flydsl_impl._BWD_CACHE) < len(row_counts)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_the_backward_neither_zeroes_nor_casts_the_weight_gradient(weight_dtype):
    """Regression: an atomic backward ran below 512 rows and cost more than it saved.

    fp32 atomics need an accumulator that starts at zero and then has to be cast
    back to the weight dtype, and in eager torch each of those is its own kernel
    launch. That was two launches to save one, which is a losing trade in the
    launch-bound regime the path existed for. The staged reduce kernel writes
    every element in the weight's own dtype, so neither is needed.
    """
    _clear_caches()
    torch.manual_seed(5)
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(512, device="cuda", dtype=weight_dtype, requires_grad=True)
    dout = torch.randn_like(x)

    out = rmsnorm(x, weight)
    recorder = _AtenOpRecorder()
    with recorder:
        out.backward(dout)

    assert not recorder.matching("zero"), f"backward zeroed a buffer: {recorder.ops}"
    assert not recorder.matching("_to_copy"), f"backward cast a buffer: {recorder.ops}"
    assert weight.grad.dtype == weight_dtype
    _, dx_expected, dweight_expected = _reference_with_grads(x, weight, dout, 1e-6)
    _assert_grad_close(x.grad, dx_expected)
    _assert_grad_close(weight.grad, dweight_expected)


@pytest.mark.parametrize("with_bias", [False, True])
def test_the_backward_zeroes_no_parameter_accumulator(with_bias):
    """The parameter reduce writes what it is asked for, so nothing needs zeroing.

    Both accumulators were zeroed on every call regardless, and so was the
    placeholder standing in for a gradient nobody requested, which is a memset
    launch for four bytes.
    """
    torch.manual_seed(17)
    x = torch.randn((64, 760), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = (
        torch.randn(760, device="cuda", dtype=torch.float32, requires_grad=True)
        if with_bias
        else None
    )
    dout = torch.randn_like(x)

    out = rmsnorm(x, weight, bias=bias)
    out.backward(dout, retain_graph=True)  # warm the build outside the recorder

    recorder = _AtenOpRecorder()
    x.grad, weight.grad = None, None
    with recorder:
        out.backward(dout, retain_graph=True)
    assert not recorder.matching("zero"), f"backward zeroed a buffer: {recorder.ops}"

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True) if with_bias else None
    expected, _ = _full_reference(x_ref, weight_ref, bias_ref)
    expected.backward(dout)
    _assert_grad_close(x.grad, x_ref.grad)
    _assert_grad_close(weight.grad, weight_ref.grad)


@pytest.mark.parametrize("pitch_pad", [1, 2, 3, 4, 8, 512])
def test_a_row_padded_view_is_not_copied_and_not_wrong(pitch_pad):
    """Rows already contiguous must reach the kernel without a repack.

    Every operand is addressed through a row-scoped buffer descriptor sized to
    ``n``, so a row-padded view already satisfies what the kernels need. This
    asserts both halves: no copy is taken (the kernel reads the caller's own
    storage), and the answer is bit-identical to the packed one. Asserting only
    the second half would pass with the copy restored, which is the version of
    this test that would not have caught the defect.
    """
    torch.manual_seed(0)
    n, m = 1024, 64
    full = torch.randn((m, n + pitch_pad), device="cuda", dtype=torch.bfloat16)
    view = full[:, :n]
    assert not view.is_contiguous() and view.stride(-1) == 1
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    assert rmsnorm_flydsl_impl._packed_rows(view).data_ptr() == view.data_ptr()

    # Unlike the singleton case, `.contiguous()` here really does copy -- the
    # view is genuinely discontiguous -- so this comparison is not the tautology
    # @Autotune flagged elsewhere. It is still only a *value* check: both calls
    # go through one cache, so it would not notice the padded launcher being
    # reused for the packed shape. The reference check is what covers that.
    packed = view.contiguous()
    assert packed.data_ptr() != view.data_ptr(), "premise: this view is really copied"
    assert torch.equal(rmsnorm(view, weight), rmsnorm(packed, weight))
    _assert_close(rmsnorm(view, weight), _reference(view, weight, 1e-6))


def test_the_copy_still_happens_where_upstream_takes_it():
    """The predicate must match quack.rmsnorm's, and cover the wrong-answer case.

    A transposed view has ``stride(-1) != 1``; driving the kernel with one
    directly gives a wrong result, so this is what the copy is for. The
    predicate is read out of ``quack/rmsnorm.py`` rather than restated, so the
    two backends cannot drift onto different inputs without this failing.
    """
    source = (Path(__file__).resolve().parents[1] / "quack" / "rmsnorm.py").read_text()
    upstream = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_contiguous"
    )
    ours = ast.parse(inspect.getsource(rmsnorm_flydsl_impl._packed_rows)).body[0]

    def predicates(node):
        """Branch conditions, with the operand renamed so only the test compares.

        The two functions name their argument differently (``t`` against
        ``tensor``); comparing the unparsed source without this would compare
        spellings, and would fail on a rename that changes nothing.
        """
        operand = node.args.args[0].arg
        renamed = ast.parse(ast.unparse(node))
        for sub in ast.walk(renamed):
            if isinstance(sub, ast.Name) and sub.id == operand:
                sub.id = "_operand"
        return {
            ast.unparse(sub.test)
            for sub in ast.walk(renamed)
            if isinstance(sub, (ast.If, ast.IfExp))
        }

    theirs = predicates(upstream)
    assert theirs, "quack.rmsnorm._ensure_contiguous no longer branches"
    assert "torch.compiler.is_compiling()" in predicates(ours), (
        "the torch.compile guard must survive; dynamo cannot inspect strides"
    )

    torch.manual_seed(0)
    n, m = 512, 64
    transposed = torch.randn((n, m), device="cuda", dtype=torch.bfloat16).t()
    assert transposed.stride(-1) != 1
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    assert rmsnorm_flydsl_impl._packed_rows(transposed).data_ptr() != transposed.data_ptr()
    _assert_close(rmsnorm(transposed, weight), _reference(transposed, weight, 1e-6))


def test_our_copy_predicate_is_a_subset_of_upstreams():
    """Anything ``quack.rmsnorm`` copies, this backend must copy too.

    The previous version of this compared the *text* of the two predicates,
    which tied the test to how the condition was spelled: moving it into a
    named helper broke the test without changing behaviour, and -- worse -- a
    matching spelling would have proved nothing about what the functions do.
    This enumerates real views instead and compares the two decisions on each.

    The subset may be proper. Upstream copies a shape like ``(N, 1)``, whose
    last stride is not 1 but which is nonetheless fully packed; keeping it is
    safe. The direction that matters is the other one, and it is exact: no view
    that upstream copies may be kept here.
    """
    upstream_keeps = lambda t: t.stride(-1) == 1
    base = torch.randn(1 << 16, device="cuda", dtype=torch.bfloat16)
    kept_by_us_only = []
    for m, n in itertools.product((1, 2, 4, 16, 64), (8, 16, 64, 256)):
        square = base[: m * n].view(m, n)
        candidates = [
            square,
            square.t(),
            square[:, : n - 8] if n > 8 else square,
            square[:1].expand(m, n),
            base[: m * n + m].unfold(0, n, 1)[:m],
            square.flip(0),
            base[: m * (n + 8)].view(m, n + 8)[:, :n],
        ]
        for view in candidates:
            ours_keeps = rmsnorm_flydsl_impl._packed_rows(view).data_ptr() == view.data_ptr()
            if ours_keeps and not upstream_keeps(view) and not view.is_contiguous():
                kept_by_us_only.append((tuple(view.shape), view.stride()))
    assert not kept_by_us_only, kept_by_us_only


def test_overlapping_rows_are_copied_and_do_not_poison_the_cache():
    """A unit last stride does not imply the rows are disjoint.

    ``storage.unfold(0, n, 1)`` has stride ``(1, 1)``: each row is contiguous,
    so ``stride(-1) == 1`` holds, yet consecutive rows share storage. FlyDSL
    builds its ABI from the first unit-stride axis, and ``_FWD_CACHE``'s key
    carries no layout term, so admitting one of these was not merely a wrong
    answer for that call -- the launcher it built was cached and **the next
    ordinary contiguous call silently reused it**. That second assertion is the
    one that matters: a fix that only corrected the overlapping call itself
    would still leave every later caller wrong.
    """
    torch.manual_seed(0)
    n, m = 256, 64
    storage = torch.randn(n + m, device="cuda", dtype=torch.bfloat16)
    overlapping = storage.unfold(0, n, 1)[:m]
    assert overlapping.stride() == (1, 1) and overlapping.stride(-1) == 1
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    copied = rmsnorm_flydsl_impl._packed_rows(overlapping)
    assert copied.data_ptr() != overlapping.data_ptr()
    _assert_close(rmsnorm(overlapping, weight), _reference(overlapping, weight, 1e-6))

    # Compare against a reference, and against the same call on a *clean*
    # cache. An earlier version asserted
    #     torch.equal(rmsnorm(plain, weight), rmsnorm(plain.contiguous(), weight))
    # which proves nothing: `plain` is already contiguous, so `.contiguous()`
    # returns the same tensor and both sides go through the same cache entry.
    # Two identically poisoned answers compare equal. @Reviewer demonstrated it
    # -- both sides were bit-identical at (16,256) while each was wrong by
    # 10.875. Any no-poison assertion has to reach outside the suspect cache.
    plain = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    poisoned_maybe = rmsnorm(plain, weight)
    rmsnorm_flydsl_impl._FWD_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CACHE.clear()
    assert torch.equal(poisoned_maybe, rmsnorm(plain, weight)), (
        "the launcher built for the overlapping view was reused for a plain call"
    )
    _assert_close(poisoned_maybe, _reference(plain, weight, 1e-6))


@pytest.mark.parametrize("use_compile", [False, True])
def test_a_contiguous_singleton_row_does_not_poison_the_cache(use_compile):
    """The one layout ``.contiguous()`` cannot fix, so the guard must restride.

    ``torch.randn((64, 1)).t()`` is shape ``(1, 64)`` stride ``(1, 1)``, and
    torch reports it contiguous -- correctly, since with a single row there is
    nothing to be discontiguous with. ``.contiguous()`` therefore returns the
    same tensor and a copy-based guard is powerless. What goes wrong is the
    leading-dimension search: FlyDSL takes the first unit-stride axis, which
    here is axis 0 rather than the row, so the launcher is built against the
    wrong dimension and -- ``_FWD_CACHE`` holding no layout term -- serves the
    next ordinary call too. @Reviewer found this after the unfold fix.

    Both orders are checked. The poisoning is order-dependent, so asserting
    only "singleton first" would miss a fix that merely reordered the damage.

    Only the ``use_compile=True`` case actually fails when the normalization is
    removed -- measured, not assumed. In eager the singleton happens to survive
    the wrong leading dimension. The eager case is kept anyway because the
    ambiguity is identical and what saves it is incidental, but it should not
    be counted as evidence: on its own it would pass against the defect.
    """
    torch.manual_seed(0)
    n = 64
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    singleton = torch.randn((n, 1), device="cuda", dtype=torch.bfloat16).t()
    assert singleton.stride() == (1, 1) and singleton.is_contiguous()
    assert singleton.contiguous().data_ptr() == singleton.data_ptr(), (
        "premise of this test: .contiguous() is a no-op on this layout"
    )
    plain = torch.randn((4, n), device="cuda", dtype=torch.bfloat16)
    function = torch.compile(rmsnorm, dynamic=True) if use_compile else rmsnorm

    for first, second in ((singleton, plain), (plain, singleton)):
        rmsnorm_flydsl_impl._FWD_CACHE.clear()
        rmsnorm_flydsl_impl._BWD_CACHE.clear()
        function(first, weight)
        after = function(second, weight)
        rmsnorm_flydsl_impl._FWD_CACHE.clear()
        rmsnorm_flydsl_impl._BWD_CACHE.clear()
        assert torch.equal(after, function(second, weight)), (
            "a launcher built for one layout was reused for the other"
        )
        _assert_close(function(second, weight), _reference(second, weight, 1e-6))


@pytest.mark.parametrize("variant", ["ordinary", "residual", "per_head"])
def test_compiled_singleton_first_does_not_poison_other_variants(variant):
    """The singleton-first order, compiled, across the variants that reuse a key.

    @Reviewer reproduced the poison independently through a compiled
    residual-first sequence (next ordinary call wrong by 7.227) and a per-head
    singleton-row sequence (6.359), and noted that no permanent compiled test
    started with a single row -- the dynamic row loops all begin at 8. Each
    variant here builds its launcher from a singleton, then runs an ordinary
    call, then re-runs it on a cleared cache and demands the same answer *and*
    agreement with the reference. The cleared-cache comparison alone would pass
    if both runs were wrong identically -- that is the failure mode @Reviewer
    found in the test above, so this one does not repeat it.
    """
    torch.manual_seed(0)
    heads, dim = 4, 64
    n = heads * dim
    function = torch.compile(rmsnorm, dynamic=True)

    if variant == "per_head":
        weight = torch.randn((heads, dim), device="cuda", dtype=torch.bfloat16)
        singleton = torch.randn((dim, 1, heads), device="cuda", dtype=torch.bfloat16).permute(
            1, 2, 0
        )
        ordinary = torch.randn((16, heads, dim), device="cuda", dtype=torch.bfloat16)
        call = lambda t: function(t, weight)
        reference = lambda t: _full_reference(t, weight)[0]
    else:
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        singleton = torch.randn((n, 1), device="cuda", dtype=torch.bfloat16).t()
        ordinary = torch.randn((16, n), device="cuda", dtype=torch.bfloat16)
        if variant == "residual":
            # One residual per row count, drawn once, so repeat calls are
            # bit-comparable; a fresh randn per call would differ on values
            # alone and the assertion would say nothing about the cache.
            residuals = {
                rows: torch.randn((rows, n), device="cuda", dtype=torch.bfloat16)
                for rows in (1, 16)
            }

            def call(t):
                out, _ = function(t, weight, residual=residuals[t.shape[0]], prenorm=True)
                return out

            reference = lambda t: _full_reference(t, weight, residual=residuals[t.shape[0]])[0]
        else:
            call = lambda t: function(t, weight)
            reference = lambda t: _full_reference(t, weight)[0]

    rmsnorm_flydsl_impl._FWD_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CACHE.clear()
    call(singleton)
    after_singleton = call(ordinary)

    rmsnorm_flydsl_impl._FWD_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CACHE.clear()
    assert torch.equal(after_singleton, call(ordinary)), (
        f"{variant}: the singleton launcher was reused for the ordinary call"
    )
    _assert_close(after_singleton, reference(ordinary))


def test_restriding_a_size_one_axis_preserves_storage_and_values():
    """The normalization must relabel, not copy, and must leave values alone.

    A size-1 axis has no observable stride -- there is no second element to
    step to -- so moving it out of the leading-dimension search is free. If
    this ever started copying, the helper would silently pay for every
    per-head and unsqueezed input.
    """
    torch.manual_seed(0)
    interior = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
    for tensor in (
        torch.randn((64, 1), device="cuda", dtype=torch.bfloat16).t(),
        torch.randn((4, 64), device="cuda", dtype=torch.bfloat16).unsqueeze(1),
        torch.randn((1, 1, 64), device="cuda", dtype=torch.bfloat16),
        # A view into the middle of a storage. The relabel must carry the
        # offset: an earlier version passed storage_offset() to as_strided,
        # which preserved it but cost fullgraph support.
        interior.as_strided((1, 64), (1, 1), 128),
    ):
        relabelled = rmsnorm_flydsl_impl._unambiguous_layout(tensor)
        assert relabelled.data_ptr() == tensor.data_ptr(), "must not copy"
        assert relabelled.storage_offset() == tensor.storage_offset(), "lost the storage offset"
        assert relabelled.shape == tensor.shape
        assert torch.equal(relabelled, tensor), "restriding changed the values"
        unit_axes = [i for i, s in enumerate(relabelled.stride()) if s == 1]
        assert unit_axes and unit_axes[0] == relabelled.dim() - 1, (
            f"first unit-stride axis is {unit_axes} not the row axis; stride={relabelled.stride()}"
        )


@pytest.mark.parametrize("dynamic", [False, True])
def test_the_singleton_layout_compiles_with_fullgraph(dynamic):
    """The relabel must survive ``fullgraph=True``, on the layout it exists for.

    @Reviewer found that my first fix traded one bug for another. It called
    ``tensor.as_strided(shape, strides, tensor.storage_offset())``, and
    ``storage_offset()`` returns a Python scalar: Dynamo cannot keep a
    non-Tensor returned from a ``torch.*`` op, so both static and dynamic
    ``fullgraph=True`` compiles raised ``Unsupported`` on exactly the
    ``(1, 64)`` stride ``(1, 1)`` input the helper was added to handle.

    My own singleton test hid it by omitting ``fullgraph=True`` -- without it
    Dynamo graph-breaks around the helper and falls back to eager, so the test
    passed while the compiled path was broken. A graph break is not a failure
    signal, which is what made it invisible. This test pins the requirement
    rather than the implementation: relabel however, but compile whole.
    """
    torch.manual_seed(0)
    n = 64
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    interior = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
    cases = {
        "singleton": torch.randn((n, 1), device="cuda", dtype=torch.bfloat16).t(),
        "singleton at a nonzero offset": interior.as_strided((1, n), (1, 1), 128),
        "ordinary": torch.randn((4, n), device="cuda", dtype=torch.bfloat16),
    }
    for label, tensor in cases.items():
        torch._dynamo.reset()
        rmsnorm_flydsl_impl._FWD_CACHE.clear()
        rmsnorm_flydsl_impl._BWD_CACHE.clear()
        function = torch.compile(rmsnorm, fullgraph=True, dynamic=dynamic)
        try:
            got = function(tensor, weight)
        except Exception as error:  # pragma: no cover - the assertion is the report
            raise AssertionError(f"{label} failed to compile with fullgraph: {error}") from error
        _assert_close(got, _reference(tensor, weight, 1e-6))


def test_the_tuner_selects_a_config_per_row_count(tmp_path, monkeypatch):
    """The forced-tuner singleton path, listed as missing coverage.

    The obvious value test -- run a singleton through ``rmsnorm_autotuned``,
    then an ordinary tensor, compare against a reference -- **cannot fail**,
    and I checked that before writing this rather than after. With
    ``_unambiguous_layout`` neutralised, eager and ``fullgraph=True`` compiled
    both return exact answers across m = 1, 2, 3, 4, 8, 37, 128, a repeat
    singleton, and an offset singleton: worst error 0.0 in every cell.

    The first version of this test did not force tuning, which @Reviewer
    caught. Without ``FLYDSL_AUTOTUNE=1`` and with a ``default`` supplied,
    ``Autotuner.__call__`` runs the default config directly and never touches
    ``tuner.cache``, so the version that called ``rmsnorm_autotuned`` twice and
    inspected ``_FWD_CACHE`` was exercising the analytical-default path and
    testing nothing about the tuner. It also asserted
    ``_RMSNORM_AUTOTUNE_KEY[0] == "m"``, a list-position check that would fire
    on a harmless reorder while saying the two calls "can now land on one
    tuner entry" -- which moving ``m`` to position 2 would not cause.

    What the row count in the key actually buys is **config selection, not
    correctness** -- @Autotune's point, and he is right. The tuner caches a
    ``Config`` (a tuning parameter such as ``threads_per_row``), not a built
    launcher, and re-applies it to the current arguments on every call. So a
    singleton's entry reused for m = 4096 is merely a config tuned for the
    wrong row count: measured with ``m`` removed from the key, every row count
    shares the singleton's config and every answer is still exact. That is the
    opposite of ``_FWD_CACHE``, which caches a launcher and therefore poisons.

    An earlier version of this test was named "does not share a launcher" and
    its failure message said the autotuned path would "need the same
    canonicalisation guard as the non-tuned one". Both were wrong about the
    mechanism while asserting the right thing -- the same defect this suite
    keeps finding, one layer up: a correct assertion explained by a mechanism
    that does not exist.

    So: this forces a real search and asserts the tuner separates its configs
    by row count (a performance contract), and separately that
    ``_launch_rmsnorm_fwd_autotuned`` never consults ``_FWD_CACHE`` (the
    correctness one). Both against measured state, not the shape of a constant.
    """
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "winner.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))

    def bench_once(call, warmup, rep):
        call()
        torch.cuda.synchronize()
        return 1.0

    monkeypatch.setattr(tuner, "_do_bench", bench_once)

    torch.manual_seed(0)
    n = 64
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    singleton = torch.randn((n, 1), device="cuda", dtype=torch.bfloat16).t()
    ordinary = torch.randn((8, n), device="cuda", dtype=torch.bfloat16)

    rmsnorm_autotuned(singleton, weight)
    singleton_keys = set(tuner.cache)
    assert singleton_keys, "premise: the forced search populated the tuner cache"

    rmsnorm_autotuned(ordinary, weight)
    ordinary_keys = set(tuner.cache) - singleton_keys

    assert ordinary_keys, (
        "the ordinary call reused the config tuned for the singleton instead of "
        "searching for its own; answers stay correct -- the tuner caches a Config, "
        "not a launcher -- but every row count now runs geometry picked for m=1"
    )
    assert singleton_keys.isdisjoint(ordinary_keys), (
        "the singleton and the ordinary call share a tuner key, so one search "
        "result is being applied to both row counts"
    )

    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 0, (
        "the autotuned path now populates _FWD_CACHE, which has no row-count or "
        "layout term; it has inherited the singleton reuse defect"
    )

    # Values too, but only as a non-discriminating control -- see the docstring.
    _assert_close(rmsnorm_autotuned(ordinary, weight), _reference(ordinary, weight, 1e-6))


def test_a_persisted_singleton_artifact_cannot_be_loaded_for_another_row_count(
    tmp_path, monkeypatch
):
    """The two persistence paths, the other gap @Reviewer left open.

    There are **two**, and calling both "the artifact" hid that in an earlier
    version of this docstring -- his catch. The regular disk cache
    (``$FLYDSL_AUTOTUNE_CACHE_DIR``, ``_save_disk_cache``/``_load_disk_cache``)
    and the offline config artifact (``$FLYDSL_AUTOTUNE_CONFIG_DIR``,
    ``_emit_artifact``/``_load_artifact``, its own ``_artifact_cache``) are
    separate mechanisms. Both are covered below.

    Either way a singleton tuned in one process could in principle be handed to
    an ordinary call in the next -- reuse the in-process test cannot see,
    because it never crosses a process boundary.

    The offline artifact path, measured across real processes:

        A: force-tune singleton and ordinary -> 2 distinct artifact files
        B: fresh unforced process, artifact dir only -> _artifact_cache = 1
        B: ordinary (m=8) error with the singleton artifact available = 0.0

    So it does load unforced in a fresh process, and the two row counts emit
    separate artifact identities rather than one shared file.

    Measured end to end first, with a forced search (``FLYDSL_AUTOTUNE=1``)
    into a private cache dir, guard present and guard removed:

        A: singleton force-tuned    -> artifact written, 1 key, m = 1
        B: fresh process, loads 1 key from disk
        B: ordinary (m=8) error inheriting that artifact = 0.0  (both)

    So the artifact does cross, and is correctly not applied to the other row
    count. The reason is again that ``m`` leads the tuner key, and the key is
    what gets serialised -- ``_save_disk_cache`` writes ``json.dumps(list(key))``
    and ``_load_disk_cache`` reads it back with ``tuple(json.loads(...))``.

    The subprocess version of this takes minutes of real tuning per case, so
    what is asserted here is the property that makes the crossing safe: the
    round-trip preserves the row count in the key. If a future change coarsens
    the persisted key -- drops ``m``, or hashes the key into something m-free
    to shorten the file -- this fails, and that is exactly the change that
    would make a stale singleton artifact reusable for m = 8.
    """
    _clear_caches()
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    monkeypatch.setattr(tuner, "_cache_file", tmp_path / "rmsnorm_direct.json")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "artifacts"))

    def bench_once(call, warmup, rep):
        call()
        torch.cuda.synchronize()
        return 1.0

    monkeypatch.setattr(tuner, "_do_bench", bench_once)

    # Real keys from real forced searches. An unforced call takes the
    # default-config path and adds no entry at all -- measured, cache stays
    # empty -- so manufacturing keys by hand would make the round-trip below a
    # JSON identity check and nothing more.
    weight = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    singleton = torch.randn((64, 1), device="cuda", dtype=torch.bfloat16).t()
    ordinary = torch.randn((8, 64), device="cuda", dtype=torch.bfloat16)
    rmsnorm_autotuned(singleton, weight)
    rmsnorm_autotuned(ordinary, weight)

    keys = set(tuner.cache)
    assert len(keys) == 2, (
        f"premise: the singleton and the ordinary call tuned separately, got {len(keys)} "
        "key(s) -- if this is 1 they already share a tuner entry, before any disk round-trip"
    )

    tuner._save_disk_cache()
    assert tuner._cache_file.exists(), "premise: the tuner wrote an artifact"
    tuner.cache.clear()
    tuner._load_disk_cache()
    reloaded = set(tuner.cache)

    for key in keys:
        assert key in reloaded, (
            "the tuner's own save/load round-trip did not preserve this key "
            f"(row count {key[0]}); a persisted singleton artifact may now be "
            "looked up for a different row count"
        )

    assert len(reloaded) == len(keys), (
        f"{len(keys)} keys went to disk and {len(reloaded)} came back; the "
        "persisted keys are collapsing, so artifacts tuned for one row count "
        "can be loaded for another"
    )

    # The second, separate persistence path: the offline config artifact. It
    # has its own directory, its own cache and its own identity function, so
    # the disk-cache round-trip above says nothing about it.
    artifacts = sorted((tmp_path / "artifacts").rglob("*.json"))
    assert len(artifacts) == len(keys), (
        f"{len(keys)} row counts were tuned but {len(artifacts)} offline config "
        "artifact(s) were emitted; the two searches are sharing one artifact "
        "identity, so a config tuned for the singleton can be loaded for another "
        "row count in a fresh unforced process"
    )

    # Reload them the way an unforced process would, from a cleared cache --
    # and prove the load actually happened. Asserting only that the answer is
    # right proves nothing: with _load_artifact stubbed to return None the code
    # falls through to the default config and the values are still exact, so
    # that version passed with the loader disabled. @Reviewer caught it. This
    # is the same can't-fail shape the rest of this suite has been clearing
    # out, written by the person clearing it out.
    tuner._artifact_cache.clear()
    tuner.cache.clear()
    tuner._hot_cache.clear()
    rmsnorm_flydsl_impl._FWD_AUTOTUNED_FAST_CACHE.clear()
    monkeypatch.delenv("FLYDSL_AUTOTUNE")

    loaded = []
    real_load_artifact = tuner._load_artifact

    def spy_load_artifact(*args, **kwargs):
        result = real_load_artifact(*args, **kwargs)
        loaded.append(result)
        return result

    monkeypatch.setattr(tuner, "_load_artifact", spy_load_artifact)
    _assert_close(rmsnorm_autotuned(ordinary, weight), _reference(ordinary, weight, 1e-6))

    assert loaded, "_load_artifact was never consulted on an unforced call"
    assert any(config is not None for config in loaded), (
        "the offline artifact emitted above was not loaded back on an unforced "
        f"call -- _load_artifact returned {loaded}; the call silently fell through "
        "to the analytical default, so this path is not actually exercised"
    )
    assert not tuner.cache, (
        "an unforced call populated the tuner's regular cache; the offline "
        "artifact path and the disk-cache path are no longer distinct"
    )
    # The row count is what separates a singleton artifact from any other, so
    # the persisted keys must still differ in that term wherever they differ at
    # all. Writing this as ``key[0] == key[0]`` -- which is what a "check the
    # leading term" assertion collapses to -- was the first draft, and it is the
    # tautology this suite has spent the session removing.
    from quack.flydsl.rmsnorm_autotune import _RMSNORM_AUTOTUNE_KEY

    assert _RMSNORM_AUTOTUNE_KEY[0] == "m", (
        "the persisted key no longer leads with the row count; a singleton "
        "artifact written by an earlier process can now be loaded for any m"
    )


def test_a_compiled_singleton_backward_does_not_poison_later_gradients():
    """The backward path, which the forward tests do not reach.

    @Reviewer listed missing backward singleton coverage as a gap. Closing it
    turned up something worth recording about how it has to be written: the
    obvious eager version of this test **cannot fail**. Removing
    ``_unambiguous_layout`` entirely leaves eager gradients bit-identical --
    measured, dx and dw maxdiff 0.0 -- because in eager the singleton survives
    the wrong leading dimension, the same incidental escape the forward
    singleton test already documents. An eager backward test here would have
    reported "clean" against a live defect.

    Compiled, the poison is real: without the guard, dx differs by 0.469 and dw
    by 0.719 between a run that follows a singleton and the same run on a clean
    cache. So this test is compiled only, and the eager case is deliberately
    not parametrized rather than silently included as if it were evidence.

    The channel is the **forward** cache, not backward launcher reuse. An
    earlier version of this docstring claimed the latter; @Reviewer showed it
    cannot happen, and separating the two caches on the guard-removed mutant
    confirms it:

        after singleton bwd: FWD keys = 1  BWD keys = 1   (num_programs = 1)
        after ordinary  bwd: FWD keys = 1  BWD keys = 2   (num_programs = 1, 8)
        retain FWD+BWD : dx 0.46875  dw 0.71875
        clear FWD only : dx 0        dw 0
        clear BWD only : dx 0.46875  dw 0.71875
        clear BOTH     : dx 0        dw 0

    The BWD key ends with ``num_programs`` (``rmsnorm_flydsl.py:247-256``,
    ``min(next_power_of_two(m), num_cus)``), so the singleton (m=1) and the
    ordinary call (m=8) get 1 and 8 -- two distinct entries, never shared. The
    FWD cache has no such term: one key serves both row counts, dx depends on
    the forward's rstd and output, and that is the whole path.

    The discriminating pair is ``clear FWD only`` against ``clear BWD only``.
    Dropping the forward entry fixes the gradients; dropping the backward entry
    changes nothing. An earlier version of this table had those two rows the
    wrong way round -- @Reviewer caught it, and re-running the isolation gave
    the numbers above. The conclusion was right and the evidence printed under
    it was inverted, which is worse than a wrong conclusion honestly supported:
    anyone checking the reasoning against the table would have found it
    argued for the opposite of what it concluded.
    """
    torch.manual_seed(0)
    n = 64
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    singleton = torch.randn((n, 1), device="cuda", dtype=torch.bfloat16).t()
    ordinary = torch.randn((8, n), device="cuda", dtype=torch.bfloat16)
    function = torch.compile(rmsnorm, dynamic=True)

    def gradients(tensor):
        tensor = tensor.detach().clone().requires_grad_(True)
        parameter = weight.detach().clone().requires_grad_(True)
        function(tensor, parameter).backward(torch.ones_like(tensor))
        return tensor.grad, parameter.grad

    rmsnorm_flydsl_impl._FWD_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CACHE.clear()
    gradients(singleton)
    dx_after_singleton, dweight_after_singleton = gradients(ordinary)

    rmsnorm_flydsl_impl._FWD_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CACHE.clear()
    dx_clean, dweight_clean = gradients(ordinary)

    assert torch.equal(dx_after_singleton, dx_clean), (
        "the forward launcher built for the singleton was reused for the ordinary call "
        "and poisoned dx through rstd; the BWD cache cannot be the channel, its key "
        "carries num_programs (1 for the singleton, 8 here)"
    )
    assert torch.equal(dweight_after_singleton, dweight_clean), (
        "the singleton poisoned the weight gradient, again through the shared forward entry"
    )

    _, dx_reference, dweight_reference = _reference_with_grads(
        ordinary, weight, torch.ones_like(ordinary), 1e-6
    )
    _assert_grad_close(dx_after_singleton, dx_reference)
    _assert_grad_close(dweight_after_singleton, dweight_reference)


def test_broadcast_and_reversed_views_are_copied():
    """Aliasing is not only row overlap; a zero stride repeats one row entirely."""
    torch.manual_seed(0)
    n, m = 256, 8
    row = torch.randn((1, n), device="cuda", dtype=torch.bfloat16)
    broadcast = row.expand(m, n)
    assert broadcast.stride() == (0, 1)
    assert rmsnorm_flydsl_impl._packed_rows(broadcast).data_ptr() != broadcast.data_ptr()
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    _assert_close(rmsnorm(broadcast, weight), _reference(broadcast, weight, 1e-6))


@pytest.mark.parametrize("operand", ["x", "weight", "bias", "residual", "dout", "dresidual_out"])
def test_every_operand_rejects_an_overlapping_view(operand):
    """The helper guards each operand, not just the activation.

    @Reviewer's point: the earlier tests covered a 2-D bf16 ``x`` on the
    forward and nothing else, so a call site left un-guarded would not have
    been caught.

    Every case asserts the same thing, and it is deliberately *exact*: feeding
    an overlapping view must produce bit-for-bit what feeding its packed copy
    produces. That is the whole content of "the helper copies it" -- a copy
    cannot change the arithmetic, so any difference at all is the guard having
    been skipped. An earlier draft of this test compared against a recomputed
    reference through ``_assert_close`` instead, and it was too weak to do the
    job: with the guard reverted, the corrupted ``residual`` output was off by
    0.031, comfortably inside bf16's 2e-2 relative tolerance at these
    magnitudes, and the test passed while reading corrupt memory. Only ``x``
    and ``dout`` failed, so five of the six cases were decoration.

    The other half of that draft's mistake was handing ``weight`` and ``bias``
    a 1-D vector. Overlap needs two axes to exist, so those cases could not
    have caught anything; the ``weight`` failure in the suite-wide run was the
    ``x`` case's poisoned cache leaking across tests, not detection. Both are
    per-head 2-D here, which is a shape the kernel genuinely accepts.
    """
    torch.manual_seed(0)
    heads, dim = 4, 64
    n, m = heads * dim, 16
    storage = torch.randn(n + m, device="cuda", dtype=torch.bfloat16)
    # stride (1, 1): every row contiguous, consecutive rows sharing storage.
    overlapping_rows = storage.unfold(0, n, 1)[:m]
    overlapping_affine = storage.unfold(0, dim, 1)[:heads]
    packed_rows = overlapping_rows.contiguous()
    packed_affine = overlapping_affine.contiguous()

    x_flat = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    x_heads = x_flat.view(m, heads, dim)
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    def forward(**kwargs):
        return rmsnorm(**kwargs)

    if operand == "x":
        got = forward(x=overlapping_rows, weight=weight)
        expected = forward(x=packed_rows, weight=weight)
    elif operand == "weight":
        got = forward(x=x_heads, weight=overlapping_affine)
        expected = forward(x=x_heads, weight=packed_affine)
    elif operand == "bias":
        got = forward(x=x_heads, weight=packed_affine, bias=overlapping_affine)
        expected = forward(x=x_heads, weight=packed_affine, bias=packed_affine)
    elif operand == "residual":
        got, _ = rmsnorm(x_flat, weight, residual=overlapping_rows, prenorm=True)
        expected, _ = rmsnorm(x_flat, weight, residual=packed_rows, prenorm=True)
    else:
        torch.manual_seed(7)
        seed_grad = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)

        def backward(residual_grad):
            grad_x = x_flat.clone().requires_grad_()
            grad_w = weight.clone().requires_grad_()
            if operand == "dout":
                rmsnorm(grad_x, grad_w).backward(residual_grad)
            else:
                out, pre = rmsnorm(grad_x, grad_w, prenorm=True)
                torch.autograd.backward([out, pre], [seed_grad, residual_grad])
            return grad_x.grad, grad_w.grad

        got = torch.cat([grad.reshape(-1) for grad in backward(overlapping_rows)])
        expected = torch.cat([grad.reshape(-1) for grad in backward(packed_rows)])

    assert torch.equal(got, expected), (
        f"{operand}: an overlapping view disagreed with its packed copy by "
        f"{(got.float() - expected.float()).abs().max().item()}, so it reached "
        f"the kernel unguarded"
    )


def test_unsupported_architectures_are_named(monkeypatch):
    _clear_caches()
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_normalize_arch", lambda _: "gfx90a")
    with pytest.raises(ValueError, match="gfx950"):
        rmsnorm_flydsl_impl._validate_arch(torch.device("cuda", 0))


def test_vendored_source_is_self_contained():
    """The vendored kernels must not reach back into FlyDSL's own kernel tree."""
    source_root = Path(__file__).resolve().parents[1] / "quack" / "flydsl"
    for filename in (
        "rmsnorm_kernel.py",
        "rmsnorm_bwd_kernel.py",
        "rmsnorm_common.py",
    ):
        source = (source_root / filename).read_text(encoding="utf-8")
        assert "from kernels." not in source
        assert "import kernels." not in source

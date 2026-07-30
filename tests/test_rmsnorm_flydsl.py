# Copyright (c) 2026, Tri Dao.

import ast
import inspect
import math
import threading
from pathlib import Path

import pytest
import torch


if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl")

import quack.rmsnorm_flydsl as rmsnorm_flydsl_impl  # noqa: E402
from quack.flydsl.rmsnorm_config import next_power_of_two  # noqa: E402
from quack.rmsnorm_flydsl import rmsnorm, rmsnorm_autotuned  # noqa: E402


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
        ((4, 4096), torch.float16, torch.float16, 1e-6),
        ((2, 4096), torch.bfloat16, torch.float32, 1e-5),
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
        (torch.empty(2, 0), torch.empty(0), ValueError, "between 1 and 8192"),
        (torch.ones(1, 8193), torch.ones(8193), ValueError, "between 1 and 8192"),
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
        ((5, 760), torch.float32, torch.float32),
        ((512, 4096), torch.bfloat16, torch.float32),
        ((512, 3584), torch.float16, torch.float16),
        # 128-bit FP32 column I/O.
        ((512, 2048), torch.float32, torch.float32),
        # Column count that is not a whole number of blocks.
        ((512, 3000), torch.bfloat16, torch.float32),
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


def test_selective_gradients_are_respected():
    x = torch.randn((4, 760), device="cuda", dtype=torch.float16)
    weight = torch.randn(760, device="cuda", dtype=torch.float32)
    bias = torch.randn(
        760,
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
def test_backward_with_no_parameter_grads(per_head):
    """Frozen parameters used to be the atomic kernel's other job.

    Nothing reduces, so the persistent kernel covers the rows and the parameter
    reduce is not launched at all.
    """
    torch.manual_seed(11)
    n = 760
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


def test_deterministic_backward_is_reproducible():
    torch.manual_seed(16)
    x = torch.randn((64, 760), device="cuda", dtype=torch.float16)
    weight = torch.randn(760, device="cuda", dtype=torch.float32)
    bias = torch.randn(760, device="cuda", dtype=torch.float32)
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
    rmsnorm_flydsl_impl._BWD_CU_COUNT_CACHE.clear()
    rmsnorm_flydsl_impl._DEVICE_ARCH_CACHE.clear()
    rmsnorm_flydsl_impl._rmsnorm_fwd_tuner.cache.clear()
    rmsnorm_flydsl_impl._rmsnorm_fwd_tuner._artifact_cache.clear()


def test_custom_ops_are_unique_mutation_only_and_fake_safe():
    fwd = torch.ops.quack._rmsnorm_flydsl_fwd.default
    bwd = torch.ops.quack._rmsnorm_flydsl_bwd.default
    for op in (fwd, bwd):
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
        fwd(
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
        bwd(
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
            False,
            1,
        )


def test_eager_fast_path_bypasses_custom_op_dispatch():
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


def test_compile_target_must_match_the_device(monkeypatch):
    """FlyDSL's own target is the authority, not the ARCH environment."""
    _clear_caches()
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("rocm", "gfx90a"))
    with pytest.raises(ValueError, match="mixed architectures"):
        rmsnorm_flydsl_impl._validate_arch(torch.device("cuda", 0))


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


def test_public_signature_matches_upstream_rmsnorm():
    """The backend must be substitutable for quack.rmsnorm, not a lookalike."""
    names, defaults = _upstream_rmsnorm_signature()
    ours = inspect.signature(rmsnorm).parameters
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

    programs = {key[-1] for key in rmsnorm_flydsl_impl._BWD_CACHE}
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

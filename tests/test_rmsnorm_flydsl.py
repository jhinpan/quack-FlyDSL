# Copyright (c) 2026, Tri Dao.

import math
from pathlib import Path

import pytest
import torch


if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl")

from quack._flydsl import FLYDSL_UPSTREAM_SHA  # noqa: E402
from quack.rmsnorm_flydsl import rmsnorm  # noqa: E402


UPSTREAM_SHA = "ddaa507f56aa3fe9c08ebe6161a717b755540248"


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_f32 * rstd * weight.float()).to(x.dtype)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("shape", "dtype", "weight_dtype", "eps"),
    [
        ((3, 127), torch.float16, torch.float16, 1e-6),
        ((2, 1024), torch.bfloat16, torch.float32, 1e-5),
        ((4, 4096), torch.float16, torch.float16, 1e-6),
        ((2, 4096), torch.bfloat16, torch.float32, 1e-5),
        ((3, 3001), torch.float16, torch.float32, 1e-6),
        ((2, 3001), torch.bfloat16, torch.bfloat16, 1e-5),
        ((2, 4096), torch.float32, torch.float32, 1e-6),
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
    n = 3001
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
        (torch.ones(8), None, TypeError, "weight must be a torch.Tensor"),
        (torch.tensor(1.0), torch.ones(1), ValueError, "at least one dimension"),
        (torch.ones(2, 8), torch.ones(2, 8), ValueError, "weight must be 1-D"),
        (torch.ones(2, 8), torch.ones(7), ValueError, "last dimension"),
        (torch.empty(2, 0), torch.empty(0), ValueError, "between 1 and 8192"),
        (torch.ones(1, 8193), torch.ones(8193), ValueError, "between 1 and 8192"),
        (
            torch.ones(2, 8, dtype=torch.float64),
            torch.ones(8, dtype=torch.float64),
            TypeError,
            "x dtype",
        ),
        (
            torch.ones(2, 8, dtype=torch.float32),
            torch.ones(8, dtype=torch.float16),
            TypeError,
            "weight dtype",
        ),
        (
            torch.ones(2, 8, dtype=torch.float16),
            torch.ones(8, dtype=torch.bfloat16),
            TypeError,
            "weight dtype",
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


def test_public_contract_rejects_mixed_devices():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two ROCm devices")
    x = torch.ones(2, 8, device="cuda:0", dtype=torch.float16)
    weight = torch.ones(8, device="cuda:1", dtype=torch.float16)
    with pytest.raises(ValueError, match="same device"):
        rmsnorm(x, weight)


def test_vendored_source_is_pinned_and_isolated():
    assert FLYDSL_UPSTREAM_SHA == UPSTREAM_SHA
    source_root = Path(__file__).resolve().parents[1] / "quack" / "_flydsl"
    for filename in ("rmsnorm_kernel.py", "rmsnorm_common.py", "kernel_utils.py"):
        source = (source_root / filename).read_text(encoding="utf-8")
        assert "from kernels." not in source
        assert "import kernels." not in source
        assert "autotune" not in source.lower()
        assert "quant" not in source.lower()
        assert "fused_add" not in source

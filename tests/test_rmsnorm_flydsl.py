# Copyright (c) 2026, Tri Dao.

import math
from pathlib import Path

import pytest
import torch


if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

pytest.importorskip("flydsl")

from quack._flydsl import FLYDSL_UPSTREAM_SHA  # noqa: E402
import quack.rmsnorm_flydsl as rmsnorm_flydsl_impl  # noqa: E402
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


@pytest.mark.parametrize(
    ("expected_path", "shape", "dtype", "weight_dtype"),
    [
        ("atomic", (17, 513), torch.float16, torch.float16),
        ("atomic", (5, 257), torch.float32, torch.float32),
        ("two_stage", (512, 4096), torch.bfloat16, torch.float32),
        ("two_stage", (512, 3001), torch.float16, torch.float16),
    ],
)
def test_backward_paths_match_fp32_reference(
    expected_path,
    shape,
    dtype,
    weight_dtype,
):
    torch.manual_seed(2)
    x = (torch.randn(shape, device="cuda", dtype=dtype) * 0.5).requires_grad_()
    weight = (
        1.0 + torch.randn(shape[-1], device="cuda", dtype=weight_dtype) * 0.1
    ).requires_grad_()
    dout = torch.randn(shape, device="cuda", dtype=dtype) * 0.1
    eps = 1e-6
    dtype_str = rmsnorm_flydsl_impl._dtype_to_str(dtype)
    path, _ = rmsnorm_flydsl_impl._select_rmsnorm_bwd_config(
        shape[0],
        shape[1],
        dtype_str,
        x.device,
    )
    assert path == expected_path

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
    n = 257
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


class _FlyDSLOpCounter(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "_rmsnorm_flydsl_" in str(func):
            self.count += 1
        return func(*args, **(kwargs or {}))


def _clear_caches():
    rmsnorm_flydsl_impl._FWD_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CACHE.clear()
    rmsnorm_flydsl_impl._BWD_CU_COUNT_CACHE.clear()


def test_custom_ops_are_unique_mutation_only_and_fake_safe():
    fwd = torch.ops.quack._rmsnorm_flydsl_fwd.default
    bwd = torch.ops.quack._rmsnorm_flydsl_bwd.default
    assert str(fwd._schema).endswith("-> ()")
    assert str(bwd._schema).endswith("-> ()")
    assert "Tensor(a2!) out" in str(fwd._schema)
    assert "Tensor(a3!) rstd" in str(fwd._schema)
    assert "Tensor(a4!) dx" in str(bwd._schema)
    assert "Tensor(a5!) dweight" in str(bwd._schema)
    assert "Tensor(a6!) partial" in str(bwd._schema)

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty((2, 64), device="cuda", dtype=torch.float16)
        weight = torch.empty(64, device="cuda", dtype=torch.float16)
        out = torch.empty_like(x)
        rstd = torch.empty(2, device="cuda", dtype=torch.float32)
        fwd(x, weight, out, rstd, 1e-6, True)

        dout = torch.empty_like(x)
        dx = torch.empty_like(x)
        dweight = torch.empty_like(weight)
        partial = torch.empty(0, device="cuda", dtype=torch.float32)
        bwd(x, weight, dout, rstd, dx, dweight, partial, 0)


def test_eager_fast_path_bypasses_custom_op_dispatch():
    x = torch.randn(
        (4, 129),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(
        129,
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
        (8, 513),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(
        513,
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
    assert {key[0] for key in rmsnorm_flydsl_impl._BWD_CACHE} == {"two_stage"}


def test_forward_cache_identity_includes_n_dtype_and_eps():
    _clear_caches()

    def call(n, dtype, eps):
        x = torch.randn((2, n), device="cuda", dtype=dtype)
        weight = torch.randn(n, device="cuda", dtype=dtype)
        return rmsnorm(x, weight, eps=eps)

    first = call(129, torch.float16, 1e-6)
    first_launcher = next(iter(rmsnorm_flydsl_impl._FWD_CACHE.values()))
    second = call(129, torch.float16, 1e-6)
    _assert_close(first, first)
    _assert_close(second, second)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1
    assert next(iter(rmsnorm_flydsl_impl._FWD_CACHE.values())) is first_launcher

    call(130, torch.float16, 1e-6)
    call(129, torch.bfloat16, 1e-6)
    call(129, torch.float16, 1e-5)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 4


def test_non_default_stream_forward_backward():
    torch.manual_seed(5)
    x = torch.randn(
        (16, 257),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        257,
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
    assert {key[1] for key in rmsnorm_flydsl_impl._BWD_CACHE} == set(range(8))


def test_architecture_overrides_must_agree(monkeypatch):
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    with pytest.raises(ValueError, match="must select the same"):
        rmsnorm_flydsl_impl._validate_arch(torch.device("cuda", 0))


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


def test_vendored_source_is_pinned_and_isolated():
    assert FLYDSL_UPSTREAM_SHA == UPSTREAM_SHA
    source_root = Path(__file__).resolve().parents[1] / "quack" / "_flydsl"
    for filename in (
        "rmsnorm_kernel.py",
        "rmsnorm_bwd_kernel.py",
        "rmsnorm_common.py",
        "kernel_utils.py",
    ):
        source = (source_root / filename).read_text(encoding="utf-8")
        assert "from kernels." not in source
        assert "import kernels." not in source
        assert "autotune" not in source.lower()
        assert "quant" not in source.lower()
        assert "fused_add" not in source

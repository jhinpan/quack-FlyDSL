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

from quack.flydsl import FLYDSL_UPSTREAM_SHA  # noqa: E402
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
        # Vectorized with a predicated final tile: 375 vectors over 256 threads.
        ((3, 3000), torch.bfloat16, torch.bfloat16, 1e-6),
        ((2, 3000), torch.float16, torch.float32, 1e-6),
        # Predicated inside a single tile: 125 vectors over a 128-thread block.
        ((4, 1000), torch.bfloat16, torch.float32, 1e-6),
        # 128-bit FP32 loads, exact and predicated.
        ((2, 2048), torch.float32, torch.float32, 1e-6),
        ((2, 1020), torch.float32, torch.float32, 1e-6),
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
        (torch.ones(8), None, NotImplementedError, "requires an explicit weight"),
        (torch.ones(8), 5.0, TypeError, "weight must be a torch.Tensor"),
        (torch.tensor(1.0), torch.ones(1), ValueError, "at least one dimension"),
        (torch.ones(2, 8), torch.ones(2, 8), NotImplementedError, "per-head"),
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
        # Staged backward with 128-bit FP32 column I/O.
        ("two_stage", (512, 2048), torch.float32, torch.float32),
        # Staged backward whose column count is not a whole number of blocks.
        ("two_stage", (512, 3000), torch.bfloat16, torch.float32),
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
    rmsnorm_flydsl_impl._DEVICE_ARCH_CACHE.clear()


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


def test_forward_cache_identity_is_shape_and_dtype_only():
    """eps is a launch argument, so it must not multiply compiled kernels."""
    _clear_caches()

    def call(n, dtype, eps):
        x = torch.randn((2, n), device="cuda", dtype=dtype)
        weight = torch.randn(n, device="cuda", dtype=dtype)
        return rmsnorm(x, weight, eps=eps)

    call(129, torch.float16, 1e-6)
    first_launcher = next(iter(rmsnorm_flydsl_impl._FWD_CACHE.values()))
    call(129, torch.float16, 1e-6)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1
    assert next(iter(rmsnorm_flydsl_impl._FWD_CACHE.values())) is first_launcher

    for eps in (1e-5, 1e-4, 3e-7, 0.5):
        call(129, torch.float16, eps)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 1

    call(130, torch.float16, 1e-6)
    call(129, torch.bfloat16, 1e-6)
    assert len(rmsnorm_flydsl_impl._FWD_CACHE) == 3


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
    """math.isfinite on a symbolic float used to break dynamic tracing."""
    torch._dynamo.reset()
    compiled = torch.compile(rmsnorm, fullgraph=True, dynamic=True)
    weight = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    for rows in (8, 16, 32):
        x = torch.randn((rows, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = compiled(x, weight, eps=1e-5)
        out.sum().backward()
        _assert_close(out, _reference(x, weight, 1e-5))
        assert x.grad is not None


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


def test_compile_target_must_match_the_device(monkeypatch):
    """FlyDSL's own target is the authority, not the ARCH environment."""
    _clear_caches()
    device_arch = rmsnorm_flydsl_impl._normalize_arch(
        torch.cuda.get_device_properties(0).gcnArchName
    )
    other = "gfx942" if device_arch != "gfx942" else "gfx950"
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("rocm", other))
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

    device_arch = rmsnorm_flydsl_impl._normalize_arch(
        torch.cuda.get_device_properties(0).gcnArchName
    )
    other = "gfx942" if device_arch != "gfx942" else "gfx950"
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_flydsl_compile_target", lambda: ("rocm", other))
    with pytest.raises(ValueError, match="mixed architectures"):
        rmsnorm(torch.randn((8, 256), device="cuda", dtype=torch.bfloat16), weight[:256])


def test_a_warp_size_mismatch_is_rejected():
    """The reductions bake in a wavefront size; a disagreeing target must fail loudly."""
    from quack.flydsl.rmsnorm_common import assert_arch_matches_reductions

    assert_arch_matches_reductions("gfx942")
    assert_arch_matches_reductions("gfx950")
    with pytest.raises(RuntimeError, match="wavefront"):
        assert_arch_matches_reductions("gfx1100")


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


@pytest.mark.parametrize(
    ("kwargs", "feature"),
    [
        ({"bias": "tensor"}, "bias"),
        ({"residual": "tensor"}, "residual"),
        ({"out_dtype": torch.float32}, "out_dtype"),
        ({"residual_dtype": torch.float32}, "residual_dtype"),
        ({"prenorm": True}, "prenorm"),
        ({"weight_offset": 1.0}, "weight_offset"),
    ],
)
def test_unsupported_upstream_features_name_themselves(kwargs, feature):
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    if kwargs.get("bias") == "tensor":
        kwargs["bias"] = weight
    if kwargs.get("residual") == "tensor":
        kwargs["residual"] = x
    with pytest.raises(NotImplementedError, match=feature):
        rmsnorm(x, weight, **kwargs)


def test_omitting_the_weight_is_rejected():
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="weight"):
        rmsnorm(x)


def test_per_head_weight_is_rejected():
    x = torch.randn((2, 4, 32), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4, 32), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="per-head"):
        rmsnorm(x, weight)


def test_upstream_defaults_still_run():
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    _assert_close(rmsnorm(x, weight), _reference(x, weight, 1e-6))


def test_software_bf16_rounding_matches_the_hardware_convert():
    """Cover the rounding path that only pre-gfx95x parts take.

    gfx942 has no packed fp32->bf16 convert, so the kernel rounds to nearest
    even by hand. That branch is otherwise dead on this machine.
    """
    from quack.flydsl.kernel_utils import run_compiled
    from quack.flydsl.rmsnorm_kernel import build_rmsnorm_module

    torch.manual_seed(3)
    n = 4096
    x = torch.randn((64, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream

    rounded = {}
    for arch in ("gfx950", "gfx942"):
        out = torch.empty_like(x)
        launcher = build_rmsnorm_module(
            n,
            "bf16",
            store_rstd=False,
            weight_dtype_str="f32",
            arch=arch,
        )
        run_compiled(launcher, x, weight, out, x.shape[0], 1e-6, stream)
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


def test_deterministic_mode_avoids_the_atomic_weight_reduction():
    """Unordered fp32 atomics make dweight vary run to run.

    Without this, whether a backward is reproducible depends on the batch
    size, because the atomic path is only chosen below 512 rows.
    """
    torch.manual_seed(4)
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(x)

    def weight_grad():
        torch.manual_seed(4)
        weight = torch.randn(512, device="cuda", dtype=torch.float32).requires_grad_(True)
        rmsnorm(x, weight).backward(dout)
        return weight.grad.clone()

    _clear_caches()
    torch.use_deterministic_algorithms(True)
    try:
        grads = [weight_grad() for _ in range(6)]
        assert {key[0] for key in rmsnorm_flydsl_impl._BWD_CACHE} == {"two_stage"}
    finally:
        torch.use_deterministic_algorithms(False)

    for later in grads[1:]:
        torch.testing.assert_close(later, grads[0], rtol=0, atol=0)


def test_small_batches_still_use_the_atomic_backward_by_default():
    _clear_caches()
    torch.manual_seed(5)
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn_like(x)

    out = rmsnorm(x, weight)
    out.backward(dout)

    assert {key[0] for key in rmsnorm_flydsl_impl._BWD_CACHE} == {"atomic"}
    _, dx_expected, dweight_expected = _reference_with_grads(x, weight, dout, 1e-6)
    _assert_grad_close(x.grad, dx_expected)
    _assert_grad_close(weight.grad, dweight_expected)


def test_unsupported_architectures_are_named(monkeypatch):
    _clear_caches()
    monkeypatch.setattr(rmsnorm_flydsl_impl, "_normalize_arch", lambda _: "gfx90a")
    with pytest.raises(ValueError, match="gfx942, gfx950"):
        rmsnorm_flydsl_impl._validate_arch(torch.device("cuda", 0))


def test_vendored_source_is_pinned_and_isolated():
    assert FLYDSL_UPSTREAM_SHA == UPSTREAM_SHA
    source_root = Path(__file__).resolve().parents[1] / "quack" / "flydsl"
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

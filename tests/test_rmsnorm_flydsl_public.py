# Copyright (c) 2026, Tri Dao.

"""Focused public-contract coverage for the FlyDSL RMSNorm backend."""

import importlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from flydsl.autotune import Config

if torch.version.hip is None:
    pytest.skip("FlyDSL RMSNorm requires a ROCm PyTorch build", allow_module_level=True)

try:
    import flydsl.compiler  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "flydsl":
        raise
    pytest.skip("flydsl is not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6
quack = importlib.import_module("quack")
rmsnorm = quack.rmsnorm
rmsnorm_flydsl_impl = importlib.import_module("quack.rmsnorm_flydsl")
rmsnorm_autotuned = rmsnorm_flydsl_impl.rmsnorm_autotuned


def _reference(
    x,
    weight=None,
    bias=None,
    residual=None,
    *,
    eps=EPS,
    weight_offset=0.0,
    out_dtype=None,
    residual_dtype=None,
):
    value = x.float()
    if residual is not None:
        value = value + residual.float()
    output = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        output = output * (weight.float() + weight_offset)
    if bias is not None:
        output = output + bias.float()
    output = output.to(x.dtype if out_dtype is None else out_dtype)
    residual_out = value.to(
        residual_dtype
        if residual_dtype is not None
        else (residual.dtype if residual is not None else x.dtype)
    )
    return output, residual_out


def _assert_close(actual, expected):
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _assert_grad_close(actual, expected):
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_package_export_is_lazy_and_matches_the_cute_signature():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import ast
                import inspect
                import sys
                from pathlib import Path

                import torch

                assert torch.version.hip is not None
                import quack

                assert "quack.rmsnorm" not in sys.modules
                assert "quack.rmsnorm_flydsl" not in sys.modules
                assert not any(
                    name == "flydsl" or name.startswith("flydsl.") for name in sys.modules
                )

                public = quack.rmsnorm
                assert public is quack.rmsnorm
                assert public.__module__ == "quack.rmsnorm_flydsl"
                assert "quack.rmsnorm" not in sys.modules
                assert "quack.rmsnorm_flydsl" in sys.modules

                source = Path(quack.__file__).with_name("rmsnorm.py").read_text()
                upstream = next(
                    node
                    for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "rmsnorm"
                )
                names = [argument.arg for argument in upstream.args.args]
                defaults = [ast.unparse(default) for default in upstream.args.defaults]
                expected_defaults = dict(zip(names[-len(defaults):], defaults))
                ours = inspect.signature(public).parameters

                assert list(ours) == names
                for name, default in expected_defaults.items():
                    assert repr(ours[name].default) == default, name
                """
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_forward_matches_fp32_reference():
    torch.manual_seed(0)
    x = torch.randn((3, 1024), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, device=x.device, dtype=torch.float32)

    actual = rmsnorm(x, weight, eps=1e-5)
    expected, _ = _reference(x, weight, eps=1e-5)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    _assert_close(actual, expected)


def test_repeated_metadata_reuses_validated_input_plan(monkeypatch):
    rmsnorm_flydsl_impl._VALIDATED_INPUT_LAST[0] = None
    calls = 0
    real_validate = rmsnorm_flydsl_impl._validate_inputs

    def count_validate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(rmsnorm_flydsl_impl, "_validate_inputs", count_validate)
    weight = torch.randn(512, device="cuda", dtype=torch.float32)
    for _ in range(2):
        x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
        rmsnorm(x, weight)
    assert calls == 1

    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
    rmsnorm(x, weight, eps=1e-5)
    assert calls == 2


def test_plain_inference_fast_path_is_shape_generic(monkeypatch):
    x = torch.randn((63, 2048), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device=x.device, dtype=torch.float32)
    expected, _ = _reference(x, weight)

    def forbid_layout(_tensor):
        raise AssertionError("plain packed inference entered layout canonicalization")

    monkeypatch.setattr(rmsnorm_flydsl_impl, "_packed_rows", forbid_layout)
    _assert_close(rmsnorm(x, weight), expected)
    _assert_close(rmsnorm_autotuned(x, weight), expected)


def test_resolved_generic_winner_uses_the_lower_overhead_public_launcher(monkeypatch):
    calls = []
    validation_entry = (("metadata",), ("result",))
    rmsnorm_flydsl_impl._VALIDATED_INPUT_LAST[0] = validation_entry
    rmsnorm_flydsl_impl._FWD_PUBLIC_GENERIC_LAST[0] = None
    rmsnorm_flydsl_impl._FWD_PUBLIC_RESOLVED_LAST[0] = None
    monkeypatch.setattr(
        rmsnorm_flydsl_impl,
        "_launch_rmsnorm_fwd",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    tensor = object()
    rmsnorm_flydsl_impl._launch_resolved_fwd_entry(
        (Config(threads_per_row=64), None, None),
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        64,
        1e-6,
        0.0,
        has_weight=True,
        has_bias=False,
        has_residual=False,
        store_residual=False,
        store_rstd=False,
        per_head=False,
        num_heads=1,
    )

    assert len(calls) == 1
    assert rmsnorm_flydsl_impl._FWD_PUBLIC_GENERIC_LAST[0] is validation_entry


def test_plain_inference_uses_resolved_public_entry_before_tuner_keys(monkeypatch):
    rmsnorm_flydsl_impl._VALIDATED_INPUT_LAST[0] = None
    rmsnorm_flydsl_impl._FWD_PUBLIC_RESOLVED_LAST[0] = None
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device=x.device, dtype=torch.float32)
    rmsnorm_flydsl_impl._validated_inputs(
        x,
        weight,
        None,
        None,
        None,
        None,
        EPS,
        False,
        0.0,
    )
    calls = []
    config = Config(
        threads_per_row=64,
        input_cache_modifier=2,
        output_cache_modifier=2,
    )
    rmsnorm_flydsl_impl._FWD_PUBLIC_RESOLVED_LAST[0] = (
        rmsnorm_flydsl_impl._VALIDATED_INPUT_LAST[0],
        rmsnorm_flydsl_impl._public_runtime_guard(),
        (config, lambda *args: calls.append(args), ()),
    )

    def forbid_context():
        raise AssertionError("rebuilt tuner context")

    monkeypatch.setattr(
        rmsnorm_flydsl_impl._rmsnorm_fwd_tuner,
        "fast_context_token",
        forbid_context,
    )

    rmsnorm_autotuned(x, weight)

    assert len(calls) == 1


def test_autotuned_forward_last_hit_bypasses_tuner_on_runtime_stream(monkeypatch):
    tuner = rmsnorm_flydsl_impl._rmsnorm_fwd_tuner
    rmsnorm_flydsl_impl._FWD_AUTOTUNED_FAST_CACHE.clear()
    rmsnorm_flydsl_impl._FWD_AUTOTUNED_LAST[0] = None
    rmsnorm_flydsl_impl._FWD_PUBLIC_GENERIC_LAST[0] = None
    rmsnorm_flydsl_impl._FWD_PUBLIC_RESOLVED_LAST[0] = None
    rmsnorm_flydsl_impl._VALIDATED_INPUT_LAST[0] = None
    tuner.cache.clear()
    tuner._hot_cache.clear()
    torch.manual_seed(4)
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device=x.device, dtype=torch.float32)

    first = rmsnorm_autotuned(x, weight)
    assert rmsnorm_flydsl_impl._FWD_AUTOTUNED_LAST[0] is not None

    def forbid_tuner_entry(*args, **kwargs):
        raise AssertionError("warm forward call re-entered the tuner")

    monkeypatch.setattr(type(tuner), "__call__", forbid_tuner_entry)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        second = rmsnorm_autotuned(x, weight)
    stream.synchronize()

    torch.testing.assert_close(second, first, rtol=0.0, atol=0.0)


def test_default_launch_policy_contains_no_benchmark_shape_literals():
    policy = "\n".join(
        inspect.getsource(function)
        for function in (
            rmsnorm_flydsl_impl._launch_rmsnorm_fwd,
            rmsnorm_flydsl_impl._launch_rmsnorm_bwd,
            rmsnorm_flydsl_impl._rmsnorm_impl,
        )
    )

    assert "32768" not in policy
    assert "_FWD_PERSISTENT_CONFIGS" not in policy
    assert "measured_" not in policy


def test_backward_matches_reference_and_is_deterministic():
    torch.manual_seed(1)
    x = torch.randn((64, 760), device="cuda", dtype=torch.bfloat16) * 0.5
    weight = 1.0 + torch.randn(760, device=x.device, dtype=torch.float32) * 0.1
    dout = torch.randn_like(x) * 0.1

    def run():
        x_i = x.detach().clone().requires_grad_(True)
        weight_i = weight.detach().clone().requires_grad_(True)
        output = rmsnorm(x_i, weight_i)
        output.backward(dout)
        return output.detach(), x_i.grad, weight_i.grad

    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        first = run()
        second = run()
    finally:
        torch.use_deterministic_algorithms(was_deterministic)

    x_ref = x.float().detach().requires_grad_(True)
    weight_ref = weight.float().detach().requires_grad_(True)
    output_ref, _ = _reference(x_ref, weight_ref)
    dx_ref, dweight_ref = torch.autograd.grad(
        output_ref,
        (x_ref, weight_ref),
        dout.float(),
    )

    _assert_close(first[0], output_ref.to(first[0].dtype))
    _assert_grad_close(first[1], dx_ref.to(first[1].dtype))
    _assert_grad_close(first[2], dweight_ref.to(first[2].dtype))
    for original, repeated in zip(first, second):
        torch.testing.assert_close(original, repeated, rtol=0.0, atol=0.0)


def test_autotuned_backward_reuses_resolved_callable(monkeypatch):
    tuner = rmsnorm_flydsl_impl._rmsnorm_bwd_tuner
    rmsnorm_flydsl_impl._FWD_AUTOTUNED_FAST_CACHE.clear()
    rmsnorm_flydsl_impl._FWD_AUTOTUNED_LAST[0] = None
    rmsnorm_flydsl_impl._BWD_AUTOTUNED_FAST_CACHE.clear()
    tuner.cache.clear()
    tuner._hot_cache.clear()
    torch.manual_seed(3)
    x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device=x.device, dtype=torch.float32)
    dout = torch.randn_like(x)

    def run():
        x_i = x.detach().clone().requires_grad_(True)
        weight_i = weight.detach().clone().requires_grad_(True)
        output = rmsnorm_autotuned(x_i, weight_i)
        output.backward(dout)
        return output.detach(), x_i.grad, weight_i.grad

    first = run()
    assert len(rmsnorm_flydsl_impl._BWD_AUTOTUNED_FAST_CACHE) == 1

    def forbid_tuner_entry(*args, **kwargs):
        raise AssertionError("warm backward call re-entered the tuner")

    monkeypatch.setattr(type(tuner), "__call__", forbid_tuner_entry)
    second = run()
    for original, repeated in zip(first, second):
        torch.testing.assert_close(original, repeated, rtol=0.0, atol=0.0)


def test_per_head_bias_residual_and_prenorm_contract():
    torch.manual_seed(2)
    x = torch.randn((2, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn((4, 64), device=x.device, dtype=torch.float32)
    bias = torch.randn_like(weight)

    actual, residual_out = rmsnorm(
        x,
        weight,
        bias=bias,
        residual=residual,
        out_dtype=torch.float16,
        residual_dtype=torch.float32,
        prenorm=True,
        weight_offset=1.0,
    )
    expected, expected_residual = _reference(
        x,
        weight,
        bias,
        residual,
        out_dtype=torch.float16,
        residual_dtype=torch.float32,
        weight_offset=1.0,
    )

    assert actual.dtype == torch.float16
    assert residual_out.dtype == torch.float32
    _assert_close(actual, expected)
    _assert_close(residual_out, expected_residual)

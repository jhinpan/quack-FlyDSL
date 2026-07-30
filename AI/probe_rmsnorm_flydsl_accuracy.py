"""Probe: does the FlyDSL backend agree with an eager fp32 reference?

The test suite asserts this too, but at the tolerance each dtype deserves.
This probe pins the number itself, in fp32 where there is nowhere for an
error to hide, so a change that quietly costs a digit is visible rather than
merely still-passing. It is what the error figures in
flydsl_rmsnorm_notes.md are measured with.

It also covers the shapes that are easy to get wrong and cheap to check: a
caller that ignores the second output under prenorm, a non-contiguous input,
and per-head parameters with a fused residual, where the saved rstd has to be
grouped by (row, head) rather than by row.
"""

import torch

from quack.rmsnorm_flydsl import rmsnorm

results = []


def record(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def reference(x, w=None, bias=None, residual=None, eps=1e-6, weight_offset=0.0):
    summed = x.float()
    if residual is not None:
        summed = summed + residual.float()
    normalized = summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + eps)
    out = normalized * (w.float() + weight_offset) if w is not None else normalized
    if bias is not None:
        out = out + bias.float()
    return out, summed


def rel_err(actual, expected):
    return ((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-6)).item()


# Autograd materializes the unused residual_out gradient as zeros rather than
# passing None, so ignoring the second output is not the crash it looks like.
for has_residual in (True, False):
    x = torch.randn(64, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
    res = (
        torch.randn(64, 512, device="cuda", dtype=torch.float32, requires_grad=True)
        if has_residual
        else None
    )
    try:
        out, _ = rmsnorm(x, w, residual=res, prenorm=True)
        out.sum().backward()
        record(f"prenorm, second output ignored (residual={has_residual})", True)
    except Exception as exc:  # noqa: BLE001
        record(
            f"prenorm, second output ignored (residual={has_residual})",
            False,
            f"{type(exc).__name__}: {exc}",
        )

x = torch.randn(64, 512, device="cuda", dtype=torch.float32, requires_grad=True)
w = torch.randn(512, device="cuda", dtype=torch.float32, requires_grad=True)
res = torch.randn(64, 512, device="cuda", dtype=torch.float32, requires_grad=True)
try:
    out, _ = rmsnorm(x, w, residual=res, prenorm=True)
    torch.autograd.grad(out.sum(), x)
    record("prenorm, autograd.grad against x alone", True)
except Exception as exc:  # noqa: BLE001
    record("prenorm, autograd.grad against x alone", False, f"{type(exc).__name__}: {exc}")


CASES = [
    dict(name="bias only", weight=True, bias=True, residual=False, offset=0.0),
    dict(name="residual fused", weight=True, bias=False, residual=True, offset=0.0),
    dict(name="gemma offset", weight=True, bias=False, residual=False, offset=1.0),
    dict(name="no weight", weight=False, bias=False, residual=False, offset=0.0),
    dict(name="every feature at once", weight=True, bias=True, residual=True, offset=1.0),
]
for case in CASES:
    torch.manual_seed(0)
    m, n = 256, 1024
    x = torch.randn(m, n, device="cuda", dtype=torch.float32, requires_grad=True)
    w = (
        torch.randn(n, device="cuda", dtype=torch.float32, requires_grad=True)
        if case["weight"]
        else None
    )
    bias = (
        torch.randn(n, device="cuda", dtype=torch.float32, requires_grad=True)
        if case["bias"]
        else None
    )
    res = (
        torch.randn(m, n, device="cuda", dtype=torch.float32, requires_grad=True)
        if case["residual"]
        else None
    )
    leaves = [t for t in (x, w, bias, res) if t is not None]

    out = rmsnorm(x, w, bias, res, weight_offset=case["offset"])
    want_out, _ = reference(x, w, bias, res, weight_offset=case["offset"])
    seed_grad = torch.randn_like(out)
    got = torch.autograd.grad((out * seed_grad).sum(), leaves)
    want = torch.autograd.grad((want_out * seed_grad).sum(), leaves)

    forward = rel_err(out, want_out)
    backward = max(rel_err(a, b) for a, b in zip(got, want))
    record(
        f"fp32 parity: {case['name']}",
        forward < 1e-5 and backward < 1e-4,
        f"fwd={forward:.2e} bwd={backward:.2e}",
    )


# The adapter flattens and copies a non-contiguous input rather than refusing
# it, so the result has to match the same math on a contiguous one.
try:
    n = 1024
    x = torch.randn(64, n, device="cuda", dtype=torch.float32).t().contiguous().t()
    assert not x.is_contiguous()
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    out = rmsnorm(x, w, bias)
    want_out, _ = reference(x, w, bias)
    err = rel_err(out, want_out)
    record("non-contiguous input", err < 1e-5, f"err={err:.2e}")
except Exception as exc:  # noqa: BLE001
    record("non-contiguous input", False, f"{type(exc).__name__}: {exc}")


torch.manual_seed(3)
batch, seq, heads, dim = 2, 8, 4, 128
x = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float32, requires_grad=True)
w = torch.randn(heads, dim, device="cuda", dtype=torch.float32, requires_grad=True)
res = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float32, requires_grad=True)
out, residual_out = rmsnorm(x, w, residual=res, prenorm=True)
want_out, want_residual = reference(x, w, residual=res)
err_out = rel_err(out, want_out)
err_residual = rel_err(residual_out, want_residual)
record(
    "per-head parameters with a fused residual",
    err_out < 1e-5 and err_residual < 1e-5,
    f"out={err_out:.2e} residual={err_residual:.2e}",
)

print()
print(f"{sum(results)}/{len(results)} checks passed")
torch.cuda.synchronize()

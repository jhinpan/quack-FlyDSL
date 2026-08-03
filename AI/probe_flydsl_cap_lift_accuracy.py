"""Does the FlyDSL kernel stay correct above MAX_N, and can you prove which one ran?

The notes assert the forward runs to N=262144 at bf16 accuracy indistinguishable
from shapes under the cap. The first version of this probe measured that and was
still not evidence, for reasons @CrossVendor gave when he held two older MI355X
sidecars to a provenance standard and then applied the same standard to this one:

  "Its JSON also omits the executed commit/tree, interpreter, device UUID,
   toolchain, source hashes, and raw samples, so this audit does not use it to
   upgrade the a0d capability claim."

He was right, and there was a worse version of the same defect he could not see
from the artifact: the probe did ``sys.path.insert(0, "/root/quack-FlyDSL-review")``,
a *shared* checkout that other agents move. So the JSON could not name the code
that produced it even in principle -- an accuracy number attributed to a tree
that may not have been the tree. That is this file's recurring defect wearing
provenance clothes: a measurement correct about some checkout other than the one
its label names.

So: the repo under test is passed in and asserted clean and at an expected
commit/tree before any GPU call, every load-bearing source file is hashed, the
imported modules are asserted to resolve inside that repo, the environment is
recorded, raw per-row samples are retained, and backward is measured too --
@CrossVendor's H100 wide row covered fwd+bwd and this covered only fwd, which
made the two sides unequal in scope as well as in rigour.

Nothing is written back to the tree under test and both MAX_N bindings are
restored in a finally.

Usage:
    python AI/probe_flydsl_cap_lift_accuracy.py --repo <clean worktree> \
        --expect-commit <sha> --output <path>
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

# The suite does not have "a bf16 tolerance"; it has two, and they differ.
# ``_assert_close`` (tests/test_rmsnorm_flydsl.py:58-62) checks the forward
# output at rtol=atol=2e-2. ``_assert_grad_close`` (:77-81) checks gradients at
# 3e-2. Both are applied as assert_close applies them:
# ``|a - b| <= atol + rtol * |b|``.
#
# The previous version used 3e-2 for all three tensors, which is 50% too loose
# on the forward output, and cited :94 as its source -- that is
# ``_assert_fused_residual_grad_close``, the helper for gradients recomputed
# from a rounded residual, which is neither of the two that apply here and
# whose own docstring says the no-residual tests deliberately do not borrow it.
# So the constant was wrong for one tensor and the citation named a third
# helper. @CrossVendor caught it by reading the suite rather than my summary of
# it. One number standing in for a set that has two, again.
_FWD_RTOL = _FWD_ATOL = 2e-2
_GRAD_RTOL = _GRAD_ATOL = 3e-2

_HASHED_SOURCES = (
    "quack/rmsnorm_flydsl.py",
    "quack/flydsl/rmsnorm_config.py",
    "quack/flydsl/rmsnorm_kernel.py",
    "quack/flydsl/rmsnorm_bwd_kernel.py",
    "quack/flydsl/rmsnorm_common.py",
)


def _sha256(path):
    """Hash or die. A null hash beside a real one reads as 'file unchanged'."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _pin_repo(repo, expect_commit):
    """Refuse to measure anything until the tree under test is pinned and clean."""
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise SystemExit(f"{repo} is dirty; refusing to attribute numbers to {commit}:\n{dirty}")
    if expect_commit and not commit.startswith(expect_commit):
        raise SystemExit(f"{repo} is at {commit}, not the expected {expect_commit}")
    return commit, tree


def _environment(repo, commit, tree):
    import torch

    props = torch.cuda.get_device_properties(0)
    try:
        uuid = str(props.uuid)
    except AttributeError:
        uuid = None
    return {
        "repo": os.path.abspath(repo),
        "commit": commit,
        "tree": tree,
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "device_name": props.name,
        "device_uuid": uuid,
        "device_count_visible": torch.cuda.device_count(),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": {p: _sha256(os.path.join(repo, p)) for p in _HASHED_SOURCES},
    }


def ref(x, w, eps):
    import torch

    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _worst_samples(torch, actual, expected, rtol, atol, k=8):
    """Paired records at the k worst elements, enough to recompute the verdict.

    @CrossVendor: the previous ``samples`` field was the first eight forward
    outputs with no paired reference, index, difference or threshold, so the
    aggregate metrics could not be recomputed from the JSON. They were
    representative values wearing the name "raw samples" -- and, being the
    *first* eight, they were the elements least likely to carry the answer.
    These are the worst offenders under the same criterion the verdict uses,
    each carrying everything needed to check the arithmetic by hand.

    Two orderings, because they are different elements and each backs a
    different reported number. ``margin`` sorts by ``diff - threshold`` and so
    holds the elements nearest to failing, which is what the outside-counts and
    the boolean rest on. ``abs`` sorts by raw difference and includes the
    element that produced ``max_abs_err``, which margin-ordering can miss
    entirely -- a large difference beside a large reference has a comfortable
    margin. Reporting only the first would leave a headline number
    uncheckable from the JSON, which is the whole complaint being answered.
    Caveat that the records carry themselves: at these shapes most elements
    round to zero in both tensors, so ``diff - thr`` ties at exactly ``-atol``
    across millions of them and ``argsort`` returns eight arbitrary members of
    that tie set. Those rows are checkable but not reproducible -- a rerun may
    name eight different indices with identical numbers. ``margin_tied_at_min``
    counts the tie set so a reader can see that, and ``worst_margin`` is the
    tensor-wide extremum, which is a fact about the tensor rather than about
    which representative argsort happened to pick.
    """
    a, b = actual.float().flatten(), expected.float().flatten()
    diff = (a - b).abs()
    thr = atol + rtol * b.abs()
    margin = diff - thr
    by_margin = torch.argsort(margin, descending=True)[:k].tolist()
    by_abs = torch.argsort(diff, descending=True)[:k].tolist()
    idx = list(dict.fromkeys(by_margin + by_abs))
    records = [
        {
            "index": int(i),
            "actual": float(a[i]),
            "expected": float(b[i]),
            "abs_diff": float(diff[i]),
            "threshold": float(thr[i]),
            "outside": bool(diff[i] > thr[i]),
            "why": "+".join(
                ([" margin"] if i in by_margin else []) + ([" abs"] if i in by_abs else [])
            ).strip(),
        }
        for i in idx
    ]
    return records, {
        "worst_margin": float(margin.max()),
        "margin_tied_at_min": int((margin <= margin.min() + 1e-9).sum()),
        "n_elements": int(a.numel()),
    }


def _row(fd, torch, n, m, dtype, eps, shipped):
    """One shape, forward and backward, with paired worst-case samples retained."""
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
    rec = {"N": n, "m": m, "over_cap": n > shipped, "status": "raised", "bwd_status": "not_reached"}
    try:
        got = fd.rmsnorm(x, w, eps=eps)
        exp = ref(x, w, eps)
        d = (got.float() - exp.float()).abs()
        # Forward gets the same criterion as backward. It did not before: the
        # forward branch recorded max/mean error and never applied the suite's
        # test, while the row-level boolean was named for the whole comparison
        # and computed from dx and dw alone. So "zero elements outside
        # tolerance, forward and backward" was a claim about a set the field
        # never examined -- this file's recurring defect, in the field I had
        # just added to fix the previous instance of it. @CrossVendor caught it
        # in the committed bytes.
        fwd_thr = _FWD_ATOL + _FWD_RTOL * exp.float().abs()
        s_out, g_out = _worst_samples(torch, got, exp, _FWD_RTOL, _FWD_ATOL)
        rec.update(
            status="ok",
            mean_rel_err=(d / exp.float().abs().clamp_min(1e-6)).mean().item(),
            max_abs_err=d.max().item(),
            n_out_outside_combined=int((d > fwd_thr).sum().item()),
            finite=bool(torch.isfinite(got).all().item()),
            samples_out_worst=s_out,
            margin_out=g_out,
        )

        # Backward too: @CrossVendor's H100 wide row covered fwd+bwd, and a
        # forward-only counterpart is not the same claim.
        g = torch.randn_like(got)
        xr = x.detach().clone().requires_grad_(True)
        wr = w.detach().clone().requires_grad_(True)
        ref(xr, wr, eps).backward(g)
        got.backward(g)
        # The verdict is the suite's own combined criterion,
        # ``|a - b| <= atol + rtol * |b|``, which is what assert_close applies
        # at each tensor's own tolerance: 2e-2 for the forward output
        # (_assert_close), 3e-2 for the gradients (_assert_grad_close).
        #
        # Two wrong single-number verdicts came before this one, in opposite
        # directions. Bare max_abs_err_dw = 0.125 looked like a breach; the dw
        # it sits on has magnitude 124, so as a fraction it is 7.5e-3, fine.
        # Switching to bare max relative error then flagged N=131072 and
        # 262144 as failures at 1.7e-1 and 4.9e-2 -- but those maxima land on
        # single elements whose reference dw is ~1e-5, one bf16 ulp, where a
        # ratio measures rounding rather than accuracy: 1 element of 131072 and
        # 2 of 262144, with the top-magnitude decile at 5e-3 and 6.8e-3. So the
        # bare-relative verdict was as wrong as the bare-absolute one, in the
        # other direction, and neither is what the suite asserts. Both raw
        # numbers are still recorded; only ``within_suite_tolerance`` is the
        # claim, and it is the conjunction over out, dx and dw.
        dxr, dwr = xr.grad.float(), wr.grad.float()
        dxd = (x.grad.float() - dxr).abs()
        dwd = (w.grad.float() - dwr).abs()
        s_dx, g_dx = _worst_samples(torch, x.grad, dxr, _GRAD_RTOL, _GRAD_ATOL)
        s_dw, g_dw = _worst_samples(torch, w.grad, dwr, _GRAD_RTOL, _GRAD_ATOL)
        n_dx_out = int((dxd > _GRAD_ATOL + _GRAD_RTOL * dxr.abs()).sum().item())
        n_dw_out = int((dwd > _GRAD_ATOL + _GRAD_RTOL * dwr.abs()).sum().item())
        rec.update(
            bwd_status="ok",
            max_abs_err_dx=dxd.max().item(),
            max_abs_err_dw=dwd.max().item(),
            max_rel_err_dx=(dxd / dxr.abs().clamp_min(1e-6)).max().item(),
            max_rel_err_dw=(dwd / dwr.abs().clamp_min(1e-6)).max().item(),
            mean_rel_err_dw=(dwd / dwr.abs().clamp_min(1e-6)).mean().item(),
            dw_ref_absmax=dwr.abs().max().item(),
            n_dx_outside_combined=n_dx_out,
            n_dw_outside_combined=n_dw_out,
            samples_dx_worst=s_dx,
            margin_dx=g_dx,
            samples_dw_worst=s_dw,
            margin_dw=g_dw,
            bwd_finite=bool(torch.isfinite(x.grad).all().item())
            and bool(torch.isfinite(w.grad).all().item()),
        )
        # The conjunction over all three tensors, named for exactly that set.
        rec["within_suite_tolerance"] = (
            rec["n_out_outside_combined"] == 0 and n_dx_out == 0 and n_dw_out == 0
        )
        rec["within_suite_tolerance_covers"] = ["out", "dx", "dw"]
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        del x, w
        torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="clean worktree to measure")
    ap.add_argument("--expect-commit", default="", help="refuse to run unless HEAD matches")
    ap.add_argument("--output", default="", help="write JSON here instead of stdout")
    args = ap.parse_args()

    commit, tree = _pin_repo(args.repo, args.expect_commit)
    sys.path.insert(0, os.path.abspath(args.repo))

    import torch

    from quack import rmsnorm_flydsl as fd
    from quack.flydsl import rmsnorm_config

    for mod in (fd, rmsnorm_config):
        origin = os.path.abspath(mod.__file__)
        if not origin.startswith(os.path.abspath(args.repo)):
            raise SystemExit(f"imported {mod.__name__} from {origin}, outside the pinned repo")

    shipped = rmsnorm_config.MAX_N
    assert fd.MAX_N == shipped, "the two MAX_N bindings already disagree"

    out = {
        "what": "FlyDSL fwd+bwd accuracy with MAX_N lifted, vs the same kernel under the cap",
        "kind": "functional/correctness probe, not a benchmark",
        "tolerance": {
            "criterion": "abs(a - b) <= atol + rtol * abs(b), as assert_close applies it",
            "out": {
                "rtol": _FWD_RTOL,
                "atol": _FWD_ATOL,
                "source": "tests/test_rmsnorm_flydsl.py:58-62, _assert_close, bf16 branch",
            },
            "dx": {
                "rtol": _GRAD_RTOL,
                "atol": _GRAD_ATOL,
                "source": "tests/test_rmsnorm_flydsl.py:77-81, _assert_grad_close, bf16 branch",
            },
            "dw": {
                "rtol": _GRAD_RTOL,
                "atol": _GRAD_ATOL,
                "source": "tests/test_rmsnorm_flydsl.py:77-81, _assert_grad_close, bf16 branch",
            },
        },
        "dtype": "bfloat16",
        "shipped_max_n": shipped,
        "environment": _environment(args.repo, commit, tree),
        "probe_sha256": _sha256(__file__),
        "rows": [],
    }

    rmsnorm_config.MAX_N = 1 << 20
    fd.MAX_N = 1 << 20
    try:
        for n in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            out["rows"].append(
                _row(fd, torch, n, max(1, (1 << 24) // n), torch.bfloat16, 1e-6, shipped)
            )
    finally:
        rmsnorm_config.MAX_N = shipped
        fd.MAX_N = shipped
    assert fd.MAX_N == shipped and rmsnorm_config.MAX_N == shipped, "cap not restored"

    text = json.dumps(out, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()

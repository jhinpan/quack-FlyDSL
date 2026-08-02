"""CHECK 1 + CHECK 2 of AI/probe_restore_value_needed.py, against the CUTEDSL
kernel instead of the FlyDSL one.

The MI355X run could only test quack.rmsnorm_flydsl: quack/rmsnorm.py imports
cuda.bindings.driver at module scope, which does not exist on ROCm. So the
cutedsl half of the restore_value verdict rested on reading the schema
(:367 / :1210 mutate output buffers exclusively; inputs carry no (a!) alias)
rather than on hardware. This closes that gap.

Same two checks, same logic, different kernel:

  CHECK 1  replay _rmsnorm_fwd / _rmsnorm_bwd N times on the SAME buffers and
           compare every tensor bit-for-bit against the first replay. If the
           outputs are stable, nothing accumulates and restore_value has
           nothing to restore.

  CHECK 2  the falsifier. Alias the residual input onto residual_out, which is
           exactly the read-and-write dataflow restore_value exists to undo.
           If check 2 ALSO came back "stable", check 1 would be vacuous.

Timing is deliberately absent: the cost half of the verdict (the regime switch
from _bench_cuda_graph_l2_rotate to do_bench) is a property of the autotuner
and the harness, not of the kernel, and it was already measured on MI355X.
What could NOT be measured there is whether the cutedsl kernel mutates its
inputs. That is what this asks, and only that.

Where it ran, and the mistake that nearly stopped it running at all. I first
reported this run as blocked: both tailscale H200 boxes have
``nvidia-cutlass-dsl`` 4.5.2, ``pyproject.toml:10`` pins ``==4.6.1``, and
``quack/pipeline.py:13`` imports ``alloc_reserved_mbarrier``, which 4.5.2 does
not export. Every one of those facts is true, and the conclusion I drew from
them was still wrong: I had checked the SYSTEM python and reported the result
as a property of the MACHINE. hyper00 has ``/root/quack-FlyDSL-h200-test/.venv``
on cutlass 4.6.1 and hyper01 has ``/root/crossvendor-cu-venv`` on 4.6.0; under
either, ``import quack.rmsnorm`` at current head succeeds. This probe ran under
the first one. That is the file's own catalogued defect class -- a check that
is correct about a set other than the one its label names -- committed while
writing up an instance of it.

Provenance. Run on hyper00 (`100.101.70.115`) GPU 0, verified 0 MiB / 0% before
launch, from a scratch clone of @CrossVendor's checkout at ``4f36477``. That is
not my HEAD, but ``quack/rmsnorm.py`` and ``quack/autotuner.py`` -- the only
two files this probe exercises -- are byte-identical between the two
(blob ``26e30eff…`` and ``67875584…`` at both commits), so the difference
cannot reach this result. No shared checkout was modified; the scratch clone is
removed after collection.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

N_REPLAYS = 5
SHAPES = [(8192, 2048), (4096, 4096)]


def fp(t):
    if t is None:
        return None
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def check_fwd(M, N, dtype, alias_residual):
    from quack.rmsnorm import rmsnorm_fwd

    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    residual = torch.randn(M, N, device="cuda", dtype=dtype)

    prints = []
    carry = residual
    for _ in range(N_REPLAYS):
        out, residual_out, rstd = rmsnorm_fwd(x, w, None, carry, eps=1e-6, store_rstd=True)
        torch.cuda.synchronize()
        if alias_residual:
            carry = residual_out
        prints.append(
            {
                "out": fp(out),
                "residual_out": fp(residual_out),
                "rstd": fp(rstd),
                "x": fp(x),
                "weight": fp(w),
                "residual_in": fp(carry),
            }
        )
    first = prints[0]
    stable = all(p == first for p in prints[1:])
    drifted = sorted({k for p in prints[1:] for k in first if p[k] != first[k]})
    return {
        "kernel": "quack.rmsnorm.rmsnorm_fwd (cutedsl)",
        "shape": [M, N],
        "dtype": str(dtype).replace("torch.", ""),
        "feedback_aliased": alias_residual,
        "replays": N_REPLAYS,
        "all_replays_bit_identical": stable,
        "tensors_that_drifted": drifted,
        "restore_value_would_be_a_noop": stable,
        "first_replay_fingerprints": first,
    }


def check_bwd(M, N, dtype):
    """Replay the backward on the same INPUTS. Fresh outputs each time.

    Read the scoping carefully, because an earlier version of this docstring
    claimed a test this function does not perform. It said "replaying on the
    same dw_partial buffer drifts" -- but ``rmsnorm_bwd`` allocates dx and
    dw_partial internally on every call (:1382, :1395, and on the fast path
    inside ``quack::rmsnorm_bwd_f`` at :1443-1444). So no ``dw_partial``
    buffer is ever reused here and this check CANNOT see a read-modify-write
    of one. What it does establish is the weaker, still necessary condition:
    the backward does not mutate x / dout / rstd, and produces bit-identical
    dx / dw from identical inputs.

    That weaker statement is the one ``restore_value`` actually turns on:
    restore exists to undo mutation of the arguments the autotuner passes
    across config evaluations. If x, dout and rstd survive a replay unchanged,
    there is nothing for it to restore at this level.

    The dw_partial question is separate and stays where @Reviewer (b3053d0a)
    put it -- source-level, conditional on no aliasing. ``check_bwd_shared``
    below closes the part of it that a caller-side probe can close.
    """
    from quack.rmsnorm import rmsnorm_bwd

    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    dout = torch.randn(M, N, device="cuda", dtype=dtype)
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-6)

    prints = []
    for _ in range(N_REPLAYS):
        res = rmsnorm_bwd(x, w, dout, rstd)
        torch.cuda.synchronize()
        dx, dw = res[0], res[1]
        prints.append({"dx": fp(dx), "dw": fp(dw), "x": fp(x), "dout": fp(dout), "rstd": fp(rstd)})
    first = prints[0]
    stable = all(p == first for p in prints[1:])
    drifted = sorted({k for p in prints[1:] for k in first if p[k] != first[k]})
    return {
        "kernel": "quack.rmsnorm.rmsnorm_bwd (cutedsl)",
        "shape": [M, N],
        "dtype": str(dtype).replace("torch.", ""),
        "replays": N_REPLAYS,
        "outputs_are_freshly_allocated_each_call": True,
        "so_this_does_not_test": (
            "reuse of a dw_partial buffer across calls -- rmsnorm_bwd "
            "allocates one per call, so no such reuse occurs here"
        ),
        "all_replays_bit_identical": stable,
        "tensors_that_drifted": drifted,
        "restore_value_would_be_a_noop": stable,
        "first_replay_fingerprints": first,
    }


def check_bwd_shared(M, N, dtype):
    """Drive _rmsnorm_bwd directly, reusing ONE dw_partial across replays.

    This is the check the previous docstring described but did not run. It
    calls the custom op underneath the wrapper, so the output buffers are
    mine and I can hand it the same dw_partial / dx every time. If
    ``copy(tXrdW, tXgdW)`` at :1175/:1200 were a read-modify-write rather than
    a store, dw_partial would drift across replays from the same inputs.

    It also pre-fills dw_partial and dx with a recognisable sentinel. A pure
    store overwrites every element it owns; an accumulate would carry the
    sentinel into the result. So this distinguishes "wrote all of it" from
    "added to what was there", which replay-stability alone cannot.
    """
    from quack.rmsnorm import _rmsnorm_bwd, get_sm_count

    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    dout = torch.randn(M, N, device="cuda", dtype=dtype)
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-6)

    sm_count = get_sm_count(N, x.device)
    dx = torch.empty_like(x)
    dw_partial = torch.empty((sm_count, N), device=x.device, dtype=torch.float32)

    SENTINEL = 12345.0
    prints = []
    for _ in range(N_REPLAYS):
        dw_partial.fill_(SENTINEL)
        dx.fill_(0)
        _rmsnorm_bwd(x, w, dout, rstd, dx, dw_partial, None, None, None, sm_count)
        torch.cuda.synchronize()
        prints.append(
            {
                "dx": fp(dx),
                "dw_partial": fp(dw_partial),
                "x": fp(x),
                "dout": fp(dout),
                "rstd": fp(rstd),
            }
        )
    first = prints[0]
    stable = all(p == first for p in prints[1:])
    drifted = sorted({k for p in prints[1:] for k in first if p[k] != first[k]})
    sentinel_survived = bool((dw_partial == SENTINEL).any().item())
    return {
        "kernel": "quack.rmsnorm._rmsnorm_bwd (cutedsl, caller-owned buffers)",
        "shape": [M, N],
        "dtype": str(dtype).replace("torch.", ""),
        "replays": N_REPLAYS,
        "dw_partial_buffer_reused_across_replays": True,
        "dw_partial_prefilled_with_sentinel": SENTINEL,
        "sentinel_survives_anywhere_in_dw_partial": sentinel_survived,
        "all_replays_bit_identical": stable,
        "tensors_that_drifted": drifted,
        "dw_partial_is_stored_not_accumulated": stable and not sentinel_survived,
        "first_replay_fingerprints": first,
    }


def main():
    import importlib.metadata as md

    props = torch.cuda.get_device_properties(0)
    fwd_plain = [check_fwd(M, N, torch.bfloat16, False) for M, N in SHAPES]
    fwd_alias = [check_fwd(M, N, torch.bfloat16, True) for M, N in SHAPES]
    bwd = [check_bwd(M, N, torch.bfloat16) for M, N in SHAPES]
    bwd_shared = [check_bwd_shared(M, N, torch.bfloat16) for M, N in SHAPES]

    control_worked = all(not r["all_replays_bit_identical"] for r in fwd_alias)
    main_stable = (
        all(r["all_replays_bit_identical"] for r in fwd_plain)
        and all(r["all_replays_bit_identical"] for r in bwd)
        and all(r["all_replays_bit_identical"] for r in bwd_shared)
    )
    dw_is_stored = all(r["dw_partial_is_stored_not_accumulated"] for r in bwd_shared)

    out = {
        "probe": Path(__file__).name,
        "source_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "what_this_is": (
            "The cutedsl half of the restore_value verdict, measured on Hopper "
            "hardware instead of argued from the schema. MI355X cannot import "
            "quack.rmsnorm at all (cuda.bindings.driver at :7)."
        ),
        "env": {
            "gpu": props.name,
            "l2_cache_bytes_reported": props.L2_cache_size,
            "l2_is_the_llc_here": True,
            "torch": torch.__version__,
            "cutlass_dsl": md.version("nvidia-cutlass-dsl"),
            "python": sys.executable,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "why_python_is_recorded": (
                "Because I got this wrong once. The interpreter, not the host, "
                "decides whether cutlass is 4.6.1 or 4.5.2 here: hyper00's "
                "/usr/bin/python3 has 4.5.2 and cannot import quack.rmsnorm, "
                "while /root/quack-FlyDSL-h200-test/.venv/bin/python has 4.6.1 "
                "and can. Reporting 'the H200 boxes are blocked' from the "
                "former was a claim about a set other than the one the label "
                "named. Any rerun must state which interpreter it used."
            ),
        },
        "scope": (
            "Kernel mutation only. The COST half of the restore_value verdict "
            "-- the has_hooks gate at autotuner.py:199-204/:363 dropping "
            "scoring from _bench_cuda_graph_l2_rotate onto do_bench -- is a "
            "property of the autotuner and the harness, not of this kernel, "
            "and was measured on MI355X (AI/data/restore_value_needed.json). "
            "It is NOT re-measured here and the two artifacts' timings must "
            "not be compared: different device, different LLC (60 MiB L2 as "
            "true LLC on H200 vs 4 MiB L2 behind a 256 MiB MALL on gfx950), "
            "so the rotation band sits in a different place on each."
        ),
        "check1_idempotence_fwd": fwd_plain,
        "check1_idempotence_bwd": bwd,
        "check1b_bwd_shared_dw_partial": bwd_shared,
        "check2_aliased_control_fwd": fwd_alias,
        "verdict": {
            "control_can_fail": control_worked,
            "kernels_are_idempotent": main_stable,
            "dw_partial_is_stored_not_accumulated": dw_is_stored,
            "restore_value_needed_for_cutedsl_rmsnorm": not main_stable,
            "interpretation": (
                "check 1 stable AND check 2 drifting means the probe is "
                "sensitive and the cutedsl kernels still do not mutate their "
                "inputs -- the source reading of schema :367/:1210 is "
                "confirmed on hardware. check 2 stable would make check 1 "
                "vacuous and this result must then be discarded."
                if control_worked
                else "CONTROL DID NOT DRIFT -- check 1 proves nothing here."
            ),
            "what_this_still_does_not_establish": (
                "Aliasing. @Reviewer (b3053d0a) is right that every call site "
                "checked so far hands these kernels freshly allocated, "
                "non-aliased outputs, and that _clone_l2_rotate_inputs "
                "(bench_utils.py:174-176) clones each tensor arg "
                "INDEPENDENTLY, which destroys any alias graph the real call "
                "had. This probe inherits that limitation exactly: check 2 "
                "aliases residual onto residual_out through the public "
                "wrapper, but nothing here exercises out-is-x or "
                "overlapping-view arguments to the tuned entry points. So the "
                "no-restore result is conditional on no aliasing, and that "
                "condition belongs in the design boundary of any future "
                "production tuned wrapper rather than being extrapolated from "
                "today's fresh-allocation habit."
            ),
        },
    }
    print(json.dumps(out, indent=2))
    Path(sys.argv[1] if len(sys.argv) > 1 else "cutedsl_restore.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

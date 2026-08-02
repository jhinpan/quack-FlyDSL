"""Does rmsnorm need ``restore_value``, and what would turning it on cost?

Background. @Autotune (`b12b184f` follow-up) found that ``quack/autotuner.py``'s
``restore_value=`` is not free: setting it installs ``pre_hook``/``post_hook``
(:136/:145), and BOTH sites that choose the benchmark path gate on
``has_hooks`` --

    :199-204  in ``_bench``:    use_l2_cold = (self._do_bench is None
                                               and ... and not has_hooks)
    :363      in ``__call__``:  if self._do_bench is None and not has_hooks:

-- so enabling ``restore_value`` silently drops autotune scoring from the
L2-cold CUDA-graph rotation path onto ``partial(triton.testing.do_bench,
warmup=5, rep=25)``. That is not a small accuracy delta. ``bench_utils.py:98``
states the direction as a known bias: "the L2-hot number favours wider layouts
/ deeper smem stages that don't actually win once data has to come from HBM."
The harness side (``results.csv``: ``rotation_buffers``,
``l2_eviction_between_calls``) reports the L2-cold regime, so flipping only the
selection side manufactures a regime mismatch between how configs are CHOSEN
and how they are REPORTED.

That much is settled by reading. What reading alone does NOT settle is whether
rmsnorm needs ``restore_value`` at all -- and if it does not, the whole
tradeoff is moot rather than merely unfavourable. This probe settles that on
hardware, and measures what the switch would cost if someone turned it on
anyway.

``restore_value`` exists to undo in-place input mutation between config
evaluations. It is load-bearing only if replaying the kernel on the SAME
buffers changes the result -- i.e. if some argument is both read and written.
The source says it is not: every mutated name in either custom-op schema is a
distinct output buffer,

    fwd  (:367)   mutates out, rstd, mean, residual_out
    bwd  (:1210)  mutates dx, dw_partial, db_partial, dresidual

while the inputs it reads (``residual``, ``dresidual_out``) carry no ``(a!)``
alias annotation, and the only global write of the accumulator-shaped tensor is
``copy(tXrdW, tXgdW)`` (:1175, :1200) -- a rmem->gmem store, not a
read-modify-write; the cross-row reduction it feeds goes through smem and never
re-reads global ``dw_partial``. Consistent with that, every allocation site
uses ``torch.empty``, never ``torch.zeros`` (the only ``zeros_like`` in the
file are the post-reduction ``dw``/``db``, outside the tuned kernels).

CHECK 1 (idempotence) tests that claim directly: run the kernel N times on the
same buffers and compare every output against the first run bit-for-bit. If
outputs are stable, no argument accumulates, and ``restore_value`` is a no-op
for rmsnorm -- there is nothing to restore.

CHECK 2 (aliased control) is the falsifier for check 1. A pure-overwrite kernel
is idempotent for the trivial reason that it never reads its outputs; to show
check 1 can FAIL rather than being vacuously true, we re-run with the residual
input deliberately aliased onto ``residual_out``, which is exactly the
read-and-write pattern ``restore_value`` is for. If check 2 also came back
"stable", check 1 would prove nothing about the probe's sensitivity.

CHECK 3 (the cost) measures both benchmark paths on the same config to put a
number on the regime switch: ``_bench_cuda_graph_l2_rotate`` (what autotune
uses today) vs ``partial(do_bench, warmup=5, rep=25)`` (what it would use the
moment ``restore_value`` is set). It runs the full 2x2 -- rotate/single x
graph/do_bench -- because the two paths differ in launch MECHANISM as well as
cache state, and comparing only the corners confounds them. On this device the
mechanism term turns out to carry essentially the whole gap, and the rotation
term is ~1.00x, which this probe reports without claiming to explain: see
``the_rotation_term_is_~1.00_and_this_probe_does_not_explain_it`` in the
output. Note the cache of interest on gfx950 is the 256 MiB MALL, not the
4 MiB L2 that ``torch.cuda.get_device_properties()`` reports -- reasoning from
the latter is defect 1 of ``AI/gfx950_mall_evictor_defect.md``, and an earlier
version of this file did exactly that.

WHICH KERNEL THIS ACTUALLY TESTS, AND WHICH IT DOES NOT. The two decorators at
issue -- ``rmsnorm_fwd_tuned`` (rmsnorm.py:514) and ``rmsnorm_bwd_tuned``
(:1530) -- live in ``quack/rmsnorm.py``, which is the cutedsl path and imports
``cuda.bindings.driver`` at module scope (:7). That import fails on MI355X, so
on this box those two decorators cannot be executed at all. Running this probe
here therefore CANNOT directly demonstrate idempotence of the cutedsl kernels
that ``restore_value`` would actually be attached to. Saying otherwise would be
the exact label-vs-set error this file is trying to avoid: the measurement
would be of the FlyDSL kernel while the claim named the cutedsl one.

What this probe does on ROCm is the pair of things that ARE in scope here:

  (a) the idempotence check against the FlyDSL rmsnorm, whose custom ops
      declare the same pure-output mutation structure
      (``rmsnorm_flydsl.py:400`` mutates out/residual_out/rstd; ``:619``
      mutates dx/dresidual/dweight/dbias), so a drift here would be evidence
      the structural argument is wrong for BOTH paths; and

  (b) the regime-gap measurement, which is a property of
      ``quack/bench/bench_utils.py`` -- shared, imports fine on ROCm, and is
      what ``autotuner.py`` would switch away from.

(a) is corroboration, not proof, for the cutedsl decorators. The proof for
those needs an H100/H200 run; ``verdict.cutedsl_status`` records that gap
explicitly rather than letting the ROCm result stand in for it.

Scope. This probe does NOT modify ``quack/autotuner.py``; that file is under
@Reviewer's `dbab028` hold and is @Autotune's to change. It only reads, and
runs rmsnorm through its public custom ops. Timing here is device-side, so the
+-1 us host-dispatch bar from ``probe_rmsnorm_stage_stubs.py`` does NOT apply
(see that file's ``this_does_NOT_generalize_to_device_side_timing``).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = REPO / "AI" / "data" / "restore_value_needed.json"

N_REPLAYS = 8
SHAPES = [(32768, 4096), (8192, 2048)]

# gfx950 last-level cache. NOT torch's L2_cache_size (4 MiB), which is the
# per-XCD L2 and is the fallback value AI/gfx950_mall_evictor_defect.md
# identifies as defect 1. Hardcoded here for the same reason that document
# hardcodes it, and flagged as hardcoded in the output.
MALL_BYTES = 256 * 2**20


def _fingerprint(t):
    """Exact bit-level fingerprint; NaN-safe (view as int bits)."""
    if t is None:
        return None
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def check_idempotence(M, N, dtype, alias_residual: bool):
    """Replay the FlyDSL rmsnorm N_REPLAYS times on the SAME buffers.

    alias_residual=True feeds the previous output back in as the residual
    input, creating a genuine read-and-write dataflow -- the control that
    proves this check is capable of failing. Without it, "stable" would be
    unfalsifiable.
    """
    from quack.rmsnorm_flydsl import rmsnorm

    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(M, N, device=dev, dtype=dtype)
    w = torch.randn(N, device=dev, dtype=dtype)
    residual = torch.randn(M, N, device=dev, dtype=dtype)

    prints = []
    carry = residual
    for _ in range(N_REPLAYS):
        out = rmsnorm(x, w, None, carry, eps=1e-6)
        torch.cuda.synchronize()
        if alias_residual:
            # Feed the result back in: now an argument is both read and
            # written across iterations, which is what restore_value exists
            # to undo.
            carry = out
        prints.append(
            {
                "out": _fingerprint(out),
                "x": _fingerprint(x),
                "weight": _fingerprint(w),
                "residual_in": _fingerprint(carry),
            }
        )

    first = prints[0]
    stable = all(p == first for p in prints[1:])
    drifted = sorted({k for p in prints[1:] for k in first if p[k] != first[k]})
    return {
        "kernel": "quack.rmsnorm_flydsl.rmsnorm",
        "shape": [M, N],
        "dtype": str(dtype).replace("torch.", ""),
        "feedback_aliased": alias_residual,
        "replays": N_REPLAYS,
        "all_replays_bit_identical": stable,
        "tensors_that_drifted": drifted,
        "restore_value_would_be_a_noop": stable,
        "first_replay_fingerprints": first,
    }


def check_bench_path_gap(M, N, dtype):
    """Same kernel+config under both paths autotune could pick.

    Path A: _bench_cuda_graph_l2_rotate  (today; has_hooks False)
    Path B: partial(do_bench, warmup=5, rep=25)  (the moment restore_value is set)
    """
    import triton

    from quack.bench.bench_utils import (
        _bench_cuda_graph_l2_rotate,
        _clone_l2_rotate_inputs,
        _pick_l2_rotate_count,
    )
    from quack.rmsnorm_flydsl import rmsnorm

    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(M, N, device=dev, dtype=dtype)
    w = torch.randn(N, device=dev, dtype=dtype)

    def call(x_, w_):
        rmsnorm(x_, w_, eps=1e-6)

    args = (x, w)
    one_t = M * N * torch.tensor([], dtype=dtype).element_size()
    n_bufs = _pick_l2_rotate_count(args, {})
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(args, {}, n_bufs)

    # The two paths autotune can pick differ in TWO ways at once: L2 state
    # (rotating clones vs one buffer) AND launch mechanism (single graph
    # replay of 200 recorded calls vs one event pair per launch, plus a fresh
    # output allocation each iteration). Comparing only A-vs-B confounds them.
    # So measure the 2x2 and report the two effects separately.
    def bench_graph(sets_a, sets_k):
        r = _bench_cuda_graph_l2_rotate(call, sets_a, sets_k, extra_kwargs={})
        return r[0] if isinstance(r, (list, tuple)) else r

    do_bench = partial(triton.testing.do_bench, warmup=5, rep=25)

    def bench_do(rotate: bool):
        if not rotate:
            r = do_bench(lambda: call(*args), quantiles=(0.5, 0.2, 0.8))
        else:
            ctr = {"i": 0}

            def rot():
                i = ctr["i"] % len(arg_sets)
                ctr["i"] += 1
                call(*arg_sets[i])

            r = do_bench(rot, quantiles=(0.5, 0.2, 0.8))
        return r[0] if isinstance(r, (list, tuple)) else r

    graph_rot = [bench_graph(arg_sets, kwarg_sets) for _ in range(5)]
    # Single-set graph bench: same mechanism, no rotation. Deliberately NOT
    # labelled "L2-hot" -- on gfx950 the last-level cache is the 256 MiB MALL,
    # not the 4 MiB L2 that torch reports, so whether either cell is warm is a
    # question about MALL residency. See the note below.
    graph_one = [
        bench_graph([arg_sets[0]] * len(arg_sets), [kwarg_sets[0]] * len(kwarg_sets))
        for _ in range(5)
    ]
    do_rot = [bench_do(rotate=True) for _ in range(5)]
    do_one = [bench_do(rotate=False) for _ in range(5)]

    g_rot, g_one = statistics.median(graph_rot), statistics.median(graph_one)
    d_rot, d_one = statistics.median(do_rot), statistics.median(do_one)

    return {
        "shape": [M, N],
        "dtype": str(dtype).replace("torch.", ""),
        "rotation_buffers": n_bufs,
        "torch_reported_L2_cache_size_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "mall_bytes_assumed": MALL_BYTES,
        "mall_bytes_is_hardcoded": True,
        "one_x_tensor_bytes": one_t,
        "rotation_working_set_bytes": n_bufs * one_t,
        "single_set_fits_mall": one_t <= MALL_BYTES,
        "rotation_set_fits_mall": n_bufs * one_t <= MALL_BYTES,
        "cells_ms": {
            "graph_rotate__what_autotune_uses_today": {"median": g_rot, "runs": graph_rot},
            "graph_single__same_mechanism_no_rotation": {"median": g_one, "runs": graph_one},
            "do_bench_rotate__other_mechanism_rotated": {"median": d_rot, "runs": do_rot},
            "do_bench_single__what_restore_value_switches_to": {"median": d_one, "runs": do_one},
        },
        "effects": {
            "rotation_effect_within_graph__single_over_rotate": g_one / g_rot if g_rot else None,
            "rotation_effect_within_do_bench__single_over_rotate": d_one / d_rot if d_rot else None,
            "mechanism_effect_at_rotate__do_bench_over_graph": d_rot / g_rot if g_rot else None,
            "end_to_end_switch__do_single_over_graph_rotate": d_one / g_rot if g_rot else None,
        },
        "note": (
            "The end-to-end ratio is NOT a cache measurement and must not be "
            "quoted as one. do_bench_single is SLOWER than graph_rotate here, "
            "which looks backwards for a 'warm is faster' story -- because the "
            "mechanism term dominates: do_bench pays one event pair and one "
            "fresh output allocation per launch, while the graph replays 200 "
            "recorded calls with no Python in the window. Also: "
            "bench_utils.py:98 is a claim about which CONFIG wins (relative "
            "ranking across layouts), not about the magnitude of any single "
            "config, so none of these ratios confirm or refute it -- that "
            "needs the whole config set scored under both regimes."
        ),
        "the_rotation_term_is_~1.00_and_this_probe_does_not_explain_it": (
            "CORRECTION of an earlier version of this field, which argued the "
            "null result from torch's L2_cache_size = 4 MiB and concluded 'the "
            "working set blows the cache, rotation cannot make an already-cold "
            "read colder.' That reasoning is wrong. On gfx950 the last-level "
            "cache is the 256 MiB MALL, not the 4 MiB per-XCD L2 (hierarchy: "
            "per-CU L1D -> 4 MiB L2 x8 XCDs -> 256 MiB MALL -> HBM, ROCm Kernel "
            "Wiki hw-chiplet-xcd). Reading L2_cache_size and sizing a cache "
            "argument on it is defect 1 of AI/gfx950_mall_evictor_defect.md, "
            "fixed harness-side in 31c1fd4. Against the right cache the claim "
            "inverts for the small shape: 8192x2048 rotated is 128 MiB, which "
            "FITS the MALL comfortably -- nearly the same 128 MiB example that "
            "document uses -- so rotation there produces no cold read at all. "
            "32768x4096 is the opposite: single sits at 256 MiB (exactly at "
            "capacity) while rotated is 1024 MiB and does cross it, which is "
            "where a contrast should have shown. It did not. This probe "
            "therefore records a ~1.00x within-graph ratio WITHOUT a surviving "
            "explanation. Candidates to separate: rmsnorm at these sizes is "
            "HBM-bound enough that MALL residency moves little; the evictor gap "
            "(defect 1) means neither cell is genuinely warm; or graph_single "
            "is not single-buffer in the way assumed. That is Experiment No.002 "
            "territory, which gfx950_mall_evictor_defect.md blocks pending "
            "re-collection. The MALL size above is hardcoded, not measured -- "
            "same caveat that document flags about its own 256 MiB."
        ),
        "rotation_within_do_bench_is_confounded": (
            "The do_bench rotate cell is NOT a clean contrast: rotation there "
            "is a Python closure doing a modulo and an index per call, inside "
            "the timed region, whereas do_bench_single calls straight through. "
            "Its ratio is that overhead plus allocator behaviour, not a cache "
            "effect. Only the within-graph ratio is like-for-like, because "
            "there the rotation is baked into the recorded graph either way."
        ),
    }


def main():
    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    src = Path(__file__).read_bytes()
    result = {
        "probe": Path(__file__).name,
        "source_sha256_16": hashlib.sha256(src).hexdigest()[:16],
        "device": torch.cuda.get_device_name(0),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "question": (
            "Does rmsnorm need autotuner.restore_value? If not, the L2-cold -> "
            "L2-hot regime switch it forces (via has_hooks at autotuner.py:199 "
            "and :363) is a cost paid for nothing."
        ),
        "idempotence": [],
        "aliased_control": [],
        "bench_path_gap": [],
    }

    for M, N in SHAPES:
        result["idempotence"].append(check_idempotence(M, N, torch.bfloat16, alias_residual=False))

    result["aliased_control"].append(
        check_idempotence(SHAPES[0][0], SHAPES[0][1], torch.bfloat16, alias_residual=True)
    )

    for M, N in SHAPES:
        result["bench_path_gap"].append(check_bench_path_gap(M, N, torch.bfloat16))

    clean = all(r["all_replays_bit_identical"] for r in result["idempotence"])
    control_moved = not result["aliased_control"][0]["all_replays_bit_identical"]

    result["verdict"] = {
        "kernel_measured": "quack.rmsnorm_flydsl.rmsnorm (ROCm)",
        "flydsl_outputs_stable_under_replay": clean,
        "aliased_control_did_move": control_moved,
        "control_is_meaningful": control_moved,
        "restore_value_needed_for_flydsl_rmsnorm": not clean,
        "cutedsl_status": (
            "NOT MEASURED HERE. rmsnorm_fwd_tuned (rmsnorm.py:514) and "
            "rmsnorm_bwd_tuned (:1530) are the decorators restore_value would "
            "attach to, and quack/rmsnorm.py imports cuda.bindings.driver at "
            "module scope (:7), which does not exist on MI355X. The argument "
            "that they need no restore is source-level only: schema :367 and "
            ":1210 mutate output buffers exclusively, inputs carry no (a!) "
            "alias, dw_partial is stored via copy(tXrdW, tXgdW) at :1175/:1200 "
            "rather than accumulated, and every allocation is torch.empty. "
            "Confirming it on hardware needs an H100/H200 run of this probe "
            "against quack.rmsnorm._rmsnorm_fwd."
        ),
        "reading": (
            "If outputs are stable AND the aliased control moved, the check is "
            "sensitive and the FlyDSL rmsnorm genuinely has nothing to "
            "restore. Combined with restore_value having zero callers "
            "tree-wide (inherited whole in b69bb67) and the identical "
            "pure-output mutation structure on the cutedsl side, that makes "
            "'set restore_value on the rmsnorm decorators' a change that buys "
            "no correctness while costing the measurement regime. If the "
            "control did NOT move, this probe proves nothing -- the check "
            "would be vacuous and the stable result uninformative."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["verdict"], indent=2))
    for g in result["bench_path_gap"]:
        c, e = g["cells_ms"], g["effects"]
        print(f"  {g['shape']} (bufs={g['rotation_buffers']}):")
        for k, v in c.items():
            print(f"      {v['median']:.4f} ms  {k}")
        print(
            f"      -> rotation within graph "
            f"{e['rotation_effect_within_graph__single_over_rotate']:.3f}x, "
            f"within do_bench {e['rotation_effect_within_do_bench__single_over_rotate']:.3f}x, "
            f"mechanism {e['mechanism_effect_at_rotate__do_bench_over_graph']:.3f}x"
        )
        print(
            f"      -> rotation set {g['rotation_working_set_bytes'] / 2**20:.1f} MiB, "
            f"fits 256 MiB MALL: {g['rotation_set_fits_mall']}"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

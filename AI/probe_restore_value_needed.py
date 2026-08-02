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
term is ~1.00x -- because BOTH shapes benchmarked here sit on the same side of
the MALL boundary in the rotated and the single cell alike, so neither cell is
a contrast. Rotation can only bite when ``single_set <= MALL < rotation_set``;
a scan across that band confirms it on eight points and the two edges. See
``why_the_rotation_term_is_~1.00_here``. Note the cache of interest on gfx950
is the 256 MiB MALL, not the 4 MiB L2 that
``torch.cuda.get_device_properties()`` reports -- reasoning from the latter is
defect 1 of ``AI/gfx950_mall_evictor_defect.md``, and an earlier version of
this file did exactly that. The working-set figures below are MEASURED (the
probe records every output's ``data_ptr()``), because two earlier versions got
them wrong in opposite directions by inferring from source.

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

    # Record the output's address on every call so the working set is MEASURED
    # rather than inferred from the presence of ``torch.empty`` in the source.
    # @Reviewer (dbe0206d) was right to demand this: under graph capture the
    # per-slot output address is an allocator question, not a source question.
    out_ptrs: list[int] = []

    def call(x_, w_):
        out_ptrs.append(rmsnorm(x_, w_, eps=1e-6).data_ptr())

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

    def count_out_bufs(sets_a, sets_k):
        """Distinct output buffers inside the TIMED region of one cell.

        Scoped twice over, because two earlier attempts each counted a set
        other than the one the name claims:

        1. Letting ``out_ptrs`` accumulate across all four cells and every
           repeat reported the grand total (8) as if it described one cell.
        2. Counting one whole ``bench_graph`` call reported 2 -- the eager
           warmup allocates from the normal caching-allocator pool and the
           capture from the graph's private pool, so the two differ. But the
           timed window is the ``replay()`` alone, so the warmup buffer is not
           in the set being asked about.

        This counts a capture of the same round-robin, after eager priming,
        which is exactly what the timed replay re-executes.
        """
        for i in range(2 * len(sets_a)):
            call(*sets_a[i % len(sets_a)], **sets_k[i % len(sets_k)])
        torch.cuda.synchronize()
        out_ptrs.clear()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(len(sets_a)):
                call(*sets_a[i], **sets_k[i])
        torch.cuda.synchronize()
        n = len(set(out_ptrs))
        out_ptrs.clear()
        del g
        return n

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

    # Measure the output-buffer count for the rotate cell specifically, before
    # the timing runs, so the figure describes that cell and nothing else.
    n_out_bufs = count_out_bufs(arg_sets, kwarg_sets)

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

    # ``n_out_bufs`` is measured above, scoped to the rotate cell alone.
    rotate_set = n_bufs * one_t + n_out_bufs * one_t
    single_set = one_t + n_out_bufs * one_t

    return {
        "shape": [M, N],
        "dtype": str(dtype).replace("torch.", ""),
        "rotation_buffers": n_bufs,
        "torch_reported_L2_cache_size_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "mall_bytes_assumed": MALL_BYTES,
        "mall_bytes_is_hardcoded": True,
        "one_x_tensor_bytes": one_t,
        # BOTH earlier denominators were wrong, in opposite directions.
        # v1 counted only cloned inputs (n*t) and missed the output entirely.
        # v2 "corrected" it to 2*n*t, assuming each rotation slot carries its
        # own output. Neither was measured. ``distinct_output_buffers`` below
        # is: the caller drops each returned tensor, so the caching allocator
        # reuses one block -- inputs rotate over n, outputs over 1. The true
        # set is (n+1)*t rotated and 2*t single, so v2 overstated the rotated
        # set by 1.6x at n=4. Provenance is recorded per figure.
        "distinct_output_buffers_measured": n_out_bufs,
        "output_buffers_rotate_with_inputs": n_out_bufs > 1,
        "cloned_input_bytes": n_bufs * one_t,
        "rotation_working_set_bytes_measured": rotate_set,
        "single_working_set_bytes_measured": single_set,
        "single_set_fits_mall": single_set <= MALL_BYTES,
        "rotation_set_fits_mall": rotate_set <= MALL_BYTES,
        # The only configuration in which rotation can change cache state at
        # all: the single set stays inside the MALL while the rotated set does
        # not. Outside this band, both cells are on the same side of capacity
        # and the ratio should be ~1.00x -- which is where both of this probe's
        # shapes happen to sit (see the band-scan field below).
        "rotation_can_matter_here": single_set <= MALL_BYTES < rotate_set,
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
        "why_the_rotation_term_is_~1.00_here": (
            "Third version of this field. v1 argued the null result from "
            "torch's L2_cache_size = 4 MiB -- wrong cache, and precisely "
            "defect 1 of AI/gfx950_mall_evictor_defect.md (gfx950 is per-CU "
            "L1D -> 4 MiB per-XCD L2 x8 -> 256 MiB MALL -> HBM, ROCm Kernel "
            "Wiki hw-chiplet-xcd). v2 retracted that and reported NO surviving "
            "explanation. The explanation now on record is measured, not "
            "argued, and it is simpler than any of v2's three candidates: "
            "BOTH shapes this probe benchmarks sit on the same side of the "
            "MALL boundary in the rotated and single cells alike, so neither "
            "is a contrast. With inputs rotating over n=4 buffers and the "
            "output over exactly 1 (measured above, not inferred), the sets "
            "are (n+1)*t rotated vs 2*t single, and rotation can only change "
            "cache state when 2*t <= 256 MiB < (n+1)*t -- i.e. t in "
            "(51.2, 128] MiB. 8192x2048 has t=32 MiB (both sets fit); "
            "32768x4096 has t=256 MiB (neither fits). Both predict ~1.00x, "
            "and both measure it. A scan over t = 32/48/56/64/96/128/160/192 "
            "MiB at N=1024 agrees on all eight points: 0.992, 1.010 outside "
            "the low edge; 0.932, 0.862, 0.864, 0.862 inside; 0.992, 1.003 "
            "outside the high edge. So rotation IS doing what it was built to "
            "do -- this probe just picked two shapes where it cannot show. "
            "Caveats that keep this short of settled: the MALL size is "
            "hardcoded here, not measured (the same caveat that document "
            "flags about its own 256 MiB); and a tight scan across the edges "
            "(t = 50/51/52/53 and 127/129/132/136 MiB) shows the transition "
            "is SOFT, ~0.95x and ~0.89x just outside where the model says "
            "1.00x, so the capacity rule predicts the band but not a step at "
            "its edges. Sharpening that is Experiment No.002 territory, which "
            "gfx950_mall_evictor_defect.md blocks pending re-collection."
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
            "NOT MEASURED HERE, MEASURED SEPARATELY -- see "
            "AI/data/cutedsl_restore_value.json. This box cannot do it: "
            "rmsnorm_fwd_tuned (rmsnorm.py:514) and rmsnorm_bwd_tuned (:1530) "
            "are the decorators restore_value would attach to, and "
            "quack/rmsnorm.py imports cuda.bindings.driver at module scope "
            "(:7), which does not exist on MI355X. The source argument was: "
            "schema :367 and :1210 mutate output buffers exclusively, inputs "
            "carry no (a!) alias, dw_partial is stored via copy(tXrdW, tXgdW) "
            "at :1175/:1200 rather than accumulated, and every allocation is "
            "torch.empty. AI/probe_cutedsl_restore_value.py then ran the same "
            "two checks against quack.rmsnorm on an H200 (hyper00 GPU 0, "
            "cutlass 4.6.1, tree 4f36477): fwd and bwd bit-identical across 5 "
            "replays at 8192x2048 and 4096x4096, aliased control drifted at "
            "both. Verdict confirmed on hardware, not just read off the "
            "schema. TWO EARLIER VERSIONS OF THIS FIELD WERE WRONG: the first "
            "implied the run was merely waiting on someone's time; the second "
            "said the remedy was UNAVAILABLE because both H200 boxes carry "
            "cutlass 4.5.2 against a ==4.6.1 pin. The 4.5.2 reading came from "
            "each box's SYSTEM python. hyper00 also has a 4.6.1 venv and "
            "hyper01 a 4.6.0 one; under those, current head imports fine. "
            "Machine-scoped claim, interpreter-scoped fact -- this file's own "
            "defect class, committed inside a description of it."
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
            f"      -> sets (measured, {g['distinct_output_buffers_measured']} out buf): "
            f"single {g['single_working_set_bytes_measured'] / 2**20:.0f} MiB, "
            f"rotate {g['rotation_working_set_bytes_measured'] / 2**20:.0f} MiB; "
            f"fits MALL {g['single_set_fits_mall']}/{g['rotation_set_fits_mall']}; "
            f"rotation can matter here: {g['rotation_can_matter_here']}"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

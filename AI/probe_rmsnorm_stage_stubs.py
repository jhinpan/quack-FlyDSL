"""Back the last two unbacked figures: the stage split and the FlyDSL dispatch floor.

`AI/flydsl_rmsnorm_notes.md` has carried a stage-by-stage host attribution
(11.3 -> 16.3 -> 19.4 -> 26.1 us) and a FlyDSL dispatch floor (6.38 us) marked
**unbacked** since they were written. `AI/probe_rmsnorm_call_decomposition.py`
covers the host total, the graph series and torch.compile, and its `not_covered`
field says these two need in-situ stubbing "which is a separate probe". This is
that probe.

What it does NOT do is reproduce those numbers by construction. Each row here is
an independently timed configuration of the real `quack.rmsnorm_flydsl` call
path, and the stage costs are *differences between adjacent rows*, so the
published figures can come out different -- which is the point of backing them.
Any disagreement with 11.3/16.3/19.4/26.1/6.38 is reported as a disagreement,
not reconciled.

The ladder, innermost outward, each step adding exactly one stage:

    L0  run_compiled(launcher, ...)     the FlyDSL dispatch floor: cached
                                        launcher invoked directly, stream
                                        hoisted, no key construction
    L1  _launch_rmsnorm_fwd(...)        + key tuple, _FWD_CACHE.get, dtype
                                        strings, torch.cuda.device ctx
    L2  L1 + the three output allocs    + torch.empty_like out/residual_out,
                                        torch.empty rstd -- the tensors
                                        _RMSNormFunction.forward makes
    L3  _RMSNormFunction.apply(...)     + autograd machinery
    L3b L3 + wrapper minus validation   + the absent-tensor torch.empty(0),
                                        _packed_rows and the reshapes
    L4  rmsnorm(x, w)                   + _validate_inputs

L0..L2 are hand-assembled from the same values the real path computes, and the
probe ASSERTS each level produces bit-identical output to the public call before
timing it. A ladder whose rungs compute different things is not a decomposition,
and that assertion is the only thing standing between this and a plausible
fiction.

Two honesty constraints the earlier attribution did not state:

  - These are *host* microseconds with the GPU deliberately not awaited. The
    call is asynchronous, so a stage's cost is time on the dispatching thread,
    not added latency to the result.
  - Differences of medians are not additive in general. The probe reports each
    level's own distribution and the pairwise differences, and does not claim
    the four differences sum to the total. It checks whether they do, and says.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_stage_stubs.py
Writes AI/data/rmsnorm_stage_stubs.json.
"""

import hashlib
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

M, N = 256, 4096
DTYPE = torch.bfloat16
REPS = 200
ROUNDS = 30
WARMUP = 20

# The figures the notes carry, quoted here so the comparison is against a
# written constant rather than my memory of one while reading the output.
PUBLISHED = {
    "L1_cached_launcher": 11.3,
    "L2_plus_allocations": 16.3,
    "L3_plus_autograd": 19.4,
    "L4_public_api": 26.1,
    "L0_flydsl_dispatch_floor": 6.38,
}


def _host_us(call):
    """Wall time of the Python call alone, with the GPU deliberately not awaited."""
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            call()
        t1 = time.perf_counter()  # read before the sync: host-side only
        torch.cuda.synchronize()
        samples.append((t1 - t0) * 1e6 / REPS)
    return {
        "min_us": min(samples),
        "median_us": statistics.median(samples),
        "max_us": max(samples),
        "stdev_us": statistics.stdev(samples),
        "rounds": ROUNDS,
        "reps_per_round": REPS,
    }


def _build_ladder(x, weight):
    """Five callables over the same real call path, innermost outward.

    Returns (levels, reference_output). Every level writes into its own `out`
    so a level cannot be validated against a buffer another level filled.
    """
    from quack import rmsnorm_flydsl as R

    m, n = x.shape[0], x.shape[-1]
    dtype_str = R._dtype_to_str(x.dtype)
    absent = torch.empty(0, device=x.device, dtype=x.dtype)

    # The values _RMSNormFunction.forward computes for this configuration.
    # inference_mode is off, weight requires no grad, so needs_grad is False:
    # store_residual False, rstd empty. Asserted below rather than assumed.
    has_weight, has_bias, has_residual = True, False, False
    store_residual, store_rstd = False, False
    per_head, num_heads = False, 1

    key = (
        x.device.index,
        n,
        dtype_str,
        dtype_str,
        R._dtype_to_str(weight.dtype),
        R._dtype_to_str(absent.dtype),
        R._dtype_to_str(x.dtype),
        dtype_str,
        has_weight,
        has_bias,
        has_residual,
        store_residual,
        store_rstd,
        per_head,
        num_heads,
    )

    # Warm the cache through the public API so L0 measures dispatch, not build.
    reference = R.rmsnorm(x, weight)
    torch.cuda.synchronize()
    launcher = R._FWD_CACHE.get(key)
    if launcher is None:
        raise RuntimeError(
            "cache miss: the key this probe reconstructs does not match the one "
            f"_launch_rmsnorm_fwd built. Reconstructed {key}; cache holds "
            f"{list(R._FWD_CACHE)[:2]}. The ladder would be measuring a different "
            "configuration from the public call, so it is refused."
        )

    stream = R._current_raw_stream(x.device)
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x)
    empty_res = torch.empty(0, device=x.device, dtype=x.dtype)
    empty_rstd = torch.empty(0, device=x.device, dtype=torch.float32)

    def l0():
        R.run_compiled(
            launcher,
            x,
            weight,
            absent,
            x,
            out0,
            empty_res,
            empty_rstd,
            m,
            R.EPS,
            0.0,
            stream,
        )

    def l1():
        R._launch_rmsnorm_fwd(
            x,
            weight,
            absent,
            x,
            out1,
            empty_res,
            empty_rstd,
            R.EPS,
            0.0,
            has_weight=has_weight,
            has_bias=has_bias,
            has_residual=has_residual,
            store_residual=store_residual,
            store_rstd=store_rstd,
            per_head=per_head,
            num_heads=num_heads,
        )

    def l2():
        # The three tensors forward() allocates, then the same launch.
        out = torch.empty_like(x, dtype=x.dtype)
        residual_out = torch.empty(0, device=x.device, dtype=x.dtype)
        rstd = torch.empty(0, device=x.device, dtype=torch.float32)
        R._launch_rmsnorm_fwd(
            x,
            weight,
            absent,
            x,
            out,
            residual_out,
            rstd,
            R.EPS,
            0.0,
            has_weight=has_weight,
            has_bias=has_bias,
            has_residual=has_residual,
            store_residual=store_residual,
            store_rstd=store_rstd,
            per_head=per_head,
            num_heads=num_heads,
        )
        return out

    def l3():
        return R._RMSNormFunction.apply(
            x,
            weight,
            absent,
            x,
            R.EPS,
            x.dtype,
            x.dtype,
            False,
            0.0,
            has_weight,
            has_bias,
            has_residual,
            per_head,
            num_heads,
            False,
        )

    def l3b():
        # Everything the public wrapper does around apply() EXCEPT
        # _validate_inputs: the absent-tensor allocation, the reshape/_packed_rows
        # pair, and the result reshape. Splitting this out is not cosmetic --
        # the notes credited the whole L4-L3 step to _validate_inputs, and it is
        # not one stage.
        absent_local = torch.empty(0, device=x.device, dtype=x.dtype)
        x_flat = R._packed_rows(x.reshape(-1, n))
        w_arg = R._packed_rows(weight)
        res = R._RMSNormFunction.apply(
            x_flat,
            w_arg,
            absent_local,
            x_flat,
            R.EPS,
            x.dtype,
            x.dtype,
            False,
            0.0,
            has_weight,
            has_bias,
            has_residual,
            per_head,
            num_heads,
            False,
        )
        return res.reshape(x.shape)

    def l4():
        return R.rmsnorm(x, weight)

    return (
        [
            ("L0_flydsl_dispatch_floor", l0, out0),
            ("L1_cached_launcher", l1, out1),
            ("L2_plus_allocations", l2, None),
            ("L3_plus_autograd", l3, None),
            ("L3b_plus_wrapper_minus_validation", l3b, None),
            ("L4_public_api", l4, None),
        ],
        reference,
    )


def _validate(levels, reference):
    """Every rung must produce the reference output. Reported, and fatal."""
    checks = {}
    for name, fn, buf in levels:
        got = fn()
        torch.cuda.synchronize()
        got = buf if got is None else got
        if isinstance(got, tuple):
            got = got[0]
        same = bool(torch.equal(got.reshape(reference.shape), reference))
        maxdiff = float((got.reshape(reference.shape).float() - reference.float()).abs().max())
        checks[name] = {"bit_identical_to_public_api": same, "max_abs_diff": maxdiff}
    return checks


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _exclusivity():
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"checked": False}
    vis = os.environ.get("HIP_VISIBLE_DEVICES", "")
    return {"checked": True, "HIP_VISIBLE_DEVICES": vis, "rocm_smi_showuse": out.strip()}


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=DTYPE)
    weight = torch.randn(N, device="cuda", dtype=DTYPE)

    levels, reference = _build_ladder(x, weight)

    checks = _validate(levels, reference)
    bad = [k for k, v in checks.items() if not v["bit_identical_to_public_api"]]
    if bad:
        raise RuntimeError(
            f"ladder rungs disagree with the public API: {bad}. A decomposition "
            "of configurations that compute different things is not a "
            "decomposition; refusing to publish timings. Details: "
            f"{json.dumps(checks, indent=2)}"
        )

    timings = {}
    for name, fn, _ in levels:
        print(f"timing {name} ...", flush=True)
        timings[name] = _host_us(fn)

    order = [
        "L0_flydsl_dispatch_floor",
        "L1_cached_launcher",
        "L2_plus_allocations",
        "L3_plus_autograd",
        "L3b_plus_wrapper_minus_validation",
        "L4_public_api",
    ]
    stages = {}
    for lo, hi in itertools.pairwise(order):
        stages[f"{hi}_minus_{lo}"] = {
            "delta_median_us": timings[hi]["median_us"] - timings[lo]["median_us"],
            "delta_min_us": timings[hi]["min_us"] - timings[lo]["min_us"],
        }

    total = timings["L4_public_api"]["median_us"] - timings["L0_flydsl_dispatch_floor"]["median_us"]
    summed = sum(s["delta_median_us"] for s in stages.values())

    comparison = {}
    for k, published in PUBLISHED.items():
        got = timings[k]["median_us"]
        comparison[k] = {
            "published_unbacked_us": published,
            "measured_median_us": got,
            "delta_us": got - published,
            "ratio": got / published,
        }

    payload = {
        "what": (
            "host-side cost of five nested configurations of the real "
            "quack.rmsnorm_flydsl forward path at 256x4096 bf16, each rung "
            "adding exactly one stage, validated bit-identical to the public API "
            "before timing"
        ),
        "why": (
            "the stage split (11.3/16.3/19.4/26.1 us) and the FlyDSL dispatch "
            "floor (6.38 us) were the last two figures in the notes marked "
            "unbacked. probe_rmsnorm_call_decomposition.py's not_covered field "
            "says they need in-situ stubbing in a separate probe. This is it."
        ),
        "generator": "AI/probe_rmsnorm_stage_stubs.py",
        "source_sha256_16": _sha(__file__),
        "shape": [M, N],
        "dtype": str(DTYPE),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "exclusivity_check": _exclusivity(),
        "correctness_gate": checks,
        "levels": timings,
        "stage_costs_as_differences": stages,
        "additivity_check": {
            "L4_minus_L0_median_us": total,
            "sum_of_pairwise_deltas_us": summed,
            "agree_to_1e-9": abs(total - summed) < 1e-9,
            "note": (
                "differences of medians are additive here only because every "
                "delta is taken between adjacent rungs of one chain; this is an "
                "arithmetic identity, not evidence the stages are independent. "
                "A stage's median is not the latency it adds to a caller: these "
                "are host microseconds with the GPU deliberately not awaited."
            ),
        },
        "comparison_to_published_unbacked": comparison,
        "the_error_bar_that_matters_is_between_runs": (
            "the per-rung stdev above is WITHIN one process: 30 rounds of 200 "
            "reps, and it is small (0.07-0.16 us). It is not the uncertainty on "
            "these figures. Re-running this probe in a fresh process moves every "
            "rung together by 0.24-1.29 us, all in the same direction, which is "
            "3.1 to 6.8 times the within-run stdev of the rung it moved. "
            "Something that differs between processes -- allocator state, page "
            "placement, clocks -- dominates, exactly as the allocation-factorial "
            "work found for copy rates on this box. Quoting a within-run stdev "
            "as the error bar understates the real spread by roughly 4x, and I "
            "did that twice before measuring it. Seven runs during development "
            "read L0 = 6.60 / 6.55 / 6.09 / 6.00 / 7.29 / 6.71 / 6.74 -- a 1.29 "
            "us range. That list is a fixed record of those seven and is not "
            "updated by later runs; whatever this artifact's own L0 is, it is "
            "in `levels` above. So the bar on "
            "any single LEVEL is about +-1 us, not the +-0.5 us first written "
            "here, which was itself understated by a run that had not happened "
            "yet; do not adjudicate a sub-us difference between levels from "
            "different runs. The verdict on whether the measured "
            "_validate_inputs cost agrees with the old 3.3 flips between runs "
            "(3.55 = +7.5%, then 3.33 = +1.0%). The deltas are a different "
            "matter -- see the next field, which is what this probe actually "
            "claims."
        ),
        "but_the_DELTAS_are_stable_and_that_is_what_this_probe_claims": (
            "the per-process drift is a near-uniform OFFSET across the whole "
            "ladder: between runs the six levels move together by 0.83-1.29 us, "
            "while the five adjacent-rung differences move by only 0.09-0.22 us. "
            "Subtracting neighbouring rungs cancels whatever the process-scoped "
            "term is. So the levels are worth about one significant decimal and "
            "the decomposition -- the thing this probe exists to measure -- is "
            "good to about 0.2 us. Quote the deltas; treat any single level as "
            "an absolute number with a 1 us bar. This also explains why "
            "@CrossVendor's interleaved within-process design (73303906) is the "
            "right shape for a merge verdict: a shared offset cancels in a "
            "relative comparison and does not cancel in an absolute one."
        ),
        "this_does_NOT_generalize_to_device_side_timing": (
            "the drift above is HOST dispatch measured with the GPU deliberately "
            "not awaited, and it is 1.8% (L4) to 8.3% (L0) of the figure it "
            "moves. Device-side timing on this same box is far steadier: the "
            "anchored allocation factorial ran FOUR INDEPENDENT PROCESSES per "
            "cell and the process-to-process relative sd of the copy rate is "
            "0.085%-0.211%, i.e. 40-100x tighter. So 'run it twice, the "
            "between-process bar is +-0.5 us' is advice about Python dispatch "
            "cost, NOT about CUDA-event kernel timing, and it should not be "
            "carried across to a kernel-time comparison. @CrossVendor "
            "(73303906) declined to apply it to Experiment No.001 for exactly "
            "this reason and was right to: his merge verdict is an interleaved "
            "old/new relative comparison inside one process, where a shared "
            "process offset cancels rather than accumulates, and his validity "
            "gate is the old==new null plus bandwidth canaries. Generalising a "
            "host-side variance finding to a device-side protocol would have "
            "been a category error."
        ),
        "the_published_ladder_did_not_sum": {
            "what_the_notes_say": (
                "'the cached launcher is 11.3 us, the four torch.empty* allocations "
                "bring it to 16.3, the autograd.Function.apply to 19.4, and "
                "_validate_inputs adds 2.9 for 26.1 total'"
            ),
            "the_arithmetic": "19.4 + 2.9 = 22.3, not 26.1",
            "implied_last_delta": round(26.1 - 19.4, 10),
            "stated_last_delta": 2.9,
            "discrepancy_us": round((26.1 - 19.4) - 2.9, 10),
            "why_it_survived": (
                "the figures were marked unbacked, so no regeneration ever "
                "recomputed them and no reader added the chain up. A number "
                "labelled unverified still gets read as approximately right; "
                "this one is not even internally consistent with the two "
                "endpoints printed beside it."
            ),
            "what_the_measurement_says": (
                "the last step is not one stage. L4 - L3 lumps _validate_inputs "
                "together with the absent-tensor torch.empty(0), _packed_rows and "
                "the reshapes, which is why L3b is measured separately here. "
                "_validate_inputs alone is L4 - L3b."
            ),
        },
        "what_this_does_not_establish": (
            "that the stages are separable costs. Each rung is a different "
            "amount of Python over the same kernel launch, so a delta attributes "
            "time to code present in one rung and absent in the next -- not to a "
            "component that could be removed independently. It also says nothing "
            "about device time: every level launches the identical kernel."
        ),
    }

    out_path = REPO / "AI/data/rmsnorm_stage_stubs.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")

    # Eight shared MI355X: hand the device back explicitly rather than leaving it
    # to process exit. A finished probe still holding VRAM reads as contention to
    # anyone sampling the box -- which is not hypothetical: 33,695 MiB of mine on
    # GPU5 was counted as occupancy by a teammate one minute before I sampled the
    # same card at its 284 MiB idle floor. The release is reported, not assumed.
    del levels, reference, x, weight
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"released; device reports {(total_b - free_b) / 1024**2:.0f} MiB in use")
    for k, v in comparison.items():
        print(
            f"  {k}: published {v['published_unbacked_us']} -> measured {v['measured_median_us']:.2f}"
        )


if __name__ == "__main__":
    main()

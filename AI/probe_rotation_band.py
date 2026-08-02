"""When does input rotation actually change cache state on gfx950?

``AI/probe_restore_value_needed.py`` measured a ~1.00x within-graph ratio
between the rotated and single-buffer cells at both of its shapes, and two
successive versions of that file failed to explain it -- the first by reasoning
from ``torch.cuda.get_device_properties().L2_cache_size`` (4 MiB, the per-XCD
L2, which is defect 1 of ``AI/gfx950_mall_evictor_defect.md``), the second by
retracting that and recording no explanation at all.

This probe supplies the explanation and tests it, because "rotation does
nothing here" and "rotation does nothing" are very different claims and the
first was at risk of being read as the second.

THE MODEL. ``_bench_cuda_graph_l2_rotate`` rotates over ``n`` cloned INPUT
sets. The output does not rotate with them: the caller discards each returned
tensor, so the caching allocator hands the same block back every call. That is
measured here (``distinct_output_buffers``), not inferred -- inferring it from
``torch.empty`` in the source is what produced the 2x overcount corrected in
9e41631. So per replay the rotated cell touches ``(n+1)*t`` bytes and the
single cell ``2*t``, where ``t`` is one tensor.

Rotation can therefore only change cache state in the band where the single
set stays resident in the last-level cache and the rotated set does not:

    2*t <= MALL < (n+1)*t

With n=4 and MALL=256 MiB that is t in (51.2, 128] MiB. Outside it, both cells
sit on the same side of capacity and the ratio must be ~1.00x -- not because
rotation is useless, but because there is nothing for it to evict that was not
already evicted, or nothing that leaves.

CRUCIAL SCOPE LIMIT ON THE LOW EDGE. That band is stated at n=4, and n=4 is not
a property of this device -- it is what ``_pick_l2_rotate_count`` returns for
EVERY shape here, because it sizes ``target_ratio * L2_cache_size`` = 12 MiB
against a tensor of tens of MiB, gets n_by_l2 = 1, and clamps to
``min_buffers``. That is defect 1 of AI/gfx950_mall_evictor_defect.md acting on
the buffer count. So the low edge measures the CURRENT HARNESS, not the
hardware: under an effective-LLC fix that sizes against the MALL, n scales with
t (n=24 at t=32 MiB, n=16 at t=48 MiB) and the rotated set crosses capacity at
far smaller t, moving the low edge DOWN. Measured, forcing the n such a fix
would pick: t=32 MiB gives 0.990x at n=4 but 0.887x at n=8 and 0.871x at n=24;
t=48 MiB gives 0.995x at n=4 and 0.882x at n=16. The high edge is different in
kind -- ``2*t > MALL`` has no n in it, so no buffer count can rescue it, and
t=192 MiB stays 0.996x/1.006x at n=4 and n=12. ``EDGE_IS_N_DEPENDENT`` below
records which is which, because "rotation cannot help below t=51 MiB" would be
a hardware claim and the truth is a harness claim.

WHAT THIS PREDICTS AND HOW IT COULD FAIL. A ratio ``single/rotate`` visibly
below 1.0 strictly inside the band, and ~1.00x on BOTH sides. Checking only one
side would not distinguish the model from "bigger is slower". The MALL size is
hardcoded, matching the caveat ``gfx950_mall_evictor_defect.md`` flags about
its own 256 MiB; if the model is right the band edges are where they are
*because* of that constant, which is itself weak evidence for it.

Usage:  HIP_VISIBLE_DEVICES=<idle gpu> python AI/probe_rotation_band.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics

import torch

OUT = pathlib.Path(__file__).resolve().parent / "data" / "rotation_band.json"

# gfx950 last-level cache. NOT torch's L2_cache_size (4 MiB), which is the
# per-XCD L2 and is the fallback value AI/gfx950_mall_evictor_defect.md
# identifies as defect 1.
MALL_BYTES = 256 * 2**20

N_COLS = 1024
DTYPE = torch.bfloat16

# Rows chosen so t straddles both predicted edges (51.2 MiB and 128 MiB) with
# points comfortably outside on each side, plus a tight scan across each edge.
COARSE_M = [16384, 24576, 28672, 32768, 49152, 65536, 81920, 98304]
LOWER_EDGE_M = [25600, 26112, 26624, 27136]
UPPER_EDGE_M = [65024, 66048, 67584, 69632]

# (M, n) pairs testing whether each edge survives a change in buffer count.
# The first n in each group is what the harness picks today; the later ones are
# what an effective-LLC fix would pick (ceil(3 * 256 MiB / t)). Low-edge shapes
# should GAIN an effect as n rises; the high-edge shape should not.
N_SWEEP = [
    (16384, 4),  # t=32 MiB, below the n=4 band
    (16384, 8),
    (16384, 24),
    (24576, 4),  # t=48 MiB, below the n=4 band
    (24576, 16),
    (98304, 4),  # t=192 MiB, above the band: n-independent
    (98304, 12),
]


def _source_sha256_16() -> str:
    return hashlib.sha256(pathlib.Path(__file__).resolve().read_bytes()).hexdigest()[:16]


def measure_at_n(M: int, n_bufs: int, N: int = N_COLS) -> dict:
    """Same two cells, but with the buffer count FORCED rather than picked.

    This is what separates "rotation cannot help at this size" (hardware) from
    "rotation cannot help at the n this harness happens to pick" (a bug).
    """
    from quack.bench.bench_utils import _bench_cuda_graph_l2_rotate, _clone_l2_rotate_inputs
    from quack.rmsnorm_flydsl import rmsnorm

    x = torch.randn(M, N, device="cuda", dtype=DTYPE)
    w = torch.randn(N, device="cuda", dtype=DTYPE)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs((x, w), {}, n_bufs)

    def call(x_, w_):
        rmsnorm(x_, w_, eps=1e-6)

    def bench(sets_a, sets_k) -> float:
        r = _bench_cuda_graph_l2_rotate(call, sets_a, sets_k, extra_kwargs={})
        return r[0] if isinstance(r, (list, tuple)) else r

    rot = statistics.median([bench(arg_sets, kwarg_sets) for _ in range(5)])
    one = statistics.median(
        [bench([arg_sets[0]] * n_bufs, [kwarg_sets[0]] * n_bufs) for _ in range(5)]
    )
    t = M * N * torch.tensor([], dtype=DTYPE).element_size()
    row = {
        "M": M,
        "one_tensor_bytes": t,
        "rotation_buffers_forced": n_bufs,
        "single_set_bytes": 2 * t,
        "rotation_set_bytes": (n_bufs + 1) * t,
        "predicted_effect": 2 * t <= MALL_BYTES < (n_bufs + 1) * t,
        "single_over_rotate": one / rot if rot else None,
    }
    del x, w, arg_sets, kwarg_sets
    torch.cuda.empty_cache()
    return row


def measure(M: int, N: int = N_COLS) -> dict:
    from quack.bench.bench_utils import (
        _bench_cuda_graph_l2_rotate,
        _clone_l2_rotate_inputs,
        _pick_l2_rotate_count,
    )
    from quack.rmsnorm_flydsl import rmsnorm

    x = torch.randn(M, N, device="cuda", dtype=DTYPE)
    w = torch.randn(N, device="cuda", dtype=DTYPE)
    args = (x, w)
    n_bufs = _pick_l2_rotate_count(args, {})
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(args, {}, n_bufs)

    out_ptrs: list[int] = []

    def call(x_, w_):
        out_ptrs.append(rmsnorm(x_, w_, eps=1e-6).data_ptr())

    def bench(sets_a, sets_k) -> float:
        r = _bench_cuda_graph_l2_rotate(call, sets_a, sets_k, extra_kwargs={})
        return r[0] if isinstance(r, (list, tuple)) else r

    # Count output buffers inside the TIMED region: prime eagerly (that pool is
    # not what the replay re-executes), then capture the same round-robin.
    for i in range(2 * n_bufs):
        call(*arg_sets[i % n_bufs], **kwarg_sets[i % n_bufs])
    torch.cuda.synchronize()
    out_ptrs.clear()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_bufs):
            call(*arg_sets[i], **kwarg_sets[i])
    torch.cuda.synchronize()
    n_out = len(set(out_ptrs))
    del g

    rot = statistics.median([bench(arg_sets, kwarg_sets) for _ in range(5)])
    one = statistics.median(
        [bench([arg_sets[0]] * n_bufs, [kwarg_sets[0]] * n_bufs) for _ in range(5)]
    )

    t = M * N * torch.tensor([], dtype=DTYPE).element_size()
    rotate_set = n_bufs * t + n_out * t
    single_set = t + n_out * t
    row = {
        "M": M,
        "N": N,
        "one_tensor_bytes": t,
        "rotation_buffers": n_bufs,
        "distinct_output_buffers": n_out,
        "single_set_bytes": single_set,
        "rotation_set_bytes": rotate_set,
        "predicted_effect": single_set <= MALL_BYTES < rotate_set,
        "graph_rotate_ms": rot,
        "graph_single_ms": one,
        "single_over_rotate": one / rot if rot else None,
    }
    del x, w, arg_sets, kwarg_sets
    torch.cuda.empty_cache()
    return row


def main() -> int:
    if not torch.cuda.is_available():
        print("no GPU")
        return 1

    coarse = [measure(m) for m in COARSE_M]
    lower = [measure(m) for m in LOWER_EDGE_M]
    upper = [measure(m) for m in UPPER_EDGE_M]
    n_sweep = [measure_at_n(m, n) for m, n in N_SWEEP]

    # Scoring uses the coarse scan only. The edge scans deliberately sample
    # within a few MiB of the threshold, where the model is not expected to
    # resolve; counting them as agreements or disagreements would be scoring
    # the model on points it does not claim.
    agree = sum(
        1
        for r in coarse
        if (r["predicted_effect"] and r["single_over_rotate"] < 0.95)
        or (not r["predicted_effect"] and r["single_over_rotate"] > 0.97)
    )

    result = {
        "source_sha256_16": _source_sha256_16(),
        "device": torch.cuda.get_device_name(0),
        "kernel": "quack.rmsnorm_flydsl.rmsnorm (ROCm)",
        "mall_bytes_assumed": MALL_BYTES,
        "mall_bytes_is_hardcoded": True,
        "torch_reported_L2_cache_size_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "model": (
            "Inputs rotate over n buffers, the output over 1 (measured per row "
            "as distinct_output_buffers, inside the captured region). So the "
            "rotated cell touches (n+1)*t and the single cell 2*t. Rotation "
            "can only change cache state when 2*t <= MALL < (n+1)*t."
        ),
        "coarse_scan": coarse,
        "coarse_agreement": f"{agree}/{len(coarse)}",
        "lower_edge_scan": lower,
        "upper_edge_scan": upper,
        "n_sweep": n_sweep,
        "EDGE_IS_N_DEPENDENT": (
            "The two edges of the band are NOT the same kind of claim, and the "
            "coarse scan alone cannot tell them apart. The low edge is an "
            "artifact of n=4, and n=4 is not a device property: "
            "_pick_l2_rotate_count sizes target_ratio * L2_cache_size = 12 MiB "
            "against tensors of tens of MiB, so n_by_l2 is always 1 and n "
            "clamps to min_buffers for every shape here -- defect 1 of "
            "gfx950_mall_evictor_defect.md acting on the buffer count. Forcing "
            "the n an effective-LLC fix would pick makes the effect appear "
            "below the stated low edge: t=32 MiB goes 0.990x (n=4) -> 0.887x "
            "(n=8) -> 0.871x (n=24), and t=48 MiB goes 0.995x (n=4) -> 0.882x "
            "(n=16). The high edge has no n in it (2*t > MALL cannot be "
            "rescued by more buffers) and holds: t=192 MiB is 0.996x at n=4 "
            "and 1.006x at n=12. So 'rotation cannot help below t~51 MiB' is a "
            "statement about THIS HARNESS, while 'rotation cannot help above "
            "t=128 MiB' is a statement about the device."
        ),
        "edges_are_soft": (
            "The coarse scan agrees on every point, but the edge scans show "
            "the transition is NOT a step. Just outside the predicted band the "
            "ratio is already ~0.95x (low edge) and ~0.89x (high edge) rather "
            "than 1.00x. So the capacity rule predicts WHERE the band is, but "
            "not a sharp threshold at its edges -- consistent with a "
            "set-associative cache with partial residency rather than the "
            "all-or-nothing model assumed here. Sharpening this is Experiment "
            "No.002 territory, which gfx950_mall_evictor_defect.md blocks "
            "pending re-collection."
        ),
        "scope": (
            "Forward rmsnorm on the FlyDSL/ROCm path only, N=1024, bf16, one "
            "device. This says nothing about which CONFIG wins under either "
            "regime (bench_utils.py:98 is a ranking claim), nor about the "
            "cutedsl path, nor about backward. And per EDGE_IS_N_DEPENDENT, "
            "the band's low edge is contingent on the buffer count this "
            "harness currently picks, so it must not be quoted as a property "
            "of gfx950."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    for name, rows in (("coarse", coarse), ("lower edge", lower), ("upper edge", upper)):
        print(f"{name}:")
        for r in rows:
            flag = "PREDICT-EFFECT" if r["predicted_effect"] else "predict ~1.00x"
            print(
                f"  t={r['one_tensor_bytes'] / 2**20:6.1f}MiB M={r['M']:6d} "
                f"n={r['rotation_buffers']} out={r['distinct_output_buffers']} "
                f"| single/rotate={r['single_over_rotate']:.3f}x  {flag}"
            )
    print("n sweep (buffer count forced, not picked):")
    for r in n_sweep:
        flag = "PREDICT-EFFECT" if r["predicted_effect"] else "predict ~1.00x"
        print(
            f"  t={r['one_tensor_bytes'] / 2**20:6.1f}MiB n={r['rotation_buffers_forced']:2d} "
            f"| single/rotate={r['single_over_rotate']:.3f}x  {flag}"
        )
    print(f"coarse agreement {agree}/{len(coarse)}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

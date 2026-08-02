"""Regenerate the bandwidth-cliff sidecar -- and retire the starvation control.

Two tables in `AI/flydsl_rmsnorm_notes.md`'s "`MAX_N = 8192` is a real cliff,
but not where the constant says" section had no artifact and no generator at
all. Writing one did not merely archive them. It showed that the way both were
timed cannot support what they were used to claim, so this file measures the
confounds explicitly rather than quietly producing a second set of numbers.

Three separate floors sit under the published figures, and the published
protocol (eager calls, `torch.cuda.Event` around a rep loop) sees all three:

  1. **The eager path is host-bound below ~30 us.** Wall-clock time of the
     Python call with the GPU never awaited is ~28-29 us, and measured eager
     time at N=8192 is ~28-31 us for every `m` from 256 to 4096. Sixteen times
     the work, no change in wall clock: that is not the GPU. `HOST_BOUND_US`
     below is measured, not assumed, and every row records `host_us` so a
     reader can see which rows are affected instead of trusting this note.

  2. **Graph replay has its own floor of ~1.5 us per kernel**, and ~9.5 us if
     only one kernel is captured per graph. A 64-element `add_` costs 9.49 us
     at K=1 and converges to 1.48 us at K=256, so it is dispatch, not work.
     Capturing K calls per graph amortises it; K is chosen per shape and the
     floor is re-measured here as a control rather than cited.

  3. **MALL residency differs across rows.** gfx950 keeps ~256 MiB resident.
     Every shape in the starvation control (8-128 MiB) fits; the width sweep's
     anchor row at N=8192 (128 MiB) fits and every other width does not. So
     "share of ceiling" was comparing cache-resident rows against a 512 MiB
     DRAM-resident probe. `fits_in_mall` is recorded per row.

What this does to the two tables:

  * **The width cliff survives.** It sits between two rows that are both far
    past the MALL (768 and 896 MiB) and both far above the host floor, and it
    reproduces under eager and graph timing alike. It is the one conclusion in
    that section the new measurement leaves standing.

  * **The starvation control does not survive as published.** Its numbers
    (78.5% -> 5.3% of ceiling from m=4096 to m=256) were read as block-count
    starvation. Under the eager protocol the wall time is *flat* across that
    entire range, so the falling TB/s is a fixed per-call cost divided by a
    shrinking byte count -- arithmetic, not starvation. Real starvation does
    exist and is visible once the floor is removed, but it is a different
    curve, and every row of it is MALL-resident, so it is not a DRAM-bandwidth
    measurement at all. Both the flawed and the corrected series are emitted:
    the point of the artifact is to let the two be compared.

The ceiling is `two_read_one_write`, re-measured in-process before any large
allocation. `copy` cannot be a denominator; see `copy_variability` and
`copy_probe_caveat` in `rmsnorm_32768x4096_bf16_roofline.json`. This docstring
used to give the reason as "allocator-state dependent (4.718 / 5.363 / 4.833
TB/s for one buffer size)". Those were three hand-typed constants copied here
from a prose field that had never measured them, and when they were finally
measured the stated mechanism was false: allocation history moves the copy rate
by ~0.5%, not 14%. The true reason is stronger for this file's purpose -- copy
varies by ~16% across *identically-sized, identically-filled buffers* once they
exceed the MALL working set, which is precisely the regime every row of this
sweep's ceiling lives in. Read the live numbers from the sidecar rather than
from this sentence; the point of that field is that it re-runs and this comment
does not. `share_of_ceiling_pct` is a share of
this measured probe on this device, not of a hardware constant, and
`achieved_TBps` is computed from bytes and time alone for readers who would
rather not inherit the denominator.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_width_cliff.py
Writes AI/data/rmsnorm_fwd_width_cliff.json.
"""

import functools
import hashlib
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
sys.path.insert(0, str(REPO))

from quack.flydsl import rmsnorm_config  # noqa: E402  (follows the sys.path insert)
from quack.flydsl.rmsnorm_config import RmsNormRowConfig  # noqa: E402
import quack.rmsnorm_flydsl as flydsl_rmsnorm  # noqa: E402

DTYPE = torch.bfloat16

# Width sweep: m pinned so block count is constant and only the row changes.
WIDTH_M = 4096
WIDTH_NS = [8192, 32768, 49152, 57344, 65536, 98304]

# Block-count sweep: N pinned at a width with no register pressure
# (elems_per_thread = 32), sweeping the block count instead.
STARVE_N = 8192
STARVE_MS = [4096, 2048, 1024, 512, 256]

ROUNDS = 5
REPS = 20
WARMUP = 10

CEILING_BYTES = 512 * 1024 * 1024

# gfx950 keeps a working set of about this size resident; see
# AI/gfx950_mall_evictor_defect.md. Used only to flag rows, never to correct
# one -- a row that fits is reported as fitting, not adjusted.
MALL_RESIDENT_BYTES = 256 * 1024 * 1024

# Captured calls per graph are chosen so the amortised dispatch floor is small
# against the measurement, subject to a memory budget for the held outputs.
GRAPH_TARGET_US = 400.0
GRAPH_MAX_K = 64
GRAPH_OUTPUT_BUDGET_BYTES = 4 << 30


def _bench(call, reps=REPS, rounds=ROUNDS, warmup=WARMUP):
    """Min-over-rounds of the round mean, in microseconds. All rounds kept."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / reps)
    return samples


def _host_us(call, reps=REPS, rounds=ROUNDS, warmup=WARMUP):
    """Wall time of the Python call alone, with the GPU deliberately not awaited.

    If this is close to the event-timed number, the event timer is measuring
    the host and not the kernel. That is the check the published protocol never
    ran, and it is why the block-count table said what it said.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            call()
        t1 = time.perf_counter()  # read before the sync: host-side only
        torch.cuda.synchronize()
        samples.append((t1 - t0) * 1e6 / reps)
    return min(samples)


def _capture(call, k):
    """Capture k sequential calls into one graph, holding every output alive.

    The outputs are retained because letting them fall out of scope lets the
    caching allocator hand the same block back on the next call, so all k
    writes would land on one buffer and the residency being measured would be
    an artifact of the harness rather than of the shape.
    """
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    held = []
    with torch.cuda.graph(graph):
        for _ in range(k):
            held.append(call())
    return graph, held


def _replay_of(graph):
    """Bind the graph at definition time.

    A bare `lambda: graph.replay()` would resolve `graph` when it is called,
    not when it is written, so a later `del` or loop rebinding would silently
    change which graph is being timed. Ruff flags exactly this (B023/F821) and
    it is a real hazard here, not a style complaint: several of these lambdas
    outlive the name they close over.
    """
    return graph.replay


def _graph_per_call_us(call, out_bytes, floor_us):
    """Per-call time under graph replay, with the dispatch floor amortised."""
    graph1, held1 = _capture(call, 1)
    t1 = min(_bench(_replay_of(graph1), reps=max(2, REPS)))
    del graph1, held1
    torch.cuda.empty_cache()

    if t1 >= GRAPH_TARGET_US:
        k = 1
    else:
        k = min(
            GRAPH_MAX_K,
            max(1, int(GRAPH_TARGET_US // max(t1, 1e-6))),
            max(1, GRAPH_OUTPUT_BUDGET_BYTES // max(out_bytes, 1)),
        )
    if k == 1:
        return t1, 1, [t1], floor_us / max(t1, 1e-9) * 100.0

    graph, held = _capture(call, k)
    samples = [s / k for s in _bench(_replay_of(graph), reps=max(2, 40 // k))]
    del graph, held
    torch.cuda.empty_cache()
    per_call = min(samples)
    return per_call, k, samples, floor_us / max(per_call, 1e-9) * 100.0


def _summarise(samples, nbytes):
    lo, hi = min(samples), max(samples)
    return {
        "samples_us": samples,
        "min_us": lo,
        "median_us": statistics.median(samples),
        "max_us": hi,
        "spread_pct_of_min": (hi - lo) / lo * 100.0,
        "bytes_moved": nbytes,
        "TBps_at_min": nbytes / (lo * 1e-6) / 1e12,
    }


def _measure_ceiling():
    """two_read_one_write, measured before the sweep allocates anything large."""
    n = CEILING_BYTES // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    b = torch.empty(n, device="cuda", dtype=torch.float32)
    c = torch.empty(n, device="cuda", dtype=torch.float32)
    a.fill_(1.0)
    b.fill_(2.0)
    result = _summarise(_bench(functools.partial(torch.add, a, b, out=c)), 3 * n * 4)
    del a, b, c
    torch.cuda.empty_cache()
    return result


def _measure_dispatch_floor():
    """Per-kernel launch cost inside a graph, from a kernel that does no work.

    A 64-element add_ moves 512 bytes. Whatever time it takes is dispatch, so
    the K -> large limit is the floor under every other row in this file.
    """
    tiny = torch.zeros(64, device="cuda")
    add_one = functools.partial(tiny.add_, 1.0)
    series = {}
    for k in (1, 4, 16, 64, 256):
        graph, held = _capture(add_one, k)
        series[f"K={k}"] = min(_bench(_replay_of(graph), reps=max(2, 40 // k))) / k
        del graph, held
        torch.cuda.empty_cache()
    return {
        "per_call_us_by_calls_per_graph": series,
        "amortised_floor_us": series["K=256"],
        "single_call_graph_floor_us": series["K=1"],
        "what": "a 64-element add_ inside a graph; 512 bytes moved, so this is "
        "dispatch cost, not work. The K=1 value is what a naive one-call-per-graph "
        "harness pays; the K=256 value is the irreducible per-kernel floor.",
    }


def _geometry(n):
    """Launch geometry as the kernel derives it, not as the notes assumed.

    The published `elems/thread` column was `n // 256`, right only while the
    heuristic saturates at MAX_NUM_THREADS and the vector size is full width.
    Reading it from the shipped config keeps the column tracking the code, and
    consulting `batch_short_rows` stops a batched shape being reported with
    one-block-per-row geometry.
    """
    bits = torch.finfo(DTYPE).bits
    batched = rmsnorm_config.batch_short_rows(n, bits)
    config = (
        RmsNormRowConfig.for_lane_group(n, bits)
        if batched
        else RmsNormRowConfig.from_analytical_heuristic(n, bits)
    )
    rows_per_block = rmsnorm_config.multi_row_block_rows(config.num_threads) if batched else 1
    return {
        "batched_short_rows": batched,
        "vecsize": config.vecsize,
        "threads_per_row": config.num_threads,
        "rows_per_block": rows_per_block,
        "block_threads": rows_per_block * config.num_threads,
        "num_tiles": config.num_tiles,
        "elems_per_thread": config.elems_per_thread,
    }


def _sweep(shapes, ceiling_tbps, floor_us):
    rows = []
    for m, n in shapes:
        print(f"  m={m} n={n} ...", flush=True)
        geom = _geometry(n)
        x = torch.randn((m, n), device="cuda", dtype=DTYPE)
        weight = torch.randn(n, device="cuda", dtype=DTYPE)
        call = functools.partial(flydsl_rmsnorm.rmsnorm, x, weight, eps=1e-6)
        moved = 2 * m * n * x.element_size()
        out_bytes = m * n * x.element_size()

        eager = _summarise(_bench(call), moved)
        host = _host_us(call)
        graph_us, k, graph_samples, floor_pct = _graph_per_call_us(call, out_bytes, floor_us)
        graph_tbps = moved / (graph_us * 1e-6) / 1e12

        rows.append(
            {
                "m": m,
                "n": n,
                "blocks": -(-m // geom["rows_per_block"]),
                **geom,
                "working_set_bytes": moved,
                "fits_in_mall": moved <= MALL_RESIDENT_BYTES,
                "eager_min_us": eager["min_us"],
                "eager_achieved_TBps": eager["TBps_at_min"],
                "eager_share_of_ceiling_pct": eager["TBps_at_min"] / ceiling_tbps * 100.0,
                "host_only_us": host,
                "host_bound": host >= 0.9 * eager["min_us"],
                "graph_calls_captured": k,
                "graph_per_call_us": graph_us,
                "graph_samples_us": graph_samples,
                "graph_achieved_TBps": graph_tbps,
                "graph_share_of_ceiling_pct": graph_tbps / ceiling_tbps * 100.0,
                "dispatch_floor_pct_of_graph_time": floor_pct,
                "eager_rounds_us": eager["samples_us"],
                "eager_spread_pct_of_min": eager["spread_pct_of_min"],
            }
        )
        del x, weight
        torch.cuda.empty_cache()
    return rows


def _sha(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()[:16]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.manual_seed(0)

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    smi_start = subprocess.run(
        ["rocm-smi", "--showuse"], capture_output=True, text=True, check=False
    ).stdout

    print("measuring ceiling (before any large allocation) ...", flush=True)
    ceiling = _measure_ceiling()
    ceiling_tbps = ceiling["TBps_at_min"]

    print("measuring dispatch floor ...", flush=True)
    floor = _measure_dispatch_floor()
    floor_us = floor["amortised_floor_us"]

    # Same double-binding trap as the VGPR probe: `quack.rmsnorm_flydsl` does
    # `from ... import MAX_N`, so patching only the defining module leaves the
    # validator rejecting every width above the cap. Both raised, both restored,
    # and asserted to agree before either is touched.
    shipped_max_n = rmsnorm_config.MAX_N
    assert flydsl_rmsnorm.MAX_N == shipped_max_n, "the two MAX_N bindings already disagree"
    rmsnorm_config.MAX_N = 1 << 20  # probe only; nothing is written back
    flydsl_rmsnorm.MAX_N = 1 << 20

    try:
        print("width sweep (m fixed, N varies) ...", flush=True)
        width_rows = _sweep([(WIDTH_M, n) for n in WIDTH_NS], ceiling_tbps, floor_us)
        print("block-count sweep (N fixed, m varies) ...", flush=True)
        starve_rows = _sweep([(m, STARVE_N) for m in STARVE_MS], ceiling_tbps, floor_us)
    finally:
        rmsnorm_config.MAX_N = shipped_max_n
        flydsl_rmsnorm.MAX_N = shipped_max_n

    payload = {
        "what": "forward throughput against row width at fixed block count, the "
        "block-count sweep it was contrasted against, and the three timing floors "
        "that sit under both",
        "generator": "AI/probe_rmsnorm_width_cliff.py",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(),
        "device_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", None)),
        "compute_units": torch.cuda.get_device_properties(0).multi_processor_count,
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "host": platform.node(),
        "commit": head,
        "worktree_dirty": bool(dirty),
        "shipped_max_n": shipped_max_n,
        "source_sha256_16": {
            "quack/rmsnorm_flydsl.py": _sha("quack/rmsnorm_flydsl.py"),
            "quack/flydsl/rmsnorm_config.py": _sha("quack/flydsl/rmsnorm_config.py"),
            "AI/probe_rmsnorm_width_cliff.py": _sha("AI/probe_rmsnorm_width_cliff.py"),
        },
        "protocol": {
            "reps_per_round": REPS,
            "rounds": ROUNDS,
            "warmup_calls": WARMUP,
            "statistic": "min over rounds of (round mean over reps); all rounds retained",
            "eager_timer": "torch.cuda.Event around a rep loop -- the published protocol",
            "host_timer": "time.perf_counter around the same loop, read BEFORE the sync, "
            "so it measures the Python call and not the kernel",
            "graph_timer": "K calls captured into one graph, replayed; per-call time is "
            "the replay time divided by K. K is chosen per shape to amortise dispatch, "
            "and is recorded per row.",
            "graph_outputs_held": "every captured output is retained during capture so the "
            "allocator cannot alias all K writes onto one buffer",
            "eviction": "none; residency is reported per row via fits_in_mall rather than "
            "forced, since forcing it would change the thing being measured",
            "bytes_moved": "2 * m * n * 2 (x in, out out); the bf16 weight is omitted, "
            "worth at most 0.0024% at the narrowest shape here",
            "exclusivity": "single visible device; rocm-smi utilisation at start recorded below",
        },
        "exclusivity_check": smi_start,
        "ceiling_probe": {
            "name": "two_read_one_write",
            "why_not_copy": (
                "copy cannot be a denominator: measured on device 5, five identically-"
                "sized and identically-filled buffers read by one destination spread "
                "~16% at 512 MiB and ~5% at 2 GiB, collapsing to <1% at 64 MiB where "
                "they fit the MALL working set. Every ceiling this sweep uses lives in "
                "the past-MALL regime, so copy would contribute that spread directly to "
                "share_of_ceiling_pct. Live figures: copy_variability in the roofline "
                "sidecar. This field previously read 'allocator-state dependent (4.718 "
                "/ 5.363 / 4.833 TB/s ... depending only on residency)' and asserted "
                "that two_read_one_write 'reproduces across processes to better than "
                "1%'. Both were hand-typed and both are wrong: allocation history moves "
                "copy by ~0.5%, and two_read_one_write spans 1.37% over three processes "
                "against write's 0.50%. The ordering against copy's 13.35% is what "
                "justifies the choice, and it holds by an order of magnitude."
            ),
            "measured_before_sweep_allocations": True,
            **ceiling,
        },
        "dispatch_floor": floor,
        "ceiling_caveat": (
            "share_of_ceiling_pct is a share of THIS measured probe on THIS device, not "
            "of a hardware constant. achieved_TBps is computed from bytes and time alone "
            "and is the figure to quote if the denominator is in doubt. Note also that "
            "the ceiling probe's 512 MiB working set is past the MALL while some swept "
            "rows are not: for those rows the ratio compares a cache-resident numerator "
            "against a DRAM-resident denominator and is not a meaningful efficiency."
        ),
        "interpretation_note": (
            "The width sweep pins m so block count is identical across rows; the "
            "block-count sweep pins N at a width with no register pressure. Neither "
            "measures occupancy or residency. A step in the width sweep locates a "
            "boundary in achieved bandwidth; whether register-limited occupancy causes "
            "it is not established here, only made coincident with the computed capacity "
            "step in rmsnorm_fwd_vgpr_by_n.json. The eager and graph columns are both "
            "emitted because the published tables used eager numbers whose small-shape "
            "rows are host-bound -- compare host_only_us against eager_min_us per row "
            "rather than taking either column on trust."
        ),
        "width_sweep_m_fixed": {"m": WIDTH_M, "rows": width_rows},
        "block_count_sweep_n_fixed": {"n": STARVE_N, "rows": starve_rows},
    }

    out_path = REPO / "AI/data/rmsnorm_fwd_width_cliff.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

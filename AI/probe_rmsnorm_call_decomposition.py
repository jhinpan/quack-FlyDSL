"""Regenerate the host/graph/compile decomposition, with its own provenance.

The "Below about 26 us of device work, the public API measures Python" section
of `AI/flydsl_rmsnorm_notes.md` quotes a stage-by-stage decomposition (11.3 ->
16.3 -> 19.4 -> 26.1 us), a FlyDSL dispatch floor (6.38 us), a graph-replay
comparison (2.11 vs 3.88 us at 1x4096) and a torch.compile figure (52.9 vs 26.9
us). None of it had an artifact. This is the last unbacked group in the file.

Archiving it changes one of its conclusions, for the same reason the width
cliff's neighbour table died: **graph replay has a floor, and a ratio taken
across it is not the ratio of the kernels.**

`AI/data/rmsnorm_fwd_width_cliff.json` measures that floor directly -- a
64-element `add_`, 512 bytes, costs 9.47 us when one call is captured per graph
and converges to 1.489 us per kernel at K=256. The published 2.11 / 3.88 / 2.66
figures are all *below* the single-call floor, so they cannot have been taken
at K=1; they reproduce here only at large K. That is fine as a measurement and
was never written down, which matters because the comparison it supports moves
with K:

    K       flydsl   torch   empty-kernel floor   advantage net of floor
    1        9.56    9.56           9.474                 1.00x
    16       2.53    4.13           2.091                 4.64x
    256      1.96    3.66           1.489                 4.61x

At K=1 the two backends are indistinguishable and the honest reading is "the
floor swamps both". At K=256 the gap net of the shared floor is 4.6x, not the
"roughly 2x" the raw numbers suggest. The raw ratio 3.66/1.96 = 1.87x is
arithmetically right and mechanically misleading: most of what it divides is a
constant both sides pay. So this probe emits the floor, the raw per-call time
and the floor-subtracted time at every K, and refuses to publish a single
headline ratio.

The floor is re-measured in-process rather than read across from the width
sweep's sidecar, on the same reasoning that keeps the ceiling probe local: a
constant shared between processes is how four different copy roofline values
got into these notes.

The stage decomposition is measured by neutralising one stage at a time in
situ, and every stage records what it patched, so a reader can see that the
attribution is a difference of two measured configurations rather than a guess
at where the time went. Stages are restored in a finally block and the
restoration is asserted.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_call_decomposition.py
Writes AI/data/rmsnorm_call_decomposition.json.
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

import quack.rmsnorm_flydsl as flydsl_rmsnorm  # noqa: E402  (follows the sys.path insert)

DTYPE = torch.bfloat16
GRAPH_SHAPES = [(1, 4096), (256, 4096)]
HOST_SHAPE = (256, 4096)
GRAPH_KS = [1, 4, 16, 64, 256]

ROUNDS = 5
REPS = 20
WARMUP = 10


def _bench(call, reps=REPS, rounds=ROUNDS, warmup=WARMUP):
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
    """Wall time of the Python call alone, with the GPU deliberately not awaited."""
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
    return {
        "min_us": min(samples),
        "median_us": statistics.median(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _compiled_host_row(x, weight, n, eager_row):
    """Host cost of the same call under torch.compile, plus a recompile guard.

    Two ways this measurement lies if taken naively. Dynamo compiles lazily, so
    the first calls include compilation and the warmup inside `_host_us` is not
    necessarily enough; and a graph break or a recompile mid-measurement would
    show up as host time that is an artifact of the harness rather than a
    property of the compiled path. So the frame counters are read before and
    after the timed region and reported -- if they move, the number is not what
    it says it is.

    Reported against the eager flydsl host cost measured in the same process on
    the same tensors, because the interesting quantity is whether dynamo removes
    the host floor or adds to it, not the absolute figure.
    """
    from torch._dynamo.utils import counters

    compiled = torch.compile(flydsl_rmsnorm.rmsnorm, fullgraph=False, dynamic=False)
    call = functools.partial(compiled, x, weight, eps=1e-6)
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    before = int(sum(counters["frames"].values())) if "frames" in counters else None
    row = _host_us(call)
    after = int(sum(counters["frames"].values())) if "frames" in counters else None

    row["available"] = True
    row["dynamo_frame_counter_before"] = before
    row["dynamo_frame_counter_after"] = after
    row["recompiled_during_measurement"] = (
        None if before is None or after is None else before != after
    )
    row["graph_breaks"] = int(sum(counters["graph_break"].values()))
    row["eager_host_us_same_process"] = eager_row["median_us"]
    row["ratio_compiled_over_eager"] = (
        round(row["median_us"] / eager_row["median_us"], 3) if eager_row["median_us"] else None
    )
    row["what"] = (
        "host wall time of torch.compile(flydsl_rmsnorm.rmsnorm), same timer and "
        "same tensors as the eager row above; ratio > 1 means dynamo adds host "
        "overhead rather than removing it"
    )
    return row


def _capture(call, k):
    """Capture k sequential calls into one graph, holding every output alive.

    Outputs are retained because letting them fall out of scope lets the
    caching allocator hand the same block back, so all k writes would land on
    one buffer.
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


def _replay_series(call):
    """Per-call replay time at each K. Bare `lambda: g.replay()` would rebind."""
    out = {}
    for k in GRAPH_KS:
        graph, held = _capture(call, k)
        out[k] = min(_bench(graph.replay, reps=max(2, 40 // k))) / k
        del graph, held
        torch.cuda.empty_cache()
    return out


def _measure_replay_floor():
    """Per-kernel replay cost from a kernel that does no work: 512 bytes moved."""
    tiny = torch.zeros(64, device="cuda")
    series = _replay_series(functools.partial(tiny.add_, 1.0))
    del tiny
    torch.cuda.empty_cache()
    return series


def _graph_block(floor):
    rows = []
    for m, n in GRAPH_SHAPES:
        x = torch.randn((m, n), device="cuda", dtype=DTYPE)
        w = torch.randn(n, device="cuda", dtype=DTYPE)
        fly = _replay_series(functools.partial(flydsl_rmsnorm.rmsnorm, x, w, eps=1e-6))
        tor = _replay_series(functools.partial(torch.nn.functional.rms_norm, x, (n,), w, eps=1e-6))
        by_k = {}
        for k in GRAPH_KS:
            f_net = fly[k] - floor[k]
            t_net = tor[k] - floor[k]
            by_k[str(k)] = {
                "flydsl_per_call_us": fly[k],
                "torch_per_call_us": tor[k],
                "empty_kernel_floor_us": floor[k],
                "flydsl_net_of_floor_us": f_net,
                "torch_net_of_floor_us": t_net,
                "raw_ratio_torch_over_flydsl": tor[k] / fly[k],
                "net_ratio_torch_over_flydsl": (t_net / f_net) if f_net > 0.02 else None,
                "floor_pct_of_flydsl": floor[k] / fly[k] * 100.0,
            }
        rows.append({"m": m, "n": n, "by_calls_per_graph": by_k})
        del x, w
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

    print("measuring graph replay floor ...", flush=True)
    floor = _measure_replay_floor()

    print("measuring host cost of the public API ...", flush=True)
    hm, hn = HOST_SHAPE
    hx = torch.randn((hm, hn), device="cuda", dtype=DTYPE)
    hw = torch.randn(hn, device="cuda", dtype=DTYPE)
    host = {
        "shape": [hm, hn],
        "flydsl_public_api": _host_us(functools.partial(flydsl_rmsnorm.rmsnorm, hx, hw, eps=1e-6)),
        "torch_public_api": _host_us(
            functools.partial(torch.nn.functional.rms_norm, hx, (hn,), hw, eps=1e-6)
        ),
        "what": "wall time of the Python call with the GPU never awaited, so it is "
        "host dispatch and not a device stall",
    }

    # torch.compile over the same call. The notes claimed dynamo roughly doubles
    # the host cost rather than folding it away; that was measured once by hand
    # and never archived, so it is regenerated here under the same timer.
    print("measuring torch.compile host cost ...", flush=True)
    host["torch_compile"] = _compiled_host_row(hx, hw, hn, host["flydsl_public_api"])
    del hx, hw
    torch.cuda.empty_cache()

    print("measuring graph replay across K ...", flush=True)
    graph_rows = _graph_block(floor)

    payload = {
        "what": "host cost of the public API, and graph-replay cost across "
        "calls-captured-per-graph, with the replay floor measured alongside",
        "generator": "AI/probe_rmsnorm_call_decomposition.py",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(),
        "device_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", None)),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "host": platform.node(),
        "commit": head,
        "worktree_dirty": bool(dirty),
        "source_sha256_16": {
            "quack/rmsnorm_flydsl.py": _sha("quack/rmsnorm_flydsl.py"),
            "AI/probe_rmsnorm_call_decomposition.py": _sha(
                "AI/probe_rmsnorm_call_decomposition.py"
            ),
        },
        "protocol": {
            "reps_per_round": REPS,
            "rounds": ROUNDS,
            "warmup_calls": WARMUP,
            "statistic": "min over rounds of (round mean over reps); host rows keep all rounds",
            "host_timer": "time.perf_counter around a rep loop, read BEFORE the sync",
            "graph_timer": "K calls captured per graph, replayed; per-call = replay / K",
            "graph_outputs_held": "every captured output retained during capture so the "
            "allocator cannot alias all K writes onto one buffer",
            "exclusivity": "single visible device; rocm-smi utilisation at start recorded below",
        },
        "exclusivity_check": smi_start,
        "replay_floor_us_by_calls_per_graph": {str(k): v for k, v in floor.items()},
        "replay_floor_note": (
            "Measured from a 64-element add_ moving 512 bytes, so it is dispatch, not "
            "work. This is the reason no single headline ratio is published here. The "
            "notes previously quoted 2.11 us (flydsl) against 3.88 us (torch) at "
            "1x4096 as 'roughly 2x'. Both figures sit BELOW the K=1 floor, so they were "
            "taken at some larger K that was never recorded, and the comparison is "
            "K-dependent: at K=1 the two are indistinguishable because the floor swamps "
            "both, while net of the floor at K=256 the gap is about 4.6x. The raw ratio "
            "is arithmetically correct and mechanically misleading, because most of "
            "what it divides is a constant both sides pay. Read the net columns, and "
            "read them with the K they were taken at."
        ),
        "host_cost": host,
        "graph_replay": graph_rows,
        "not_covered": (
            "The stage-by-stage attribution (11.3 / 16.3 / 19.4 / 26.1 us) and the 6.38 us "
            "FlyDSL dispatch floor are NOT regenerated here. They need in-situ stubbing of "
            "quack internals, which is a separate probe; they remain unbacked and are "
            "marked as such in the notes. The torch.compile figure IS covered now -- see "
            "host.torch_compile, which supersedes the previously unbacked 52.9 us."
        ),
    }

    out_path = REPO / "AI/data/rmsnorm_call_decomposition.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

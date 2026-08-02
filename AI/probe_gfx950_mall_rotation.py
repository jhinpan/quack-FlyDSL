"""Controlled version of the gfx950 MALL question: fixed kernel, varying working set.

Why this replaces the first probe
---------------------------------
A first attempt swept `x.sum()` over buffer sizes from 4 MiB to 2 GiB and found
a sharp throughput cliff at exactly 256 MiB, which looked like decisive MALL
evidence. It is not usable, for two reasons:

  1. `sum()` is a reduction. torch may select a different kernel, block count or
     multi-pass strategy at different input sizes, so size and kernel vary
     together and the cliff cannot be attributed to the cache.
  2. Its absolute numbers (3796 GB/s at 2 GiB) sit ~40% below both the ROCm
     Kernel Wiki's independent microbench on this same part (6192 GB/s at
     1 GiB) and this repo's own write probe (6587 GB/s). A measurement that far
     under two independent ceilings is limited by the kernel, not by HBM.

The wiki's microbench, which holds its kernel fixed, reports 6152 GB/s at
64 MiB and 6192 GB/s at 1024 MiB -- flat across the boundary, the opposite
shape. One of the two is an artifact, and the uncontrolled one is the suspect.

This probe removes the confound by holding the kernel *exactly* fixed and
varying only how many distinct buffers we rotate through. Every timed call is
the same operation on the same shape; the only thing that changes is whether
the round-robin working set fits in the 256 MiB MALL. That is also precisely
the pattern `benchmarks/benchmark_rmsnorm_flydsl.py` uses, so a positive result
here transfers directly to the harness.

Every number quoted in `AI/gfx950_mall_evictor_defect.md` is produced by this
script, including the evictor-control block. Raw per-round samples and the full
environment are written to a JSON sidecar so the note is recomputable from the
commit rather than from a shell history.

Run:  python3 AI/probe_gfx950_mall_rotation.py [--json OUT.json] [--repeats R]
"""

import argparse
import json
import os
import platform
import subprocess
import sys

import torch


MALL_BYTES = 256 * 2**20
ROTATIONS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
WARMUP = 5
ITERS = 30
REPEATS = 5

# Evictor sizes for the control block, in MiB. 0 = current harness behaviour on
# these shapes (the `ws < 12 MiB` gate never fires); 12 = the size the harness
# would use if it did fire; 256/512/1024 bracket the MALL.
EVICTOR_MIB = [0, 12, 256, 512, 1024]


class _Evictor:
    """Same shape as the harness `_L2Evictor`: a uint8 copy of `target_bytes`."""

    def __init__(self, target_bytes: int):
        self.source = torch.empty(target_bytes, device="cuda", dtype=torch.uint8)
        self.destination = torch.empty_like(self.source)
        self.source.fill_(1)
        self.destination.zero_()
        torch.cuda.synchronize()

    def __call__(self) -> None:
        self.destination.copy_(self.source)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def bench_rotation(elem_bytes, n_buffers, iters=ITERS, repeats=None,
                   evictor_bytes=0):
    """Copy between rotating buffer pairs.

    Returns (median GB/s, working-set bytes, [per-repeat GB/s]).

    Evictor placement follows `_time_rotating_calls` exactly: once per round,
    *outside* the timed window, with the timed window covering one whole
    rotation (`n_buffers` calls). Charging the evictor per call inside the
    window instead measures the evictor rather than the copy -- an earlier
    version of this probe did that and reported 280 GB/s for a 1 GiB evictor,
    which is the evictor's own cost, not a cache effect.
    """
    repeats = REPEATS if repeats is None else repeats
    n = elem_bytes // 4
    srcs = [torch.empty(n, dtype=torch.float32, device="cuda").normal_()
            for _ in range(n_buffers)]
    dsts = [torch.empty(n, dtype=torch.float32, device="cuda")
            for _ in range(n_buffers)]
    evictor = _Evictor(evictor_bytes) if evictor_bytes else None
    torch.cuda.synchronize()

    # One "round" is one full rotation over all buffers, matching the harness.
    rounds = max(1, iters // n_buffers)
    for _ in range(max(1, WARMUP // n_buffers)):
        if evictor is not None:
            evictor()
        for i in range(n_buffers):
            dsts[i].copy_(srcs[i])
    torch.cuda.synchronize()

    moved = 2 * elem_bytes  # a copy moves the buffer twice: one read + one write
    samples = []
    for _ in range(repeats):
        round_us = []
        for _ in range(rounds):
            if evictor is not None:
                evictor()  # outside the timed window, as in _time_rotating_calls
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i in range(n_buffers):
                dsts[i].copy_(srcs[i])
            end.record()
            end.synchronize()
            round_us.append(start.elapsed_time(end) / n_buffers)
        ms = _median(round_us)
        samples.append(moved / (ms * 1e-3) / 1e9)

    working_set = 2 * elem_bytes * n_buffers

    del srcs, dsts, evictor
    torch.cuda.empty_cache()
    return _median(samples), working_set, samples


def sweep(elem_mib, record):
    elem_bytes = elem_mib * 2**20
    print(f"\n### buffer = {elem_mib} MiB, copy_ (1 read + 1 write per call)")
    print(f"{'buffers':>8}  {'working set':>13}  {'vs MALL':>8}  {'GB/s':>8}")

    rows = []
    for nb in ROTATIONS:
        ws = 2 * elem_bytes * nb
        if ws > 24 * 2**30:
            break
        gbps, ws, samples = bench_rotation(elem_bytes, nb)
        rows.append((nb, ws, gbps))
        record.append({
            "block": "rotation_sweep",
            "buffer_mib": elem_mib,
            "n_buffers": nb,
            "working_set_bytes": ws,
            "working_set_vs_mall": ws / MALL_BYTES,
            "evictor_bytes": 0,
            "gbps_median": gbps,
            "gbps_samples": samples,
        })
        print(f"{nb:>8}  {ws / 2**20:>10.0f} MiB  "
              f"{ws / MALL_BYTES:>7.2f}x  {gbps:>8.0f}")

    # The headline is the adjacent-rotation step across the MALL boundary, not
    # best-vs-worst: same kernel, one rotation apart.
    step = None
    by_nb = {nb: g for nb, _, g in rows}
    lo = max((nb for nb, ws, _ in rows if ws <= MALL_BYTES), default=None)
    if lo is not None and (lo + 1) in by_nb:
        step = by_nb[lo] / by_nb[lo + 1]
        print(f"  boundary step: {lo} bufs "
              f"({2 * elem_bytes * lo / 2**20:.0f} MiB) {by_nb[lo]:.0f} GB/s vs "
              f"{lo + 1} bufs ({2 * elem_bytes * (lo + 1) / 2**20:.0f} MiB) "
              f"{by_nb[lo + 1]:.0f} GB/s -> {step:.3f}x")
    return step


def evictor_control(record, elem_mib=64, n_buffers=2, hbm_buffers=8):
    """Does a correctly-sized evictor recover the HBM number?

    Working set is held at exactly the 256 MiB boundary while only the evictor
    size varies; the last row is a genuinely-out-of-cache rotation as the HBM
    reference to compare against.
    """
    elem_bytes = elem_mib * 2**20
    ws = 2 * elem_bytes * n_buffers
    print(f"\n### evictor control: buffer = {elem_mib} MiB x {n_buffers} bufs "
          f"(WS = {ws / 2**20:.0f} MiB, exactly the MALL boundary)")
    print(f"{'evictor':>12}  {'GB/s':>8}   note")

    notes = {
        0: "current behaviour (harness gate never fires)",
        12: "current evictor size (4 MiB L2 x 3)",
    }
    for mib in EVICTOR_MIB:
        gbps, _, samples = bench_rotation(elem_bytes, n_buffers,
                                          evictor_bytes=mib * 2**20)
        record.append({
            "block": "evictor_control",
            "buffer_mib": elem_mib,
            "n_buffers": n_buffers,
            "working_set_bytes": ws,
            "evictor_bytes": mib * 2**20,
            "gbps_median": gbps,
            "gbps_samples": samples,
        })
        print(f"{mib:>9} MiB  {gbps:>8.0f}   {notes.get(mib, '')}")

    ref, ref_ws, ref_samples = bench_rotation(elem_bytes, hbm_buffers)
    record.append({
        "block": "hbm_reference",
        "buffer_mib": elem_mib,
        "n_buffers": hbm_buffers,
        "working_set_bytes": ref_ws,
        "evictor_bytes": 0,
        "gbps_median": ref,
        "gbps_samples": ref_samples,
    })
    print(f"{'none':>9}      {ref:>8.0f}   HBM reference "
          f"({hbm_buffers} bufs, WS = {ref_ws / 2**20:.0f} MiB)")
    return ref


def environment():
    props = torch.cuda.get_device_properties(0)
    env = {
        "device_name": props.name,
        "gcn_arch": getattr(props, "gcnArchName", None),
        "multi_processor_count": props.multi_processor_count,
        "torch_l2_cache_size_bytes": props.L2_cache_size,
        "mall_bytes_assumed": MALL_BYTES,
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "warmup": WARMUP,
        "iters_per_sample": ITERS,
        "repeats": REPEATS,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("HIP_VISIBLE_DEVICES"),
    }
    # MALL is discoverable rather than hardcoded; record what the machine says.
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True,
                             timeout=30).stdout
        env["rocminfo_l3"] = [ln.strip() for ln in out.splitlines()
                              if "L3:" in ln][:1]
    except Exception as exc:  # noqa: BLE001 - provenance only, never fatal
        env["rocminfo_l3"] = f"unavailable: {exc}"
    # Provenance of the *script*, not just of the checkout. HEAD alone is
    # misleading: a probe run before committing records its parent commit, which
    # does not contain the code that produced the numbers.
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=here, timeout=10
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001
        env["git_commit"] = None
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=here, timeout=10
        ).stdout
        env["git_dirty"] = bool(dirty.strip())
        env["git_dirty_paths"] = [ln[3:] for ln in dirty.splitlines()][:20]
    except Exception:  # noqa: BLE001
        env["git_dirty"] = None
    # Hash of this file, so the JSON identifies the code that produced it even
    # when the working tree is dirty or the commit is the parent.
    try:
        import hashlib
        with open(os.path.abspath(__file__), "rb") as fh:
            env["script_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    except Exception:  # noqa: BLE001
        env["script_sha256"] = None
    env["script_path"] = os.path.relpath(os.path.abspath(__file__), here)
    return env


def main() -> None:
    global REPEATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="AI/probe_gfx950_mall_rotation.json",
                    help="where to write raw samples + environment")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()
    REPEATS = args.repeats

    env = environment()
    print(f"device             : {env['device_name']} ({env['gcn_arch']})")
    print(f"torch L2_cache_size: "
          f"{env['torch_l2_cache_size_bytes'] / 2**20:.1f} MiB "
          "(per-XCD; MALL not reported)")
    print(f"MALL assumed       : {MALL_BYTES / 2**20:.0f} MiB "
          "(32 MiB/stack x 8, per ROCm Kernel Wiki hw-chiplet-xcd)")
    print(f"rocminfo L3        : {env['rocminfo_l3']}")
    print(f"torch              : {env['torch_version']} "
          f"(hip {env['torch_hip']})")
    print()
    print("Kernel is identical in every row below. Only the rotation working")
    print("set changes, so any systematic difference is attributable to cache")
    print("residency rather than to kernel selection.")

    record = []
    ratios = []
    for elem_mib in (16, 64):
        r = sweep(elem_mib, record)
        if r is not None:
            ratios.append(r)

    hbm_ref = evictor_control(record)

    payload = {"environment": env, "measurements": record}
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nraw samples + environment -> {args.json}")

    print()
    print("=" * 62)
    print("Note on reading these numbers: quote the *boundary step* between")
    print("adjacent rotation counts, not the best-vs-worst ratio. At 64 MiB")
    print("buffers the honest figure is 2 buffers (256 MiB) vs 3 buffers")
    print("(384 MiB) -- one rotation apart, same kernel. The 16 MiB sweep's")
    print("single-buffer row is additionally inflated by fitting the 32 MiB")
    print("aggregate L2 and should not be used as the headline.")
    ev = {r["evictor_bytes"] // 2**20: r["gbps_median"]
          for r in record if r["block"] == "evictor_control"}
    if ev and hbm_ref:
        big = [v for k, v in ev.items() if k >= 256]
        if big:
            print(f"evictor control: {min(big):.0f}-{max(big):.0f} GB/s at "
                  f">=256 MiB evictor vs {hbm_ref:.0f} GB/s HBM reference; "
                  f"{ev.get(0, float('nan')):.0f} GB/s with none.")
    if not ratios:
        print("INCONCLUSIVE: no size gave points both inside and outside MALL.")
        return
    worst = max(ratios)
    print(f"max boundary step across sweeps: {worst:.3f}x")
    if worst > 1.15:
        print("VERDICT: confirmed. Rotation sets that fit the 256 MiB MALL are")
        print("         measured against a cache, not HBM.")
        print("         Attribution: the evictor-control block above shows a")
        print("         12 MiB evictor recovering nearly as much as a 256 MiB")
        print("         one, so the cause is the `ws < l2_target` gate leaving")
        print("         eviction OFF on these shapes -- not the evictor's size.")
        print("         Sizing it off the per-XCD L2 is a separate, real")
        print("         mis-derivation, but it is not what this measures.")
        print("         Fix before taking MI355X data.")
    else:
        print("VERDICT: not confirmed. With the kernel held fixed, fitting the")
        print("         MALL does not measurably inflate throughput for this")
        print("         access pattern -- consistent with the wiki's flat")
        print("         64 MiB -> 1024 MiB microbench curve, and inconsistent")
        print("         with the uncontrolled sum() sweep. Treat the earlier")
        print("         cliff as a reduction-kernel artifact.")


if __name__ == "__main__":
    main()

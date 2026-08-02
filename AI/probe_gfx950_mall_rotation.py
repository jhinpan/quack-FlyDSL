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

Run:  python3 AI/probe_gfx950_mall_rotation.py
"""

import torch


MALL_BYTES = 256 * 2**20
ROTATIONS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
WARMUP = 5
ITERS = 30


def bench_rotation(elem_bytes: int, n_buffers: int, iters: int = ITERS):
    """Copy between rotating buffer pairs; return (GB/s, working-set bytes)."""
    n = elem_bytes // 4
    srcs = [torch.empty(n, dtype=torch.float32, device="cuda").normal_()
            for _ in range(n_buffers)]
    dsts = [torch.empty(n, dtype=torch.float32, device="cuda")
            for _ in range(n_buffers)]
    torch.cuda.synchronize()

    for i in range(WARMUP):
        dsts[i % n_buffers].copy_(srcs[i % n_buffers])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        dsts[i % n_buffers].copy_(srcs[i % n_buffers])
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    # a copy moves the buffer twice: one read + one write
    moved = 2 * elem_bytes
    gbps = moved / (ms * 1e-3) / 1e9
    working_set = 2 * elem_bytes * n_buffers

    del srcs, dsts
    torch.cuda.empty_cache()
    return gbps, working_set


def sweep(elem_mib: int) -> None:
    elem_bytes = elem_mib * 2**20
    print(f"\n### buffer = {elem_mib} MiB, copy_ (1 read + 1 write per call)")
    print(f"{'buffers':>8}  {'working set':>13}  {'vs MALL':>8}  {'GB/s':>8}")

    rows = []
    for nb in ROTATIONS:
        ws = 2 * elem_bytes * nb
        if ws > 24 * 2**30:
            break
        gbps, ws = bench_rotation(elem_bytes, nb)
        rows.append((nb, ws, gbps))
        print(f"{nb:>8}  {ws / 2**20:>10.0f} MiB  "
              f"{ws / MALL_BYTES:>7.2f}x  {gbps:>8.0f}")

    inside = [g for _, ws, g in rows if ws <= MALL_BYTES]
    outside = [g for _, ws, g in rows if ws >= 2 * MALL_BYTES]
    if inside and outside:
        ratio = max(inside) / max(outside)
        print(f"  best fitting MALL {max(inside):.0f} GB/s vs "
              f"clearly exceeding it {max(outside):.0f} GB/s -> {ratio:.3f}x")
        return ratio
    return None


def main() -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"device             : {props.name} ({props.gcnArchName})")
    print(f"torch L2_cache_size: {props.L2_cache_size / 2**20:.1f} MiB "
          "(per-XCD; MALL not reported)")
    print(f"MALL assumed       : {MALL_BYTES / 2**20:.0f} MiB "
          "(32 MiB/stack x 8, per ROCm Kernel Wiki hw-chiplet-xcd)")
    print()
    print("Kernel is identical in every row below. Only the rotation working")
    print("set changes, so any systematic difference is attributable to cache")
    print("residency rather than to kernel selection.")

    ratios = []
    for elem_mib in (16, 64):
        r = sweep(elem_mib)
        if r is not None:
            ratios.append(r)

    print()
    print("=" * 62)
    print("Note on reading these numbers: quote the *boundary step* between")
    print("adjacent rotation counts, not the best-vs-worst ratio. At 64 MiB")
    print("buffers the honest figure is 2 buffers (256 MiB) vs 3 buffers")
    print("(384 MiB) -- one rotation apart, same kernel. The 16 MiB sweep's")
    print("single-buffer row is additionally inflated by fitting the 32 MiB")
    print("aggregate L2 and should not be used as the headline.")
    if not ratios:
        print("INCONCLUSIVE: no size gave points both inside and outside MALL.")
        return
    worst = max(ratios)
    print(f"max inflation across sweeps: {worst:.3f}x")
    if worst > 1.15:
        print("VERDICT: confirmed. Rotation sets that fit the 256 MiB MALL are")
        print("         measured against a cache, not HBM. The harness evictor")
        print("         is sized from torch's 4 MiB per-XCD figure and therefore")
        print("         under-sizes on gfx950. Fix before taking MI355X data.")
    else:
        print("VERDICT: not confirmed. With the kernel held fixed, fitting the")
        print("         MALL does not measurably inflate throughput for this")
        print("         access pattern -- consistent with the wiki's flat")
        print("         64 MiB -> 1024 MiB microbench curve, and inconsistent")
        print("         with the uncontrolled sum() sweep. Treat the earlier")
        print("         cliff as a reduction-kernel artifact.")


if __name__ == "__main__":
    main()

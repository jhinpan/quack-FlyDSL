"""Separate allocation ordinal from timing order, and record the addresses.

Every copy artifact in this tree so far reports an "allocation slot" effect, and
in every one of them slot `i` was allocated `i`-th and then measured `i`-th. So
three candidate causes are one index:

  * allocation ordinal -- where the allocator placed the `i`-th buffer
  * timing order       -- when in the sequence the measurement happened
  * address            -- what the buffer's actual virtual address is

@Reviewer raised this against `9899e9d` and again against `682e103`, and it was
right both times: the artifact called the axis "slot" and read the effect as
placement, which is a claim about addresses, while recording neither addresses
nor any variation in timing order.

This probe breaks the tie two ways at once, both cheap, neither of which needed
a new mechanism -- only for someone to stop taking the index at face value:

  1. Buffers are allocated 0..4 and then measured in a PERMUTED order, so
     allocation ordinal and timing position vary independently. Pooled across
     processes with different permutations, the variance attributable to each can
     be compared directly.
  2. Every buffer's `data_ptr()` is recorded, so "placement" becomes a testable
     statement about addresses rather than an interpretation of an index.

Prediction if the effect is placement: rate tracks allocation ordinal (and
address), and timing position explains little. Prediction if it is warmup, clock
ramp or drift: rate tracks timing position and allocation ordinal explains
little. These are distinguishable, which is the whole point -- the previous
design could not have told them apart, and reported the placement reading anyway.

Deliberately does NOT set FLYDSL_AUTOTUNE=1.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# Must follow the sys.path insert above, so it cannot sit in the header. The
# repo's pinned ruff config does not enable E402 while @Reviewer's invocation
# does, and an inline suppression for it trips RUF100 under the repo config --
# no inline directive satisfies both. A function-scoped import satisfies both.
def _roofline():
    from AI.probe_rmsnorm_roofline import (
        _bench,
        _summarise,
    )

    return (_bench, _summarise)


_bench, _summarise = _roofline()


N_BUFFERS = 5
SIZE_MIB = 2048


def _run(seed):
    """One process: allocate in order, measure in a permuted order, keep addresses."""
    cnt = (SIZE_MIB * 1024 * 1024) // 4
    nbytes = 2 * cnt * 4
    dst = torch.empty(cnt, device="cuda", dtype=torch.float32)
    # Allocation order is always 0..N-1. Only the measurement order is permuted,
    # because allocation order is what the allocator sees and the thing under
    # test is whether the *measurement* sequence matters independently of it.
    srcs = [
        torch.empty(cnt, device="cuda", dtype=torch.float32).fill_(1.0) for _ in range(N_BUFFERS)
    ]
    addrs = [s.data_ptr() for s in srcs]

    order = list(range(N_BUFFERS))
    random.Random(seed).shuffle(order)

    rows = []
    for time_pos, alloc_idx in enumerate(order):
        s = _summarise(_bench(lambda t=srcs[alloc_idx]: dst.copy_(t)), nbytes)
        rows.append(
            {
                "alloc_ordinal": alloc_idx,
                "time_position": time_pos,
                "address": addrs[alloc_idx],
                "TBps_at_min": s["TBps_at_min"],
                "rounds_us": s["samples_us"],
                "within_spread_pct": round(s["spread_pct_of_min"], 3),
            }
        )
    return {
        "seed": seed,
        "measurement_order": order,
        "dst_address": dst.data_ptr(),
        "size_mib": SIZE_MIB,
        "device_name": torch.cuda.get_device_name(0),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    Path(args.out).write_text(json.dumps(_run(args.seed)) + "\n")


if __name__ == "__main__":
    main()

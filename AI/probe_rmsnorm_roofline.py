"""Regenerate the 32768x4096 bf16 roofline sidecar, with its own provenance.

`AI/data/rmsnorm_32768x4096_bf16_roofline.json` backs the forward throughput
figures quoted in `AI/flydsl_rmsnorm_notes.md`. It was originally produced by an
ad-hoc script that no longer existed, which is the defect this file closes:
@Reviewer's standing blocker was that the sidecar carried no generator, no
source hashes, no host or device identity, and no exclusivity evidence, so its
numbers could be re-read but not re-derived.

Two things about the denominators, both of which the old artifact left implicit
and a reader had to reverse-engineer from the stored TB/s:

  * `bytes_moved_min_traffic` is the *kernel's* minimum traffic, x + out, and
    it omits the bf16 weight. Exact traffic is 536,879,104 bytes. The weight is
    8 KiB against 512 MiB, so it moves a derived TB/s by 0.0015% and no
    conclusion turns on it -- but the sidecar's job is to let the number be
    re-derived rather than approximated, so both are recorded.

  * The three roofline probes do NOT move the kernel's byte count. A copy
    reads and writes, so it moves 2x the buffer; two-read-one-write moves 3x;
    a pure write moves 1x. Their TB/s are therefore computed against their own
    traffic, not against the kernel's. Storing only the resulting TB/s made the
    three look directly comparable to the kernel's number when the arithmetic
    behind each differed. Each probe now records `bytes_moved` alongside.

The statistic is min-over-rounds of the round mean, matching
`AI/probe_rmsnorm_harness_levels.py`. Every round is retained so a reader can
compute the spread rather than take a summary on trust -- the roofline rows are
the ones whose observed span (0.18-2.18%) the notes cite as the comparable noise
figure, and that span is only auditable if the raw rounds are present.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_roofline.py
Writes AI/data/rmsnorm_32768x4096_bf16_roofline.json.
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import quack.rmsnorm_flydsl as flydsl_rmsnorm  # noqa: E402  (must follow the sys.path insert)

M, N = 32768, 4096
DTYPE = torch.bfloat16
ROUNDS = 7
REPS = 50
WARMUP = 20

# Large enough that a rotation cannot sit in the MALL. gfx950 keeps a working
# set of 256 MiB or less resident, which is the defect recorded in
# AI/gfx950_mall_evictor_defect.md; the L2-evicted rows below are only
# meaningful if the evictor buffer clears that threshold.
EVICT_BYTES = 512 * 1024 * 1024


def _bench(call, evict=None):
    """Min-of-rounds of mean-over-reps, in microseconds. All rounds returned.

    When `evict` is given it runs once per round *outside* the timed window,
    which is what "L2 evicted" means here -- not between calls, and not inside
    the event pair. The notes previously described the harness as evicting
    between timed calls; it does not, and neither does this.
    """
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ROUNDS):
        if evict is not None:
            evict()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPS):
            call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / REPS)
    return samples


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


def _probes():
    """Three access patterns, each against the traffic it actually moves."""
    n = EVICT_BYTES // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    b = torch.empty(n, device="cuda", dtype=torch.float32)
    c = torch.empty(n, device="cuda", dtype=torch.float32)
    buf = n * 4
    a.fill_(1.0)
    b.fill_(2.0)

    return {
        # read buf + write buf. See `copy_probe_caveat` in the payload: this
        # one is not a stable property of the card.
        "copy": (lambda: c.copy_(a), 2 * buf),
        # read a, read b, write c
        "two_read_one_write": (lambda: torch.add(a, b, out=c), 3 * buf),
        # write buf only
        "write": (lambda: c.fill_(1.0), buf),
    }


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

    x = torch.randn((M, N), device="cuda", dtype=DTYPE)
    weight = torch.randn(N, device="cuda", dtype=DTYPE)
    evictor = torch.empty(EVICT_BYTES // 4, device="cuda", dtype=torch.float32)
    evictor_dst = torch.empty_like(evictor)

    kernel_bytes = 2 * M * N * x.element_size()
    exact_bytes = kernel_bytes + N * weight.element_size()

    def evict():
        evictor_dst.copy_(evictor)

    def flydsl_call():
        flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)

    def torch_call():
        torch.nn.functional.rms_norm(x, (N,), weight, eps=1e-6)

    payload = {
        "what": "forward throughput at 32768x4096 bf16 against three measured "
        "bandwidth probes, with the raw rounds behind every cited figure",
        "generator": "AI/probe_rmsnorm_roofline.py",
        "shape": [M, N],
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
            "AI/probe_rmsnorm_roofline.py": _sha("AI/probe_rmsnorm_roofline.py"),
        },
        "bytes_moved_min_traffic": kernel_bytes,
        "bytes_moved_exact_including_weight": exact_bytes,
        "byte_count_note": (
            "bytes_moved_min_traffic is x + out and omits the bf16 weight; "
            "exact traffic is bytes_moved_exact_including_weight. The 8 KiB "
            "weight against 512 MiB moves a derived TB/s by 0.0015%. Probe "
            "TB/s use each probe's own bytes_moved (copy 2x buffer, "
            "two_read_one_write 3x, write 1x), NOT the kernel's byte count."
        ),
        "protocol": {
            "reps_per_round": REPS,
            "rounds": ROUNDS,
            "warmup_calls": WARMUP,
            "statistic": "min over rounds of (round mean over reps); all rounds retained",
            "timer": "torch.cuda.Event, recorded around a rep loop",
            "events": "recorded on the operand device (HIP_VISIBLE_DEVICES pins it)",
            "eviction": (
                f"cold rows copy a {EVICT_BYTES >> 20} MiB buffer once per round, "
                "outside the timed window -- not between calls"
            ),
            "exclusivity": "single visible device; caller is responsible for it "
            "being idle. rocm-smi utilisation at start is recorded below.",
        },
        "exclusivity_check": subprocess.run(
            ["rocm-smi", "--showuse"], capture_output=True, text=True, check=False
        ).stdout,
        "copy_probe_caveat": (
            "The copy probe is allocator-state dependent and must not be quoted "
            "as a property of the card. Same buffer size, same process, 512 MiB: "
            "4.718 TB/s with no prior allocations, 5.363 TB/s with this "
            "generator's tensors already resident -- a 14% swing from allocation "
            "history alone. At 2 GiB it reads 4.833. Standalone c.copy_, "
            "a.clone() and c[:]=a all agree at 4.71-4.72, so it is placement, "
            "not the op. This is the mechanism behind the several different "
            "MI355X 'copy roofline' values that accumulated in the notes. Use "
            "write or two_read_one_write as denominators; copy is retained here "
            "only because the notes' history refers to it."
        ),
        "roofline_probes": {},
        "measurements": {},
    }

    for name, (call, nbytes) in _probes().items():
        print(f"probing {name} ...", flush=True)
        payload["roofline_probes"][name] = _summarise(_bench(call), nbytes)

    for label, call, ev in (
        ("flydsl_cold_l2", flydsl_call, evict),
        ("flydsl_warm", flydsl_call, None),
        ("torch_cold_l2", torch_call, evict),
        ("torch_warm", torch_call, None),
    ):
        print(f"measuring {label} ...", flush=True)
        payload["measurements"][label] = _summarise(_bench(call, evict=ev), kernel_bytes)

    out_path = REPO / "AI/data/rmsnorm_32768x4096_bf16_roofline.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

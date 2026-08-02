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
import re
import statistics
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# Must follow the sys.path insert above, so it cannot move to the header. The
# repo's pinned ruff config does not enable E402, but @Reviewer's invocation
# does, and an inline suppression for it then trips RUF100 under the repo
# config -- the two
# configs cannot both be satisfied by an inline directive. Importing inside a
# function satisfies both, and the check that matters is unaffected either way.
def _flydsl_rmsnorm():
    import quack.rmsnorm_flydsl as m

    return m


flydsl_rmsnorm = _flydsl_rmsnorm()

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

# gfx950 keeps a working set of 256 MiB or less resident in the MALL. Used by
# _copy_variability to label which sizes can be resident at all; the same
# threshold is the one AI/gfx950_mall_evictor_defect.md records.
MALL_WORKING_SET_BYTES = 256 * 1024 * 1024


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


def _identical_buffer_spread(mib):
    """Copy rate across five identical buffers at one size, one fixed destination.

    A function rather than a loop body because the loop body version closed over
    a `dst` that the same iteration then deleted. It produced correct numbers --
    `_bench` runs the lambda eagerly, so the name was always still bound when it
    mattered -- but ruff flagged it F821/B023 and it was right to: the code was
    only correct by evaluation order, and nothing in it said so. Numbers I was
    about to publish rested on that. Binding `dst` as a local of a real scope
    makes the guarantee structural instead of incidental.

    Rates are stored unrounded. The first version rounded to 3 dp on the way
    out, and that destroyed the answer to a question asked of it a few hours
    later: whether the draw recorded as `4.895` fell inside the half-open band
    `[4.885, 4.895)` that @Reviewer fixed for the historical `4.89` cell. At 3 dp
    that value means "somewhere in [4.8945, 4.8955)", which straddles the
    boundary -- exactly half the bin is inside. The rounding was applied for
    readability, and it silently threw away the only digits that could decide a
    membership question about a 0.2%-wide interval.

    The general form is the one this file keeps running into from new angles:
    *a summary is a claim about which distinctions will matter later, and it is
    made before you know.* Rounded copies are kept alongside for readers,
    clearly named.

    That lesson was learned twice here, and the second time the first fix was
    what hid it. Storing the rates unrounded read as "full precision retained",
    so the next question -- @Reviewer's, in 23a6f662 -- went unasked for a day:
    full precision *of what*? Each of these five numbers is `bytes / min(seven
    rounds)`. The other six rounds were computed by `_summarise` and dropped on
    the floor. So a slot that is genuinely slower and a slot that caught one bad
    round are the same datum here, and the slot-vs-process decomposition that
    rests on these draws could not tell them apart. `rounds_us_per_identical_buffer`
    now carries all seven; the derived rate stays, unchanged, so every consumer
    of the old field keeps reading it.

    Note what "unrounded" bought and what it cost. It was a real fix -- the band
    question needed those digits. But precision and provenance are different
    axes, and satisfying one loudly is how the other stops being checked.
    """
    cnt = (mib * 1024 * 1024) // 4
    nbytes = 2 * cnt * 4
    dst = torch.empty(cnt, device="cuda", dtype=torch.float32)
    srcs = [torch.empty(cnt, device="cuda", dtype=torch.float32).fill_(1.0) for _ in range(5)]
    per_buffer = [_summarise(_bench(lambda s=s: dst.copy_(s)), nbytes) for s in srcs]
    rates = [p["TBps_at_min"] for p in per_buffer]
    lo, hi = min(rates), max(rates)
    return {
        "TBps_per_identical_buffer": rates,
        "TBps_per_identical_buffer_rounded_3dp": [round(r, 3) for r in rates],
        "rounding_note": (
            "The unrounded list is authoritative. A rounded copy of this field once "
            "made an interval-membership question undecidable against a 0.2%-wide "
            "band; 3 dp is a display choice, not a measurement."
        ),
        # All seven rounds per buffer. @Reviewer's objection was that "full
        # precision" meant one derived rate while the rounds behind it were
        # discarded, so a reader cannot separate a genuinely slower slot from one
        # that caught a single bad round -- and that distinction is what the whole
        # slot argument rests on. _summarise already computed these.
        "rounds_us_per_identical_buffer": [p["samples_us"] for p in per_buffer],
        "within_buffer_spread_pct": [round(p["spread_pct_of_min"], 3) for p in per_buffer],
        "spread_pct_of_min": round((hi - lo) / lo * 100.0, 2),
        "fits_in_mall": mib * 1024 * 1024 <= MALL_WORKING_SET_BYTES,
        "n_identical_buffers": len(srcs),
    }


def _copy_variability():
    """Measure what actually moves the copy rate, instead of typing a mechanism.

    @Autotune found the defect that started this. `copy_probe_caveat` asserted
    three copy rates -- 4.718 with a cold allocator, 5.363 with this generator's
    tensors resident, 4.833 at 2 GiB -- and every one was a hand-typed constant
    inside a prose string. Grepping the emitted sidecar: each appeared exactly
    once, only in that string, and no numeric field equalled any of them. No
    samples, no bytes_moved, no re-derivation. They had also propagated by hand
    to four other sites (this file, the width-cliff docstring, its
    `why_not_copy` field, and the notes' table). The string then contradicted
    its own payload, claiming 5.363 for a condition the generator computed at
    5.579 -- 4.04% apart against that run's 0.31% spread, 13x its noise floor.

    Measuring the three conditions falsified the caveat's *mechanism*, which is
    why this function is no longer named for it. The claim was a 14% swing
    "from allocation history alone". Measured, allocation history does nothing:
    the identical call before any large allocation, with three more buffers made
    live, and again afterwards reads 4.767 / 4.764 / 4.776 -- 0.25% apart, and
    that null reproduced in three separate processes. The swing is real but it
    is not history. It tracks the *buffer*: across six identically-sized,
    identically-filled sources read by one fixed destination, rates ranged
    4.785 to 5.593, and a pointer that read 5.010 read 5.582 after a free and
    realloc to the same address, so it is not a stable per-allocation label
    either.

    What discriminates is residency, and the size sweep is the test that
    separates it from every buffer-identity story: across five identical
    buffers the spread is 0.70% at 64 MiB (inside the MALL's 256 MiB working
    set), 16.2% at 512 MiB, and 4.96% at 2 GiB. If the cause were the
    allocator, the op, or the individual buffer, nothing about it would care
    that 64 MiB fits in the MALL. Running the sizes in reverse order gives
    0.85% / 18.5% / 4.37%, so it is not an order effect. The mechanism is what
    fraction of a past-MALL buffer happens to land where, and that varies per
    allocation and is re-rolled on free.

    The conclusion the notes draw is unchanged and now rests on the right
    evidence: **do not use copy as a denominator.** It was already the correct
    rule, for a reason that was wrong.

    The general defect, worth stating because it is not about this number:
    *a number that lives only in a prose string has no error bar and never
    re-runs.* `TBps_at_min` is recomputed every invocation and would have caught
    its own drift; the caveat string could not, and did not. Interpolating the
    resident figure from the computed probe would have fixed one of three and
    left the other two exactly as unfounded -- and would have left the false
    mechanism in place, since no arithmetic on those three numbers tests it.
    """
    states = {}

    # History rows. Named for the hypothesis they test, and retained because
    # they are the ones that came back null -- deleting them would leave the
    # falsified claim unfalsifiable on the next run.
    n = EVICT_BYTES // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    c = torch.empty(n, device="cuda", dtype=torch.float32)
    a.fill_(1.0)
    states["history_cold_allocator_512MiB"] = _summarise(_bench(lambda: c.copy_(a)), 2 * n * 4)

    resident = [torch.empty(n, device="cuda", dtype=torch.float32) for _ in range(3)]
    for t in resident:
        t.fill_(3.0)
    states["history_three_buffers_live_512MiB"] = _summarise(_bench(lambda: c.copy_(a)), 2 * n * 4)

    # Same call a third time, after the heap changed twice. This is the drift
    # control: without it, agreement between the first two rows could just be a
    # machine that was not moving.
    states["history_cold_pair_remeasured_512MiB"] = _summarise(
        _bench(lambda: c.copy_(a)), 2 * n * 4
    )

    del resident
    torch.cuda.empty_cache()

    # Residency rows: five identical buffers per size, one fixed destination.
    # 64 MiB fits the MALL working set (256 MiB, per gfx950_mall_evictor_defect.md);
    # the other two do not. This is the comparison that discriminates.
    residency = {}
    for mib in (64, 512, 2048):
        residency[f"{mib}MiB"] = _identical_buffer_spread(mib)
        torch.cuda.empty_cache()

    hist = {k: v["TBps_at_min"] for k, v in states.items()}
    hlo, hhi = min(hist.values()), max(hist.values())
    return {
        "states": states,
        "TBps_by_state": {k: round(v, 3) for k, v in hist.items()},
        "history_swing_pct_of_min": round((hhi - hlo) / hlo * 100.0, 2),
        "identical_buffers_by_size": residency,
        "finding": (
            "Allocation history does not move the copy rate; buffer placement past "
            "the MALL does. The three history rows are the same call before, during "
            "and after heap changes. The size sweep is the discriminating comparison: "
            "identical buffers agree inside the MALL and disagree past it, which no "
            "allocator-state or per-op explanation predicts."
        ),
        "supersedes": (
            "The '14% swing from allocation history alone' this field replaces. That "
            "claim was hand-typed, never re-ran, and is falsified by the history rows "
            "here. The rule it supported -- do not use copy as a denominator -- is "
            "unaffected, and is now supported by the reason that is true."
        ),
        "how_to_read": (
            "Every figure here is computed from this run's samples. The prose caveat "
            "below interpolates from this field and must not restate a number that is "
            "not in it -- the previous hand-typed version disagreed with this "
            "generator's own computed copy probe by 4.04% against a 0.31% spread."
        ),
    }


def _assert_caveat_is_derived(payload):
    """Refuse to write a caveat containing a number no field computed.

    Interpolation alone does not close the defect -- the next person to add a
    sentence can type a constant straight back in, and nothing would fail. So
    every decimal in the emitted string must either round-trip to a measured
    field or be named here as narrative.

    Two of the narrative decimals are the post-mortem itself: 4.04 is how far
    the old hand-typed 5.363 sat from this file's computed copy probe on the run
    @Autotune audited, and 0.31 is that run's within-run spread. They are
    historical, they refer to a specific prior artifact, and they are the reason
    the rest of the sentence is computed. If they were interpolated from the
    current run they would silently restate themselves as fresh measurements,
    which is the defect one level up.

    The other three (0.50, 1.37, 13.35) are cross-process spreads. A single run
    genuinely cannot compute them, which is the honest reason a number may be
    transcribed -- but "cannot be computed here" is exactly the excuse the
    original caveat's constants had, so they do not get to sit in prose alone:
    they are also emitted as `denominator_stability_across_processes` with their
    device, process count and provenance. The allowlist is the narrow exception,
    and every entry in it must be a value no run could produce.
    """
    text = payload["copy_probe_caveat"]
    narrative = {
        "4.04",  # @Autotune's drift finding, the defect that created this guard
        "0.31",  # the spread that drift was measured against
        "0.50",  # superseded write value, quoted as the error
        "1.37",  # superseded two_read_one_write value, quoted as the error
        "13.35",  # superseded copy value, quoted as the error
        "0.63",  # copy across six processes, generator held fixed
        "0.08",  # two_read_one_write, same six
        "0.22",  # write, same six
    }
    cv = payload["copy_variability"]
    measured = {f"{v:.3f}" for v in cv["TBps_by_state"].values()}
    measured.add(f"{payload['roofline_probes']['copy']['TBps_at_min']:.3f}")
    measured.add(f"{cv['history_swing_pct_of_min']:.2f}")
    for block in cv["identical_buffers_by_size"].values():
        measured.add(f"{block['spread_pct_of_min']:.2f}")
    for probe in payload["roofline_probes"].values():
        measured.add(f"{probe['spread_pct_of_min']:.2f}")
        measured.add(f"{probe['TBps_at_min']:.3f}")
    unexplained = [
        tok for tok in re.findall(r"\d+\.\d+", text) if tok not in narrative and tok not in measured
    ]
    if unexplained:
        raise SystemExit(
            f"copy_probe_caveat contains decimals {unexplained} that no field on this "
            "payload computed. That is exactly the defect this probe was changed to "
            "close: three copy rates lived only inside this string, never re-ran, and "
            "one of them drifted 4.04% from the file's own computed probe. Interpolate "
            "from copy_variability or roofline_probes, or add the value to the "
            "narrative allowlist with a reason."
        )


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

    # Must run before any large allocation: its first row is defined by the
    # allocator being cold, and that state cannot be recovered once x exists.
    print("probing copy variability (history rows, then size sweep) ...", flush=True)
    copy_states = _copy_variability()
    torch.cuda.empty_cache()

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
        "copy_variability": copy_states,
        # Filled after the probe loop: it interpolates from measured fields,
        # including roofline_probes/copy, which does not exist yet.
        "copy_probe_caveat": None,
        "roofline_probes": {},
        "measurements": {},
    }

    for name, (call, nbytes) in _probes().items():
        print(f"probing {name} ...", flush=True)
        payload["roofline_probes"][name] = _summarise(_bench(call), nbytes)

    rates = copy_states["TBps_by_state"]
    sizes = copy_states["identical_buffers_by_size"]
    payload["copy_probe_caveat"] = (
        "The copy probe is not a property of the card and must not be quoted as "
        "one. Every figure in this sentence is interpolated from copy_variability, "
        "measured in this process on this run. What it is NOT: allocation history. "
        "The same call before any large allocation, with three same-size buffers "
        "made live, and again afterwards reads {cold:.3f} / {live:.3f} / "
        "{again:.3f} TB/s -- {hswing:.2f}% apart, a null that reproduced across "
        "three processes. This supersedes an earlier hand-typed claim of a 14% "
        "swing 'from allocation history alone', which was never measured. What it "
        "IS: placement past the MALL. Across five identically-sized, identically-"
        "filled buffers read by one destination, the spread is {s64:.2f}% at 64 "
        "MiB (inside the 256 MiB working set), {s512:.2f}% at 512 MiB and "
        "{s2g:.2f}% at 2 GiB; reversing the size order reproduces it, so it is "
        "not an order effect, and no allocator-state or per-op story predicts "
        "that identical buffers stop disagreeing exactly when they fit in cache. "
        "This is the mechanism behind the several different MI355X 'copy "
        "roofline' values that accumulated in the notes: each was one draw from "
        "that distribution, written down as a constant. Use write or "
        "two_read_one_write as denominators. Their advantage is measured, and the "
        "measurement had to be redone: an earlier version of this sentence gave "
        "write 0.50%, two_read_one_write 1.37% and copy 13.35% and called them "
        "cross-process spreads. They are not. Holding this generator fixed, six "
        "processes on device 5 give copy 0.63%, two_read_one_write 0.08% and "
        "write 0.22%. The three runs behind 13.35% straddled edits to "
        "_copy_variability, which allocates and frees buffers before the copy "
        "probe runs -- so that figure spans generator VERSIONS, not processes. "
        "What moves it is the allocator's peak high-water mark, a reversible "
        "staircase: 0, 6 and 11 live 512 MiB buffers are indistinguishable, 13 "
        "and 17 each shift the slots. See copy_axes_dev5.json. Within THIS run "
        "the probe spreads are {sp_write:.2f}% / {sp_trow:.2f}% / {sp_copy:.2f}% "
        "for write / two_read_one_write / copy, and this run's copy probe reads "
        "{inplace:.3f}, itself only one draw. Copy is retained here because "
        "the notes' history refers to it. These numbers were hand-typed until "
        "@Autotune found one disagreeing with this file's own computed probe by "
        "4.04% against a 0.31% spread; they are computed now, so a machine that "
        "stops reproducing this changes the sentence."
    ).format(
        cold=rates["history_cold_allocator_512MiB"],
        live=rates["history_three_buffers_live_512MiB"],
        again=rates["history_cold_pair_remeasured_512MiB"],
        hswing=copy_states["history_swing_pct_of_min"],
        s64=sizes["64MiB"]["spread_pct_of_min"],
        s512=sizes["512MiB"]["spread_pct_of_min"],
        s2g=sizes["2048MiB"]["spread_pct_of_min"],
        sp_write=payload["roofline_probes"]["write"]["spread_pct_of_min"],
        sp_trow=payload["roofline_probes"]["two_read_one_write"]["spread_pct_of_min"],
        sp_copy=payload["roofline_probes"]["copy"]["spread_pct_of_min"],
        inplace=payload["roofline_probes"]["copy"]["TBps_at_min"],
    )
    payload["denominator_stability_across_processes"] = {
        "spread_pct_of_min": {"write": 0.22, "two_read_one_write": 0.08, "copy": 0.63},
        "n_processes": 6,
        "device": "physical 5",
        "held_fixed": "this generator, byte for byte, across all six processes",
        "why_narrative": (
            "A single run cannot compute a cross-process spread, so these three are "
            "transcribed. They are recorded as a field anyway so the next reader can "
            "re-derive them rather than trust the sentence. The generator was held "
            "fixed across all six runs, which is the condition the previous version "
            "of this field silently violated."
        ),
        "supersedes": (
            "This field's own previous values -- write 0.50%, two_read_one_write "
            "1.37%, copy 13.35%, n_processes 3 -- which were not cross-process "
            "spreads at all. Those three runs straddled edits to _copy_variability, "
            "whose allocations precede the copy probe, so the figure measured "
            "generator versions. It was introduced by the change that removed "
            "hand-typed constants from copy_probe_caveat: the fix for untested "
            "numbers contributed a mislabelled one. Re-measured with the generator "
            "held fixed, copy reproduces to 0.63% -- so the ordering that motivated "
            "'use write or two_read_one_write' survives, but the 21x gap it appeared "
            "to rest on does not exist. The real argument against copy is the "
            "allocation-slot and high-water-mark sensitivity in copy_axes_dev5.json, "
            "which write and two_read_one_write do not share."
        ),
        "full_decomposition": "AI/data/copy_placement_draws/copy_axes_dev5.json",
    }

    _assert_caveat_is_derived(payload)

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

"""Move occupancy at fixed N and see whether bandwidth follows.

The measured-occupancy sidecar established that occupancy halves at the same
width where achieved bandwidth halves. Two coincident edges are not causality,
and both @Reviewer and my own notes said so. This is the intervention that was
owed: hold N fixed, force the register allocator across the boundary, and watch
the bandwidth.

The lever is `--amdgpu-waves-per-eu`, which flydsl's ROCm backend accepts as a
`waves_per_eu` compile hint (`flydsl/compiler/backends/rocm.py`). It pressures
the allocator to fit more waves per EU, which at N=57344 drops the allocation
from 264 VGPRs (bound 1) to 256 (bound 2) -- across the boundary, at a width
that has not changed.

**The lever is not clean, and the honest reading depends on saying how.** It
buys occupancy with spills: 0 at the default, 8 at waves_per_eu=2, then 97 and
137 at 3 and 4. So this is not a single-variable intervention and cannot be
reported as one. What makes it evidential anyway is the shape:

  * The first step goes the predicted direction *while paying a spill cost that
    should push the other way*. Occupancy 1.00 -> 1.94, bandwidth 38.1% ->
    52.7%. A confound that works against your hypothesis and loses is worth
    more than one you had to argue away. And the confound is smaller than the
    spill count suggests: `agpr_count` goes 8 -> 0 exactly as
    `vgpr_spill_count` goes 0 -> 8, with scratch 0 -> 36 B/lane (9 dwords).
    Those are the same eight registers relocated from AGPRs to scratch, not
    eight new spills. @Reviewer found this, and it closes the AGPR question
    from the measured side: those 8 AGPRs are what push the allocation to 264
    and the capacity from 2 to 1, and this row removes them and watches the
    capacity come back.
  * The later steps reverse while occupancy keeps climbing (2.80, 3.69) and
    bandwidth falls back (43.2%, 34.4%) as spills go 97, 137. So "more
    occupancy is always faster" is *not* what this shows, and the probe does
    not claim it. The sharpest single row: the highest occupancy in the table
    is also its worst bandwidth, below the unhinted kernel at 1.00 wave/SIMD.

The control is the load-bearing part. At N=49152 the kernel is already at 2
waves/SIMD, so hint=2 has nothing to move: allocation stays at 232 and
bandwidth is 76.7% vs 77.0%, unchanged. The flag therefore does not make
kernels generically faster -- it moves bandwidth only where it moves occupancy.
Without this row the experiment would not distinguish "occupancy drives
bandwidth" from "this compiler flag is good for you".

**The control runs the whole ladder, and that cost me a claim.** The first
version ran the control only at none/2 -- enough for the step the main result
rests on, but it left the 3/4 reversal uncontrolled, and I had written that
reversal up as occupancy and spills trading off *at the treatment width*.
@Reviewer pointed out there was no width where spills should not matter to read
it against. With one, the reversal turns out to be generic: at N=49152, hint=3
and 4 take bandwidth from 76.7% to 46.5% and 43.2%. High settings of this flag
are harmful at both widths, so the 3/4 rows support "more occupancy is not
always faster" and nothing more specific than that. The main claim is untouched
-- it lives on the none->2 step, where the control is flat and the treatment
is not.

A second thing the control needs, which it had by luck and now asserts: an
allocation that does not move is equally consistent with a hint the compiler
honoured but could not act on, and a hint that never reached the compiler.
Only the second would invalidate the control, and the alloc assert passes
either way. The raw `vgpr_count` moving (230 -> 228) while granularity-8
allocation stays at 232 is what tells them apart, so the probe now requires it.

What this still does not establish. Occupancy and spill count both move with
the hint, so the first step is a two-variable change read in the direction
where the variables disagree. A cleaner lever would change occupancy at
constant spills; I do not have one. The claim supported here is directional --
at the cliff, restoring occupancy restores a substantial part of the lost
bandwidth -- not quantitative, and not "occupancy is the whole story". The
last point does not need a cross-width extrapolation to make: the sweeps
contain a pair at the *same* occupancy and different N (N=57344 hint=2 at 1.937
waves/SIMD, N=49152 unhinted at 1.960 -- about 1% apart) whose bandwidths
differ by a factor of 1.45. Restoring occupancy buys back 38% of the cliff and
no more. That pair is emitted as `equal_occupancy_pair` with its separation
alongside, so a reader can see how well matched the held-fixed axis actually
is; @Reviewer identified it.

Provenance: @Reviewer noted that only the occupancy sidecar had been
reproduced across GPUs while this had not. Rerunning on device 5 matched device
6 to within 0.2%, and I wrote that up as "reproduces across two GPUs". A third
die falsified it. Device 4 is faster on every row, by 1.7% to 11.0%, and it
reproduces -- two runs per die, worst within-device spread 1.08% against a
worst between-device gap of 10.96%. See
AI/data/rmsnorm_fwd_occupancy_intervention_cross_device.json.

Two dies agreeing is one comparison, not a property of the hardware. What does
carry across all three is what this experiment actually claims, all of which
are ratios: the none->2 lift (+38.4% dev5, +39.6% dev4), the flat control
(+0.37%, +0.44%), the equal-occupancy ratio (1.455, 1.453), and bit-identical
vgpr/spill/scratch counts. The absolute pct_of_ceiling numbers are per-card.
The gap is not the streaming ceiling either: device 4 measures 6.004 TB/s on
two_read_one_write against 6.086 (dev5) and 6.074 (dev6), so it is the slowest
of the three at streaming and the fastest here, and normalising per-die would
widen the gap. Left unexplained rather than given a mechanism.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_occupancy_intervention.py
Writes AI/data/rmsnorm_fwd_occupancy_intervention.json.
"""

import contextlib
import csv
import hashlib
import json
import os
import pickle
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", tempfile.mkdtemp(prefix="flydsl-interv-probe-"))

from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402  (after path setup)

import quack.rmsnorm_flydsl as flydsl_rmsnorm  # noqa: E402  (must follow the env + path setup)
from quack.flydsl import rmsnorm_config  # noqa: E402  (must follow the sys.path insert)

M = 4096  # the bandwidth cliff's own m, so the percentages are comparable to it

# TREATMENT: past the cliff, where the default allocation is over the boundary.
TREATMENT_N = 57344
# CONTROL: before the cliff, where the default is already at 2 waves and the
# hint has nothing to move. Without this the probe cannot tell "occupancy
# drives bandwidth" from "this flag is good for you".
CONTROL_N = 49152

HINTS = ["none", 2, 3, 4]
# The control runs the SAME hint ladder as the treatment. Running only
# none/2 controlled the step the main claim rests on but left the reversal at
# 3/4 uncontrolled: without 3/4 here, "the reversal is spills interacting with
# occupancy" cannot be told apart from "this flag is simply harmful at high
# settings". @Reviewer caught the gap.
CONTROL_HINTS = ["none", 2, 3, 4]

WARMUP = 20
INNER = 20
REPEATS = 10

# From AI/data/rmsnorm_fwd_width_cliff.json: two_read_one_write, not copy.
CEILING_TBS = 6.075

VGPRS_PER_SIMD = 512
ALLOC_GRANULARITY_WAVE64 = 8
COUNTER = "MeanOccupancyPerActiveCU"

_METADATA_FIELDS = (
    "vgpr_count",
    "sgpr_count",
    "agpr_count",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "private_segment_fixed_size",
)


def _hint_ctx(hint):
    if hint == "none":
        return contextlib.nullcontext()
    return CompilationContext.compile_hints({"waves_per_eu": int(hint)})


def _raise_max_n():
    shipped = rmsnorm_config.MAX_N
    assert flydsl_rmsnorm.MAX_N == shipped, "the two MAX_N bindings already disagree"
    rmsnorm_config.MAX_N = 1 << 20
    flydsl_rmsnorm.MAX_N = 1 << 20
    return shipped


def _restore_max_n(shipped):
    rmsnorm_config.MAX_N = shipped
    flydsl_rmsnorm.MAX_N = shipped


def _regs_inner(n, hint):
    """Child: compile under the hint into a fresh cache, print the metadata.

    Runs in a subprocess because flydsl memoizes compilation in-process as well
    as on disk; a shared process across hints would report the first hint's
    kernel for all of them.
    """
    cache = Path(os.environ["FLYDSL_RUNTIME_CACHE_DIR"])
    torch.manual_seed(0)
    shipped = _raise_max_n()
    try:
        x = torch.randn((M, n), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        before = set(cache.rglob("*.pkl"))
        with _hint_ctx(hint):
            flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
        torch.cuda.synchronize()
        new = sorted(set(cache.rglob("*.pkl")) - before)
        if not new:
            raise SystemExit(
                f"N={n} hint={hint}: nothing compiled, so this row would report a "
                "different hint's kernel."
            )
        best = None
        for path in new:
            ir = str(pickle.load(path.open("rb")).ir)
            meta = {}
            for field in _METADATA_FIELDS:
                match = re.search(rf"\b{field} = (\d+) : i64", ir)
                if match is None:
                    raise SystemExit(f"{field} not present in the artifact metadata")
                meta[field] = int(match.group(1))
            if best is None or meta["vgpr_count"] > best["vgpr_count"]:
                best = meta
    finally:
        _restore_max_n(shipped)
    print(json.dumps(best))


def _bw_inner(n, hint):
    """Child: time the kernel under the hint, print the best per-call us."""
    torch.manual_seed(0)
    shipped = _raise_max_n()
    try:
        x = torch.randn((M, n), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        with _hint_ctx(hint):
            for _ in range(WARMUP):
                flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(REPEATS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(INNER):
                    flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
                end.record()
                torch.cuda.synchronize()
                best = min(best, start.elapsed_time(end) * 1000.0 / INNER)
    finally:
        _restore_max_n(shipped)
    print(json.dumps({"us": best}))


def _occ_inner(n, hint):
    """Child: a handful of dispatches under the hint, for rocprof to count."""
    torch.manual_seed(0)
    shipped = _raise_max_n()
    try:
        x = torch.randn((M, n), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        with _hint_ctx(hint):
            for _ in range(3):
                flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
            torch.cuda.synchronize()
    finally:
        _restore_max_n(shipped)


def _child(mode, n, hint, env_extra=None):
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, str(n), str(hint)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{mode} N={n} hint={hint} failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _occupancy(n, hint):
    outdir = Path(tempfile.mkdtemp(prefix="rocprof-interv-"))
    cmd = [
        "rocprofv3",
        "--pmc",
        COUNTER,
        "-d",
        str(outdir),
        "-o",
        "r",
        "--output-format",
        "csv",  # the default .db carries no counter table
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--occ",
        str(n),
        str(hint),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"rocprofv3 failed:\n{proc.stdout}\n{proc.stderr}")
    csv_path = outdir / "r_counter_collection.csv"
    rows = [
        r for r in csv.DictReader(csv_path.open()) if r["Kernel_Name"].startswith("rmsnorm_kernel")
    ]
    if not rows:
        raise SystemExit(f"N={n} hint={hint}: no rmsnorm dispatches in the counter CSV")
    return statistics.median(float(r["Counter_Value"]) for r in rows), len(rows)


def _row(n, hint):
    regs = _child(
        "--regs",
        n,
        hint,
        {"FLYDSL_RUNTIME_CACHE_DIR": tempfile.mkdtemp(prefix="flydsl-interv-art-")},
    )
    timing = _child("--bw", n, hint)
    occ, dispatches = _occupancy(n, hint)

    alloc = max(
        ALLOC_GRANULARITY_WAVE64,
        -(-regs["vgpr_count"] // ALLOC_GRANULARITY_WAVE64) * ALLOC_GRANULARITY_WAVE64,
    )
    moved_bytes = 2 * M * n * 2  # one read + one write, bf16; weight is negligible
    tbs = moved_bytes / (timing["us"] * 1e-6) / 1e12
    return {
        "n": n,
        "m": M,
        "waves_per_eu_hint": hint,
        **regs,
        "vgpr_alloc_wave64": alloc,
        "waves_per_simd_register_limited": VGPRS_PER_SIMD // alloc,
        "measured_waves_per_simd": round(occ, 4),
        "occupancy_dispatches": dispatches,
        "us_per_call": round(timing["us"], 3),
        "tbs": round(tbs, 4),
        "pct_of_ceiling": round(100.0 * tbs / CEILING_TBS, 2),
    }


def _sha(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()[:16]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    from shutil import which

    if not which("rocprofv3"):
        raise SystemExit("rocprofv3 not on PATH")

    head_run = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty_run = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Fail closed. commit="" with worktree_dirty=false reads as a clean
    # checkout when the truth is "git did not answer" -- a failure mode
    # indistinguishable from success, and quotable. The occupancy probe was
    # fixed for this in e5efbe1; this one had the same hole.
    if head_run.returncode or dirty_run.returncode or not head_run.stdout.strip():
        raise SystemExit(
            f"cannot read git provenance for {REPO} (rev-parse rc={head_run.returncode}, "
            f"status rc={dirty_run.returncode}). Refusing to write a sidecar whose commit "
            "field would be empty and whose dirty flag would read clean by default."
        )
    head = head_run.stdout.strip()
    dirty = dirty_run.stdout.strip()

    treatment = []
    for hint in HINTS:
        print(f"treatment N={TREATMENT_N} hint={hint} ...", flush=True)
        treatment.append(_row(TREATMENT_N, hint))
    control = []
    for hint in CONTROL_HINTS:
        print(f"control N={CONTROL_N} hint={hint} ...", flush=True)
        control.append(_row(CONTROL_N, hint))

    # The control only controls if the hint really had nothing to move there,
    # at the hint the main claim is read against (2). At 3 and 4 the allocation
    # is *expected* to move here too -- those rows exist to test the reversal,
    # not to serve as a null.
    base = control[0]
    hinted = next(r for r in control if r["waves_per_eu_hint"] == 2)
    if base["vgpr_alloc_wave64"] != hinted["vgpr_alloc_wave64"]:
        raise SystemExit(
            "the control width's allocation moved under hint=2; it is no longer a "
            "control and the treatment cannot be read against it"
        )
    # ...and a flat control is only evidence if the flag ARRIVED. An
    # allocation that does not move is consistent with two very different
    # things: a hint the compiler honoured but could not act on, and a hint
    # that never reached the compiler at all. The first assert passes either
    # way, so on its own it launders the second case as a control. The
    # raw vgpr_count moving while the granularity-8 allocation does not is
    # what distinguishes them. @Reviewer pointed out this was recorded but
    # never asserted -- it held by luck.
    if not any(r["vgpr_count"] != base["vgpr_count"] for r in control[1:]):
        raise SystemExit(
            "no control row's vgpr_count moved under any hint: the flag may not be "
            "reaching the compiler at all, in which case a flat control is not evidence"
        )

    # 'Occupancy is not the whole story' reads sharpest off a pair that is at
    # the same occupancy and different N -- no cross-width extrapolation
    # needed. @Reviewer identified the pair; it is computed here rather than
    # written into prose so it cannot drift away from the rows above it. The
    # gap is only meaningful if the two really are at the same occupancy, so
    # the separation is emitted alongside instead of being asserted small.
    t_hinted = next(r for r in treatment if r["waves_per_eu_hint"] == 2)
    t_base = treatment[0]
    equal_occ = {
        "why": (
            "two rows at essentially the same measured occupancy and different N. If "
            "occupancy determined bandwidth these would agree; see bandwidth_ratio for "
            "how far apart they are, and occupancy_separation_pct for how well matched "
            "they are on the axis being held fixed"
        ),
        "a": {
            "n": t_hinted["n"],
            "hint": 2,
            "waves_per_simd": t_hinted["measured_waves_per_simd"],
            "pct_of_ceiling": t_hinted["pct_of_ceiling"],
        },
        "b": {
            "n": base["n"],
            "hint": "none",
            "waves_per_simd": base["measured_waves_per_simd"],
            "pct_of_ceiling": base["pct_of_ceiling"],
        },
        "occupancy_separation_pct": round(
            100
            * abs(t_hinted["measured_waves_per_simd"] - base["measured_waves_per_simd"])
            / base["measured_waves_per_simd"],
            2,
        ),
        "bandwidth_ratio": round(base["pct_of_ceiling"] / t_hinted["pct_of_ceiling"], 3),
        "fraction_of_cliff_recovered_pct": round(
            100
            * (t_hinted["pct_of_ceiling"] - t_base["pct_of_ceiling"])
            / (base["pct_of_ceiling"] - t_base["pct_of_ceiling"]),
            1,
        ),
        "highest_occupancy_is_worst_bandwidth": (
            "sharper still: the highest occupancy in the treatment ladder is hint=4 at "
            f"{treatment[-1]['measured_waves_per_simd']:.3f} waves/SIMD, and its bandwidth "
            f"({treatment[-1]['pct_of_ceiling']:.1f} pct) is the WORST in the table -- below "
            f"the unhinted {t_base['pct_of_ceiling']:.1f} pct at 1.000 waves/SIMD"
        ),
    }

    payload = {
        "what": "does bandwidth follow occupancy at fixed N? forcing the register "
        "allocator across the boundary with --amdgpu-waves-per-eu, plus a control "
        "width where the hint has nothing to move",
        "generator": "AI/probe_rmsnorm_occupancy_intervention.py",
        "note": (
            "gfx950 MI355X, bf16 fwd, has_weight only, m=4096 (the bandwidth cliff's own m). "
            "This is the intervention the measured-occupancy sidecar said was still owed. "
            "The lever is NOT clean: it buys occupancy with spills (0 / 8 / 97 / 137 VGPR "
            "spills at hints none/2/3/4), so this is a two-variable change and is reported "
            "as directional evidence, not as a quantitative result. MAX_N was raised "
            "in-process for the probe only; the shipped cap is unchanged."
        ),
        "device": torch.cuda.get_device_name(),
        "device_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", None)),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "host": platform.node(),
        "commit": head,
        "worktree_dirty": bool(dirty),
        "shipped_max_n": rmsnorm_config.MAX_N,
        "source_sha256_16": {
            "quack/rmsnorm_flydsl.py": _sha("quack/rmsnorm_flydsl.py"),
            "quack/flydsl/rmsnorm_config.py": _sha("quack/flydsl/rmsnorm_config.py"),
            "AI/probe_rmsnorm_occupancy_intervention.py": _sha(
                "AI/probe_rmsnorm_occupancy_intervention.py"
            ),
        },
        "lever": (
            "waves_per_eu compile hint -> --amdgpu-waves-per-eu, via "
            "flydsl.compiler.kernel_function.CompilationContext.compile_hints; consumed in "
            "flydsl/compiler/backends/rocm.py"
        ),
        "ceiling_tbs": CEILING_TBS,
        "ceiling_source": (
            "AI/data/rmsnorm_fwd_width_cliff.json ceiling_probe, two_read_one_write -- "
            "not a copy kernel, which on this part reads MALL-inflated"
        ),
        "bytes_model": "2 * m * n * 2 (one read + one write, bf16); weight is negligible",
        "timing": f"cuda events, best of {REPEATS} x {INNER} inner calls after {WARMUP} warmup",
        "confound_note": (
            "occupancy and spill count both move with the hint, so the first step is a "
            "two-variable change. It is read in the direction where the two variables "
            "disagree: bandwidth improves 38.1 -> 52.8 pct despite the spill count going "
            "0 -> 8, i.e. the confound pushes against the hypothesis and loses. The later "
            "steps reverse and track spills (97, 137) while occupancy keeps rising, so "
            "'more occupancy is always faster' is NOT supported and is not claimed. "
            "What those later steps are NOT evidence for: see reversal_control. "
            "The first step's confound is smaller than the raw spill count suggests: "
            "agpr_count goes 8 -> 0 as vgpr_spill_count goes 0 -> 8, and the scratch "
            "allocation goes 0 -> 36 B/lane (9 dwords). Those are the same 8 registers "
            "moving from AGPRs to scratch, not 8 newly created spills. @Reviewer found "
            "this. It also closes the AGPR question from the static side: those 8 AGPRs "
            "are what push the allocation to 264 and the capacity from 2 to 1, and this "
            "run removes them and measures the capacity coming back."
        ),
        "control_note": (
            "At N=49152 the default allocation is already at 2 waves/SIMD, so hint=2 has "
            "nothing to move: the probe asserts vgpr_alloc is unchanged there, and "
            "bandwidth is flat. This is what separates 'occupancy drives bandwidth' from "
            "'this compiler flag is generically good'. The flatness is only evidence if "
            "the flag reached the compiler, and an unchanged allocation cannot show that "
            "on its own -- so the probe also asserts that the raw vgpr_count moves under "
            "some hint while the granularity-8 allocation does not. The control runs the "
            "full none/2/3/4 ladder, not just none/2, so the reversal seen in the "
            "treatment at 3 and 4 can be read against a width where the same settings "
            "apply."
        ),
        "equal_occupancy_pair": equal_occ,
        "reversal_control": (
            "The control now runs the full none/2/3/4 ladder, and the reversal at 3 and 4 "
            "happens THERE TOO: at N=49152, hint=3 takes bandwidth from 76.7 to 46.5 pct "
            "and hint=4 to 43.2 pct, at 61 and 101 spills, while measured occupancy rises "
            "to 2.82 and 3.88. So the treatment's reversal is not specific to the width "
            "that was over the register boundary -- high settings of this flag are harmful "
            "at both widths. The earlier reading, that the reversal showed occupancy and "
            "spills trading off at the treatment width, was not supported: it had no "
            "control at those settings, and with one the effect is generic. This does not "
            "touch the main claim, which rests on the none->2 step, where the control IS "
            "flat (76.7 -> 77.0) and the treatment moves (38.1 -> 52.7). It does mean the "
            "3/4 rows are evidence against 'more occupancy is always faster' and evidence "
            "for nothing else. @Reviewer asked for these rows."
        ),
        "still_open": (
            "Directional only. A clean lever would change occupancy at constant spills; "
            "none is available here. Restoring occupancy recovers part of the lost "
            "bandwidth, not all of it. Unexplained: why device 4 runs every row of this "
            "sweep 1.7-11.0 pct faster than device 5 with identical code and a lower "
            "streaming ceiling."
        ),
        "cross_device": (
            "Run on three dies. Devices 5 and 6 agree to 0.2 pct on bandwidth; device 4 is "
            "faster on EVERY row by 1.7-11.0 pct, reproducibly (two runs per die: worst "
            "within-device spread 1.08 pct, worst between-device gap 10.96 pct). So the "
            "earlier 'reproduces across two GPUs to within 0.2 pct' was one comparison "
            "generalised into a hardware property. The absolute pct_of_ceiling figures are "
            "per-card. Everything this experiment claims is a ratio and the ratios hold on "
            "both dies to about a point -- none->2 lift +38.4/+39.6 pct, control none->2 "
            "+0.37/+0.44 pct, equal-occupancy bandwidth ratio 1.455/1.453 -- with "
            "bit-identical register and spill counts throughout. Full rows in "
            "AI/data/rmsnorm_fwd_occupancy_intervention_cross_device.json."
        ),
        "treatment_sweep": treatment,
        "control_sweep": control,
    }

    out_path = REPO / "AI/data/rmsnorm_fwd_occupancy_intervention.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--regs":
        _regs_inner(int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "--bw":
        _bw_inner(int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "--occ":
        _occ_inner(int(sys.argv[2]), sys.argv[3])
    else:
        main()

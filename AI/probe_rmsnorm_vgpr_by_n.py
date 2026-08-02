"""Regenerate the forward VGPR-by-N sidecar, with its own provenance.

`AI/data/rmsnorm_fwd_vgpr_by_n.json` backs the register-budget section of
`AI/flydsl_rmsnorm_notes.md`. Like the roofline sidecar before it, it was
produced by an ad-hoc script that no longer existed: the numbers could be read
but not re-derived, and it carried no host, device, commit or source identity.
This closes that half of @Reviewer's evidence blocker.

Where the numbers come from. FlyDSL pickles a `CompiledArtifact` per cache
entry under `$FLYDSL_RUNTIME_CACHE_DIR` (default `~/.flydsl/cache`), and the
artifact's MLIR carries a `gpu.kernel_metadata` attribute dictionary with the
register counts in structured form:

    metadata = {agpr_count = 0 : i64, ..., sgpr_count = 16 : i64,
                sgpr_spill_count = 0 : i64, vgpr_count = 10 : i64, ...}

That attribute block is what this probe parses. The same numbers also appear
in the embedded msgpack `amdhsa` note inside the ELF, but as escaped binary
(`\\AB.vgpr_count\\0A`), which is a worse thing to regex. The two agree.

**The derived occupancy column is where the previous version went wrong.** It
stored `floor(vgprs_per_simd / vgpr_alloc)` under the name `waves_per_simd`,
which gave 21, 21, 12 for N=1024/2048/4096 -- more waves than the hardware can
hold. This host reports `max_waves_per_simd 8` on every GPU node. The register
file's limit and the achievable occupancy are two different quantities and only
one of them was being computed, so both are now emitted under names that say
which is which, and the hardware cap is read from the KFD topology at runtime
rather than assumed.

Neither column is a measurement. Both are arithmetic on a register count, and
they ignore workgroup-slot, LDS and barrier limits, so the upper bound can be
loose. `agpr_count` is recorded now too -- on gfx950 the AGPRs share the
register file, so a nonzero value would make the VGPR-only arithmetic wrong,
and the old sidecar could not even tell you it was zero.

N above the shipped `MAX_N` requires raising the cap in-process. That is done
here for the probe only; nothing is written back and the shipped constant is
untouched.

Run:  HIP_VISIBLE_DEVICES=<idle> python AI/probe_rmsnorm_vgpr_by_n.py
Writes AI/data/rmsnorm_fwd_vgpr_by_n.json.
"""

import hashlib
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Point the runtime cache at an empty directory BEFORE importing flydsl, or
# every row reads "cache hit" and the probe silently measures nothing -- it
# would report the register counts of whatever happened to be compiled
# earlier, which is a wrong answer rather than a missing one.
_CACHE_DIR = tempfile.mkdtemp(prefix="flydsl-vgpr-probe-")
os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = _CACHE_DIR

import quack.rmsnorm_flydsl as flydsl_rmsnorm  # noqa: E402  (must follow the env + path setup)
from quack.flydsl import rmsnorm_config  # noqa: E402  (must follow the sys.path insert)

# (N, m) pairs, m chosen to keep the working set roughly constant.
SHAPES = [
    (1024, 16384),
    (2048, 8192),
    (4096, 4096),
    (8192, 2048),
    (16384, 1024),
    (32768, 512),
    (49152, 341),
    (57344, 292),
    (65536, 256),
]

VGPRS_PER_SIMD = 512
ALLOC_GRANULARITY_WAVE64 = 8

_METADATA_FIELDS = (
    "agpr_count",
    "sgpr_count",
    "sgpr_spill_count",
    "vgpr_count",
    "vgpr_spill_count",
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "max_flat_workgroup_size",
    "wavefront_size",
)


def _hw_max_waves_per_simd():
    """Read the cap from KFD rather than assuming it.

    Nodes with no GPU report 0; those are CPU nodes and are skipped. If the
    GPU nodes disagree the probe refuses rather than picking one.
    """
    values = set()
    for props in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        for line in props.read_text().splitlines():
            if line.startswith("max_waves_per_simd "):
                value = int(line.split()[1])
                if value:
                    values.add(value)
    if len(values) != 1:
        raise SystemExit(f"could not read a single max_waves_per_simd from KFD: {values or 'none'}")
    return values.pop()


def _parse_metadata(ir):
    """Pull the gpu.kernel_metadata attribute dictionary out of the MLIR."""
    found = {}
    for field in _METADATA_FIELDS:
        match = re.search(rf"\b{field} = (\d+) : i64", ir)
        if match is None:
            raise SystemExit(f"{field} not present in the compiled artifact's metadata")
        found[field] = int(match.group(1))
    return found


def _compile_and_read(n, m):
    """Compile one forward and read the register counts back out of the cache."""
    before = set(Path(_CACHE_DIR).rglob("*.pkl"))
    x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    flydsl_rmsnorm.rmsnorm(x, weight, eps=1e-6)
    torch.cuda.synchronize()
    new = sorted(set(Path(_CACHE_DIR).rglob("*.pkl")) - before)
    if not new:
        raise SystemExit(
            f"N={n}: nothing was compiled, so this row would report a stale "
            "kernel's registers. The runtime cache was not empty."
        )
    # One forward may compile more than one kernel; take the widest, which is
    # the normalization itself rather than any helper.
    best = None
    for path in new:
        artifact = pickle.load(path.open("rb"))
        meta = _parse_metadata(str(artifact.ir))
        if best is None or meta["vgpr_count"] > best["vgpr_count"]:
            best = meta
    return best, len(new)


def _sha(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()[:16]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.manual_seed(0)
    hw_cap = _hw_max_waves_per_simd()

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    # `quack.rmsnorm_flydsl` does `from ... import MAX_N`, so it holds its own
    # binding and patching only the defining module would leave the validator
    # rejecting wide rows. Both are raised, and both are restored.
    shipped_max_n = rmsnorm_config.MAX_N
    assert flydsl_rmsnorm.MAX_N == shipped_max_n, "the two MAX_N bindings already disagree"
    rmsnorm_config.MAX_N = 1 << 20  # probe only; nothing is written back
    flydsl_rmsnorm.MAX_N = 1 << 20

    rows = []
    try:
        for n, m in SHAPES:
            print(f"compiling N={n} ...", flush=True)
            meta, kernels = _compile_and_read(n, m)
            alloc = max(
                ALLOC_GRANULARITY_WAVE64,
                -(-meta["vgpr_count"] // ALLOC_GRANULARITY_WAVE64) * ALLOC_GRANULARITY_WAVE64,
            )
            register_limited = VGPRS_PER_SIMD // alloc
            rows.append(
                {
                    "n": n,
                    "m": m,
                    "kernels_compiled": kernels,
                    **meta,
                    "vgpr_alloc": alloc,
                    "waves_per_simd_register_limited": register_limited,
                    "occupancy_upper_bound_waves_per_simd": min(register_limited, hw_cap),
                }
            )
    finally:
        rmsnorm_config.MAX_N = shipped_max_n
        flydsl_rmsnorm.MAX_N = shipped_max_n

    payload = {
        "what": "forward register usage against row width, and the occupancy "
        "upper bound it implies, for the register-budget section",
        "generator": "AI/probe_rmsnorm_vgpr_by_n.py",
        "note": (
            "gfx950 MI355X, bf16 fwd, has_weight only. Register counts parsed from the "
            "gpu.kernel_metadata attribute dictionary in flydsl's jit cache pickle. "
            "FLYDSL_RUNTIME_CACHE_DIR is pointed at a fresh empty directory and each row "
            "asserts that something was actually compiled, so a cache hit cannot masquerade "
            "as a measurement. MAX_N was raised in-process for the probe only; the shipped "
            "cap is unchanged."
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
        "shipped_max_n": shipped_max_n,
        "source_sha256_16": {
            "quack/rmsnorm_flydsl.py": _sha("quack/rmsnorm_flydsl.py"),
            "quack/flydsl/rmsnorm_config.py": _sha("quack/flydsl/rmsnorm_config.py"),
            "AI/probe_rmsnorm_vgpr_by_n.py": _sha("AI/probe_rmsnorm_vgpr_by_n.py"),
        },
        "vgprs_per_simd": VGPRS_PER_SIMD,
        "alloc_granularity_wave64": ALLOC_GRANULARITY_WAVE64,
        "max_waves_per_simd_hw": hw_cap,
        "max_waves_per_simd_hw_source": (
            "/sys/class/kfd/kfd/topology/nodes/*/properties, read at runtime; all GPU nodes agree"
        ),
        "derived_field_note": (
            "occupancy_upper_bound_waves_per_simd = min(floor(vgprs_per_simd / vgpr_alloc), "
            "max_waves_per_simd_hw), where vgpr_alloc = roundup(vgpr_count, "
            "alloc_granularity_wave64). NEITHER column is measured occupancy: both are "
            "arithmetic on a register count, and both ignore workgroup-slot, LDS and barrier "
            "limits, so the bound can be loose. A previous version of this file stored the "
            "uncapped value under the name 'waves_per_simd', giving 21/21/12 for "
            "N=1024/2048/4096, which exceeds what the hardware can hold; @Reviewer caught it."
        ),
        "rows": rows,
    }

    out_path = REPO / "AI/data/rmsnorm_fwd_vgpr_by_n.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

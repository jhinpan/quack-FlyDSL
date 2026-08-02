# Copyright (c) 2026, Tri Dao.

"""Provider-lazy plain RMSNorm benchmark for the FlyDSL ROCm backend."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence
import uuid
import warnings


COMPACT_SHAPES = (
    (1, 4096),
    (256, 4096),
    (512, 4096),
    (4096, 3000),
    (4096, 4096),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
)
DTYPE_WEIGHT_MODES = (
    ("float16", "same"),
    ("float16", "float32"),
    ("bfloat16", "same"),
    ("bfloat16", "float32"),
    ("float32", "same"),
)
# Multiple of the last-level cache a rotation must exceed before the evictor is
# considered unnecessary. See the gate comment in _run() for why 2 and not 1.
EVICTOR_LLC_MARGIN = 2
OPERATIONS = ("fwd", "bwd")
PROVIDERS = ("flydsl", "quack", "torch")
# Schema history, so a consumer can tell what a given artifact's fields mean:
#   v2  l2_target_bytes (actually carrying 3 x LLC), l2_eviction_between_calls
#   v3  l2_target_bytes -> rotation_target_bytes; + evictor_threshold_bytes,
#       last_level_cache_provenance, rotation_target_bytes, evictor_gate
#   v4  l2_eviction_between_calls -> evictor_ran_per_rotation
#   v5  + peak_bw_probe (which probe peak_bw_pct's denominator came from)
#
# v4 is a rename with no behavioural change, and it is a *breaking* rename on
# purpose. The old name asserted the evictor ran between individually timed
# calls; it never did -- it runs once per rotation, before the event window
# opens. v3 corrected that in the prose methodology string while leaving the
# field name saying the opposite, so the machine-readable contract still
# claimed the thing the prose had just withdrawn. @Reviewer's blocker 3. No
# alias is kept: a consumer reading `l2_eviction_between_calls` should fail
# loudly against a v4 artifact rather than silently read a field whose meaning
# it has wrong. The v2/v3 artifacts under AI/gate_llc_before_after/ keep the
# old name and are frozen; their schema_version distinguishes them.
#
# v5 is additive. It exists because `comparison_scope` told the reader to
# compare `peak_bw_pct` across vendors, and that column's denominator is
# whichever of the three probes won on that host -- two_read_one_write on H200,
# write on MI355X, measured the same day with this harness on both.
#
# What v5 adds is the probe's *name*. The first version of this note claimed a
# reader with only results.csv "could not see that the two percentages divide
# by different references", and that was false: v4 already wrote peak_bw_gbps
# on every row, 4314.018124 on the H200 file and 6664.195243 on the MI355X one,
# so the difference was visible and pct was recomputable from the CSV alone.
# @Reviewer checked and refused the overstatement. The real gain is semantic:
# 4314 vs 6664 shows the denominators differ, while `two_read_one_write` vs
# `write` says *why*, and distinguishes "different probe won" from "same probe,
# different hardware" -- which are different facts about the comparison and
# were not separable from the numbers alone. Recording it still does not make
# the columns comparable; it makes the incomparability legible instead of
# merely visible. Overstating an increment as an absence is the same failure
# this schema exists to prevent, pointed the other way.
SCHEMA_VERSION = 5
RESULT_FIELDS = (
    "schema_version",
    "provider",
    "provider_detail",
    "operation",
    "m",
    "n",
    "activation_dtype",
    "weight_dtype",
    "weight_mode",
    "eps",
    "cold_compile_ms",
    "cold_compile_reused",
    "median_us",
    "p10_us",
    "p90_us",
    "logical_bytes",
    "logical_gbps",
    "peak_bw_gbps",
    "peak_bw_pct",
    "peak_bw_probe",
    "rotation_buffers",
    "rotation_working_set_bytes",
    "rotation_target_bytes",
    "evictor_threshold_bytes",
    "evictor_ran_per_rotation",
    "timed_samples",
)

_ITEMSIZES = {"float16": 2, "bfloat16": 2, "float32": 4}


@dataclass(frozen=True)
class MatrixCell:
    m: int
    n: int
    activation_dtype: str
    weight_mode: str
    operation: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.m, self.n

    @property
    def weight_dtype(self) -> str:
        return self.activation_dtype if self.weight_mode == "same" else "float32"


@dataclass
class PreparedCase:
    calls: list[Callable[[], None]]
    reset: Callable[[], None]
    outputs: Callable[[], tuple[Any, ...]]
    provider_detail: str
    cold_compile_ms: float | None
    cold_compile_reused: bool


def build_matrix(
    shapes: Sequence[tuple[int, int]] = COMPACT_SHAPES,
    dtype_weight_modes: Sequence[tuple[str, str]] = DTYPE_WEIGHT_MODES,
    operations: Sequence[str] = OPERATIONS,
) -> list[MatrixCell]:
    """Build the distinct v1 matrix without duplicating fp32/fp32."""
    return [
        MatrixCell(m, n, activation_dtype, weight_mode, operation)
        for m, n in shapes
        for activation_dtype, weight_mode in dtype_weight_modes
        for operation in operations
    ]


def logical_bytes(
    operation: str,
    m: int,
    n: int,
    activation_itemsize: int,
    weight_itemsize: int,
) -> int:
    """Return provider-independent logical I/O bytes for plain RMSNorm."""
    activation_bytes = m * n * activation_itemsize
    weight_bytes = n * weight_itemsize
    if operation == "fwd":
        return 2 * activation_bytes + weight_bytes
    if operation == "bwd":
        # Read x, dout, weight, and fp32 rstd; write dx and dweight.
        return 3 * activation_bytes + 2 * weight_bytes + m * 4
    raise ValueError(f"unsupported operation: {operation}")


def write_artifacts(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    environment: dict[str, Any],
) -> tuple[Path, Path]:
    """Write the stable CSV result contract and environment JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    environment_path = output_dir / "environment.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, environment_path


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_us(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_us": _percentile(values, 0.5),
        "p10_us": _percentile(values, 0.1),
        "p90_us": _percentile(values, 0.9),
    }


def _rotation_count(
    bytes_per_set: int,
    target_bytes: int,
    free_bytes: int,
    *,
    min_buffers: int,
    max_buffers: int,
) -> int:
    by_target = max(1, math.ceil(target_bytes / max(1, bytes_per_set)))
    by_memory = max(1, int(free_bytes * 0.2) // max(1, bytes_per_set))
    desired = max(min_buffers, by_target)
    return max(1, min(max_buffers, by_memory, desired))


def _torch_unique_id(properties: Any) -> int | None:
    """KFD's ``unique_id`` for this device, from torch's misnamed ``uuid``.

    The field is not a UUID. Its sixteen bytes are the ASCII text of a hex
    string -- ``b"a60c2956cd9dd4c5"`` -- which parses to exactly the
    ``unique_id`` KFD publishes for the same card (verified 8/8 on this host).
    Note this is NOT the value ``rocm-smi --showuniqueid`` prints, which is a
    third number agreeing with neither.

    Returns ``None`` rather than guessing if the field is missing or does not
    look like the hex text this decoding assumes.
    """
    raw = getattr(properties, "uuid", None)
    data = getattr(raw, "bytes", None)
    if data is None:
        return None
    try:
        text = bytes(data).decode("ascii").strip()
    except (UnicodeDecodeError, TypeError, ValueError):
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def _reported_device_uuid(properties: Any) -> str | None:
    """The device identity as torch reports it, for the artifact to record.

    Deliberately not ``_torch_unique_id``: that decodes the field the AMD way
    to match KFD's ``unique_id``, whereas this is provenance and has to work on
    whatever the runtime hands over. On ROCm the attribute is a bytes-like of
    ASCII hex text; on CUDA it is a real ``uuid.UUID``. A build exposing
    neither records ``None`` rather than a fabricated value.

    ``uuid.UUID`` also has a ``.bytes``, and the first version of this function
    reached for it first -- which decoded a CUDA UUID's raw binary as ASCII and
    returned mojibake where the artifact should have carried
    ``12345678-1234-...``. A test written for the vendor I was not on caught
    it. ``UUID`` is therefore checked by type before anything duck-typed: the
    two objects answer the same attribute name with different kinds of thing,
    which is exactly the case a hasattr check cannot distinguish.
    """
    raw = getattr(properties, "uuid", None)
    if raw is None:
        return None
    if isinstance(raw, uuid.UUID):
        return str(raw)
    data = getattr(raw, "bytes", None)
    if data is not None:
        try:
            text = bytes(data).decode("ascii").strip()
        except (UnicodeDecodeError, TypeError, ValueError):
            return str(raw)
        if text:
            return text
    return str(raw)


def evictor_is_needed(rotation_working_set_bytes: int, llc_bytes: int) -> bool:
    """Whether a rotation of this size still needs the cache evicted for it.

    A function rather than an inline expression so a test can call the real
    predicate. A test that re-implements the comparison only proves the test
    agrees with itself -- which is how the false claim about `32768x1024` fwd
    survived review in the first place.

    See the call site in ``_run`` for why the margin is 2 and what it does not
    do.
    """
    return rotation_working_set_bytes <= EVICTOR_LLC_MARGIN * llc_bytes


KFD_NODE_ROOT = "/sys/class/kfd/kfd/topology/nodes"

# Ceilings for the two cache fields, in the units KFD publishes them in. These
# bound what a cache hierarchy can physically be, not what the field is wide
# enough to hold -- a `size` of 2**32 KB fits the driver's u32 and is still not
# a cache. That distinction is the whole point: the width check stops a
# corrupt read, and these stop a well-formed impossible one.
#
# Live topology for scale: the deepest level published is 3 and the largest
# entry is 262144 KB (the 256 MiB MALL). 8 and 16 GiB leave several orders of
# magnitude of headroom for future parts while still refusing the failure that
# actually occurred -- a 4 TiB "last level" that sized every rotation buffer.
MAX_CACHE_LEVEL = 8
MAX_CACHE_SIZE_KB = 16 * 1024 * 1024


_KFD_LINE = re.compile(r"\A([a-z0-9_]+) (-?[0-9]+)\Z")


def _read_kfd_properties(path: str) -> tuple[dict[str, str], set[str]]:
    """Parse a KFD ``properties`` file strictly, returning (fields, malformed).

    ``dict(line.split()[:2] for line in handle if len(line.split()) >= 2)``
    was lenient in three ways that each turn a malformed file into a
    confident-looking read:

    - a **duplicate key silently last-wins**. Two ``size`` lines in one entry
      returned the second, so a file asserting both 262144 and 1 read as a
      clean 1 KB cache with nothing recorded.
    - **trailing junk was discarded** by the ``[:2]`` slice, so
      ``size 262144 extra`` was indistinguishable from ``size 262144``.
    - the value was handed to ``int()``, whose accepted grammar is wider than
      the driver's output: ``+262144``, ``262_144`` and ``" 262144 "`` all
      parse. ``262_144`` is the sharp one -- a file the driver could not have
      written produced a *correct-looking* number, which is the same
      right-answer-wrong-provenance shape as the laundering fix.

    Measured on the live topology before tightening, as with the widths:
    39702 lines across all nodes and their 4370 cache entries. Every line that
    is not exactly ``key<space>integer`` is a ``sibling_map`` CSV row (4370 of
    them, a field nothing here reads), and there are **zero** duplicate keys in
    any file. So the strict form reclassifies nothing that exists.

    A malformed or contradicted line **withholds that one field** rather than
    condemning the file, and the field's name is returned in the second element
    so the caller can tell *withheld* from *never present*.

    Both halves of that were learned by getting it wrong. The first version
    returned an anomaly list and every caller refused the whole node on
    ``anomalies[0]``, which quietly rebuilt the over-refusal 10d9480 had just
    fixed: a UID-matched node with a lexically bad ``domain`` was thrown away
    while the same node with an out-of-range ``domain -1`` was correctly kept,
    and a duplicated ``max_waves_per_simd`` -- a field nothing here reads --
    killed the node outright.

    The second version dropped the malformed field and returned the dict alone.
    That fixed the over-refusal and broke the opposite case: a node whose
    ``unique_id`` is unreadable became byte-identical to a node that never
    stated one, so instead of being recorded as unidentifiable it was quietly
    passed over and the read fell through to a same-BDF neighbour -- the
    wrong-device answer c597e11, then 337bdbd, then this. Which is this defect
    class again, at the smallest scale yet: *a field that failed to parse must
    not become indistinguishable from a field that was never there.* Absent and
    malformed get the same **treatment** at the call site; they are not the
    same **observation**, and the reader must not erase the difference before
    the caller has had a chance to weigh it.
    """
    fields: dict[str, str] = {}
    dropped: set[str] = set()
    with open(path) as handle:
        for line in handle:
            text = line.rstrip("\n")
            if not text:
                continue
            match = _KFD_LINE.match(text)
            if match is None:
                # sibling_map is a CSV row on every node and nothing here reads
                # it, so this is the common case and not in itself a problem.
                #
                # The key is recovered with whitespace-agnostic splitting even
                # though the match above deliberately is not: matching stays
                # strict (a padded or tab-separated line is malformed and must
                # not be accepted), while naming is best-effort, because this
                # string only decides which tag a reader sees. Splitting on a
                # literal space made `  size 262144  ` yield "" and `size\t...`
                # yield the whole line, so both reported as missing_size -- the
                # withheld-vs-absent conflation this function exists to prevent,
                # reintroduced in the label instead of in the value. A
                # whitespace-only line claims no field, so there is nothing to
                # withhold.
                parts = text.split(maxsplit=1)
                if parts:
                    dropped.add(parts[0])
                continue
            key, value = match.group(1), match.group(2)
            if key in fields and fields[key] != value:
                # Last-wins would pick one silently. Two different values for
                # one key is a contradiction in the file, and the honest read
                # of a contradiction is that the field is unavailable.
                dropped.add(key)
                continue
            fields[key] = value
    return {k: v for k, v in fields.items() if k not in dropped}, dropped


def _bounded(props: dict[str, str], name: str, ceiling: int) -> int | None:
    """A present, parseable, non-negative value strictly below ``ceiling``.

    The cache-entry counterpart to ``_field``. Same rule: absent, malformed and
    out-of-range are the same amount of evidence, which is none, so they take
    the same path and the caller records a skip.
    """
    if name not in props:
        return None
    try:
        value = int(props[name])
    except ValueError:
        return None
    return value if 0 <= value < ceiling else None


def _last_level_cache_bytes(
    torch: Any, properties: Any, node_root: str = KFD_NODE_ROOT
) -> tuple[int, dict[str, Any]]:
    """Bytes that must be turned over to actually miss the last-level cache.

    Returns ``(bytes, provenance)``. The provenance dict is not decoration: the
    fallback here returns a number that is *known wrong* on gfx950, so a caller
    that only sees the number cannot tell a real reading from a silent
    degradation. It is recorded in the artifact and, on gfx950, treated as
    fatal -- see ``_resolve_llc``.

    ``properties.L2_cache_size`` is not the last-level cache on a part with a
    memory-side cache behind L2. On gfx950 torch reports the 4 MiB per-XCD L2
    while a 256 MiB MALL sits behind it, so a rotation sized against 4 MiB stays
    resident and the benchmark measures cache-warm while reporting cache-cold.

    Nodes are matched by identity, not by index: ``HIP_VISIBLE_DEVICES``
    renumbers torch's ordinals but not KFD's nodes, so index matching reads
    another card's topology under any masking. On this host the two orders
    disagree already (torch 0 is rocm-smi GPU 3).

    ``unique_id`` is tried first and PCI ``domain:bus:device`` second. The PCI
    key is not unique under CPX, where the eight logical devices of one card
    share one address; ``unique_id`` distinguishes them. Ambiguity on either key
    is refused, not resolved by ``os.listdir`` order.
    """
    fallback = properties.L2_cache_size
    if not getattr(torch.version, "hip", None):
        return fallback, {"source": "torch_l2_fallback", "reason": "not_a_hip_build"}
    want_uid = _torch_unique_id(properties)
    # Absent on torch's side is also not zero. `getattr(..., 0)` defaulted a
    # missing pci_domain_id to 0 and then *compared* it against a node
    # asserting domain 0, so an unverifiable field read as a verified match.
    # Same defect as the KFD-side ones, on the other operand of the comparison.
    # Latent on this host (torch 2.9.1+rocm7.2 supplies it and every node is
    # domain 0), which is exactly why it needed looking for rather than
    # waiting for.
    domain = getattr(properties, "pci_domain_id", None)
    bus = getattr(properties, "pci_bus_id", None)
    device = getattr(properties, "pci_device_id", None)
    by_uid: list[str] = []
    by_bdf: list[str] = []
    unparsed_nodes: list[str] = []
    contradicted: list[str] = []

    def _field(props: dict[str, str], name: str, bits: int = 32) -> int | None:
        """An unsigned integer field that must be present, parse, and be in range.

        None means none of the three -- absent, malformed and out-of-range are
        the same amount of evidence, which is none.

        ``bits`` is the field's width in the driver, and it is a required part
        of the range: an unsigned check alone is only half an invariant.
        @Reviewer found the other half against @Autotune's tree and it
        reproduced verbatim here -- ``domain``, ``location_id`` and
        ``gfx_target_version`` at 2**32 all clean-matched, and a cache ``size``
        of 2**32 KiB returned 4 TiB as a clean ``kfd_topology`` read with no
        degradation recorded. A rotation sized against a 4 TiB last level is
        every allocation the harness makes.

        Widths measured on the live topology before tightening, not asserted
        from the header: ``unique_id`` reaches 18206932166487137716 and needs
        64, while ``gfx_target_version`` (90500), ``domain`` (0),
        ``location_id`` (62720), ``level`` (3) and ``size`` (262144 KB) all sit
        far inside 32. Defaulting to 32 and widening only ``unique_id`` is
        deliberate: a new field added without thought gets the tighter bound
        and fails loudly, rather than the looser one and passing quietly.

        Every integer KFD publishes here is unsigned, so a negative value is a
        malformed read and not a small one. Parsing alone let three wrong
        answers through as clean matches (@Reviewer, against 53c1d4d):
        ``gfx_target_version -1`` passed the ``!= 0`` CPU test and was accepted
        as a GPU; ``unique_id -1`` was accepted as an identity; and
        ``location_id -35584`` is the worst of the three, because Python's
        arithmetic shift on a negative gives ``(-35584 >> 8) & 0xFF == 117``
        and ``(-35584 >> 3) & 0x1F == 0`` -- it *aliases* this host's real bus
        0x75, device 0 and clean-wins the PCI match. A masked bitfield cannot
        reject its own garbage, so the range check has to happen before the
        mask, not after.

        Verified against the live topology before tightening rather than after:
        35332 integer fields across all 10 KFD nodes and their 4370 cache
        entries, zero negative. ``unique_id`` reaches 18206932166487137716,
        above 2**63, confirming these are unsigned 64-bit and that signedness
        is the bug rather than the format.
        """
        if name not in props:
            return None
        try:
            value = int(props[name])
        except ValueError:
            return None
        return value if 0 <= value < (1 << bits) else None

    try:
        for node in sorted(os.listdir(node_root)):
            base = os.path.join(node_root, node)
            try:
                props, malformed = _read_kfd_properties(os.path.join(base, "properties"))
            except OSError:
                unparsed_nodes.append(f"{node}:unreadable")
                continue

            # Validate the fields; do not infer from what int() happens to
            # raise. `.get("gfx_target_version", 0)` read a *missing* field as
            # 0 and silently classified the node as a CPU -- so a GPU node with
            # that field absent vanished from the scan without being recorded,
            # and the search fell through to a same-BDF neighbour. @Reviewer
            # demonstrated that against 337bdbd: it is the same wrong-device
            # answer c597e11 closed, reached through a missing field instead of
            # a bad integer. Absent and malformed are the same amount of
            # evidence, so they take the same path.
            gfx_version = _field(props, "gfx_target_version")
            if gfx_version is None:
                unparsed_nodes.append(f"{node}:unparseable_properties")
                continue
            if gfx_version == 0:
                continue  # genuinely known to be a CPU node

            # A node's identity fields are validated whether or not this
            # caller's branch happens to consult them. @Reviewer's
            # branch-asymmetry point, which reproduced here: the UID path
            # accepted a node without ever looking at `domain`/`location_id`,
            # so the same malformed node was clean when torch supplied a UUID
            # and degraded when it did not. Validation strength that depends on
            # which branch the caller took is not validation -- it is a
            # coincidence of routing, and it hides exactly the node that is
            # least trustworthy.
            #
            # `unique_id` is validated even on the PCI path, where it is not
            # read: a node asserting an impossible identity is not a node whose
            # other fields are more believable.
            #
            # But the converse does not follow, and the first version of this
            # block got it wrong by treating every malformed field as
            # disqualifying. A node whose unique_id *is* the one we asked for
            # has asserted the identity directly; an unreadable `domain` on
            # that node is a field the match never rested on, and dropping the
            # match over it is over-refusal. So the PCI fields disqualify a
            # node only when the PCI address is what would identify it. The
            # asymmetry @Reviewer objected to was a field being unchecked on
            # one path; this is a field being *irrelevant* on one path, which
            # is a different thing and has to be argued rather than assumed --
            # a test asserting the opposite is what made me look.
            #
            # `unique_id` is the one field where absent and malformed must not
            # be conflated *before* this point, and the reader is what makes
            # that possible. A node with no `unique_id` line states no identity
            # and legitimately falls through to the PCI key. A node whose
            # `unique_id` line is unreadable *did* state one -- we simply cannot
            # tell whether it says this card -- so it can be neither confirmed
            # nor excluded, and letting it fall through is how the read lands on
            # a same-BDF neighbour. Hence `malformed`: without it a corrupted
            # identity looks exactly like an unstated one.
            stated_uid = "unique_id" in props or "unique_id" in malformed
            node_uid = _field(props, "unique_id", bits=64) if "unique_id" in props else None
            if stated_uid and node_uid is None:
                unparsed_nodes.append(f"{node}:unparseable_properties")
                continue
            if want_uid is not None and node_uid is not None and node_uid != want_uid:
                # This node states an identity, and it is not the one we asked
                # for. That is positive evidence of a *different* device, not
                # missing evidence -- so it must not later be accepted by PCI
                # address. Recording it is what distinguishes "no unique_id
                # information available" from "the only candidate says no".
                contradicted.append(node)
                continue
            if want_uid is not None and node_uid == want_uid:
                by_uid.append(base)
                # The identity was asserted directly by this node, so the PCI
                # fields are not what identifies it and cannot disqualify it.
                continue

            if domain is not None and bus is not None and device is not None:
                location = _field(props, "location_id")
                node_domain = _field(props, "domain")
                if location is None or node_domain is None:
                    unparsed_nodes.append(f"{node}:unparseable_properties")
                    continue
                if (
                    node_domain == domain
                    and ((location >> 8) & 0xFF) == bus
                    and ((location >> 3) & 0x1F) == device
                ):
                    by_bdf.append(base)
    except OSError:
        return fallback, {"source": "torch_l2_fallback", "reason": "no_kfd_topology"}

    if len(by_uid) == 1:
        matched, key = by_uid[0], "unique_id"
    elif len(by_uid) > 1:
        return fallback, {"source": "torch_l2_fallback", "reason": "ambiguous_unique_id"}
    elif len(by_bdf) == 1:
        matched, key = by_bdf[0], "pci_domain_bus_device"
    elif len(by_bdf) > 1:
        # Reachable under CPX. unique_id would have separated these; getting
        # here means it was unavailable or matched nothing.
        return fallback, {"source": "torch_l2_fallback", "reason": "ambiguous_pci_address"}
    elif contradicted:
        # Distinguishable from a plain absence: candidates existed at this
        # address and each stated a different identity. "No node matched" would
        # read as "the topology does not describe this card", which is the
        # opposite of what happened.
        miss: dict[str, Any] = {
            "source": "torch_l2_fallback",
            "reason": "unique_id_contradicted",
            "contradicting_nodes": contradicted[:20],
        }
        if unparsed_nodes:
            miss["skipped_nodes"] = unparsed_nodes[:20]
        return fallback, miss
    else:
        miss = {"source": "torch_l2_fallback", "reason": "no_matching_node"}
        if unparsed_nodes:
            # Why nothing matched is the actionable part: a tree we could not
            # fully read is a different situation from one that genuinely does
            # not contain this card.
            miss["skipped_nodes"] = unparsed_nodes[:20]
        return fallback, miss

    # The two match keys are not equally robust to a skipped node, and the
    # difference is positive evidence versus absence of competing evidence:
    #
    #   unique_id  -- the returned node *asserted* the identity we asked for. A
    #                 node that failed to parse asserted nothing, so it cannot
    #                 take that away.
    #   PCI b:d.f  -- the returned node is the only *surviving* candidate at
    #                 that address. Under CPX the address is shared, so a
    #                 skipped node is a candidate we neither confirmed nor
    #                 excluded, and the elimination no longer eliminates.
    #
    # The demonstrated failure is the second one: with the real card's node
    # corrupt, asking for its unique_id fell through to a PCI match on a
    # different node and returned that device's 268435456 with matched_by
    # pci_domain_bus_device and no degradation marked at all.
    #
    # A contradiction is stronger than a skip and is handled above rather than
    # here: a node that states a *different* unique_id is not a candidate we
    # failed to evaluate, it is one we evaluated and rejected, so accepting it
    # by PCI address afterwards would override direct evidence with a weaker
    # key. Those nodes never enter by_bdf.
    node_scan_degraded = bool(unparsed_nodes) and key != "unique_id"

    # Entries skipped inside the *matched* node are recorded, not just skipped.
    # Skipping silently and then reporting success is how a corrupt level-3
    # entry -- exactly the MALL line this helper exists to read -- degrades to
    # torch's 4 MiB while the provenance still says "kfd_topology". @Reviewer
    # demonstrated that against fad422c: a valid L2 entry beside an unparseable
    # level-3 entry returned (4194304, source=kfd_topology) and _resolve_llc
    # saw nothing to fail on. A skipped entry is missing evidence; the caller
    # has to be able to tell that from a complete read.
    best = 0
    skipped: list[str] = []
    try:
        cache_root = os.path.join(matched, "caches")
        for cache in sorted(os.listdir(cache_root)):
            try:
                cprops, cmalformed = _read_kfd_properties(
                    os.path.join(cache_root, cache, "properties")
                )
            except OSError:
                skipped.append(f"{cache}:unreadable")
                continue
            # Validate the fields, do not just catch what int() happens to
            # raise on. `.get("level", 0)` silently turns a *missing* level into
            # L0 and drops the entry as uninteresting -- but a missing level is
            # unknown, not small, and the MALL is exactly the entry we cannot
            # afford to drop. @Reviewer found this against a7eec93, along with
            # nonpositive sizes, which parse fine and then vanish into `max`.
            # Absent and malformed take the same path -- both are missing
            # evidence and both are recorded -- but they are tagged apart,
            # because the tag is what a reader uses to decide where to look. A
            # `caches/` entry with no `level` line is a driver that did not
            # publish one; a `level` line that does not parse is a file to go
            # read. Same treatment, different observation.
            if "level" not in cprops:
                tag = "bad_level" if "level" in cmalformed else "missing_level"
                skipped.append(f"{cache}:{tag}")
                continue
            # Bounded above as well as below, and bounded by what a cache
            # hierarchy can actually be rather than by the field's width. A
            # `level` of 2**32 parsed cleanly and passed `level >= 2`, so the
            # entry was consulted as if it were a last level.
            level_value = _bounded(cprops, "level", MAX_CACHE_LEVEL)
            if level_value is None:
                skipped.append(f"{cache}:bad_level")
                continue
            level = level_value
            if level < 1:
                # A cache cannot be below L1. `level < 2 -> skip` treated 0 and
                # -1 as ordinary low-level entries and dropped them silently,
                # so a topology carrying nonsense still read as complete
                # (@Reviewer, against 337bdbd). Only levels that are genuinely
                # below the last level may be discarded without a record.
                skipped.append(f"{cache}:invalid_level{level}")
                continue
            if level < 2:
                continue  # genuinely known to be below the last level
            if "size" not in cprops:
                tag = "bad_size" if "size" in cmalformed else "missing_size"
                skipped.append(f"{cache}:{tag}_level{level}")
                continue
            # The upper bound matters more here than anywhere else in this
            # function, because this value is not just an identity check -- it
            # is returned and then used to size every rotation buffer the
            # harness allocates. `size 4294967296` parsed, passed `> 0`, and
            # came back as a clean 4 TiB `kfd_topology` read with no
            # degradation recorded.
            size_value = _bounded(cprops, "size", MAX_CACHE_SIZE_KB)
            if size_value is None:
                skipped.append(f"{cache}:bad_size_level{level}")
                continue
            size_kb = size_value  # KFD reports KB
            if size_kb <= 0:
                skipped.append(f"{cache}:nonpositive_size_level{level}")
                continue
            best = max(best, size_kb * 1024)
    except OSError:
        return fallback, {"source": "torch_l2_fallback", "reason": "no_caches_directory"}
    if best <= 0:
        # Carry the skips out with the failure. Without this the caller is told
        # "this node reports no cache above L2" -- a statement about the
        # hardware -- when what happened may be "every entry that would have
        # answered failed to parse", a statement about the read. The gfx950
        # path fails closed either way, so this is not a soundness hole, but
        # the two send a reader to entirely different places and only one of
        # them is where the problem is. Enumeration completeness: an empty
        # `caches/` and a `caches/` whose every entry was rejected produced
        # byte-identical provenance before this.
        miss = {"source": "torch_l2_fallback", "reason": "no_level2_plus_cache"}
        if skipped:
            miss["skipped_cache_entries"] = skipped
            miss["reason"] = "no_usable_level2_plus_cache"
        return fallback, miss
    provenance: dict[str, Any] = {
        "source": "kfd_topology",
        "matched_by": key,
        "matched_node": matched,
    }
    # A list, not a string: the two degradations are independent and a run can
    # hit both. The earlier single-string field would have reported whichever
    # one was assigned last and hidden the other.
    degraded: list[str] = []
    if node_scan_degraded:
        degraded.append("unidentified_nodes")
        provenance["skipped_nodes"] = unparsed_nodes[:20]
    if skipped:
        degraded.append("unparseable_cache_entries")
        provenance["skipped_cache_entries"] = skipped
    if degraded:
        provenance["degraded"] = degraded
    # `max(best, fallback)` laundered the fallback's value through the
    # topology's source label. @Reviewer's counterexample against 53c1d4d: a
    # clean unique_id-matched node reporting only a 4 MiB L2, with torch
    # claiming a 256 MiB L2_cache_size, returned 268435456 tagged
    # `source: kfd_topology` and no degradation -- so _resolve_llc accepted a
    # MALL that KFD never observed. The label said where the number came from
    # and it was not true of that number.
    #
    # A value and its provenance have to travel together or the provenance is
    # decoration. `best` is what the topology actually reported; the fallback
    # is recorded beside it, never merged into it. This is the same defect
    # class again, at the point of return rather than the point of parse: two
    # different situations -- observed and assumed -- collapsing into one
    # indistinguishable number.
    if fallback > best:
        # This is evidence *conflict*, not evidence absence, and the two are
        # not the same defect. `torch_l2_fallback` means the topology could not
        # be read; this means it was read cleanly and disagrees with torch. Only
        # the second one says some component is wrong.
        #
        # An earlier version of this comment called it benign -- "on a part with
        # no memory-side cache torch's L2 and KFD's last level are the same
        # line, so a larger torch figure just means we could not see what torch
        # saw". Measured on the live topology instead of assumed: all eight
        # gfx950 nodes publish a level-2 entry of 4194304 and torch reports
        # L2_cache_size 4194304 for the same card. An L2-only part is the case
        # where the two sources are *equal*. `fallback > best` is not that part;
        # it is a source contradicting the driver, and there is no reading of it
        # under which both numbers are right.
        provenance["torch_l2_exceeds_topology"] = {
            "topology_bytes": best,
            "torch_l2_bytes": fallback,
        }
    return best, provenance


GFX950_MALL_BYTES = 256 * 1024**2


def _resolve_llc(
    torch: Any,
    properties: Any,
    args: argparse.Namespace,
    node_root: str = KFD_NODE_ROOT,
) -> tuple[int, dict]:
    """Fail closed on gfx950 rather than silently reporting cache-warm numbers.

    Falling back to ``L2_cache_size`` on gfx950 reinstates the exact defect this
    helper exists to fix: the run completes, every row looks normal, and the
    numbers are measured against a resident MALL. That is worse than a crash,
    because a crash is noticed. On any other architecture the fallback is just a
    conservative guess and is allowed through with the reason recorded.

    Five ways to end up with a wrong number, not one. The first version of this
    function only caught the first:

    1. the whole topology read fails and we fall back;
    2. the read *succeeds* but an individual cache entry was unparseable, so the
       MALL line may be the one that was skipped;
    3. the read succeeds completely and simply reports no cache above the
       per-XCD L2;
    4. the read succeeds and returns a plausible 256 MiB -- from the wrong node,
       because the node that was this card failed to parse and the fallback PCI
       key is not unique under CPX;
    5. the read succeeds completely and *contradicts* torch, which reports a
       larger ``L2_cache_size`` for the same card. Nothing is missing here and
       nothing is undersized -- two sources describe one device incompatibly,
       and at most one of them is right.

    On gfx950 all five are fatal, and the acceptance test is a conjunction of
    four independent things -- a trustworthy *source*, no recorded degradation,
    no recorded conflict, and a value at or above the MALL -- none of which
    subsumes the others.

    An earlier version of this paragraph said "for (2) and (3) the test is the
    value itself", and that was the wrong lesson to draw from a true
    observation. The value catches a *wrong-sized* answer. It cannot catch a
    right-sized one, and there are two ways to get one:

    - (4), where the number is right because every card on this host is the
      same part, so the wrong node reports the same 256 MiB. What is unsound
      is the identification, not the magnitude. This is why ``degraded`` is
      fatal rather than advisory.
    - source laundering, which @Reviewer found against ``53c1d4d``: the parser
      merged torch's fallback into the topology's number with ``max()`` and
      kept the topology's label, so a node reporting only 4 MiB returned
      268435456 marked ``kfd_topology``. The value test passed on a number
      KFD never observed. Fixed at the source -- the parser now returns only
      what the topology reported -- but the docstring had been asserting a
      robustness the code did not have, which is the recurring shape here.

    So: the value is a *necessary* condition, never a sufficient one, and the
    reason to check the source first is that a number which is right by
    coincidence is not evidence.

    (5) is the same lesson a third time and it caught this function again. The
    conflict was recorded in the provenance and the accept branch did not read
    it, so a topology reporting exactly 256 MiB against a torch claim of 512 MiB
    was accepted -- source clean, no degradation, value right. Recording
    evidence and not gating on it leaves the artifact honest and the decision
    unchanged, which is the failure mode this whole helper exists to prevent.

    ``--llc-bytes`` is the escape hatch for a gfx950 host whose topology this
    helper cannot read.
    """
    if args.llc_bytes is not None:
        return args.llc_bytes, {"source": "explicit_override", "flag": "--llc-bytes"}
    llc_bytes, provenance = _last_level_cache_bytes(torch, properties, node_root)
    if _device_arch(torch) != "gfx950":
        # Off gfx950 the LLC figure is a sizing hint, not a soundness gate, and
        # the conservative choice is the larger of the two -- a rotation sized
        # against too *large* a cache is merely wasteful, while one sized too
        # small is silently cache-warm. The parser no longer merges them (that
        # is what laundered the source on gfx950), so the choice is made here.
        #
        # And it has to be *relabelled* here, not merely annotated. @Reviewer's
        # counterexample against 50db350, reproduced before this edit: this
        # branch returned torch's 268435456 with the provenance still reading
        # `source: kfd_topology` from a node that reported 4 MiB. Adding
        # `torch_l2_exceeds_topology` beside an unchanged `source` is the same
        # laundering one level up -- I fixed the parser and then re-did it in
        # the caller, which is why a side field is not a substitute for the
        # field that is actually named `source`. A consumer reads `source`.
        larger = provenance.get("torch_l2_exceeds_topology")
        if larger is not None:
            return larger["torch_l2_bytes"], {
                **provenance,
                "source": "torch_l2_over_topology",
                "topology_bytes": larger["topology_bytes"],
            }
        return llc_bytes, provenance
    # Three independent conditions, and the value is only one of them. An
    # earlier version accepted any figure at or above the MALL size with no
    # degradation marked, which let the *fallback* through whenever torch
    # happened to report a large enough L2 -- flatly contradicting the promise
    # one paragraph up that a whole-topology failure is fatal. @Reviewer
    # constructed it: absent topology plus `L2_cache_size = 256 MiB` returned
    # `source: torch_l2_fallback` and was accepted. A number that is right by
    # coincidence is not evidence, so the source has to be checked first.
    # A recorded conflict is unresolved evidence, so it belongs in the accept
    # test and not merely in the artifact. Found while acting on @Autotune's
    # correction: a topology reporting exactly the MALL against a torch
    # `L2_cache_size` of 512 MiB passed all three conditions and returned
    # 268435456, because the early return fired before the conflict reason was
    # ever built. Writing `torch_l2_exceeds_topology` into the provenance and
    # then not gating on it is the recording-without-acting half of the same
    # defect -- the evidence was captured and the decision ignored it.
    #
    # Not over-refusal on this host: all eight live nodes report 262144 KB
    # against torch's 4194304 B, so `fallback > best` is false and the flag is
    # absent (verified 8/8 before and after this change).
    if (
        provenance["source"] == "kfd_topology"
        and "degraded" not in provenance
        and "torch_l2_exceeds_topology" not in provenance
        and llc_bytes >= GFX950_MALL_BYTES
    ):
        return llc_bytes, provenance
    # Every applicable reason, not the first one that matches. The `elif` chain
    # this replaces reported half the problem when both degradations occurred
    # -- and the half it dropped was arbitrary, decided by clause order rather
    # than by severity (@Reviewer). A run that fails closed for two independent
    # reasons and names one sends its reader to fix half of it.
    degraded = provenance.get("degraded", [])
    reasons = []
    if provenance["source"] == "torch_l2_fallback":
        reasons.append(f"the KFD topology could not be read ({provenance['reason']})")
    if "unidentified_nodes" in degraded:
        reasons.append(
            f"the node was matched only by {provenance['matched_by']}, which is not "
            "unique under CPX, while "
            f"{', '.join(provenance['skipped_nodes'])} could not be identified, so "
            f"{provenance.get('matched_node')} may belong to a different device"
        )
    if "unparseable_cache_entries" in degraded:
        entries = provenance["skipped_cache_entries"]
        reasons.append(
            "the KFD topology was read but "
            f"{len(entries)} cache entr{'y was' if len(entries) == 1 else 'ies were'} "
            f"unusable ({', '.join(entries)}), so the MALL entry may be among them"
        )
    conflict = provenance.get("torch_l2_exceeds_topology")
    if conflict is not None:
        # Name the disagreement rather than only its consequence. "Reports no
        # cache at or above the MALL size" is true here and sends the reader to
        # look for a missing level-3 entry, when what actually happened is that
        # two sources described the same card incompatibly -- and which one is
        # wrong is the question worth printing.
        reasons.append(
            f"the KFD topology reports {conflict['topology_bytes']} B as this card's "
            f"last level while torch reports L2_cache_size {conflict['torch_l2_bytes']} B; "
            "the two sources contradict each other, so neither can be used"
        )
    if not reasons:
        reasons.append(
            f"the KFD topology was read cleanly from {provenance.get('matched_node')} "
            "but reports no cache at or above the MALL size"
        )
    detail = "; and ".join(reasons)
    # Do not claim the value is too small when it is not. Two of these paths
    # reach a value that is the right size and still untrustworthy -- a
    # wrong-card read, or a degraded read that happened to find the MALL anyway.
    # "268435456 B, below the known 268435456 B MALL" is what the first version
    # of this message printed, and a reader chasing that sentence would look for
    # a magnitude defect that is not there.
    if "unidentified_nodes" in degraded:
        consequence = (
            f"The value {llc_bytes} B was obtained from a node this run cannot prove "
            "is the device it benchmarked; sizing the rotation from another device's "
            "topology is not detectable in the results."
        )
    elif llc_bytes >= GFX950_MALL_BYTES:
        consequence = (
            f"The value {llc_bytes} B is the expected size, but evidence was missing "
            "from the read that produced it, so it cannot be distinguished from a "
            "coincidence -- a larger cache may be the entry that was skipped."
        )
    else:
        consequence = (
            f"The best value available is {llc_bytes} B, below the known "
            f"{GFX950_MALL_BYTES} B MALL, so using it would size the rotation against "
            "a cache that is not the last level and silently measure cache-warm."
        )
    raise RuntimeError(
        f"cannot determine the last-level cache on gfx950: {detail}. {consequence} "
        "Pass --llc-bytes to override explicitly."
    )


class _L2Evictor:
    def __init__(self, torch: Any, target_bytes: int):
        self.source = torch.empty(target_bytes, device="cuda", dtype=torch.uint8)
        self.destination = torch.empty_like(self.source)
        self.source.fill_(1)
        self.destination.zero_()
        torch.cuda.synchronize()

    def __call__(self) -> None:
        self.destination.copy_(self.source)


def _time_rotating_calls(
    torch: Any,
    prepared: PreparedCase,
    *,
    warmup_rounds: int,
    sample_rounds: int,
    evictor: _L2Evictor | None,
) -> list[float]:
    """Time a whole rotation with one event pair, not one pair per call.

    A ``torch.cuda.Event`` record carries barrier semantics, so bracketing
    every launch charges two pipeline drains to each kernel. Measured against
    rocprofv3 hardware timestamps on gfx950 by
    ``AI/probe_event_timing_calibration.py``, sidecar committed beside it.
    Quoting the field the probe defines and stores, ``over_read_vs_hardware``
    -- profiled event median over the hardware median of that same profiled
    phase -- at ``512x4096`` / ``4096x4096`` / ``32768x1024``:

        per-call      +138% / +50% / +22%
        per-rotation  +103% / +14% /  +5%

    The over-read shrinks as the kernel grows, which is the expected shape for
    a fixed per-launch cost, and per-rotation is the smaller over-read at every
    shape, which is the ordering this function's design rests on.

    Two corrections have now landed here, and the second is the more
    instructive. The original text asserted "178% high on a 6us kernel and 9%
    high on a 29us one" from a shell that was thrown away; @Reviewer refused it
    (blocker 4) and the probe exists because he was right.

    The replacement was archived but still wrong. It read +52% / +18% / +6%
    and +16% / +5% / +1%, computed by dividing ``event_median_us_unprofiled``
    by ``per_rotation.hardware_median_us`` -- in the per-call row too. That
    crosses profiler regimes (unprofiled numerator, profiled denominator) and,
    worse, charges the per-call figure against a *different process's* hardware
    baseline, which is why +52% appeared where the self-consistent pair reads
    +138%. @Reviewer caught it by recomputing from the committed JSON, and the
    contradiction was available to anyone who did: the probe stores the honest
    ratio in every record and its own doc says the over-read "should be read
    against the profiled pair, which is self-consistent".

    Being archived is what made the second error checkable, not what made it
    right. A committed sidecar removes the excuse for a remembered number; it
    does not license deriving a new one whose halves come from different runs.
    If a figure is not the field the artifact stores, it needs its own
    derivation shown -- and this one could not have survived showing it.

    The evictor runs outside the window. Keeping the operands out of L2 is the
    rotation's job -- ``_rotation_count`` sizes it against the L2 target for
    exactly that reason -- and the evictor only covers the case where memory
    capped the rotation short of it.
    """
    for _ in range(warmup_rounds):
        prepared.reset()
        if evictor is not None:
            evictor()
        for call in prepared.calls:
            call()
    torch.cuda.synchronize()

    calls_per_round = len(prepared.calls)
    samples_us = []
    for _ in range(sample_rounds):
        prepared.reset()
        if evictor is not None:
            evictor()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for call in prepared.calls:
            call()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / calls_per_round)
    return samples_us


def _measure_achievable_bandwidth(
    torch: Any,
    *,
    probe_bytes: int,
    warmup_rounds: int,
    sample_rounds: int,
) -> dict[str, Any]:
    """Best sustained bandwidth over several access patterns.

    A same-device copy alone understates the memory system badly enough that
    the RMSNorm forward exceeds it, which makes the resulting percentage
    useless as a ceiling. Probe a copy, a two-read one-write elementwise, and
    a pure write, and report the best; the elementwise probe is the closest
    match to what a normalization kernel actually does.
    """
    elements = probe_bytes // 2
    left = torch.empty(elements, device="cuda", dtype=torch.bfloat16).fill_(1)
    right = torch.empty_like(left).fill_(2)
    out = torch.empty_like(left)

    probes = {
        "copy": (lambda: out.copy_(left), 2 * probe_bytes),
        "two_read_one_write": (lambda: torch.add(left, right, out=out), 3 * probe_bytes),
        "write": (lambda: out.zero_(), probe_bytes),
    }
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    results = {}
    for name, (call, logical) in probes.items():
        for _ in range(warmup_rounds):
            call()
        torch.cuda.synchronize()
        samples_us = []
        for _ in range(sample_rounds):
            start.record()
            call()
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000.0)
        median_us = _percentile(samples_us, 0.5)
        results[name] = {
            "median_us": round(median_us, 6),
            "gbps": round(logical / (median_us * 1e-6) / 1e9, 6),
        }
    best = max(results, key=lambda name: results[name]["gbps"])
    return {
        "probe_bytes": probe_bytes,
        "probes": results,
        "best_probe": best,
        "median_gbps": results[best]["gbps"],
        "median_us": results[best]["median_us"],
        "samples": sample_rounds,
    }


def _torch_dtype(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _make_inputs(torch: Any, cell: MatrixCell, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    activation_dtype = _torch_dtype(torch, cell.activation_dtype)
    weight_dtype = _torch_dtype(torch, cell.weight_dtype)
    x = torch.randn((cell.m, cell.n), device="cuda", dtype=activation_dtype) * 0.5
    weight = 1.0 + torch.randn(cell.n, device="cuda", dtype=weight_dtype) * 0.1
    dout = torch.randn_like(x) * 0.1 if cell.operation == "bwd" else None
    return {"x": x, "weight": weight, "dout": dout}


def _reference(torch: Any, cell: MatrixCell, inputs: dict[str, Any], eps: float) -> tuple:
    with torch.no_grad():
        x = inputs["x"]
        weight = inputs["weight"]
        x_f32 = x.float()
        rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
        if cell.operation == "fwd":
            return ((x_f32 * rstd * weight.float()).to(x.dtype),)

        dout = inputs["dout"]
        dout_f32 = dout.float()
        weighted_dout = dout_f32 * weight.float()
        correction = (weighted_dout * x_f32).mean(dim=-1, keepdim=True)
        dx = (rstd * (weighted_dout - x_f32 * rstd.square() * correction)).to(x.dtype)
        dweight = (dout_f32 * x_f32 * rstd).sum(dim=0).to(weight.dtype)
        return dx, dweight, rstd.flatten()


def _assert_correct(torch: Any, cell: MatrixCell, actual: tuple, expected: tuple) -> None:
    if cell.operation == "fwd":
        tolerances = (2e-4, 2e-5) if cell.activation_dtype == "float32" else (2e-2, 2e-2)
    else:
        tolerances = (5e-3, 5e-3) if cell.activation_dtype == "float32" else (3e-2, 3e-2)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=tolerances[0],
            atol=tolerances[1],
        )


def _cold_launch(
    torch: Any,
    call: Callable[[], None],
    key: tuple,
    compile_timings: dict[tuple, float],
) -> tuple[float, bool]:
    reused = key in compile_timings
    torch.cuda.synchronize()
    started = time.perf_counter()
    call()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not reused:
        compile_timings[key] = elapsed_ms
    return compile_timings[key], reused


class _FlyDSLProvider:
    def __init__(self, torch: Any):
        self.torch = torch
        self.impl = importlib.import_module("quack.rmsnorm_flydsl")
        self.compile_timings: dict[tuple, float] = {}

    def prepare(
        self,
        cell: MatrixCell,
        inputs: dict[str, Any],
        reference: tuple,
        *,
        eps: float,
        rotation_buffers: int,
    ) -> PreparedCase:
        torch = self.torch
        if cell.operation == "fwd":
            tensor_sets = []
            calls = []
            for _ in range(rotation_buffers):
                x = inputs["x"].clone()
                weight = inputs["weight"].clone()
                out = torch.empty_like(x)
                rstd = torch.empty(0, device=x.device, dtype=torch.float32)
                absent = torch.empty(0, device=x.device, dtype=x.dtype)
                tensor_sets.append((out,))

                def call(x=x, weight=weight, out=out, rstd=rstd, absent=absent):
                    self.impl._launch_rmsnorm_fwd(
                        x,
                        weight,
                        absent,
                        absent,
                        out,
                        absent,
                        rstd,
                        eps,
                        0.0,
                        has_weight=True,
                        has_bias=False,
                        has_residual=False,
                        store_residual=False,
                        store_rstd=False,
                        per_head=False,
                        num_heads=1,
                    )

                calls.append(call)

            compile_key = (
                "fwd",
                cell.n,
                cell.activation_dtype,
                cell.weight_dtype,
                eps,
            )
            cold_ms, reused = _cold_launch(
                torch,
                calls[0],
                compile_key,
                self.compile_timings,
            )
            _assert_correct(torch, cell, tensor_sets[0], reference)
            return PreparedCase(
                calls=calls,
                reset=lambda: None,
                outputs=lambda: tensor_sets[0],
                provider_detail="FlyDSL low-level forward",
                cold_compile_ms=cold_ms,
                cold_compile_reused=reused,
            )

        dtype_str = self.impl._dtype_to_str(inputs["x"].dtype)
        num_programs = self.impl._select_rmsnorm_bwd_programs(
            cell.m,
            cell.n,
            dtype_str,
            inputs["x"].device,
        )
        tensor_sets = []
        calls = []
        for _ in range(rotation_buffers):
            x = inputs["x"].clone()
            weight = inputs["weight"].clone()
            dout = inputs["dout"].clone()
            rstd = reference[2].clone()
            dx = torch.empty_like(x)
            dweight = torch.empty_like(weight)
            absent = torch.empty(0, device=x.device, dtype=x.dtype)
            dbias = torch.empty(1, device=x.device, dtype=weight.dtype)
            tensor_sets.append((dx, dweight))

            # The launcher allocates its own partials, so that allocation is
            # inside the timed region, which is where the operation pays it.
            def call(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                dx=dx,
                dweight=dweight,
                absent=absent,
                dbias=dbias,
            ):
                self.impl._launch_rmsnorm_bwd(
                    x,
                    weight,
                    dout,
                    x,
                    rstd,
                    dx,
                    absent,
                    dweight,
                    dbias,
                    0.0,
                    has_weight=True,
                    has_bias=False,
                    compute_dweight=True,
                    compute_dbias=False,
                    has_residual=False,
                    has_dresidual_out=False,
                    per_head=False,
                    num_heads=1,
                )

            calls.append(call)

        def reset() -> None:
            return

        compile_key = (
            "bwd",
            cell.n,
            cell.activation_dtype,
            cell.weight_dtype,
            num_programs,
        )
        cold_ms, reused = _cold_launch(
            torch,
            calls[0],
            compile_key,
            self.compile_timings,
        )
        _assert_correct(torch, cell, tensor_sets[0], reference[:2])
        return PreparedCase(
            calls=calls,
            reset=reset,
            outputs=lambda: tensor_sets[0],
            provider_detail=f"FlyDSL low-level backward ({num_programs} programs)",
            cold_compile_ms=cold_ms,
            cold_compile_reused=reused,
        )


def _torch_rms_norm(torch: Any, x: Any, weight: Any, eps: float) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mismatch dtype between input and weight.*",
            category=UserWarning,
        )
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


class _TorchProvider:
    def __init__(self, torch: Any):
        self.torch = torch

    def prepare(
        self,
        cell: MatrixCell,
        inputs: dict[str, Any],
        reference: tuple,
        *,
        eps: float,
        rotation_buffers: int,
    ) -> PreparedCase:
        torch = self.torch
        tensor_sets = []
        calls = []
        if cell.operation == "fwd":
            for _ in range(rotation_buffers):
                x = inputs["x"].clone()
                weight = inputs["weight"].clone()
                output = [None]
                tensor_sets.append(output)

                def call(x=x, weight=weight, output=output):
                    output[0] = _torch_rms_norm(torch, x, weight, eps)

                calls.append(call)
            calls[0]()
            torch.cuda.synchronize()
            _assert_correct(torch, cell, (tensor_sets[0][0],), reference)
            return PreparedCase(
                calls=calls,
                reset=lambda: None,
                outputs=lambda: (tensor_sets[0][0],),
                provider_detail="torch.nn.functional.rms_norm",
                cold_compile_ms=None,
                cold_compile_reused=False,
            )

        for _ in range(rotation_buffers):
            x = inputs["x"].clone().requires_grad_(True)
            weight = inputs["weight"].clone().requires_grad_(True)
            dout = inputs["dout"].clone()
            output = _torch_rms_norm(torch, x, weight, eps)
            gradients = [None]
            tensor_sets.append(gradients)

            def call(
                x=x,
                weight=weight,
                dout=dout,
                output=output,
                gradients=gradients,
            ):
                gradients[0] = torch.autograd.grad(
                    output,
                    (x, weight),
                    dout,
                    retain_graph=True,
                )

            calls.append(call)
        torch.cuda.synchronize()
        calls[0]()
        torch.cuda.synchronize()
        _assert_correct(torch, cell, tensor_sets[0][0], reference[:2])
        return PreparedCase(
            calls=calls,
            reset=lambda: None,
            outputs=lambda: tensor_sets[0][0],
            provider_detail="torch.nn.functional.rms_norm autograd",
            cold_compile_ms=None,
            cold_compile_reused=False,
        )


class _QuackProvider:
    """Quack's own CuTe RMSNorm, so a CUDA box can be measured the same way.

    Calls the same low-level ``rmsnorm_fwd`` / ``rmsnorm_bwd`` entry points
    that ``benchmarks/benchmark_rmsnorm.py`` times, which is also the level
    the FlyDSL provider measures.
    """

    def __init__(self, torch: Any):
        self.torch = torch
        module = importlib.import_module("quack.rmsnorm")
        self._fwd = module.rmsnorm_fwd
        self._bwd = module.rmsnorm_bwd

    def prepare(
        self,
        cell: MatrixCell,
        inputs: dict[str, Any],
        reference: tuple,
        *,
        eps: float,
        rotation_buffers: int,
    ) -> PreparedCase:
        torch = self.torch
        tensor_sets = []
        calls = []
        if cell.operation == "fwd":
            for _ in range(rotation_buffers):
                x = inputs["x"].clone()
                weight = inputs["weight"].clone()
                slot = [None]
                tensor_sets.append(slot)

                def call(x=x, weight=weight, slot=slot):
                    slot[0] = self._fwd(x, weight, eps=eps)[0]

                calls.append(call)
            cold_start = time.perf_counter()
            calls[0]()
            torch.cuda.synchronize()
            cold_compile_ms = (time.perf_counter() - cold_start) * 1000.0
            _assert_correct(torch, cell, (tensor_sets[0][0],), reference)
            return PreparedCase(
                calls=calls,
                reset=lambda: None,
                outputs=lambda: (tensor_sets[0][0],),
                provider_detail="quack.rmsnorm.rmsnorm_fwd (CuTe)",
                cold_compile_ms=cold_compile_ms,
                cold_compile_reused=False,
            )

        for _ in range(rotation_buffers):
            x = inputs["x"].clone()
            weight = inputs["weight"].clone()
            dout = inputs["dout"].clone()
            # rmsnorm_fwd returns (out, residual_out, rstd).
            rstd = self._fwd(x, weight, eps=eps, store_rstd=True)[2]
            slot = [None]
            tensor_sets.append(slot)

            def call(x=x, weight=weight, dout=dout, rstd=rstd, slot=slot):
                dx, dweight, _, _ = self._bwd(x, weight, dout, rstd)
                slot[0] = (dx, dweight)

            calls.append(call)
        torch.cuda.synchronize()
        cold_start = time.perf_counter()
        calls[0]()
        torch.cuda.synchronize()
        cold_compile_ms = (time.perf_counter() - cold_start) * 1000.0
        _assert_correct(torch, cell, tensor_sets[0][0], reference[:2])
        return PreparedCase(
            calls=calls,
            reset=lambda: None,
            outputs=lambda: tensor_sets[0][0],
            provider_detail="quack.rmsnorm.rmsnorm_bwd (CuTe)",
            cold_compile_ms=cold_compile_ms,
            cold_compile_reused=False,
        )


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        m_text, n_text = value.lower().split("x", 1)
        shape = int(m_text), int(n_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must use MxN, for example 512x4096") from error
    if shape[0] < 1 or not 1 <= shape[1] <= 8192:
        raise argparse.ArgumentTypeError("shape requires M >= 1 and 1 <= N <= 8192")
    return shape


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmark_artifacts") / f"rmsnorm_flydsl_gfx950_{timestamp}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _imported_provider_modules() -> dict[str, Any]:
    """Where the provider packages were actually imported from, and their hash.

    ``git_commit`` records the repository the harness was launched inside,
    which is not the same thing and on the cross-vendor H200 run was not even
    consistent with it: the artifact carries ambient commit 4f36477, whose
    tree declares ``quack.__version__ = "0.6.1"``, while the run imported
    installed ``quack`` 0.5.0 from ``dist-packages`` -- and the repo tree at
    that commit does not import at all on that host. So the recorded commit
    could not have been the provider under test, and nothing in the artifact
    said so. @Reviewer's blocker 5.

    ``_executed_source`` already applies this principle to the harness; the
    providers are the code the benchmark exists to measure and had weaker
    provenance than the file measuring them. Hashing ``__init__.py`` rather
    than the whole package is deliberate: it is cheap, it is stable, and it is
    enough to tell two installs apart. It is not a build fingerprint, and this
    field should not be read as one.
    """
    modules: dict[str, Any] = {}
    for name in ("quack", "flydsl"):
        module = sys.modules.get(name)
        if module is None:
            continue
        entry: dict[str, Any] = {
            "path": getattr(module, "__file__", None),
            "version_attr": getattr(module, "__version__", None),
        }
        path = entry["path"]
        if path:
            try:
                entry["init_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError:
                entry["init_sha256"] = None
        modules[name] = entry
    return modules


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> dict[str, Any]:
    """Whether the tree differs from the recorded commit, and where.

    ``git_commit`` alone is ambient: it names HEAD, which says nothing about
    what was actually executed if the tree is dirty or if the file being run is
    a scratch copy. Recording HEAD as if it identified the code is how the
    ``AI/gate_llc_before_after/`` isolate runs ended up unverifiable.
    """
    try:
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_dirty": None, "git_dirty_paths": []}
    paths = [line[3:] for line in porcelain.splitlines() if line]
    return {"git_dirty": bool(paths), "git_dirty_paths": paths[:20]}


def _executed_source() -> dict[str, Any]:
    """Hash of the file actually being run, not of the commit it resembles.

    This is the field that distinguishes ``benchmark_rmsnorm_flydsl.py`` from a
    one-line-edited scratch copy of it. Without it an artifact can only assert
    its provenance; with it a reader can check it.
    """
    try:
        source = Path(__file__).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError:
        return {"script_path": None, "script_sha256": None}
    return {"script_path": source.name, "script_sha256": digest}


def _device_arch(torch: Any) -> str:
    """Architecture string for the visible device, on either vendor.

    A CUDA build also exposes ``gcnArchName``, where it holds the marketing
    name, so the build is the discriminator rather than the attribute.
    """
    properties = torch.cuda.get_device_properties(0)
    if torch.version.hip is not None:
        return properties.gcnArchName.split(":", 1)[0]
    return f"sm_{properties.major}{properties.minor}"


def _describe_measured_scope(peak_bw: dict[str, Any]) -> str:
    """The part of ``comparison_scope`` that is a measurement, not a caveat.

    Split out and appended after the probe runs, rather than interpolated into
    the literal in ``_environment``, because ``_environment`` is built before
    any measurement exists -- so the only numbers available to it are ones from
    somewhere else, which is precisely how the v5 string came to quote another
    host's results in every artifact. Keeping the two apart makes that mistake
    hard to repeat: the caveat cannot cite a number, and this cannot be written
    without one in hand.
    """
    probes = ", ".join(
        f"{name} {probe['gbps']:.1f}" for name, probe in sorted(peak_bw["probes"].items())
    )
    return (
        f". This run divided by {peak_bw['best_probe']} at "
        f"{peak_bw['median_gbps']:.1f} GB/s; the probes measured here were {probes} GB/s"
    )


def _environment(torch: Any, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        # Every provider is checked against the fp32 reference before it is
        # timed, and a mismatch aborts the sweep. Recorded once here rather
        # than as a per-row column that could only ever say "passed".
        "correctness_gate": "required",
        "runtime_scope": f"{properties.name} / {_device_arch(torch)}",
        # Only what is true of every run. The v5 version of this string
        # hard-coded one experiment -- a date, two host names, fwd bf16, one
        # shape and three deltas -- into every artifact the harness would ever
        # write. @Reviewer constructed an H100 / bwd / fp32 / torch-only run and
        # got the entire paragraph verbatim, including "Measured 2026-08-02,
        # fwd bf16" and both vendors' probe winners. That is a machine-readable
        # field asserting measurements the run did not make, which is this
        # defect class exactly, committed by the person who had just written a
        # commit message about it.
        #
        # The measured numbers that motivated it belong in
        # AI/crossvendor_rmsnorm_fwd_bf16/README.md, attached to the artifacts
        # they came from. What THIS run observed is appended by
        # `_describe_measured_scope` once the probe has actually run -- see
        # there for why it cannot be interpolated here.
        "comparison_scope": (
            "same-device providers. RMSNorm is memory bound, so a cross-vendor "
            "comparison of microseconds mostly reports the HBM bandwidth ratio. "
            "peak_bw_pct is NOT a cross-vendor fix: its denominator is whichever "
            "achievable-bandwidth probe won on this host, and the winner can "
            "differ by host, so two hosts' percentages may be ratios against "
            "different references. For a cross-vendor statement pick one probe "
            "present on both hosts and divide by that same probe on each; "
            "achievable_bandwidth.probes carries all of them for exactly this "
            "purpose, and peak_bw_probe records per row which one this run used"
        ),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "git_commit": _git_commit(),
        **_git_dirty(),
        **_executed_source(),
        "artifact_directory": str(output_dir.resolve()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "flydsl": _package_version("flydsl"),
            "quack": _package_version("quack-kernels") or _package_version("quack"),
        },
        "gpu": {
            "visible_index": 0,
            "visible_count": torch.cuda.device_count(),
            "name": properties.name,
            "arch": _device_arch(torch),
            "compute_units": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "l2_cache_bytes_reported": properties.L2_cache_size,
            # The ordinal identifies a card only relative to a visibility mask
            # that is itself set by the environment, so an artifact saying
            # "index 0" pins nothing -- on an 8-GPU host it is whichever card
            # the mask happened to expose. @Reviewer's blocker 5: the frozen
            # MI355X artifact recorded ordinal 6 and node 8 without ever
            # storing the UID those were matched on, so the identity the
            # resolver worked hard to establish was dropped before it reached
            # the reader. None on a build whose runtime does not expose one,
            # which is itself worth recording rather than omitting.
            #
            # Stored as torch reports it. _torch_unique_id's int decode is the
            # AMD reading of this field and is what the KFD match uses; the raw
            # text is what identifies the card on either vendor, so that is
            # what goes in the artifact.
            "uuid": _reported_device_uuid(properties),
        },
        "visibility": {
            name: os.environ.get(name)
            for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        },
        "architecture_overrides": {
            name: os.environ.get(name) for name in ("ARCH", "FLYDSL_GPU_ARCH")
        },
        "host": {
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "matrix": {
            "shapes": [list(shape) for shape in args.shapes],
            "dtype_weight_modes": [list(mode) for mode in args.dtype_weight_modes],
            "operations": args.operations,
            "providers": args.providers,
            "eps": args.eps,
        },
        # Each string describes what this run did, not what the gfx950/FlyDSL
        # path does. @Reviewer's blocker 7 against the frozen H200 artifact:
        # `last_level_cache` said the value was "read from the KFD topology,
        # matched by unique_id" on a host with no HIP, whose own
        # last_level_cache_provenance in the same file says
        # not_a_hip_build/torch_l2_fallback; `cache` asserted the 256 MiB MALL
        # and the 4 MiB per-XCD L2 on an sm_90 card; `steady_state` described
        # FlyDSL first-launch JIT on a quack+torch run. A methodology field
        # that documents the template rather than the execution is a claim the
        # artifact cannot support -- the same defect as the hard-coded
        # comparison_scope above, and it was in the very artifact I collected
        # to close a different blocker.
        "methodology": {
            "steady_state": (
                "one torch.cuda.Event pair per timed rotation, divided by the number "
                "of calls in it -- NOT one pair per call, which carries barrier "
                "semantics and over-reads short kernels. Magnitude is measured, not "
                "asserted here: see AI/probe_event_timing_calibration.py and its "
                "committed sidecar for the per-call and per-rotation over-read against "
                "rocprofv3 hardware timestamps on gfx950. After provider warmup"
                + (
                    ", with FlyDSL first-launch JIT synchronized, recorded separately, and excluded"
                    if "flydsl" in args.providers
                    else " (no FlyDSL provider in this run, so no JIT phase to exclude)"
                )
            ),
            "cache": (
                "round-robin cloned tensor sets sized against rotation_target_bytes "
                "(= --l2-target-ratio x the last-level cache, taken from "
                "last_level_cache_bytes with last_level_cache_provenance recording "
                "where it came from); when a set's working set is at most "
                "evictor_threshold_bytes (= EVICTOR_LLC_MARGIN x LLC) a device copy "
                "evicts the cache once per timed rotation, before the event window "
                "opens, not between individual calls"
            ),
            "last_level_cache": (
                "on a HIP build, read from the KFD topology, matched by unique_id and "
                "then by PCI domain:bus:device; a gfx950 host that cannot be matched "
                "aborts rather than falling back to torch's L2, which would silently "
                "measure cache-warm. Elsewhere torch's reported L2 is used. "
                "last_level_cache_provenance records which of these this run did"
            ),
            "logical_bytes": {
                "fwd": "read x + weight; write y",
                "bwd": "read x + dout + weight + fp32 rstd; write dx + dweight",
            },
            "peak_bandwidth": (
                "best of a same-device copy, a two-read one-write elementwise, and "
                "a pure write; a copy alone understates the memory system"
            ),
            "warmup_rounds": args.warmup_rounds,
            "sample_rounds": args.sample_rounds,
            "max_rotation_buffers": args.max_rotation_buffers,
            "l2_target_ratio": args.l2_target_ratio,
        },
        "result_schema": list(RESULT_FIELDS),
    }


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=None,
        help="Default: the vendor's own kernels plus torch (flydsl on ROCm, quack on CUDA)",
    )
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument(
        "--activation-dtypes",
        nargs="+",
        choices=tuple(_ITEMSIZES),
        default=list(_ITEMSIZES),
    )
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        choices=("same", "float32"),
        default=["same", "float32"],
    )
    parser.add_argument(
        "--shape",
        dest="shapes",
        action="append",
        type=_parse_shape,
        help="Restrict to an MxN shape; repeat for multiple shapes",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    # One sample per round now that a round is timed as a whole, so this is
    # also the sample count the percentiles are drawn from.
    parser.add_argument("--sample-rounds", type=int, default=40)
    parser.add_argument("--copy-mib", type=int, default=512)
    parser.add_argument("--copy-samples", type=int, default=30)
    parser.add_argument("--min-rotation-buffers", type=int, default=2)
    parser.add_argument("--max-rotation-buffers", type=int, default=4)
    parser.add_argument("--l2-target-ratio", type=int, default=3)
    parser.add_argument(
        "--llc-bytes",
        type=int,
        default=None,
        help=(
            "Override the last-level cache size in bytes. Only needed on a host "
            "whose KFD topology cannot be matched to the visible device; on "
            "gfx950 that case aborts rather than guessing."
        ),
    )
    parser.add_argument("--expected-arch", default="gfx950")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _validate_runtime(torch: Any, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU is visible")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark requires exactly one visible GPU; isolate it with HIP/ROCR visibility"
        )
    if args.providers is None:
        args.providers = ["flydsl" if torch.version.hip is not None else "quack", "torch"]
    actual_arch = _device_arch(torch)
    if actual_arch != args.expected_arch:
        raise RuntimeError(f"expected {args.expected_arch}, found {actual_arch}")
    if "flydsl" in args.providers and torch.version.hip is None:
        raise RuntimeError("the FlyDSL provider requires a ROCm PyTorch build")
    if "quack" in args.providers and torch.version.hip is not None:
        raise RuntimeError("the Quack CuTe provider requires a CUDA PyTorch build")
    if not math.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("--eps must be finite and positive")
    positive_values = {
        "--warmup-rounds": args.warmup_rounds,
        "--sample-rounds": args.sample_rounds,
        "--copy-mib": args.copy_mib,
        "--copy-samples": args.copy_samples,
        "--min-rotation-buffers": args.min_rotation_buffers,
        "--max-rotation-buffers": args.max_rotation_buffers,
        "--l2-target-ratio": args.l2_target_ratio,
    }
    if args.llc_bytes is not None and args.llc_bytes < 1:
        raise ValueError("--llc-bytes must be positive")
    for name, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if args.min_rotation_buffers > args.max_rotation_buffers:
        raise ValueError("--min-rotation-buffers cannot exceed --max-rotation-buffers")


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _run(
    torch: Any,
    args: argparse.Namespace,
    environment: dict[str, Any],
) -> list[dict[str, Any]]:
    properties = torch.cuda.get_device_properties(0)
    llc_bytes, llc_provenance = _resolve_llc(torch, properties, args)
    environment["last_level_cache_bytes"] = llc_bytes
    environment["last_level_cache_provenance"] = llc_provenance
    environment["torch_l2_cache_size"] = properties.L2_cache_size
    rotation_target_bytes = llc_bytes * args.l2_target_ratio
    environment["rotation_target_bytes"] = rotation_target_bytes
    environment["evictor_gate"] = {
        "predicate": "rotation_working_set_bytes <= margin * last_level_cache_bytes",
        "margin": EVICTOR_LLC_MARGIN,
        "threshold_bytes": EVICTOR_LLC_MARGIN * llc_bytes,
    }
    peak_bw = _measure_achievable_bandwidth(
        torch,
        probe_bytes=args.copy_mib * 1024**2,
        warmup_rounds=args.warmup_rounds,
        sample_rounds=args.copy_samples,
    )
    environment["achievable_bandwidth"] = peak_bw
    environment["comparison_scope"] += _describe_measured_scope(peak_bw)
    torch.cuda.empty_cache()
    evictor = _L2Evictor(torch, rotation_target_bytes)

    providers = {}
    for name in args.providers:
        factory = {
            "flydsl": _FlyDSLProvider,
            "quack": _QuackProvider,
            "torch": _TorchProvider,
        }[name]
        providers[name] = factory(torch)

    # After construction, so it reflects what the providers actually imported
    # rather than what was importable before they ran.
    environment["provider_modules"] = _imported_provider_modules()

    cells = build_matrix(
        args.shapes,
        args.dtype_weight_modes,
        args.operations,
    )
    rows = []
    for cell_index, cell in enumerate(cells):
        inputs = _make_inputs(torch, cell, args.seed + cell_index)
        reference = _reference(torch, cell, inputs, args.eps)
        byte_count = logical_bytes(
            cell.operation,
            cell.m,
            cell.n,
            _ITEMSIZES[cell.activation_dtype],
            _ITEMSIZES[cell.weight_dtype],
        )
        for provider_name in args.providers:
            free_bytes, _ = torch.cuda.mem_get_info()
            rotation_buffers = _rotation_count(
                byte_count,
                rotation_target_bytes,
                free_bytes,
                min_buffers=args.min_rotation_buffers,
                max_buffers=args.max_rotation_buffers,
            )
            prepared = providers[provider_name].prepare(
                cell,
                inputs,
                reference,
                eps=args.eps,
                rotation_buffers=rotation_buffers,
            )
            rotation_working_set_bytes = rotation_buffers * byte_count
            # Evict unless the rotation alone already clears the last-level
            # cache by a margin. The comparison is against the LLC itself, not
            # against rotation_target_bytes (= ratio * LLC): a set larger than the
            # cache is self-evicting, and nothing between LLC and ratio*LLC
            # needs the evictor switched off. Comparing against the target left
            # the evictor off for every set in that band.
            #
            # The margin is a margin, not a measured threshold. The archived
            # fine-boundary sweep walks 256.000-256.008 MiB and every point
            # still reads above the HBM reference, so an exact-fit or
            # few-KiB-over set measures MALL-warm and a bare `> llc_bytes` test
            # would trust it. The 256->288 MiB decay is gradual rather than a
            # cliff, so no single crossing point is defensible; 2x is chosen to
            # sit clear of the soft region. Erring high costs time, erring low
            # costs correctness.
            #
            # What the margin does NOT do -- an earlier version of this comment
            # claimed it did, and @Reviewer showed the claim is false against
            # the code: it does not turn the evictor on for `32768x1024` fwd.
            # That cell's rotation grows 2->4 under the MALL-sized target, so
            # its working set is 512.008 MiB, which *exceeds* 2x256 MiB and
            # leaves the evictor OFF. It is rescued by the larger rotation, not
            # by this gate. Across the 90-cell matrix the target change moves 53
            # rotation counts while this gate flips 37 cells false->true, and
            # the two sets are not the same cells. Do not describe a cell as
            # "now evicted" without checking which of the two changes reached
            # it. The 13 cells where 2x differs from a bare 1x are the 4096-row
            # shapes between 256.03 and 384.19 MiB.
            use_evictor = evictor_is_needed(rotation_working_set_bytes, llc_bytes)
            samples_us = _time_rotating_calls(
                torch,
                prepared,
                warmup_rounds=args.warmup_rounds,
                sample_rounds=args.sample_rounds,
                evictor=evictor if use_evictor else None,
            )
            stats = _summarize_us(samples_us)
            logical_gbps = byte_count / stats["median_us"] / 1000.0
            peak_bw_pct = logical_gbps / peak_bw["median_gbps"] * 100.0
            row = {
                "schema_version": SCHEMA_VERSION,
                "provider": provider_name,
                "provider_detail": prepared.provider_detail,
                "operation": cell.operation,
                "m": cell.m,
                "n": cell.n,
                "activation_dtype": cell.activation_dtype,
                "weight_dtype": cell.weight_dtype,
                "weight_mode": cell.weight_mode,
                "eps": args.eps,
                "cold_compile_ms": _round(prepared.cold_compile_ms),
                "cold_compile_reused": prepared.cold_compile_reused,
                "median_us": _round(stats["median_us"]),
                "p10_us": _round(stats["p10_us"]),
                "p90_us": _round(stats["p90_us"]),
                "logical_bytes": byte_count,
                "logical_gbps": _round(logical_gbps),
                "peak_bw_gbps": _round(peak_bw["median_gbps"]),
                "peak_bw_pct": _round(peak_bw_pct),
                # Which probe the denominator came from. Without it a reader of
                # results.csv alone cannot tell that two hosts' peak_bw_pct are
                # ratios against different references -- and the winner does
                # differ by host, so a cross-vendor delta computed from this
                # column is not a comparison unless both rows agree here.
                "peak_bw_probe": peak_bw["best_probe"],
                "rotation_buffers": rotation_buffers,
                "rotation_working_set_bytes": rotation_working_set_bytes,
                "rotation_target_bytes": rotation_target_bytes,
                "evictor_threshold_bytes": EVICTOR_LLC_MARGIN * llc_bytes,
                "evictor_ran_per_rotation": use_evictor,
                "timed_samples": len(samples_us),
            }
            rows.append(row)
            print(
                f"PASS {provider_name:6s} {cell.operation} "
                f"M={cell.m:<5d} N={cell.n:<4d} "
                f"{cell.activation_dtype}/{cell.weight_dtype}: "
                f"{stats['median_us']:.3f} us, {logical_gbps:.1f} GB/s, "
                f"{peak_bw_pct:.1f}% of peak BW",
                flush=True,
            )
            del prepared
        del inputs, reference
        torch.cuda.empty_cache()
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    args.shapes = args.shapes or list(COMPACT_SHAPES)
    args.dtype_weight_modes = [
        mode
        for mode in DTYPE_WEIGHT_MODES
        if mode[0] in args.activation_dtypes and mode[1] in args.weight_modes
    ]
    if not args.dtype_weight_modes:
        parser.error("dtype and weight-mode filters select no supported combinations")
    args.output_dir = args.output_dir or _default_output_dir()

    torch = importlib.import_module("torch")
    _validate_runtime(torch, args)
    environment = _environment(torch, args, args.output_dir)
    rows: list[dict[str, Any]] = []
    try:
        rows = _run(torch, args, environment)
    except Exception as error:
        environment["status"] = "failed"
        environment["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        csv_path, environment_path = write_artifacts(args.output_dir, rows, environment)
        print(f"Partial artifacts: {args.output_dir.resolve()}")
        print(f"CSV: {csv_path.resolve()}")
        print(f"Environment: {environment_path.resolve()}")
        raise

    # Contention canary: re-probe the memory system the sweep was normalised
    # against. A shared node can pick up a co-tenant partway through, and every
    # peak_bw_pct in the CSV is then measured against a ceiling that no longer
    # holds. Cheaper to record the drift than to discover it later.
    closing_bw = _measure_achievable_bandwidth(
        torch,
        probe_bytes=args.copy_mib * 1024**2,
        warmup_rounds=args.warmup_rounds,
        sample_rounds=args.copy_samples,
    )
    opening_gbps = environment["achievable_bandwidth"]["median_gbps"]
    closing_gbps = closing_bw["median_gbps"]
    drift = closing_gbps / opening_gbps
    environment["contention_canary"] = {
        "opening_gbps": _round(opening_gbps),
        "closing_gbps": _round(closing_gbps),
        "closing_over_opening": _round(drift),
        "quiet": 0.9 <= drift <= 1.1,
    }
    if not 0.9 <= drift <= 1.1:
        print(
            f"\nWARNING: achievable bandwidth moved {drift:.2f}x during the sweep "
            f"({opening_gbps:.0f} -> {closing_gbps:.0f} GB/s). "
            "The node was not quiet; treat these numbers as indicative only.",
            file=sys.stderr,
        )

    environment["status"] = "passed"
    environment["result_rows"] = len(rows)
    environment["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    csv_path, environment_path = write_artifacts(args.output_dir, rows, environment)
    print(f"Artifacts: {args.output_dir.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Environment: {environment_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

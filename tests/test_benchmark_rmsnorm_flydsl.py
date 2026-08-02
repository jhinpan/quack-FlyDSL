# Copyright (c) 2026, Tri Dao.

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_rmsnorm_flydsl.py"
SPEC = importlib.util.spec_from_file_location("benchmark_rmsnorm_flydsl_contract", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_default_matrix_covers_compact_shapes_and_distinct_dtype_weight_modes():
    cells = benchmark.build_matrix()

    assert {cell.shape for cell in cells} == set(benchmark.COMPACT_SHAPES)
    assert {(cell.activation_dtype, cell.weight_mode) for cell in cells} == {
        ("float16", "same"),
        ("float16", "float32"),
        ("bfloat16", "same"),
        ("bfloat16", "float32"),
        ("float32", "same"),
    }
    assert {cell.operation for cell in cells} == {"fwd", "bwd"}
    assert len(cells) == len(benchmark.COMPACT_SHAPES) * 5 * 2


def test_logical_byte_accounting_uses_the_public_fwd_bwd_contract():
    # M=2, N=4, two-byte activations, four-byte weights.
    # fwd: read x + weight and write y.
    assert benchmark.logical_bytes("fwd", 2, 4, 2, 4) == 48
    # bwd: read x/dy/weight/rstd, write dx/dweight.
    assert benchmark.logical_bytes("bwd", 2, 4, 2, 4) == 88


def test_result_contract_and_artifact_writers(tmp_path):
    row = {
        field: index if field not in {"provider", "operation"} else field
        for index, field in enumerate(benchmark.RESULT_FIELDS)
    }
    environment = {"schema_version": 2, "correctness_gate": "required"}

    csv_path, environment_path = benchmark.write_artifacts(tmp_path, [row], environment)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == benchmark.RESULT_FIELDS
        assert len(list(reader)) == 1
    assert json.loads(environment_path.read_text(encoding="utf-8")) == environment


def test_module_import_is_provider_lazy():
    script = (
        "import importlib.util, sys;"
        f"p={str(BENCHMARK_PATH)!r};"
        "s=importlib.util.spec_from_file_location('bench_lazy', p);"
        "m=importlib.util.module_from_spec(s);"
        "sys.modules[s.name]=m;"
        "s.loader.exec_module(m);"
        "assert 'torch' not in sys.modules;"
        "assert 'quack.rmsnorm_flydsl' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


# --- last-level cache resolution -------------------------------------------
#
# These cover the paths that decide whether the benchmark measures cache-cold
# or cache-warm. Before this block there were zero: the parser, the identity
# matching, the ambiguity refusal, the fail-closed policy and the gate
# arithmetic were all exercised only by running the benchmark on one host,
# where every one of them happens to take its happy path.


def _with_caches_count(properties, count):
    """Add the `caches_count` line the driver always writes, unless the test
    is deliberately saying something else about it.

    Every real node publishes this field and it agrees exactly with its
    `caches/` listing -- measured on all 10 nodes of this host (0/0 twice, then
    546/546 seven times and 548/548). A fixture that omits it is not a smaller
    version of a real node, it is a node in a state the driver never produces,
    and building the whole suite on one would mean the enumeration check is
    only ever exercised by its failure path.
    """
    if any(line.startswith("caches_count ") for line in properties.splitlines()):
        return properties
    return properties + f"caches_count {count}\n"


def _extra_cache(node, name, body):
    """Add one `caches/` entry to an already-written node, keeping the node's
    declared `caches_count` in step with the directory.

    Tests that appended an entry by hand used to leave the count behind. That
    is now a mismatch the reader refuses -- correctly, since it is exactly the
    partial-listing case the check exists for -- so a fixture that wants a node
    with a *bad entry* rather than a *bad count* has to say so here.
    """
    cache = node / "caches" / name
    cache.mkdir()
    (cache / "properties").write_text(body)
    properties = node / "properties"
    text = properties.read_text()
    count = len(list((node / "caches").iterdir()))
    kept = [line for line in text.splitlines() if not line.startswith("caches_count ")]
    properties.write_text("\n".join([*kept, f"caches_count {count}"]) + "\n")
    return cache


def _write_node(root, index, *, gfx=90000, domain=0, location=29952, unique_id=None, caches=()):
    node = root / str(index)
    (node / "caches").mkdir(parents=True)
    lines = [f"gfx_target_version {gfx}", f"domain {domain}", f"location_id {location}"]
    if unique_id is not None:
        lines.append(f"unique_id {unique_id}")
    lines.append(f"caches_count {len(caches)}")
    (node / "properties").write_text("\n".join(lines) + "\n")
    for cache_index, (level, size_kb) in enumerate(caches):
        cache = node / "caches" / str(cache_index)
        cache.mkdir()
        (cache / "properties").write_text(f"level {level}\nsize {size_kb}\n")
    return node


def _properties(*, bus=0x75, device=0, uuid_text=b"a60c2956cd9dd4c5", l2=4 * 1024**2):
    uuid = types.SimpleNamespace(bytes=uuid_text) if uuid_text is not None else None
    return types.SimpleNamespace(
        L2_cache_size=l2, pci_domain_id=0, pci_bus_id=bus, pci_device_id=device, uuid=uuid
    )


def _hip_torch():
    return types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2"))


def test_torch_uuid_is_ascii_hex_not_a_uuid():
    # The 16 bytes are the text "a60c2956cd9dd4c5", which is KFD's unique_id
    # for the same card. Decoding them as a binary UUID gives a different
    # number and matches no node.
    assert benchmark._torch_unique_id(_properties()) == 0xA60C2956CD9DD4C5
    assert benchmark._torch_unique_id(_properties(uuid_text=b"not hex at all!!")) is None
    assert benchmark._torch_unique_id(_properties(uuid_text=None)) is None


def test_gate_margin_and_motivating_cell_is_not_evicted_by_the_gate():
    # Guards the claim that used to be wrong in the source comment. The cell
    # this whole investigation started from -- 32768x1024 fwd -- ends up with a
    # 512.008 MiB rotation, which is ABOVE 2 x 256 MiB, so the gate leaves the
    # evictor OFF. It is rescued by the larger rotation, not by the gate.
    llc = 256 * 1024**2
    per_set = benchmark.logical_bytes("fwd", 32768, 1024, 2, 2)
    buffers = benchmark._rotation_count(
        per_set, llc * 3, free_bytes=10**12, min_buffers=2, max_buffers=4
    )
    working_set = buffers * per_set
    assert buffers == 4
    assert working_set == 512 * 1024**2 + 8 * 1024
    # Calls the real predicate, not a copy of it.
    assert benchmark.evictor_is_needed(working_set, llc) is False
    # ...and it clears the threshold by only 8 KiB, so the margin is not what
    # saves this cell either way.
    assert working_set - benchmark.EVICTOR_LLC_MARGIN * llc == 8 * 1024


def test_gate_flips_exactly_the_cells_the_docs_claim():
    # 90 cells: the target change moves 53 rotation counts, the gate flips 37
    # false->true, and those are different sets. Conflating them is the error
    # @Reviewer caught in the commit message and the source comment.
    llc = 256 * 1024**2
    rotation_changed = gate_flipped = 0
    for cell in benchmark.build_matrix():
        act = benchmark._ITEMSIZES[cell.activation_dtype]
        weight = 4 if cell.weight_mode == "float32" else act
        per_set = benchmark.logical_bytes(cell.operation, cell.m, cell.n, act, weight)
        kwargs = {"free_bytes": 10**12, "min_buffers": 2, "max_buffers": 4}
        old_n = benchmark._rotation_count(per_set, 4 * 1024**2 * 3, **kwargs)
        new_n = benchmark._rotation_count(per_set, llc * 3, **kwargs)
        old_on = old_n * per_set < 4 * 1024**2 * 3  # the pre-fix predicate
        new_on = benchmark.evictor_is_needed(new_n * per_set, llc)
        rotation_changed += old_n != new_n
        gate_flipped += (not old_on) and new_on
    assert rotation_changed == 53
    assert gate_flipped == 37


def _gfx950_torch(arch="gfx950:sramecc+:xnack-"):
    """A torch stub whose _device_arch resolves, so _resolve_llc runs for real.

    The previous version of the fail-closed test monkeypatched
    _last_level_cache_bytes out of the way and asserted against a hand-written
    provenance dict. It therefore tested the branch, not the pipeline, and
    passed while a corrupt cache entry walked straight through the real parser.
    Everything below drives the actual sysfs reader over a temporary tree.
    """
    return types.SimpleNamespace(
        version=types.SimpleNamespace(hip="7.2"),
        cuda=types.SimpleNamespace(
            get_device_properties=lambda _: types.SimpleNamespace(gcnArchName=arch)
        ),
    )


def _gfx950_node(tmp_path, caches):
    root = tmp_path / "nodes"
    root.mkdir()
    node = root / "2"
    (node / "caches").mkdir(parents=True)
    (node / "properties").write_text(
        _with_caches_count(
            "gfx_target_version 90500\n"
            "unique_id 11964983762810164421\n"
            "domain 0\n"
            "location_id 30720\n",
            len(caches),
        )
    )
    for index, body in enumerate(caches):
        cache = node / "caches" / str(index)
        cache.mkdir()
        (cache / "properties").write_text(body)
    return str(root)


def _node(root, index, properties, caches=()):
    """A node written from a literal properties string, so a test can omit a
    field entirely -- which is the case _write_node's keyword arguments cannot
    express, and therefore the case that went untested."""
    node = root / str(index)
    (node / "caches").mkdir(parents=True)
    (node / "properties").write_text(_with_caches_count(properties, len(caches)))
    for cache_index, (level, size_kb) in enumerate(caches):
        cache = node / "caches" / str(cache_index)
        cache.mkdir()
        (cache / "properties").write_text(f"level {level}\nsize {size_kb}\n")
    return node


REAL_UID = 11964983762810164421  # 0xa60c2956cd9dd4c5, as KFD prints it
GPU_AT_BDF = f"gfx_target_version 90500\nunique_id {REAL_UID}\ndomain 0\nlocation_id 29952\n"


def test_a_cache_level_below_one_is_recorded_not_silently_dropped(tmp_path):
    # `level < 2 -> skip` treated 0 and -1 as ordinary low-level entries. No
    # cache is below L1, so those are nonsense, and a topology carrying
    # nonsense is not a topology that was read completely -- even when the
    # surviving entries happen to produce the right number. @Reviewer's case:
    # valid L2 + valid MALL + {level 0, size 524288} was accepted clean.
    root = _gfx950_node(
        tmp_path,
        ["level 2\nsize 4096\n", "level 3\nsize 262144\n", "level 0\nsize 524288\n"],
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)
    assert value == 256 * 1024**2  # right answer, incomplete evidence
    assert provenance["degraded"] == ["unparseable_cache_entries"]
    assert provenance["skipped_cache_entries"] == ["2:invalid_level0"]

    with pytest.raises(RuntimeError, match="cache entr(y was|ies were) unusable"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), root
        )


def test_a_genuine_l1_entry_is_still_dropped_silently(tmp_path):
    # The negative control @Reviewer asked for. Level 1 is a real cache level
    # and the live tree is full of them -- node 2 has 544 L1 entries
    # (144x16 KiB + 256x32 KiB + 144x64 KiB). Recording those would mark every
    # healthy read degraded.
    root = _gfx950_node(
        tmp_path,
        [
            "level 1\nsize 16\n",
            "level 1\nsize 64\n",
            "level 2\nsize 4096\n",
            "level 3\nsize 262144\n",
        ],
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)
    assert value == 256 * 1024**2
    assert "degraded" not in provenance


def test_a_missing_node_field_cannot_hand_the_read_to_a_neighbour(tmp_path):
    # Same wrong-device answer c597e11 closed, reached through a missing field
    # instead of a bad integer: `.get("gfx_target_version", 0)` read absent as
    # 0 and classified this card's node as a CPU, so it left the scan without
    # being recorded and a same-BDF neighbour was accepted clean.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, f"unique_id {REAL_UID}\ndomain 0\nlocation_id 29952\n", ((2, 4096), (3, 262144)))
    _node(
        root,
        5,
        "gfx_target_version 90500\ndomain 0\nlocation_id 29952\nunique_id 999\n",
        ((2, 4096), (3, 262144)),
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 4 * 1024**2
    assert provenance["source"] == "torch_l2_fallback"
    assert provenance["skipped_nodes"] == ["2:unparseable_properties"]

    with pytest.raises(RuntimeError, match="could not be read"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), str(root)
        )


def test_a_node_that_states_a_different_uid_is_not_accepted_by_pci_address(tmp_path):
    # A node asserting unique_id B when we asked for A is not missing evidence,
    # it is evidence of a different device. Falling back to the PCI key there
    # overrides a direct answer with a weaker one -- and the PCI key is shared
    # under CPX, so it cannot overrule an explicit identity.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root,
        5,
        "gfx_target_version 90500\ndomain 0\nlocation_id 29952\nunique_id 12345\n",
        ((2, 4096), (3, 262144)),
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 4 * 1024**2
    assert provenance["reason"] == "unique_id_contradicted"
    assert provenance["contradicting_nodes"] == ["5"]


def test_a_node_with_no_uid_at_all_still_matches_by_pci(tmp_path):
    # Over-refusal negative for the rule above: silence is not contradiction.
    # A node that states no unique_id has not denied being this card, so the
    # PCI key is still the best evidence available and must keep working.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root, 5, "gfx_target_version 90500\ndomain 0\nlocation_id 29952\n", ((2, 4096), (3, 262144))
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "pci_domain_bus_device"
    assert "degraded" not in provenance


def test_an_absent_torch_pci_domain_is_not_treated_as_domain_zero(tmp_path):
    # The same defect on the other operand of the comparison, found by auditing
    # rather than by review. `getattr(properties, "pci_domain_id", 0)` supplied
    # a 0 that torch never reported, and it was then compared against a node
    # asserting domain 0 -- so an unverifiable field produced a verified match.
    #
    # Latent on this host: torch 2.9.1+rocm7.2 supplies pci_domain_id and every
    # KFD node is domain 0, so it cannot fire here. It is fixed because the
    # class is what is being fixed, not the reachable instances of it.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root, 5, "gfx_target_version 90500\ndomain 0\nlocation_id 29952\n", ((2, 4096), (3, 262144))
    )

    without_domain = types.SimpleNamespace(
        L2_cache_size=4 * 1024**2, pci_bus_id=0x75, pci_device_id=0, uuid=None
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), without_domain, str(root))
    assert value == 4 * 1024**2
    assert provenance["reason"] == "no_matching_node"

    # Over-refusal negative: a domain torch *did* report still matches.
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "pci_domain_bus_device"


def test_gfx950_requires_a_trustworthy_source_not_merely_a_large_number(tmp_path):
    # The resolver accepted any value >= the MALL with nothing marked degraded,
    # checking the number before the provenance. On a host where torch reports
    # a 256 MiB L2 the *fallback* satisfied that -- contradicting the promise
    # that a whole-topology failure is fatal on gfx950. A number that is right
    # by coincidence is not evidence.
    big_l2 = _properties(l2=256 * 1024**2)
    with pytest.raises(RuntimeError, match="could not be read"):
        benchmark._resolve_llc(
            _gfx950_torch(), big_l2, argparse.Namespace(llc_bytes=None), str(tmp_path / "absent")
        )


def test_all_applicable_reasons_are_named_not_just_the_first(tmp_path):
    # The elif chain reported whichever clause came first, so a run failing
    # closed for two independent reasons named one and sent its reader to fix
    # half the problem. Clause order is not severity order.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, "gfx_target_version 90500\ndomain 0\nlocation_id not-a-number\n")
    good = _node(root, 5, "gfx_target_version 90500\ndomain 0\nlocation_id 29952\n", ((2, 4096),))
    _extra_cache(good, "1", "level 3\nsize 0\n")

    _, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert provenance["degraded"] == ["unidentified_nodes", "unparseable_cache_entries"]

    with pytest.raises(RuntimeError) as excinfo:
        benchmark._resolve_llc(
            _gfx950_torch(),
            _properties(uuid_text=None),
            argparse.Namespace(llc_bytes=None),
            str(root),
        )
    message = str(excinfo.value)
    assert "may belong to a different device" in message
    assert "unusable" in message


def test_resolve_llc_fails_closed_on_gfx950_rather_than_measuring_cache_warm(tmp_path):
    # The whole point of the helper. Returning torch's 4 MiB L2 here would let
    # the run complete with every row silently measured against a resident
    # 256 MiB MALL -- the original defect, reintroduced quietly.
    args = argparse.Namespace(llc_bytes=None)
    absent = str(tmp_path / "absent")
    with pytest.raises(RuntimeError, match="could not be read"):
        benchmark._resolve_llc(_gfx950_torch(), _properties(), args, absent)

    # Another architecture has no known-wrong fallback, so it is allowed.
    value, provenance = benchmark._resolve_llc(_gfx950_torch("gfx942"), _properties(), args, absent)
    assert value == 4 * 1024**2
    assert provenance["reason"] == "no_kfd_topology"


def test_a_corrupt_mall_entry_beside_a_good_l2_entry_is_not_reported_as_success(tmp_path):
    # @Reviewer's counterexample against fad422c. The matched node parses, the
    # L2 entry parses, and the level-3 MALL entry does not -- so `best` is a
    # real reading of the wrong cache. The old code returned
    # (4194304, source=kfd_topology) with nothing marking it degraded, and
    # _resolve_llc had no reason to fail. The run then measured cache-warm
    # while its own artifact claimed a successful topology read.
    root = _gfx950_node(tmp_path, ["level 2\nsize 4096\n", "level 3\nsize not-a-number\n"])
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)

    # The parser still returns its best effort -- callers on other parts may
    # legitimately use it -- but it no longer claims the read was complete.
    assert value == 4 * 1024**2
    assert provenance["source"] == "kfd_topology"
    assert provenance["degraded"] == ["unparseable_cache_entries"]
    assert provenance["skipped_cache_entries"] == ["1:bad_size_level3"]

    with pytest.raises(RuntimeError, match="cache entr(y was|ies were) unusable"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), root
        )


@pytest.mark.parametrize(
    "entry,tag",
    [
        # @Reviewer's escapes against a7eec93. Each one parsed or defaulted
        # successfully and then vanished, so `best` stayed at the L2 value with
        # nothing recorded -- byte-identical to a healthy L2-only part.
        ("size 262144\n", "missing_level"),
        ("level 3\n", "missing_size_level3"),
        ("level 3\nsize 0\n", "nonpositive_size_level3"),
        # `size -1` moved from nonpositive_size to bad_size when the reader
        # gained a range. Both record the skip, so this is the tag and not the
        # behaviour -- and bad_size is the truer one: under the same unsigned
        # reasoning that governs the node fields, a negative size is a
        # malformed read rather than a small cache. Zero stays nonpositive
        # because it parses in range and is genuinely a stated size of nothing.
        ("level 3\nsize -1\n", "bad_size_level3"),
        ("", "missing_level"),
        ("level 3\nsize not-a-number\n", "bad_size_level3"),
        # The upper bound, which did not exist before. This one is not a
        # labelling question: 2**32 KB parsed, passed `> 0`, and returned 4 TiB
        # as a clean kfd_topology read that then sized every rotation buffer.
        ("level 3\nsize 4294967296\n", "bad_size_level3"),
        ("level 4294967296\nsize 262144\n", "bad_level"),
    ],
)
def test_every_way_an_entry_can_go_missing_is_recorded(tmp_path, entry, tag):
    # `.get("level", 0)` treated a missing level as L0 and dropped the entry as
    # uninteresting; a missing level is unknown, not small. Nonpositive sizes
    # parsed fine and disappeared into `max`. Validating the fields catches the
    # class; catching whatever int() raises only catches the instances.
    root = _gfx950_node(tmp_path, ["level 2\nsize 4096\n", entry])
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)
    assert value == 4 * 1024**2
    assert provenance["degraded"] == ["unparseable_cache_entries"]
    assert provenance["skipped_cache_entries"] == [f"1:{tag}"]

    with pytest.raises(RuntimeError, match="cache entr(y was|ies were) unusable"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), root
        )


def test_a_valid_mall_beside_a_malformed_entry_is_still_refused(tmp_path):
    # The value guard cannot carry this one: `best` reaches the correct
    # 268435456 from the good entry, so a rule that only checks magnitude sees
    # a perfect read. What is missing is evidence -- the skipped entry could
    # have been larger -- and the refusal has to come from `degraded` alone.
    root = _gfx950_node(
        tmp_path,
        ["level 2\nsize 4096\n", "level 3\nsize 262144\n", "level 3\nsize 0\n"],
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)
    assert value == 256 * 1024**2
    assert provenance["degraded"] == ["unparseable_cache_entries"]

    # And the message must not call 268435456 "below the known 268435456".
    # That sentence sends the reader after a magnitude defect that is not there.
    with pytest.raises(RuntimeError, match="is the expected size, but evidence was missing"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), root
        )


def test_level_one_entries_are_dropped_without_being_called_degraded(tmp_path):
    # Over-refusal negative for the validation above. The real node has 546
    # entries, 544 of them L1 at 32 KB. Recording those as "skipped" would mark
    # every healthy read degraded and fail closed on the actual hardware.
    root = _gfx950_node(
        tmp_path,
        [
            "level 1\nsize 32\n",
            "level 1\nsize 32\n",
            "level 2\nsize 4096\n",
            "level 3\nsize 262144\n",
        ],
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)
    assert value == 256 * 1024**2
    assert "degraded" not in provenance


def test_provenance_reaches_the_artifact_and_survives_json(tmp_path):
    # @Autotune's handover: source/reason have to be *in* environment.json, not
    # merely computed. A provenance dict that never leaves the process cannot
    # tell a later reader which of the four LLC paths a run took, and that
    # reader is the whole audience for this field.
    #
    # It also has to survive json.dumps. `degraded` became a list and
    # `skipped_cache_entries` holds strings for exactly that reason -- a set,
    # or a tuple keyed by node object, would raise or silently reshape at write
    # time, i.e. at the one moment nobody is watching.
    root = _gfx950_node(tmp_path, ["level 2\nsize 4096\n", "level 3\nsize 262144\n"])
    _, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)
    environment = {"last_level_cache_provenance": provenance}
    _, path = benchmark.write_artifacts(tmp_path, [], environment)

    written = json.loads(path.read_text(encoding="utf-8"))["last_level_cache_provenance"]
    assert written == {
        "source": "kfd_topology",
        "matched_by": "unique_id",
        "matched_node": provenance["matched_node"],
    }

    # And the degraded shape, which is the one a reader acts on.
    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    bad = _gfx950_node(bad_root, ["level 2\nsize 4096\n", "level 3\nsize 0\n"])
    _, degraded = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), bad)
    round_tripped = json.loads(json.dumps(degraded))
    assert round_tripped["degraded"] == ["unparseable_cache_entries"]
    assert round_tripped["skipped_cache_entries"] == ["1:nonpositive_size_level3"]


def test_gfx950_refuses_a_clean_read_that_reports_no_mall(tmp_path):
    # Distinct from the case above: nothing failed to parse, KFD simply reports
    # only a 4 MiB L2. Enumerating failure reasons would let this through,
    # because there is no failure. gfx950 has a 256 MiB MALL, so any resolved
    # value below it is wrong however cleanly it was obtained.
    root = _gfx950_node(tmp_path, ["level 2\nsize 4096\n"])
    _, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)
    assert "degraded" not in provenance

    with pytest.raises(RuntimeError, match="no cache at or above"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), root
        )


def test_a_complete_gfx950_read_is_accepted_and_not_marked_degraded(tmp_path):
    # The over-refusal negative case: three of the four gfx950 paths raise, so
    # this pins the one that must not.
    root = _gfx950_node(tmp_path, ["level 2\nsize 4096\n", "level 3\nsize 262144\n"])
    value, provenance = benchmark._resolve_llc(
        _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), root
    )
    assert value == 256 * 1024**2
    assert provenance["source"] == "kfd_topology"
    assert "degraded" not in provenance


def test_override_still_wins_over_a_corrupt_topology(tmp_path):
    # The documented escape hatch has to work in exactly the situation that
    # makes the run fail closed, or fail-closed is just a wall.
    root = _gfx950_node(tmp_path, ["level 2\nsize 4096\n", "level 3\nsize not-a-number\n"])
    value, provenance = benchmark._resolve_llc(
        _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=256 * 1024**2), root
    )
    assert value == 256 * 1024**2
    assert provenance["source"] == "explicit_override"


def test_explicit_override_bypasses_topology_and_is_recorded():
    args = argparse.Namespace(llc_bytes=123456)
    value, provenance = benchmark._resolve_llc(_hip_torch(), _properties(), args)
    assert value == 123456
    assert provenance == {"source": "explicit_override", "flag": "--llc-bytes"}


def test_matches_by_unique_id_even_when_the_pci_address_is_shared(tmp_path):
    # Reachable under CPX: eight logical devices behind one PCI address. Before
    # unique_id matching this returned the fallback for every one of them.
    root = tmp_path / "nodes"
    root.mkdir()
    _write_node(root, 0, unique_id=0xAAAA, caches=((1, 32), (2, 4096)))
    _write_node(root, 1, unique_id=0xA60C2956CD9DD4C5, caches=((2, 4096), (3, 262144)))
    _write_node(root, 2, unique_id=0xBBBB, caches=((2, 4096),))
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "unique_id"


def test_ambiguous_pci_address_without_unique_id_refuses_rather_than_guessing(tmp_path):
    # Same shared address, but no unique_id to separate them. Returning the
    # first match would make the answer depend on directory iteration order.
    root = tmp_path / "nodes"
    root.mkdir()
    _write_node(root, 0, caches=((2, 4096), (3, 262144)))
    _write_node(root, 1, caches=((2, 4096),))
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 4 * 1024**2
    assert provenance["reason"] == "ambiguous_pci_address"


def test_one_node_per_address_still_resolves_without_unique_id(tmp_path):
    # The over-refusal negative case: a guard written as "more than one node
    # anywhere -> refuse" would break every normal eight-card host.
    root = tmp_path / "nodes"
    root.mkdir()
    _write_node(root, 0, location=29952, caches=((2, 4096), (3, 262144)))
    _write_node(root, 1, location=30208, caches=((2, 4096), (3, 262144)))
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "pci_domain_bus_device"


def test_a_malformed_unrelated_node_is_skipped_not_fatal(tmp_path):
    # Regression: the parser used to let ValueError escape _run() and kill the
    # benchmark, and the offending node need not even be a candidate match.
    root = tmp_path / "nodes"
    root.mkdir()
    bad = root / "0"
    (bad / "caches").mkdir(parents=True)
    (bad / "properties").write_text("gfx_target_version not-a-number\ndomain 0\n")
    _write_node(root, 1, unique_id=0xA60C2956CD9DD4C5, caches=((2, 4096), (3, 262144)))
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2

    # And the over-refusal case for the guard below: a unique_id match is a
    # positive identification, so an unrelated unparseable node cannot make it
    # wrong. Marking this degraded would fail closed on every host with one
    # malformed node anywhere in the tree.
    assert provenance["matched_by"] == "unique_id"
    assert "degraded" not in provenance


def test_a_corrupt_node_does_not_hand_the_read_to_a_pci_neighbour(tmp_path):
    # @Autotune's argument about cache entries, applied one level up. He noted
    # that a skipped cache entry only loses information in one direction --
    # `best` is a max, so a skip can only make the answer smaller. A skipped
    # *node* is not like that: it can be neither confirmed nor excluded as this
    # card, and if it was this card the search falls through to the PCI key,
    # which is not unique under CPX.
    #
    # Demonstrated on the real shape: node 2 is this card by unique_id but its
    # properties do not parse, and node 5 is a different device at the same
    # address. The previous code returned node 5's topology with
    # matched_by=pci_domain_bus_device and no degradation marked -- a plausible
    # 256 MiB read off the wrong card, which no assertion on the value can
    # catch because every card on this host is the same part.
    #
    # This test later caught a regression in the strict KFD parser, which is
    # worth recording because it is why the parser returns its withheld-field
    # set rather than just omitting the field. Omitting made a node whose
    # `unique_id` line is corrupt byte-identical to a node that never stated
    # one -- and those two must diverge here: an unstated identity legitimately
    # falls through to the PCI key, a corrupt one can be neither confirmed nor
    # excluded as this card. I wrote a separate test for that before noticing
    # this one already expresses it exactly, so the coverage stayed here.
    root = tmp_path / "nodes"
    root.mkdir()
    real = root / "2"
    (real / "caches").mkdir(parents=True)
    (real / "properties").write_text(
        "gfx_target_version 90500\nunique_id a60c2956cd9dd4c5\ndomain 0\nlocation_id 29952\n"
    )
    _write_node(root, 5, gfx=90500, location=29952, caches=((2, 4096), (3, 262144)))

    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2  # right number
    assert provenance["matched_by"] == "pci_domain_bus_device"  # wrong card
    assert provenance["degraded"] == ["unidentified_nodes"]
    assert provenance["skipped_nodes"] == ["2:unparseable_properties"]

    with pytest.raises(RuntimeError, match="may belong to a different device"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), str(root)
        )


def test_both_degradations_are_reported_when_both_occur(tmp_path):
    # `degraded` is a list because the two are independent. A single string
    # field would report whichever was assigned last and hide the other, and
    # the message _resolve_llc raises would then name only half the problem.
    root = tmp_path / "nodes"
    root.mkdir()
    bad = root / "2"
    (bad / "caches").mkdir(parents=True)
    (bad / "properties").write_text("gfx_target_version 90500\nlocation_id not-a-number\n")
    good = _write_node(root, 5, gfx=90500, location=29952, caches=((2, 4096),))
    _extra_cache(good, "1", "level 3\nsize not-a-number\n")

    _, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert provenance["degraded"] == ["unidentified_nodes", "unparseable_cache_entries"]
    assert provenance["skipped_nodes"] == ["2:unparseable_properties"]
    assert provenance["skipped_cache_entries"] == ["1:bad_size_level3"]


def test_missing_topology_reports_a_reason_rather_than_a_bare_number(tmp_path):
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(), str(tmp_path / "absent")
    )
    assert value == 4 * 1024**2
    assert provenance == {"source": "torch_l2_fallback", "reason": "no_kfd_topology"}


def test_non_hip_build_uses_the_torch_value_without_touching_sysfs():
    torch = types.SimpleNamespace(version=types.SimpleNamespace(hip=None))
    value, provenance = benchmark._last_level_cache_bytes(torch, _properties(), "/nonexistent")
    assert value == 4 * 1024**2
    assert provenance["reason"] == "not_a_hip_build"


def test_artifact_identifies_the_file_that_ran_not_just_the_commit(tmp_path):
    # The gate_llc_before_after isolate runs recorded git_commit=d167701 while
    # actually executing four deleted one-line-edited scratch copies. HEAD was
    # identical across all of them, so the artifact could not distinguish the
    # arms of its own experiment. A hash of the executed file can.
    source = benchmark._executed_source()
    on_disk = hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest()
    assert source["script_sha256"] == on_disk
    assert source["script_path"] == Path(benchmark.__file__).name

    # And it must discriminate: a copy differing by one line hashes differently
    # even though git_commit would be byte-identical for both.
    variant = tmp_path / "_v_scratch.py"
    variant.write_bytes(Path(benchmark.__file__).read_bytes() + b"\n# one edit\n")
    assert hashlib.sha256(variant.read_bytes()).hexdigest() != on_disk


def test_dirty_tree_is_recorded_so_a_commit_hash_is_not_read_as_provenance():
    dirty = benchmark._git_dirty()
    assert set(dirty) == {"git_dirty", "git_dirty_paths"}
    # None means "could not tell" and is distinct from False, which is a claim.
    assert dirty["git_dirty"] in (True, False, None)
    if dirty["git_dirty"]:
        assert dirty["git_dirty_paths"]
    assert len(dirty["git_dirty_paths"]) <= 20


def test_the_evictor_field_name_says_what_the_evictor_actually_does():
    # @Reviewer's blocker 3. v3 corrected the prose methodology string to say
    # the evictor runs once per rotation *before* the event window, and left
    # the CSV column named `l2_eviction_between_calls` -- so the machine
    # contract still asserted the thing the prose had just withdrawn. A
    # consumer reads the column, not the paragraph.
    assert "evictor_ran_per_rotation" in benchmark.RESULT_FIELDS
    assert "l2_eviction_between_calls" not in benchmark.RESULT_FIELDS


def test_the_rename_bumped_the_schema_so_a_reader_can_tell_the_versions_apart():
    # A breaking rename that kept schema_version at 3 would leave v3 artifacts
    # from before and after the rename indistinguishable, which is the same
    # failure as an unrecorded skip: two different things reading identically.
    # The frozen artifacts under AI/gate_llc_before_after/ are v2 and v3 and
    # legitimately carry the old name.
    # The parent emitted v3 under both field names, so this asserts the bump
    # itself and fails against it -- a frozen-artifact check alone would pass
    # on the parent and guard nothing.
    #
    # Asserted as an invariant rather than against the literal 4 it was written
    # for: the first version pinned `count('"schema_version": 4,') == 2` and
    # broke the moment v5 was added, which would have trained the next person
    # to edit the number until the test went quiet.
    #
    # But that over-corrected, and @Reviewer caught it: `current >= 4` also
    # passes when a NEW field is added and the version is left where it is.
    # He constructed it -- peak_bw_probe present, both sites still 4 -- and all
    # 79 tests passed. "Future bumps should not require editing the test" is a
    # reason to assert the invariant; it is not a reason to stop guarding the
    # bump in front of you. So both hold here: the invariant, and a floor that
    # rises when a contract change lands. The floor is a separate assertion
    # from the field checks so that a failure says which one broke.
    #
    # The two emission sites now read one constant, which is what makes "the
    # sites agree" structural instead of textual.
    assert benchmark.SCHEMA_VERSION >= 5, "peak_bw_probe added a field; v5 is the floor"
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    emitted = re.findall(r'"schema_version": ([A-Za-z_0-9]+),', source)
    assert len(emitted) == 2, "row and environment are the two emission sites"
    assert emitted[0] == emitted[1] == "SCHEMA_VERSION", (
        f"both sites must read the single constant, got {emitted}"
    )
    current = benchmark.SCHEMA_VERSION
    frozen = Path(benchmark.__file__).resolve().parents[1] / "AI" / "gate_llc_before_after"
    for environment_path in sorted(frozen.glob("*/environment.json")):
        recorded = json.loads(environment_path.read_text(encoding="utf-8"))["schema_version"]
        assert recorded < 4, f"{environment_path} predates the rename but claims v{recorded}"


def test_a_row_records_which_probe_its_peak_percentage_divides_by():
    # Measured cross-vendor on 2026-08-02, fwd bf16, this harness on both
    # hosts: the achievable-bandwidth probe that wins differs by host
    # (two_read_one_write on H200 at 4314 GB/s, write on MI355X at 6664), so
    # peak_bw_pct on the two hosts divides by different references. Under
    # `copy` FlyDSL leads cutedsl by +30.0 points at 32768x8192; under `write`
    # it trails by -32.7. The denominator inverts the conclusion.
    #
    # comparison_scope used to recommend peak_bw_pct as the cross-vendor
    # measure, which made the one number it named the least comparable one.
    # The column cannot be made comparable here -- the fix is to record which
    # probe it came from so the incomparability is visible in results.csv
    # alone, rather than only to a reader who also opens environment.json.
    assert "peak_bw_probe" in benchmark.RESULT_FIELDS
    assert benchmark.RESULT_FIELDS.index("peak_bw_probe") == (
        benchmark.RESULT_FIELDS.index("peak_bw_pct") + 1
    ), "the probe belongs beside the number it qualifies"


def test_comparison_scope_does_not_recommend_the_column_it_cannot_compare():
    # The prose half of the same defect, and the shape that keeps recurring
    # here: a methodology string asserting a property the data does not have.
    # "compare peak_bw_pct instead" was advice that produced a sign error.
    torch = types.SimpleNamespace(
        __version__="2.9.1",
        version=types.SimpleNamespace(hip="7.2", cuda=None),
        cuda=types.SimpleNamespace(
            device_count=lambda: 1,
            get_device_properties=lambda _: types.SimpleNamespace(
                name="AMD Instinct MI355X",
                gcnArchName="gfx950:sramecc+:xnack-",
                total_memory=309220868096,
                multi_processor_count=256,
                L2_cache_size=4 * 1024**2,
            ),
        ),
    )
    environment = benchmark._environment(
        torch,
        argparse.Namespace(
            shapes=[(512, 4096)],
            dtype_weight_modes=[("bfloat16", "same")],
            operations=["fwd"],
            providers=["flydsl"],
            eps=1e-6,
            warmup_rounds=3,
            sample_rounds=40,
            max_rotation_buffers=4,
            l2_target_ratio=3.0,
        ),
        Path("/tmp"),
    )
    scope = environment["comparison_scope"]
    assert "compare peak_bw_pct instead" not in scope
    assert "peak_bw_probe" in scope or "same probe" in scope


def test_the_methodology_string_points_at_evidence_instead_of_asserting_a_number():
    # @Reviewer's blocker 4: a machine-readable methodology should not state an
    # unarchived diagnostic as fact. The 178% figure was never archived and did
    # not reproduce. The string now cites the committed probe rather than
    # quoting a magnitude -- which is also why this test survived the
    # correction below unchanged, and why it is written against the *shape* of
    # the string rather than its numbers.
    #
    # This comment used to gloss the measured per-call over-read as
    # "+52%/+18%/+6%". Those were mine and they were wrong: unprofiled event
    # medians divided by the per_rotation phase's hardware median, mixing
    # profiler regimes and borrowing another run's baseline. It then quoted the
    # stored field as +138%/+50%/+22% per-call, which was the right field but
    # still a single run: with 5 repeats the probe records ranges, and the
    # launch-bound cell spans +132..157%. See over_read_range in the sidecar.
    # A test comment is not machine-readable, but it is read by the next person
    # deciding what the artifact says, so it gets the same standard as the
    # string it is testing.
    torch = types.SimpleNamespace(
        __version__="2.9.1",
        version=types.SimpleNamespace(hip="7.2", cuda=None),
        cuda=types.SimpleNamespace(
            device_count=lambda: 1,
            get_device_properties=lambda _: types.SimpleNamespace(
                name="AMD Instinct MI355X",
                gcnArchName="gfx950:sramecc+:xnack-",
                multi_processor_count=256,
                total_memory=309220868096,
                L2_cache_size=4 * 1024**2,
            ),
        ),
    )
    args = argparse.Namespace(
        shapes=[(512, 4096)],
        dtype_weight_modes=[("bfloat16", "same")],
        operations=["fwd"],
        providers=["torch"],
        eps=1e-6,
        warmup_rounds=1,
        sample_rounds=1,
        max_rotation_buffers=4,
        l2_target_ratio=3.0,
    )
    steady = benchmark._environment(torch, args, Path("/tmp"))["methodology"]["steady_state"]
    assert "178%" not in steady
    assert "probe_event_timing_calibration" in steady
    assert "torch.cuda.Event" in steady


def test_a_topology_value_is_never_the_torch_fallback_wearing_its_label(tmp_path):
    # @Reviewer's blocker 1 against 53c1d4d. `max(best, fallback)` merged the
    # two numbers and kept the topology's source label, so a node reporting
    # only a 4 MiB L2 returned 268435456 tagged `kfd_topology` whenever torch
    # happened to claim a large L2 -- and _resolve_llc's value test passed on a
    # MALL that KFD never observed. A value and its provenance have to travel
    # together.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, GPU_AT_BDF, ((2, 4096),))  # topology sees only 4 MiB
    torch = _hip_torch()
    properties = _properties(l2=256 * 1024**2)  # torch claims 256 MiB

    value, provenance = benchmark._last_level_cache_bytes(torch, properties, str(root))

    assert value == 4 * 1024**2, "the returned number must be the one KFD reported"
    assert provenance["source"] == "kfd_topology"
    # The larger torch figure is not discarded, it is labelled -- a caller that
    # wants it must be able to see that taking it leaves the topology.
    assert provenance["torch_l2_exceeds_topology"] == {
        "topology_bytes": 4 * 1024**2,
        "torch_l2_bytes": 256 * 1024**2,
    }


def test_gfx950_refuses_a_mall_sized_number_the_topology_never_reported(tmp_path):
    # The end-to-end half of the same blocker: the laundered value reached
    # _resolve_llc and was accepted, because source, degradation and value all
    # looked clean. Only the value was a lie.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, GPU_AT_BDF, ((2, 4096),))
    with pytest.raises(RuntimeError, match="the two sources contradict each other"):
        benchmark._resolve_llc(
            _gfx950_torch(),
            _properties(l2=256 * 1024**2),
            argparse.Namespace(llc_bytes=None),
            str(root),
        )


def test_gfx950_refuses_a_conflict_even_when_the_topology_value_is_right(tmp_path):
    # The hole @Autotune's correction led me to, in my own 50db350. The three
    # accept conditions all held -- source kfd_topology, no degradation, value
    # at the MALL -- so the early return fired before the conflict reason was
    # ever built, and a topology contradicted by torch was accepted because it
    # happened to be the right size. Recording `torch_l2_exceeds_topology` and
    # then not gating on it is the recording-without-acting half of the defect.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, GPU_AT_BDF, ((3, 262144),))  # topology reports exactly the MALL
    with pytest.raises(RuntimeError, match="the two sources contradict each other"):
        benchmark._resolve_llc(
            _gfx950_torch(),
            _properties(l2=512 * 1024**2),  # torch claims twice it
            argparse.Namespace(llc_bytes=None),
            str(root),
        )


def test_gfx950_still_accepts_the_topology_when_nothing_contradicts_it(tmp_path):
    # Over-refusal control for the gate above, and the case the live host is
    # in: KFD reports 256 MiB, torch reports its 4 MiB L2, `fallback > best` is
    # false, no conflict flag, accepted. Verified 8/8 on real hardware.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, GPU_AT_BDF, ((2, 4096), (3, 262144)))
    value, provenance = benchmark._resolve_llc(
        _gfx950_torch(),
        _properties(l2=4 * 1024**2),
        argparse.Namespace(llc_bytes=None),
        str(root),
    )
    assert value == 256 * 1024**2
    assert provenance["source"] == "kfd_topology"
    assert "torch_l2_exceeds_topology" not in provenance


@pytest.mark.parametrize(
    "properties_text,why",
    [
        (
            f"gfx_target_version 4294967296\nunique_id {REAL_UID}\ndomain 0\nlocation_id 29952\n",
            "2**32 passed the != 0 CPU test and was read as a GPU",
        ),
    ],
)
def test_an_identity_field_above_its_driver_width_is_unreadable(tmp_path, properties_text, why):
    # The upper half of the range. @Reviewer raised it against @Autotune's tree
    # and it reproduced here verbatim: `_field` checked `>= 0` only, so
    # "unsigned" was half an invariant and 2**32 sailed through. Widths
    # measured on the live topology first -- gfx_target_version tops out at
    # 90500, domain at 0, location_id at 62720 -- so nothing healthy is
    # reclassified by a 32-bit bound.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, properties_text, ((3, 262144),))
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 4 * 1024**2, why
    assert provenance["source"] == "torch_l2_fallback", why


@pytest.mark.parametrize(
    "properties_text,why",
    [
        (
            "gfx_target_version 90500\ndomain 4294967296\nlocation_id 29952\n",
            "an impossible domain was treated as an ordinary non-match",
        ),
        (
            "gfx_target_version 90500\ndomain 0\nlocation_id 4294967296\n",
            "2**32 masks to bus 0 device 0 and the mask cannot reject it",
        ),
    ],
)
def test_an_oversized_pci_field_is_unreadable_on_the_path_that_uses_it(
    tmp_path, properties_text, why
):
    # These have no unique_id, so the PCI address is what would identify the
    # node and its fields are load-bearing. I first wrote them with a matching
    # unique_id and they failed -- correctly: on that path the node asserts its
    # identity directly and the PCI fields are not what the match rests on, so
    # refusing over them is over-refusal. The same malformed field is
    # disqualifying or irrelevant depending on what the match is standing on,
    # which is the distinction @Reviewer's branch-asymmetry point turns on.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, properties_text, ((3, 262144),))
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert value == 4 * 1024**2, why
    assert provenance["source"] == "torch_l2_fallback", why
    # The value alone does not discriminate here, and asserting only it would
    # have been a test that passes on the parent while guarding nothing: the
    # parent also declines, but by *accidental non-match* -- 2**32 simply fails
    # the comparison -- and records nothing, so an unreadable node is
    # indistinguishable from an absent one. What changed is that the skip is
    # now evidence the caller can see.
    assert provenance["skipped_nodes"] == ["2:unparseable_properties"], why


def test_a_unique_id_above_sixty_four_bits_is_refused_on_the_pci_path(tmp_path):
    # Validated even where it is not read. A node asserting an identity that
    # cannot exist is not a node whose other fields are more believable, and
    # leaving it unchecked on this path is what makes validation depend on
    # which branch the caller took.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root,
        2,
        "gfx_target_version 90500\nunique_id 18446744073709551616\ndomain 0\nlocation_id 29952\n",
        ((3, 262144),),
    )
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert value == 4 * 1024**2
    assert provenance["source"] == "torch_l2_fallback"


def test_the_real_sixty_four_bit_unique_id_is_still_accepted(tmp_path):
    # Over-refusal control for the bound above, from the live topology: the
    # largest unique_id this host publishes is 18206932166487137716, which
    # needs all 64 bits. A blanket 32-bit rule would have refused every node on
    # the machine, which is why the width is per-field and measured.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root,
        2,
        "gfx_target_version 90500\nunique_id 18206932166487137716\ndomain 0\nlocation_id 29952\n",
        ((3, 262144),),
    )
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=b"fcac045749195db4"), str(root)
    )
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "unique_id"


def test_off_gfx950_the_larger_torch_value_carries_torch_as_its_source(tmp_path):
    # Over-refusal control, and @Reviewer's blocker against 50db350 in one
    # test. Not merging the two numbers must not silently make every
    # non-gfx950 host size its rotation against a smaller cache than before;
    # off gfx950 the LLC is a sizing hint and the conservative choice is still
    # the larger figure. But when this branch takes torch's number it must say
    # so in `source` -- returning 268435456 under `source: kfd_topology` from a
    # node that reported 4 MiB is the laundering of 53c1d4d done one level up,
    # in the caller I moved the choice into. A side field is not a substitute
    # for the field a consumer actually reads.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, GPU_AT_BDF, ((2, 4096),))
    value, provenance = benchmark._resolve_llc(
        _gfx950_torch(arch="gfx942"),
        _properties(l2=256 * 1024**2),
        argparse.Namespace(llc_bytes=None),
        str(root),
    )
    assert value == 256 * 1024**2
    assert provenance["source"] == "torch_l2_over_topology"
    assert provenance["topology_bytes"] == 4 * 1024**2
    assert provenance["torch_l2_exceeds_topology"]["topology_bytes"] == 4 * 1024**2


@pytest.mark.parametrize(
    "properties_text,why",
    [
        (
            f"gfx_target_version -1\nunique_id {REAL_UID}\ndomain 0\nlocation_id 29952\n",
            "a negative gfx_target_version passed the != 0 CPU test and was read as a GPU",
        ),
        (
            "gfx_target_version 90500\ndomain 0\nlocation_id -35584\n",
            "(-35584 >> 8) & 0xFF is 117 and (-35584 >> 3) & 0x1F is 0, so a negative "
            "location_id aliases this host's real bus 0x75 device 0 and clean-wins PCI",
        ),
        (
            "gfx_target_version 90500\nunique_id -1\ndomain 0\nlocation_id 29952\n",
            "a negative unique_id was accepted as an identity",
        ),
    ],
)
def test_a_negative_identity_field_is_unreadable_not_a_small_number(tmp_path, properties_text, why):
    # @Reviewer's blocker 2 against 53c1d4d. Every integer KFD publishes is
    # unsigned -- verified against the live topology, 35332 integer fields
    # across 10 nodes and 4370 cache entries, zero negative -- so a negative
    # value is a malformed read, not a small one. The location_id case is the
    # sharp one: a masked bitfield cannot reject its own garbage, so the range
    # check has to run before the mask.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, properties_text, ((3, 262144),))
    uuid_text = b"a60c2956cd9dd4c5" if "unique_id -1" not in properties_text else None
    if "location_id -35584" in properties_text:
        uuid_text = None  # force the PCI path, which is what aliases

    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=uuid_text), str(root)
    )

    assert value == 4 * 1024**2, why
    assert provenance["source"] == "torch_l2_fallback"
    assert provenance["skipped_nodes"] == ["2:unparseable_properties"], why


def test_a_real_unique_id_above_two_to_the_63_still_parses():
    # Over-refusal control for the unsigned check. These are unsigned 64-bit:
    # this host publishes 18206932166487137716, which is above 2**63 and would
    # be negative if anything treated it as signed. Rejecting negatives must
    # not reject the top half of the legitimate range.
    assert 18206932166487137716 > 2**63
    root = Path("/sys/class/kfd/kfd/topology/nodes")
    if not root.is_dir():
        pytest.skip("no KFD topology on this host")
    seen = []
    for node in sorted(root.iterdir()):
        try:
            text = (node / "properties").read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "unique_id":
                seen.append(int(parts[1]))
    assert all(value >= 0 for value in seen)


def test_a_uid_match_survives_a_malformed_field_it_did_not_need(tmp_path):
    # Over-refusal control, and the invariant the resolver documents: a node
    # that *asserted* the identity we asked for is positive evidence, and a
    # node that failed to parse asserted nothing, so an unreadable `domain` on
    # the very node whose unique_id matched cannot retract the match. I wrote
    # this case first asserting the opposite -- that any negative field makes
    # the node unreadable -- and the code was right and the test was wrong.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root,
        2,
        f"gfx_target_version 90500\nunique_id {REAL_UID}\ndomain -1\nlocation_id 29952\n",
        ((3, 262144),),
    )
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "unique_id"
    assert "degraded" not in provenance


def test_a_malformed_domain_still_cannot_hand_the_read_to_a_pci_neighbour(tmp_path):
    # The same negative `domain`, with torch's unique_id unavailable so the PCI
    # key is all there is. Now the malformed node is a candidate we neither
    # confirmed nor excluded, its neighbour at the same address must not win
    # clean, and on gfx950 that is fatal rather than advisory -- both cards
    # report the same 256 MiB, so the number cannot reveal the mix-up.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root,
        2,
        f"gfx_target_version 90500\nunique_id {REAL_UID}\ndomain -1\nlocation_id 29952\n",
        ((3, 262144),),
    )
    _node(root, 3, "gfx_target_version 90500\ndomain 0\nlocation_id 29952\n", ((3, 262144),))

    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "pci_domain_bus_device"
    assert provenance["degraded"] == ["unidentified_nodes"]
    assert provenance["skipped_nodes"] == ["2:unparseable_properties"]

    with pytest.raises(RuntimeError, match="may belong to a different device"):
        benchmark._resolve_llc(
            _gfx950_torch(),
            _properties(uuid_text=None),
            argparse.Namespace(llc_bytes=None),
            str(root),
        )


@pytest.mark.parametrize(
    "body,tag",
    [
        # Every one of these produced a confident-looking read before the
        # parser was made strict. `dict(line.split()[:2] ...)` was lenient in
        # three ways at once and each way ends the same place: a number the
        # driver never wrote, returned as a clean kfd_topology value that then
        # sizes every rotation buffer the harness allocates.
        #
        # Duplicate key: last-wins picked the second silently, so a file
        # asserting both 262144 and 1 read as a 1 KB last level.
        ("level 3\nsize 262144\nsize 1\n", "bad_size_level3"),
        # Trailing junk: the [:2] slice discarded it, making `size 262144 extra`
        # indistinguishable from `size 262144`.
        ("level 3\nsize 262144 extra\n", "bad_size_level3"),
        # int()'s grammar is wider than the driver's output. The underscore
        # form is the sharp one -- it yields a *correct-looking* 262144 from a
        # file the driver could not have produced, which is the same
        # right-answer-wrong-provenance shape as the laundering fix.
        ("level 3\nsize 262_144\n", "bad_size_level3"),
        ("level 3\nsize +262144\n", "bad_size_level3"),
        ("level 3\n  size 262144  \n", "bad_size_level3"),
        ("level 3\nsize\t262144\n", "bad_size_level3"),
        # A repeated key whose copies AGREE. This was on the accept list until
        # @Reviewer pointed out that agreement between two copies is a fact
        # about the corruption, not about the value: the measured grammar has
        # no duplicates at all, so the file is not one this driver wrote and
        # the reader should not be the one deciding that is fine.
        ("level 3\nsize 262144\nsize 262144\n", "bad_size_level3"),
        # Lexical signedness, which no numeric range check can recover:
        # `int("-0") == 0` passes every `0 <= value` test. Here it is the
        # narrow case -- `size -0` -- but the same grammar hole let
        # `gfx_target_version -0` classify a GPU node as a CPU one.
        ("level 3\nsize -0\n", "bad_size_level3"),
        ("level 3\nsize -1\n", "bad_size_level3"),
        ("level -0\nsize 262144\n", "bad_level"),
    ],
)
def test_a_value_the_driver_could_not_have_written_is_not_read_as_clean(tmp_path, body, tag):
    root = _gfx950_node(tmp_path, [body])
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)

    # Not merely "does not return the bad number" -- it must also say so. All
    # six of these previously returned a value with no degradation recorded.
    assert value == 4 * 1024**2
    assert provenance["source"] == "torch_l2_fallback"
    assert provenance["skipped_cache_entries"] == [f"0:{tag}"]

    with pytest.raises(RuntimeError, match="could not be read"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), root
        )


@pytest.mark.parametrize(
    "body",
    [
        # The strictness must not reclassify anything the driver actually
        # writes. Measured on this host before tightening: 39702 lines across
        # 10 nodes and 4370 cache entries, every non-conforming line a
        # sibling_map CSV row, and zero duplicate keys anywhere. A trailing
        # blank line is the shape a reader could plausibly meet without the
        # file being wrong.
        #
        # A repeated *identical* key used to be in this list, on the reasoning
        # that the two copies agree so nothing is being chosen between them. It
        # has moved to the rejection list above: the measured grammar has zero
        # duplicates of any kind, so the repeat itself is the evidence that
        # this file is not one the driver wrote, and the agreement of its two
        # copies is a property of the corruption rather than of the value.
        "level 3\nsize 262144\nsibling_map 1,0,0,1\n",
        "level 3\nsize 262144\n\n",
        "level 3\nsize 262144\n   \n",
    ],
)
def test_the_strict_parser_still_accepts_what_the_driver_writes(tmp_path, body):
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(), _gfx950_node(tmp_path, [body])
    )
    assert value == 256 * 1024**2
    assert provenance["source"] == "kfd_topology"
    assert "degraded" not in provenance


@pytest.mark.parametrize(
    "extra",
    [
        # Over-refusal controls for the same reader. The first version of the
        # strict parser returned an anomaly list and every caller refused the
        # whole node on anomalies[0], which rebuilt the over-refusal 10d9480
        # had just fixed -- a UID-matched node died over a field the match
        # never rested on, and over fields this module does not even read.
        "domain not-a-number\nlocation_id 29952\n",  # unread on the UID path
        "domain 0\nlocation_id 29952\nmax_waves_per_simd 8\nmax_waves_per_simd 9\n",
        "domain 0\nlocation_id 29952\nsibling_map 1,0,0,1\n",
    ],
)
def test_a_uid_match_survives_a_malformed_line_it_did_not_need(tmp_path, extra):
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, f"gfx_target_version 90500\nunique_id {REAL_UID}\n" + extra, ((3, 262144),))

    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "unique_id"
    assert "degraded" not in provenance


def test_an_empty_cache_directory_is_not_reported_as_a_part_without_a_mall(tmp_path):
    # Enumeration completeness. `no_level2_plus_cache` is a claim about the
    # *hardware* -- this part has no cache above L2 -- and it was also what got
    # reported when every entry that would have answered failed to parse, which
    # is a claim about the *read*. The gfx950 path fails closed either way, so
    # this is not a soundness hole; it is a reader being sent to the wrong place
    # to look, which is the same defect class in the diagnostics.
    empty = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(), _gfx950_node(tmp_path, [])
    )[1]
    assert empty["reason"] == "no_level2_plus_cache"
    assert "skipped_cache_entries" not in empty


def test_a_cache_directory_whose_entries_all_failed_says_so(tmp_path):
    unusable = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(), _gfx950_node(tmp_path, ["level 3\nsize 262_144\n"])
    )[1]
    assert unusable["reason"] == "no_usable_level2_plus_cache"
    assert unusable["skipped_cache_entries"] == ["0:bad_size_level3"]


def test_the_over_read_figures_match_the_field_the_probe_actually_stores():
    # The blocker @Reviewer found against 43ffc5b, turned into something that
    # cannot come back quietly. The docstring justifying per-rotation timing
    # quoted six percentages that no field in the committed sidecar contains:
    # they divided the *unprofiled* event median by the *per_rotation* phase's
    # hardware median -- crossing profiler regimes, and in the per-call row
    # charging one process's event timing against another process's hardware
    # baseline. The probe defines over_read_vs_hardware as the profiled pair
    # and says so in its own docstring.
    #
    # Archiving the data is what let him recompute and refute it. It did not by
    # itself make the derived number honest, which is the point worth keeping:
    # a figure that is not the stored field needs its derivation shown.
    sidecar = json.loads(
        (BENCHMARK_PATH.parents[1] / "AI" / "probe_event_timing_calibration.json").read_text()
    )
    # The first version of this test asserted the six percentages as an
    # unordered bag -- `f"+{pct}%" in doc` for each, with no idea which shape
    # or which mode it belonged to. @Reviewer demonstrated the hole by swapping
    # the first two per-call values to `+50 / +138 / +22`; the test still
    # passed 1/1. A set membership check cannot see a permutation, and a
    # permutation here is the whole claim: the numbers are cited to show the
    # over-read *shrinks as the kernel grows* and that per-rotation is smaller
    # than per-call at every shape. Both of those are statements about order.
    #
    # So the mapping is reconstructed positionally from the docstring's own
    # table and compared shape-by-shape against the sidecar.
    source = BENCHMARK_PATH.read_text()
    doc = source[source.index("def _time_rotating_calls") :]
    doc = doc[: doc.index('"""', doc.index('"""') + 3)]

    shapes = re.search(r"at ``(\d+x\d+)`` / ``(\d+x\d+)`` / ``(\d+x\d+)``", doc)
    assert shapes, "the docstring must name the shapes its table is ordered by"
    order = list(shapes.groups())

    # Ranges, not points. The probe now runs each phase 5 times, and the
    # repeats showed the previous single figures were quoted to a precision
    # the measurement does not have: `512x4096` per-call moves 132-157% run to
    # run, so the `+138%` this test used to pin was one draw from a spread. A
    # test that pins a single draw to the integer *enforces* that overclaim --
    # it would fail on an honest re-run and pass only on a lucky one.
    quoted: dict[tuple[str, str], tuple[int, int]] = {}
    for mode, label in (("per_call", "per-call"), ("per_rotation", "per-rotation")):
        row = re.search(rf"^\s*{label}\s+(.+)$", doc, re.MULTILINE)
        assert row, f"the docstring must carry a {label} row"
        values = re.findall(r"\+(\d+)\.\.(\d+)%", row.group(1))
        assert len(values) == len(order), f"{label} row must cover every shape named"
        for shape, (lo, hi) in zip(order, values):
            quoted[(shape, mode)] = (int(lo), int(hi))

    measured: dict[tuple[str, str], tuple[float, float]] = {}
    for record in sidecar["measurements"]:
        shape = f"{record['m']}x{record['n']}"
        for mode in ("per_call", "per_rotation"):
            entry = record[mode]
            # The stored field is self-consistent: both halves come from the
            # same profiled phase of the same process.
            recomputed = entry["event_median_us_profiled"] / entry["hardware_median_us"] - 1.0
            assert recomputed == pytest.approx(entry["over_read_vs_hardware"], rel=1e-9)
            # And the range is the range of the repeats, not a wider band
            # someone widened by hand to make a stale citation fit.
            spread = [rep["over_read_vs_hardware"] for rep in entry["repeats"]]
            assert len(spread) >= 2, (shape, mode, "a single run cannot show a range")
            assert entry["over_read_range"] == [min(spread), max(spread)]
            assert entry["over_read_vs_hardware"] == spread[0]
            measured[(shape, mode)] = (min(spread), max(spread))

    # Every quoted bound is that shape's and that mode's own measured bound,
    # rounded outward. A swap still fails, because the key is what is asserted.
    assert set(quoted) == set(measured)
    for key, (lo, hi) in quoted.items():
        m_lo, m_hi = measured[key]
        assert lo == math.floor(m_lo * 100), key
        assert hi == math.ceil(m_hi * 100), key

    # And the two orderings the figures are cited to support, asserted rather
    # than left to the reader to spot in the table. Against the *ranges*: the
    # docstring claims these hold in every repeat, so overlapping bands would
    # make that sentence false even if the medians were ordered correctly.
    for shape in order:
        assert measured[(shape, "per_rotation")][1] < measured[(shape, "per_call")][0], shape
    kernel_order = sorted(order, key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))
    for mode in ("per_call", "per_rotation"):
        for larger, smaller in zip(kernel_order, kernel_order[1:]):
            assert measured[(larger, mode)][0] > measured[(smaller, mode)][1], (
                mode,
                larger,
                smaller,
            )


def test_the_noise_floor_a_canary_is_read_against_is_the_measured_one():
    # @Reviewer, 2026-08-02: he will not take a canary as stability evidence.
    # Correct, and I had just done it -- reported a canary matching an archived
    # run to 0.40% as if that showed nothing regressed, in the same message
    # arguing that a single draw cannot show reproducibility. The canary is a
    # single draw. The reasoning I applied to the probe's figures is the
    # reasoning I failed to apply to my own verification step.
    #
    # The repeats make the missing number available: how far the same
    # measurement moves with nothing changed. The notes now carry it as a
    # table, and a table of measured figures goes stale exactly the way the
    # over-read figures did, so it is pinned to the sidecar rather than typed.
    sidecar = json.loads(
        (BENCHMARK_PATH.parents[1] / "AI" / "probe_event_timing_calibration.json").read_text()
    )
    notes = (BENCHMARK_PATH.parents[1] / "AI" / "flydsl_rmsnorm_notes.md").read_text()

    record = next(r for r in sidecar["measurements"] if (r["m"], r["n"]) == (32768, 1024))
    reps = record["per_rotation"]["repeats"]
    fields = {
        "unprofiled event median": "event_median_us_unprofiled",
        "profiled event median": "event_median_us_profiled",
        "rocprofv3 hardware median": "hardware_median_us",
    }
    for label, field in fields.items():
        values = [rep[field] for rep in reps]
        spread = (max(values) / min(values) - 1.0) * 100
        row = re.search(rf"^\|\s*{label}\s*\|\s*([\d.]+)%\s*\|$", notes, re.MULTILINE)
        assert row, label
        assert float(row.group(1)) == pytest.approx(spread, abs=0.005), label

    # And the point the table exists to make, which is the part a reader acts
    # on: the canary agreement I quoted is *inside* that floor. If a future
    # run tightened the noise enough for 0.40% to sit outside it, this claim
    # would need rewriting rather than silently continuing to look supported.
    unprofiled = [rep["event_median_us_unprofiled"] for rep in reps]
    floor = (max(unprofiled) / min(unprofiled) - 1.0) * 100
    assert floor > 0.40, floor
    assert "is not stability evidence" in notes


def test_the_sidecar_names_the_code_and_the_card_that_produced_it():
    # @Reviewer's last item on this artifact: it was *checkable* but not
    # *authenticated*. He recomputed from it and refuted a figure I had quoted,
    # which is exactly what it was written for -- and that worked without the
    # file pinning a script, a commit, a torch build or a card. So the
    # arithmetic inside could be verified while "is this a measurement of the
    # tree in front of me?" stayed open.
    #
    # That is the recurring defect one level up: an artifact that cannot say
    # which code produced it is indistinguishable from one that describes a
    # different tree, in the same way an assumed field is indistinguishable
    # from an observed one.
    sidecar = json.loads(
        (BENCHMARK_PATH.parents[1] / "AI" / "probe_event_timing_calibration.json").read_text()
    )
    env = sidecar["environment"]
    probe = BENCHMARK_PATH.parents[1] / "AI" / "probe_event_timing_calibration.py"

    # The hash is of the probe as committed. If the script is edited without
    # re-running it, this fails -- which is the point: the sidecar would then
    # be describing code that is no longer here.
    assert env["script_path"] == "AI/probe_event_timing_calibration.py"
    assert env["script_sha256"] == hashlib.sha256(probe.read_bytes()).hexdigest()

    # A commit is only evidence if the tree was clean, so the flag travels with
    # it rather than being inferred. `source_dirty` true is honest, not a
    # failure -- the sidecar is regenerated before the commit that carries it,
    # so the tree necessarily has uncommitted changes at that moment. What
    # would be dishonest is a hash with no flag beside it.
    assert re.fullmatch(r"[0-9a-f]{40}", env["source_commit"])
    assert isinstance(env["source_dirty"], bool)

    assert env["torch_version"] and env["torch_hip"]

    # The physical card, not an ordinal a visibility mask renumbers -- the same
    # identity the harness matches KFD nodes on, and the same shape rule.
    gpu = env["gpu"]
    assert benchmark._TORCH_UID_TEXT.match(gpu["uuid"]), gpu["uuid"]
    assert re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}", gpu["bdf"]), gpu["bdf"]
    assert gpu["name"] and gpu["arch"]

    # And the reduction from trace to median is auditable rather than trusted.
    # Every repeat records what the trace contained and what the window
    # discarded, so a selection that silently dropped dispatches it should not
    # have is visible instead of looking identical to a clean one.
    for record in sidecar["measurements"]:
        for mode in ("per_call", "per_rotation"):
            for rep in record[mode]["repeats"]:
                assert re.fullmatch(r"[0-9a-f]{64}", rep["trace"]["sha256"])
                assert rep["trace"]["bytes"] > 0
                census = rep["trace_census"]
                # The window is a suffix of the selected kernel's dispatches,
                # and what it dropped is a number rather than an assurance.
                assert census["selected_kernel_dispatches"] >= census["window_requested"]
                assert census["leading_dispatches_dropped"] == (
                    census["selected_kernel_dispatches"] - census["window_requested"]
                )
                assert census["window_requested"] == rep["hardware_dispatches"]
                assert census["window_requested"] == len(rep["hardware_samples_us"])
                # Every dispatch row is attributed to some kernel name, so a
                # second kernel appearing cannot hide inside the total.
                assert sum(census["dispatches_by_kernel"].values()) == census["dispatch_rows"]
                assert (
                    census["dispatches_by_kernel"][rep["kernel"]]
                    == census["selected_kernel_dispatches"]
                )


def test_the_withdrawn_hybrid_figures_are_gone_from_every_artifact_that_quoted_them():
    # The blocker was fixed where it was reported and left standing where it
    # was not. `AI/gate_llc_before_after/README.md` carried the same withdrawn
    # arithmetic -- `+16%` at 512x4096 and `+1%` at 32768x1024, both the
    # unprofiled numerator over the profiled denominator -- for as long as the
    # docstring did, and no test read that file. @Reviewer's point: a
    # correction scoped to the place the blocker pointed at is not a
    # correction, and the prose artifact is the one a reader reaches for first.
    sidecar = json.loads(
        (BENCHMARK_PATH.parents[1] / "AI" / "probe_event_timing_calibration.json").read_text()
    )
    readme = (BENCHMARK_PATH.parents[1] / "AI" / "gate_llc_before_after" / "README.md").read_text()

    stored = {
        f"{r['m']}x{r['n']}": {
            m: (
                math.floor(r[m]["over_read_range"][0] * 100),
                math.ceil(r[m]["over_read_range"][1] * 100),
            )
            for m in ("per_call", "per_rotation")
        }
        for r in sidecar["measurements"]
    }
    # The hybrid a reader could recompute by mixing the two regimes, which is
    # exactly how both wrong versions were produced. Every repeat, since a
    # stale figure only has to match *some* run to look plausible.
    withdrawn = set()
    for record in sidecar["measurements"]:
        for mode in ("per_call", "per_rotation"):
            for rep, denom in zip(record[mode]["repeats"], record["per_rotation"]["repeats"]):
                hybrid = rep["event_median_us_unprofiled"] / denom["hardware_median_us"] - 1.0
                withdrawn.add(f"+{round(hybrid * 100):d}%")

    # Minus the ones that collide with a correct figure. A hybrid can round to
    # the same integer as a legitimate bound -- when this test was written,
    # `4096x4096`'s hybrid per-rotation value and `32768x1024`'s stored
    # per-rotation value were both `+5%`, the same four characters, one wrong
    # and one right. Banning the string outright would forbid the correct
    # number, so for those the string carries no information and the keyed
    # check below is what does the work. Worth naming rather than quietly
    # excluding: a ban list that cannot tell two figures apart is exactly the
    # conflation this suite keeps finding, and pretending otherwise would make
    # the test look stronger than it is. The collision set is computed, not
    # hard-coded, so it tracks whatever the current data happens to collide on.
    legitimate = {
        f"+{bound:d}%" for modes in stored.values() for bounds in modes.values() for bound in bounds
    }
    withdrawn -= legitimate

    # The paragraph explaining the withdrawal is allowed to name them -- a
    # correction that cannot say what it is correcting is not much of a record.
    # Everything before that paragraph is the file's own claim, and the
    # withdrawn figures must not appear there. The marker is the paragraph's
    # opening `**`, not the phrase inside it, so the figures it quotes fall on
    # the explanatory side of the split rather than the asserting side.
    marker = "**The `+"
    assert marker in readme, "the withdrawal has to stay on the record"
    assert "were withdrawn on" in readme[readme.index(marker) :]
    claim = readme[: readme.index(marker)]
    for figure in withdrawn:
        assert figure not in claim, figure

    # And the figures it does quote are the stored ones, keyed by shape --
    # which is what catches the collision case the ban list cannot. The
    # paragraph must state each over-read *adjacent to the shape it belongs
    # to*, so quoting one shape's figure while calling it another fails on the
    # pairing even when the string is a legitimate figure somewhere in the file.
    #
    # Ranges, matching the sidecar's own `over_read_range`. A README that
    # quotes a single number here would be asserting a reproducibility the
    # probe now explicitly measures and contradicts.
    paragraph = readme[readme.index("What the same probe") : readme.index(marker)]
    for shape in ("512x4096", "32768x1024"):
        lo, hi = stored[shape]["per_rotation"]
        assert re.search(rf"\*\*\+{lo}\.\.{hi}%\*\*\s+at\s+`{shape}`", paragraph), shape


def _scope_environment(
    *,
    name="AMD Instinct MI355X",
    major=9,
    minor=5,
    hip="7.2",
    cuda=None,
    operations=("fwd",),
    providers=("flydsl", "torch"),
    modes=(("bfloat16", "same"),),
):
    torch = types.SimpleNamespace(
        __version__="2.9.1",
        version=types.SimpleNamespace(hip=hip, cuda=cuda),
        cuda=types.SimpleNamespace(
            device_count=lambda: 1,
            get_device_properties=lambda _: types.SimpleNamespace(
                name=name,
                gcnArchName="gfx950:sramecc+:xnack-",
                major=major,
                minor=minor,
                multi_processor_count=256,
                total_memory=309220868096,
                L2_cache_size=4 * 1024**2,
            ),
        ),
    )
    args = argparse.Namespace(
        shapes=[(512, 4096)],
        dtype_weight_modes=[list(mode) for mode in modes],
        operations=list(operations),
        providers=list(providers),
        eps=1e-6,
        seed=0,
        warmup_rounds=3,
        sample_rounds=10,
        copy_mib=256,
        copy_samples=5,
        min_rotation_buffers=2,
        max_rotation_buffers=8,
        l2_target_ratio=4.0,
        llc_bytes=None,
        expected_arch=None,
        output_dir="/tmp/unused",
    )
    return benchmark._environment(torch, args, Path("/tmp/unused"))


def test_the_machine_contract_does_not_assert_measurements_this_run_did_not_make():
    # @Reviewer's blocker 2 against fdf2530, reproduced with his construction:
    # an H100 / bwd / fp32 / torch-only run emitted a comparison_scope naming
    # 2026-08-02, H200 and MI355X, fwd bf16, 32768x8192 and +30.0/-32.7 -- an
    # experiment it had no part in. A hard-coded paragraph in a
    # machine-readable field is an assumed value that reads exactly like an
    # observed one, which is the defect this very string was rewritten to warn
    # about.
    scope = _scope_environment(
        name="NVIDIA H100 80GB HBM3",
        major=9,
        minor=0,
        hip=None,
        cuda="12.8",
        operations=("bwd",),
        providers=("torch",),
        modes=(("float32", "same"),),
    )["comparison_scope"]

    for foreign in ("2026-08-02", "fwd bf16", "MI355X", "H200", "32768x8192", "+30.0", "-32.7"):
        assert foreign not in scope, f"{foreign!r} is another run's result"
    # The general caveat must survive -- the fix is to stop asserting a
    # measurement, not to stop warning about the column.
    assert "peak_bw_pct is NOT a cross-vendor fix" in scope
    assert "peak_bw_probe" in scope


def test_the_measured_half_of_the_scope_is_appended_from_the_probe_that_ran():
    # And the numbers are not lost: they move to where they are true. The
    # caveat carries no figure, so it cannot go stale; the measurement is
    # written from live values once the probe exists.
    peak_bw = {
        "best_probe": "write",
        "median_gbps": 6664.195243,
        "probes": {
            "copy": {"gbps": 4644.1},
            "two_read_one_write": {"gbps": 6068.6},
            "write": {"gbps": 6664.195243},
        },
    }
    measured = benchmark._describe_measured_scope(peak_bw)
    assert "divided by write at 6664.2 GB/s" in measured
    for name in peak_bw["probes"]:
        assert name in measured
    assert "copy 4644.1" in measured and "two_read_one_write 6068.6" in measured


@pytest.mark.parametrize(
    "kwargs,absent,present",
    [
        # Blocker 7: the frozen H200 artifact said the LLC was "read from the
        # KFD topology, matched by unique_id" while its own provenance in the
        # same file said not_a_hip_build/torch_l2_fallback, asserted the gfx950
        # MALL on an sm_90 card, and described FlyDSL JIT on a quack+torch run.
        # A methodology that documents the template instead of the execution is
        # a claim the artifact cannot support.
        (
            {
                "hip": None,
                "cuda": "12.8",
                "providers": ("quack", "torch"),
                "name": "NVIDIA H200",
                "major": 9,
                "minor": 0,
            },
            ("256 MiB MALL", "4 MiB per-XCD"),
            ("last_level_cache_provenance",),
        ),
        (
            {"providers": ("torch",)},
            ("FlyDSL first-launch JIT synchronized",),
            ("no FlyDSL provider in this run",),
        ),
    ],
)
def test_methodology_describes_the_run_and_not_the_template(kwargs, absent, present):
    methodology = _scope_environment(**kwargs)["methodology"]
    blob = json.dumps(methodology)
    for text in absent:
        assert text not in blob, f"{text!r} is asserted regardless of what ran"
    for text in present:
        assert text in blob


def test_the_artifact_records_which_provider_code_was_imported():
    # Blocker 5. The frozen H200 artifact carries ambient git_commit 4f36477,
    # whose tree declares quack 0.6.1, while the run imported installed 0.5.0
    # from dist-packages -- and that tree does not import at all on that host,
    # so the recorded commit provably was not the provider under test. Nothing
    # in the artifact said so; establishing it needed a shell on the machine.
    # _executed_source already made the harness checkable rather than merely
    # asserted; the providers are what the harness exists to measure and had
    # weaker provenance than the file measuring them.
    import quack  # noqa: F401  -- the point is that it is in sys.modules

    modules = benchmark._imported_provider_modules()
    assert "quack" in modules
    entry = modules["quack"]
    assert entry["path"] and entry["path"].endswith("__init__.py")
    assert entry["init_sha256"] and len(entry["init_sha256"]) == 64
    # version_attr comes from the imported module, versions.quack from installed
    # distribution metadata. Recording both is the point: on the H200 they
    # disagreed, and only one of them was the code that ran.
    assert "version_attr" in entry


def test_a_run_records_the_card_and_not_just_the_ordinal():
    # The other half of blocker 5. visible_index 0 identifies a card only
    # relative to a visibility mask the environment sets, so on an 8-GPU host
    # it pins nothing. The MI355X artifact recorded ordinal 6 and node 8 while
    # dropping the UID those were matched on -- the resolver established the
    # identity and then did not carry it to the reader.
    rocm = types.SimpleNamespace(bytes=b"a60c2956cd9dd4c5")
    assert benchmark._reported_device_uuid(types.SimpleNamespace(uuid=rocm)) == "a60c2956cd9dd4c5"
    # CUDA hands over a real UUID object rather than hex text; provenance has
    # to survive either, so this must not go through the AMD-specific decode.
    import uuid as uuid_module

    nvidia = uuid_module.UUID("12345678-1234-5678-1234-567812345678")
    assert benchmark._reported_device_uuid(types.SimpleNamespace(uuid=nvidia)) == str(nvidia)
    # A runtime exposing nothing records None rather than a fabricated value.
    assert benchmark._reported_device_uuid(types.SimpleNamespace()) is None


@pytest.mark.parametrize(
    "field,line",
    [
        # `int("-0") == 0`, so a negative-zero field passes every `0 <= value`
        # range check ever written. @Reviewer's four cases, each of which
        # returned a clean, undegraded read before the grammar was tightened:
        #
        #   gfx_target_version -0  -- the worst. The node is silently
        #     classified a CPU, drops out of the scan without being recorded,
        #     and the search falls through to a same-BDF neighbour: the
        #     c597e11 wrong-device answer with skipped=None.
        ("gfx_target_version", "gfx_target_version -0\n"),
        #   unique_id -0 -- clean-matches torch bytes 0000000000000000.
        ("unique_id", "unique_id -0\n"),
        #   domain -0 / location_id -0 -- clean PCI matches.
        ("domain", "domain -0\n"),
        ("location_id", "location_id -0\n"),
    ],
)
def test_a_negative_zero_field_is_not_read_as_a_valid_zero(field, line):
    # Asserted at the reader, because that is where the fix has to live: a
    # numeric range check downstream cannot recover lexical signedness, so the
    # grammar is the only place the sign can be rejected. The withheld-key set
    # is the difference between "the driver did not publish this" and "the file
    # says something we refuse to read", and -0 must land in the second.
    path = Path(tempfile.mkdtemp()) / "properties"
    path.write_text(line)
    fields, malformed = benchmark._read_kfd_properties(str(path))
    assert field not in fields
    assert field in malformed


def test_a_negative_zero_gfx_version_cannot_hand_the_read_to_a_pci_neighbour(tmp_path):
    # The end-to-end shape of the case above, since the reader-level assertion
    # only shows the field is withheld and not what the pipeline then does with
    # it. Two nodes at one PCI address, as under CPX. The card we want has
    # `gfx_target_version -0`; before the tightening that parsed to 0, read as
    # "this is a CPU node", and skipped it with no record -- leaving the
    # neighbour as the sole surviving candidate at that address, which then won
    # the PCI match and answered with its 256 MiB. Right-looking number, wrong
    # device, `degraded` absent.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root,
        2,
        f"gfx_target_version -0\nunique_id {REAL_UID}\ndomain 0\nlocation_id 29952\n",
        ((3, 262144),),
    )
    _node(
        root,
        3,
        "gfx_target_version 90500\nunique_id 1\ndomain 0\nlocation_id 29952\n",
        ((3, 262144),),
    )

    # torch supplies no UUID, so the PCI key is the only one available -- which
    # is exactly the situation the neighbour wins in.
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    # The neighbour still wins the address, and the value is still the
    # plausible 256 MiB -- no assertion on the number can catch this, because
    # every card on this host is the same part. What changes is that the answer
    # now carries the fact that a candidate at this address went unread, which
    # is the whole difference between an elimination and a guess.
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "pci_domain_bus_device"
    assert provenance["degraded"] == ["unidentified_nodes"]
    assert provenance["skipped_nodes"] == ["2:unparseable_properties"]

    # Before the grammar was tightened this returned byte-identical provenance
    # with `degraded` absent and `skipped_nodes` absent: `-0` parsed to 0, the
    # node read as a CPU, and it left the scan without being recorded at all.
    # On gfx950 that difference is the difference between a run that stops and
    # a run that reports another card's cache as its own.
    with pytest.raises(RuntimeError, match="may belong to a different device"):
        benchmark._resolve_llc(
            _gfx950_torch(),
            _properties(uuid_text=None),
            argparse.Namespace(llc_bytes=None),
            str(root),
        )


@pytest.mark.parametrize(
    "text",
    [
        b"+a60c2956cd9dd4c5",  # int() takes a sign
        b"a60c2956_cd9dd4c5",  # int() takes underscore separators
        b" a60c2956cd9dd4c5 ",  # int() takes surrounding whitespace, as did strip()
        b"0a60c2956cd9dd4c5",  # 17 digits: a different field padded to look like this one
        b"a60c2956cd9dd4c",  # 15 digits
        b"",
    ],
)
def test_a_uuid_shape_the_runtime_could_not_have_reported_is_refused(text):
    # All four of @Reviewer's shapes decoded to the same 11964983762810164421
    # under `.strip()` + `int(text, 16)` and each returned a clean
    # matched_by=unique_id. `int` is a far wider grammar than the format being
    # recognised, and this is the identity key: a false match here answers with
    # another card's topology, silently.
    assert benchmark._torch_unique_id(_properties(uuid_text=text)) is None


def test_the_uid_shape_check_still_accepts_what_the_runtime_reports(tmp_path):
    # The over-refusal control. Exactly sixteen ASCII hex bytes is what
    # torch 2.9.1+rocm7.2 hands over on this host, verified 8/8, and both cases
    # must keep resolving -- the driver prints unique_id in lowercase but the
    # hex text itself is case-insensitive, so refusing uppercase would be
    # inventing a constraint rather than enforcing one.
    #
    # @Autotune raised uppercase as a residual in the sibling module on
    # 2026-08-02: two byte strings mapping to one identity. Checked here rather
    # than assumed to transfer, because it is a different defect class from the
    # four shapes above and the difference is the whole argument. Those decoded
    # a string the driver *cannot emit* into a valid id, so a match was against
    # a value nothing measured. Case folding maps two spellings of the *same*
    # sixteen hex digits onto the same card -- many-to-one on notation, not on
    # identity -- so it cannot produce a false match with a different device.
    # Live torch emits lowercase on all 8 (measured), so tightening would only
    # refuse input nothing sends; that is the over-refusal @Autotune's own
    # commit was undoing. Left accepting, deliberately, with the reason stated.
    assert benchmark._torch_unique_id(_properties()) == REAL_UID
    assert benchmark._torch_unique_id(_properties(uuid_text=b"A60C2956CD9DD4C5")) == REAL_UID
    assert benchmark._torch_unique_id(_properties(uuid_text=b"A60c2956Cd9dD4c5")) == REAL_UID
    # The claim that makes case folding safe, asserted rather than argued: a
    # spelling change must not move the answer to another card. Both cases
    # resolve against the same tree, and neither goes ambiguous.
    folded = tmp_path / "folded"
    folded.mkdir()
    _node(folded, 2, GPU_AT_BDF, ((3, 262144),))
    for text in (b"a60c2956cd9dd4c5", b"A60C2956CD9DD4C5"):
        value, provenance = benchmark._last_level_cache_bytes(
            _hip_torch(), _properties(uuid_text=text), str(folded)
        )
        assert value == 256 * 1024**2, text
        assert provenance["matched_by"] == "unique_id", text
        assert provenance["matched_node"].endswith("/2"), text
        assert "degraded" not in provenance, text
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, GPU_AT_BDF, ((3, 262144),))
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2
    assert provenance["matched_by"] == "unique_id"
    assert "degraded" not in provenance


def test_a_partial_cache_listing_is_not_read_as_a_complete_one(tmp_path):
    # Recording rejected entries only covers entries the loop saw. An entry
    # that never appeared in the listing is invisible to it, so a `caches/`
    # missing its level-3 entry returned a clean 4 MiB kfd_topology read: no
    # skip, no degradation, and provenance asserting a complete read of a
    # topology one line of which was never there. The count was in the file the
    # whole time and nothing consulted it. @Reviewer's two cases:
    #
    #   caches_count 546 beside a single directory -> clean 256 MiB.
    root = tmp_path / "nodes"
    root.mkdir()
    node = _node(root, 2, GPU_AT_BDF, ((3, 262144),))
    (node / "properties").write_text(GPU_AT_BDF + "caches_count 546\n")
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 4 * 1024**2
    assert provenance["reason"] == "caches_count_mismatch"
    assert provenance["declared_cache_entries"] == 546
    assert provenance["enumerated_cache_entries"] == 1

    #   caches_count 0 beside a single directory -> also clean 256 MiB. Worth
    #   its own case: an under-count is the direction a `>=` check would miss,
    #   and the reason it must be equality rather than a floor.
    (node / "properties").write_text(GPU_AT_BDF + "caches_count 0\n")
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 4 * 1024**2
    assert provenance["reason"] == "caches_count_mismatch"
    assert provenance["declared_cache_entries"] == 0

    # And on gfx950 the disagreement stops the run rather than sizing every
    # rotation buffer against a cache we could not confirm we finished reading.
    with pytest.raises(RuntimeError, match="could not be read"):
        benchmark._resolve_llc(
            _gfx950_torch(), _properties(), argparse.Namespace(llc_bytes=None), str(root)
        )


def test_an_unreadable_cache_count_is_told_apart_from_an_absent_one(tmp_path):
    # Same absent-vs-withheld distinction as everywhere else in this reader,
    # applied to the count itself: one sends a reader to the driver, the other
    # to the file. Both refuse -- the count is what makes the enumeration
    # checkable, so a node that does not supply one readably cannot be read
    # completely -- but they are not the same observation.
    root = tmp_path / "nodes"
    root.mkdir()
    node = _node(root, 2, GPU_AT_BDF, ((3, 262144),))

    (node / "properties").write_text(GPU_AT_BDF)  # no caches_count line at all
    provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))[1]
    assert provenance["reason"] == "caches_count_absent"
    assert provenance["enumerated_cache_entries"] == 1

    (node / "properties").write_text(GPU_AT_BDF + "caches_count not-a-number\n")
    provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))[1]
    assert provenance["reason"] == "caches_count_unreadable"


def test_the_enumeration_check_accepts_the_agreement_this_host_actually_has(tmp_path):
    # Over-refusal control, and the reason exact equality is safe to require:
    # measured live on all 10 KFD nodes of this host before tightening, every
    # node's caches_count equals its directory count exactly -- 0/0 twice, then
    # 546/546 seven times and 548/548. Nothing real is reclassified.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 2, GPU_AT_BDF, ((2, 4096), (3, 262144)))
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), str(root))
    assert value == 256 * 1024**2
    assert provenance["source"] == "kfd_topology"
    assert "degraded" not in provenance

    # Including the node that legitimately has none, which the live host has
    # two of. `caches_count 0` with an empty directory agrees, so this must
    # reach the hardware statement rather than the mismatch refusal.
    elsewhere = tmp_path / "empty"
    elsewhere.mkdir()
    empty = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(), _gfx950_node(elsewhere, [])
    )[1]
    assert empty["reason"] == "no_level2_plus_cache"


def test_a_packed_field_is_bounded_by_what_is_packed_into_it_not_its_storage(tmp_path):
    # @Reviewer's packed-location addendum. `location_id` is declared u32 in
    # kfd_topology.h, so a u32 ceiling looks like the right invariant -- but
    # the driver writes `pci_dev_id()` into it, which returns u16
    # (`PCI_DEVID(bus, devfn)` = `(bus << 8) | devfn`, plus `node_id` ORed into
    # the same low byte on a multi-node GPU). Nothing above bit 15 is ever set.
    # Read from the driver source on this host, amdgpu-6.16.13
    # kfd_topology.c:2162 with pci.h:70/686, and confirmed against all 10 live
    # nodes: zero high bits on every one.
    #
    # The extra room is not merely unused, it is exploitable, because this
    # field is masked before it is compared: 0x17500 is an in-range u32 that
    # aliases this host's real 0x7500 under `>> 8 & 0xFF`, so it clean-matched
    # bus 117 device 0 and answered with that node's 256 MiB. A mask cannot
    # reject what it discards.
    root = tmp_path / "nodes"
    root.mkdir()
    _node(
        root,
        2,
        f"gfx_target_version 90500\ndomain 0\nlocation_id {0x17500}\n",
        ((3, 262144),),
    )
    value, provenance = benchmark._last_level_cache_bytes(
        _hip_torch(), _properties(uuid_text=None), str(root)
    )
    assert value == 4 * 1024**2
    assert provenance["source"] == "torch_l2_fallback"
    assert provenance["skipped_nodes"] == ["2:unparseable_properties"]

    # Boundary controls in both directions, so the check is pinned to 16 rather
    # than to "somewhere between the real values and 2**32". The largest
    # location_id the packed form can hold must still resolve; the smallest it
    # cannot must not.
    for packed, resolves in ((0xFFFF, True), (0x10000, False)):
        bus, device = (packed >> 8) & 0xFF, (packed >> 3) & 0x1F
        other = tmp_path / f"nodes_{packed:x}"
        other.mkdir()
        _node(
            other, 2, f"gfx_target_version 90500\ndomain 0\nlocation_id {packed}\n", ((3, 262144),)
        )
        got = benchmark._last_level_cache_bytes(
            _hip_torch(), _properties(uuid_text=None, bus=bus, device=device), str(other)
        )[1]
        assert (got["source"] == "kfd_topology") is resolves, packed


@pytest.mark.parametrize(
    "level,size_kb,resolves",
    [
        # @Reviewer's blocker 6 against 10d9480: the ceiling tests all used
        # values at or above the ABI width, so they were passing on the width
        # check and would have passed with MAX_CACHE_LEVEL and MAX_CACHE_SIZE_KB
        # deleted entirely. The physical ceilings were argued for in a comment
        # and guarded by nothing of their own.
        #
        # Confirmed by mutation rather than by reading: widening both constants
        # to `1 << 32` -- which is deleting them as independent bounds, since
        # the width check then subsumes them -- leaves every pre-existing
        # ceiling test green (11/11) and fails exactly the two cases below.
        # That is the difference between a bound being present and a bound
        # being tested, which is this defect class applied to the test suite:
        # a check nothing exercises is indistinguishable from a check that is
        # not there.
        #
        # These sit well inside u32, so only the physical bound can reject
        # them, and they bracket it from both sides.
        (7, 262144, True),  # deepest level the ceiling admits
        (8, 262144, False),  # first it does not -- and far below 2**32
        (3, 16 * 1024 * 1024 - 1, True),  # largest size the ceiling admits
        (3, 16 * 1024 * 1024, False),  # first it does not
    ],
)
def test_the_physical_ceilings_are_load_bearing_on_their_own(tmp_path, level, size_kb, resolves):
    root = _gfx950_node(tmp_path, [f"level {level}\nsize {size_kb}\n"])
    value, provenance = benchmark._last_level_cache_bytes(_hip_torch(), _properties(), root)
    if resolves:
        assert provenance["source"] == "kfd_topology"
        assert value == size_kb * 1024
    else:
        assert provenance["source"] == "torch_l2_fallback"
        assert provenance["skipped_cache_entries"]

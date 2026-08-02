# Copyright (c) 2026, Tri Dao.

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
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


def _write_node(root, index, *, gfx=90000, domain=0, location=29952, unique_id=None, caches=()):
    node = root / str(index)
    (node / "caches").mkdir(parents=True)
    lines = [f"gfx_target_version {gfx}", f"domain {domain}", f"location_id {location}"]
    if unique_id is not None:
        lines.append(f"unique_id {unique_id}")
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
        "gfx_target_version 90500\nunique_id 11964983762810164421\ndomain 0\nlocation_id 30720\n"
    )
    for index, body in enumerate(caches):
        cache = node / "caches" / str(index)
        cache.mkdir()
        (cache / "properties").write_text(body)
    return str(root)


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

    with pytest.raises(RuntimeError, match="unparseable"):
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
        ("level 3\nsize -1\n", "nonpositive_size_level3"),
        ("", "missing_level"),
        ("level 3\nsize not-a-number\n", "bad_size_level3"),
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

    with pytest.raises(RuntimeError, match="unparseable"):
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
    _write_node(root, 5, gfx=90500, location=29952, caches=((2, 4096),))
    (root / "5" / "caches" / "1").mkdir()
    (root / "5" / "caches" / "1" / "properties").write_text("level 3\nsize not-a-number\n")

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

"""Tests for the provenance ledger + dataset fingerprint (root of trust).

These lock in the *rigor* properties, not just the happy path:
- canonical hashing is key-order independent and value-sensitive;
- the dataset fingerprint is order-independent but seed/label/case-sensitive;
- fingerprints use relative paths -> re-derivable under a different absolute root;
- timestamps never affect any fingerprint;
- the ledger is append-only and round-trips;
- digests are pixel-free (work from metadata with no file on disk).
"""
from __future__ import annotations

from brats2026.provenance import (
    LedgerEntry,
    append_entry,
    canonical_digest,
    case_fingerprint,
    dataset_fingerprint,
    fingerprint_from_ledger,
    make_entry,
    read_ledger,
    write_manifest,
)

LABEL_MAP = {"NCR": 1, "ED": 2, "ET": 3}


# --------------------------------------------------------------------------------------- #
# canonical_digest
# --------------------------------------------------------------------------------------- #
def test_canonical_digest_is_key_order_independent():
    a = canonical_digest({"a": 1, "b": 2, "c": {"x": 1, "y": 2}})
    b = canonical_digest({"c": {"y": 2, "x": 1}, "b": 2, "a": 1})
    assert a == b


def test_canonical_digest_changes_when_value_changes():
    a = canonical_digest({"a": 1, "b": 2})
    b = canonical_digest({"a": 1, "b": 3})
    assert a != b


# --------------------------------------------------------------------------------------- #
# case_fingerprint — pixel-free, metadata only
# --------------------------------------------------------------------------------------- #
def test_case_fingerprint_works_without_any_file_present():
    # No file on disk: digest is computed purely from metadata args -> pixel-free by design.
    fp = case_fingerprint("BraTS-GLI-1-0/t1n.nii.gz", 12345, header=None)
    assert isinstance(fp, str) and len(fp) == 64


def test_case_fingerprint_sensitive_to_size_and_header():
    base = case_fingerprint("c/t1n.nii.gz", 100, None)
    assert base != case_fingerprint("c/t1n.nii.gz", 101, None)
    assert base != case_fingerprint("c/t1n.nii.gz", 100, {"shape": [1, 2, 3]})


def test_case_fingerprint_normalises_relative_path():
    # A path with a redundant "./" segment normalises to the same POSIX form -> same hash,
    # so cosmetic path differences don't change the fingerprint.
    assert case_fingerprint("a/b/t1n.nii.gz", 5, None) == case_fingerprint("a/./b/t1n.nii.gz", 5, None)


# --------------------------------------------------------------------------------------- #
# dataset_fingerprint — order-independent, decision-sensitive
# --------------------------------------------------------------------------------------- #
def _fps():
    return [
        case_fingerprint("BraTS-GLI-1-0", 10, None),
        case_fingerprint("BraTS-SSA-2-0", 20, None),
        case_fingerprint("BraTS-PED-3-0", 30, None),
    ]


def test_dataset_fingerprint_is_order_independent():
    fps = _fps()
    extra = {"label_map": LABEL_MAP, "seed": 1234}
    a = dataset_fingerprint(fps, extra)
    b = dataset_fingerprint(list(reversed(fps)), extra)
    assert a == b


def test_dataset_fingerprint_changes_on_seed():
    fps = _fps()
    a = dataset_fingerprint(fps, {"label_map": LABEL_MAP, "seed": 1234})
    b = dataset_fingerprint(fps, {"label_map": LABEL_MAP, "seed": 9999})
    assert a != b


def test_dataset_fingerprint_changes_on_label_map():
    fps = _fps()
    a = dataset_fingerprint(fps, {"label_map": LABEL_MAP, "seed": 1})
    b = dataset_fingerprint(fps, {"label_map": {"NCR": 1, "ED": 2, "ET": 4}, "seed": 1})
    assert a != b


def test_dataset_fingerprint_changes_when_a_case_fingerprint_changes():
    fps = _fps()
    extra = {"label_map": LABEL_MAP, "seed": 1}
    a = dataset_fingerprint(fps, extra)
    changed = fps[:-1] + [case_fingerprint("BraTS-PED-3-0", 31, None)]
    assert a != dataset_fingerprint(changed, extra)


# --------------------------------------------------------------------------------------- #
# Container re-derivability: relative paths -> same hash under a different absolute root
# --------------------------------------------------------------------------------------- #
def test_same_relative_structure_under_different_roots_yields_same_fingerprint(tmp_path):
    extra = {"label_map": LABEL_MAP, "seed": 7}
    rel_cases = [("BraTS-GLI-1-0/t1n.nii.gz", 11), ("BraTS-SSA-2-0/t1n.nii.gz", 22)]

    def build(root_name: str) -> str:
        root = tmp_path / root_name  # different absolute roots, identical relative structure
        ledger = root / "provenance" / "ledger.jsonl"
        for case_id, (rel, size) in zip(("a", "b"), rel_cases):
            append_entry(make_entry(case_id, rel, size, LABEL_MAP), ledger)
        return fingerprint_from_ledger(read_ledger(ledger), extra)

    assert build("/abs/root/one".lstrip("/")) == build("a_totally_different_root")


# --------------------------------------------------------------------------------------- #
# Timestamps never affect any fingerprint
# --------------------------------------------------------------------------------------- #
def test_recorded_at_does_not_affect_case_fingerprint():
    e1 = LedgerEntry("c", "rel/t1n.nii.gz", 9, "fp", LABEL_MAP, recorded_at="2020-01-01T00:00:00+00:00")
    e2 = LedgerEntry("c", "rel/t1n.nii.gz", 9, "fp", LABEL_MAP, recorded_at="2099-12-31T23:59:59+00:00")
    # The fingerprint field is recomputed from metadata only; timestamps are not an input.
    assert case_fingerprint(e1.relative_path, e1.size, None) == case_fingerprint(
        e2.relative_path, e2.size, None
    )


def test_recorded_at_does_not_affect_dataset_fingerprint(tmp_path):
    extra = {"label_map": LABEL_MAP, "seed": 3}
    led_a = tmp_path / "a.jsonl"
    led_b = tmp_path / "b.jsonl"
    e_early = make_entry("c", "rel/t1n.nii.gz", 50, LABEL_MAP)
    e_late = LedgerEntry(
        e_early.case_id, e_early.relative_path, e_early.size, e_early.fingerprint,
        e_early.label_harmonization, recorded_at="2099-01-01T00:00:00+00:00",
    )
    append_entry(e_early, led_a)
    append_entry(e_late, led_b)
    assert fingerprint_from_ledger(read_ledger(led_a), extra) == fingerprint_from_ledger(
        read_ledger(led_b), extra
    )


# --------------------------------------------------------------------------------------- #
# Append-only ledger + round-trip
# --------------------------------------------------------------------------------------- #
def test_append_entry_preserves_existing_lines(tmp_path):
    ledger = tmp_path / "provenance" / "ledger.jsonl"
    e1 = make_entry("c1", "BraTS-GLI-1-0/t1n.nii.gz", 10, LABEL_MAP)
    e2 = make_entry("c2", "BraTS-SSA-2-0/t1n.nii.gz", 20, LABEL_MAP)
    append_entry(e1, ledger)
    first_pass = ledger.read_text()
    append_entry(e2, ledger)
    # The first line is untouched (append-only), and both lines are present.
    assert ledger.read_text().startswith(first_pass)
    assert ledger.read_text().count("\n") == 2


def test_read_ledger_round_trips(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    entries = [
        make_entry("c1", "BraTS-GLI-1-0/t1n.nii.gz", 10, LABEL_MAP),
        make_entry("c2", "BraTS-SSA-2-0/t1n.nii.gz", 20, LABEL_MAP),
    ]
    for e in entries:
        append_entry(e, ledger)
    got = read_ledger(ledger)
    assert [e.case_id for e in got] == ["c1", "c2"]
    assert [e.fingerprint for e in got] == [e.fingerprint for e in entries]


def test_read_ledger_missing_file_returns_empty(tmp_path):
    assert read_ledger(tmp_path / "nope.jsonl") == []


# --------------------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------------------- #
def test_write_manifest_records_fingerprint_and_optional_git_sha(tmp_path):
    import json

    out = write_manifest("deadbeef", {"num_cases": 2}, tmp_path / "manifest.json", git_sha="abc123")
    data = json.loads(out.read_text())
    assert data["dataset_fingerprint"] == "deadbeef"
    assert data["git_sha"] == "abc123"
    assert data["summary"] == {"num_cases": 2}


def test_write_manifest_git_sha_defaults_to_none(tmp_path):
    import json

    out = write_manifest("fp", {}, tmp_path / "m.json")
    assert json.loads(out.read_text())["git_sha"] is None

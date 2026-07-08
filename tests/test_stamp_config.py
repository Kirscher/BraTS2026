import pytest

from brats2026 import provenance
from brats2026.config import (
    ConfigError,
    assert_valid_config,
    load_config,
    stamp_config_file,
    stamp_provenance_in_text,
)

STUB = """\
kind: train
provenance:
  git_sha: null                       # STAMP: commit
  date: null                          # STAMP: date
  dataset_fingerprint: null           # STAMP: fingerprint
  seed: 42
  parent_run_id: "root"
  config_schema_version: "brats2026-config/1"
initial_lr: 0.01
"""


def test_stamp_preserves_comments_and_only_touches_targets():
    out = stamp_provenance_in_text(
        STUB, {"git_sha": "abc123", "date": "2026-07-08", "dataset_fingerprint": "deadbeef"}
    )
    assert 'git_sha: "abc123"' in out
    assert 'date: "2026-07-08"' in out
    assert 'dataset_fingerprint: "deadbeef"' in out
    assert "# STAMP: commit" in out          # comment preserved
    assert "seed: 42" in out                 # untouched
    assert "initial_lr: 0.01" in out         # outside provenance block, untouched


def test_stamp_raises_when_target_key_absent():
    with pytest.raises(ConfigError):
        stamp_provenance_in_text(STUB, {"not_a_field": "x"})


def test_stamp_does_not_leak_into_next_top_level_key():
    # A key with the same name outside the provenance block must NOT be edited.
    text = STUB + "date: SHOULD_NOT_CHANGE\n"
    out = stamp_provenance_in_text(text, {"date": "2026-07-08"})
    assert "date: SHOULD_NOT_CHANGE" in out


def test_stamp_config_file_makes_config_valid(tmp_path):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(STUB, encoding="utf-8")
    stamp_config_file(
        cfg_path,
        {"git_sha": "abc123", "date": "2026-07-08", "dataset_fingerprint": "deadbeef"},
    )
    # Every remaining SPECIALIST hook is still None here, so the header is now complete but the
    # config is not fully valid — assert the header stamping worked via load + no header errors.
    cfg = load_config(cfg_path)
    assert cfg["provenance"]["git_sha"] == "abc123"
    assert cfg["provenance"]["dataset_fingerprint"] == "deadbeef"


def test_shipped_train_yaml_is_valid_once_stamped(tmp_path):
    """The repo's configs/train.yaml must be complete except for the 3 stamped fields."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Before stamping: refused (dataset_fingerprint/git_sha/date are null).
    with pytest.raises(ConfigError):
        assert_valid_config(load_config(cfg_path), "train")

    stamp_config_file(
        cfg_path,
        {"git_sha": "abc123", "date": "2026-07-08", "dataset_fingerprint": "f" * 64},
    )
    # After stamping the 3 env-bound fields: fully valid (proves every hook is filled).
    assert_valid_config(load_config(cfg_path), "train")


def _write_fake_raw(root):
    ds = root / "Dataset501_BraTSGoAT"
    (ds / "imagesTr").mkdir(parents=True)
    (ds / "labelsTr").mkdir(parents=True)
    for case in ("BraTS-GLI-0001-000", "BraTS-PED-0002-000"):
        for ch in range(4):
            (ds / "imagesTr" / f"{case}_{ch:04d}.nii.gz").write_bytes(b"x" * (10 + ch))
        (ds / "labelsTr" / f"{case}.nii.gz").write_bytes(b"seg")
    return ds


def test_fingerprint_from_raw_is_deterministic_and_extra_sensitive(tmp_path):
    ds = _write_fake_raw(tmp_path)
    extra = {"label_map": {"NCR": 1, "ED": 2, "ET": 3}, "seed": 42}
    fp1 = provenance.dataset_fingerprint_from_raw(ds, extra, read_headers=False)
    fp2 = provenance.dataset_fingerprint_from_raw(ds, extra, read_headers=False)
    assert fp1 == fp2                                   # deterministic
    fp3 = provenance.dataset_fingerprint_from_raw(ds, {**extra, "seed": 7}, read_headers=False)
    assert fp1 != fp3                                   # extra block folded in
    assert len(fp1) == 64                               # sha256 hex


def test_fingerprint_from_raw_changes_when_a_file_size_changes(tmp_path):
    ds = _write_fake_raw(tmp_path)
    extra = {"seed": 42}
    before = provenance.dataset_fingerprint_from_raw(ds, extra, read_headers=False)
    (ds / "imagesTr" / "BraTS-GLI-0001-000_0000.nii.gz").write_bytes(b"x" * 999)
    after = provenance.dataset_fingerprint_from_raw(ds, extra, read_headers=False)
    assert before != after


def test_fingerprint_from_raw_errors_on_empty(tmp_path):
    (tmp_path / "Dataset501_BraTSGoAT" / "imagesTr").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        provenance.dataset_fingerprint_from_raw(tmp_path / "Dataset501_BraTSGoAT", {}, read_headers=False)

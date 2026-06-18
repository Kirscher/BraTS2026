import copy

import pytest

from brats2026.config import (
    CONFIG_SCHEMA_VERSION,
    INFER_HOOK_KEYS,
    PROVENANCE_FIELDS,
    TRAIN_HOOK_KEYS,
    ConfigError,
    assert_valid_config,
    check_fingerprint_binding,
    make_template,
    validate_config,
    validate_provenance_header,
)
from brats2026.nnunet.trainer import SPECIALIST_HOOKS


# --------------------------------------------------------------------------------------- #
# Helpers — fully-filled configs (no invented ML meaning; just non-None placeholders)
# --------------------------------------------------------------------------------------- #
def _filled_header() -> dict:
    return {
        "git_sha": "abc1234",
        "dataset_fingerprint": "deadbeef",
        "seed": 1234,
        "parent_run_id": "root",
        "date": "2026-06-16",
        "config_schema_version": CONFIG_SCHEMA_VERSION,
    }


def _filled_train() -> dict:
    cfg = make_template("train")
    cfg["provenance"] = _filled_header()
    for hook in TRAIN_HOOK_KEYS:
        cfg[hook] = 1  # any non-None value; validity is about being set, not the value
    return cfg


def _filled_infer() -> dict:
    cfg = make_template("infer")
    cfg["provenance"] = _filled_header()
    for hook in INFER_HOOK_KEYS:
        cfg[hook] = 1
    return cfg


# --------------------------------------------------------------------------------------- #
# Lockstep guard: config train hooks ARE the trainer's SPECIALIST_HOOKS
# --------------------------------------------------------------------------------------- #
def test_train_hook_keys_in_lockstep_with_specialist_hooks():
    # Regression guard against code/config drift: equality, verbatim.
    assert TRAIN_HOOK_KEYS == SPECIALIST_HOOKS


# --------------------------------------------------------------------------------------- #
# Templates are born unusable
# --------------------------------------------------------------------------------------- #
def test_train_template_has_all_hooks_none_and_full_empty_header():
    cfg = make_template("train")
    # every SPECIALIST hook present and None
    for hook in SPECIALIST_HOOKS:
        assert hook in cfg
        assert cfg[hook] is None
    # complete provenance sub-dict, all empty
    assert set(cfg["provenance"]) == set(PROVENANCE_FIELDS)
    assert all(v is None for v in cfg["provenance"].values())


def test_train_template_is_not_ok():
    cfg = make_template("train")
    result = validate_config(cfg, "train")
    assert result.ok is False
    # both failure axes flagged: empty header AND unset hooks
    assert set(result.missing_header) == set(PROVENANCE_FIELDS)
    assert set(result.unset_hooks) == set(TRAIN_HOOK_KEYS)


def test_infer_template_is_not_ok():
    cfg = make_template("infer")
    result = validate_config(cfg, "infer")
    assert result.ok is False
    assert set(result.unset_hooks) == set(INFER_HOOK_KEYS)


def test_make_template_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_template("bogus")


# --------------------------------------------------------------------------------------- #
# A fully-filled config validates; breaking one piece is flagged precisely
# --------------------------------------------------------------------------------------- #
def test_filled_train_config_validates_ok():
    cfg = _filled_train()
    result = validate_config(cfg, "train")
    assert result.ok is True
    assert result.missing_header == []
    assert result.unset_hooks == []
    assert_valid_config(cfg, "train")  # no raise


def test_removing_one_hook_flags_unset_and_raises():
    cfg = _filled_train()
    victim = TRAIN_HOOK_KEYS[0]
    cfg[victim] = None
    result = validate_config(cfg, "train")
    assert result.ok is False
    assert result.unset_hooks == [victim]
    with pytest.raises(ConfigError) as exc:
        assert_valid_config(cfg, "train")
    assert victim in str(exc.value)


def test_blanking_one_header_field_flags_missing_and_raises():
    cfg = _filled_train()
    cfg["provenance"]["git_sha"] = ""  # empty string counts as missing
    result = validate_config(cfg, "train")
    assert result.ok is False
    assert result.missing_header == ["git_sha"]
    with pytest.raises(ConfigError) as exc:
        assert_valid_config(cfg, "train")
    assert "git_sha" in str(exc.value)


def test_validate_provenance_header_on_filled_and_empty():
    assert validate_provenance_header(_filled_train()) == []
    assert set(validate_provenance_header(make_template("train"))) == set(PROVENANCE_FIELDS)


def test_filled_infer_config_validates_ok():
    cfg = _filled_infer()
    result = validate_config(cfg, "infer")
    assert result.ok is True
    assert_valid_config(cfg, "infer")


# --------------------------------------------------------------------------------------- #
# Fingerprint binding
# --------------------------------------------------------------------------------------- #
def test_fingerprint_binding_true_on_match_false_on_mismatch():
    cfg = _filled_train()
    assert check_fingerprint_binding(cfg, "deadbeef") is True
    assert check_fingerprint_binding(cfg, "feedface") is False


def test_fingerprint_binding_false_when_unset():
    cfg = make_template("train")
    assert check_fingerprint_binding(cfg, "deadbeef") is False


# --------------------------------------------------------------------------------------- #
# YAML round-trip + committed stubs (skipped if PyYAML absent)
# --------------------------------------------------------------------------------------- #
yaml = pytest.importorskip("yaml")  # noqa: F841


def test_dump_load_round_trip(tmp_path):
    from brats2026.config import dump_config, load_config

    cfg = _filled_train()
    path = tmp_path / "round.yaml"
    dump_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg


def test_committed_train_stub_loads_and_is_not_ok():
    import pathlib

    from brats2026.config import load_config

    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = load_config(root / "configs" / "train.yaml")
    assert validate_config(cfg, "train").ok is False
    with pytest.raises(ConfigError):
        assert_valid_config(cfg, "train")


def test_committed_infer_stub_loads_and_is_not_ok():
    import pathlib

    from brats2026.config import load_config

    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = load_config(root / "configs" / "infer.yaml")
    assert validate_config(cfg, "infer").ok is False

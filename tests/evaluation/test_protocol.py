import pytest

from brats2026.evaluation.protocol import (
    ProtocolError,
    read_sealed_config,
    reveal_outer_lodo,
    score_inner_dev,
    seal_config,
)
from brats2026.evaluation.report import CaseScore


def _case(case_id, cohort, dice=0.8, hd95=1.0):
    regions = {r: {"dice": dice, "hd95": hd95} for r in ("ET", "TC", "WT")}
    return CaseScore(case_id=case_id, cohort=cohort, legacy=regions)


def _cohort_scores():
    scores = []
    for cohort, n in (("GLI", 20), ("SSA", 10), ("MEN", 10), ("MET", 5), ("PED", 5)):
        scores += [_case(f"BraTS-{cohort}-{i:05d}-000", cohort) for i in range(n)]
    return scores


# --------------------------------------------------------------------------- #
# Inner dev — freely / repeatedly callable
# --------------------------------------------------------------------------- #
def test_inner_dev_callable_repeatedly_and_deterministic():
    scores = _cohort_scores()
    a = score_inner_dev(scores, k=5, seed=0)
    b = score_inner_dev(scores, k=5, seed=0)
    assert a == b  # deterministic, no side effects
    assert len(a["folds"]) == 5
    assert a["stage"] == "inner_dev"


def test_inner_dev_has_no_outer_information():
    scores = _cohort_scores()
    report = score_inner_dev(scores, k=5, seed=0)
    # nothing in the inner-dev report references the outer LODO stage
    assert "outer_lodo" not in repr(report)
    assert report["pooled_worst_cohort"]["region"] == "WT"


# --------------------------------------------------------------------------- #
# seal_config — write-once pre-registration
# --------------------------------------------------------------------------- #
def test_seal_config_writes_and_reads_back(tmp_path):
    seal_config("abc123", tmp_path)
    assert read_sealed_config(tmp_path) == "abc123"


def test_seal_config_refuses_overwrite(tmp_path):
    seal_config("abc123", tmp_path)
    with pytest.raises(ProtocolError, match="already sealed"):
        seal_config("def456", tmp_path)
    # original hash is intact
    assert read_sealed_config(tmp_path) == "abc123"


def test_seal_config_rejects_empty_hash(tmp_path):
    with pytest.raises(ProtocolError):
        seal_config("", tmp_path)


# --------------------------------------------------------------------------- #
# reveal_outer_lodo — the anti-leakage seal
# --------------------------------------------------------------------------- #
def _by_held_out():
    return {
        "GLI": [_case("BraTS-GLI-1-0", "GLI", 0.9, 1.0)],
        "MEN": [_case("BraTS-MEN-1-0", "MEN", 0.4, 3.0)],
    }


def test_reveal_succeeds_with_correct_hash(tmp_path):
    seal_config("frozen-hash", tmp_path)
    report = reveal_outer_lodo("frozen-hash", tmp_path, _by_held_out())
    assert report["stage"] == "outer_lodo"
    assert report["config_hash"] == "frozen-hash"
    assert [r["held_out"] for r in report["lodo"]["rows"]] == ["GLI", "MEN"]


def test_reveal_second_call_refuses_write_once(tmp_path):
    seal_config("frozen-hash", tmp_path)
    reveal_outer_lodo("frozen-hash", tmp_path, _by_held_out())
    with pytest.raises(ProtocolError, match="already revealed"):
        reveal_outer_lodo("frozen-hash", tmp_path, _by_held_out())


def test_reveal_wrong_hash_refuses(tmp_path):
    seal_config("frozen-hash", tmp_path)
    with pytest.raises(ProtocolError, match="mismatch"):
        reveal_outer_lodo("WRONG-hash", tmp_path, _by_held_out())


def test_reveal_wrong_hash_does_not_burn_the_one_shot(tmp_path):
    seal_config("frozen-hash", tmp_path)
    with pytest.raises(ProtocolError):
        reveal_outer_lodo("WRONG-hash", tmp_path, _by_held_out())
    # a failed guard must not consume the single legitimate reveal
    report = reveal_outer_lodo("frozen-hash", tmp_path, _by_held_out())
    assert report["stage"] == "outer_lodo"


def test_reveal_without_sealed_config_refuses(tmp_path):
    with pytest.raises(ProtocolError, match="no sealed config"):
        reveal_outer_lodo("any-hash", tmp_path, _by_held_out())

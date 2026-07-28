import pytest

from brats2026.domains import COHORTS, cohort_from_case_id
from brats2026.synthesis.generate import (
    assert_synthetic_compliant,
    generate_synthetic_case,
    synthetic_case_id,
)


# --- synthetic_case_id -------------------------------------------------------------------

def test_synthetic_case_id_round_trips_cohort():
    for cohort in COHORTS:
        cid = synthetic_case_id(cohort, 1)
        assert cohort_from_case_id(cid) == cohort


def test_synthetic_case_id_marks_synthetic_numeric_block():
    cid = synthetic_case_id("GLI", 42)
    # BraTS-GLI-9..... : the numeric block starts with the synthetic marker 9
    numeric = cid.rsplit("-", 1)[-1]
    assert numeric.startswith("9")
    assert cid.startswith("BraTS-GLI-")


def test_synthetic_case_id_is_unique_per_index():
    ids = {synthetic_case_id("MET", i) for i in range(50)}
    assert len(ids) == 50


# --- assert_synthetic_compliant ----------------------------------------------------------

def test_assert_synthetic_compliant_accepts_known_cohorts():
    ids = [synthetic_case_id(c, i) for i, c in enumerate(COHORTS)]
    assert_synthetic_compliant(ids)  # no raise


def test_assert_synthetic_compliant_rejects_foreign_id():
    ids = [synthetic_case_id("GLI", 0), "Foreign-Case-0001"]
    with pytest.raises(ValueError):
        assert_synthetic_compliant(ids)


# --- generate_synthetic_case scaffold ----------------------------------------------------

def test_generate_synthetic_case_is_scaffold():
    with pytest.raises(NotImplementedError):
        generate_synthetic_case(None, None, None, None, None)

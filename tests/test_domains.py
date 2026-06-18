from brats2026.domains import (
    COHORTS,
    UNKNOWN_COHORT,
    cohort_from_case_id,
    count_cohorts,
    domain_table,
)


def test_cohort_extraction_for_each_known_cohort():
    for cohort in COHORTS:
        case_id = f"BraTS-{cohort}-00123-000"
        assert cohort_from_case_id(case_id) == cohort


def test_cohort_extraction_is_case_insensitive():
    assert cohort_from_case_id("brats-gli-00001-000") == "GLI"


def test_unknown_prefix_returns_unk_not_error():
    assert cohort_from_case_id("BraTS-XYZ-00001-000") == UNKNOWN_COHORT
    assert cohort_from_case_id("") == UNKNOWN_COHORT


def test_domain_table_maps_every_case():
    ids = ["BraTS-GLI-1-0", "BraTS-PED-2-0"]
    assert domain_table(ids) == {"BraTS-GLI-1-0": "GLI", "BraTS-PED-2-0": "PED"}


def test_count_cohorts_keeps_order_and_tracks_unknown():
    ids = [
        "BraTS-GLI-1-0", "BraTS-GLI-2-0", "BraTS-MEN-1-0",
        "BraTS-PED-1-0", "BraTS-WAT-9-0",
    ]
    counts = count_cohorts(ids)
    assert list(counts.counts.keys()) == list(COHORTS)
    assert counts.counts["GLI"] == 2
    assert counts.counts["MEN"] == 1
    assert counts.counts["MET"] == 0
    assert counts.unknown == 1
    assert counts.total == 5

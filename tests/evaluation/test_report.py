import json
import math

from brats2026.evaluation.report import (
    CaseScore,
    hd95_summary,
    leave_one_domain_out_table,
    per_cohort_table,
    worst_cohort,
    write_report,
)


def _case(case_id, cohort, dice, hd95, family="legacy"):
    """Build a CaseScore with the same dice/hd95 across ET/TC/WT for the given family."""
    regions = {r: {"dice": dice, "hd95": hd95} for r in ("ET", "TC", "WT")}
    kwargs = {"case_id": case_id, "cohort": cohort, family: regions}
    return CaseScore(**kwargs)


# --------------------------------------------------------------------------- #
# CaseScore
# --------------------------------------------------------------------------- #
def test_case_score_infers_cohort_from_id():
    cs = CaseScore(case_id="BraTS-MEN-00012-000")
    assert cs.cohort == "MEN"


def test_case_score_carries_both_families():
    cs = CaseScore(
        case_id="x",
        cohort="GLI",
        legacy={"WT": {"dice": 0.9, "hd95": 2.0}},
        lesion={"WT": {"dice": 0.5, "hd95": 4.0}},
    )
    assert cs.family("legacy")["WT"]["dice"] == 0.9
    assert cs.family("lesion")["WT"]["dice"] == 0.5


# --------------------------------------------------------------------------- #
# per-cohort means
# --------------------------------------------------------------------------- #
def test_per_cohort_means_are_correct():
    scores = [
        _case("BraTS-GLI-1-0", "GLI", 0.8, 1.0),
        _case("BraTS-GLI-2-0", "GLI", 0.6, 3.0),
        _case("BraTS-MEN-1-0", "MEN", 1.0, 0.0),
    ]
    table = per_cohort_table(scores)
    gli = table["cohorts"]["GLI"]["regions"]["WT"]
    assert gli["dice"] == 0.7  # mean(0.8, 0.6)
    assert gli["hd95"]["mean"] == 2.0  # mean(1.0, 3.0)
    assert table["cohorts"]["GLI"]["n_cases"] == 2
    assert table["cohorts"]["MEN"]["regions"]["WT"]["dice"] == 1.0


def test_per_cohort_empty_cohort_is_graceful():
    scores = [_case("BraTS-GLI-1-0", "GLI", 0.8, 1.0)]
    table = per_cohort_table(scores)
    # GLI present, the other 4 cohorts are not -> not in the table (no cases).
    assert "GLI" in table["cohorts"]
    assert "SSA" not in table["cohorts"]


def test_per_cohort_ordering_is_cohorts_order():
    # deliberately out of order at input
    scores = [
        _case("BraTS-PED-1-0", "PED", 0.5, 1.0),
        _case("BraTS-GLI-1-0", "GLI", 0.5, 1.0),
        _case("BraTS-MEN-1-0", "MEN", 0.5, 1.0),
    ]
    table = per_cohort_table(scores)
    assert list(table["cohorts"]) == ["GLI", "MEN", "PED"]


# --------------------------------------------------------------------------- #
# worst cohort
# --------------------------------------------------------------------------- #
def test_worst_cohort_identified():
    scores = [
        _case("BraTS-GLI-1-0", "GLI", 0.9, 1.0),
        _case("BraTS-MEN-1-0", "MEN", 0.4, 1.0),  # worst
        _case("BraTS-SSA-1-0", "SSA", 0.7, 1.0),
    ]
    worst = worst_cohort(scores, region="WT")
    assert worst["cohort"] == "MEN"
    assert worst["dice"] == 0.4


# --------------------------------------------------------------------------- #
# inf-aware HD95 aggregation
# --------------------------------------------------------------------------- #
def test_hd95_summary_excludes_inf_from_mean():
    summary = hd95_summary([2.0, 4.0, float("inf")])
    assert summary["mean"] == 3.0  # mean of finite only
    assert summary["n"] == 3
    assert summary["n_miss"] == 1
    assert summary["miss_rate"] == 1 / 3


def test_hd95_summary_all_miss_has_none_mean():
    summary = hd95_summary([float("inf"), float("inf")])
    assert summary["mean"] is None
    assert summary["n_miss"] == 2
    assert summary["miss_rate"] == 1.0


def test_miss_does_not_poison_per_cohort_finite_mean():
    scores = [
        _case("BraTS-GLI-1-0", "GLI", 0.8, 2.0),
        _case("BraTS-GLI-2-0", "GLI", 0.0, float("inf")),  # a total miss
    ]
    wt = per_cohort_table(scores)["cohorts"]["GLI"]["regions"]["WT"]["hd95"]
    assert math.isfinite(wt["mean"])
    assert wt["mean"] == 2.0  # the inf is excluded from the finite mean
    assert wt["n_miss"] == 1
    assert wt["miss_rate"] == 0.5


# --------------------------------------------------------------------------- #
# leave-one-domain-out table
# --------------------------------------------------------------------------- #
def test_lodo_table_one_row_per_held_out():
    by_held = {
        "GLI": [_case("BraTS-GLI-1-0", "GLI", 0.9, 1.0)],
        "MEN": [_case("BraTS-MEN-1-0", "MEN", 0.5, 2.0)],
    }
    table = leave_one_domain_out_table(by_held)
    assert [row["held_out"] for row in table["rows"]] == ["GLI", "MEN"]
    men = next(r for r in table["rows"] if r["held_out"] == "MEN")
    assert men["n_cases"] == 1
    assert men["regions"]["WT"]["dice"] == 0.5


def test_lodo_rows_in_cohorts_order():
    by_held = {
        "PED": [_case("BraTS-PED-1-0", "PED", 0.5, 1.0)],
        "GLI": [_case("BraTS-GLI-1-0", "GLI", 0.5, 1.0)],
    }
    table = leave_one_domain_out_table(by_held)
    assert [row["held_out"] for row in table["rows"]] == ["GLI", "PED"]


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def test_write_report_is_deterministic_json(tmp_path):
    scores = [_case("BraTS-GLI-1-0", "GLI", 0.8, 1.0)]
    table = per_cohort_table(scores)
    path = write_report(table, tmp_path / "reports" / "cohort.json")
    assert path.exists()
    reloaded = json.loads(path.read_text())
    assert reloaded == table
    # sorted keys -> stable across writes
    assert path.read_text() == write_report(table, tmp_path / "again.json").read_text()

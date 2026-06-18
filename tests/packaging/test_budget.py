import pytest

from brats2026.packaging.budget import (
    BUDGET_HOURS,
    BudgetError,
    assert_within_budget,
    estimate_total_seconds,
    fits_budget,
    headroom_seconds,
)

EIGHT_HOURS = BUDGET_HOURS * 3600


def test_budget_hours_is_eight():
    assert BUDGET_HOURS == 8


def test_estimate_total_seconds():
    assert estimate_total_seconds(60, 100) == 6000


def test_config_that_fits_passes_and_reports_headroom():
    # 100 cases * 60s = 6000s, well under 28800s.
    proj = assert_within_budget(per_case_seconds=60, n_cases=100)
    assert proj.fits
    assert proj.total_seconds == 6000
    assert proj.headroom_seconds == EIGHT_HOURS - 6000
    assert fits_budget(6000)
    assert headroom_seconds(6000) == EIGHT_HOURS - 6000


def test_config_that_overruns_raises_with_overage():
    # 1000 cases * 60s = 60000s > 28800s, overage 31200s.
    with pytest.raises(BudgetError, match="31200s"):
        assert_within_budget(per_case_seconds=60, n_cases=1000)
    assert not fits_budget(60000)
    assert headroom_seconds(60000) == EIGHT_HOURS - 60000  # negative


def test_boundary_exactly_eight_hours_fits():
    # Exactly on budget counts as fitting (<=).
    assert fits_budget(EIGHT_HOURS)
    proj = assert_within_budget(per_case_seconds=EIGHT_HOURS, n_cases=1)
    assert proj.fits
    assert proj.headroom_seconds == 0


def test_just_over_boundary_raises():
    with pytest.raises(BudgetError):
        assert_within_budget(per_case_seconds=EIGHT_HOURS + 1, n_cases=1)

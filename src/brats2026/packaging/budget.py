"""Phase 6 — the hard 8-hour inference time-budget guard.

The GoAT container must finish inference over the whole hidden test set within **8 hours**.
Unlike thresholds/TTA, this is a fixed CHALLENGE RULE, not a ``# SPECIALIST:`` knob — so
``BUDGET_HOURS = 8`` is a constant here, not a config value. The per-case timing comes from
the inference-packager's profiling run.

Boundary convention: a projection that lands *exactly* on the budget is considered to FIT
(``<=``), so an estimate of exactly 8h passes with zero headroom.
"""
from __future__ import annotations

from dataclasses import dataclass

BUDGET_HOURS: int = 8
SECONDS_PER_HOUR: int = 3600


class BudgetError(RuntimeError):
    """Raised when projected inference time exceeds the fixed 8-hour budget."""


@dataclass(frozen=True)
class BudgetProjection:
    """Result of a budget check: the projected total and the (signed) headroom."""

    total_seconds: float
    budget_seconds: float

    @property
    def headroom_seconds(self) -> float:
        return self.budget_seconds - self.total_seconds

    @property
    def fits(self) -> bool:
        return self.total_seconds <= self.budget_seconds


def estimate_total_seconds(per_case_seconds: float, n_cases: int) -> float:
    """Projected wall-clock seconds for ``n_cases`` at ``per_case_seconds`` each."""
    return float(per_case_seconds) * int(n_cases)


def fits_budget(total_seconds: float, budget_hours: int = BUDGET_HOURS) -> bool:
    """True iff ``total_seconds`` is within the budget (exactly-on-budget counts as fitting)."""
    return float(total_seconds) <= budget_hours * SECONDS_PER_HOUR


def headroom_seconds(total_seconds: float, budget_hours: int = BUDGET_HOURS) -> float:
    """Signed spare seconds: positive when under budget, negative when over."""
    return budget_hours * SECONDS_PER_HOUR - float(total_seconds)


def assert_within_budget(
    per_case_seconds: float,
    n_cases: int,
    budget_hours: int = BUDGET_HOURS,
) -> BudgetProjection:
    """Raise :class:`BudgetError` (with projection + overage) if it won't fit; else return it."""
    total = estimate_total_seconds(per_case_seconds, n_cases)
    budget = budget_hours * SECONDS_PER_HOUR
    projection = BudgetProjection(total_seconds=total, budget_seconds=budget)
    if not projection.fits:
        overage = total - budget
        raise BudgetError(
            f"projected inference {total:.0f}s ({total / SECONDS_PER_HOUR:.2f}h) over "
            f"{n_cases} cases exceeds the {budget_hours}h budget ({budget:.0f}s) by "
            f"{overage:.0f}s ({overage / SECONDS_PER_HOUR:.2f}h)"
        )
    return projection

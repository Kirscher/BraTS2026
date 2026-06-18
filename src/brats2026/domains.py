"""Phase 0/1 — cohort (domain) handling for BraTS-GoAT.

The cohort is encoded for free in the case-ID prefix ``BraTS-<COHORT>-...``. We use it for
domain-balanced sampling and leave-one-domain-out validation. This module is pure stdlib so
it stays importable and testable without any imaging dependency.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# The five GoAT cohorts. Order is stable so it can key deterministic splits/reports.
COHORTS: tuple[str, ...] = ("GLI", "SSA", "MEN", "MET", "PED")

UNKNOWN_COHORT = "UNK"


def cohort_from_case_id(case_id: str) -> str:
    """Extract the cohort code from a ``BraTS-<COHORT>-<id>-<tp>`` style case ID.

    Matching is case-insensitive and tolerant of an optional ``BraTS-`` prefix. Returns
    :data:`UNKNOWN_COHORT` when no known cohort token is present, rather than raising — the
    caller (e.g. discovery) decides whether an unknown cohort is fatal.
    """
    if not case_id:
        return UNKNOWN_COHORT
    tokens = case_id.upper().replace("_", "-").split("-")
    for token in tokens:
        if token in COHORTS:
            return token
    return UNKNOWN_COHORT


def domain_table(case_ids: list[str]) -> dict[str, str]:
    """Map every case ID to its cohort code."""
    return {case_id: cohort_from_case_id(case_id) for case_id in case_ids}


@dataclass(frozen=True)
class CohortCounts:
    """Per-cohort case counts plus the count of unresolved IDs."""

    counts: dict[str, int]
    unknown: int

    @property
    def total(self) -> int:
        return sum(self.counts.values()) + self.unknown


def count_cohorts(case_ids: list[str]) -> CohortCounts:
    """Count cases per cohort, keeping :data:`COHORTS` order and tracking unknowns."""
    raw = Counter(cohort_from_case_id(cid) for cid in case_ids)
    counts = {cohort: raw.get(cohort, 0) for cohort in COHORTS}
    return CohortCounts(counts=counts, unknown=raw.get(UNKNOWN_COHORT, 0))

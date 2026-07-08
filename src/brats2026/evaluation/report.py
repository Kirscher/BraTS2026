"""Phase 3 — aggregation layer the evaluator owns.

Turns a bag of per-case region scores into the GoAT-relevant tables:

- **per-cohort** mean Dice + NSD + (inf-aware) HD95 for ET/TC/WT (in :data:`domains.COHORTS`
  order). NSD is the surface metric the leaderboard rank-aggregates with Dice; it is bounded
  ``[0, 1]`` with no ``inf`` (a total miss is 0.0), so it averages as a plain mean;
- the **worst cohort** — the GoAT generalisation signal, kept as a first-class result rather
  than buried inside a pooled mean;
- a **leave-one-domain-out** table (one row per held-out cohort).

Two scoring families are carried side by side per case (*legacy-overlap* and *lesion-wise*),
because which one the portal ranks on is unconfirmed — we never silently pick one.

**HD95 inf-aware aggregation.** A total miss yields ``inf`` (see :mod:`metrics`). Averaging
that in would nuke a whole cohort's HD95 to ``inf`` and hide the rest of the distribution. So
HD95 is summarised as the **mean over finite values** alongside an explicit **miss count /
miss-rate** — the misses are reported, not allowed to poison the finite mean.

Pure stdlib + numpy (no torch/nnU-Net). Output is deterministic and JSON-serialisable so it
can be written under ``work/reports/``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..domains import cohort_from_case_id, COHORTS

# The two scoring families carried per case. "legacy" = global overlap; "lesion" = lesion-wise.
SCORING_FAMILIES: tuple[str, ...] = ("legacy", "lesion")
REGION_NAMES: tuple[str, ...] = ("ET", "TC", "WT")


# --------------------------------------------------------------------------------------- #
# Per-case score record
# --------------------------------------------------------------------------------------- #
@dataclass
class CaseScore:
    """All scores for one case, both scoring families side by side.

    ``legacy`` and ``lesion`` map region name -> ``{"dice": float, "hd95": float, "nsd": float}``
    (``nsd`` optional for back-compat). ``cohort`` defaults to the cohort decoded from ``case_id``
    when not given explicitly.
    """

    case_id: str
    cohort: str = ""
    legacy: dict[str, dict[str, float]] = field(default_factory=dict)
    lesion: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cohort:
            self.cohort = cohort_from_case_id(self.case_id)

    def family(self, family: str) -> dict[str, dict[str, float]]:
        if family not in SCORING_FAMILIES:
            raise ValueError(f"unknown scoring family {family!r}; expected {SCORING_FAMILIES}")
        return getattr(self, family)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "cohort": self.cohort,
            "legacy": self.legacy,
            "lesion": self.lesion,
        }


# --------------------------------------------------------------------------------------- #
# inf-aware summary helpers
# --------------------------------------------------------------------------------------- #
def _mean_or_none(values: list[float]) -> float | None:
    """Plain mean of ``values``; ``None`` when empty (so an empty cohort doesn't crash)."""
    return float(np.mean(values)) if values else None


def hd95_summary(values: list[float]) -> dict:
    """inf-aware HD95 summary: finite mean + miss count/rate.

    ``inf`` entries are total misses (one mask empty). They are counted but excluded from the
    ``mean`` so a single miss cannot drive the whole summary to ``inf``. ``mean`` is ``None``
    when *every* value is a miss (no finite distance exists).
    """
    finite = [v for v in values if math.isfinite(v)]
    n_total = len(values)
    n_miss = sum(1 for v in values if math.isinf(v))
    return {
        "mean": _mean_or_none(finite),
        "n": n_total,
        "n_miss": n_miss,
        "miss_rate": (n_miss / n_total) if n_total else None,
    }


# --------------------------------------------------------------------------------------- #
# Per-cohort table
# --------------------------------------------------------------------------------------- #
def _region_summary(scores: list[CaseScore], cohort: str, family: str, region: str) -> dict:
    fam = [s.family(family) for s in scores if s.cohort == cohort]
    dices = [r[region]["dice"] for r in fam if region in r]
    hd95s = [r[region]["hd95"] for r in fam if region in r]
    nsds = [r[region]["nsd"] for r in fam if region in r and "nsd" in r[region]]
    return {"dice": _mean_or_none(dices), "nsd": _mean_or_none(nsds), "hd95": hd95_summary(hd95s)}


def per_cohort_table(scores: list[CaseScore], family: str = "legacy") -> dict:
    """Per-cohort mean Dice + (inf-aware) HD95 for ET/TC/WT, in :data:`COHORTS` order.

    Cohorts with no cases are emitted with ``n_cases == 0`` and ``None`` summaries rather than
    dropped, so the table shape is stable regardless of which cohorts are present.
    """
    if family not in SCORING_FAMILIES:
        raise ValueError(f"unknown scoring family {family!r}; expected {SCORING_FAMILIES}")
    present = {s.cohort for s in scores}
    # Keep COHORTS order, then append any stray cohort (e.g. UNK) deterministically.
    ordered = [c for c in COHORTS if c in present] + sorted(present - set(COHORTS))
    cohorts: dict[str, dict] = {}
    for cohort in ordered:
        n_cases = sum(1 for s in scores if s.cohort == cohort)
        cohorts[cohort] = {
            "n_cases": n_cases,
            "regions": {
                region: _region_summary(scores, cohort, family, region)
                for region in REGION_NAMES
            },
        }
    return {"family": family, "cohorts": cohorts}


def worst_cohort(scores: list[CaseScore], family: str = "legacy", region: str = "WT") -> dict:
    """First-class worst-cohort result: the lowest mean Dice across cohorts (the GoAT signal).

    Returns ``{"cohort", "dice", "region", "family"}``. Cohorts with no cases or no scorable
    Dice for ``region`` are skipped. ``cohort`` is ``None`` when nothing is scorable.
    """
    table = per_cohort_table(scores, family)
    worst_name: str | None = None
    worst_dice: float | None = None
    for cohort, body in table["cohorts"].items():
        dice = body["regions"][region]["dice"]
        if dice is None:
            continue
        if worst_dice is None or dice < worst_dice:
            worst_name, worst_dice = cohort, dice
    return {"cohort": worst_name, "dice": worst_dice, "region": region, "family": family}


# --------------------------------------------------------------------------------------- #
# Leave-one-domain-out table
# --------------------------------------------------------------------------------------- #
def leave_one_domain_out_table(
    scores_by_held_out: dict[str, list[CaseScore]], family: str = "legacy"
) -> dict:
    """One row per held-out cohort.

    ``scores_by_held_out`` maps the held-out cohort name -> the CaseScores produced when that
    cohort was the validation domain (typically scored on that cohort's own cases). Rows are
    emitted in :data:`COHORTS` order for the held-out cohorts that are present.
    """
    present = [c for c in COHORTS if c in scores_by_held_out]
    present += sorted(set(scores_by_held_out) - set(COHORTS))
    rows = []
    for held in present:
        held_scores = scores_by_held_out[held]
        rows.append(
            {
                "held_out": held,
                "n_cases": len(held_scores),
                "regions": {
                    region: {
                        "dice": _mean_or_none(
                            [
                                s.family(family)[region]["dice"]
                                for s in held_scores
                                if region in s.family(family)
                            ]
                        ),
                        "nsd": _mean_or_none(
                            [
                                s.family(family)[region]["nsd"]
                                for s in held_scores
                                if region in s.family(family) and "nsd" in s.family(family)[region]
                            ]
                        ),
                        "hd95": hd95_summary(
                            [
                                s.family(family)[region]["hd95"]
                                for s in held_scores
                                if region in s.family(family)
                            ]
                        ),
                    }
                    for region in REGION_NAMES
                },
            }
        )
    return {"family": family, "rows": rows}


# --------------------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------------------- #
def write_report(table: dict, path: Path) -> Path:
    """Write a report table as deterministic, pretty JSON (sorted keys) under ``work/``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

"""Phase 1/2 — domain-grouped validation splits for BraTS-GoAT.

Two split products, both keyed off the cohort encoded in the case ID:

1. **Domain-balanced K-fold** — each fold holds a proportional slice of every cohort, so no
   fold is dominated by the majority GLI cohort. Written in nnU-Net ``splits_final.json``
   shape (a list of ``{"train": [...], "val": [...]}``).
2. **Leave-one-domain-out** — one entry per cohort: validate on that cohort, train on the
   rest. This measures out-of-domain generalisation directly (Phase 2/3).

Deterministic given ``seed``; pure stdlib (no numpy needed).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..domains import COHORTS, cohort_from_case_id

Fold = dict[str, list[str]]


def _group_by_cohort(case_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for cid in case_ids:
        groups.setdefault(cohort_from_case_id(cid), []).append(cid)
    return groups


def domain_balanced_kfold(case_ids: list[str], k: int = 5, seed: int = 42) -> list[Fold]:
    """Build ``k`` folds, each with a proportional share of every cohort.

    Within each cohort the cases are shuffled (seeded) then dealt round-robin across folds,
    so cohort proportions are preserved per fold and the assignment is reproducible.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    rng = random.Random(seed)
    val_buckets: list[list[str]] = [[] for _ in range(k)]
    for cohort in sorted(_group_by_cohort(case_ids)):
        members = sorted(_group_by_cohort(case_ids)[cohort])
        rng.shuffle(members)
        for i, cid in enumerate(members):
            val_buckets[i % k].append(cid)

    all_ids = set(case_ids)
    folds: list[Fold] = []
    for val in val_buckets:
        val_set = set(val)
        train = sorted(all_ids - val_set)
        folds.append({"train": train, "val": sorted(val)})
    return folds


def leave_one_domain_out(case_ids: list[str]) -> list[Fold]:
    """One fold per present cohort: validate on it, train on all others.

    Cohorts are emitted in :data:`COHORTS` order (only those actually present). Each fold
    carries a ``held_out`` key naming the validation cohort.
    """
    groups = _group_by_cohort(case_ids)
    present = [c for c in COHORTS if groups.get(c)]
    all_ids = set(case_ids)
    folds: list[Fold] = []
    for cohort in present:
        val = sorted(groups[cohort])
        train = sorted(all_ids - set(val))
        folds.append({"held_out": cohort, "train": train, "val": val})
    return folds


def write_splits(folds: list[Fold], output: Path) -> Path:
    """Write folds as JSON (nnU-Net ``splits_final.json`` shape for the K-fold case)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(folds, indent=2, sort_keys=False) + "\n")
    return output

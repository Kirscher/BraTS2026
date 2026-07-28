"""Post-hoc operating-point search for lesion detection (the ET-recall lever).

**Why this exists.** The two scored submissions differ by ~0.003 global Dice, which is empirical
confirmation that hyperparameter tuning has saturated. But their lesion-level breakdown shows the
real problem is untouched: across 453 cases and 1716 ground-truth ET lesions we detect ~570 and
miss ~1150 — **lesion recall 0.33 at precision 0.52-0.57**. The newer submission bought precision
(false positives 1.17 -> 0.96 per case) and gained nothing, because false negatives outnumber
false positives more than two to one. The binding constraint is *recall*, and the last change
pushed the wrong way.

That is a decision-threshold problem, not a model problem, and it is fixable without retraining:
the same probability maps re-thresholded at a different operating point yield a different
recall/precision trade. This module searches that trade-off directly.

Two knobs, both already SPECIALIST hooks in ``configs/infer.yaml``:

- the **probability threshold** — lowering it recovers faint lesions (raises recall);
- the **minimum component size** — the small-component filter that suppresses false positives.
  Note it also deletes true small lesions, which is very likely where the missing ET recall went.

Everything here is pure numpy on plain arrays, so the search is unit-testable with no GPU, no
nnU-Net and no torch. I/O (loading nnU-Net's saved softmax) lives in the CLI, not here.

**Caveat on lesion matching.** The challenge's exact instance-matching rule is not published in
the material we hold; this module counts a ground-truth lesion as detected when *any* predicted
component overlaps it, mirroring :func:`brats2026.evaluation.metrics.lesion_wise_dice`. Absolute
counts may therefore differ slightly from the leaderboard's, but the *ranking of operating
points* — which is what this is for — is robust to that choice. Confirm before quoting counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import connected_components

# Objective names accepted by `best_operating_point`. "f2" weights recall twice as heavily as
# precision, which is the setting this dataset's FN:FP ratio (~2.6:1) actually calls for.
OBJECTIVES: tuple[str, ...] = ("f1", "f2", "recall", "precision")


@dataclass(frozen=True)
class LesionCounts:
    """Instance-level confusion for one region of one case (or summed over a cohort)."""

    tp: int
    fp: int
    fn: int

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    def fbeta(self, beta: float = 1.0) -> float:
        """F-beta. ``beta > 1`` weights recall higher — the right choice when FN dominates."""
        p, r = self.precision, self.recall
        if p == 0 and r == 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)

    def __add__(self, other: "LesionCounts") -> "LesionCounts":
        return LesionCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def lesion_confusion(pred_mask, gt_mask) -> LesionCounts:
    """Count detected / spurious / missed lesions between two boolean masks.

    A ground-truth component overlapped by at least one predicted voxel is a TP; one with no
    overlap is a FN; a predicted component touching no ground-truth component is a FP. Multiple
    predicted fragments landing on a single ground-truth lesion count as one TP and no FPs —
    fragmentation is a boundary-quality problem that Dice and NSD already penalise, and counting
    it here as well would double-charge it and bias the search toward over-suppression, which is
    the exact failure we are trying to undo.
    """
    gt_labels, n_gt = connected_components(np.asarray(gt_mask, dtype=bool))
    pred_labels, n_pred = connected_components(np.asarray(pred_mask, dtype=bool))

    matched_pred: set[int] = set()
    tp = 0
    for gi in range(1, n_gt + 1):
        overlap = pred_labels[gt_labels == gi]
        hits = {int(x) for x in np.unique(overlap) if x > 0}
        if hits:
            tp += 1
            matched_pred |= hits
    return LesionCounts(tp=tp, fp=n_pred - len(matched_pred), fn=n_gt - tp)


def apply_operating_point(prob, threshold: float, min_voxels: int):
    """Binarise a probability map, then drop connected components below ``min_voxels``.

    This is the decision rule the sweep varies. ``min_voxels <= 1`` disables the size filter
    entirely, which is a meaningful setting to include: it is the only way to see how much recall
    the filter is currently costing.
    """
    mask = np.asarray(prob) >= threshold
    if min_voxels <= 1 or not mask.any():
        return mask
    labels, n = connected_components(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n + 1):
        component = labels == i
        if int(component.sum()) >= min_voxels:
            keep |= component
    return keep


@dataclass(frozen=True)
class SweepResult:
    """Aggregate performance of one (threshold, min_voxels) operating point over all cases."""

    threshold: float
    min_voxels: int
    counts: LesionCounts
    mean_dice: float

    @property
    def recall(self) -> float:
        return self.counts.recall

    @property
    def precision(self) -> float:
        return self.counts.precision

    def objective(self, name: str) -> float:
        if name == "f1":
            return self.counts.fbeta(1.0)
        if name == "f2":
            return self.counts.fbeta(2.0)
        if name == "recall":
            return self.counts.recall
        if name == "precision":
            return self.counts.precision
        raise ValueError(f"unknown objective {name!r}; expected one of {OBJECTIVES}")


def _dice(pred_mask, gt_mask) -> float:
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_mask, gt_mask).sum() / total)


def sweep_operating_points(cases, thresholds, min_voxels_grid) -> list[SweepResult]:
    """Evaluate every (threshold, min_voxels) combination over ``cases``.

    ``cases`` is an iterable of ``(probability_array, ground_truth_mask)`` pairs for ONE region —
    the caller selects the channel, so this stays agnostic to nnU-Net's region ordering. It is
    materialised once because the grid re-reads it for every operating point.

    Counts are summed across cases (micro-averaged) rather than averaged per case: a case with one
    lesion should not carry the same weight as a case with twenty when the question is "how many
    lesions do we find in total". Dice is macro-averaged, matching how the leaderboard reports it.
    """
    cases = list(cases)
    results: list[SweepResult] = []
    for threshold in thresholds:
        for min_voxels in min_voxels_grid:
            total = LesionCounts(0, 0, 0)
            dices: list[float] = []
            for prob, gt in cases:
                pred = apply_operating_point(prob, threshold, min_voxels)
                total = total + lesion_confusion(pred, gt)
                dices.append(_dice(pred, gt))
            results.append(
                SweepResult(
                    threshold=float(threshold),
                    min_voxels=int(min_voxels),
                    counts=total,
                    mean_dice=float(np.mean(dices)) if dices else 1.0,
                )
            )
    return results


def best_operating_point(results: list[SweepResult], objective: str = "f1") -> SweepResult:
    """The operating point maximising ``objective``.

    Ties break toward the HIGHER threshold and LARGER min_voxels, i.e. toward the more
    conservative rule — if two settings score the same, prefer the one that predicts less, since
    the extra predictions bought nothing.
    """
    if not results:
        raise ValueError("no sweep results to choose from")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    return max(results, key=lambda r: (r.objective(objective), r.threshold, r.min_voxels))


def summarise(results: list[SweepResult], baseline: SweepResult | None = None) -> dict:
    """JSON-ready summary: the best point per objective, plus the full grid.

    ``baseline`` is the currently-shipped operating point; when given, each recommendation
    carries its delta against it, so the report answers "is this worth changing" rather than
    just "what scored best".
    """
    out: dict = {
        "n_operating_points": len(results),
        "best": {},
        "grid": [
            {
                "threshold": r.threshold,
                "min_voxels": r.min_voxels,
                "tp": r.counts.tp,
                "fp": r.counts.fp,
                "fn": r.counts.fn,
                "recall": r.recall,
                "precision": r.precision,
                "f1": r.objective("f1"),
                "f2": r.objective("f2"),
                "mean_dice": r.mean_dice,
            }
            for r in results
        ],
    }
    for name in OBJECTIVES:
        best = best_operating_point(results, name)
        entry = {
            "threshold": best.threshold,
            "min_voxels": best.min_voxels,
            "value": best.objective(name),
            "recall": best.recall,
            "precision": best.precision,
            "mean_dice": best.mean_dice,
        }
        if baseline is not None:
            entry["delta_vs_baseline"] = {
                "value": best.objective(name) - baseline.objective(name),
                "recall": best.recall - baseline.recall,
                "precision": best.precision - baseline.precision,
                "mean_dice": best.mean_dice - baseline.mean_dice,
            }
        out["best"][name] = entry
    if baseline is not None:
        out["baseline"] = {
            "threshold": baseline.threshold,
            "min_voxels": baseline.min_voxels,
            "recall": baseline.recall,
            "precision": baseline.precision,
            "f1": baseline.objective("f1"),
            "mean_dice": baseline.mean_dice,
        }
    return out

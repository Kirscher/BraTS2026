"""Phase 3 — score a directory of predicted segmentations into the GoAT region tables.

Bridges the raw metrics (:mod:`brats2026.evaluation.metrics`) and the aggregation layer
(:mod:`brats2026.evaluation.report`): match predicted vs ground-truth NIfTI label maps by case
ID, compute per-region Dice/HD95/NSD (both scoring families), and fold them into the
per-cohort + worst-cohort report the evaluator owns. This is what ``brats2026 evaluate`` runs on
an nnU-Net fold's ``validation/`` output to produce the first GoAT result.

Surface metrics (HD95, NSD) are region-level, so they are shared by both scoring families; only
Dice differs (global overlap for ``legacy`` vs lesion-wise for ``lesion``). numpy is required;
nibabel is needed only for the file-loading entry points (behind a guarded import).

Reads label masks (our own / nnU-Net's derived segmentations under ``work/``), never the
controlled NAS image pixels.
"""
from __future__ import annotations

from pathlib import Path

from .metrics import REGIONS, lesion_wise_dice, region_scores
from .report import (
    CaseScore,
    per_cohort_table,
    worst_cohort,
    write_report,
)


def score_case_arrays(
    case_id: str,
    pred_arr,
    gt_arr,
    spacing=(1.0, 1.0, 1.0),
    nsd_tolerances_mm: dict[str, float] | None = None,
) -> CaseScore:
    """Score one case from in-memory integer label arrays → a :class:`CaseScore`.

    ``legacy`` carries global-overlap Dice; ``lesion`` carries lesion-wise Dice. Both families
    share the region-level HD95 and NSD (surface metrics are not lesion-decomposed here).
    """
    legacy = region_scores(pred_arr, gt_arr, spacing, nsd_tolerances_mm)
    lesion = {
        name: {
            "dice": lesion_wise_dice(pred_arr, gt_arr, REGIONS[name]),
            "hd95": legacy[name]["hd95"],
            "nsd": legacy[name]["nsd"],
        }
        for name in REGIONS
    }
    return CaseScore(case_id=case_id, legacy=legacy, lesion=lesion)


def _load_label(path: Path):
    """Load an integer label NIfTI as an int array + its voxel spacing (mm). nibabel-guarded."""
    try:
        import nibabel as nib  # noqa: PLC0415 - optional, guarded
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without nibabel
        raise RuntimeError("nibabel + numpy are required to load segmentations for scoring") from exc

    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)  # label map — reading our derived seg, not NAS pixels
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return np.rint(arr).astype(int), spacing


def score_directory(
    pred_dir: Path,
    gt_dir: Path,
    nsd_tolerances_mm: dict[str, float] | None = None,
) -> list[CaseScore]:
    """Score every case present in BOTH ``pred_dir`` and ``gt_dir`` (matched by ``<case>.nii.gz``).

    Returns a list of :class:`CaseScore`. Cases missing from either side are skipped (they are
    not silently scored as misses — the caller sees the count via the returned length). Spacing
    is read from the ground-truth header so anisotropic voxels are honoured in mm.
    """
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    pred_ids = {p.name[: -len(".nii.gz")] for p in pred_dir.glob("*.nii.gz")}
    gt_ids = {p.name[: -len(".nii.gz")] for p in gt_dir.glob("*.nii.gz")}
    common = sorted(pred_ids & gt_ids)

    scores: list[CaseScore] = []
    for case_id in common:
        gt_arr, spacing = _load_label(gt_dir / f"{case_id}.nii.gz")
        pred_arr, _ = _load_label(pred_dir / f"{case_id}.nii.gz")
        scores.append(score_case_arrays(case_id, pred_arr, gt_arr, spacing, nsd_tolerances_mm))
    return scores


def build_report(scores: list[CaseScore]) -> dict:
    """Assemble the GoAT report: per-cohort tables (both families) + worst-cohort (lesion WT)."""
    return {
        "n_cases": len(scores),
        "per_cohort": {
            "legacy": per_cohort_table(scores, "legacy"),
            "lesion": per_cohort_table(scores, "lesion"),
        },
        "worst_cohort": {
            region: worst_cohort(scores, "lesion", region) for region in ("ET", "TC", "WT")
        },
    }


def evaluate_directory(pred_dir: Path, gt_dir: Path, out_path: Path | None = None) -> dict:
    """Score ``pred_dir`` vs ``gt_dir`` and (optionally) write the GoAT report JSON. Returns it."""
    scores = score_directory(pred_dir, gt_dir)
    report = build_report(scores)
    if out_path is not None:
        write_report(report, Path(out_path))
    return report

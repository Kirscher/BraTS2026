#!/usr/bin/env python3
"""Cross-fitted ET decision-threshold sweep on nnU-Net OOF probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from surface_distance import metrics as surface_metrics


REGIONS = ("et", "tc", "wt")
BASELINE_THRESHOLD = 0.5
BASELINE_MIN_VOXELS = 1
CONNECTIVITY = ndimage.generate_binary_structure(3, 3)
EXPECTED_DICE = {
    "et": 0.8673700676277436,
    "tc": 0.9125093419046888,
    "wt": 0.9267405267937684,
}


@dataclass(frozen=True)
class CaseInput:
    case_index: int
    fold: int
    probability_path: Path
    label_path: Path
    exported_path: Path


def parse_csv_numbers(value: str, cast) -> tuple:
    parsed = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
    )
    parser.add_argument("--min-voxels", default="1,5,10,25,50")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--source-commit",
        help="Optional commit identifying the source implementation being benchmarked.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    denominator = int(prediction.sum()) + int(reference.sum())
    if denominator == 0:
        return float("nan")
    return float(2 * np.logical_and(prediction, reference).sum() / denominator)


def nsd(prediction: np.ndarray, reference: np.ndarray, tolerance_mm: float = 1.0) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if not prediction.any() and not reference.any():
        return float("nan")
    if not prediction.any() or not reference.any():
        return 0.0
    distances = surface_metrics.compute_surface_distances(
        reference,
        prediction,
        spacing_mm=(1.0, 1.0, 1.0),
    )
    return float(
        surface_metrics.compute_surface_dice_at_tolerance(distances, tolerance_mm)
    )


def label_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    return labels, sizes


def filter_components(
    component_labels: np.ndarray,
    component_sizes: np.ndarray,
    min_voxels: int,
) -> np.ndarray:
    if min_voxels <= 1:
        return component_labels > 0
    keep = component_sizes >= min_voxels
    keep[0] = False
    return keep[component_labels]


def lesion_counts(
    prediction: np.ndarray,
    reference_labels: np.ndarray,
    n_reference: int,
) -> tuple[int, int, int]:
    prediction_labels, n_prediction = ndimage.label(
        np.asarray(prediction, dtype=bool),
        structure=CONNECTIVITY,
    )
    overlap = (reference_labels > 0) & (prediction_labels > 0)
    if overlap.any():
        reference_hits = np.unique(reference_labels[overlap])
        prediction_hits = np.unique(prediction_labels[overlap])
        tp = int(len(reference_hits))
        matched_predictions = int(len(prediction_hits))
    else:
        tp = 0
        matched_predictions = 0
    return tp, int(n_prediction - matched_predictions), int(n_reference - tp)


def probability_to_segmentation(
    probabilities: np.ndarray,
    et_mask: np.ndarray,
) -> np.ndarray:
    """Replicate nnU-Net region export, replacing only the ET decision rule."""
    segmentation = np.zeros(probabilities.shape[1:], dtype=np.uint8)
    segmentation[probabilities[0] > 0.5] = 2  # WT -> edema label
    segmentation[probabilities[1] > 0.5] = 1  # TC -> NCR label
    segmentation[et_mask] = 3
    return segmentation


def region_masks(segmentation: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "et": segmentation == 3,
        "tc": np.isin(segmentation, (1, 3)),
        "wt": segmentation > 0,
    }


def load_case(case: CaseInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(case.probability_path) as handle:
        probabilities = np.asarray(handle["probabilities"], dtype=np.float32)
    reference = sitk.GetArrayFromImage(sitk.ReadImage(str(case.label_path)))
    exported = sitk.GetArrayFromImage(sitk.ReadImage(str(case.exported_path)))
    if probabilities.shape != (3, *reference.shape):
        raise ValueError(
            f"case {case.case_index}: probability shape {probabilities.shape} "
            f"does not match reference {reference.shape}"
        )
    if exported.shape != reference.shape:
        raise ValueError(f"case {case.case_index}: exported/reference shape mismatch")
    return probabilities, np.asarray(reference), np.asarray(exported)


def analyze_grid_case(payload: tuple[CaseInput, tuple[float, ...], tuple[int, ...]]):
    case, thresholds, minimum_sizes = payload
    probabilities, reference, exported = load_case(case)
    references = region_masks(reference)
    reference_et_labels, reference_et_sizes = label_components(references["et"])
    n_reference_et = len(reference_et_sizes) - 1
    rows = []
    baseline_matches = None
    for threshold in thresholds:
        threshold_labels, threshold_sizes = label_components(
            probabilities[2] > threshold
        )
        for min_voxels in minimum_sizes:
            et_mask = filter_components(
                threshold_labels,
                threshold_sizes,
                min_voxels,
            )
            segmentation = probability_to_segmentation(probabilities, et_mask)
            predictions = region_masks(segmentation)
            tp, fp, fn = lesion_counts(
                predictions["et"],
                reference_et_labels,
                n_reference_et,
            )
            rows.append(
                {
                    "case_index": case.case_index,
                    "fold": case.fold,
                    "threshold": threshold,
                    "min_voxels": min_voxels,
                    **{
                        f"dsc_{region}": dice(predictions[region], references[region])
                        for region in REGIONS
                    },
                    "tp_et": tp,
                    "fp_et": fp,
                    "fn_et": fn,
                }
            )
            if (
                threshold == BASELINE_THRESHOLD
                and min_voxels == BASELINE_MIN_VOXELS
            ):
                baseline_matches = bool(np.array_equal(segmentation, exported))
    if baseline_matches is None:
        raise ValueError("baseline operating point is absent from the grid")
    return rows, baseline_matches


def analyze_surface_case(
    payload: tuple[CaseInput, float, int],
) -> list[dict[str, float | int | str]]:
    case, selected_threshold, selected_min_voxels = payload
    probabilities, reference, _ = load_case(case)
    references = region_masks(reference)
    output = []
    for configuration, threshold, min_voxels in (
        ("baseline", BASELINE_THRESHOLD, BASELINE_MIN_VOXELS),
        ("crossfit", selected_threshold, selected_min_voxels),
    ):
        labels, sizes = label_components(probabilities[2] > threshold)
        et_mask = filter_components(labels, sizes, min_voxels)
        predictions = region_masks(probability_to_segmentation(probabilities, et_mask))
        record: dict[str, float | int | str] = {
            "case_index": case.case_index,
            "fold": case.fold,
            "configuration": configuration,
            "threshold": threshold,
            "min_voxels": min_voxels,
        }
        for region in REGIONS:
            record[f"dsc_{region}"] = dice(predictions[region], references[region])
            record[f"nsd_{region}"] = nsd(predictions[region], references[region])
        output.append(record)
    return output


def discover_cases(predictions_root: Path, labels_dir: Path) -> list[CaseInput]:
    cases = []
    case_index = 0
    seen_names = set()
    for fold in range(5):
        validation = predictions_root / f"fold_{fold}" / "validation"
        probability_paths = sorted(validation.glob("*.npz"))
        expected = 271 if fold == 0 else 270
        if len(probability_paths) != expected:
            raise ValueError(
                f"fold {fold}: expected {expected} probability files, "
                f"got {len(probability_paths)}"
            )
        for probability_path in probability_paths:
            if probability_path.stem in seen_names:
                raise ValueError("a case appears in more than one OOF fold")
            seen_names.add(probability_path.stem)
            label_path = labels_dir / f"{probability_path.stem}.nii.gz"
            exported_path = validation / f"{probability_path.stem}.nii.gz"
            if not label_path.exists() or not exported_path.is_file():
                raise FileNotFoundError("missing label or exported OOF prediction")
            cases.append(
                CaseInput(
                    case_index=case_index,
                    fold=fold,
                    probability_path=probability_path,
                    label_path=label_path,
                    exported_path=exported_path,
                )
            )
            case_index += 1
    if len(cases) != 1351:
        raise ValueError(f"expected 1351 unique OOF cases, got {len(cases)}")
    return cases


def grid_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (threshold, min_voxels), group in frame.groupby(
        ["threshold", "min_voxels"],
        sort=True,
    ):
        tp = int(group["tp_et"].sum())
        fp = int(group["fp_et"].sum())
        fn = int(group["fn_et"].sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f2 = (
            5 * precision * recall / (4 * precision + recall)
            if 4 * precision + recall
            else 0.0
        )
        dsc_means = {region: group[f"dsc_{region}"].mean() for region in REGIONS}
        records.append(
            {
                "threshold": threshold,
                "min_voxels": int(min_voxels),
                **{f"mean_dsc_{region}": value for region, value in dsc_means.items()},
                "mean_region_dsc": float(np.mean(list(dsc_means.values()))),
                "tp_et": tp,
                "fp_et": fp,
                "fn_et": fn,
                "lesion_precision_et": precision,
                "lesion_recall_et": recall,
                "lesion_f1_et": f1,
                "lesion_f2_et": f2,
            }
        )
    return pd.DataFrame(records)


def select_operating_points(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for held_out_fold in range(5):
        calibration = frame[frame["fold"] != held_out_fold]
        summary = grid_summary(calibration)
        best = summary.sort_values(
            ["mean_region_dsc", "mean_dsc_et", "threshold", "min_voxels"],
            ascending=[False, False, False, False],
        ).iloc[0]
        records.append(
            {
                "held_out_fold": held_out_fold,
                "threshold": float(best["threshold"]),
                "min_voxels": int(best["min_voxels"]),
                "calibration_mean_region_dsc": float(best["mean_region_dsc"]),
                "calibration_mean_dsc_et": float(best["mean_dsc_et"]),
            }
        )
    return pd.DataFrame(records)


def paired_bootstrap(
    differences: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float, float, int]:
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        raise ValueError("no finite paired differences")
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    samples = differences[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(differences.mean()), float(low), float(high), int(len(differences))


def effect_summary(
    surface_frame: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    baseline = surface_frame[surface_frame["configuration"] == "baseline"].set_index(
        "case_index"
    )
    crossfit = surface_frame[surface_frame["configuration"] == "crossfit"].set_index(
        "case_index"
    )
    if not baseline.index.equals(crossfit.index):
        raise ValueError("baseline and cross-fitted case indices do not align")
    rng = np.random.default_rng(seed)
    records = []
    for family in ("dsc", "nsd"):
        for region in REGIONS:
            metric = f"{family}_{region}"
            paired = pd.concat(
                [crossfit[metric].rename("crossfit"), baseline[metric].rename("baseline")],
                axis=1,
            ).dropna()
            mean, low, high, count = paired_bootstrap(
                (paired["crossfit"] - paired["baseline"]).to_numpy(),
                rng,
                bootstrap_replicates,
            )
            records.append(
                {
                    "metric": metric,
                    "n": count,
                    "baseline_mean": float(paired["baseline"].mean()),
                    "crossfit_mean": float(paired["crossfit"].mean()),
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(records)


def map_parallel(function, payloads: Iterable, workers: int):
    if workers == 1:
        return [function(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, payloads, chunksize=1))


def main() -> None:
    args = parse_args()
    thresholds = parse_csv_numbers(args.thresholds, float)
    minimum_sizes = parse_csv_numbers(args.min_voxels, int)
    if BASELINE_THRESHOLD not in thresholds or BASELINE_MIN_VOXELS not in minimum_sizes:
        raise ValueError("the grid must contain the baseline (0.5, 1)")
    if args.workers < 1 or args.bootstrap_replicates < 1:
        raise ValueError("workers and bootstrap replicates must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing existing output root: {args.output}")
    if not args.predictions_root.is_dir() or not args.labels_dir.is_dir():
        raise FileNotFoundError("prediction or label input root is missing")

    cases = discover_cases(args.predictions_root, args.labels_dir)
    args.output.mkdir(parents=True)

    grid_results = map_parallel(
        analyze_grid_case,
        ((case, thresholds, minimum_sizes) for case in cases),
        args.workers,
    )
    grid_rows = [row for rows, _ in grid_results for row in rows]
    baseline_match_count = sum(match for _, match in grid_results)
    if baseline_match_count != len(cases):
        raise RuntimeError(
            f"baseline reconstruction matched only {baseline_match_count}/{len(cases)} "
            "exported segmentations"
        )
    grid = pd.DataFrame(grid_rows)
    summary = grid_summary(grid)
    selection = select_operating_points(grid)
    selected_by_fold = selection.set_index("held_out_fold")

    surface_results = map_parallel(
        analyze_surface_case,
        (
            (
                case,
                float(selected_by_fold.loc[case.fold, "threshold"]),
                int(selected_by_fold.loc[case.fold, "min_voxels"]),
            )
            for case in cases
        ),
        args.workers,
    )
    surface = pd.DataFrame([row for rows in surface_results for row in rows])
    effects = effect_summary(surface, args.bootstrap_replicates, args.seed)

    baseline = summary[
        np.isclose(summary["threshold"], BASELINE_THRESHOLD)
        & summary["min_voxels"].eq(BASELINE_MIN_VOXELS)
    ].iloc[0]
    for region, expected in EXPECTED_DICE.items():
        observed = float(baseline[f"mean_dsc_{region}"])
        if not np.isclose(observed, expected, atol=1e-10, rtol=0):
            raise RuntimeError(
                f"baseline {region} DSC mismatch: observed {observed}, expected {expected}"
            )

    grid.to_csv(args.output / "grid_per_case.csv", index=False)
    summary.to_csv(args.output / "grid_summary.csv", index=False)
    selection.to_csv(args.output / "fold_selection.csv", index=False)
    surface.to_csv(args.output / "crossfit_per_case.csv", index=False)
    effects.to_csv(args.output / "paired_effects.csv", index=False)

    input_counts = {
        f"fold_{fold}": sum(case.fold == fold for case in cases) for fold in range(5)
    }
    metadata = {
        "experiment_id": "BRATS2026-20260729-ETTHR001",
        "analysis": "cross_fitted_et_threshold_and_component_size_sweep",
        "created_at": pd.Timestamp.now(tz="Europe/Paris").isoformat(),
        "command": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "source": {
            "source_commit": args.source_commit,
            "local_git_commit": os.environ.get("BRATS_GIT_COMMIT", "unknown"),
            "local_git_status_sha256": os.environ.get(
                "BRATS_GIT_STATUS_SHA256",
                "unknown",
            ),
        },
        "inputs": {
            "predictions_root": str(args.predictions_root.resolve()),
            "labels_dir": str(args.labels_dir.resolve()),
            "case_counts": input_counts,
            "probability_channel_order": ["WT", "TC", "ET"],
            "et_probability_channel": 2,
            "array_order": "z,y,x for both npz probabilities and SimpleITK labels",
        },
        "grid": {
            "thresholds": thresholds,
            "minimum_component_voxels": minimum_sizes,
            "baseline": {
                "threshold": BASELINE_THRESHOLD,
                "min_voxels": BASELINE_MIN_VOXELS,
            },
            "connectivity": 26,
        },
        "selection": {
            "scheme": "five-fold cross-fit by OOF fold",
            "objective": "calibration-fold arithmetic mean of ET, TC, WT DSC",
            "limitation": (
                "calibration-fold models were trained on some held-out-fold cases; "
                "inference is OOF per evaluated case but the procedure is not fully nested"
            ),
        },
        "metrics": {
            "dice": "case-wise; both-empty NaN; otherwise 2TP/(2TP+FP+FN)",
            "nsd": (
                "surface-distance 0.1 at 1 mm; both-empty NaN; one-empty 0; "
                "case-wise mean"
            ),
            "lesion_matching": (
                "26-connected components; a reference lesion is detected by any overlap; "
                "unmatched predicted components are false positives"
            ),
            "bootstrap": {
                "replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "method": "paired case percentile interval",
            },
        },
        "validation": {
            "baseline_export_exact_matches": baseline_match_count,
            "expected_cases": len(cases),
            "baseline_expected_dice": EXPECTED_DICE,
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "simpleitk": sitk.Version_VersionString(),
        },
        "outputs": [
            "grid_per_case.csv",
            "grid_summary.csv",
            "fold_selection.csv",
            "crossfit_per_case.csv",
            "paired_effects.csv",
            "metadata.json",
            "report.md",
        ],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Cross-fitted ET threshold benchmark",
        "",
        "The baseline hard-label reconstruction matched all exported OOF segmentations.",
        "",
        "## Fold-wise selected operating points",
        "",
        selection.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Cross-fitted paired effects",
        "",
        effects.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best full-grid descriptive operating points",
        "",
        summary.sort_values("mean_region_dsc", ascending=False)
        .head(10)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "Selection and estimation are exploratory and conditional on the fitted OOF models.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")

    for path in sorted(args.output.iterdir()):
        if path.is_file():
            print(f"OUTPUT {path.name} sha256={sha256(path)}")
    print(f"Analysis complete: {args.output}")


if __name__ == "__main__":
    main()

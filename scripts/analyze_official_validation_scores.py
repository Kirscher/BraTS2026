#!/usr/bin/env python3
"""Analyze paired official BraTS validation scores and the source-target gap."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = ("et", "tc", "wt")
METRIC_FAMILIES = ("global_dsc", "global_nsd", "global_hd95", "all_instance_f1")
PRIMARY_FAMILIES = ("global_dsc", "global_nsd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1-notta", required=True, type=Path)
    parser.add_argument("--f1-tta", required=True, type=Path)
    parser.add_argument("--f5-notta", required=True, type=Path)
    parser.add_argument("--f5-tta", required=True, type=Path)
    parser.add_argument("--resenc-f5-tta", required=True, type=Path)
    parser.add_argument("--oof-features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    case_rows = frame["subject_id"].str.match(
        r"^BraTS-GoAT-\d{5}-seg\.nii\.gz$", na=False
    )
    frame = frame[case_rows].copy()
    if len(frame) != 450 or frame["subject_id"].nunique() != 450:
        raise ValueError(f"{path}: expected the 450 case rows returned by the scorer")
    return frame.set_index("subject_id").sort_index()


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_summary(
    left: pd.Series,
    right: pd.Series,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, float | int]:
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1)
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
    differences = paired["left"].to_numpy() - paired["right"].to_numpy()
    if not len(differences):
        raise ValueError("no finite paired observations")
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    low, high = percentile_ci(bootstrap)
    return {
        "n": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "ci_low": low,
        "ci_high": high,
    }


def independent_summary(
    target: pd.Series,
    source: pd.Series,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, float | int]:
    target_values = target.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    source_values = source.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if not len(target_values) or not len(source_values):
        raise ValueError("source-target comparison has no finite observations")
    target_indices = rng.integers(
        0, len(target_values), size=(replicates, len(target_values))
    )
    source_indices = rng.integers(
        0, len(source_values), size=(replicates, len(source_values))
    )
    bootstrap = target_values[target_indices].mean(axis=1) - source_values[
        source_indices
    ].mean(axis=1)
    low, high = percentile_ci(bootstrap)
    return {
        "n_target": int(len(target_values)),
        "n_source": int(len(source_values)),
        "target_mean": float(target_values.mean()),
        "source_mean": float(source_values.mean()),
        "mean_difference": float(target_values.mean() - source_values.mean()),
        "ci_low": low,
        "ci_high": high,
    }


def independent_region_mean_summary(
    target: pd.DataFrame,
    source: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, float | int]:
    target_values = target.replace([np.inf, -np.inf], np.nan).to_numpy(float)
    source_values = source.replace([np.inf, -np.inf], np.nan).to_numpy(float)
    target_mean = float(np.nanmean(target_values, axis=0).mean())
    source_mean = float(np.nanmean(source_values, axis=0).mean())
    bootstrap = np.empty(replicates, dtype=float)
    batch_size = 500
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        batch = stop - start
        target_indices = rng.integers(
            0, len(target_values), size=(batch, len(target_values))
        )
        source_indices = rng.integers(
            0, len(source_values), size=(batch, len(source_values))
        )
        target_bootstrap = np.nanmean(
            target_values[target_indices], axis=1
        ).mean(axis=1)
        source_bootstrap = np.nanmean(
            source_values[source_indices], axis=1
        ).mean(axis=1)
        bootstrap[start:stop] = target_bootstrap - source_bootstrap
    low, high = percentile_ci(bootstrap)
    return {
        "n_target": int(len(target_values)),
        "n_source": int(len(source_values)),
        "target_mean": target_mean,
        "source_mean": source_mean,
        "mean_difference": target_mean - source_mean,
        "ci_low": low,
        "ci_high": high,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing existing output root: {args.output}")

    source_paths = {
        "f1_notta": args.f1_notta,
        "f1_tta": args.f1_tta,
        "f5_notta": args.f5_notta,
        "f5_tta": args.f5_tta,
        "resenc_f5_tta": args.resenc_f5_tta,
    }
    args.output.mkdir(parents=True)
    input_root = args.output / "inputs"
    input_root.mkdir()
    paths = {}
    for name, source_path in source_paths.items():
        destination = input_root / f"{name}_all_scores.csv"
        shutil.copyfile(source_path, destination)
        paths[name] = destination
    scores = {name: load_scores(path) for name, path in paths.items()}
    reference_ids = scores["f5_notta"].index
    for name, frame in scores.items():
        if not frame.index.equals(reference_ids):
            raise ValueError(f"{name}: case identifiers do not align")

    comparisons = {
        "f1_tta_minus_f1_notta": ("f1_tta", "f1_notta"),
        "f5_tta_minus_f5_notta": ("f5_tta", "f5_notta"),
        "f5_notta_minus_f1_notta": ("f5_notta", "f1_notta"),
        "f5_tta_minus_f1_tta": ("f5_tta", "f1_tta"),
        "resenc_f5_tta_minus_plain_f5_tta": ("resenc_f5_tta", "f5_tta"),
    }
    rng = np.random.default_rng(args.seed)
    paired_records: list[dict[str, object]] = []
    for comparison, (left_name, right_name) in comparisons.items():
        for family in METRIC_FAMILIES:
            for region in REGIONS:
                metric = f"{family}_{region}"
                summary = paired_summary(
                    scores[left_name][metric],
                    scores[right_name][metric],
                    rng,
                    args.bootstrap_replicates,
                )
                paired_records.append(
                    {
                        "comparison": comparison,
                        "metric_family": family,
                        "region": region.upper(),
                        "metric": metric,
                        **summary,
                    }
                )
    paired = pd.DataFrame(paired_records)
    primary = paired["metric_family"].isin(PRIMARY_FAMILIES)

    oof = pd.read_csv(args.oof_features)
    source = oof[oof["fold"].eq(0)]
    if len(source) != 271:
        raise ValueError(f"expected 271 fold-0 OOF cases, got {len(source)}")
    gap_records = []
    for region in REGIONS:
        summary = independent_summary(
            scores["f1_tta"][f"global_dsc_{region}"],
            source[f"{region}_dice"],
            rng,
            args.bootstrap_replicates,
        )
        gap_records.append({"region": region.upper(), **summary})
    gap_records.append(
        {
            "region": "MEAN",
            **independent_region_mean_summary(
                scores["f1_tta"][
                    [f"global_dsc_{region}" for region in REGIONS]
                ],
                source[[f"{region}_dice" for region in REGIONS]],
                rng,
                args.bootstrap_replicates,
            ),
        }
    )
    gap = pd.DataFrame(gap_records)

    aggregate_records = []
    for name, frame in scores.items():
        for family in METRIC_FAMILIES:
            for region in REGIONS:
                metric = f"{family}_{region}"
                finite = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
                aggregate_records.append(
                    {
                        "configuration": name,
                        "metric": metric,
                        "n": int(len(finite)),
                        "mean": float(finite.mean()),
                        "sd": float(finite.std(ddof=1)),
                    }
                )
    aggregate = pd.DataFrame(aggregate_records)

    paired.to_csv(args.output / "paired_comparisons.csv", index=False)
    gap.to_csv(args.output / "source_target_gap.csv", index=False)
    aggregate.to_csv(args.output / "aggregate_scores.csv", index=False)

    input_paths = {**paths, "oof_features": args.oof_features}
    metadata = {
        "analysis": "official_validation_paired_statistics",
        "entrypoint": Path(__file__).name,
        "seed": args.seed,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "paired_method": "percentile bootstrap of paired case differences",
            "source_target_method": (
                "independent percentile bootstrap of regional and region-mean "
                "differences; region mean preserves within-case covariance"
            ),
        },
        "hypothesis_tests": "none; post-selection contrasts are effect estimates",
        "scorer_case_rows": {
            "archive_predictions": 451,
            "rows_returned": 450,
            "handling": "analysis restricted to case rows shared by all score files",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "inputs": {
            name: {
                "file": (
                    str(path.relative_to(args.output))
                    if path.is_relative_to(args.output)
                    else path.name
                ),
                "sha256": sha256(path),
            }
            for name, path in input_paths.items()
        },
        "outputs": [
            "inputs/*.csv",
            "paired_comparisons.csv",
            "source_target_gap.csv",
            "aggregate_scores.csv",
            "metadata.json",
            "report.md",
        ],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Official validation paired statistics",
        "",
        "All inferential summaries are descriptive and post-selection.",
        "",
        "## Source-to-pooled-target DSC gap",
        "",
        gap.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired DSC/NSD comparisons",
        "",
        paired[primary][
            [
                "comparison",
                "metric",
                "n",
                "mean_difference",
                "ci_low",
                "ci_high",
            ]
        ].to_markdown(index=False, floatfmt=".6g"),
        "",
        "Intervals quantify case-sampling uncertainty only.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Analysis complete: {args.output}")


if __name__ == "__main__":
    main()

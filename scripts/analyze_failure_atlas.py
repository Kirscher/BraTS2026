#!/usr/bin/env python3
"""Exploratory failure analysis for the stock nnU-Net out-of-fold predictions.

This script deliberately analyzes only labeled training cases with their
out-of-fold prediction. The header-derived groups are acquisition proxies, not
official tumor-population labels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DATASET = "Dataset501_BraTSGoAT"
MODEL = "nnUNetTrainer__nnUNetPlans__3d_fullres"
RESULTS = ROOT / "work/nnUNet_results" / DATASET / MODEL
GT_ROOT = ROOT / "work/nnUNet_preprocessed" / DATASET / "gt_segmentations"
SPLITS = ROOT / "work/nnUNet_preprocessed" / DATASET / "splits_final.json"
PROXY_GROUPS = ROOT / "work/reports/cohort_map.json"
PROXY_METADATA = ROOT / "work/reports/cohort_map_metadata.json"
OUTPUT: Path

REGIONS = {
    "wt": (1, 2, 3),
    "tc": (1, 3),
    "et": (3,),
}
SUMMARY_KEYS = {
    "wt": "(1, 2, 3)",
    "tc": "(1, 3)",
    "et": "(3,)",
}
PROXY_RENAME = {
    "GLI": "Domain A (q1/s0)",
    "SSA": "Domain B (q2/s1)",
    "MEN": "Domain C (q0/s2)",
    "PED": "Domain D (q1/s1)",
}
MORPH_FEATURES = [
    "log_wt_volume",
    "log_tc_volume",
    "log_et_volume",
    "tc_wt_ratio",
    "et_wt_ratio",
    "wt_components_10",
    "tc_components_10",
    "et_components_10",
    "wt_largest_fraction",
    "tc_largest_fraction",
    "et_largest_fraction",
    "wt_compactness",
    "tc_compactness",
    "et_compactness",
    "wt_extent",
    "wt_span_x",
    "wt_span_y",
    "wt_span_z",
    "wt_centroid_x",
    "wt_centroid_y",
    "wt_centroid_z",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_id(path: str | Path) -> str:
    name = Path(path).name
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem


def load_metrics() -> pd.DataFrame:
    split_data = json.loads(SPLITS.read_text())
    fold_for_case: dict[str, int] = {}
    for fold, split in enumerate(split_data):
        for case in split["val"]:
            if case in fold_for_case:
                raise RuntimeError(f"case occurs in multiple validation folds: {case}")
            fold_for_case[case] = fold

    records: list[dict[str, object]] = []
    for fold in range(5):
        summary_path = RESULTS / f"fold_{fold}/validation/summary.json"
        summary = json.loads(summary_path.read_text())
        for item in summary["metric_per_case"]:
            case = case_id(item["prediction_file"])
            if fold_for_case.get(case) != fold:
                raise RuntimeError(
                    f"{case}: summary fold {fold} != split fold {fold_for_case.get(case)}"
                )
            row: dict[str, object] = {
                "case": case,
                "fold": fold,
                "prediction_path": item["prediction_file"],
                "reference_path": item["reference_file"],
            }
            for region, key in SUMMARY_KEYS.items():
                row[f"{region}_dice"] = item["metrics"][key]["Dice"]
                row[f"{region}_fp"] = item["metrics"][key]["FP"]
                row[f"{region}_fn"] = item["metrics"][key]["FN"]
            records.append(row)
    frame = pd.DataFrame(records)
    if len(frame) != 1351 or frame["case"].nunique() != 1351:
        raise RuntimeError(
            f"expected 1,351 unique OOF cases, got {len(frame)} rows and "
            f"{frame['case'].nunique()} unique cases"
        )
    return frame


def morphology(array: np.ndarray, labels: tuple[int, ...], prefix: str) -> dict[str, float]:
    mask = np.isin(array, labels)
    volume = int(mask.sum())
    result = {f"{prefix}_volume": float(volume)}
    if volume == 0:
        result.update(
            {
                f"{prefix}_components": 0.0,
                f"{prefix}_components_10": 0.0,
                f"{prefix}_largest_fraction": np.nan,
                f"{prefix}_surface_voxels": 0.0,
                f"{prefix}_compactness": np.nan,
                f"{prefix}_extent": np.nan,
                f"{prefix}_span_x": 0.0,
                f"{prefix}_span_y": 0.0,
                f"{prefix}_span_z": 0.0,
                f"{prefix}_centroid_x": np.nan,
                f"{prefix}_centroid_y": np.nan,
                f"{prefix}_centroid_z": np.nan,
            }
        )
        return result

    coords = np.argwhere(mask)
    lower = coords.min(axis=0)
    upper = coords.max(axis=0) + 1
    slices = tuple(slice(max(0, int(lo) - 1), min(mask.shape[i], int(hi) + 1))
                   for i, (lo, hi) in enumerate(zip(lower, upper)))
    crop = mask[slices]
    components, count = ndimage.label(crop, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(components.ravel())[1:]
    surface = crop & ~ndimage.binary_erosion(crop, structure=np.ones((3, 3, 3)))
    span = upper - lower
    centroid = coords.mean(axis=0) / np.asarray(mask.shape)
    bbox_volume = int(np.prod(span))
    result.update(
        {
            f"{prefix}_components": float(count),
            f"{prefix}_components_10": float(np.count_nonzero(sizes >= 10)),
            f"{prefix}_largest_fraction": float(sizes.max() / volume),
            f"{prefix}_surface_voxels": float(surface.sum()),
            f"{prefix}_compactness": float(surface.sum() / (volume ** (2 / 3))),
            f"{prefix}_extent": float(volume / bbox_volume),
            f"{prefix}_span_x": float(span[0] / mask.shape[0]),
            f"{prefix}_span_y": float(span[1] / mask.shape[1]),
            f"{prefix}_span_z": float(span[2] / mask.shape[2]),
            f"{prefix}_centroid_x": float(centroid[0]),
            f"{prefix}_centroid_y": float(centroid[1]),
            f"{prefix}_centroid_z": float(centroid[2]),
        }
    )
    return result


def extract_case(metric: dict[str, object]) -> dict[str, object]:
    reference_path = Path(metric["reference_path"])
    prediction_path = Path(metric["prediction_path"])
    if not reference_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"missing OOF pair for {metric['case']}")
    reference = np.asarray(nib.load(reference_path).dataobj, dtype=np.uint8)
    prediction = np.asarray(nib.load(prediction_path).dataobj, dtype=np.uint8)
    if reference.shape != prediction.shape:
        raise RuntimeError(f"{metric['case']}: shape mismatch")

    row = dict(metric)
    for region, labels in REGIONS.items():
        row.update(morphology(reference, labels, region))
        pred_features = morphology(prediction, labels, f"pred_{region}")
        row.update(pred_features)
        reference_volume = row[f"{region}_volume"]
        predicted_volume = row[f"pred_{region}_volume"]
        row[f"{region}_volume_bias"] = (
            float((predicted_volume - reference_volume) / reference_volume)
            if reference_volume
            else np.nan
        )
    row["tc_wt_ratio"] = row["tc_volume"] / row["wt_volume"] if row["wt_volume"] else np.nan
    row["et_wt_ratio"] = row["et_volume"] / row["wt_volume"] if row["wt_volume"] else np.nan
    row["et_tc_ratio"] = row["et_volume"] / row["tc_volume"] if row["tc_volume"] else np.nan
    for region in REGIONS:
        row[f"log_{region}_volume"] = np.log10(1 + row[f"{region}_volume"])
    return row


def extract_features(metrics: pd.DataFrame, workers: int) -> pd.DataFrame:
    proxy = json.loads(PROXY_GROUPS.read_text())
    metrics = metrics.copy()
    metrics["proxy_header_group_raw"] = metrics["case"].map(proxy).fillna("UNKNOWN")
    metrics["header_domain"] = metrics["proxy_header_group_raw"].map(PROXY_RENAME).fillna(
        "Unknown domain"
    )
    records = metrics.to_dict(orient="records")
    rows: list[dict[str, object]] = []
    if workers == 1:
        iterator = map(extract_case, records)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(extract_case, records, chunksize=2)
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if index % 100 == 0 or index == len(records):
                print(f"features: {index}/{len(records)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iterations: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_domains(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for domain, group in frame.groupby("header_domain", observed=True):
        row: dict[str, object] = {"header_domain": domain, "n": len(group)}
        for outcome in ("mean_dice", "wt_dice", "tc_dice", "et_dice", "failure_bottom10"):
            series = group[outcome]
            if outcome == "failure_bottom10":
                series = series.where(group["failure_eligible"])
            values = series.to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            low, high = bootstrap_ci(finite_values, rng)
            row[f"{outcome}_n"] = len(finite_values)
            row[f"{outcome}_mean"] = (
                float(np.mean(finite_values)) if len(finite_values) else float("nan")
            )
            row[f"{outcome}_ci_low"] = low
            row[f"{outcome}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("header_domain")


def association_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlations = []
    for feature in MORPH_FEATURES:
        for outcome in ("mean_dice", "wt_dice", "tc_dice", "et_dice"):
            subset = frame[[feature, outcome]].dropna()
            rho, p_value = stats.spearmanr(subset[feature], subset[outcome])
            correlations.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "rho": rho,
                    "p_value": p_value,
                    "n": len(subset),
                }
            )
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame["p_fdr"] = multipletests(
        correlation_frame["p_value"], method="fdr_bh"
    )[1]

    effects = []
    eligible = frame["failure_eligible"]
    failed = eligible & frame["failure_bottom10"]
    nonfailed = eligible & ~frame["failure_bottom10"]
    for feature in MORPH_FEATURES:
        x = frame.loc[failed, feature].dropna().to_numpy()
        y = frame.loc[nonfailed, feature].dropna().to_numpy()
        statistic, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        rank_biserial = 2 * statistic / (len(x) * len(y)) - 1
        effects.append(
            {
                "feature": feature,
                "failure_median": float(np.median(x)),
                "nonfailure_median": float(np.median(y)),
                "rank_biserial_failure_higher": float(rank_biserial),
                "p_value": float(p_value),
                "n_failure": len(x),
                "n_nonfailure": len(y),
            }
        )
    effect_frame = pd.DataFrame(effects)
    effect_frame["p_fdr"] = multipletests(effect_frame["p_value"], method="fdr_bh")[1]
    return correlation_frame, effect_frame


def partial_spearman_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Run targeted exploratory associations after rank-based size adjustment."""
    hypotheses = (
        ("log_et_volume", "mean_dice", ("log_wt_volume",)),
        ("et_wt_ratio", "mean_dice", ("log_et_volume", "log_wt_volume")),
        (
            "et_largest_fraction",
            "mean_dice",
            ("log_et_volume", "log_wt_volume"),
        ),
        ("et_components_10", "mean_dice", ("log_et_volume", "log_wt_volume")),
        ("et_compactness", "et_dice", ("log_et_volume",)),
    )
    rows = []
    for feature, outcome, controls in hypotheses:
        columns = [feature, outcome, *controls]
        ranked = frame[columns].dropna().rank(method="average")
        design = np.column_stack(
            [np.ones(len(ranked)), ranked[list(controls)].to_numpy()]
        )
        feature_values = ranked[feature].to_numpy()
        outcome_values = ranked[outcome].to_numpy()
        feature_residual = feature_values - design @ np.linalg.lstsq(
            design, feature_values, rcond=None
        )[0]
        outcome_residual = outcome_values - design @ np.linalg.lstsq(
            design, outcome_values, rcond=None
        )[0]
        rho, p_value = stats.pearsonr(feature_residual, outcome_residual)
        rows.append(
            {
                "feature": feature,
                "outcome": outcome,
                "controls": ",".join(controls),
                "partial_rho": float(rho),
                "p_value": float(p_value),
                "n": len(ranked),
            }
        )
    result = pd.DataFrame(rows)
    result["p_fdr"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def failure_sensitivity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Repeat rank-biserial morphology comparisons at 5%, 10%, and 20% cutoffs."""
    rows = []
    eligible = frame["failure_eligible"]
    for percentile, flag in (
        (5, "failure_bottom05"),
        (10, "failure_bottom10"),
        (20, "failure_bottom20"),
    ):
        failed = eligible & frame[flag]
        nonfailed = eligible & ~frame[flag]
        for feature in MORPH_FEATURES:
            x = frame.loc[failed, feature].dropna().to_numpy()
            y = frame.loc[nonfailed, feature].dropna().to_numpy()
            statistic, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "percentile": percentile,
                    "feature": feature,
                    "rank_biserial_failure_higher": float(
                        2 * statistic / (len(x) * len(y)) - 1
                    ),
                    "p_value": float(p_value),
                    "n_failure": len(x),
                    "n_nonfailure": len(y),
                }
            )
    result = pd.DataFrame(rows)
    result["p_fdr_within_percentile"] = result.groupby("percentile")[
        "p_value"
    ].transform(lambda values: multipletests(values, method="fdr_bh")[1])
    return result


def build_embedding(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    values = frame[MORPH_FEATURES]
    imputed = SimpleImputer(strategy="median").fit_transform(values)
    scaled = StandardScaler().fit_transform(imputed)
    pca = PCA(n_components=min(10, scaled.shape[1]), random_state=seed)
    pca_values = pca.fit_transform(scaled)
    tsne_values = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        max_iter=1500,
        random_state=seed,
    ).fit_transform(pca_values)
    embedded = frame[
        ["case", "fold", "header_domain", "mean_dice", "wt_dice", "tc_dice", "et_dice",
         "failure_bottom10"]
    ].copy()
    embedded["pca1"] = pca_values[:, 0]
    embedded["pca2"] = pca_values[:, 1]
    embedded["tsne1"] = tsne_values[:, 0]
    embedded["tsne2"] = tsne_values[:, 1]
    return embedded, pca.explained_variance_ratio_


def plot_embedding(embedded: pd.DataFrame, explained: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)
    common = dict(s=10, alpha=0.72, linewidths=0)
    scatter = axes[0].scatter(
        embedded["pca1"], embedded["pca2"], c=embedded["mean_dice"],
        cmap="viridis", vmin=0, vmax=1, **common
    )
    axes[0].set(
        title="PCA of reference morphology",
        xlabel=f"PC1 ({explained[0] * 100:.1f}%)",
        ylabel=f"PC2 ({explained[1] * 100:.1f}%)",
    )
    fig.colorbar(scatter, ax=axes[0], label="Mean WT/TC/ET Dice", fraction=0.05)

    scatter = axes[1].scatter(
        embedded["tsne1"], embedded["tsne2"], c=embedded["mean_dice"],
        cmap="viridis", vmin=0, vmax=1, **common
    )
    axes[1].set(title="t-SNE (visualization only)", xlabel="t-SNE 1", ylabel="t-SNE 2")
    fig.colorbar(scatter, ax=axes[1], label="Mean WT/TC/ET Dice", fraction=0.05)

    palette = {
        "Domain A (q1/s0)": "#0072B2",
        "Domain B (q2/s1)": "#D55E00",
        "Domain C (q0/s2)": "#009E73",
        "Domain D (q1/s1)": "#CC79A7",
        "Unknown domain": "#777777",
    }
    for domain, group in embedded.groupby("header_domain", observed=True):
        axes[2].scatter(
            group["tsne1"], group["tsne2"], s=10, alpha=0.65,
            color=palette.get(domain, "#777777"), label=domain, linewidths=0
        )
    failed = embedded[embedded["failure_bottom10"]]
    axes[2].scatter(
        failed["tsne1"], failed["tsne2"], s=24, facecolors="none",
        edgecolors="black", linewidths=0.65, label="Bottom-decile failure"
    )
    axes[2].set(
        title="Header domains (not populations)",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    axes[2].legend(frameon=False, fontsize=7, loc="best")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "morphology_embedding.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "morphology_embedding.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_associations(frame: pd.DataFrame, effects: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4), constrained_layout=True)
    for axis, region, color in (
        (axes[0, 0], "wt", "#0072B2"),
        (axes[0, 1], "et", "#D55E00"),
    ):
        x = frame[f"log_{region}_volume"]
        y = frame[f"{region}_dice"]
        axis.scatter(x, y, s=7, alpha=0.28, color=color, linewidths=0)
        bins = pd.qcut(x, 12, duplicates="drop")
        grouped = frame.assign(_bin=bins).groupby("_bin", observed=True)
        axis.plot(
            grouped[f"log_{region}_volume"].median(),
            grouped[f"{region}_dice"].median(),
            color="black", marker="o", markersize=3, linewidth=1.3,
        )
        axis.set(
            xlabel=f"log10(1 + {region.upper()} voxels)",
            ylabel=f"{region.upper()} Dice",
            title=f"{region.upper()} size and segmentation performance",
            ylim=(-0.02, 1.02),
        )

    eligible_frame = frame[frame["failure_eligible"]]
    failure_rates = (
        eligible_frame.assign(
            volume_quintile=pd.qcut(eligible_frame["log_wt_volume"], 5, labels=False)
        )
        .groupby("volume_quintile", observed=True)["failure_bottom10"]
        .agg(["mean", "count"])
        .reset_index()
    )
    axes[1, 0].bar(
        failure_rates["volume_quintile"].astype(int) + 1,
        failure_rates["mean"],
        color="#56B4E9",
    )
    axes[1, 0].set(
        xlabel="WT volume quintile (small → large)",
        ylabel="Bottom-decile failure rate",
        title="Failure concentration by tumor size",
        ylim=(0, max(0.25, failure_rates["mean"].max() * 1.15)),
    )

    selected = effects.reindex(
        effects["rank_biserial_failure_higher"].abs().sort_values().index
    ).tail(10)
    colors = np.where(
        selected["rank_biserial_failure_higher"] > 0, "#D55E00", "#0072B2"
    )
    axes[1, 1].barh(
        selected["feature"].str.replace("_", " "),
        selected["rank_biserial_failure_higher"],
        color=colors,
    )
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set(
        xlabel="Rank-biserial effect (positive = higher in failures)",
        title="Largest morphology differences",
        xlim=(-1, 1),
    )
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "failure_associations.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "failure_associations.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    domains: pd.DataFrame,
    correlations: pd.DataFrame,
    effects: pd.DataFrame,
    partial_correlations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    explained: np.ndarray,
) -> None:
    strongest_correlations = correlations.sort_values("rho", key=abs, ascending=False).head(12)
    strongest_effects = effects.sort_values(
        "rank_biserial_failure_higher", key=abs, ascending=False
    ).head(12)
    kruskal = {}
    for outcome in ("mean_dice", "wt_dice", "tc_dice", "et_dice"):
        groups = [
            group[outcome].dropna().to_numpy()
            for _, group in frame.groupby("header_domain", observed=True)
        ]
        statistic, p_value = stats.kruskal(*groups)
        kruskal[outcome] = {"H": float(statistic), "p": float(p_value)}

    lines = [
        "# Failure atlas",
        "",
        "Exploratory analysis of 1,351 stock nnU-Net out-of-fold predictions.",
        "The four header domains are acquisition/preprocessing proxies and must not",
        "be interpreted as official tumor-population labels.",
        "",
        "## Cohort and failure definitions",
        "",
        f"- Mean Dice bottom-decile threshold: `{frame['mean_dice'].quantile(0.10):.6f}`.",
        f"- Bottom-decile failures: `{int(frame['failure_bottom10'].sum())}/"
        f"{int(frame['failure_eligible'].sum())}` complete cases.",
        f"- ET Dice < 0.5: `{int((frame['et_dice'] < 0.5).sum())}/"
        f"{int(frame['et_dice'].notna().sum())}` evaluable cases.",
        f"- WT Dice < 0.8: `{int((frame['wt_dice'] < 0.8).sum())}/{len(frame)}`.",
        f"- PCA variance: PC1 `{explained[0]:.4f}`, PC2 `{explained[1]:.4f}`.",
        "",
        "## Strongest Spearman associations",
        "",
        strongest_correlations.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Largest bottom-decile morphology effects",
        "",
        strongest_effects.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Targeted exploratory partial Spearman associations",
        "",
        partial_correlations.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Failure-threshold sensitivity (selected features)",
        "",
        sensitivity[
            sensitivity["feature"].isin(
                (
                    "log_et_volume",
                    "et_wt_ratio",
                    "et_largest_fraction",
                    "et_components_10",
                    "log_wt_volume",
                )
            )
        ].to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Header-domain summaries",
        "",
        domains.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Kruskal-Wallis tests across header domains",
        "",
        "```json",
        json.dumps(kruskal, indent=2),
        "```",
        "",
        "## Interpretation rules",
        "",
        "- Associations are exploratory and non-causal.",
        "- t-SNE axes and apparent clusters have no direct biological meaning.",
        "- Population claims require an official case-to-population mapping.",
        "- Shape features use the reference labels and therefore characterize",
        "  failure conditions; they are not available to a deployed model.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


def write_provenance() -> None:
    summary_paths = [RESULTS / f"fold_{fold}/validation/summary.json" for fold in range(5)]
    lines = [
        "# Provenance graph and gap report",
        "",
        "Inspection timestamp: 2026-07-29 Europe/Paris",
        "",
        "## Evidence graph",
        "",
        "```text",
        "organizer-provided labels + splits_final.json",
        "  -> five nnUNetTrainer fold runs (1,000 epochs; historical Git commit unknown)",
        "  -> fold_<k>/validation/<case>.nii.gz",
        "  -> fold_<k>/validation/summary.json",
        "  -> scripts/analyze_failure_atlas.py",
        "  -> case_features.csv / statistical tables / figures",
        "  -> candidate manuscript analysis (not yet inserted)",
        "```",
        "",
        "## Verified chain",
        "",
        f"- `splits_final.json`: `{SPLITS}`, SHA-256 `{sha256(SPLITS)}`.",
        f"- Proxy mapping: `{PROXY_GROUPS}`, SHA-256 `{sha256(PROXY_GROUPS)}`.",
        "- The split contains one validation assignment per analyzed case.",
        "- Each metric row is matched to its fold-specific prediction and reference path.",
        "- Fold debug metadata identifies stock `nnUNetTrainer`, `nnUNetPlans`,",
        "  `3d_fullres`, PyTorch 2.6.0+cu124, and 1,000 epochs.",
    ]
    for path in summary_paths:
        lines.append(f"- `{path}`: SHA-256 `{sha256(path)}`.")
    lines += [
        "",
        "## Conflicts and alternate variants",
        "",
        "- Residual-encoder and experimental GoAT summaries also exist but are not",
        "  inputs to failure_atlas_v1.",
        "- `cohort_map.json` uses tumor-like names, but its own metadata states that",
        "  assignments are inferred from qform/sform signatures and are not official.",
        "",
        "## Missing links",
        "",
        "- Historical Git commit for the five training runs is not recorded.",
        "- Official case-to-population mapping is unavailable locally.",
        "- Training scheduler job IDs are not embedded in the summary files.",
        "",
        "## Reproduction readiness",
        "",
        "The analysis is reproducible from the current frozen prediction, reference,",
        "split, and mapping files. The historical model-training run is not fully",
        "reproducible because its exact Git state is unknown.",
        "",
        "## Next bounded checks",
        "",
        "- Request the official case-to-population mapping from the organizers.",
        "- Repeat the analysis with official groups before making population claims.",
        "- Confirm any manuscript claim against `case_features.csv` and the saved",
        "  statistical tables.",
    ]
    (OUTPUT / "provenance.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output root. The command refuses an existing path.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> None:
    global OUTPUT
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    OUTPUT = args.output.expanduser().resolve()
    approved_parent = (ROOT / "work/analysis").resolve()
    if OUTPUT.parent != approved_parent:
        raise RuntimeError(f"output must be a direct child of {approved_parent}: {OUTPUT}")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing output root: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    metrics = load_metrics()
    frame = extract_features(metrics, workers=args.workers)
    frame["mean_dice"] = frame[["wt_dice", "tc_dice", "et_dice"]].mean(
        axis=1, skipna=False
    )
    frame["failure_eligible"] = frame["mean_dice"].notna()
    threshold = frame["mean_dice"].quantile(0.10)
    frame["failure_bottom10"] = frame["mean_dice"] <= threshold
    frame["failure_bottom05"] = frame["mean_dice"] <= frame["mean_dice"].quantile(0.05)
    frame["failure_bottom20"] = frame["mean_dice"] <= frame["mean_dice"].quantile(0.20)
    frame["failure_et_lt_05"] = frame["et_dice"] < 0.5
    frame["failure_wt_lt_08"] = frame["wt_dice"] < 0.8
    frame.sort_values("case").to_csv(OUTPUT / "case_features.csv", index=False)

    domains = summarize_domains(frame, seed=args.seed)
    correlations, effects = association_tables(frame)
    partial_correlations = partial_spearman_table(frame)
    sensitivity = failure_sensitivity_table(frame)
    domains.to_csv(OUTPUT / "header_domain_summary.csv", index=False)
    correlations.to_csv(OUTPUT / "spearman_associations.csv", index=False)
    effects.to_csv(OUTPUT / "failure_effects.csv", index=False)
    partial_correlations.to_csv(OUTPUT / "partial_spearman_associations.csv", index=False)
    sensitivity.to_csv(OUTPUT / "failure_threshold_sensitivity.csv", index=False)

    embedded, explained = build_embedding(frame, seed=args.seed)
    embedded.to_csv(OUTPUT / "morphology_embedding.csv", index=False)
    plot_embedding(embedded, explained)
    plot_associations(frame, effects)
    write_report(
        frame,
        domains,
        correlations,
        effects,
        partial_correlations,
        sensitivity,
        explained,
    )
    write_provenance()

    metadata = {
        "analysis": "failure_atlas",
        "created": "2026-07-29",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "random_seed": args.seed,
        "workers": args.workers,
        "command": [sys.executable, *sys.argv],
        "repository": {
            "path": str(ROOT),
            "git_commit": os.environ.get("BRATS_ANALYSIS_GIT_COMMIT"),
            "dirty_at_launch": os.environ.get("BRATS_ANALYSIS_GIT_DIRTY"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "slurm_script_sha256": os.environ.get("BRATS_ANALYSIS_SLURM_SCRIPT_SHA256"),
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_ACCOUNT",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURMD_NODENAME",
            )
        },
        "n_cases": len(frame),
        "failure_definition": {
            "primary": "mean WT/TC/ET Dice <= empirical 10th percentile",
            "threshold": float(threshold),
        },
        "warning": "Header domains are acquisition proxies, not official populations.",
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Analysis complete: {OUTPUT}")


if __name__ == "__main__":
    main()

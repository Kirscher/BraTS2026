#!/usr/bin/env python3
"""Build the paper figure and fold-stratified failure-analysis intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def partial_spearman(
    frame: pd.DataFrame, feature: str, outcome: str, controls: tuple[str, ...]
) -> float:
    ranked = frame[[feature, outcome, *controls]].dropna().rank(method="average")
    design = np.column_stack(
        [np.ones(len(ranked)), ranked[list(controls)].to_numpy()]
    )
    x = ranked[feature].to_numpy()
    y = ranked[outcome].to_numpy()
    x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(stats.pearsonr(x, y).statistic)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    frame = pd.read_csv(args.features)
    frame["mean_dice"] = frame[["wt_dice", "tc_dice", "et_dice"]].mean(
        axis=1, skipna=False
    )

    hypotheses = (
        ("ET volume", "log_et_volume", "mean_dice", ("log_wt_volume",)),
        (
            "ET/WT ratio",
            "et_wt_ratio",
            "mean_dice",
            ("log_et_volume", "log_wt_volume"),
        ),
        (
            "Largest ET fraction",
            "et_largest_fraction",
            "mean_dice",
            ("log_et_volume", "log_wt_volume"),
        ),
        (
            "ET components",
            "et_components_10",
            "mean_dice",
            ("log_et_volume", "log_wt_volume"),
        ),
    )
    point_estimates = np.array(
        [
            partial_spearman(frame, feature, outcome, controls)
            for _, feature, outcome, controls in hypotheses
        ]
    )
    rng = np.random.default_rng(args.seed)
    fold_indices = [
        indices.to_numpy()
        for _, indices in frame.groupby("fold", observed=True).groups.items()
    ]
    bootstrap = np.empty((args.bootstrap_replicates, len(hypotheses)), dtype=float)
    for replicate in range(args.bootstrap_replicates):
        sampled_indices = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in fold_indices]
        )
        sampled = frame.loc[sampled_indices]
        for column, (_, feature, outcome, controls) in enumerate(hypotheses):
            bootstrap[replicate, column] = partial_spearman(
                sampled, feature, outcome, controls
            )
    lows, highs = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    estimates = [
        (hypothesis[0], point_estimates[index], lows[index], highs[index])
        for index, hypothesis in enumerate(hypotheses)
    ]

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for index, (label, feature, outcome, controls) in enumerate(hypotheses):
            records.append(
                {
                    "label": label,
                    "feature": feature,
                    "outcome": outcome,
                    "controls": list(controls),
                    "partial_rho": float(point_estimates[index]),
                    "ci_low": float(lows[index]),
                    "ci_high": float(highs[index]),
                }
            )
        args.stats_output.write_text(
            json.dumps(
                {
                    "input": args.features.name,
                    "seed": args.seed,
                    "bootstrap": {
                        "method": "percentile bootstrap stratified by OOF fold",
                        "replicates": args.bootstrap_replicates,
                        "scope": "case-sampling uncertainty only",
                    },
                    "estimates": records,
                },
                indent=2,
            )
            + "\n"
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": (1.05, 1)}
    )
    valid = frame.loc[frame["et_volume"].gt(0), ["et_volume", "et_dice"]].dropna()
    valid = valid.assign(log_et_volume_ml=np.log10(valid["et_volume"] / 1000.0))
    axes[0].scatter(
        valid["log_et_volume_ml"],
        valid["et_dice"],
        s=5,
        alpha=0.18,
        color="#D55E00",
        linewidths=0,
    )
    bins = pd.qcut(valid["log_et_volume_ml"], 12, duplicates="drop")
    grouped = valid.assign(_bin=bins).groupby("_bin", observed=True)
    axes[0].plot(
        grouped["log_et_volume_ml"].median(),
        grouped["et_dice"].median(),
        color="black",
        marker="o",
        markersize=2.8,
        linewidth=1.2,
    )
    axes[0].set(
        xlabel=r"$\log_{10}(\mathrm{ET\ volume\ [mL]})$",
        ylabel="ET Dice",
        title="a  Smaller ET has lower Dice",
        ylim=(-0.03, 1.03),
    )

    labels = [item[0] for item in estimates]
    values = np.array([item[1] for item in estimates])
    lows = np.array([item[2] for item in estimates])
    highs = np.array([item[3] for item in estimates])
    positions = np.arange(len(labels))[::-1]
    colors = np.where(values >= 0, "#0072B2", "#D55E00")
    for value, low, high, position, color in zip(
        values, lows, highs, positions, colors, strict=True
    ):
        axes[1].errorbar(
            value,
            position,
            xerr=np.array([[value - low], [high - value]]),
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=2.2,
        )
    axes[1].scatter(values, positions, c=colors, s=26, zorder=3)
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set(
        yticks=positions,
        yticklabels=labels,
        xlabel=r"Partial Spearman $\rho$ (95% CI)",
        title="b  Size-adjusted associations",
        xlim=(-0.4, 0.6),
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(8.5)
        axis.yaxis.label.set_size(8.5)
        axis.title.set_size(9.5)
    fig.tight_layout(pad=0.7, w_pad=1.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if args.output.suffix.lower() == ".pdf":
        fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

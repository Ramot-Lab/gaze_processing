"""
Per-feature mean-vs-SD regression plots from a saved participant_features.csv (one row
per participant; each feature's per-participant mean and SD across its trials/panels
as <feature>_mean / <feature>_std). One scatter per feature: participant mean (x) vs
participant SD (y), colored by group, with the pooled OLS fit + 95% CI and Pearson r -
shows how strongly a feature's variability scales with its level (a tight positive
relationship means the SD mostly re-encodes the mean). Same participant exclusions as
feature_pipeline's correlation analysis.

Usage:
    python plot_mean_vs_sd.py [participant_features.csv] [out_dir]
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, t as t_dist

import pipeline_config as pc
from feature_pipeline import EXCLUDED_PARTICIPANTS, demographics_text, annotate_demographics

# Defaults (decision 2026-11-11): the newest consolidated run's participant table, with the
# plots written next to it - resolved at call time, not a hard-coded run date.

GROUP_COLORS = {"HC": "tab:blue", "MS": "tab:orange", "pwMS": "tab:orange"}

# Columns that don't follow the <feature>_mean / <feature>_std naming.
EXTRA_PAIRS = {"typicality": ("mean_typicality", "typicality_std")}


def find_mean_sd_pairs(columns):
    pairs = {c[:-len("_mean")]: (c, c[:-len("_mean")] + "_std")
             for c in columns if c.endswith("_mean") and c[:-len("_mean")] + "_std" in columns}
    pairs.update({k: v for k, v in EXTRA_PAIRS.items() if set(v) <= set(columns)})
    return pairs


def plot_pair(df, feature, mean_col, sd_col, out_path):
    d = df[[mean_col, sd_col, "group", "Gender"]].dropna(subset=[mean_col, sd_col])
    if len(d) < 3 or d[mean_col].std() == 0:
        print(f"Skipping {feature}: not enough valid data (N={len(d)}).")
        return None
    x, y = d[mean_col].values, d[sd_col].values
    fit = linregress(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    for group, g in d.groupby("group"):
        ax.scatter(g[mean_col], g[sd_col], s=40, alpha=0.75, edgecolors="black", linewidths=0.5,
                   color=GROUP_COLORS.get(group, "gray"), label=group)

    # Pooled OLS fit with 95% confidence band for the mean prediction.
    xs = np.linspace(x.min(), x.max(), 200)
    n = len(x)
    resid_se = np.sqrt(np.sum((y - (fit.intercept + fit.slope * x)) ** 2) / (n - 2))
    se_fit = resid_se * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    tcrit = t_dist.ppf(0.975, n - 2)
    ys = fit.intercept + fit.slope * xs
    ax.plot(xs, ys, "k-", lw=1.5)
    ax.fill_between(xs, ys - tcrit * se_fit, ys + tcrit * se_fit, color="gray", alpha=0.25)

    stars = "***" if fit.pvalue < 0.001 else "**" if fit.pvalue < 0.01 else "*" if fit.pvalue < 0.05 else "ns"
    ax.set_title(f"{feature}: mean vs SD\nr = {fit.rvalue:.3f} ({stars}), p = {fit.pvalue:.2g}, "
                 f"R² = {fit.rvalue ** 2:.2f}, slope = {fit.slope:.3g}")
    ax.set_xlabel(f"{mean_col} (participant mean)")
    ax.set_ylabel(f"{sd_col} (participant SD)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    annotate_demographics(demographics_text(d), ax=ax, loc="lower left")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"feature": feature, "mean_col": mean_col, "sd_col": sd_col, "N": n,
            "r": fit.rvalue, "p_value": fit.pvalue, "R2": fit.rvalue ** 2,
            "slope": fit.slope, "intercept": fit.intercept}


def main(features_csv=None, out_dir=None):
    if features_csv is None:
        features_csv = os.path.join(pc.latest_feature_tables_dir(), "participant_features.csv")
    out_dir = out_dir or os.path.join(os.path.dirname(features_csv), "mean_vs_sd")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(features_csv)
    df = df[~df["participant"].isin(EXCLUDED_PARTICIPANTS)]
    pairs = find_mean_sd_pairs(df.columns)
    print(f"Read {features_csv}: {len(df)} participants, {len(pairs)} mean/SD feature pairs.")

    rows = []
    for feature, (mean_col, sd_col) in sorted(pairs.items()):
        row = plot_pair(df, feature, mean_col, sd_col, os.path.join(out_dir, f"{feature}_mean_vs_sd.png"))
        if row:
            rows.append(row)
    summary = pd.DataFrame(rows).sort_values("R2", ascending=False)
    summary.to_csv(os.path.join(out_dir, "mean_vs_sd_regression_summary.csv"), index=False)
    print(summary[["feature", "N", "r", "p_value", "R2", "slope"]].to_string(index=False))
    print(f"Saved {len(rows)} plots + summary to {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

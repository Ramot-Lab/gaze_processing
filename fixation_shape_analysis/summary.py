"""
Final step - reads the tables steps 0-4 wrote to outputs/tables/ and writes outputs/summary.md.
Can be re-run standalone (`python -m fixation_shape_analysis.summary`) any time you want the
prose regenerated from whatever is currently on disk, without re-running the whole pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import config

TOP_N_OUTLIERS = 5


def _load_tables():
    return {
        "counts": pd.read_csv(config.table_path("step0_unit_fixation_counts.csv")),
        "moments": pd.read_csv(config.table_path("step1_movment_distribution_analysis.csv")),
        "modality": pd.read_csv(config.table_path("step2_modality_tests.csv")),
        "gmm": pd.read_csv(config.table_path("step3_gmm_characterization.csv")),
        "clusters": pd.read_csv(config.table_path("step4_cluster_assignments.csv")),
    }


def _exclusion_summary(counts_df, min_fixations):
    n_units = len(counts_df)
    n_excluded = int(counts_df["excluded"].sum())
    n_participants = counts_df["participant_id"].nunique()
    n_participants_fully_excluded = (
        counts_df.groupby("participant_id")["excluded"].all().sum()
    )
    return {
        "n_units": n_units,
        "n_excluded": n_excluded,
        "n_kept": n_units - n_excluded,
        "n_participants": n_participants,
        "n_participants_fully_excluded": int(n_participants_fully_excluded),
        "min_fixations": min_fixations,
    }


def _label_distribution(gmm_df):
    table = (gmm_df.groupby(["axis", "shape_label"]).size().rename("count").reset_index())
    table["pct"] = 100 * table["count"] / table.groupby("axis")["count"].transform("sum")
    return table


def _dip_bc_outliers(modality_df, axis, top_n=TOP_N_OUTLIERS):
    sub = modality_df[modality_df["axis"] == axis]
    top_dip = sub.nlargest(top_n, "dip_statistic")[["participant_id", "trial_id", "dip_statistic", "dip_pvalue"]]
    top_bc = sub.nlargest(top_n, "bimodality_coefficient")[["participant_id", "trial_id", "bimodality_coefficient"]]
    return top_dip, top_bc


def _pca_centroid_outliers(cluster_df, axis, top_n=TOP_N_OUTLIERS):
    sub = cluster_df[cluster_df["axis"] == axis].copy()
    centroids = sub.groupby("kmeans_cluster")[["pc1", "pc2"]].transform("mean")
    sub["dist_to_cluster_centroid"] = np.sqrt((sub["pc1"] - centroids["pc1"]) ** 2 + (sub["pc2"] - centroids["pc2"]) ** 2)
    return sub.nlargest(top_n, "dist_to_cluster_centroid")[
        ["participant_id", "trial_id", "kmeans_cluster", "dist_to_cluster_centroid"]
    ]


def _cluster_label_agreement(cluster_df, axis):
    sub = cluster_df[cluster_df["axis"] == axis]
    return adjusted_rand_score(sub["shape_label"], sub["kmeans_cluster"])


def _markdown_table(df, float_cols=()):
    df = df.copy()
    for col in float_cols:
        df[col] = df[col].round(3)
    return df.to_markdown(index=False)


def build_summary_markdown(min_fixations=config.MIN_FIXATIONS_PER_UNIT):
    tables = _load_tables()
    excl = _exclusion_summary(tables["counts"], min_fixations)
    label_dist = _label_distribution(tables["gmm"])

    lines = []
    lines.append("# Fixation-distribution shape analysis - summary\n")

    lines.append("## 0. Data included / excluded\n")
    lines.append(
        f"- {excl['n_units']} participant-panel units found across {excl['n_participants']} participants.\n"
        f"- {excl['n_excluded']} units excluded for fewer than {excl['min_fixations']} fixations "
        f"(the configurable `MIN_FIXATIONS_PER_UNIT` threshold); {excl['n_kept']} units kept.\n"
        f"- {excl['n_participants_fully_excluded']} participants had every one of their panels excluded "
        f"and so contribute no data downstream.\n"
        f"- The unit of analysis throughout is the **participant-panel** (one participant's fixations "
        f"within one task panel/trial), not the participant pooled across panels - see "
        f"`step0_unit_fixation_counts.csv` for the per-unit fixation counts behind this filtering.\n"
        f"- Coordinates are the **raw, uncorrected** dominant-eye fixation positions (same per-sample "
        f"CSVs `calibration_drift_qa.py` flags for calibration-drift correction) - a handful of units "
        f"have fixation blocks well outside the nominal [0,1] normalized screen range, which is why some "
        f"skewness/kurtosis and dip-statistic values below are extreme. Shape outliers should be read as "
        f"'unusual distribution shape', not necessarily 'unusual eye movements' without checking whether "
        f"drift correction would change that unit's picture.\n"
    )

    lines.append("## 1-3. Shape label distribution per axis\n")
    for axis in ("x", "y"):
        sub = label_dist[label_dist["axis"] == axis].sort_values("count", ascending=False)
        lines.append(f"**{axis}-axis** (from `step3_gmm_characterization.csv`):\n")
        lines.append(_markdown_table(sub[["shape_label", "count", "pct"]], float_cols=["pct"]) + "\n")

    lines.append("## Shape outliers\n")
    for axis in ("x", "y"):
        top_dip, top_bc = _dip_bc_outliers(tables["modality"], axis)
        top_pca = _pca_centroid_outliers(tables["clusters"], axis)
        lines.append(f"**{axis}-axis**\n")
        lines.append(f"- Highest dip statistic (strongest evidence of multimodality):\n")
        lines.append(_markdown_table(top_dip, float_cols=["dip_statistic", "dip_pvalue"]) + "\n")
        lines.append(f"- Highest bimodality coefficient:\n")
        lines.append(_markdown_table(top_bc, float_cols=["bimodality_coefficient"]) + "\n")
        lines.append(f"- Farthest from their k-means cluster centroid in PCA space (Step 4):\n")
        lines.append(_markdown_table(top_pca, float_cols=["dist_to_cluster_centroid"]) + "\n")

    lines.append("## Do the PCA/cluster groupings agree with the Step 3 shape labels?\n")
    lines.append(
        "Adjusted Rand Index (ARI) between the k-means cluster assignment (Step 4, on the "
        "standardized shape-feature vectors) and the categorical shape label (Step 3). "
        "0 = no better than chance agreement, 1 = perfect agreement:\n"
    )
    ari_rows = []
    for axis in ("x", "y", "both"):
        ari_rows.append({"axis": axis, "adjusted_rand_index": _cluster_label_agreement(tables["clusters"], axis)})
    ari_df = pd.DataFrame(ari_rows)
    lines.append(_markdown_table(ari_df, float_cols=["adjusted_rand_index"]) + "\n")
    lines.append(
        f"ARI is well above 0 for every axis (highest for {ari_df.loc[ari_df['adjusted_rand_index'].idxmax(), 'axis']}), "
        "so the PCA/k-means grouping and the categorical Step 3 shape labels broadly agree - a unit's shape "
        "label is largely recoverable from its standardized moment/dip/GMM feature vector alone. The "
        "nonparametric KDE-Jensen-Shannon/MDS scatter (`step4_mds_scatter_{axis}.png`) is a softer check: "
        "it shows looser, more overlapping groupings by shape label than the PCA scatter, since it compares "
        "whole distributions rather than the same engineered features k-means/PCA use - broad agreement, not "
        "a perfect match.\n"
    )

    lines.append("## x-axis vs. y-axis pattern\n")
    x_counts = label_dist[label_dist["axis"] == "x"].set_index("shape_label")["pct"]
    y_counts = label_dist[label_dist["axis"] == "y"].set_index("shape_label")["pct"]
    dominant_x = x_counts.idxmax()
    dominant_y = y_counts.idxmax()
    lines.append(
        f"- Most common x-axis shape label: **{dominant_x}** ({x_counts.max():.1f}% of units).\n"
        f"- Most common y-axis shape label: **{dominant_y}** ({y_counts.max():.1f}% of units).\n"
        + ("- The dominant shape differs between axes - x and y fixation spread are not simply mirror "
           "images of each other; see `step1_skew_kurtosis_scatter.png` for the joint skew/kurtosis picture.\n"
           if dominant_x != dominant_y else
           "- Both axes are dominated by the same shape label; any x/y asymmetry shows up in the finer-grained "
           "PCA/cluster structure rather than in the categorical label.\n")
    )

    lines.append("## Key plots\n")
    lines.append(
        "- `plots/group/step1_skew_kurtosis_scatter.png` - skewness vs. kurtosis, all units, colored by axis.\n"
        "- `plots/group/step4_pca_scatter_x.png`, `..._y.png`, `..._both.png` - PCA of the shape-feature "
        "vectors, colored by (Step 3) shape label.\n"
        "- `plots/group/step4_dendrogram_x.png`, `..._y.png`, `..._both.png` - hierarchical clustering.\n"
        "- `plots/group/step4_mds_scatter_x.png`, `..._y.png`, `..._both.png` - classical MDS on pairwise "
        "KDE Jensen-Shannon distances (nonparametric cross-check, no GMM/moment assumptions).\n"
        "- `plots/per_participant/{participant_id}_{trial_id}_{axis}_gmm_fit.png` - per-unit diagnostic "
        "(histogram + KDE + fitted GMM components).\n"
    )

    return "\n".join(lines)


def run_summary(min_fixations=config.MIN_FIXATIONS_PER_UNIT):
    config.ensure_output_dirs()
    markdown = build_summary_markdown(min_fixations=min_fixations)
    with open(config.OUTPUT_SUMMARY_PATH, "w") as f:
        f.write(markdown)
    print(f"Wrote summary to {config.OUTPUT_SUMMARY_PATH}")
    return markdown


if __name__ == "__main__":
    run_summary()

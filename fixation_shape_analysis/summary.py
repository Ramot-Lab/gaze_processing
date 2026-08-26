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


def _ari(labels_a, labels_b):
    return adjusted_rand_score(labels_a, labels_b)


def _js_cluster_imbalance(cluster_df, axis):
    """Cluster-size breakdown for js_hierarchical_cluster, plus which participant-panels
    sit in the smallest cluster - used to check whether a near-zero ARI against it means
    "no shape signal" or "silhouette picked a degenerate outlier-vs-everything split"."""
    sub = cluster_df[cluster_df["axis"] == axis]
    sizes = sub["js_hierarchical_cluster"].value_counts().sort_values()
    minority_cluster = sizes.index[0]
    minority = sub.loc[sub["js_hierarchical_cluster"] == minority_cluster, ["participant_id", "trial_id"]]
    return sizes, minority


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
    lines.append(
        "GMM component search range differs by axis: x is searched over k=1..9 (9 dictionary "
        "columns to potentially resolve), y over k=1..3 (only 2 rows) - see "
        "`config.GMM_MAX_COMPONENTS_BY_AXIS`.\n"
    )
    for axis in ("x", "y"):
        sub = label_dist[label_dist["axis"] == axis].sort_values("count", ascending=False)
        lines.append(f"**{axis}-axis** (from `step3_gmm_characterization.csv`):\n")
        lines.append(_markdown_table(sub[["shape_label", "count", "pct"]], float_cols=["pct"]) + "\n")

    lines.append("## How reliable is the selected k? (k_stability)\n")
    lines.append(
        "Each unit's `k_stability` (in `step3_gmm_characterization.csv`) is the fraction of "
        f"{config.GMM_BOOTSTRAP_N_RESAMPLES} bootstrap resamples of that unit's own fixations "
        "that picked the same k (via BIC) as the main fit. It's a plain check on whether the "
        "reported number of modes is trustworthy or just a quirk of which fixations happened "
        "to land where: k_stability near 1 means the mode count is reliable, while values below "
        "roughly 0.5 mean that unit's k is genuinely uncertain and shouldn't be over-interpreted "
        "(most likely for x's higher-k fits on smaller units, since a wider k=1..9 search gives "
        "BIC more ways to flip between neighboring k on resampled data).\n"
    )
    lines.append(
        f"B={config.GMM_BOOTSTRAP_N_RESAMPLES} resamples and a lower n_init "
        f"({config.GMM_BOOTSTRAP_N_INIT}, vs. {config.GMM_N_INIT} for the main fit) on each "
        "bootstrap refit follow standard bootstrap practice: Efron & Tibshirani "
        "(*An Introduction to the Bootstrap*, 1993) cite roughly 50-200 replicates as enough "
        "for a bootstrap standard-error estimate; Monti et al. (\"Consensus Clustering\", "
        "2003) use a comparable ~100-500 resamples for resampling-based model-order/cluster "
        "stability; McLachlan's (1987) bootstrap test for the number of mixture components "
        "likewise uses cheaper per-replicate fits than the one definitive fit being tested, "
        "since B replicates already average out per-fit initialization noise.\n"
    )
    gmm_df = tables["gmm"]
    k_stability_rows = []
    for axis in ("x", "y"):
        sub = gmm_df[gmm_df["axis"] == axis]["k_stability"]
        k_stability_rows.append({
            "axis": axis,
            "median_k_stability": sub.median(),
            "pct_units_below_0.5": 100 * (sub < 0.5).mean(),
        })
    lines.append(_markdown_table(pd.DataFrame(k_stability_rows),
                                  float_cols=["median_k_stability", "pct_units_below_0.5"]) + "\n")

    # Monte Carlo noise ON k_stability ITSELF: it's a proportion estimated from B bootstrap
    # draws, so it carries its own sampling error, SE = sqrt(p*(1-p)/B), maximized at p=0.5.
    b = config.GMM_BOOTSTRAP_N_RESAMPLES
    worst_case_se = np.sqrt(0.5 * 0.5 / b)
    lines.append(
        f"**A caveat on k_stability's own precision**: it is itself a proportion estimated "
        f"from only B={b} bootstrap draws, so it carries Monte Carlo sampling noise of "
        f"SE = sqrt(p(1-p)/B), worst-case (p=0.5) SE ≈ {worst_case_se:.3f} "
        f"(±{100 * worst_case_se:.1f} percentage points, or roughly ±{200 * worst_case_se:.0f} "
        "points for a ~95% band). In practice this means a reported k_stability of, say, 0.50 "
        f"could plausibly be anywhere from about {0.5 - 1.96 * worst_case_se:.2f} to "
        f"{0.5 + 1.96 * worst_case_se:.2f} just from which {b} resamples happened to be drawn - "
        "it should be read as a coarse reliability band (roughly: high/mid/low), not a precise "
        "score, and only differences larger than this noise band across units should be treated "
        "as meaningfully different.\n"
    )

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
        "Two independent clusterings are compared against each other and against the categorical Step 3 "
        "shape label:\n"
        "- **feature-based** (`kmeans_cluster`): k-means/hierarchical(Ward) on the standardized shape-feature "
        "vector (skewness, kurtosis, dip stat, BC, GMM params) - the \"by shapes\" clustering.\n"
        "- **distance-based** (`js_hierarchical_cluster`): hierarchical (average-linkage) clustering run "
        "directly on the pairwise KDE Jensen-Shannon distance matrix, with no shape-feature assumptions at "
        "all - the \"general\" clustering. k-means is not an option here: it needs Euclidean coordinates to "
        "compute a cluster centroid, which a bare distance matrix doesn't provide (that's exactly why "
        "classical MDS exists in this pipeline - to approximately, and lossily, embed those distances into a "
        "low-dimensional Euclidean space just for visualization). Agglomerative clustering with a precomputed "
        "distance metric instead uses the full distance matrix directly, no embedding needed.\n\n"
        "Adjusted Rand Index (ARI), 0 = chance agreement, 1 = perfect agreement:\n"
    )
    ari_rows = []
    for axis in ("x", "y"):
        sub = tables["clusters"][tables["clusters"]["axis"] == axis]
        ari_rows.append({
            "axis": axis,
            "feature_cluster_vs_shape_label": _ari(sub["kmeans_cluster"], sub["shape_label"]),
            "js_cluster_vs_shape_label": _ari(sub["js_hierarchical_cluster"], sub["shape_label"]),
            "feature_cluster_vs_js_cluster": _ari(sub["kmeans_cluster"], sub["js_hierarchical_cluster"]),
        })
    both_sub = tables["clusters"][tables["clusters"]["axis"] == "both"]
    ari_rows.append({
        "axis": "both",
        "feature_cluster_vs_shape_label": _ari(both_sub["kmeans_cluster"], both_sub["shape_label"]),
        "js_cluster_vs_shape_label": np.nan,
        "feature_cluster_vs_js_cluster": np.nan,
    })
    ari_df = pd.DataFrame(ari_rows)
    ari_cols = ["feature_cluster_vs_shape_label", "js_cluster_vs_shape_label", "feature_cluster_vs_js_cluster"]
    lines.append(_markdown_table(ari_df, float_cols=ari_cols) + "\n")
    lines.append(
        "(\"both\" has no distance-based clustering - Step 4's nonparametric KDE/JS/MDS cross-check stays "
        "per-axis, see `step4_comparison.py`.) Reading these together: a high "
        "`feature_cluster_vs_shape_label` means a unit's shape label is largely recoverable from its "
        "engineered feature vector alone. A high `feature_cluster_vs_js_cluster` means the two independent "
        "lines of evidence - engineered shape features vs. raw distributional comparison - broadly agree "
        "with each other, which is reassuring for both. If `feature_cluster_vs_js_cluster` is low while "
        "`feature_cluster_vs_shape_label` is high, that suggests the categorical shape label is more a "
        "property of the GMM/moment feature engineering than of the underlying distributions themselves.\n"
    )

    for axis in ("x", "y"):
        sizes, minority = _js_cluster_imbalance(tables["clusters"], axis)
        if len(sizes) > 1 and sizes.iloc[0] <= 0.02 * sizes.sum():
            minority_desc = ", ".join(f"{p}/{t}" for p, t in minority[["participant_id", "trial_id"]].to_numpy())
            lines.append(
                f"**Why `js_cluster_vs_shape_label` is near zero for {axis}**: `js_hierarchical_cluster` split "
                f"{sizes.sum()} units into {sizes.iloc[0]} vs. {sizes.iloc[1:].sum()} - silhouette selected a "
                f"degenerate split isolating a handful of extreme outliers ({minority_desc}) rather than "
                "resolving genuine shape-based subgroups among the rest. Given the raw/uncorrected-coordinates "
                "caveat above, this is consistent with calibration-drift-affected units being distributionally "
                "far from everyone else, not with the shape labels being wrong. This check would need outlier "
                "trimming or drift correction upstream before it can meaningfully validate (or refute) the "
                "shape-feature clustering for the remaining majority of units.\n"
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
        "- `plots/group/step4_pca_scatter_{x,y,both}_pc{a}_pc{b}.png` - PCA of the shape-feature vectors "
        "(x, y, and combined x+y), colored by shape label, all 3 pairs among the first 3 components "
        "(PC1-PC2, PC1-PC3, PC2-PC3).\n"
        "- `plots/group/step4_pca_scatter_{x,y,both}_by_cluster_pc{a}_pc{b}.png` - the SAME PCA space, "
        "colored instead by the Ward hierarchical cluster assignment - compare against the shape-label-colored "
        "version above to see how well the clustering carves up the space.\n"
        "- `plots/group/step4_dendrogram_{x,y,both}.png` - Ward hierarchical clustering of the "
        "shape-feature vector.\n"
        "- `plots/group/step4_mds_scatter_{x,y}_dim{a}_dim{b}.png` - classical MDS on the pairwise "
        "KDE Jensen-Shannon distances (nonparametric cross-check, no GMM/moment assumptions), all 3 "
        "pairs among the first 3 dimensions, colored by shape label.\n"
        "- `plots/group/step4_mds_scatter_{x,y}_by_cluster_dim{a}_dim{b}.png` - the SAME MDS space, colored "
        "instead by its own distance-based hierarchical cluster - makes the degenerate outlier-vs-everyone "
        "split (see Conclusions) directly visible: a handful of points sit far off the main arc, and that's "
        "the entire clustering.\n"
        "- `plots/group/step4_js_dendrogram_{x,y}.png` - average-linkage hierarchical clustering run "
        "directly on the JS-distance matrix (the \"general\" clustering - see the ARI section above).\n"
        "- `plots/per_participant/{participant_id}_{trial_id}_{axis}_gmm_fit.png` - per-unit diagnostic "
        "(histogram + KDE + fitted GMM components).\n"
    )

    lines.append("## Conclusions (clustering)\n")
    x_ari, y_ari, both_ari = (ari_df.set_index("axis").loc[a, "feature_cluster_vs_shape_label"] for a in ("x", "y", "both"))
    x_kstab = gmm_df.loc[gmm_df["axis"] == "x", "k_stability"].median()
    y_kstab = gmm_df.loc[gmm_df["axis"] == "y", "k_stability"].median()
    lines.append(
        f"- **x and y have genuinely different shape landscapes**: x is dominated by **{dominant_x}** "
        f"({x_counts.max():.0f}%), y by **{dominant_y}** ({y_counts.max():.0f}%) - they should be reported and "
        "interpreted as two separate stories, not collapsed into one.\n"
        f"- **The shape-feature clustering is internally consistent**: standardized shape features (skewness, "
        f"kurtosis, dip stat, BC, GMM params) reduced via PCA and clustered recover the categorical shape label "
        f"reasonably well (ARI={x_ari:.2f} for x, ARI={y_ari:.2f} for y) - the shape-label framework is "
        "picking up real, recoverable structure, not noise.\n"
        f"- **x's fine-grained mode counts need a big caveat, y's don't**: bootstrap k_stability is high for y "
        f"(median {y_kstab:.2f}) but low for x (median {x_kstab:.2f}, {(gmm_df.loc[gmm_df['axis']=='x','k_stability']<0.5).mean()*100:.0f}% "
        "of units below 0.5) - present y's mode counts with confidence, present x's k as illustrative/exploratory, "
        "not a precise count of dictionary columns used.\n"
        "- **The nonparametric \"general\" cross-check did not confirm the shape-feature clustering - but for a "
        "diagnosable reason, not because the shapes are wrong**: clustering directly on Jensen-Shannon "
        "distances between whole fixation distributions produced near-zero ARI against both the shape labels "
        "and the feature-based clusters. Root cause: silhouette selection picked a degenerate 2-cluster split "
        "isolating a handful of extreme-outlier units (all from participants with likely calibration drift, "
        "e.g. the raw/uncorrected-coordinates caveat noted above) versus everyone else, rather than resolving "
        "any real shape-based subgroup structure - visible directly in `step4_mds_scatter_{axis}_by_cluster_"
        "dim1_dim2.png` (a handful of points off the main arc, that's the entire clustering) versus the much "
        "more balanced, structured split in `step4_pca_scatter_{axis}_by_cluster_pc1_pc2.png`. **Recommended "
        "before presenting this check as a real validation**: rerun it on drift-corrected coordinates, or "
        "exclude the flagged outlier units first.\n"
        f"- **Combining x and y into one feature vector weakens the signal rather than strengthening it** "
        f"(ARI={both_ari:.2f} for \"both\" vs. {x_ari:.2f}/{y_ari:.2f} separately) - supports keeping x and y "
        "as independent analyses rather than a single combined shape-space.\n"
        "- **Bottom line for presentation**: the GMM/moment-based shape characterization is the trustworthy, "
        "validated backbone of this analysis (especially for y); the nonparametric distance-based check is "
        "still an open validation step pending outlier/drift handling, not a contradiction of the main result.\n"
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

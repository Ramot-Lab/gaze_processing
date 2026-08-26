"""
Step 4 - between-participant(-panel) comparison of fixation-distribution shape. Three
passes: x-axis only, y-axis only, and a combined x+y pass (PCA + clustering on
concatenated per-axis feature vectors - no bivariate KDE/MDS, see run_step4_both).

Each per-axis pass runs two INDEPENDENT lines of evidence, deliberately kept separate:

  1. PCA + k-means/hierarchical(Ward) clustering on the engineered shape-feature vector
     (skewness, kurtosis, dip stat, BC, GMM params from Steps 1-3). This is "by shapes":
     it can only ever be as good as that feature engineering.
  2. KDE -> Jensen-Shannon distance -> classical MDS (visualization) -> hierarchical
     (average-linkage) clustering RUN DIRECTLY ON THE PRECOMPUTED DISTANCE MATRIX. This
     compares whole empirical distributions with no shape-feature assumptions at all -
     the "general, not by shapes" check.

Why hierarchical clustering (not k-means) for line 2: k-means needs Euclidean coordinates
to compute a centroid (a cluster mean) - a distance matrix alone doesn't provide any. That
is exactly why classical MDS exists here: to approximately embed the Jensen-Shannon
distances into a low-dimensional Euclidean space, for visualization - but that embedding
is lossy (note _classical_mds clips negative eigenvalues; JS distances aren't perfectly
Euclidean-embeddable). Agglomerative clustering with metric="precomputed" instead operates
on the FULL distance matrix directly, no embedding or approximation needed, so it's the
more faithful "general" clustering to set against the shape-feature-based one. (The
k-means analog that WOULD work on raw distances is k-medoids/PAM, not used here since
hierarchical was the agreed choice - see conversation log.)

Both PCA and classical MDS are computed to 3 components and plotted pairwise (PC1-PC2,
PC1-PC3, PC2-PC3, and the same for MDS dims), not just the first two, per request.

Re-running: the normal path (via run_pipeline.py, or this file's __main__) re-fits Steps
0-3 first (~2 hours end to end). To iterate on Step 4's own methodology without paying
that cost again, use run_step4_from_csvs() (or `python -m fixation_shape_analysis.
step4_comparison --from-csvs`), which reads the already-saved Step 0-3 CSVs instead.
"""

import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import jensenshannon, squareform
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

N_DISPLAY_COMPONENTS = 3  # PCA/MDS components to plot pairwise


def _gmm_feature_cols(axis):
    """Component_1..max_components weight/mean/sd column names for this axis's own k range
    (x: up to 9, y: up to 3 - see config.GMM_MAX_COMPONENTS_BY_AXIS)."""
    cols = ["selected_k", "ashmans_d"]
    for i in range(1, config.GMM_MAX_COMPONENTS_BY_AXIS[axis] + 1):
        cols += [f"component_{i}_weight", f"component_{i}_mean", f"component_{i}_sd"]
    return cols


def build_feature_table(moments_df, modality_df, gmm_df, axis):
    """
    One row per unit_id: skewness/kurtosis (step1) + dip/BC (step2) +
    selected_k/ashmans_d/component params (step3) + shape_label, for a single axis.
    Missing component_i values (when selected_k < i) are imputed: weight -> 0 (component
    doesn't exist), mean/sd -> component_1's mean/sd (always present, a neutral filler so
    standardized values center near zero rather than injecting a spurious signal).
    """
    gmm_cols = _gmm_feature_cols(axis)
    m = moments_df[moments_df["axis"] == axis][["participant_id", "trial_id", "unit_id", "skewness", "kurtosis"]]
    d = modality_df[modality_df["axis"] == axis][["unit_id", "dip_statistic", "dip_pvalue", "bimodality_coefficient", "modality_label"]]
    g = gmm_df[gmm_df["axis"] == axis][["unit_id", "shape_label", "k_stability"] + gmm_cols]

    feature_df = m.merge(d, on="unit_id").merge(g, on="unit_id")

    for i in range(2, config.GMM_MAX_COMPONENTS_BY_AXIS[axis] + 1):
        feature_df[f"component_{i}_weight"] = feature_df[f"component_{i}_weight"].fillna(0.0)
        feature_df[f"component_{i}_mean"] = feature_df[f"component_{i}_mean"].fillna(feature_df["component_1_mean"])
        feature_df[f"component_{i}_sd"] = feature_df[f"component_{i}_sd"].fillna(feature_df["component_1_sd"])

    return feature_df.reset_index(drop=True)


def build_feature_table_both(moments_df, modality_df, gmm_df):
    """Combined x+y feature table: x_/y_-prefixed columns from build_feature_table for each
    axis, merged on unit_id. Combined shape_label is "x_label | y_label"."""
    fx = build_feature_table(moments_df, modality_df, gmm_df, "x").add_prefix("x_").rename(
        columns={"x_participant_id": "participant_id", "x_trial_id": "trial_id", "x_unit_id": "unit_id"})
    fy = build_feature_table(moments_df, modality_df, gmm_df, "y").add_prefix("y_").rename(
        columns={"y_participant_id": "participant_id", "y_trial_id": "trial_id", "y_unit_id": "unit_id"})
    feature_df = fx.merge(fy, on=["participant_id", "trial_id", "unit_id"])
    feature_df["shape_label"] = feature_df["x_shape_label"] + " | " + feature_df["y_shape_label"]
    return feature_df


def _numeric_feature_cols(axis):
    return ["skewness", "kurtosis", "dip_statistic", "bimodality_coefficient"] + _gmm_feature_cols(axis)


def _standardize(feature_df, numeric_cols):
    X = feature_df[numeric_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    return StandardScaler().fit_transform(X)


def run_pca(X, feature_names):
    pca = PCA(n_components=min(len(feature_names), X.shape[0], 10))
    scores = pca.fit_transform(X)

    loadings_rows = []
    for pc_idx in range(pca.n_components_):
        row = {"component": f"PC{pc_idx + 1}", "explained_variance_ratio": pca.explained_variance_ratio_[pc_idx]}
        for feat_idx, feat_name in enumerate(feature_names):
            row[feat_name] = pca.components_[pc_idx, feat_idx]
        loadings_rows.append(row)
    loadings_df = pd.DataFrame(loadings_rows)

    return pca, scores, loadings_df


def plot_scatter_2d(x_vals, y_vals, labels, save_path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in sorted(pd.unique(labels)):
        mask = labels == label
        ax.scatter(x_vals[mask], y_vals[mask], label=label, alpha=0.7, s=25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _cluster_label_strings(cluster_ids, prefix):
    """Integer cluster ids (0,1,2,...) -> descriptive strings ("cluster 0", ...) so the
    scatter legend reads the same way the shape_label-colored plots do."""
    return np.array([f"{prefix} {c}" for c in cluster_ids])


def plot_pairwise_components(scores, labels, dim_prefix, save_path_fn, title_fn,
                              n_components=N_DISPLAY_COMPONENTS):
    """Scatter every pair among the first n_components columns of `scores` (e.g. PC1-PC2,
    PC1-PC3, PC2-PC3) rather than just the first two - falls back gracefully if fewer than
    n_components are actually available (e.g. a small smoke-test subset)."""
    available = min(n_components, scores.shape[1])
    for a, b in itertools.combinations(range(1, available + 1), 2):
        plot_scatter_2d(scores[:, a - 1], scores[:, b - 1], labels,
                         save_path_fn(a, b), title_fn(a, b),
                         f"{dim_prefix}{a}", f"{dim_prefix}{b}")


def run_clustering(X, k_range=config.KMEANS_K_RANGE, random_state=config.CLUSTERING_RANDOM_STATE):
    """K-means: pick k by silhouette score over k_range. Hierarchical (Ward) clustering
    re-uses that same k so the two methods (both operating on the same Euclidean feature
    vector) are directly comparable."""
    best_k, best_score, best_labels = None, -np.inf, None
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    hierarchical_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    return best_k, best_score, best_labels, hierarchical_labels


def run_distance_clustering(distance_matrix, k_range=config.KMEANS_K_RANGE):
    """
    Hierarchical (average-linkage) clustering run DIRECTLY on a precomputed distance
    matrix - the "general, not by [engineered] shapes" clustering. k-means is not an
    option here: it requires Euclidean coordinates to compute a cluster centroid, which a
    bare distance matrix doesn't provide (see module docstring for the full reasoning).
    Picks k by silhouette score (also computed directly on the distance matrix).
    """
    best_k, best_score, best_labels = None, -np.inf, None
    for k in k_range:
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(distance_matrix)
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_k, best_score, best_labels


def plot_dendrogram(X, save_path, title):
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, no_labels=True, ax=ax, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_xlabel("Participant-panel units")
    ax.set_ylabel("Ward distance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_distance_dendrogram(distance_matrix, save_path, title):
    """Same idea as plot_dendrogram but for a precomputed distance matrix (average
    linkage - Ward requires true Euclidean coordinates, which this doesn't have)."""
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, no_labels=True, ax=ax, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_xlabel("Participant-panel units")
    ax.set_ylabel("Average-linkage JS distance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _classical_mds(distance_matrix, n_components=N_DISPLAY_COMPONENTS):
    """Torgerson's classical (metric) MDS via double-centering + eigendecomposition."""
    D2 = distance_matrix ** 2
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order][:n_components], eigvecs[:, order][:, :n_components]
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs * np.sqrt(eigvals)


# Raw (uncorrected) fixation coordinates have real calibration-drift outliers far outside the
# nominal [0,1] screen range (see calibration_drift_qa.py), which widens the common KDE grid a
# lot. Units with a tight KDE evaluated way out in that grid's tail underflow to exact 0.0 in
# double precision; averaging two such pdfs can then underflow too even where one side is
# nonzero, and scipy's jensenshannon (via rel_entr) reports that as an infinite divergence. A
# tiny floor before normalizing avoids the underflow without materially changing the divergence
# (those bins carry ~0 mass either way).
_KDE_PROB_FLOOR = 1e-300


def _normalize_pdf(pdf):
    pdf = np.clip(pdf, _KDE_PROB_FLOOR, None)
    return pdf / pdf.sum()


def kde_js_distance_matrix_1d(fixation_df, unit_ids, axis, grid_points=config.KDE_GRID_POINTS):
    all_values = fixation_df[fixation_df["unit_id"].isin(unit_ids)][axis].to_numpy()
    grid = np.linspace(all_values.min(), all_values.max(), grid_points)

    pdfs = np.zeros((len(unit_ids), grid_points))
    for i, unit_id in enumerate(unit_ids):
        values = fixation_df.loc[fixation_df["unit_id"] == unit_id, axis].to_numpy()
        pdfs[i] = _normalize_pdf(gaussian_kde(values)(grid))

    return _pairwise_js(pdfs)


def _pairwise_js(pdfs):
    n = pdfs.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(pdfs[i], pdfs[j])
            dist[i, j] = dist[j, i] = 0.0 if np.isnan(d) else d
    return dist


def run_step4_for_axis(fixation_df, moments_df, modality_df, gmm_df, axis):
    """Full Step 4 for a single univariate axis ('x' or 'y'): PCA + clustering on the
    shape-feature vector, and the nonparametric KDE/JS/MDS/distance-clustering cross-check."""
    config.ensure_output_dirs()

    numeric_cols = _numeric_feature_cols(axis)
    feature_df = build_feature_table(moments_df, modality_df, gmm_df, axis)
    X = _standardize(feature_df, numeric_cols)
    color_labels = feature_df["shape_label"].to_numpy()

    # --- Line 1: PCA + clustering on the engineered shape-feature vector ---
    pca, scores, loadings_df = run_pca(X, numeric_cols)
    loadings_df.insert(0, "axis", axis)
    plot_pairwise_components(
        scores, color_labels, "PC",
        save_path_fn=lambda a, b: config.group_plot_path(f"step4_pca_scatter_{axis}_pc{a}_pc{b}.png"),
        title_fn=lambda a, b: f"PCA (PC{a} vs PC{b}) of {axis}-axis shape features, colored by Step 3 shape label",
    )

    best_k, silhouette, kmeans_labels, hier_labels = run_clustering(X)
    plot_dendrogram(X, config.group_plot_path(f"step4_dendrogram_{axis}.png"),
                     f"Hierarchical clustering (Ward) of {axis}-axis fixation-shape features")
    plot_pairwise_components(
        scores, _cluster_label_strings(hier_labels, "cluster"), "PC",
        save_path_fn=lambda a, b: config.group_plot_path(f"step4_pca_scatter_{axis}_by_cluster_pc{a}_pc{b}.png"),
        title_fn=lambda a, b: f"PCA (PC{a} vs PC{b}) of {axis}-axis shape features, colored by Ward hierarchical cluster",
    )

    # --- Line 2: nonparametric KDE -> Jensen-Shannon -> classical MDS + distance clustering ---
    unit_ids = feature_df["unit_id"].tolist()
    js_dist = kde_js_distance_matrix_1d(fixation_df, unit_ids, axis)
    js_dist_df = pd.DataFrame(js_dist, index=unit_ids, columns=unit_ids)

    mds_coords = _classical_mds(js_dist)
    plot_pairwise_components(
        mds_coords, color_labels, "MDS dim ",
        save_path_fn=lambda a, b: config.group_plot_path(f"step4_mds_scatter_{axis}_dim{a}_dim{b}.png"),
        title_fn=lambda a, b: f"Classical MDS (dim{a} vs dim{b}) on {axis}-axis KDE Jensen-Shannon distances",
    )

    js_best_k, js_silhouette, js_hier_labels = run_distance_clustering(js_dist)
    plot_distance_dendrogram(
        js_dist, config.group_plot_path(f"step4_js_dendrogram_{axis}.png"),
        f"Hierarchical clustering (average linkage, precomputed JS distance) - {axis}-axis")
    plot_pairwise_components(
        mds_coords, _cluster_label_strings(js_hier_labels, "cluster"), "MDS dim ",
        save_path_fn=lambda a, b: config.group_plot_path(f"step4_mds_scatter_{axis}_by_cluster_dim{a}_dim{b}.png"),
        title_fn=lambda a, b: f"Classical MDS (dim{a} vs dim{b}) on {axis}-axis JS distances, colored by its own "
                               f"distance-based hierarchical cluster",
    )

    cluster_df = feature_df[["participant_id", "trial_id", "unit_id", "shape_label"]].copy()
    cluster_df.insert(0, "axis", axis)
    cluster_df["kmeans_cluster"] = kmeans_labels
    cluster_df["hierarchical_cluster"] = hier_labels
    cluster_df["kmeans_k"] = best_k
    cluster_df["kmeans_silhouette"] = silhouette
    cluster_df["js_hierarchical_cluster"] = js_hier_labels
    cluster_df["js_hierarchical_k"] = js_best_k
    cluster_df["js_hierarchical_silhouette"] = js_silhouette
    for i in range(min(N_DISPLAY_COMPONENTS, scores.shape[1])):
        cluster_df[f"pc{i + 1}"] = scores[:, i]
    for i in range(min(N_DISPLAY_COMPONENTS, mds_coords.shape[1])):
        cluster_df[f"mds{i + 1}"] = mds_coords[:, i]

    print(f"Step 4 ({axis}): PCA explained variance (PC1-3) = {pca.explained_variance_ratio_[:3].round(3)}, "
          f"feature-kmeans best_k={best_k} (silhouette={silhouette:.3f}), "
          f"JS-distance-hierarchical best_k={js_best_k} (silhouette={js_silhouette:.3f})")

    return {
        "feature_df": feature_df,
        "loadings_df": loadings_df,
        "cluster_df": cluster_df,
        "js_dist_df": js_dist_df,
    }


def run_step4_both(moments_df, modality_df, gmm_df):
    """
    Combined x+y pass: PCA + clustering on the concatenated per-axis feature vectors
    (x_-/y_-prefixed). No bivariate KDE/Jensen-Shannon/MDS here - that would need a
    genuinely 2D (x,y) density estimate per unit, a heavier step not requested for this
    pass; the nonparametric cross-check (Line 2 above) stays per-axis.
    """
    config.ensure_output_dirs()
    axis = "both"

    feature_df = build_feature_table_both(moments_df, modality_df, gmm_df)
    numeric_cols = [f"x_{c}" for c in _numeric_feature_cols("x")] + [f"y_{c}" for c in _numeric_feature_cols("y")]
    X = _standardize(feature_df, numeric_cols)
    color_labels = feature_df["shape_label"].to_numpy()

    pca, scores, loadings_df = run_pca(X, numeric_cols)
    loadings_df.insert(0, "axis", axis)
    plot_pairwise_components(
        scores, color_labels, "PC",
        save_path_fn=lambda a, b: config.group_plot_path(f"step4_pca_scatter_{axis}_pc{a}_pc{b}.png"),
        title_fn=lambda a, b: f"PCA (PC{a} vs PC{b}) of combined x+y shape features, colored by combined shape label",
    )

    best_k, silhouette, kmeans_labels, hier_labels = run_clustering(X)
    plot_dendrogram(X, config.group_plot_path(f"step4_dendrogram_{axis}.png"),
                     "Hierarchical clustering (Ward) of combined x+y fixation-shape features")
    plot_pairwise_components(
        scores, _cluster_label_strings(hier_labels, "cluster"), "PC",
        save_path_fn=lambda a, b: config.group_plot_path(f"step4_pca_scatter_{axis}_by_cluster_pc{a}_pc{b}.png"),
        title_fn=lambda a, b: f"PCA (PC{a} vs PC{b}) of combined x+y shape features, colored by Ward hierarchical cluster",
    )

    cluster_df = feature_df[["participant_id", "trial_id", "unit_id", "shape_label"]].copy()
    cluster_df.insert(0, "axis", axis)
    cluster_df["kmeans_cluster"] = kmeans_labels
    cluster_df["hierarchical_cluster"] = hier_labels
    cluster_df["kmeans_k"] = best_k
    cluster_df["kmeans_silhouette"] = silhouette
    for i in range(min(N_DISPLAY_COMPONENTS, scores.shape[1])):
        cluster_df[f"pc{i + 1}"] = scores[:, i]

    print(f"Step 4 (both): PCA explained variance (PC1-3) = {pca.explained_variance_ratio_[:3].round(3)}, "
          f"kmeans best_k={best_k} (silhouette={silhouette:.3f})")

    return {"feature_df": feature_df, "loadings_df": loadings_df, "cluster_df": cluster_df}


def run_step4(fixation_df, moments_df, modality_df, gmm_df):
    results_x = run_step4_for_axis(fixation_df, moments_df, modality_df, gmm_df, "x")
    results_y = run_step4_for_axis(fixation_df, moments_df, modality_df, gmm_df, "y")
    results_both = run_step4_both(moments_df, modality_df, gmm_df)

    loadings_df = pd.concat([results_x["loadings_df"], results_y["loadings_df"], results_both["loadings_df"]],
                             ignore_index=True)
    loadings_df.to_csv(config.table_path("step4_pca_loadings.csv"), index=False)

    cluster_df = pd.concat([results_x["cluster_df"], results_y["cluster_df"], results_both["cluster_df"]],
                            ignore_index=True)
    cluster_df.to_csv(config.table_path("step4_cluster_assignments.csv"), index=False)

    long_rows = []
    for axis, results in (("x", results_x), ("y", results_y)):
        js_dist_df = results["js_dist_df"]
        stacked = js_dist_df.where(np.triu(np.ones(js_dist_df.shape, dtype=bool), k=1)).stack()
        stacked = stacked.rename("js_distance").reset_index()
        stacked.columns = ["unit_id_1", "unit_id_2", "js_distance"]
        stacked.insert(0, "axis", axis)
        long_rows.append(stacked)
    pd.concat(long_rows, ignore_index=True).to_csv(config.table_path("step4_js_distance_matrix.csv"), index=False)

    return {"x": results_x, "y": results_y, "both": results_both}


def run_step4_from_csvs():
    """
    Re-runs Step 4 only, reading the CSVs Steps 0-3 already wrote to outputs/tables/
    instead of refitting them (~2 hours end to end). For iterating on Step 4's own
    methodology (PCA/clustering/MDS) without paying that cost again.
    """
    fixation_df = pd.read_csv(config.table_path("step0_fixation_data_included.csv"))
    moments_df = pd.read_csv(config.table_path("step1_movment_distribution_analysis.csv"))
    modality_df = pd.read_csv(config.table_path("step2_modality_tests.csv"))
    gmm_df = pd.read_csv(config.table_path("step3_gmm_characterization.csv"))
    return run_step4(fixation_df, moments_df, modality_df, gmm_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csvs", action="store_true",
                         help="Skip re-running Steps 0-3 (~2hr) and read their already-saved CSVs instead.")
    args = parser.parse_args()

    if args.from_csvs:
        run_step4_from_csvs()
    else:
        from .step0_load_data import run_step0
        from .step1_moments import run_step1
        from .step2_modality import run_step2
        from .step3_gmm import run_step3
        included_df, _ = run_step0()
        moments_df = run_step1(included_df)
        modality_df = run_step2(included_df, moments_df)
        gmm_df = run_step3(included_df, moments_df, make_plots=False)
        run_step4(included_df, moments_df, modality_df, gmm_df)

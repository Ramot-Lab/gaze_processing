"""
Step 4 - between-participant(-panel) comparison of fixation-distribution shape: PCA,
k-means + hierarchical clustering, and a nonparametric KDE/Jensen-Shannon/classical-MDS
cross-check. Runs independently for x and y - no combined x+y pass.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

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


def plot_pca_scatter(scores, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in sorted(pd.unique(labels)):
        mask = labels == label
        ax.scatter(scores[mask, 0], scores[mask, 1], label=label, alpha=0.7, s=25)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_clustering(X, k_range=config.KMEANS_K_RANGE, random_state=config.CLUSTERING_RANDOM_STATE):
    """K-means: pick k by silhouette score over k_range. Hierarchical (Ward) clustering re-uses
    that same k so the two methods are directly comparable."""
    best_k, best_score, best_labels = None, -np.inf, None
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    hierarchical_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    return best_k, best_score, best_labels, hierarchical_labels


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


def _classical_mds(distance_matrix, n_components=2):
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


def plot_mds_scatter(coords, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in sorted(pd.unique(labels)):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.7, s=25)
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_step4_for_axis(fixation_df, moments_df, modality_df, gmm_df, axis):
    """Full Step 4 (PCA, clustering, KDE/JS/MDS) for a single univariate axis ('x' or 'y')."""
    config.ensure_output_dirs()

    numeric_cols = _numeric_feature_cols(axis)
    feature_df = build_feature_table(moments_df, modality_df, gmm_df, axis)
    X = _standardize(feature_df, numeric_cols)

    pca, scores, loadings_df = run_pca(X, numeric_cols)
    loadings_df.insert(0, "axis", axis)
    plot_pca_scatter(scores, feature_df["shape_label"].to_numpy(),
                      config.group_plot_path(f"step4_pca_scatter_{axis}.png"),
                      f"PCA of {axis}-axis fixation-shape features, colored by Step 3 shape label")

    best_k, silhouette, kmeans_labels, hier_labels = run_clustering(X)
    plot_dendrogram(X, config.group_plot_path(f"step4_dendrogram_{axis}.png"),
                     f"Hierarchical clustering (Ward) of {axis}-axis fixation-shape features")

    cluster_df = feature_df[["participant_id", "trial_id", "unit_id", "shape_label"]].copy()
    cluster_df.insert(0, "axis", axis)
    cluster_df["kmeans_cluster"] = kmeans_labels
    cluster_df["hierarchical_cluster"] = hier_labels
    cluster_df["kmeans_k"] = best_k
    cluster_df["kmeans_silhouette"] = silhouette
    cluster_df["pc1"] = scores[:, 0]
    cluster_df["pc2"] = scores[:, 1]

    unit_ids = feature_df["unit_id"].tolist()
    js_dist = kde_js_distance_matrix_1d(fixation_df, unit_ids, axis)
    js_dist_df = pd.DataFrame(js_dist, index=unit_ids, columns=unit_ids)

    mds_coords = _classical_mds(js_dist)
    plot_mds_scatter(mds_coords, feature_df["shape_label"].to_numpy(),
                      config.group_plot_path(f"step4_mds_scatter_{axis}.png"),
                      f"Classical MDS on KDE Jensen-Shannon distances ({axis}-axis), colored by shape label")

    print(f"Step 4 ({axis}): PCA explained variance (PC1,PC2) = "
          f"{pca.explained_variance_ratio_[:2].round(3)}, kmeans best_k={best_k} (silhouette={silhouette:.3f})")

    return {
        "feature_df": feature_df,
        "loadings_df": loadings_df,
        "cluster_df": cluster_df,
        "js_dist_df": js_dist_df,
        "mds_coords": mds_coords,
    }


def run_step4(fixation_df, moments_df, modality_df, gmm_df):
    results_x = run_step4_for_axis(fixation_df, moments_df, modality_df, gmm_df, "x")
    results_y = run_step4_for_axis(fixation_df, moments_df, modality_df, gmm_df, "y")

    loadings_df = pd.concat([results_x["loadings_df"], results_y["loadings_df"]], ignore_index=True)
    loadings_df.to_csv(config.table_path("step4_pca_loadings.csv"), index=False)

    cluster_df = pd.concat([results_x["cluster_df"], results_y["cluster_df"]], ignore_index=True)
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

    return {"x": results_x, "y": results_y}


if __name__ == "__main__":
    from .step0_load_data import run_step0
    from .step1_moments import run_step1
    from .step2_modality import run_step2
    from .step3_gmm import run_step3
    included_df, _ = run_step0()
    moments_df = run_step1(included_df)
    modality_df = run_step2(included_df, moments_df)
    gmm_df = run_step3(included_df, moments_df, make_plots=False)
    run_step4(included_df, moments_df, modality_df, gmm_df)

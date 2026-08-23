"""
Step 3 - Gaussian Mixture Model characterization per participant-panel/axis: fit k=1,2,3
components, pick k by BIC, derive a categorical shape label, and save one diagnostic plot
per unit/axis (histogram + KDE + weighted GMM components).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

from . import config


def _fit_best_gmm(values, max_components=config.GMM_MAX_COMPONENTS, n_init=config.GMM_N_INIT,
                   random_state=config.GMM_RANDOM_STATE):
    """Fits k=1..max_components GaussianMixtures, returns the one with lowest BIC."""
    x = values.reshape(-1, 1)
    best_model, best_bic = None, np.inf
    for k in range(1, max_components + 1):
        model = GaussianMixture(n_components=k, n_init=n_init, random_state=random_state)
        model.fit(x)
        bic = model.bic(x)
        if bic < best_bic:
            best_model, best_bic = model, bic
    return best_model


def _ordered_components(model):
    """Component (weight, mean, sd) triples, ordered by ascending mean."""
    means = model.means_.flatten()
    sds = np.sqrt(model.covariances_.flatten())
    weights = model.weights_.flatten()
    order = np.argsort(means)
    return [(weights[i], means[i], sds[i]) for i in order]


def ashmans_d(mean1, sd1, mean2, sd2):
    return np.sqrt(2) * abs(mean1 - mean2) / np.sqrt(sd1 ** 2 + sd2 ** 2)


def _shape_label(selected_k, skewness, components, ashman_d):
    if selected_k == 1:
        if abs(skewness) < 0.5:
            return "unimodal-symmetric"
        elif skewness >= 0.5:
            return "unimodal-right-skewed"
        else:
            return "unimodal-left-skewed"
    elif selected_k == 2:
        weight_diff = abs(components[0][0] - components[1][0])
        if ashman_d > config.ASHMANS_D_BIMODAL_THRESHOLD:
            return "bimodal-symmetric" if weight_diff < config.CLUSTER_WEIGHT_SYMMETRY_THRESHOLD else "bimodal-asymmetric"
        else:
            return "weakly-bimodal-or-overlapping"
    else:
        return "multimodal"


def _gmm_row(participant_id, trial_id, unit_id, axis, values, skewness, max_components=config.GMM_MAX_COMPONENTS):
    model = _fit_best_gmm(values, max_components=max_components)
    selected_k = model.n_components
    components = _ordered_components(model)

    row = {
        "participant_id": participant_id,
        "trial_id": trial_id,
        "unit_id": unit_id,
        "axis": axis,
        "n_fixations": len(values),
        "selected_k": selected_k,
        "bic": model.bic(values.reshape(-1, 1)),
    }
    for i in range(max_components):
        if i < len(components):
            weight, mean, sd = components[i]
        else:
            weight, mean, sd = np.nan, np.nan, np.nan
        row[f"component_{i + 1}_weight"] = weight
        row[f"component_{i + 1}_mean"] = mean
        row[f"component_{i + 1}_sd"] = sd

    d = np.nan
    if selected_k == 2:
        (_, m1, s1), (_, m2, s2) = components
        d = ashmans_d(m1, s1, m2, s2)
    row["ashmans_d"] = d
    row["shape_label"] = _shape_label(selected_k, skewness, components, d)
    row["_model"] = model  # consumed by plotting, dropped before saving to CSV
    return row


def compute_gmm_characterization(fixation_df, moments_df, max_components=config.GMM_MAX_COMPONENTS):
    rows = []
    for _, moment_row in moments_df.iterrows():
        participant_id, trial_id, axis = moment_row["participant_id"], moment_row["trial_id"], moment_row["axis"]
        values = fixation_df.loc[
            (fixation_df["participant_id"] == participant_id) & (fixation_df["trial_id"] == trial_id), axis
        ].to_numpy()
        rows.append(_gmm_row(participant_id, trial_id, moment_row["unit_id"], axis, values, moment_row["skewness"],
                              max_components=max_components))
    return pd.DataFrame(rows)


def plot_gmm_diagnostic(values, gmm_row, save_path):
    model = gmm_row["_model"]
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(values, bins=30, density=True, color="lightgray", edgecolor="white", label="fixations")

    grid = np.linspace(values.min(), values.max(), 500)
    kde = stats.gaussian_kde(values)
    ax.plot(grid, kde(grid), color="black", linewidth=1.5, label="KDE")

    n_components_reported = max(
        int(key.split("_")[1]) for key in gmm_row.index if key.startswith("component_") and key.endswith("_weight")
    )

    mixture_pdf = np.zeros_like(grid)
    for i in range(n_components_reported):
        weight = gmm_row[f"component_{i + 1}_weight"]
        if pd.isna(weight):
            continue
        mean, sd = gmm_row[f"component_{i + 1}_mean"], gmm_row[f"component_{i + 1}_sd"]
        component_pdf = weight * stats.norm.pdf(grid, loc=mean, scale=sd)
        mixture_pdf += component_pdf
        ax.plot(grid, component_pdf, linestyle="--", linewidth=1, label=f"component {i + 1} (w={weight:.2f})")
    ax.plot(grid, mixture_pdf, color="tab:red", linewidth=1.5, label=f"GMM mixture (k={model.n_components})")

    ax.set_xlabel(f"{gmm_row['axis']} position (normalized)")
    ax.set_ylabel("Density")
    ax.set_title(f"{gmm_row['participant_id']} / {gmm_row['trial_id']} - {gmm_row['axis']}-axis\n{gmm_row['shape_label']}", fontsize=9)
    ax.legend(fontsize=6, ncol=2 if n_components_reported > 3 else 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def run_step3(fixation_df, moments_df, make_plots=True, max_components=config.GMM_MAX_COMPONENTS):
    config.ensure_output_dirs()

    gmm_df = compute_gmm_characterization(fixation_df, moments_df, max_components=max_components)

    if make_plots:
        for _, gmm_row in gmm_df.iterrows():
            values = fixation_df.loc[
                (fixation_df["participant_id"] == gmm_row["participant_id"]) &
                (fixation_df["trial_id"] == gmm_row["trial_id"]), gmm_row["axis"]
            ].to_numpy()
            filename = f"{gmm_row['participant_id']}_{gmm_row['trial_id']}_{gmm_row['axis']}_gmm_fit.png"
            plot_gmm_diagnostic(values, gmm_row, config.per_participant_plot_path(filename))

    gmm_df = gmm_df.drop(columns=["_model"])
    gmm_df.to_csv(config.table_path("step3_gmm_characterization.csv"), index=False)

    print(f"Step 3: fit GMMs for {len(gmm_df)} participant-panel/axis rows. "
          f"Shape label counts:\n{gmm_df.groupby(['axis', 'shape_label']).size()}")
    return gmm_df


if __name__ == "__main__":
    from .step0_load_data import run_step0
    from .step1_moments import run_step1
    included_df, _ = run_step0()
    moments_df = run_step1(included_df)
    run_step3(included_df, moments_df)

"""
LRT vs BIC check - a separate validation branch (outputs under config.LRT_VS_BIC_DIR, next
to the main pipeline's outputs/) that checks whether Step 3's BIC-based GMM component
count (k) agrees with a more rigorous, but far more expensive, alternative: a sequential
parametric-bootstrap likelihood-ratio test (LRT) for the number of mixture components.

Why BIC alone isn't the last word
----------------------------------
BIC = -2*logL + p*log(n) is a cheap, closed-form approximation to "which k best balances
fit against complexity." It's the standard choice for Step 3 because it's fast enough to
run on hundreds of units. But its usual justification relies on regularity conditions
(a smooth, identifiable parameter space) that do NOT hold when comparing a k-component
mixture to a (k+1)-component one: under the null hypothesis of k components, the extra
component's mixing weight can sit right at the boundary of the parameter space (weight->0)
or its mean can coincide with an existing component's, so the classical asymptotic theory
BIC's derivation leans on breaks down. This is a well-known, specific problem for mixture
"order" (component count) selection - not a general knock against BIC.

McLachlan's bootstrap LRT (McLachlan, G.J., 1987, "On bootstrapping the likelihood ratio
test statistic for the number of components in a normal mixture", Journal of the Royal
Statistical Society: Series C (Applied Statistics), 36(3), 318-324; also McLachlan & Peel,
2000, "Finite Mixture Models", Wiley, ch. 6) sidesteps that problem entirely by not relying
on any asymptotic distribution at all:

  1. Fit k components (H0) and k+1 components (H1) to the real data.
  2. Compute the likelihood-ratio statistic: LRT = 2 * (logL(k+1) - logL(k)).
  3. Simulate many synthetic datasets of the same size FROM THE FITTED H0 (k-component)
     model - a "parametric bootstrap" - since if H0 is true, this is what the data-
     generating process actually looks like.
  4. Refit k and k+1 to each synthetic dataset and compute its own LRT statistic.
  5. The bootstrap p-value is the fraction of synthetic LRT statistics that are >= the
     real one (using the (count+1)/(B+1) bias correction from Davison & Hinkley, 1997,
     "Bootstrap Methods and Their Application", Cambridge University Press, section 4.2 -
     avoids ever reporting an impossible p-value of exactly 0).
  6. If p < alpha, k components isn't enough - reject H0, move on to test k+1 vs k+2.
     Stop at the first k that fails to reject (or at max_components).

This literally decides k via a proper hypothesis test, rather than approximating it with a
penalized-likelihood formula. It's a DIFFERENT question from Step 3's k_stability (which
asks "would BIC's pick survive a different sample of the same fixations?" - a finite-sample
robustness check, not a test of whether the selection rule itself is theoretically sound).
Both are useful; this module exists to see how much they actually disagree in practice.

Cost note: unlike the stability check, this is sequential and its cost is data-dependent
(it stops at the first non-rejected transition), so - per the same B/n_init reasoning as
Step 3's bootstrap (Efron & Tibshirani 1993; Monti et al. 2003; McLachlan 1987) - a smoke
test on a small subset should be run first to get real timing before committing to the
full ~640-unit dataset.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from . import config


def _fit_gmm(values, k, n_init, random_state):
    return GaussianMixture(n_components=k, n_init=n_init, random_state=random_state).fit(values.reshape(-1, 1))


def _log_likelihood(model, values):
    return model.score_samples(values.reshape(-1, 1)).sum()


def bic_curve(values, max_components, n_init=config.GMM_N_INIT, random_state=config.GMM_RANDOM_STATE):
    """BIC (and the fitted model) for every k=1..max_components on the real data - "which k
    best balances fit vs. complexity," across the whole candidate range, not just the winner."""
    x = values.reshape(-1, 1)
    rows = []
    for k in range(1, max_components + 1):
        model = GaussianMixture(n_components=k, n_init=n_init, random_state=random_state).fit(x)
        rows.append({"k": k, "bic": model.bic(x), "model": model})
    return rows


def bootstrap_k_distribution(values, max_components, n_resamples=config.GMM_BOOTSTRAP_N_RESAMPLES,
                              n_init=config.GMM_BOOTSTRAP_N_INIT, random_state=config.GMM_RANDOM_STATE):
    """
    Full histogram of "which k wins BIC" across n_resamples bootstrap resamples (with
    replacement) of `values` - the "certainty of k according to the bootstrap." This is
    the same nonparametric resampling Step 3's k_stability uses, just keeping every
    resample's winning k instead of only whether it matched the main fit's k.
    """
    rng = np.random.default_rng(random_state)
    n = len(values)
    counts = {k: 0 for k in range(1, max_components + 1)}
    for _ in range(n_resamples):
        resample = values[rng.integers(0, n, size=n)]
        x = resample.reshape(-1, 1)
        best_k, best_bic = 1, np.inf
        for k in range(1, max_components + 1):
            bic = GaussianMixture(n_components=k, n_init=n_init, random_state=random_state).fit(x).bic(x)
            if bic < best_bic:
                best_k, best_bic = k, bic
        counts[best_k] += 1
    return {k: c / n_resamples for k, c in counts.items()}


def _sample_from_gmm(model, n, rng):
    """
    Draws n samples from a fitted 1D GaussianMixture using our own rng - NOT model.sample(),
    which reseeds from model.random_state on every call (a fixed int gives byte-identical
    "random" draws every time it's called), which would make every bootstrap resample here
    identical and silently collapse the whole test.
    """
    weights = model.weights_.flatten()
    means = model.means_.flatten()
    sds = np.sqrt(model.covariances_.flatten())
    component_choices = rng.choice(len(weights), size=n, p=weights)
    return rng.normal(loc=means[component_choices], scale=sds[component_choices])


def bootstrap_lrt_transition_test(values, k, n_resamples=config.LRT_N_RESAMPLES, n_init=config.LRT_N_INIT,
                                   fit_n_init=config.GMM_N_INIT, random_state=config.GMM_RANDOM_STATE):
    """
    McLachlan (1987) parametric bootstrap LRT of H0: k components vs H1: k+1 components.
    Returns (observed_lrt_statistic, bootstrap_p_value). The two real-data fits use the
    main pipeline's full n_init (fit_n_init); the B simulated-data refits use the cheaper
    LRT_N_INIT, per the same reasoning as Step 3's bootstrap (see module docstring).
    """
    model_k = _fit_gmm(values, k, fit_n_init, random_state)
    model_k1 = _fit_gmm(values, k + 1, fit_n_init, random_state)
    observed_lrt = 2 * (_log_likelihood(model_k1, values) - _log_likelihood(model_k, values))

    rng = np.random.default_rng(random_state)
    n = len(values)
    count_ge = 0
    for _ in range(n_resamples):
        simulated = _sample_from_gmm(model_k, n, rng)
        sim_k = _fit_gmm(simulated, k, n_init, random_state)
        sim_k1 = _fit_gmm(simulated, k + 1, n_init, random_state)
        sim_lrt = 2 * (_log_likelihood(sim_k1, simulated) - _log_likelihood(sim_k, simulated))
        if sim_lrt >= observed_lrt:
            count_ge += 1

    p_value = (count_ge + 1) / (n_resamples + 1)  # Davison & Hinkley (1997) bias correction
    return observed_lrt, p_value


def select_k_via_bootstrap_lrt(values, max_components, alpha=config.LRT_ALPHA,
                                n_resamples=config.LRT_N_RESAMPLES, n_init=config.LRT_N_INIT,
                                fit_n_init=config.GMM_N_INIT, random_state=config.GMM_RANDOM_STATE):
    """
    Sequential test (McLachlan 1987): test k=1 vs 2, and if rejected keep testing k vs k+1
    until a test fails to reject (that k is returned) or max_components is reached.
    Returns (selected_k, list of per-transition dicts).
    """
    transitions = []
    for k in range(1, max_components):
        lrt_stat, p_value = bootstrap_lrt_transition_test(values, k, n_resamples=n_resamples, n_init=n_init,
                                                            fit_n_init=fit_n_init, random_state=random_state)
        rejected = p_value < alpha
        transitions.append({"k_tested": k, "lrt_statistic": lrt_stat, "p_value": p_value, "rejected": rejected})
        if not rejected:
            return k, transitions
    return max_components, transitions


def plot_bic_bootstrap_diagnostic(bic_rows, bootstrap_dist, bic_selected_k, lrt_selected_k, save_path, title):
    """
    Two stacked panels sharing the k-axis (never a dual-y-axis - see dataviz skill's
    anti-pattern on mixing two scales in one plot):
      top:    BIC(k) on the real data - "which k best balances fit vs. complexity"
      bottom: bootstrap_k_distribution(k) - "certainty of k according to the bootstrap"
    Both panels mark the BIC-selected k; the bottom panel also marks the LRT-selected k
    (McLachlan 1987) when it differs, so agreement/disagreement is visible at a glance.
    """
    ks = [row["k"] for row in bic_rows]
    bics = [row["bic"] for row in bic_rows]

    fig, (ax_bic, ax_boot) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax_bic.plot(ks, bics, color="tab:blue", linewidth=2, marker="o", markersize=4)
    ax_bic.axvline(bic_selected_k, color="tab:blue", linestyle="--", linewidth=1, alpha=0.6)
    ax_bic.set_ylabel("BIC (lower is better)")
    ax_bic.set_title(title, fontsize=9)

    bar_colors = ["tab:orange" if k != bic_selected_k else "tab:blue" for k in ks]
    ax_boot.bar(ks, [bootstrap_dist.get(k, 0.0) for k in ks], color=bar_colors, width=0.6)
    ax_boot.axvline(bic_selected_k, color="tab:blue", linestyle="--", linewidth=1, alpha=0.6,
                     label=f"BIC-selected k={bic_selected_k}")
    if lrt_selected_k != bic_selected_k:
        ax_boot.axvline(lrt_selected_k, color="tab:green", linestyle=":", linewidth=1.5,
                         label=f"LRT-selected k={lrt_selected_k}")
    ax_boot.set_xlabel("k (number of GMM components)")
    ax_boot.set_ylabel("P(bootstrap resample\npicks this k)")
    ax_boot.set_xticks(ks)
    ax_boot.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def compute_lrt_vs_bic_row(participant_id, trial_id, unit_id, axis, values):
    max_components = config.GMM_MAX_COMPONENTS_BY_AXIS[axis]

    bic_rows = bic_curve(values, max_components)
    bic_selected_k = min(bic_rows, key=lambda r: r["bic"])["k"]

    bootstrap_dist = bootstrap_k_distribution(values, max_components)
    lrt_selected_k, transitions = select_k_via_bootstrap_lrt(values, max_components)

    row = {
        "participant_id": participant_id,
        "trial_id": trial_id,
        "unit_id": unit_id,
        "axis": axis,
        "n_fixations": len(values),
        "bic_selected_k": bic_selected_k,
        "lrt_selected_k": lrt_selected_k,
        "agree": bic_selected_k == lrt_selected_k,
        "n_lrt_transitions_tested": len(transitions),
        "k_stability_at_bic_k": bootstrap_dist.get(bic_selected_k, 0.0),
    }
    row["_bic_rows"] = bic_rows
    row["_bootstrap_dist"] = bootstrap_dist
    return row


def run_lrt_vs_bic_check(fixation_df, moments_df, make_plots=True, limit_units=None):
    config.ensure_lrt_vs_bic_dirs()

    units = moments_df[["participant_id", "trial_id", "unit_id", "axis"]].drop_duplicates()
    if limit_units is not None:
        seen_units = units["unit_id"].unique()[:limit_units]
        units = units[units["unit_id"].isin(seen_units)]

    rows = []
    for _, u in units.iterrows():
        values = fixation_df.loc[
            (fixation_df["participant_id"] == u["participant_id"]) & (fixation_df["trial_id"] == u["trial_id"]),
            u["axis"]
        ].to_numpy()
        rows.append(compute_lrt_vs_bic_row(u["participant_id"], u["trial_id"], u["unit_id"], u["axis"], values))

    result_df = pd.DataFrame(rows)

    if make_plots:
        for _, row in result_df.iterrows():
            title = f"{row['participant_id']} / {row['trial_id']} - {row['axis']}-axis"
            filename = f"{row['participant_id']}_{row['trial_id']}_{row['axis']}_bic_bootstrap.png"
            plot_bic_bootstrap_diagnostic(row["_bic_rows"], row["_bootstrap_dist"], row["bic_selected_k"],
                                           row["lrt_selected_k"],
                                           config.lrt_vs_bic_per_participant_plot_path(filename), title)

    table_df = result_df.drop(columns=["_bic_rows", "_bootstrap_dist"])
    table_df.to_csv(config.lrt_vs_bic_table_path("gmm_lrt_vs_bic_k_selection.csv"), index=False)

    print(f"LRT vs BIC check: {len(table_df)} participant-panel/axis rows.\n"
          f"Agreement rate by axis:\n{table_df.groupby('axis')['agree'].mean()}")
    return table_df


if __name__ == "__main__":
    from .step0_load_data import run_step0
    from .step1_moments import run_step1
    included_df, _ = run_step0()
    moments_df = run_step1(included_df)
    run_lrt_vs_bic_check(included_df, moments_df)

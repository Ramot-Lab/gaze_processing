"""
bcea_individuality.py

Tests whether the standard deviation of BCEA (Bivariate Contour Ellipse Area)
computed per panel is an "individual" trait: is a subject's own panels more
similar to each other than to other subjects' panels?

Two complementary analyses are run:

1. ICC(3,1) -- Intraclass Correlation Coefficient (two-way, single measures,
   consistency), computed on the subjects x panels matrix of per-panel
   whole-panel SD-of-BCEA. This is the standard psychometric statistic for
   "is this scalar metric a stable trait of the individual" and uses all
   6 repeated panels per subject with proper statistical power (Shrout &
   Fleiss, 1979; McGraw & Wong, 1996).

2. Self-vs-other correlation test -- the literal question "is a person's
   correlation to themself higher than to someone else". A single SD number
   per panel can't be meaningfully correlated (nothing to correlate against),
   so each 90s panel is split into K equal-count time bins and the local SD
   of BCEA is computed per bin, giving a K-length "SD-BCEA temporal profile"
   per panel. Every pair of panels (across all subjects) is Pearson-correlated;
   pairs are labeled "self" (same subject) or "other" (different subjects).
   Because panels from the same subject are not independent observations, the
   self-vs-other difference is tested with a label-permutation test (not a
   plain t-test). Identification accuracy (top-1 nearest-neighbour matches
   correctly to the same subject) is also reported, matching the standard
   "gaze fingerprinting" metric.

Input CSV format (long format), one row per fixation:
    subject_id, panel_id, fixation_index, bcea

    - subject_id     : any hashable subject identifier
    - panel_id       : panel/trial identifier (expects 6 per subject)
    - fixation_index : the fixation's order within the 90s panel
                       (an integer counter, or a timestamp -- anything
                       monotonically increasing within a panel works)
    - bcea           : the BCEA value for that fixation

Usage:
    python bcea_individuality.py --input your_data.csv --bins 8 --n_perm 5000
    python bcea_individuality.py --demo                 # run on synthetic data
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1. Data prep
# --------------------------------------------------------------------------

def load_long_csv(path):
    df = pd.read_csv(path)
    required = {"subject_id", "panel_id", "fixation_index", "bcea"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    df = df.sort_values(["subject_id", "panel_id", "fixation_index"]).reset_index(drop=True)
    return df


def panel_scalar_sd(df):
    """Subjects x panels matrix of the whole-panel SD of BCEA."""
    g = df.groupby(["subject_id", "panel_id"])["bcea"].std().reset_index()
    mat = g.pivot(index="subject_id", columns="panel_id", values="bcea")
    return mat  # rows = subjects, cols = panels, values = SD(BCEA)


def panel_binned_profile(df, n_bins=8, min_per_bin=2):
    """
    For every (subject, panel), split fixations into n_bins equal-COUNT bins
    (ordered by fixation_index) and compute local SD(BCEA) per bin.
    Returns a dict {(subject_id, panel_id): np.array(length n_bins)} and
    drops panels that don't have enough fixations for the requested binning.
    """
    profiles = {}
    dropped = []
    for (sid, pid), grp in df.groupby(["subject_id", "panel_id"]):
        vals = grp.sort_values("fixation_index")["bcea"].to_numpy()
        n = len(vals)
        if n < n_bins * min_per_bin:
            dropped.append((sid, pid, n))
            continue
        bin_idx = np.floor(np.linspace(0, n_bins - 1e-9, n)).astype(int)
        profile = np.array([vals[bin_idx == b].std(ddof=1) for b in range(n_bins)])
        profiles[(sid, pid)] = profile
    return profiles, dropped


# --------------------------------------------------------------------------
# 2. ICC(3,1) -- Shrout & Fleiss (1979) / McGraw & Wong (1996)
# --------------------------------------------------------------------------

def icc_3_1(mat, alpha=0.05):
    """
    mat: subjects (rows) x panels (cols), no NaNs.
    Returns dict with icc estimate, CI, F, p-value.
    """
    X = mat.to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("ICC requires a complete subjects x panels matrix (no missing panels).")
    n, k = X.shape  # n subjects, k panels

    grand_mean = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)

    ss_total = ((X - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)

    msr = ss_rows / df_rows
    msc = ss_cols / df_cols
    mse = ss_error / df_error

    icc = (msr - mse) / (msr + (k - 1) * mse)

    f_stat = msr / mse
    p_value = 1 - stats.f.cdf(f_stat, df_rows, df_error)

    f_crit_u = stats.f.ppf(1 - alpha / 2, df_rows, df_error)
    f_crit_l = stats.f.ppf(1 - alpha / 2, df_error, df_rows)
    f_l = f_stat / f_crit_u
    f_u = f_stat * f_crit_l
    icc_low = (f_l - 1) / (f_l + k - 1)
    icc_high = (f_u - 1) / (f_u + k - 1)

    return {
        "icc": icc, "ci_low": icc_low, "ci_high": icc_high,
        "F": f_stat, "df1": df_rows, "df2": df_error, "p": p_value,
        "n_subjects": n, "n_panels": k,
    }


# --------------------------------------------------------------------------
# 3. Self vs other correlation test + identification accuracy
# --------------------------------------------------------------------------

def self_vs_other_test(profiles, n_perm=5000, seed=0, alternative="greater"):
    keys = list(profiles.keys())
    subj = np.array([k[0] for k in keys], dtype=object)
    mats = np.vstack([profiles[k] for k in keys])  # (n_panels_total, n_bins)

    n_panels_total = mats.shape[0]
    # pairwise Pearson correlation matrix
    corr = np.corrcoef(mats)
    iu = np.triu_indices(n_panels_total, k=1)
    r_vals = corr[iu]
    subj_i = subj[iu[0]]
    subj_j = subj[iu[1]]
    self_mask = subj_i == subj_j

    self_r = r_vals[self_mask]
    other_r = r_vals[~self_mask]

    def fisher_z(r):
        r = np.clip(r, -0.999999, 0.999999)
        return np.arctanh(r)

    z_self, z_other = fisher_z(self_r), fisher_z(other_r)
    observed_stat = z_self.mean() - z_other.mean()

    pooled_sd = np.sqrt(((len(z_self) - 1) * z_self.var(ddof=1) +
                          (len(z_other) - 1) * z_other.var(ddof=1)) /
                         (len(z_self) + len(z_other) - 2))
    cohens_d = observed_stat / pooled_sd if pooled_sd > 0 else np.nan

    mw_stat, mw_p = stats.mannwhitneyu(self_r, other_r, alternative=alternative)

    # Label-permutation test: shuffle which subject each panel "belongs to"
    # (preserves the full correlation matrix and each subject's panel count),
    # then recompute the self/other split. This respects the non-independence
    # of same-subject panel pairs, unlike treating each pair as an iid sample.
    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm)
    for p in range(n_perm):
        perm_subj = rng.permutation(subj)
        pi, pj = perm_subj[iu[0]], perm_subj[iu[1]]
        pm = pi == pj
        if pm.sum() == 0 or (~pm).sum() == 0:
            null_stats[p] = 0.0
            continue
        null_stats[p] = fisher_z(r_vals[pm]).mean() - fisher_z(r_vals[~pm]).mean()

    if alternative == "greater":
        perm_p = (1 + np.sum(null_stats >= observed_stat)) / (n_perm + 1)
    else:
        perm_p = (1 + np.sum(np.abs(null_stats) >= abs(observed_stat))) / (n_perm + 1)

    # Identification accuracy: for each panel, does its best-matching OTHER
    # panel (max r, excluding itself) belong to the same subject?
    corr_noself = corr.copy()
    np.fill_diagonal(corr_noself, -np.inf)
    best_match = np.argmax(corr_noself, axis=1)
    correct = subj[best_match] == subj
    id_accuracy = correct.mean()
    counts_per_subject = pd.Series(subj).value_counts()
    n_subj = len(counts_per_subject)
    chance = (counts_per_subject.mean() - 1) / (n_panels_total - 1)

    return {
        "n_panels_total": n_panels_total, "n_subjects": n_subj,
        "mean_self_r": self_r.mean(), "mean_other_r": other_r.mean(),
        "median_self_r": np.median(self_r), "median_other_r": np.median(other_r),
        "cohens_d": cohens_d,
        "mannwhitney_U": mw_stat, "mannwhitney_p": mw_p,
        "perm_observed_stat": observed_stat, "perm_p": perm_p, "n_perm": n_perm,
        "id_accuracy": id_accuracy, "chance_level": chance,
        "self_r": self_r, "other_r": other_r,
    }


# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------

def plot_self_vs_other(res, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-1, 1, 41)
    ax.hist(res["other_r"], bins=bins, alpha=0.6, label="other subject", density=True)
    ax.hist(res["self_r"], bins=bins, alpha=0.6, label="same subject", density=True)
    ax.axvline(res["mean_self_r"], linestyle="--", linewidth=1)
    ax.axvline(res["mean_other_r"], linestyle="--", linewidth=1)
    ax.set_xlabel("Pearson r between panel SD-BCEA profiles")
    ax.set_ylabel("density")
    ax.set_title("Self vs. other panel-profile correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Synthetic demo data
# --------------------------------------------------------------------------

def make_demo_data(n_subjects=20, n_panels=6, n_fixations_range=(60, 100),
                    individual_effect=1.5, seed=0):
    """
    Generates synthetic fixation-level BCEA data with a subject-specific
    baseline dispersion (so the pipeline should detect individuality) plus
    panel-to-panel and fixation-to-fixation noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    subject_trait = rng.gamma(shape=4.0, scale=individual_effect, size=n_subjects)
    for s in range(n_subjects):
        base_sd = subject_trait[s]
        for p in range(n_panels):
            n_fix = rng.integers(*n_fixations_range)
            panel_sd = max(0.1, base_sd + rng.normal(0, 0.3))
            bcea_vals = rng.normal(loc=5.0, scale=panel_sd, size=n_fix)
            bcea_vals = np.abs(bcea_vals)  # BCEA is non-negative
            for f, v in enumerate(bcea_vals):
                rows.append((f"S{s:02d}", f"P{p+1}", f, v))
    return pd.DataFrame(rows, columns=["subject_id", "panel_id", "fixation_index", "bcea"])


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------

def run(df, n_bins=8, n_perm=5000, alpha=0.05, outdir=".", seed=0):
    print(f"Loaded {len(df)} fixation rows, "
          f"{df['subject_id'].nunique()} subjects, "
          f"{df.groupby('subject_id')['panel_id'].nunique().mean():.1f} panels/subject (avg).\n")

    # --- ICC on whole-panel scalar SD ---
    mat = panel_scalar_sd(df)
    mat = mat.dropna()  # ICC needs a complete matrix
    icc_res = icc_3_1(mat, alpha=alpha)
    print("=== ICC(3,1) on per-panel SD(BCEA) ===")
    print(f"  n subjects used: {icc_res['n_subjects']}, n panels: {icc_res['n_panels']}")
    print(f"  ICC(3,1) = {icc_res['icc']:.3f}  "
          f"[{100*(1-alpha):.0f}% CI: {icc_res['ci_low']:.3f}, {icc_res['ci_high']:.3f}]")
    print(f"  F({icc_res['df1']}, {icc_res['df2']}) = {icc_res['F']:.2f}, p = {icc_res['p']:.4g}")
    interp = ("poor" if icc_res['icc'] < 0.5 else
              "moderate" if icc_res['icc'] < 0.75 else
              "good" if icc_res['icc'] < 0.9 else "excellent")
    print(f"  Interpretation (Koo & Li, 2016): {interp} reliability/individuality\n")

    # --- self vs other correlation test ---
    profiles, dropped = panel_binned_profile(df, n_bins=n_bins)
    if dropped:
        print(f"  Note: dropped {len(dropped)} panel(s) with too few fixations for {n_bins} bins.\n")
    res = self_vs_other_test(profiles, n_perm=n_perm, seed=seed)
    print("=== Self vs. other panel-profile correlation ===")
    print(f"  panels analyzed: {res['n_panels_total']} across {res['n_subjects']} subjects, "
          f"{n_bins}-bin SD(BCEA) profiles")
    print(f"  mean self r  = {res['mean_self_r']:.3f} (median {res['median_self_r']:.3f})")
    print(f"  mean other r = {res['mean_other_r']:.3f} (median {res['median_other_r']:.3f})")
    print(f"  Cohen's d (Fisher-z) = {res['cohens_d']:.3f}")
    print(f"  Mann-Whitney U p (self > other) = {res['mannwhitney_p']:.4g}")
    print(f"  Permutation test p (self > other, n={res['n_perm']}) = {res['perm_p']:.4g}")
    print(f"  Identification accuracy = {res['id_accuracy']:.1%} "
          f"(chance level {res['chance_level']:.1%})\n")

    plot_path = f"{outdir}/self_vs_other_correlation.png"
    plot_self_vs_other(res, plot_path)
    print(f"Saved plot to {plot_path}")

    return icc_res, res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, help="Path to long-format CSV (see module docstring).")
    ap.add_argument("--demo", action="store_true", help="Run on generated synthetic data instead.")
    ap.add_argument("--bins", type=int, default=8, help="Time bins per panel for the profile test.")
    ap.add_argument("--n_perm", type=int, default=5000, help="Permutation test iterations.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for ICC confidence interval.")
    ap.add_argument("--outdir", type=str, default=".", help="Where to save the output plot.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = ap.parse_args()

    if args.demo:
        print("Running on synthetic demo data (subjects have a real, injected individual "
              "dispersion trait -- expect the tests below to detect it).\n")
        df = make_demo_data(seed=args.seed)
    elif args.input:
        df = load_long_csv(args.input)
    else:
        print("Provide --input path.csv or --demo. Use --help for details.", file=sys.stderr)
        sys.exit(1)

    run(df, n_bins=args.bins, n_perm=args.n_perm, alpha=args.alpha,
        outdir=args.outdir, seed=args.seed)


if __name__ == "__main__":
    main()
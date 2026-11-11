"""
Symbol count vs trial: split-half stability of the per-panel mean trajectory, and where
the MS-vs-HC trajectory difference falls relative to random halves of the cohort
(decision 2026-11-09).

Per panel, the curve is the mean (+/- SD) of symbol_counts_per_trial across participants
at each trial_idx. Truncation (user's choice, 2026-11-09): the curve stops at
trial_idx = mean + 2 SD of that subset's own panel score. trial_idx and score share a
scale here (each participant's last trial_idx equals their panel score; practice trials
0-2 are absent), so this keeps trial positions up to the score of a high (+2 SD)
performer. Every subset (whole cohort, each random half, MS, HC) gets the cutoff from its
OWN scores - "the same truncation rule" applied to that subset.

Stability metric: Pearson r between two mean trajectories, matched on trial_idx over the
range both curves cover - the same statistic as the pipeline's split-half reliability
(feature_pipeline.calculate_reliability) and #symbols_corr_to_mean
(feature_pipeline.compute_symbols_corr_to_mean, a shape measure that ignores level).

Permutation test: N_ITERATIONS unstratified random splits of the whole cohort into two
halves (odd N -> sizes n and n+1). Each split is drawn once per iteration at the
participant level and applied to every panel. MS-vs-HC r is located in the per-panel
random-split distribution; p = (b + 1) / (m + 1) with b = #null r <= observed r, the
add-one permutation p-value that never reports exactly 0 (Phipson & Smyth, 2010, Stat
Appl Genet Mol Biol 9:39; Good, 2005, "Permutation, Parametric and Bootstrap Tests of
Hypotheses"). One-sided: a LOWER r means less similar shapes, so the question is whether
MS and HC are less alike than random halves.

Usage: python symbol_count_stability.py [trial_features_dir]
"""
import os
import sys
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pipeline_config as pc
from feature_pipeline import EXCLUDED_PARTICIPANTS, PANELS, demographics_text, annotate_demographics

DEFAULT_TABLES_DIR = os.path.join(pc.noam_m_root(), "markov_features_results_27_09_26", "feature_analysis")
OUTPUT_ROOT = os.path.join(pc.noam_m_root(), "symbol_count_stability")

METRIC = "symbol_counts_per_trial"
N_SD_TRUNCATION = 2.0
N_ITERATIONS = 1000
RANDOM_SEED = 20261109

GROUP_COLORS = {"HC": "tab:blue", "MS": "tab:orange"}
HALF_COLORS = {"Half A": "#1b9e77", "Half B": "#7570b3"}


# ============================================================
# 1. DATA
# ============================================================
def load_data(tables_dir):
    trials = pd.read_csv(os.path.join(tables_dir, "trial_features.csv"))
    panels = pd.read_csv(os.path.join(tables_dir, "panel_features.csv"))
    people = pd.read_csv(os.path.join(tables_dir, "participant_features.csv"))
    trials["panel"] = trials["panel"].astype(str)
    panels["panel"] = panels["panel"].astype(str)
    people["group"] = people["group"].replace({"pwMS": "MS"})

    trials = trials[~trials["participant"].isin(EXCLUDED_PARTICIPANTS)]
    people = people[people["participant"].isin(trials["participant"].unique())].reset_index(drop=True)
    return trials, panels, people


def build_panel_matrices(trials, panels, participants):
    """Per panel: (participants x trial_idx) value matrix (NaN where a participant has no
    trial) and the matching score vector, rows aligned to `participants` so a boolean
    participant mask selects a subset on every panel at once."""
    out = {}
    for panel in PANELS:
        t = trials[trials["panel"] == panel]
        if t.empty:
            continue
        wide = t.pivot_table(index="participant", columns="trial_idx", values=METRIC).reindex(participants)
        scores = (panels[panels["panel"] == panel].set_index("participant")["score"]
                  .reindex(participants).to_numpy(dtype=float))
        out[panel] = {"values": wide.to_numpy(dtype=float), "trial_idx": wide.columns.to_numpy(),
                      "scores": scores}
    return out


# ============================================================
# 2. TRAJECTORY + METRIC
# ============================================================
def trajectory(mat, mask):
    """Mean/SD/N per trial_idx for the participants in `mask`, truncated at
    mean + N_SD_TRUNCATION * SD of their own panel scores."""
    vals = mat["values"][mask]
    scores = mat["scores"][mask]
    scores = scores[~np.isnan(scores)]
    present = ~np.isnan(vals).all(axis=1)
    vals = vals[present]
    if len(vals) == 0 or len(scores) < 2:
        return None
    cutoff = scores.mean() + N_SD_TRUNCATION * scores.std(ddof=1)
    keep = mat["trial_idx"] <= cutoff
    n = (~np.isnan(vals[:, keep])).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(vals[:, keep], axis=0)
        sd = np.nanstd(vals[:, keep], axis=0, ddof=1)
    valid = n > 0
    return pd.DataFrame({"trial_idx": mat["trial_idx"][keep][valid], "mean": mean[valid],
                         "sd": sd[valid], "n": n[valid]}).assign(cutoff=cutoff, n_participants=len(vals))


def trajectory_r(a, b):
    """Pearson r between two mean trajectories over their shared trial_idx range."""
    if a is None or b is None:
        return np.nan, 0
    m = a[["trial_idx", "mean"]].merge(b[["trial_idx", "mean"]], on="trial_idx", suffixes=("_a", "_b"))
    if len(m) < 3 or m["mean_a"].std() == 0 or m["mean_b"].std() == 0:
        return np.nan, len(m)
    return float(np.corrcoef(m["mean_a"], m["mean_b"])[0, 1]), len(m)


# ============================================================
# 3. PERMUTATION (SPLIT-HALF) TEST
# ============================================================
def run_permutations(mats, people, rng):
    n = len(people)
    n_a = n // 2  # odd N -> halves of n//2 and n - n//2 (user's choice, 2026-11-09)
    rows, splits = [], []
    for it in range(N_ITERATIONS):
        perm = rng.permutation(n)
        mask_a = np.zeros(n, dtype=bool)
        mask_a[perm[:n_a]] = True
        splits.append({"iteration": it,
                       "half_A_participants": ";".join(people.loc[mask_a, "participant"])})
        for panel, mat in mats.items():
            ta, tb = trajectory(mat, mask_a), trajectory(mat, ~mask_a)
            r, n_shared = trajectory_r(ta, tb)
            rows.append({"iteration": it, "panel": panel, "r": r, "n_shared_trials": n_shared,
                         "n_half_A": None if ta is None else int(ta["n_participants"].iloc[0]),
                         "n_half_B": None if tb is None else int(tb["n_participants"].iloc[0]),
                         "cutoff_half_A": None if ta is None else ta["cutoff"].iloc[0],
                         "cutoff_half_B": None if tb is None else tb["cutoff"].iloc[0]})
    return pd.DataFrame(rows), pd.DataFrame(splits)


def summarize(null_df, group_r):
    rows = []
    for panel, g in null_df.groupby("panel", sort=False):
        r = g["r"].dropna().to_numpy()
        obs = group_r.get(panel, np.nan)
        b = int((r <= obs).sum()) if not np.isnan(obs) else np.nan
        rows.append({"panel": panel, "n_valid_iterations": len(r),
                     "null_mean_r": r.mean(), "null_median_r": np.median(r),
                     "null_ci95_low": np.percentile(r, 2.5), "null_ci95_high": np.percentile(r, 97.5),
                     "ms_vs_hc_r": obs,
                     "ms_vs_hc_percentile": 100 * b / len(r) if not np.isnan(obs) else np.nan,
                     "p_one_sided_lower": (b + 1) / (len(r) + 1) if not np.isnan(obs) else np.nan})
    return pd.DataFrame(rows)


# ============================================================
# 4. FIGURES
# ============================================================
def _grid():
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), squeeze=False)
    return fig, axes.flatten()


def _draw(ax, traj, color, label):
    ax.plot(traj["trial_idx"], traj["mean"], color=color, linewidth=2, label=label)
    ax.fill_between(traj["trial_idx"], traj["mean"] - traj["sd"], traj["mean"] + traj["sd"],
                    color=color, alpha=0.18, linewidth=0)


def _style(ax, title, demo):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Symbols per trial (mean ± SD)")
    ax.grid(True, alpha=0.3)
    annotate_demographics(demo, ax=ax, loc="upper left")


def _demo(people, mask, mat):
    present = mask & ~np.isnan(mat["values"]).all(axis=1)
    return demographics_text(people[present])


def _finish(fig, axes, n_used, suptitle, path):
    for ax in axes[n_used:]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_all(mats, people, out_dir):
    fig, axes = _grid()
    everyone = np.ones(len(people), dtype=bool)
    for ax, (panel, mat) in zip(axes, mats.items()):
        traj = trajectory(mat, everyone)
        _draw(ax, traj, "black", "All participants")
        _style(ax, f"Panel {panel} (cutoff: trial ≤ {traj['cutoff'].iloc[0]:.1f}; "
                   f"N at last trial = {traj['n'].iloc[-1]})", _demo(people, everyone, mat))
    _finish(fig, axes, len(mats), "Symbol count vs trial - all participants\n"
            f"(curve stops at trial = mean + {N_SD_TRUNCATION:g} SD of panel score)",
            os.path.join(out_dir, "01_all_participants.png"))


def plot_example_splits(mats, people, null_df, splits, out_dir):
    """Per panel, the iteration whose r is closest to that panel's median and 5th
    percentile (so a given figure shows a different iteration on each panel)."""
    for label, q in (("median", 50), ("p05", 5)):
        fig, axes = _grid()
        for ax, (panel, mat) in zip(axes, mats.items()):
            g = null_df[(null_df["panel"] == panel)].dropna(subset=["r"])
            target = np.percentile(g["r"], q)
            row = g.iloc[(g["r"] - target).abs().argmin()]
            it = int(row["iteration"])
            half_a = set(splits.loc[it, "half_A_participants"].split(";"))
            mask_a = people["participant"].isin(half_a).to_numpy()
            for name, m in (("Half A", mask_a), ("Half B", ~mask_a)):
                traj = trajectory(mat, m)
                _draw(ax, traj, HALF_COLORS[name], f"{name} (n={int(traj['n_participants'].iloc[0])})")
            ax.legend(loc="lower right", fontsize=8)
            _style(ax, f"Panel {panel} - iteration {it}: r = {row['r']:.3f}", _demo(people, np.ones(len(people), bool), mat))
        title = "median" if q == 50 else "5th percentile"
        _finish(fig, axes, len(mats), f"Example random split near the {title} of the r distribution (per panel)",
                os.path.join(out_dir, f"02_example_split_{label}.png"))


def plot_ms_vs_hc(mats, people, summary, out_dir):
    fig, axes = _grid()
    s = summary.set_index("panel")
    for ax, (panel, mat) in zip(axes, mats.items()):
        for grp in ("HC", "MS"):
            m = (people["group"] == grp).to_numpy()
            traj = trajectory(mat, m)
            if traj is not None:
                _draw(ax, traj, GROUP_COLORS[grp], f"{grp} (n={int(traj['n_participants'].iloc[0])})")
        ax.legend(loc="lower right", fontsize=8)
        _style(ax, f"Panel {panel}: r = {s.loc[panel, 'ms_vs_hc_r']:.3f}, "
                   f"percentile = {s.loc[panel, 'ms_vs_hc_percentile']:.1f}, "
                   f"p = {s.loc[panel, 'p_one_sided_lower']:.3f}",
               _demo(people, np.ones(len(people), bool), mat))
    _finish(fig, axes, len(mats), "MS vs HC - symbol count vs trial (each group truncated at its own mean + 2 SD score)",
            os.path.join(out_dir, "03_ms_vs_hc.png"))


def plot_null_hist(mats, people, null_df, summary, out_dir):
    fig, axes = _grid()
    s = summary.set_index("panel")
    for ax, (panel, mat) in zip(axes, mats.items()):
        r = null_df.loc[null_df["panel"] == panel, "r"].dropna()
        ax.hist(r, bins=40, color="#9aa5b1", edgecolor="white")
        lo, hi = s.loc[panel, "null_ci95_low"], s.loc[panel, "null_ci95_high"]
        ax.axvspan(lo, hi, color="#9aa5b1", alpha=0.15, label=f"95% interval [{lo:.3f}, {hi:.3f}]")
        ax.axvline(r.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean r = {r.mean():.3f}")
        obs = s.loc[panel, "ms_vs_hc_r"]
        ax.axvline(obs, color="tab:red", linewidth=2.5,
                   label=f"MS vs HC r = {obs:.3f} (pctl {s.loc[panel, 'ms_vs_hc_percentile']:.1f})")
        # Demographics in the title here - a corner box would sit on top of the bars.
        demo = _demo(people, np.ones(len(people), bool), mat).replace("\n", "; ")
        ax.set_title(f"Panel {panel}\n{demo}", fontsize=10)
        ax.set_xlabel("Pearson r between half trajectories")
        ax.set_ylabel("Iterations")
        ax.legend(loc="upper left", fontsize=8)
    _finish(fig, axes, len(mats), f"Split-half stability distribution ({N_ITERATIONS} random splits) with MS vs HC marked",
            os.path.join(out_dir, "04_stability_distribution.png"))


# ============================================================
# 5. TEXT
# ============================================================
def write_readme(out_dir, tables_dir, people, summary):
    lines = [
        "# Symbol count vs trial - split-half stability",
        "",
        f"Run date: {date.today().isoformat()}  |  Random seed: {RANDOM_SEED}  |  Iterations: {N_ITERATIONS}",
        f"Source tables: `{tables_dir}`  |  Excluded (hard-coded): {', '.join(EXCLUDED_PARTICIPANTS)}",
        f"Cohort: {demographics_text(people).replace(chr(10), '; ')}",
        "",
        "## Truncation",
        f"Each curve stops at trial_idx = mean + {N_SD_TRUNCATION:g} SD of the panel score of the "
        "participants behind that curve (computed separately for the whole cohort, each random half, MS and HC). "
        "Past most participants' last trial, the late points rest on few participants (see `n` in "
        "`trajectories_*.csv`), so the tail of each curve is noisier than its start.",
        "",
        "## Stability metric: Pearson r between two mean trajectories",
        "- **What it measures:** how closely the trial-to-trial rise and fall of one subsample's mean "
        "symbol-count curve follows the other's, matched on trial_idx over the range both curves cover. "
        "It is the statistic the pipeline already uses for split-half reliability "
        "(`feature_pipeline.calculate_reliability`) and for `#symbols_corr_to_mean`.",
        "- **Why it counts as stability:** two random halves of the cohort are independent subsamples with "
        "no systematic difference between them. If their mean curves correlate highly, the trajectory's shape "
        "is reproducible and comes from the population, not from a few participants who happen to be in one "
        "half. A wide or low distribution of r means the shape depends on who is sampled.",
        "- **What it does not measure:** level. Pearson r is unchanged by adding a constant or scaling a curve, so "
        "two curves offset from each other (e.g. one group searching more symbols on every trial) can still "
        "have r near 1. Compare the levels on the MS vs HC figure directly.",
        "- **Reading the values:** r near 1 with a narrow 95% interval is a stable trajectory. A wide interval, or a "
        "lower bound well below ~0.7, means the shape is not reliably reproduced. Commonly cited cut-offs "
        "for reliability coefficients are ≥0.7 acceptable and ≥0.9 excellent (Nunnally, 1978, Psychometric Theory).",
        "",
        "## MS vs HC",
        "The same r is computed between the MS and HC mean curves and located within the random-split "
        "distribution. A percentile near 50 means MS and HC differ in shape no more than two random halves do. "
        "A low percentile (p < .05, one-sided) means their shapes differ more than random halves would explain. "
        "p = (b + 1) / (m + 1), with b = number of random-split r values ≤ the MS–HC r and m = valid iterations "
        "(Phipson & Smyth, 2010). The groups are not equal in size, while the random halves are, so the "
        "smaller group's curve is somewhat noisier than a half's. This biases the MS–HC r slightly downward.",
        "",
        "## Results",
        summary.round(4).to_markdown(index=False),
    ]
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# 6. MAIN
# ============================================================
def main(tables_dir=DEFAULT_TABLES_DIR):
    out_dir = os.path.join(OUTPUT_ROOT, date.today().strftime("%Y_%m_%d"))
    os.makedirs(out_dir, exist_ok=True)

    trials, panels, people = load_data(tables_dir)
    mats = build_panel_matrices(trials, panels, people["participant"].tolist())
    print(f"Cohort: {demographics_text(people)} | panels: {list(mats)}")

    everyone = np.ones(len(people), dtype=bool)
    traj_rows, group_r = [], {}
    for panel, mat in mats.items():
        traj_rows.append(trajectory(mat, everyone).assign(panel=panel, subset="all"))
        t_ms = trajectory(mat, (people["group"] == "MS").to_numpy())
        t_hc = trajectory(mat, (people["group"] == "HC").to_numpy())
        traj_rows += [t_ms.assign(panel=panel, subset="MS"), t_hc.assign(panel=panel, subset="HC")]
        group_r[panel] = trajectory_r(t_ms, t_hc)[0]
    pd.concat(traj_rows).to_csv(os.path.join(out_dir, "trajectories_all_MS_HC.csv"), index=False)

    rng = np.random.default_rng(RANDOM_SEED)
    null_df, splits = run_permutations(mats, people, rng)
    null_df.to_csv(os.path.join(out_dir, "per_iteration_metrics.csv"), index=False)
    splits.to_csv(os.path.join(out_dir, "per_iteration_splits.csv"), index=False)
    with open(os.path.join(out_dir, "random_seed.txt"), "w") as f:
        f.write(f"RANDOM_SEED={RANDOM_SEED}\nN_ITERATIONS={N_ITERATIONS}\nnumpy.random.default_rng\n")

    summary = summarize(null_df, group_r)
    summary.to_csv(os.path.join(out_dir, "stability_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))

    plot_all(mats, people, out_dir)
    plot_example_splits(mats, people, null_df, splits, out_dir)
    plot_ms_vs_hc(mats, people, summary, out_dir)
    plot_null_hist(mats, people, null_df, summary, out_dir)
    write_readme(out_dir, tables_dir, people, summary)
    print(f"All outputs in {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TABLES_DIR)

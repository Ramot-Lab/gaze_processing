"""
Phase A - SDMT time-budget features (per panel, then mean across panels).

Definitions (decision 2026-11-16, Noam): every panel is its own 90 s SDMT with its own
score S_panel, so every time component is computed per participant x panel from the
EXISTING trial-level table (trial_features.csv - nothing is re-extracted):

    time_T_item    = 90 000 ms / S_panel
    time_T_encode  = sum over the panel's trials of time_in_relevant_fixations_ms / S_panel
    time_T_search  = sum over the panel's trials of trial_in_search_durations_ms / S_panel
    time_T_rest    = time_T_item - time_T_encode - time_T_search

The participant value is the mean of panel values (mean-of-panel-means, decision
2026-09-29 convention), used only after the panel values are checked for split-half
reliability with the pipeline's existing feature_pipeline.calculate_reliability.

Known caveat (shown to Noam 2026-11-16): time_in_relevant_fixations_ms is the span from
the first to the last target fixation of a trial (trial_features.py
_extract_time_in_relevant_fixations), so a target -> dictionary -> target trial counts its
dictionary visit inside that span as well as in trial_in_search_durations_ms. time_T_rest
therefore goes negative wherever that double counting exceeds the true residual time; it
is reported here as a diagnostic.

Time shares (time_share_*) are descriptors only: they are compositional (sum to 1), so
never enter all three into one model (Aitchison 1986).
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402
from feature_pipeline import (calculate_reliability, build_reliability_matrix,  # noqa: E402
                              demographics_text, annotate_demographics,
                              RELIABILITY_REPETITIONS, MIN_PANELS_RELIABILITY)

# ---------------------------------------------------------------- config
GROUP = "group"                      # single group column (values "HC" / "MS")
GROUP_ORDER = ["HC", "MS"]
PANEL_SECONDS = 90.0
RUN_DATE = "16_11_26"
SOURCE_DATE = "12_11_26"
RANDOM_SEED = 20261116

ROOT = os.path.join(pipeline_config.noam_m_root(), SOURCE_DATE)
FEATURES_CSV = os.path.join(ROOT, "analysis_results", "features", "csvs")
TRIAL_CSV = os.path.join(FEATURES_CSV, "trial_features.csv")
PANEL_CSV = os.path.join(FEATURES_CSV, "panel_features.csv")
PARTICIPANT_CSV = os.path.join(FEATURES_CSV, "participant_features.csv")
SCORES_CSV = os.path.join(ROOT, "behavior_scores", "all_scores.csv")

OUT_DIR = os.path.join(ROOT, "analysis_results", f"sdmt_decomposition_{RUN_DATE}", "phase_A")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

T_COLS = ["time_T_item", "time_T_encode", "time_T_search", "time_T_rest"]
SHARE_COLS = ["time_share_encode", "time_share_search", "time_share_rest"]

sns.set_theme(style="whitegrid", context="paper")
PALETTE = dict(zip(GROUP_ORDER, sns.color_palette("colorblind", 2)))


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _save_fig(fig, name, demo_df):
    annotate_demographics(demographics_text(demo_df), on_figure=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- load
def load_inputs():
    trials = pd.read_csv(TRIAL_CSV, dtype={"panel": str})
    panels = pd.read_csv(PANEL_CSV, dtype={"panel": str})
    participants = pd.read_csv(PARTICIPANT_CSV)
    scores = pd.read_csv(SCORES_CSV, dtype={"Panel": str}).rename(
        columns={"Participant": "participant", "Panel": "panel", "SDMT_Score": "S_panel"})
    return trials, panels, participants, scores


# ---------------------------------------------------------------- compute
def build_panel_time_budget(trials, scores, participants):
    sums = (trials.groupby(["participant", "panel"])
            .agg(n_trials=("trial_idx", "size"),
                 sum_encode_ms=("time_in_relevant_fixations_ms", "sum"),
                 sum_search_ms=("trial_in_search_durations_ms", "sum"))
            .reset_index())
    # inner join: a panel needs both trials and a score; unmatched rows are logged below
    df = sums.merge(scores[["participant", "panel", "S_panel"]], on=["participant", "panel"],
                    how="left", validate="1:1", indicator=True)
    unmatched = df[df["_merge"] != "both"]
    df = df.drop(columns="_merge")
    df = df.merge(participants[["participant", GROUP, "Gender"]], on="participant", how="left", validate="m:1")

    total_ms = PANEL_SECONDS * 1000.0
    df["time_T_item"] = total_ms / df["S_panel"]
    df["time_T_encode"] = df["sum_encode_ms"] / df["S_panel"]
    df["time_T_search"] = df["sum_search_ms"] / df["S_panel"]
    df["time_T_rest"] = df["time_T_item"] - df["time_T_encode"] - df["time_T_search"]
    for comp in ["encode", "search", "rest"]:
        df[f"time_share_{comp}"] = df[f"time_T_{comp}"] / df["time_T_item"]
    df["coverage_encode_plus_search"] = (df["sum_encode_ms"] + df["sum_search_ms"]) / total_ms
    df["flag_trials_ne_S_minus_2"] = df["n_trials"] != (df["S_panel"] - 2)
    return df, unmatched


def panel_reliability(df_panel):
    """Split-half reliability of each panel-level T value with the pipeline's own
    feature_pipeline.calculate_reliability (random panel halves, RELIABILITY_REPETITIONS
    permutations), run at two half sizes: L=2 (participants with >= MIN_PANELS_RELIABILITY
    panels, the pipeline default) and L=3 (participants with all 6 panels, 3 vs 3).
    Spearman-Brown (Spearman 1910; Brown 1910) corrects the mean half-vs-half r to the
    reliability of a 2L-panel mean: r_SB = 2r / (1 + r)."""
    rows = []
    for feature in T_COLS + SHARE_COLS:
        for min_count in (MIN_PANELS_RELIABILITY, 6):
            for grp_name, sub in [("pooled", df_panel)] + [(g, df_panel[df_panel[GROUP] == g]) for g in GROUP_ORDER]:
                valid, mat = build_reliability_matrix(sub, feature, min_count)
                if valid is None:
                    continue
                np.random.seed(RANDOM_SEED)
                r_all, _ = calculate_reliability(mat, min_count // 2, RELIABILITY_REPETITIONS, min_count, feature,
                                                 plot=False, return_details=True)
                r = float(np.nanmean(r_all))
                rows.append({"feature": feature, "group": grp_name, "panels_per_half": min_count // 2,
                             "N": len(valid), "mean_r_half": r,
                             "r_half_2.5pct": float(np.nanpercentile(r_all, 2.5)),
                             "r_half_97.5pct": float(np.nanpercentile(r_all, 97.5)),
                             "spearman_brown": 2 * r / (1 + r)})
    return pd.DataFrame(rows)


def bland_altman(df_part):
    """Bland & Altman (1986): agreement between the summed components (T_encode +
    T_search) and the item time 90/S, per participant (mean of panels)."""
    a = df_part["time_T_encode"] + df_part["time_T_search"]
    b = df_part["time_T_item"]
    diff, mean = a - b, (a + b) / 2
    bias, sd = diff.mean(), diff.std(ddof=1)
    res = {"N": int(diff.notna().sum()), "bias_ms": bias, "sd_diff_ms": sd,
           "loa_low_ms": bias - 1.96 * sd, "loa_high_ms": bias + 1.96 * sd}
    for g in GROUP_ORDER:
        d = diff[df_part[GROUP] == g]
        res[f"bias_ms_{g}"] = d.mean()
        res[f"loa_low_ms_{g}"] = d.mean() - 1.96 * d.std(ddof=1)
        res[f"loa_high_ms_{g}"] = d.mean() + 1.96 * d.std(ddof=1)
    return res, diff, mean


# ---------------------------------------------------------------- plots
def plot_components(df_part):
    long = df_part.melt(id_vars=["participant", GROUP], value_vars=T_COLS, var_name="component", value_name="ms")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(data=long, x="component", y="ms", hue=GROUP, hue_order=GROUP_ORDER, palette=PALETTE,
                showfliers=False, ax=ax)
    sns.stripplot(data=long, x="component", y="ms", hue=GROUP, hue_order=GROUP_ORDER, palette=PALETTE,
                  dodge=True, size=2.5, alpha=0.6, legend=False, ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ms per item (mean of panels)")
    ax.set_xlabel("")
    ax.set_title("Time budget per item: components by group")
    _save_fig(fig, "A1_components_by_group.png", df_part)


def plot_bland_altman(df_part, diff, mean, ba):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    a = df_part["time_T_encode"] + df_part["time_T_search"]
    for g in GROUP_ORDER:
        m = df_part[GROUP] == g
        ax.scatter(df_part.loc[m, "time_T_item"], a[m], s=14, color=PALETTE[g], label=g, alpha=0.8)
    lim = [min(a.min(), df_part["time_T_item"].min()) * 0.95, max(a.max(), df_part["time_T_item"].max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--", label="identity")
    ax.set_xlabel("T_item = 90 000 / S (ms)")
    ax.set_ylabel("T_encode + T_search (ms)")
    ax.set_title("Summed components vs item time")
    ax.legend(loc="lower right")
    ax = axes[1]
    for g in GROUP_ORDER:
        m = df_part[GROUP] == g
        ax.scatter(mean[m], diff[m], s=14, color=PALETTE[g], label=g, alpha=0.8)
    ax.axhline(ba["bias_ms"], color="black", lw=1, label=f"bias {ba['bias_ms']:.0f} ms")
    for k in ("loa_low_ms", "loa_high_ms"):
        ax.axhline(ba[k], color="black", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("mean of (T_encode + T_search) and T_item (ms)")
    ax.set_ylabel("(T_encode + T_search) - T_item (ms)")
    ax.set_title(f"Bland-Altman: LoA [{ba['loa_low_ms']:.0f}, {ba['loa_high_ms']:.0f}] ms")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, "A2_bland_altman.png", df_part)


def plot_rest_distribution(df_panel, df_part):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (data, lvl) in zip(axes, [(df_panel, "participant x panel"), (df_part, "participant (mean of panels)")]):
        sns.histplot(data=data, x="time_T_rest", hue=GROUP, hue_order=GROUP_ORDER, palette=PALETTE,
                     element="step", bins=40, ax=ax)
        ax.axvline(0, color="black", lw=1)
        neg = (data["time_T_rest"] < 0).mean() * 100
        ax.set_title(f"T_rest, {lvl}: {neg:.0f}% negative")
        ax.set_xlabel("T_rest (ms per item)")
    fig.tight_layout()
    _save_fig(fig, "A3_T_rest_distribution.png", df_part)


def plot_encode_by_searches(trials, df_part):
    t = trials.copy()
    t["searches"] = t["num_of_searches_per_trial"].clip(upper=3).astype(int).astype(str).replace({"3": "3+"})
    t = t[t["time_in_relevant_fixations_ms"] > 0]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.boxplot(data=t, x="searches", y="time_in_relevant_fixations_ms", order=["0", "1", "2", "3+"],
                color=sns.color_palette("colorblind")[2], showfliers=False, ax=ax)
    ax.set_xlabel("dictionary searches in trial")
    ax.set_ylabel("time_in_relevant_fixations_ms (first-to-last span)")
    ax.set_title("Encode span grows with each dictionary visit\n(trials with >= 1 target fixation)")
    _save_fig(fig, "A4_encode_span_by_num_searches.png", df_part)


def plot_reliability(rel, df_part, df_panel):
    sub = rel[rel["panels_per_half"] == 3].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.pointplot(data=sub, y="feature", x="spearman_brown", hue="group", dodge=0.4, linestyle="none",
                  palette=["grey"] + [PALETTE[g] for g in GROUP_ORDER], hue_order=["pooled"] + GROUP_ORDER, ax=ax)
    for x, ls in [(0.7, "--"), (0.5, ":")]:
        ax.axvline(x, color="black", lw=0.8, ls=ls)
    ax.set_xlabel("Spearman-Brown split-half reliability (3 vs 3 panels)")
    ax.set_ylabel("")
    ax.set_title("Panel-level reliability of time-budget features")
    six = df_panel.groupby("participant").size().loc[lambda c: c >= 6].index
    _save_fig(fig, "A5_split_half_reliability.png", df_part[df_part["participant"].isin(six)])


def plot_shares(df_part):
    d = df_part.sort_values("time_T_item")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors = sns.color_palette("colorblind")[2:5]
    for ax, g in zip(axes, GROUP_ORDER):
        s = d[d[GROUP] == g].reset_index(drop=True)
        left = np.zeros(len(s))
        pos = s[["time_T_encode", "time_T_search"]].values
        for i, (col, c) in enumerate(zip(["time_T_encode", "time_T_search"], colors)):
            ax.barh(np.arange(len(s)), s[col], left=left, color=c, label=col.replace("time_", ""), height=0.9)
            left += s[col].values
        rest = s["time_T_rest"].values
        ax.barh(np.arange(len(s)), np.where(rest > 0, rest, 0), left=left, color=colors[2], label="T_rest (>0)", height=0.9)
        # negative T_rest = the part of encode+search beyond T_item (double-counted time)
        ax.barh(np.arange(len(s)), np.where(rest < 0, rest, 0), left=left, facecolor="none", edgecolor="black",
                hatch="////", lw=0, label="T_rest < 0 (double-counted)", height=0.9)
        ax.scatter(s["time_T_item"], np.arange(len(s)), marker="|", color="black", s=20, label="T_item = 90/S")
        ax.set_title(f"{g} (sorted by T_item)")
        ax.set_xlabel("ms per item")
        ax.set_yticks([])
    axes[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    _save_fig(fig, "A6_time_budget_per_participant.png", df_part)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    trials, panels, participants, scores = load_inputs()

    df_panel, unmatched = build_panel_time_budget(trials, scores, participants)
    rel = panel_reliability(df_panel)

    # participant level: mean of panel values
    agg_cols = T_COLS + SHARE_COLS + ["coverage_encode_plus_search"]
    df_part = (df_panel.groupby("participant")[agg_cols].mean()
               .join(df_panel.groupby("participant")["n_trials"].size().rename("n_panels"))
               .reset_index())
    df_part = df_part.merge(participants[["participant", GROUP, "Gender", "score"]], on="participant",
                            how="left", validate="1:1")
    ba, diff, mean = bland_altman(df_part)

    neg_panels = df_panel[df_panel["time_T_rest"] < 0]
    flagged = df_panel[df_panel["flag_trials_ne_S_minus_2"]]

    # merged tables (explicit join types; canonical tables are NOT overwritten)
    new_panel_cols = ["participant", "panel", "S_panel", "n_trials"] + T_COLS + SHARE_COLS + \
                     ["coverage_encode_plus_search", "flag_trials_ne_S_minus_2"]
    panels_merged = panels.merge(df_panel[new_panel_cols], on=["participant", "panel"], how="left",
                                 validate="1:1", indicator="_join_time_budget")
    part_merged = participants.merge(df_part[["participant", "n_panels"] + agg_cols], on="participant",
                                     how="left", validate="1:1", indicator="_join_time_budget")

    df_panel.to_csv(os.path.join(TAB_DIR, "panel_time_budget.csv"), index=False)
    df_part.to_csv(os.path.join(TAB_DIR, "participant_time_budget.csv"), index=False)
    rel.to_csv(os.path.join(TAB_DIR, "time_budget_split_half_reliability.csv"), index=False)
    pd.DataFrame([ba]).to_csv(os.path.join(TAB_DIR, "bland_altman.csv"), index=False)
    neg_panels.to_csv(os.path.join(TAB_DIR, "negative_T_rest_panels.csv"), index=False)
    flagged.to_csv(os.path.join(TAB_DIR, "panels_trials_ne_S_minus_2.csv"), index=False)
    panels_merged.drop(columns="_join_time_budget").to_csv(os.path.join(TAB_DIR, "panel_features_with_time_budget.csv"), index=False)
    part_merged.drop(columns="_join_time_budget").to_csv(os.path.join(TAB_DIR, "participant_features_with_time_budget.csv"), index=False)

    desc = df_part.groupby(GROUP)[T_COLS + SHARE_COLS].describe().T
    desc.to_csv(os.path.join(TAB_DIR, "time_budget_descriptives_by_group.csv"))

    plot_components(df_part)
    plot_bland_altman(df_part, diff, mean, ba)
    plot_rest_distribution(df_panel, df_part)
    plot_encode_by_searches(trials, df_part)
    plot_reliability(rel, df_part, df_panel)
    plot_shares(df_part)

    provenance = {
        "script": os.path.relpath(__file__, REPO), "git_commit": _git_commit(),
        "run_at": datetime.now().isoformat(timespec="seconds"), "random_seed": RANDOM_SEED,
        "inputs": {p: {"md5": _md5(p), "rows": int(len(pd.read_csv(p)))} for p in
                   [TRIAL_CSV, PANEL_CSV, PARTICIPANT_CSV, SCORES_CSV]},
        "joins": {
            "trial sums -> all_scores (participant, panel)": {"how": "left", "validate": "1:1",
                                                              "unmatched": int(len(unmatched))},
            "panel_features -> panel_time_budget": {"how": "left", "validate": "1:1",
                                                    **panels_merged["_join_time_budget"].value_counts().astype(int).to_dict()},
            "participant_features -> participant_time_budget": {"how": "left", "validate": "1:1",
                                                                **part_merged["_join_time_budget"].value_counts().astype(int).to_dict()},
        },
        "definitions": {
            "time_T_item": "90000 ms / S_panel",
            "time_T_encode": "sum(time_in_relevant_fixations_ms over panel trials) / S_panel (first-to-last target-fixation span)",
            "time_T_search": "sum(trial_in_search_durations_ms over panel trials) / S_panel",
            "time_T_rest": "time_T_item - time_T_encode - time_T_search",
            "participant value": "mean of panel values",
        },
    }
    with open(os.path.join(OUT_DIR, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    print(f"panels: {len(df_panel)}  participants: {len(df_part)}  unmatched score rows: {len(unmatched)}")
    print(df_part.groupby(GROUP)[T_COLS + ["coverage_encode_plus_search"]].median().round(1).to_string())
    print("Bland-Altman:", {k: round(v, 1) if isinstance(v, float) else v for k, v in ba.items()})
    print(f"negative T_rest panels: {len(neg_panels)}/{len(df_panel)}; participants with mean T_rest<0: "
          f"{(df_part['time_T_rest'] < 0).sum()}/{len(df_part)}")
    print("panels with n_trials != S-2:"); print(flagged[["participant", "panel", "S_panel", "n_trials"]].to_string())
    print(rel[rel.group == "pooled"].round(3).to_string())
    print("joins:", json.dumps(provenance["joins"]))
    print("out:", OUT_DIR)


if __name__ == "__main__":
    main()

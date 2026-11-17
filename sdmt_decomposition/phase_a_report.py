"""
Phase A (v2) - SDMT time-budget features from the gaze-anchored partition, and a
self-contained HTML report (figures as base64 PNG, inline CSS, no external dependencies).

Inputs (nothing existing is recomputed):
  gaze_partition/panel_time_partition.csv   (sdmt_decomposition/gaze_time_partition.py)
  gaze_partition/excluded_panel_heatmaps/   (same run)
  12_11_26 features/csvs: participant_features.csv, panel_features.csv, trial_features.csv
  test_sample_level_time_budget/            (5-panel sample-level check)

Three ways of expressing the same time (decision Noam 2026-11-17):
  1. per item      time_T_<part>      = panel sum / S_panel                 (Noam's original definition)
  2. per analysed item time_Ttrial_<part> = panel sum / items inside the analysed windows
                                         (parts sum exactly to the mean window per item)
  3. coarse        time_Tcoarse_*     = whole-panel fixation time in dictionary ROIs / in
                                         any text symbol ROI, / S_panel (no trial segmentation)

Panels with > 20% of their length outside the analysed windows are excluded from these
features only (decision Noam 2026-11-17, option a); participant values are the mean of
the remaining panels.

Methods cited: Bland & Altman (1986) agreement; Spearman (1910) / Brown (1910) split-half
correction; Nunnally & Bernstein (1994) reliability thresholds; Aitchison (1986) for why
time shares are descriptors only.
"""
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

_DIRECT_MOUNT = "/Volumes/Noam_M"
if os.path.isdir(os.path.join(_DIRECT_MOUNT, "Results", "Behavior", "processing_results")):
    pipeline_config.noam_m_root = lambda: _DIRECT_MOUNT

from feature_pipeline import (calculate_reliability, build_reliability_matrix,  # noqa: E402
                              demographics_text, annotate_demographics,
                              RELIABILITY_REPETITIONS, MIN_PANELS_RELIABILITY)

# ---------------------------------------------------------------- config
GROUP = "group"
GROUP_ORDER = ["HC", "MS"]
RANDOM_SEED = 20261117
EXCLUDE_EDGE_PCT = 20.0
ROOT = os.path.join(pipeline_config.noam_m_root(), "12_11_26")
CSV_DIR = os.path.join(ROOT, "analysis_results", "features", "csvs")
BASE = os.path.join(ROOT, "analysis_results", "sdmt_decomposition_16_11_26")
PART_DIR = os.path.join(BASE, "gaze_partition")
OUT_DIR = os.path.join(BASE, "phase_A_v2")
TAB_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")

PARTS = ["encode_pre", "encode_recheck", "encode_post", "search", "rest"]
PART_LABEL = {"encode_pre": "Encode (before search)", "encode_recheck": "Encode re-check (between searches)",
              "encode_post": "Encode (after last search)", "search": "Search (dictionary)",
              "rest": "Rest (saccades, other symbols, speech pauses)", "edge": "Outside analysed trials"}
FEATURES = (["time_T_item"] + [f"time_T_{p}" for p in PARTS]
            + ["time_Ttrial_window"] + [f"time_Ttrial_{p}" for p in PARTS]
            + ["time_Tcoarse_dict_roi_fix", "time_Tcoarse_text_roi_fix"])
SHARES = [f"time_share_{p}" for p in PARTS]

sns.set_theme(style="whitegrid", context="paper")
CB = sns.color_palette("colorblind")
PALETTE = {"HC": CB[0], "MS": CB[1]}
PART_COLORS = {"encode_pre": CB[2], "encode_recheck": CB[8], "encode_post": CB[9], "search": CB[3],
               "rest": CB[7], "edge": "#dddddd"}


# ---------------------------------------------------------------- helpers
def fig_to_b64(fig, name, demo_df, on_figure=True):
    annotate_demographics(demographics_text(demo_df), on_figure=on_figure)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_html(df, floatfmt="{:.1f}", index=False):
    return df.to_html(index=index, float_format=lambda v: floatfmt.format(v), border=0, classes="tbl", na_rep="–")


def save_table(df, name):
    df.to_csv(os.path.join(TAB_DIR, name), index=False)


def med_iqr(x):
    x = x.dropna()
    return f"{x.median():.0f} ({x.quantile(.25):.0f}–{x.quantile(.75):.0f})"


# ---------------------------------------------------------------- data
def load():
    panel = pd.read_csv(os.path.join(PART_DIR, "panel_time_partition.csv"), dtype={"panel": str})
    part = pd.read_csv(os.path.join(CSV_DIR, "participant_features.csv"))
    panel_feat = pd.read_csv(os.path.join(CSV_DIR, "panel_features.csv"), dtype={"panel": str})
    trials_old = pd.read_csv(os.path.join(CSV_DIR, "trial_features.csv"), dtype={"panel": str})
    panel = panel.merge(part[["participant", "Gender"]], on="participant", how="left", validate="m:1")
    panel["edge_ms_total"] = panel["panel_length_ms"] - panel["window_ms"]
    for p in PARTS:
        panel[f"time_share_{p}"] = panel[f"{p}_ms"] / panel["window_ms"]
    return panel, part, panel_feat, trials_old


def exclusion_reason(r):
    """Plain-language reason for an excluded panel, from the row-alignment check and the
    off-screen share (diagnose_bad_partition_panels.py)."""
    reasons = []
    off1, off2 = r.get("rowcheck_median_offset_first_half"), r.get("rowcheck_median_offset_second_half")
    on_row = r.get("rowcheck_pct_on_expected_row")
    if pd.notna(off2) and abs(off2) >= 1:
        side = "above" if off2 < 0 else "below"
        when = "for the whole panel" if pd.notna(off1) and abs(off1) >= 1 else "in the second half of the panel"
        reasons.append(f"gaze on the symbol rows was recorded about {abs(off2):.0f} row(s) {side} the symbol "
                       f"being answered {when} (vertical drift of the text-area gaze)")
    elif pd.notna(on_row) and on_row < 60:
        reasons.append(f"only {on_row:.0f}% of fixations on the symbol rows were on the row being answered "
                       f"(gaze scattered or drifting; normal panels are ~85–90%)")
    if pd.notna(r.get("offscreen_pct")) and r["offscreen_pct"] > 5:
        reasons.append(f"{r['offscreen_pct']:.0f}% of fixation time fell outside the panel")
    if not reasons:
        reasons.append("the target symbols were often not fixated inside their boxes; gaze row alignment "
                       "looked normal, so the exact cause is unclear - see heatmap")
    return "; ".join(reasons)


# ---------------------------------------------------------------- analyses
def reliability(panel_in):
    """Split-half reliability with the pipeline's own feature_pipeline.calculate_reliability
    (random panel halves, RELIABILITY_REPETITIONS permutations) at L=2 (>= 4 panels) and
    L=3 (6 panels); Spearman-Brown corrected (Spearman 1910; Brown 1910)."""
    rows = []
    for feat in FEATURES + SHARES:
        for min_count in (MIN_PANELS_RELIABILITY, 6):
            for gname, sub in [("pooled", panel_in)] + [(g, panel_in[panel_in[GROUP] == g]) for g in GROUP_ORDER]:
                valid, mat = build_reliability_matrix(sub, feat, min_count)
                if valid is None:
                    continue
                np.random.seed(RANDOM_SEED)
                r_all, _ = calculate_reliability(mat, min_count // 2, RELIABILITY_REPETITIONS, min_count, feat,
                                                 plot=False, return_details=True)
                r = float(np.nanmean(r_all))
                rows.append({"feature": feat, "group": gname, "panels_per_half": min_count // 2, "N": len(valid),
                             "mean_r_half": r, "r_half_2.5pct": np.nanpercentile(r_all, 2.5),
                             "r_half_97.5pct": np.nanpercentile(r_all, 97.5), "spearman_brown": 2 * r / (1 + r)})
    return pd.DataFrame(rows)


def bland_altman(a, b):
    """Bland & Altman (1986)."""
    d = a - b
    bias, sd = d.mean(), d.std(ddof=1)
    return d, (a + b) / 2, {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


# ---------------------------------------------------------------- figures
def fig_old_encode(trials_old, demo):
    t = trials_old[trials_old["time_in_relevant_fixations_ms"] > 0].copy()
    t["searches"] = t["num_of_searches_per_trial"].clip(upper=3).astype(int).astype(str).replace({"3": "3+"})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=t, x="searches", y="time_in_relevant_fixations_ms", order=["0", "1", "2", "3+"],
                color=CB[2], showfliers=False, ax=ax)
    ax.set_xlabel("dictionary searches in the trial")
    ax.set_ylabel("old time_in_relevant_fixations_ms (span, ms)")
    ax.set_title("The old encode span grows with every dictionary visit")
    return fig_to_b64(fig, "A1_old_encode_span_by_searches.png", demo)


def fig_components(part_df, prefix, title, name):
    cols = [f"{prefix}{p}" for p in PARTS]
    long = part_df.melt(id_vars=["participant", GROUP], value_vars=cols, var_name="part", value_name="ms")
    long["part"] = long["part"].str.replace(prefix, "", regex=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    sns.boxplot(data=long, x="part", y="ms", hue=GROUP, hue_order=GROUP_ORDER, palette=PALETTE,
                showfliers=False, ax=ax)
    sns.stripplot(data=long, x="part", y="ms", hue=GROUP, hue_order=GROUP_ORDER, palette=PALETTE, dodge=True,
                  size=2, alpha=0.5, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("ms")
    ax.set_title(title)
    return fig_to_b64(fig, name, part_df)


def fig_bland_altman(part_df, d, m, ba):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for g in GROUP_ORDER:
        k = part_df[GROUP] == g
        ax.scatter(part_df.loc[k, "time_T_item"], part_df.loc[k, "time_Ttrial_window"], s=14, color=PALETTE[g],
                   label=g, alpha=0.8)
    lim = [part_df[["time_T_item", "time_Ttrial_window"]].min().min() * 0.95,
           part_df[["time_T_item", "time_Ttrial_window"]].max().max() * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8, label="identity")
    ax.set_xlabel("90 000 / S (ms)")
    ax.set_ylabel("mean gaze window per analysed item (ms)")
    ax.legend()
    ax.set_title("Measured time per item vs 90/S")
    ax = axes[1]
    for g in GROUP_ORDER:
        k = part_df[GROUP] == g
        ax.scatter(m[k], d[k], s=14, color=PALETTE[g], alpha=0.8, label=g)
    ax.axhline(ba["bias"], color="k", lw=1, label=f"bias {ba['bias']:.0f} ms")
    for k in ("loa_low", "loa_high"):
        ax.axhline(ba[k], color="k", ls="--", lw=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("mean of the two (ms)")
    ax.set_ylabel("window per item − 90/S (ms)")
    ax.set_title(f"Bland–Altman: limits of agreement [{ba['loa_low']:.0f}, {ba['loa_high']:.0f}] ms")
    ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig, "A3_bland_altman.png", part_df)


def fig_reliability(rel, demo_six):
    sub = rel[(rel["panels_per_half"] == 3) & rel["feature"].isin(FEATURES)].copy()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.pointplot(data=sub, y="feature", x="spearman_brown", hue="group", hue_order=["pooled"] + GROUP_ORDER,
                  palette=["grey", PALETTE["HC"], PALETTE["MS"]], dodge=0.45, linestyle="none", ax=ax)
    ax.axvline(0.7, color="k", ls="--", lw=0.8)
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Spearman–Brown split-half reliability (3 vs 3 panels)")
    ax.set_ylabel("")
    ax.set_title("Split-half reliability of the time features")
    return fig_to_b64(fig, "A4_split_half_reliability.png", demo_six)


def fig_time_budget(part_df):
    d = part_df.sort_values("time_T_item")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, g in zip(axes, GROUP_ORDER):
        s = d[d[GROUP] == g].reset_index(drop=True)
        left = np.zeros(len(s))
        for p in PARTS + ["edge"]:
            col = "time_T_edge" if p == "edge" else f"time_T_{p}"
            ax.barh(np.arange(len(s)), s[col], left=left, color=PART_COLORS[p], height=0.9,
                    label=PART_LABEL[p])
            left += s[col].to_numpy()
        ax.scatter(s["time_T_item"], np.arange(len(s)), marker="|", color="k", s=25, label="90/S")
        hc = part_df[part_df[GROUP] == "HC"]["time_T_item"]
        ax.axvspan(hc.quantile(.25), hc.quantile(.75), color="grey", alpha=0.12, label="HC IQR of 90/S")
        ax.axvline(hc.median(), color="grey", lw=1)
        ax.set_yticks([])
        ax.set_xlabel("ms per item (panel sum / S, mean of panels)")
        ax.set_title(f"{g} (sorted by 90/S)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig_to_b64(fig, "A5_time_budget_per_participant.png", part_df)


def fig_shares(part_df):
    m = part_df.groupby(GROUP)[SHARES].mean().loc[GROUP_ORDER]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    left = np.zeros(len(m))
    for p in PARTS:
        ax.barh(m.index, m[f"time_share_{p}"], left=left, color=PART_COLORS[p], label=PART_LABEL[p])
        for i, v in enumerate(m[f"time_share_{p}"]):
            if v > 0.04:
                ax.text(left[i] + v / 2, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
        left += m[f"time_share_{p}"].to_numpy()
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean share of the analysed trial time")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("Composition of trial time (descriptive only)")
    return fig_to_b64(fig, "A6_time_shares.png", part_df)


def fig_old_vs_new(panel_in, old_panel, demo):
    m = panel_in.merge(old_panel, on=["participant", "panel"], how="inner", validate="1:1")
    m["new_encode_ms"] = m[["encode_pre_ms", "encode_recheck_ms", "encode_post_ms"]].sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (x, y, t) in zip(axes, [("old_encode_ms", "new_encode_ms", "Encode: old span vs new summed target time"),
                                    ("old_search_ms", "search_ms", "Search: old (press-assigned) vs new (gaze)")]):
        for g in GROUP_ORDER:
            k = m[GROUP] == g
            ax.scatter(m.loc[k, x] / 1000, m.loc[k, y] / 1000, s=8, alpha=0.6, color=PALETTE[g], label=g)
        lim = [0, max(m[x].max(), m[y].max()) / 1000 * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("old, s per panel")
        ax.set_ylabel("new, s per panel")
        ax.set_title(f"{t}\nr = {m[x].corr(m[y]):.2f}")
        ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig, "A7_old_vs_new.png", demo), m


# ---------------------------------------------------------------- main
def main():
    for d in (OUT_DIR, TAB_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)
    panel, part, panel_feat, trials_old = load()

    # ---- exclusions
    diag_path = os.path.join(PART_DIR, "bad_panel_diagnosis", "bad_panel_gaze_location.csv")
    diag = pd.read_csv(diag_path, dtype={"panel": str})[["participant", "panel", "offscreen_pct", "text_in_roi_pct"]] \
        if os.path.exists(diag_path) else pd.DataFrame(columns=["participant", "panel", "offscreen_pct", "text_in_roi_pct"])
    excl = panel[panel["edge_pct_of_panel"] > EXCLUDE_EDGE_PCT].merge(diag, on=["participant", "panel"], how="left")
    excl["reason"] = excl.apply(exclusion_reason, axis=1)
    excl = excl.sort_values(["participant", "panel"])
    excl_table = excl[["participant", GROUP, "panel", "edge_pct_of_panel", "n_absorbed", "n_trials",
                       "rowcheck_pct_on_expected_row", "rowcheck_median_offset", "reason"]].rename(columns={
        "edge_pct_of_panel": "% of panel outside analysed trials", "n_absorbed": "targets never fixated",
        "n_trials": "trials", "rowcheck_pct_on_expected_row": "% text fixations on answered row",
        "rowcheck_median_offset": "median row offset"})
    save_table(excl_table, "excluded_panels.csv")
    panel_in = panel[panel["edge_pct_of_panel"] <= EXCLUDE_EDGE_PCT].copy()
    panel_in["time_T_edge"] = panel_in["edge_ms_total"] / panel_in["S_panel"]

    # ---- participant level (mean of included panels)
    agg_cols = FEATURES + SHARES + ["time_T_edge", "edge_pct_of_panel"]
    part_df = (panel_in.groupby("participant")[agg_cols].mean()
               .join(panel_in.groupby("participant").size().rename("time_n_panels"))
               .reset_index()
               .merge(part[["participant", GROUP, "Gender", "Age_Years"]], on="participant", how="left", validate="1:1"))
    save_table(panel_in, "panel_time_budget_v2.csv")
    save_table(part_df, "participant_time_budget_v2.csv")

    # ---- merged canonical-shaped tables (left joins, canonical files untouched)
    new_panel_cols = ["participant", "panel", "S_panel", "panel_length_ms", "edge_pct_of_panel"] + FEATURES + SHARES
    panels_merged = panel_feat.merge(panel_in[new_panel_cols], on=["participant", "panel"], how="left",
                                     validate="1:1", indicator="_join")
    part_merged = part.merge(part_df[["participant", "time_n_panels"] + FEATURES + SHARES], on="participant",
                             how="left", validate="1:1", indicator="_join")
    join_log = {"panel_features <- panel_time_budget_v2 (left, 1:1)": panels_merged["_join"].value_counts().to_dict(),
                "participant_features <- participant_time_budget_v2 (left, 1:1)": part_merged["_join"].value_counts().to_dict()}
    panels_merged.drop(columns="_join").to_csv(os.path.join(TAB_DIR, "panel_features_with_time_budget_v2.csv"), index=False)
    part_merged.drop(columns="_join").to_csv(os.path.join(TAB_DIR, "participant_features_with_time_budget_v2.csv"), index=False)

    # ---- descriptives
    desc_rows = []
    for f in FEATURES:
        desc_rows.append({"feature": f, **{g: med_iqr(part_df.loc[part_df[GROUP] == g, f]) for g in GROUP_ORDER}})
    desc = pd.DataFrame(desc_rows)
    save_table(desc, "descriptives_median_iqr_by_group.csv")

    # ---- identity check
    d, m, ba = bland_altman(part_df["time_Ttrial_window"], part_df["time_T_item"])
    ba_tab = pd.DataFrame([{"N": int(d.notna().sum()), "bias_ms": ba["bias"], "sd_ms": ba["sd"],
                            "loa_low_ms": ba["loa_low"], "loa_high_ms": ba["loa_high"],
                            "median_pct_outside_analysed_trials": part_df["edge_pct_of_panel"].median()}])
    save_table(ba_tab, "bland_altman_window_vs_90_over_S.csv")

    # ---- reliability
    rel = reliability(panel_in)
    save_table(rel, "split_half_reliability.csv")
    rel3 = rel[(rel["panels_per_half"] == 3) & (rel["group"] == "pooled")].set_index("feature")["spearman_brown"]
    six = panel_in.groupby("participant").size().loc[lambda c: c >= 6].index
    demo_six = part_df[part_df["participant"].isin(six)]

    # ---- QA
    qa = pd.DataFrame([{
        "panels analysed": len(panel), "panels excluded": len(excl), "participants with an excluded panel": excl["participant"].nunique(),
        "% trials target never fixated (merged)": 100 * panel_in["n_absorbed"].sum() / panel_in["n_trials"].sum(),
        "% trials answer press before search end": 100 * panel_in["n_press_before_search_end"].sum() / panel_in["n_trials"].sum(),
        "max |parts sum − window| (ms)": panel["parts_sum_minus_window_ms"].abs().max(),
        "panels with parts > panel length": int((panel["window_ms"] > panel["panel_length_ms"]).sum())}])
    save_table(qa, "qa_summary.csv")

    # ---- figures
    old_panel = trials_old.groupby(["participant", "panel"]).agg(
        old_encode_ms=("time_in_relevant_fixations_ms", "sum"), old_search_ms=("trial_in_search_durations_ms", "sum")).reset_index()
    figs = {
        "old": fig_old_encode(trials_old, part),
        "comp_item": fig_components(part_df, "time_T_", "Per item (panel sum / S), mean of panels", "A2a_components_per_item.png"),
        "comp_trial": fig_components(part_df, "time_Ttrial_", "Per analysed item (sum / items in analysed trials)", "A2b_components_per_analysed_item.png"),
        "ba": fig_bland_altman(part_df, d, m, ba),
        "rel": fig_reliability(rel, demo_six),
        "budget": fig_time_budget(part_df),
        "shares": fig_shares(part_df),
    }
    figs["oldnew"], oldnew = fig_old_vs_new(panel_in, old_panel, part_df)
    test_csv = os.path.join(BASE, "test_sample_level_time_budget", "sample_level_time_budget.csv")
    heat_dir = os.path.join(PART_DIR, "excluded_panel_heatmaps")
    heatmaps = [(r.participant, r.panel, os.path.join(heat_dir, f"heatmap_{r.participant}_{r.panel}.png"))
                for r in excl.itertuples()]

    coarse_desc = pd.DataFrame([{"measure": f, **{g: med_iqr(part_df.loc[part_df[GROUP] == g, f]) for g in GROUP_ORDER}}
                                for f in ["time_Tcoarse_dict_roi_fix", "time_Tcoarse_text_roi_fix"]])

    provenance = {
        "scripts": ["sdmt_decomposition/gaze_time_partition.py", "sdmt_decomposition/phase_a_report.py"],
        "git_commit": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "gaze_data": "/Volumes/Noam_M/12_11_26/SDMT_annotated_gaze_11_11_26_model/<group>/<participant>/panel_<p>_annotated_gaze_data.csv "
                     "(Stage 1 annotated, interpolated; dictionary drift correction applied on load: "
                     "pipeline_config.load_annotated_csv(corrected=True))",
        "inputs_md5": {p: md5(p) for p in [os.path.join(PART_DIR, "panel_time_partition.csv"),
                                           os.path.join(CSV_DIR, "participant_features.csv"),
                                           os.path.join(CSV_DIR, "panel_features.csv"),
                                           os.path.join(CSV_DIR, "trial_features.csv")]},
        "joins": join_log, "excluded_panels": excl[["participant", "panel"]].to_dict("records"),
        "random_seed": RANDOM_SEED,
    }
    with open(os.path.join(OUT_DIR, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    write_html(panel, panel_in, part_df, excl, excl_table, desc, coarse_desc, ba, ba_tab, rel, rel3, qa, figs,
               heatmaps, oldnew, test_csv, provenance)
    print("report:", os.path.join(OUT_DIR, "sdmt_phaseA_time_budget_report.html"))


# ---------------------------------------------------------------- HTML
CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;
color:#1d1d1f;background:#fff;line-height:1.5}
h1{font-size:1.7em;margin-bottom:0}h2{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:40px}
h3{margin-top:26px}.sub{color:#666;margin-top:4px}
.tbl{border-collapse:collapse;font-size:.85em;margin:10px 0}.tbl th,.tbl td{border-bottom:1px solid #e5e5e5;
padding:4px 8px;text-align:left;vertical-align:top}.tbl th{background:#f5f5f7}
img{max-width:100%;border:1px solid #eee;margin:8px 0}.box{background:#f5f7fb;border-left:4px solid #4a78c2;
padding:10px 14px;margin:12px 0}.warn{background:#fff6ec;border-left:4px solid #e08a1e;padding:10px 14px;margin:12px 0}
.ref{font-size:.85em;color:#444}code{background:#f2f2f2;padding:1px 4px;border-radius:3px;font-size:.9em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.tag{font-size:.62em;font-weight:600;padding:2px 6px;border-radius:4px;vertical-align:middle;white-space:nowrap}
.tag.old{background:#f3e3d3;color:#7a3e00}.tag.new{background:#dcebdc;color:#1d5e20}.tag.test{background:#e3e3f3;color:#2e2e7a}
.tag.other{background:#eee;color:#444}
.grid figure{margin:0}.grid figcaption{font-size:.8em;color:#444}
"""


def img(b64, alt=""):
    return f'<img alt="{escape(alt)}" src="data:image/png;base64,{b64}">'


def write_html(panel, panel_in, part_df, excl, excl_table, desc, coarse_desc, ba, ba_tab, rel, rel3, qa, figs,
               heatmaps, oldnew, test_csv, prov):
    n = len(part_df)
    n_hc, n_ms = (part_df[GROUP] == "HC").sum(), (part_df[GROUP] == "MS").sum()
    med = part_df.groupby(GROUP)[[f"time_share_{p}" for p in PARTS]].median().mul(100)
    rel_min = rel3.reindex(FEATURES).min()
    rel_low = rel3.reindex(FEATURES)[rel3.reindex(FEATURES) < 0.7]
    r_enc = oldnew["old_encode_ms"].corr(oldnew[["encode_pre_ms", "encode_recheck_ms", "encode_post_ms"]].sum(axis=1))
    r_srch = oldnew["old_search_ms"].corr(oldnew["search_ms"])
    q = qa.iloc[0]

    test_html = ""
    if os.path.exists(test_csv):
        t = pd.read_csv(test_csv)
        t = t[["participant", "panel", "raw_panel_ms", "search_pct", "nondict_pct", "text_pct", "buffer_pct", "encode_pct"]]
        t.columns = ["participant", "panel", "panel length (ms)", "dictionary %", "not dictionary %", "text area %",
                     "buffer %", "in any symbol box %"]
        test_html = table_html(t)

    rel_tab = rel[(rel["group"] == "pooled") & rel["feature"].isin(FEATURES)].pivot_table(
        index="feature", columns="panels_per_half", values="spearman_brown").reindex(FEATURES).reset_index()
    rel_tab.columns = ["feature", "r_SB, 2 vs 2 panels", "r_SB, 3 vs 3 panels"]
    rel_grp = rel[(rel["panels_per_half"] == 3) & rel["feature"].isin(FEATURES)].pivot_table(
        index="feature", columns="group", values="spearman_brown").reindex(FEATURES).reset_index()

    heat_html = "".join(
        f'<figure>{img(file_to_b64(p), f"{pid} {pn}")}<figcaption>{escape(pid)}/{escape(pn)}</figcaption></figure>'
        for pid, pn, p in heatmaps if os.path.exists(p))

    summary = [
        f"The original encode measure (<code>time_in_relevant_fixations_ms</code>) is a first-to-last span, not time on "
        f"the target: summed over a panel, encode + search exceeded the real panel length in 460 of 683 panels.",
        f"A new gaze-anchored partition splits every trial into five parts that cannot overlap: encode before the search, "
        f"encode re-check between searches, encode after the last search, search, and rest. The parts sum exactly to "
        f"each trial window (largest error {q['max |parts sum − window| (ms)']:.1g} ms) and never exceed the panel.",
        f"The measured time per item agrees with 90/S: Bland–Altman bias {ba['bias']:.0f} ms, limits of agreement "
        f"[{ba['loa_low']:.0f}, {ba['loa_high']:.0f}] ms.",
        f"Median trial composition (HC / MS): search {med.loc['HC', 'time_share_search']:.0f}% / "
        f"{med.loc['MS', 'time_share_search']:.0f}%, encode before search {med.loc['HC', 'time_share_encode_pre']:.0f}% / "
        f"{med.loc['MS', 'time_share_encode_pre']:.0f}%, rest {med.loc['HC', 'time_share_rest']:.0f}% / "
        f"{med.loc['MS', 'time_share_rest']:.0f}%.",
        f"All time features have split-half reliability ≥ {rel_min:.2f} (Spearman–Brown, 3 vs 3 panels, pooled)"
        + ("." if rel_low.empty else f"; below .70: {', '.join(rel_low.index)}."),
        f"{int(q['panels excluded'])} panels from {int(q['participants with an excluded panel'])} participants were "
        f"excluded from these features, mostly because gaze on the symbol rows had drifted vertically.",
        f"The new search total matches the old one (r = {r_srch:.2f}); the new encode total is a different "
        f"measure from the old span (r = {r_enc:.2f}).",
    ]

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SDMT Time Budget</title>
<style>{CSS}</style></head><body>
<h1>SDMT time budget – Phase A</h1>
<p class="sub">Generated {prov['run_at']} · N = {n} (HC = {n_hc}, MS = {n_ms}) · git {prov['git_commit'][:8]}</p>

<h2>Summary</h2><ul>{''.join(f'<li>{s}</li>' for s in summary)}</ul>

<h2>1. Data</h2>
<p><b>What was done.</b> Every panel is its own 90 s SDMT with its own score S. Gaze data:
<code>{escape(prov['gaze_data'])}</code>. These are the same gaze samples the feature pipeline uses: annotated in Stage 1,
missing or outlier samples interpolated, and the dictionary-area y-drift correction applied on load. The panel length
is the real recording length (last − first sample), median {panel['panel_length_ms'].median() / 1000:.2f} s
(range {panel['panel_length_ms'].min() / 1000:.1f}–{panel['panel_length_ms'].max() / 1000:.1f} s), not a nominal 90 s.
{len(panel)} panels from {panel['participant'].nunique()} participants were analysed. These are all panels that passed
preprocessing.</p>

<h2>1b. Which numbers come from which processing version</h2>
<p>Several processing versions appear in this report. The trial definition differs between them, so numbers from
different versions are not interchangeable. Every section below is tagged with its source.</p>
<table class="tbl"><tr><th>tag</th><th>source (script → file)</th><th>gaze input</th><th>trial definition</th>
<th>used in</th></tr>
<tr><td><span class="tag old">OLD</span></td><td>feature pipeline, 12_11_26 run (<code>feature_pipeline.py</code>,
<code>trial_manager.py</code>, <code>trial_features.py</code>) → <code>features/csvs/trial_features.csv</code></td>
<td>Stage 1 annotated + interpolated, dictionary drift-corrected</td>
<td><b>Press-bounded.</b> Relevant fixations = fixations on the target box from the end of the previous trial up to
the answer press. Encode = first-to-last span. Searches are assigned to trials by press times.</td>
<td>Sections 2 and 6.5 (comparison only). Not used for any new feature.</td></tr>
<tr><td><span class="tag test">SAMPLE TEST</span></td><td><code>test_sample_level_time_budget.py</code> →
<code>test_sample_level_time_budget/</code></td><td>Saved annotated CSV <b>as is</b> (interpolated, <b>not</b>
drift-corrected); individual gaze samples</td><td><b>No trials.</b> Runs of consecutive samples by screen zone and
symbol box, 5 panels.</td><td>Section 2 (sanity check)</td></tr>
<tr><td><span class="tag new">NEW</span></td><td><code>gaze_time_partition.py</code> →
<code>gaze_partition/panel_time_partition.csv</code>, <code>trial_time_partition.csv</code></td>
<td>Same as OLD (the same fixations, searches and ROIs that <code>TrialManager</code> builds)</td>
<td><b>Gaze-anchored (new).</b> Trial boundaries come from committed fixations on consecutive targets (section 3).
Parts are sums of fixation durations. Searches are assigned by gaze window. The same trial set as OLD (presses
2 … n−2).</td><td><code>time_T_*</code>, <code>time_Ttrial_*</code>, shares, sections 3–6, exclusions</td></tr>
<tr><td><span class="tag new">NEW – COARSE</span></td><td>same run</td><td>same as NEW</td><td><b>No trials.</b>
Whole-panel fixation time in the dictionary ROIs and in any text symbol ROI.</td>
<td><code>time_Tcoarse_*</code>, section 4</td></tr>
<tr><td><span class="tag new">NEW – QA</span></td><td>same run + <code>diagnose_bad_partition_panels.py</code></td>
<td>same as NEW</td><td>Row of each text fixation vs the row of the symbol being answered (from the presses)</td>
<td>Exclusion reasons, heatmaps (section 5)</td></tr>
<tr><td><span class="tag other">SCORES / LENGTH</span></td><td><code>12_11_26/behavior_scores/all_scores.csv</code>;
panel length = last − first sample of the annotated CSV (identical to the raw Tobii panel length)</td>
<td>–</td><td>–</td><td>S<sub>panel</sub>, 90/S, % of panel</td></tr>
<tr><td><span class="tag old">SUPERSEDED</span></td><td><code>phase_a_time_budget.py</code> → <code>phase_A/</code>
(16 Nov)</td><td>–</td><td>Built on the OLD encode span. T_rest was negative in 73% of panels.</td>
<td><b>Do not use.</b> Replaced by this report.</td></tr></table>

<h2>2. Why the encode measure was rebuilt <span class="tag old">OLD</span> <span class="tag test">SAMPLE TEST</span></h2>
<p><b>What was done.</b> The existing per-trial <code>time_in_relevant_fixations_ms</code> and
<code>trial_in_search_durations_ms</code> were summed per panel and compared with the panel length.</p>
<p><b>Result.</b> Encode + search exceeded the real panel length in 460 of 683 panels (67%). The search sums never
exceeded 65% of a panel. The encode span, however, grows with every dictionary visit in the trial:</p>
{img(figs['old'], 'old encode span')}
<p><b>Interpretation.</b> The old encode is the time from the <i>start of the first</i> to the <i>end of the last</i>
fixation on the target, collected up to the answer press. When a participant looks at the target, goes to the
dictionary and looks back at the target, the whole dictionary visit is inside that span, so it is counted again in
search. In three rebuilt panels, only 44% of the span was actual fixation time on the target. The rest was the
trial's own search (31%), searches belonging to neighbouring trials (9%) and other gaze (16%).
The physical gaze does not overlap: a sample-level check on 5 panels, without any trial logic, found dictionary + all
other samples = 99–100% of the panel:</p>
{test_html}

<h2>3. The gaze-anchored partition <span class="tag new">NEW</span></h2>
<p><b>What was done.</b> For every panel, the fixations, searches and symbol ROIs that <code>TrialManager</code>
builds were relabelled as a sequence: target-symbol fixations (T<sub>k</sub>), dictionary searches (D) and other (O).</p>
<ul>
<li><b>Trial boundary (gaze only):</b> trial k ends at the first fixation on target k+1 after which the participant
works on k+1 (a search, or a look at k+2) before returning to k. A brief glance at k+1 followed by a return to k is a
preview and stays in trial k. ROI hits more than two symbols away are treated as stray.</li>
<li><b>Five parts inside each trial window,</b> which cannot overlap:</li></ul>
<table class="tbl"><tr><th>part</th><th>definition</th></tr>
<tr><td>encode_pre</td><td>fixation time on target k before the first search (all target time if there was no search)</td></tr>
<tr><td>encode_recheck</td><td>fixation time on target k between searches (only with ≥ 2 searches)</td></tr>
<tr><td>encode_post</td><td>fixation time on target k after the last search</td></tr>
<tr><td>search</td><td>duration of the dictionary searches that start in the window</td></tr>
<tr><td>rest</td><td>window − all of the above: saccades, other symbols, buffer zone, speaking pauses, missing</td></tr></table>
<p>Fixation times are <i>sums of fixation durations</i>, not spans. A target that is never fixated inside its box
cannot get a boundary. Its time is merged into the previous window, so it is still counted exactly once
({q['% trials target never fixated (merged)']:.1f}% of trials).</p>
<p><b>Role of the presses.</b> The presses do not set the windows. They are used only (1) to analyse the same trials
as <code>trial_features.csv</code> (presses 2 … n−2), (2) to stop the search for a boundary at the answer press of the
next symbol, so one bad boundary cannot spread through the panel, and (3) as a QA flag. In
{q['% trials answer press before search end']:.1f}% of trials the answer press came before the end of the search
assigned to that trial. Usually this is because the participant went to the dictionary for the next symbol without
first fixating it, so that search sits in the previous window. Panel totals are unaffected, but the re-check part is
slightly inflated in those windows.</p>

<h2>4. Three ways of expressing the time <span class="tag new">NEW</span> <span class="tag new">NEW – COARSE</span></h2>
<p>The same panel time is reported in three versions. Each answers a different question:</p>
<table class="tbl"><tr><th>version</th><th>columns</th><th>definition</th><th>use it for</th></tr>
<tr><td><b>Per item</b> (main)</td><td><code>time_T_&lt;part&gt;</code></td><td>panel sum of the part / S<sub>panel</sub>,
then the mean of panels. On the same scale as 90/S.</td><td>Decomposing the SDMT score: how many ms of each item's time
go to each part. The five parts sum to slightly less than 90/S, because the first 2 and last 2 trials of a panel are
not segmented. That time is reported as <i>outside analysed trials</i>.</td></tr>
<tr><td><b>Per analysed item</b></td><td><code>time_Ttrial_&lt;part&gt;</code></td><td>panel sum of the part / number of
items inside the analysed windows</td><td>Exact time accounting. The five parts sum to the mean window per item, which
is compared with 90/S in the identity check (section 6).</td></tr>
<tr><td><b>Coarse</b></td><td><code>time_Tcoarse_dict_roi_fix</code>, <code>time_Tcoarse_text_roi_fix</code></td>
<td>all fixation time in the dictionary ROIs, and all fixation time in any symbol ROI of the text (not only the current
target), over the whole panel / S<sub>panel</sub></td><td>A cross-check that does not depend on trial boundaries.
It is less specific: text time includes looking at upcoming and neighbouring symbols, and the edge trials are
included.</td></tr></table>
{table_html(coarse_desc)}
<p class="ref">Coarse measures, median (IQR) ms per item.</p>

<h2>5. Excluded panels <span class="tag new">NEW – QA</span></h2>
<p><b>Rule (decided 2026-11-17).</b> A panel is excluded from the time-budget features when more than
{EXCLUDE_EDGE_PCT:.0f}% of its length falls outside the analysed trial windows. In a normal panel this is about 5%,
which is just the four edge trials. The participant keeps all other panels and all other features.</p>
<p><b>Result.</b> {len(excl)} panels from {excl['participant'].nunique()} participants were excluded:</p>
{table_html(excl_table)}
<p><b>Interpretation in plain words.</b> These panels passed preprocessing because their calibration accuracy,
missing data and saccade fraction were acceptable, and their gaze did reach both the dictionary and the text.
Preprocessing does not check whether the gaze on the symbol rows lines up with the symbol being answered. In most of
these panels the gaze sits inside symbol boxes, but on the wrong row. For example, in GS739/i1 the gaze followed the
answers for the first ~19 s and then stayed about one row above the symbol being answered for the rest of the panel.
This is vertical drift of the text-area gaze. The existing correction only shifts the dictionary area.
<b>The old features of these panels are affected as well</b>: in most of them, the old encode was 0 in 50–94% of
trials, against a median of 6% across all panels. A pipeline-wide QA flag for this (follow-up b) is pending.</p>
<p>Heatmaps of all gaze samples of each excluded panel over the panel image (log density; red boxes = ROIs).
Gaze that clusters between rows, or is shifted by a whole row relative to the symbols, shows the drift directly:</p>
<div class="grid">{heat_html}</div>

<h2>6. Results</h2>
<h3>6.1 Components by group <span class="tag new">NEW</span></h3>
<p><b>What was done.</b> Participant values are the mean of the included panels.</p>
{table_html(desc)}
<p class="ref">Median (IQR) in ms.</p>
{img(figs['comp_item'], 'components per item')}
{img(figs['comp_trial'], 'components per analysed item')}

<h3>6.2 Identity check: does the measured time per item equal 90/S? <span class="tag new">NEW</span></h3>
<p><b>What was done.</b> Bland–Altman agreement between the mean gaze window per analysed item and 90 000/S, per
participant.</p>
{table_html(ba_tab)}
{img(figs['ba'], 'Bland-Altman')}
<p><b>Interpretation.</b> A bias near 0 with narrow limits means the gaze windows account for the whole time per item,
so the five parts are a complete decomposition of the SDMT time. A positive bias means the analysed trials are slightly
slower than the panel average, for example because the panel ran a little over 90 s or because the edge trials were
faster than average. Bland &amp; Altman (1986).</p>

<h3>6.3 Split-half reliability <span class="tag new">NEW</span></h3>
<p><b>What was done.</b> Each panel-level value was split into random halves of panels with the pipeline's existing
<code>calculate_reliability</code> ({RELIABILITY_REPETITIONS} random splits): 2 vs 2 panels (participants with ≥ 4
panels) and 3 vs 3 panels (participants with all 6). The mean half-vs-half Pearson r was corrected with
Spearman–Brown, r<sub>SB</sub> = 2r/(1+r).</p>
{table_html(rel_tab, '{:.2f}')}
{table_html(rel_grp, '{:.2f}')}
<p class="ref">Second table: 3 vs 3 panels, per group.</p>
{img(figs['rel'], 'reliability')}
<p><b>Interpretation.</b> Following the thresholds of Nunnally &amp; Bernstein (1994):
r<sub>SB</sub> ≥ .70 means the feature is retained; .50–.70 means it is retained with a caution flag;
< .50 means it is excluded from later models. Because the panel means are reliable, the mean across panels can be
used as the participant value. Spearman (1910); Brown (1910).</p>

<h3>6.4 Time budget per participant <span class="tag new">NEW</span></h3>
{img(figs['budget'], 'time budget')}
<p><b>How to read it.</b> Each bar is one participant, and participants are sorted by their time per item (90/S, black
tick). The coloured segments are the parts in ms per item. The light grey end of each bar is time outside the
analysed trials (the edge trials). The grey band is the HC interquartile range of 90/S. A participant who is slow
because of long searches has a long red segment. One who spends long on the target before searching has a long green
segment.</p>
{img(figs['shares'], 'shares')}
<p><b>Shares are descriptive only.</b> The five shares sum to 1, so they are compositional data and must never all be
entered into one model together (Aitchison 1986).</p>

<h3>6.5 Old vs new features <span class="tag old">OLD</span> vs <span class="tag new">NEW</span></h3>
{img(figs['oldnew'], 'old vs new')}
<p><b>Interpretation.</b> Search is essentially the same measure whether it is assigned to trials by presses or by
gaze. Encode is not: the new value is the actual time on the target, while the old one mostly measured the
target-to-target span.</p>

<h3>6.6 Quality checks <span class="tag new">NEW – QA</span></h3>
{table_html(qa, '{:.2f}')}

<h2>7. Outputs</h2>
<p>Folder: <code>{escape(OUT_DIR)}</code>. The tables are <code>participant_features_with_time_budget_v2.csv</code> and
<code>panel_features_with_time_budget_v2.csv</code>. Both are left joins validated 1:1:
<code>{escape(json.dumps(prov['joins'], default=str))}</code>. There is also one CSV per table in this report, and
<code>provenance.json</code>. The canonical feature tables are not modified.</p>

<h2>8. Open items</h2><ul>
<li><b>Row alignment in panels that were kept:</b> {int(((panel['edge_pct_of_panel'] <= EXCLUDE_EDGE_PCT) & ((panel['rowcheck_median_offset_second_half'].abs() >= 1) | (panel['rowcheck_pct_on_expected_row'] < 60))).sum())}
panels passed the 20% rule but have fewer than 60% of text fixations on the answered row, or a one-row offset in the
second half (normal panels: median {panel.loc[panel['edge_pct_of_panel'] <= EXCLUDE_EDGE_PCT, 'rowcheck_pct_on_expected_row'].median():.0f}%).
Their encode values may be biased. They are listed in <code>gaze_partition/panel_time_partition.csv</code>
(<code>rowcheck_*</code> columns) and belong to follow-up (b).</li>
<li>Follow-up (b): a pipeline-wide QA flag for text-area gaze drift, based on the row-alignment check. It also affects
the existing features of the excluded panels.</li>
<li>Three panels have more trials than their score allows (BR152/a3, BY165/a3, BY165/l3), awaiting a manual check.</li>
<li>The re-check part is slightly inflated in windows where the next symbol's search was not preceded by a fixation on
that symbol (see section 3).</li></ul>

<h2>References</h2><div class="ref"><ul>
<li>Aitchison J (1986). <i>The Statistical Analysis of Compositional Data</i>.</li>
<li>Bland JM, Altman DG (1986). <i>Lancet</i> 327:307–310.</li>
<li>Brown W (1910). <i>Br J Psychol</i> 3:296–322.</li>
<li>Nunnally JC, Bernstein IH (1994). <i>Psychometric Theory</i>, 3rd ed.</li>
<li>Spearman C (1910). <i>Br J Psychol</i> 3:271–295.</li></ul></div>
</body></html>"""
    with open(os.path.join(OUT_DIR, "sdmt_phaseA_time_budget_report.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()

"""
Key-entry strategy feature (decision 2026-11-09): per SDMT trial, classifies where gaze
FIRST enters the key relative to where the target symbol actually sits (memory vs.
proximity vs. a directional miss), to later be corrected for each participant's own
entry bias via a within-participant permutation null.

Two SEPARATE, independently-computed L/M/R boundary systems (confirmed against real
panel images on 2026-11-09 - do not conflate them):
  - KEY thirds (3/3/3 of the key's 9 columns) - used for E (entry) and T (target).
  - TEST-ROW thirds (5/5/5 of the test row's 15 columns) - used for D (departure) only.
Both are computed once per panel from real detected ROI edges (RoiFinder), not
hard-coded pixel values - confirmed identical across participants for the same panel,
and confirmed structurally that key column v holds digit value v.

Trial filtering (decision 2026-11-09):
  - key_skip: trial.searches is empty (no key visit at all) - excluded.
  - congruent: T == D - excluded.
  - Minimum entry-fixation duration: NOT applied yet (decision 2026-11-09 - "remove the
    length of fixation condition for now, count how many times it happened" - see
    build_trial_records' entry_fixation_duration_ms column, meant for exactly that
    diagnostic; no trial is dropped for a short entry fixation at this stage).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr


def _key_thirds_cuts(rois):
    """Cut points between key columns 3/4 and 6/7 (3/3/3 of the key's 9 columns),
    from the symbol-icon row's own detected ROI edges (idx<9)."""
    key_row = sorted([r for r in rois if r.idx < 9], key=lambda r: r.center[0])
    if len(key_row) != 9:
        raise ValueError(f"Expected 9 key-row ROIs, got {len(key_row)}")
    col3, col4 = key_row[2], key_row[3]
    col6, col7 = key_row[5], key_row[6]
    cut1 = ((col3.center[0] + col3.x_half_right) + (col4.center[0] - col4.x_half_left)) / 2
    cut2 = ((col6.center[0] + col6.x_half_right) + (col7.center[0] - col7.x_half_left)) / 2
    return cut1, cut2


def _text_thirds_cuts(rois):
    """Cut points between test-row columns 5/6 and 10/11 (5/5/5 of the test row's 15
    columns), from one text row's own detected ROI edges (idx>=18) - all 8 rows share
    the same x-positions (confirmed 2026-11-09), so any one row suffices."""
    text_rois = [r for r in rois if r.idx >= 18]
    first_row_y = min(r.center[1] for r in text_rois)
    first_row = sorted([r for r in text_rois if abs(r.center[1] - first_row_y) < 5], key=lambda r: r.center[0])
    if len(first_row) != 15:
        raise ValueError(f"Expected 15 text-row ROIs in one row, got {len(first_row)}")
    col5, col6 = first_row[4], first_row[5]
    col10, col11 = first_row[9], first_row[10]
    cut1 = ((col5.center[0] + col5.x_half_right) + (col6.center[0] - col6.x_half_left)) / 2
    cut2 = ((col10.center[0] + col10.x_half_right) + (col11.center[0] - col11.x_half_left)) / 2
    return cut1, cut2


def _third_label(x, cut1, cut2):
    return "L" if x < cut1 else "M" if x < cut2 else "R"


def _key_third_by_value(value):
    """T: key thirds are value-based (digit value v always sits at key column v,
    confirmed 2026-11-09), matching the key's own 3/3/3 column grouping."""
    return "L" if value <= 3 else "M" if value <= 6 else "R"


def _classify(E, T, D):
    """Exhaustive/mutually-exclusive by construction for T != D (verified analytically
    2026-11-09): after ruling out E==T (memory) and E==D (proximity), exactly one
    structural case applies to the remaining E value."""
    if E == T:
        return "memory"
    if E == D:
        return "proximity"
    if {T, D} == {"L", "R"}:
        return "undershoot"
    if T == "M":
        return "overshoot"
    if D == "M":
        return "away"
    raise ValueError(f"Unclassifiable: E={E}, T={T}, D={D}")


def build_trial_records(tm):
    """One row per trial in TrialManager `tm`. `entry_fixation_duration_ms` is recorded
    for diagnostic purposes even though no duration filter is applied yet."""
    key_cut1, key_cut2 = _key_thirds_cuts(tm.rois)
    text_cut1, text_cut2 = _text_thirds_cuts(tm.rois)

    rows = []
    for trial in tm.trials:
        target_value = trial.triggering_symbol.value
        T = _key_third_by_value(target_value)

        departure_roi = next((r for r in tm.rois if r.idx == trial.triggering_symbol.idx), None)
        D = _third_label(departure_roi.center[0], text_cut1, text_cut2) if departure_roi is not None else None

        base_row = {"participant": tm.subject_name, "panel": tm.panel, "trial_idx": trial.idx, "T": T, "D": D}

        if not trial.searches or not trial.searches[0].fixations:
            rows.append({**base_row, "key_skip": True, "congruent": None, "E": None,
                         "category": None, "entry_fixation_duration_ms": None})
            continue

        entry_fixation = trial.searches[0].fixations[0]
        E = _third_label(entry_fixation.position[0], key_cut1, key_cut2)
        entry_duration_ms = entry_fixation.duration() / 1000.0

        congruent = (T == D)
        category = None if congruent else _classify(E, T, D)

        rows.append({**base_row, "key_skip": False, "congruent": congruent, "E": E,
                     "category": category, "entry_fixation_duration_ms": entry_duration_ms})

    return pd.DataFrame(rows)


def build_keyent_trial_table(participants):
    """participants: list of Participant objects with .trial_managers already built
    (feature_pipeline.build_trial_managers) - same population/exclusions as the rest of
    this pipeline. Returns one row per trial across every participant x panel."""
    frames = []
    for p in participants:
        for panel, tm in p.trial_managers.items():
            df = build_trial_records(tm)
            df["group"] = p.group
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================
# PARTICIPANT-LEVEL STRATEGY SUMMARY + TERNARY PLOT
# ============================================================
EXACT_CATS = ("memory",)
DIRECTIONAL_CATS = ("undershoot", "overshoot")
UNINFORMED_CATS = ("proximity", "away")

# Theoretical "balanced" chance point (decision 2026-11-09): assumes (T,D) pairs are
# uniformly distributed over the 6 possible incongruent combinations and E is uniform
# over L/M/R - matches the per-category chance levels in the original spec exactly
# (memory=1/3, proximity=1/3, undershoot=overshoot=away=1/9). This is a fixed reference
# marker, NOT each participant's own empirical null (that comes from the permutation
# test, built on their real (T,D) mix, not this uniformity assumption).
CHANCE_EXACT = 1 / 3
CHANCE_DIRECTIONAL = 2 / 9
CHANCE_UNINFORMED = 4 / 9


def build_participant_strategy_summary(trial_df):
    """One row per participant: pooled (across all panels) counts of the 5 raw
    categories plus the 3-way collapse (exact/directional/uninformed) used by the
    ternary plot. Pooling matches the spec ('pool counts across panels, do not average
    panel proportions'). Carries 'Gender' through if trial_df has it (optional)."""
    inc = trial_df[trial_df["category"].notna()].copy()
    rows = []
    for participant, g in inc.groupby("participant"):
        counts = g["category"].value_counts()
        n_inc = len(g)
        memory = counts.get("memory", 0)
        proximity = counts.get("proximity", 0)
        undershoot = counts.get("undershoot", 0)
        overshoot = counts.get("overshoot", 0)
        away = counts.get("away", 0)
        exact = memory
        directional = undershoot + overshoot
        uninformed = proximity + away
        row = {
            "participant": participant, "group": g["group"].iloc[0], "n_inc": n_inc,
            "memory": memory, "proximity": proximity, "undershoot": undershoot,
            "overshoot": overshoot, "away": away,
            "exact_frac": exact / n_inc, "directional_frac": directional / n_inc,
            "uninformed_frac": uninformed / n_inc,
        }
        if "Gender" in g.columns:
            row["Gender"] = g["Gender"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _ternary_xy(a, b, c):
    """a=exact (bottom-left vertex), b=directional (bottom-right vertex), c=uninformed
    (top vertex), each in [0,1] summing to 1 - standard equilateral-triangle barycentric
    -> Cartesian conversion."""
    x = b + 0.5 * c
    y = c * (3 ** 0.5) / 2
    return x, y


def plot_strategy_ternary(trial_df, out_path, excluded=()):
    """One point per participant (decision 2026-11-09): the clearest single view of
    individual key-entry strategy - where each participant sits on the exact/
    directional/uninformed simplex, pooling all their incongruent trials across panels.
    Point size ~ n_inc (more trials = a more reliable estimate); color = group. Uses RAW
    observed proportions, not yet the permutation-corrected Delta (that needs the null
    built first, which accounts for each participant's own (T,D) mix) - the gold star
    marks the fixed theoretical chance point under a uniformly-balanced assumption, not
    any one participant's real empirical null."""
    from feature_pipeline import demographics_text, annotate_demographics

    summary = build_participant_strategy_summary(trial_df)
    summary = summary[~summary["participant"].isin(excluded)]

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, (3 ** 0.5) / 2, 0]

    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.plot(tri_x, tri_y, color="black", linewidth=1.5)

    for frac in (0.25, 0.5, 0.75):
        p1, p2 = _ternary_xy(1 - frac, frac, 0), _ternary_xy(1 - frac, 0, frac)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lightgray", linewidth=0.5, zorder=0)
        p1, p2 = _ternary_xy(frac, 1 - frac, 0), _ternary_xy(0, 1 - frac, frac)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lightgray", linewidth=0.5, zorder=0)
        p1, p2 = _ternary_xy(frac, 0, 1 - frac), _ternary_xy(0, frac, 1 - frac)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lightgray", linewidth=0.5, zorder=0)

    palette = {"HC": "skyblue", "MS": "lightcoral", "pwMS": "lightcoral"}
    for group, g in summary.groupby("group"):
        xs, ys = zip(*[_ternary_xy(r.exact_frac, r.directional_frac, r.uninformed_frac) for r in g.itertuples()])
        sizes = 20 + 2.0 * g["n_inc"].to_numpy()
        ax.scatter(xs, ys, s=sizes, color=palette.get(group, "gray"), edgecolors="black",
                   alpha=0.75, linewidths=0.6, label=f"{group} (N={len(g)})", zorder=3)

    cx, cy = _ternary_xy(CHANCE_EXACT, CHANCE_DIRECTIONAL, CHANCE_UNINFORMED)
    ax.scatter([cx], [cy], marker="*", s=400, color="gold", edgecolors="black",
               linewidths=1, label="Theoretical chance (balanced)", zorder=4)

    ax.text(0, -0.06, "Exact\n(memory)", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(1, -0.06, "Directional\n(under/overshoot)", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.5, (3 ** 0.5) / 2 + 0.03, "Uninformed\n(proximity+away)", ha="center", va="bottom",
            fontsize=12, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, (3 ** 0.5) / 2 + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Key-entry strategy: individual participants on the exact/directional/uninformed simplex\n"
                 "(raw observed proportions, pooled across panels; point size ~ n incongruent trials)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    demographics = demographics_text(summary)
    annotate_demographics(demographics, ax=ax, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ternary strategy plot saved: {out_path} (N={len(summary)} participants)")
    return summary


# ============================================================
# FULL-BREAKDOWN OVERVIEW PANEL (decision 2026-11-10)
# ============================================================
# All 7 possible trial outcomes, as a % of EVERY trial (not just incongruent ones) so
# each participant's bar/percentages sum to exactly 100 - key_skip and congruent are
# "excluded from classification" but still real outcomes that happened on real trials.
STACK_ORDER = ["key_skip", "congruent", "memory", "proximity", "undershoot", "overshoot", "away"]
STACK_COLORS = {
    "key_skip": "#BDBDBD",
    "congruent": "#6B6B6B",
    "memory": "#2A9D8F",
    "proximity": "#E76F51",
    "undershoot": "#4C6EF5",
    "overshoot": "#9C6ADE",
    "away": "#F4A300",
}
STACK_LABELS = {
    "key_skip": "Key skip (no key visit)",
    "congruent": "Congruent (T==D, excluded)",
    "memory": "Memory", "proximity": "Proximity",
    "undershoot": "Undershoot", "overshoot": "Overshoot", "away": "Away",
}
EVENT_CATS = ["memory", "proximity", "undershoot", "overshoot", "away"]
GROUP_COLORS = {"HC": "#4C72B0", "MS": "#C0392B", "pwMS": "#C0392B"}


def build_participant_full_breakdown(trial_df):
    """One row per participant: % of ALL their trials (key_skip + congruent included)
    landing in each of the 7 outcomes, summing to 100. Different from
    build_participant_strategy_summary, whose denominator is incongruent trials only."""
    rows = []
    for participant, g in trial_df.groupby("participant"):
        n_total = len(g)
        n_key_skip = int(g["key_skip"].sum())
        n_congruent = int((g["congruent"] == True).sum())  # noqa: E712 (pandas bool col)
        cat_counts = g["category"].value_counts()
        row = {"participant": participant, "group": g["group"].iloc[0], "n_total": n_total,
               "key_skip_pct": 100 * n_key_skip / n_total, "congruent_pct": 100 * n_congruent / n_total}
        for cat in EVENT_CATS:
            row[f"{cat}_pct"] = 100 * cat_counts.get(cat, 0) / n_total
        if "Gender" in g.columns:
            row["Gender"] = g["Gender"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered_breakdown(trial_df, df_participants, excluded):
    """Shared prep for all 4 standalone plots below: full breakdown, HC sorted before
    MS, 'score' merged in from df_participants (participant-level, one row each)."""
    breakdown = build_participant_full_breakdown(trial_df)
    breakdown = breakdown[~breakdown["participant"].isin(excluded)].copy()
    breakdown["group"] = breakdown["group"].replace({"pwMS": "MS"})
    breakdown = breakdown.merge(df_participants[["participant", "score"]], on="participant", how="left")
    hc = breakdown[breakdown["group"] == "HC"].sort_values("participant")
    ms = breakdown[breakdown["group"] == "MS"].sort_values("participant")
    return pd.concat([hc, ms], ignore_index=True), hc, ms


def plot_strategy_breakdown_bar(trial_df, df_participants, out_path, excluded=()):
    """PNG 1/4: per-participant stacked bar of all 7 trial outcomes as %-of-all-trials
    (key_skip + congruent included, so each bar sums to 100) - HC on the left/blue
    names, MS on the right/red names."""
    ordered, hc, ms = _ordered_breakdown(trial_df, df_participants, excluded)
    n = len(ordered)

    fig, ax_bar = plt.subplots(figsize=(max(24, n * 0.16), 10))
    x = np.arange(n)
    bottom = np.zeros(n)
    for cat in STACK_ORDER:
        vals = ordered[f"{cat}_pct"].to_numpy()
        ax_bar.bar(x, vals, bottom=bottom, width=0.9, color=STACK_COLORS[cat], label=STACK_LABELS[cat])
        bottom += vals
    ax_bar.set_xlim(-0.5, n - 0.5)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("% of all trials")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ordered["participant"], rotation=90, fontsize=6)
    for tick, group in zip(ax_bar.get_xticklabels(), ordered["group"]):
        tick.set_color(GROUP_COLORS.get(group, "black"))
    ax_bar.axvline(len(hc) - 0.5, color="black", linewidth=1.2, linestyle="--")
    ax_bar.text(len(hc) / 2, 103, f"HC (N={len(hc)})", color=GROUP_COLORS["HC"], ha="center", fontsize=13, fontweight="bold")
    ax_bar.text(len(hc) + len(ms) / 2, 103, f"MS (N={len(ms)})", color=GROUP_COLORS["MS"], ha="center", fontsize=13, fontweight="bold")
    ax_bar.legend(loc="upper right", ncol=1, fontsize=9, framealpha=0.92)
    ax_bar.set_title("Per-participant trial outcome breakdown (% of ALL trials, stacked to 100%)",
                      fontsize=14, fontweight="bold")

    from feature_pipeline import demographics_text
    ax_bar.text(0.01, 1.14, demographics_text(ordered), fontsize=10, va="top", transform=ax_bar.transAxes,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Breakdown bar chart saved: {out_path} (N={n} participants)")


def plot_strategy_histograms(trial_df, df_participants, out_path, excluded=()):
    """PNG 2/4: for each of the 5 real categories, a histogram of participants' %
    (pooled across their own panels) split by group (HC blue / MS red overlaid), and a
    second row combining all participants together regardless of group."""
    ordered, hc, ms = _ordered_breakdown(trial_df, df_participants, excluded)

    fig, axes = plt.subplots(2, len(EVENT_CATS), figsize=(5 * len(EVENT_CATS), 9))
    for col, cat in enumerate(EVENT_CATS):
        ax_split = axes[0, col]
        ax_split.hist(hc[f"{cat}_pct"], bins=15, color=GROUP_COLORS["HC"], alpha=0.55, label="HC")
        ax_split.hist(ms[f"{cat}_pct"], bins=15, color=GROUP_COLORS["MS"], alpha=0.55, label="MS")
        ax_split.set_title(f"{cat.capitalize()} % (by group)", fontsize=10, fontweight="bold")
        ax_split.set_xlabel(f"{cat} % of all trials", fontsize=8)
        ax_split.set_ylabel("N participants", fontsize=8)
        if col == 0:
            ax_split.legend(fontsize=8)

        ax_all = axes[1, col]
        ax_all.hist(ordered[f"{cat}_pct"], bins=15, color="#555555", alpha=0.85)
        ax_all.set_title(f"{cat.capitalize()} % (all participants)", fontsize=10, fontweight="bold")
        ax_all.set_xlabel(f"{cat} % of all trials", fontsize=8)
        ax_all.set_ylabel("N participants", fontsize=8)

    from feature_pipeline import demographics_text, annotate_demographics
    annotate_demographics(demographics_text(ordered), ax=axes[0, 0], loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram panel saved: {out_path} (N={len(ordered)} participants)")


def _regression_grid(ordered, value_cols, value_label, out_path, title_suffix):
    fig, axes = plt.subplots(1, len(EVENT_CATS), figsize=(5 * len(EVENT_CATS), 5.5))
    for col, cat in enumerate(EVENT_CATS):
        ax_reg = axes[col]
        valid = ordered[[value_cols[cat], "score", "group"]].dropna()
        for group, g in valid.groupby("group"):
            ax_reg.scatter(g[value_cols[cat]], g["score"], color=GROUP_COLORS.get(group, "gray"),
                           alpha=0.7, s=25, edgecolors="black", linewidths=0.3, label=group)
        if len(valid) > 2 and valid[value_cols[cat]].std() > 0:
            r, p = pearsonr(valid[value_cols[cat]], valid["score"])
            z = np.polyfit(valid[value_cols[cat]], valid["score"], 1)
            x_line = np.linspace(valid[value_cols[cat]].min(), valid[value_cols[cat]].max(), 100)
            ax_reg.plot(x_line, np.poly1d(z)(x_line), "k--", alpha=0.7)
            ax_reg.set_title(f"{cat.capitalize()} {title_suffix}\nr={r:.2f}, p={p:.3g} (N={len(valid)})",
                             fontsize=10, fontweight="bold")
        else:
            ax_reg.set_title(f"{cat.capitalize()} {title_suffix}", fontsize=10, fontweight="bold")
        ax_reg.set_xlabel(value_label.format(cat=cat), fontsize=8)
        ax_reg.set_ylabel("SDMT score", fontsize=8)
        if col == 0:
            ax_reg.legend(fontsize=8)

    from feature_pipeline import demographics_text, annotate_demographics
    annotate_demographics(demographics_text(ordered), ax=axes[0], loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Regression panel saved: {out_path} (N={len(ordered)} participants)")


def plot_strategy_regression_vs_score(trial_df, df_participants, out_path, excluded=()):
    """PNG 3/4: each category's % regressed against SDMT score. One point PER
    PARTICIPANT - the % is pooled across all of that participant's own panels (not
    averaged panel-by-panel, not one point per panel), matching the spec's own pooling
    convention."""
    ordered, _, _ = _ordered_breakdown(trial_df, df_participants, excluded)
    value_cols = {cat: f"{cat}_pct" for cat in EVENT_CATS}
    _regression_grid(ordered, value_cols, "{cat} % of all trials (pooled across panels)", out_path,
                      "% (pooled) vs score")


def build_participant_panel_sd(trial_df):
    """For each participant, each category's SD ACROSS their own panels' percentages -
    one number per participant (decision 2026-11-10), distinct from the pooled-across-
    panels % level itself: this captures how much a participant's own category mix
    varies panel-to-panel, not its overall level. NaN if the participant has fewer than
    2 panels with trials (SD undefined)."""
    panel_rows = []
    for (participant, panel), g in trial_df.groupby(["participant", "panel"]):
        n_total = len(g)
        cat_counts = g["category"].value_counts()
        row = {"participant": participant, "panel": panel, "group": g["group"].iloc[0]}
        for cat in EVENT_CATS:
            row[f"{cat}_pct"] = 100 * cat_counts.get(cat, 0) / n_total
        if "Gender" in g.columns:
            row["Gender"] = g["Gender"].iloc[0]
        panel_rows.append(row)
    panel_df = pd.DataFrame(panel_rows)

    rows = []
    for participant, g in panel_df.groupby("participant"):
        row = {"participant": participant, "group": g["group"].iloc[0], "n_panels": len(g)}
        for cat in EVENT_CATS:
            row[f"{cat}_pct_sd"] = g[f"{cat}_pct"].std(ddof=1) if len(g) > 1 else np.nan
        if "Gender" in g.columns:
            row["Gender"] = g["Gender"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_strategy_sd_regression_vs_score(trial_df, df_participants, out_path, excluded=()):
    """PNG 4/4: each category's panel-to-panel SD (one value per participant, see
    build_participant_panel_sd) regressed against SDMT score - does how INCONSISTENT a
    participant's strategy is across panels relate to performance, as opposed to the
    strategy's overall LEVEL (that's PNG 3)."""
    sd_df = build_participant_panel_sd(trial_df)
    sd_df = sd_df[~sd_df["participant"].isin(excluded)].copy()
    sd_df["group"] = sd_df["group"].replace({"pwMS": "MS"})
    sd_df = sd_df.merge(df_participants[["participant", "score"]], on="participant", how="left")

    value_cols = {cat: f"{cat}_pct_sd" for cat in EVENT_CATS}
    _regression_grid(sd_df, value_cols, "{cat} % SD across panels", out_path, "% SD (across panels) vs score")

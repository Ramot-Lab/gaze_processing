"""
Per-symbol search-behavior analysis (decision 2026-09-27): how search length (in
symbols/fixations/physical distance) and search entry-point behave as a function of how
many times a given target symbol has already appeared on this panel, plus histograms and
group (left/middle/right symbol) trajectories built from each search's first fixated
symbol.

Reuses feature_pipeline.py's participant discovery/trial-manager construction so the
population and exclusions match the rest of the feature-analysis pipeline exactly - this
is not a separate analysis population, just a different set of plots over the same data.

Usage: python symbol_search_analysis.py [threshold_based|model_based]
"""
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pipeline_config
from exclusion_policy import DEFAULT_TOBII_SUCKS_XLSX
import feature_pipeline as fp
from feature_pipeline import (discover_participants, load_scores, save_scores_table, build_trial_managers,
                               demographics_text, annotate_demographics, resolve_run_dir)

PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]
SYMBOL_VALUES = list(range(1, 10))
SYMBOL_PALETTE = dict(zip(SYMBOL_VALUES, sns.color_palette("tab10", 9)))
ENTRY_GROUPS = [("left", (1, 3)), ("middle", (4, 6)), ("right", (7, 9))]
COVERAGE_CUTOFF = 0.8
N_RANDOM_SAMPLES = 100
RANDOM_SEED = 27

METRICS = {
    "num_symbols": "Number of Symbols in Search",
    "num_fixations": "Number of Fixations in Search",
    "search_distance": "Physical Distance Covered in Search (px)",
}


# ============================================================
# 1. BUILD THE MASTER PER-TRIAL RECORDS TABLE
# ============================================================
def build_records(participants):
    """One row per (participant, panel, trial) whose triggering symbol is a genuine
    digit 1-9 - the unit every plot in this module is built from. occurrence_index (1-based,
    per participant/panel/symbol_value) is the "how many times has this symbol already
    appeared on this panel" axis items 1-3 plot against. first_symbol is the value of the
    first symbol fixated in the trial's first search (None if the trial had no search at
    all - "no search = 0 symbols" for num_symbols/num_fixations/distance, but there's no
    well-defined first symbol for a search that never happened, so those trials are simply
    absent from first_symbol-based analyses, not zero-filled)."""
    records = []
    for p in participants:
        for panel, tm in getattr(p, "trial_managers", {}).items():
            feats = tm.features
            for i, trial in enumerate(tm.trials):
                sym = trial.triggering_symbol
                if sym is None or sym.value not in SYMBOL_VALUES:
                    continue

                first_symbol = None
                if trial.searches:
                    for s in trial.searches:
                        if s.cleaned_sequence:
                            first_val = next(iter(s.cleaned_sequence.values()))
                            first_symbol = first_val.value if first_val is not None else None
                            break

                records.append({
                    "participant": p.name, "group": p.group, "panel": panel,
                    "trial_order": i, "symbol_value": sym.value,
                    "num_symbols": feats.symbol_counts_per_trial.get(trial.idx, 0),
                    "num_fixations": feats.fixation_counts_per_trial.get(trial.idx, 0),
                    "search_distance": feats.search_distance_per_trial.get(trial.idx, 0.0),
                    "first_symbol": first_symbol,
                    "score": p.scores.get(panel),
                })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.sort_values(["participant", "panel", "trial_order"]).reset_index(drop=True)
    df["occurrence_index"] = df.groupby(["participant", "panel", "symbol_value"]).cumcount() + 1

    # demographics_text (feature_pipeline.py) reads Gender straight off whatever
    # dataframe it's given - merge it in once here so every per-panel/per-sample
    # subset used for a plot's demographics annotation already carries it.
    try:
        demo = pd.read_excel(DEFAULT_TOBII_SUCKS_XLSX)[["Patient_ID", "Gender"]]
        demo = demo.rename(columns={"Patient_ID": "participant"}).drop_duplicates("participant")
        df = df.merge(demo, on="participant", how="left")
    except Exception as e:
        print(f"Warning: could not merge Gender demographics: {e}")

    return df


def symbol_legend_handles():
    """Proxy artists for a shared symbol->color legend (decision 2026-09-27) - used
    wherever color alone encodes which symbol a point/bar belongs to (the entry-group
    scatter has no other indicator of symbol identity at all)."""
    return [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=SYMBOL_PALETTE[s],
                        markersize=10, label=f"Symbol {s}") for s in SYMBOL_VALUES]


def entry_group(symbol_value):
    if symbol_value is None or (isinstance(symbol_value, float) and np.isnan(symbol_value)):
        return None
    for name, (lo, hi) in ENTRY_GROUPS:
        if lo <= symbol_value <= hi:
            return name
    return None


# ============================================================
# 2. COVERAGE CUTOFF (mirrors behavioral_analyzer.plot_metric_global_average's
#    80%-participation logic, applied per symbol_value/occurrence_index instead of
#    per raw trial index)
# ============================================================
def _coverage_cutoff_per_symbol(df_panel, symbol_value, total_pop, cutoff=COVERAGE_CUTOFF):
    """First occurrence_index (inclusive upper bound, i.e. return value is the last index
    still KEPT) at which fewer than cutoff*total_pop distinct participants still have a
    trial for this symbol - matches the "80% of the population we started with" rule."""
    sub = df_panel[df_panel["symbol_value"] == symbol_value]
    counts = sub.groupby("occurrence_index")["participant"].nunique().sort_index()
    threshold = cutoff * total_pop
    below = counts[counts < threshold]
    if below.empty:
        return counts.index.max() if len(counts) else 0
    return below.index.min() - 1


# ============================================================
# 3. ITEMS 1 & 3a: MEAN +/- SD PER PANEL, PER SYMBOL, VS OCCURRENCE INDEX
# ============================================================
def plot_mean_per_symbol(df, out_dir, metric, ylabel):
    os.makedirs(out_dir, exist_ok=True)
    for panel in PANELS:
        df_panel = df[df["panel"] == panel]
        if df_panel.empty:
            continue
        total_pop = df_panel["participant"].nunique()

        plt.figure(figsize=(12, 8))
        for sym in SYMBOL_VALUES:
            cutoff_idx = _coverage_cutoff_per_symbol(df_panel, sym, total_pop)
            if cutoff_idx < 1:
                continue
            sub = df_panel[(df_panel["symbol_value"] == sym) & (df_panel["occurrence_index"] <= cutoff_idx)]
            stats = sub.groupby("occurrence_index")[metric].agg(["mean", "std"]).reset_index()
            color = SYMBOL_PALETTE[sym]
            plt.plot(stats["occurrence_index"], stats["mean"], color=color, label=f"Symbol {sym}", linewidth=2)
            plt.fill_between(stats["occurrence_index"], stats["mean"] - stats["std"],
                              stats["mean"] + stats["std"], color=color, alpha=0.15)

        plt.title(f"Panel {panel}: {ylabel} vs Symbol Occurrence (Mean +/- SD)\n"
                  f"(each symbol's line stops once <{COVERAGE_CUTOFF:.0%} of the population remains)")
        plt.xlabel("Occurrence Number of Symbol on Panel")
        plt.ylabel(ylabel)
        plt.legend(title="Symbol", ncol=3, fontsize=9)
        plt.grid(True, alpha=0.3)
        annotate_demographics(demographics_text(df_panel.drop_duplicates("participant")))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"panel_{panel}_{metric}_mean.png"), dpi=150, bbox_inches="tight")
        plt.close()


def plot_mean_per_symbol_bar(df, out_dir, metric, ylabel):
    """Simpler companion to plot_mean_per_symbol: collapses across occurrence_index
    entirely - one bar per target symbol, mean +/- SD of `metric` over every trial with
    that symbol as its target, across the whole (included) population. Per panel."""
    os.makedirs(out_dir, exist_ok=True)
    for panel in PANELS:
        df_panel = df[df["panel"] == panel]
        if df_panel.empty:
            continue
        stats = df_panel.groupby("symbol_value")[metric].agg(["mean", "std"]).reindex(SYMBOL_VALUES)
        plt.figure(figsize=(9, 6))
        plt.bar(SYMBOL_VALUES, stats["mean"], yerr=stats["std"],
                color=[SYMBOL_PALETTE[s] for s in SYMBOL_VALUES], edgecolor="black", capsize=4)
        plt.title(f"Panel {panel}: Mean {ylabel} per Target Symbol")
        plt.xlabel("Target Symbol")
        plt.ylabel(ylabel)
        plt.xticks(SYMBOL_VALUES)
        plt.grid(True, axis="y", alpha=0.3)
        annotate_demographics(demographics_text(df_panel.drop_duplicates("participant")))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"panel_{panel}_{metric}_mean_per_symbol_bar.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()


def plot_mean_per_symbol_bar_individual(df, sampled_pairs, out_dir, metric, ylabel):
    """A few individual (not 100) example versions of plot_mean_per_symbol_bar - one
    participant/panel's own mean per target symbol, not averaged over the population."""
    out_dir = os.path.join(out_dir, metric)
    os.makedirs(out_dir, exist_ok=True)
    for participant, panel in sampled_pairs:
        sub = df[(df["participant"] == participant) & (df["panel"] == panel)]
        if sub.empty:
            continue
        stats = sub.groupby("symbol_value")[metric].mean().reindex(SYMBOL_VALUES)
        score = sub["score"].iloc[0] if "score" in sub.columns else None
        score_str = f"{score:.0f}" if pd.notna(score) else "N/A"
        plt.figure(figsize=(9, 6))
        plt.bar(SYMBOL_VALUES, stats.values, color=[SYMBOL_PALETTE[s] for s in SYMBOL_VALUES], edgecolor="black")
        plt.title(f"{participant} - Panel {panel} (Score: {score_str})\nMean {ylabel} per Target Symbol")
        plt.xlabel("Target Symbol")
        plt.ylabel(ylabel)
        plt.xticks(SYMBOL_VALUES)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{participant}_{panel}_{metric}_mean_per_symbol_bar.png"),
                    dpi=120, bbox_inches="tight")
        plt.close()


# ============================================================
# 4. ITEMS 2 & 3b: 100 RANDOM (PARTICIPANT, PANEL) SAMPLES - ACTUAL VALUES
# ============================================================
def sample_participant_panels(df, n=N_RANDOM_SAMPLES, seed=RANDOM_SEED):
    pairs = df[["participant", "panel"]].drop_duplicates().values.tolist()
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return [tuple(p) for p in pairs[:n]]


def plot_individual_per_symbol(df, sampled_pairs, out_dir, metric, ylabel):
    out_dir = os.path.join(out_dir, metric)
    os.makedirs(out_dir, exist_ok=True)
    for participant, panel in sampled_pairs:
        sub_pp = df[(df["participant"] == participant) & (df["panel"] == panel)]
        if sub_pp.empty:
            continue
        plt.figure(figsize=(10, 6))
        for sym in SYMBOL_VALUES:
            sub = sub_pp[sub_pp["symbol_value"] == sym].sort_values("occurrence_index")
            if sub.empty:
                continue
            plt.plot(sub["occurrence_index"], sub[metric], color=SYMBOL_PALETTE[sym],
                      marker="o", markersize=4, linewidth=1.5, label=f"Symbol {sym}")

        score = sub_pp["score"].iloc[0]
        score_str = f"{score:.0f}" if pd.notna(score) else "N/A"
        plt.title(f"{participant} - Panel {panel} (Score: {score_str}): {ylabel}\n(actual values, not averaged)")
        plt.xlabel("Occurrence Number of Symbol on Panel")
        plt.ylabel(ylabel)
        plt.legend(title="Symbol", ncol=3, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{participant}_{panel}_{metric}.png"), dpi=120, bbox_inches="tight")
        plt.close()


# ============================================================
# 5. ITEM 4: FIRST-SYMBOL-OF-SEARCH HISTOGRAMS
# ============================================================
def plot_first_symbol_individual(df, sampled_pairs, out_dir):
    """Item 4a: one histogram per sampled (participant, panel)."""
    out_dir = os.path.join(out_dir, "individual")
    os.makedirs(out_dir, exist_ok=True)
    for participant, panel in sampled_pairs:
        sub_pp = df[(df["participant"] == participant) & (df["panel"] == panel)]
        sub = sub_pp["first_symbol"].dropna()
        if sub.empty:
            continue
        score = sub_pp["score"].iloc[0]
        score_str = f"{score:.0f}" if pd.notna(score) else "N/A"
        plt.figure(figsize=(6, 5))
        plt.hist(sub, bins=np.arange(0.5, 10.5, 1), color="teal", edgecolor="black", alpha=0.8)
        plt.title(f"{participant} - Panel {panel} (Score: {score_str})\nFirst Symbol Fixated in Search")
        plt.xlabel("Symbol")
        plt.ylabel("Count")
        plt.xticks(SYMBOL_VALUES)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{participant}_{panel}_first_symbol_hist.png"), dpi=120, bbox_inches="tight")
        plt.close()


def plot_first_symbol_per_panel(df, out_dir):
    """Item 4b: one combined image, 6 subplots (one per panel), each over ALL
    participants who have data for that panel - and its own N in the subplot title,
    since panel-level participation can differ."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for ax, panel in zip(axes, PANELS):
        sub = df[df["panel"] == panel]
        vals = sub["first_symbol"].dropna()
        n_participants = sub.loc[sub["first_symbol"].notna(), "participant"].nunique()
        ax.hist(vals, bins=np.arange(0.5, 10.5, 1), color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_title(f"Panel {panel} (N={n_participants})")
        ax.set_xlabel("Symbol")
        ax.set_ylabel("Count")
        ax.set_xticks(SYMBOL_VALUES)
    plt.suptitle("First Symbol Fixated in Search, per Panel (all participants)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "first_symbol_hist_all_panels.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 6. ITEM 5 (+ combined with item 4a): LEFT/MIDDLE/RIGHT ENTRY GROUP OVER TRIALS
# ============================================================
def plot_entry_group_and_hist_combined(df, sampled_pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    group_order = [name for name, _ in ENTRY_GROUPS]
    group_y = {name: i for i, name in enumerate(group_order)}

    for participant, panel in sampled_pairs:
        sub = df[(df["participant"] == participant) & (df["panel"] == panel)].sort_values("trial_order")
        sub = sub[sub["first_symbol"].notna()]
        if sub.empty:
            continue

        fig, (ax_hist, ax_group) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: first-symbol histogram (item 4a) - bars colored per symbol, same
        # palette as the entry-group scatter, so the shared legend below explains both.
        counts = sub["first_symbol"].value_counts().reindex(SYMBOL_VALUES, fill_value=0)
        ax_hist.bar(SYMBOL_VALUES, counts.values, color=[SYMBOL_PALETTE[s] for s in SYMBOL_VALUES],
                    edgecolor="black", alpha=0.9)
        ax_hist.set_title("First Symbol Fixated in Search")
        ax_hist.set_xlabel("Symbol")
        ax_hist.set_ylabel("Count")
        ax_hist.set_xticks(SYMBOL_VALUES)

        # Right: entry group (left/middle/right) over trials (item 5)
        groups = sub["first_symbol"].apply(entry_group)
        y_vals = groups.map(group_y)
        ax_group.scatter(sub["trial_order"], y_vals, c=[SYMBOL_PALETTE[s] for s in sub["first_symbol"]], s=40)
        ax_group.plot(sub["trial_order"], y_vals, color="gray", alpha=0.4, linewidth=1)
        ax_group.set_yticks(list(group_y.values()))
        ax_group.set_yticklabels([f"{name}\n({lo}-{hi})" for name, (lo, hi) in ENTRY_GROUPS])
        ax_group.set_xlabel("Trial Number")
        ax_group.set_title("Search Entry Group Over Trials")
        ax_group.grid(True, alpha=0.3)

        score = sub["score"].iloc[0] if "score" in sub.columns else None
        score_str = f"{score:.0f}" if pd.notna(score) else "N/A"
        legend = fig.legend(handles=symbol_legend_handles(), title="Symbol", loc="lower center",
                            ncol=9, fontsize=8, bbox_to_anchor=(0.5, -0.05))
        suptitle = plt.suptitle(f"{participant} - Panel {panel} (Score: {score_str})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{participant}_{panel}_entry_group_and_hist.png"),
                    dpi=120, bbox_inches="tight", bbox_extra_artists=(legend, suptitle))
        plt.close()


# ============================================================
# MAIN DRIVER
# ============================================================
def main(annotation_method="threshold_based", run_dir=None, date_str=None, scores_csv_path=None,
         participant_whitelist=None):
    """date_str: which Stage 1 date-folder to read annotated CSVs from (None
    auto-resolves to the most recent existing one). run_dir: override where this run's
    own output goes (None uses the normal feature_analysis_base_dir default).
    participant_whitelist: restrict to just these participant names (None runs the
    full population)."""
    if run_dir is None:
        base_dir = pipeline_config.feature_analysis_base_dir(annotation_method)
        run_dir = resolve_run_dir("compute", base_dir=base_dir)
    if scores_csv_path is None:
        scores_csv_path = fp.SCORES_CSV
    out_dir = os.path.join(run_dir, "symbol_search_analysis")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    participants_dict = discover_participants()
    save_scores_table(participants_dict, out_path=scores_csv_path)
    if participant_whitelist is not None:
        participants_dict = {g: [n for n in names if n in participant_whitelist]
                              for g, names in participants_dict.items()}
    participants = load_scores(participants_dict, scores_csv=scores_csv_path)
    participants, exclusions = build_trial_managers(participants, annotation_method=annotation_method, date_str=date_str)
    print(f"Built trial managers for {len(participants)} participants ({len(exclusions)} panel exclusions).")

    df = build_records(participants)
    if df.empty:
        print("No records built - aborting.")
        return
    csv_path = os.path.join(out_dir, "search_trial_records.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} trial records -> {csv_path}")

    sampled_pairs = sample_participant_panels(df)
    pd.DataFrame(sampled_pairs, columns=["participant", "panel"]).to_csv(
        os.path.join(out_dir, "random_100_sampled_participant_panels.csv"), index=False)

    few_sample_pairs = sampled_pairs[:5]
    for metric, ylabel in METRICS.items():
        plot_mean_per_symbol(df, os.path.join(out_dir, "mean_per_symbol"), metric, ylabel)
        plot_individual_per_symbol(df, sampled_pairs, os.path.join(out_dir, "random_100_individual"), metric, ylabel)
        plot_mean_per_symbol_bar(df, os.path.join(out_dir, "mean_per_symbol_bar"), metric, ylabel)
        plot_mean_per_symbol_bar_individual(df, few_sample_pairs, os.path.join(out_dir, "mean_per_symbol_bar_individual"),
                                             metric, ylabel)

    plot_first_symbol_individual(df, sampled_pairs, os.path.join(out_dir, "first_symbol_histograms"))
    plot_first_symbol_per_panel(df, os.path.join(out_dir, "first_symbol_histograms"))
    plot_entry_group_and_hist_combined(df, sampled_pairs, os.path.join(out_dir, "entry_group_and_hist"))

    print("symbol_search_analysis complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] not in pipeline_config.ANNOTATION_METHODS:
            print(f"usage: python symbol_search_analysis.py [{'|'.join(pipeline_config.ANNOTATION_METHODS)}]")
            sys.exit(1)
        main(sys.argv[1])
    else:
        main()

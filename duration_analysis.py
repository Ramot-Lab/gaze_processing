"""
Fixation/saccade duration histograms + MS-vs-HC violin comparison (decision 2026-09-27),
computed directly from Stage 1's saved annotated CSVs (t, evt, status columns) - no
Fixation/Search/TrialManager objects needed, just consecutive same-event run lengths.

Usage: python duration_analysis.py [threshold_based|model_based]
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind

import pipeline_config
from constants import FIXATION_IDX, SACCADE_IDX, PSO_IDX
from run_preprocessing import iter_all_subject_dirs
from feature_pipeline import demographics_text, annotate_demographics, resolve_run_dir

PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]

# Standard reference lines requested for the two distributions (not a statistical
# threshold from this data - just markers for where "typical" fixation/saccade
# durations fall in the eye-movement literature, e.g. Rayner 1998).
FIXATION_REF_LINES_MS = [200, 300]
SACCADE_REF_LINES_MS = [20, 200]

# Display caps for histogram/violin plots only (never drop data from the underlying CSV
# or from run_group_tests' stats) - a handful of multi-second runs, usually a genuine
# gap in the recording (e.g. ER635/l3: a ~35s jump between two consecutive samples,
# both still labeled valid/saccade) rather than a labeling bug, otherwise dominate the
# bin width / KDE bandwidth and flatten the real distribution into a sliver.
FIXATION_DISPLAY_CAP_MS = 1000
SACCADE_DISPLAY_CAP_MS = 300


def _extract_event_runs(annotated_df):
    """Vectorized run-length extraction: groups consecutive samples sharing the same
    evt value (breaking the run at any invalid/status==0 sample, mirroring
    FixationHandler's own validity check). Returns a DataFrame, one row per contiguous
    run, with columns evt/t_min/t_max/duration_ms - duration_ms = last_t - first_t of
    that run (same convention as FixationHandler.Fixation.duration()), converted from
    the raw microsecond timestamps to milliseconds. Shared by every per-event-type
    duration extractor below (fixation/saccade/PSO) so they all use the identical
    run-detection logic."""
    evt = annotated_df["evt"].to_numpy()
    t = annotated_df["t"].to_numpy()
    valid = annotated_df["status"].to_numpy().astype(bool)
    if len(evt) == 0:
        return pd.DataFrame(columns=["evt", "t_min", "t_max", "duration_ms"])

    evt_masked = np.where(valid, evt, -1)
    change = np.empty(len(evt_masked), dtype=bool)
    change[0] = True
    change[1:] = evt_masked[1:] != evt_masked[:-1]
    run_id = np.cumsum(change)

    runs = pd.DataFrame({"run_id": run_id, "evt": evt_masked, "t": t}).groupby("run_id").agg(
        evt=("evt", "first"), t_min=("t", "min"), t_max=("t", "max"))
    runs["duration_ms"] = (runs["t_max"] - runs["t_min"]) / 1000.0
    return runs


def extract_event_durations(annotated_df):
    """Returns (fixation_durations_ms, saccade_durations_ms) - one value per contiguous
    run of each type."""
    runs = _extract_event_runs(annotated_df)
    fixation_durs = runs.loc[runs["evt"] == FIXATION_IDX, "duration_ms"].to_numpy()
    saccade_durs = runs.loc[runs["evt"] == SACCADE_IDX, "duration_ms"].to_numpy()
    return fixation_durs, saccade_durs


def extract_pso_durations(annotated_df):
    """One value per contiguous PSO (post-saccadic oscillation, evt==PSO_IDX) run -
    only meaningful for model_based annotation (threshold_based never produces this
    class). Sibling of extract_event_durations, split out since PSO is used as a
    per-panel TOTAL (compute_pso_duration_total), not an individual-event distribution."""
    runs = _extract_event_runs(annotated_df)
    return runs.loc[runs["evt"] == PSO_IDX, "duration_ms"].to_numpy()


def compute_pso_duration_total(annotated_df):
    """The feature itself (decision 2026-09-27): total time spent in PSO across every
    PSO block in one participant's panel - a single number per participant/panel, not
    a per-trial breakdown (mirrors this module's panel-level framing, not
    trial_features.py's per-trial one). 0.0 (not NaN) if the panel has no PSO blocks at
    all, consistent with the rest of this codebase's "no occurrence = 0" convention."""
    return float(extract_pso_durations(annotated_df).sum())


def build_duration_table(annotation_method, date_str=None, participant_whitelist=None):
    records = []
    for group, subject_dir in iter_all_subject_dirs(pipeline_config.main_data_path()):
        participant = os.path.basename(subject_dir)
        if participant_whitelist is not None and participant not in participant_whitelist:
            continue
        for panel in PANELS:
            try:
                df = pipeline_config.load_annotated_csv(annotation_method, group, participant, panel,
                                                          date_str=date_str, corrected=False)
            except FileNotFoundError:
                continue
            fix_durs, sacc_durs = extract_event_durations(df)
            for d in fix_durs:
                records.append({"participant": participant, "group": group, "panel": panel,
                                 "event_type": "fixation", "duration_ms": d})
            for d in sacc_durs:
                records.append({"participant": participant, "group": group, "panel": panel,
                                 "event_type": "saccade", "duration_ms": d})
    return pd.DataFrame.from_records(records)


def _merge_gender(df):
    from exclusion_policy import DEFAULT_TOBII_SUCKS_XLSX
    try:
        demo = pd.read_excel(DEFAULT_TOBII_SUCKS_XLSX)[["Patient_ID", "Gender"]]
        demo = demo.rename(columns={"Patient_ID": "participant"}).drop_duplicates("participant")
        return df.merge(demo, on="participant", how="left")
    except Exception as e:
        print(f"Warning: could not merge Gender demographics: {e}")
        return df


def plot_histogram(durations, ref_lines, title, xlabel, out_path, demographics=None, display_cap_ms=None):
    """display_cap_ms: fixed bin range/x-limit for display, so a handful of extreme
    outliers (e.g. a single mislabeled multi-second "saccade" from a recording gap)
    don't blow out the bin width and squash the real distribution into one bar. All
    data still goes into the histogram; points above the cap are just off-screen, and
    their count is reported in a corner annotation so nothing is silently hidden."""
    n_outliers = int((durations > display_cap_ms).sum()) if display_cap_ms is not None else 0
    bin_range = (0, display_cap_ms) if display_cap_ms is not None else None

    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=100, range=bin_range, color="steelblue", edgecolor="black", alpha=0.8)
    colors = ["red", "darkorange"]
    for line_ms, color in zip(ref_lines, colors):
        plt.axvline(line_ms, color=color, linestyle="--", linewidth=2, label=f"{line_ms} ms")
    if display_cap_ms is not None:
        plt.xlim(0, display_cap_ms)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    annotate_demographics(demographics)
    if n_outliers:
        plt.text(0.98, 0.98, f"{n_outliers} value(s) > {display_cap_ms:.0f} ms not shown",
                  transform=plt.gca().transAxes, fontsize=9, ha="right", va="top",
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_group_tests(ms_vals, hc_vals):
    """All 3 candidate tests - user is choosing which one gets featured on the plot,
    so all 3 are computed and reported rather than picking one silently."""
    u_stat, u_p = mannwhitneyu(ms_vals, hc_vals, alternative="two-sided")
    t_stat, t_p = ttest_ind(ms_vals, hc_vals, equal_var=False)

    rng = np.random.default_rng(27)
    combined = np.concatenate([ms_vals, hc_vals])
    n_ms = len(ms_vals)
    obs_diff = np.mean(ms_vals) - np.mean(hc_vals)
    n_perm = 2000
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(combined)
        perm_diffs[i] = np.mean(shuffled[:n_ms]) - np.mean(shuffled[n_ms:])
    perm_p = (np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) + 1) / (n_perm + 1)

    return {
        "mannwhitney_u": u_stat, "mannwhitney_p": u_p,
        "welch_t": t_stat, "welch_p": t_p,
        "permutation_mean_diff": obs_diff, "permutation_p": perm_p,
    }


def plot_violin(df, event_type, ref_lines, title, ylabel, out_path, stats_result, demographics=None,
                 display_cap_ms=None):
    sub = df[df["event_type"] == event_type].copy()
    sub["group"] = sub["group"].replace({"pwMS": "MS"})
    sub = sub[sub["group"].isin(["MS", "HC"])]

    # A handful of extreme outliers would otherwise dominate the KDE bandwidth and
    # flatten the violin into a sliver - excluded from the SHAPE (not from stats_result,
    # which is computed upstream on the full data), same principle as plot_histogram's
    # display_cap_ms, with the excluded count reported the same way.
    n_outliers = 0
    if display_cap_ms is not None:
        n_outliers = int((sub["duration_ms"] > display_cap_ms).sum())
        sub = sub[sub["duration_ms"] <= display_cap_ms]

    plt.figure(figsize=(8, 7))
    sns.violinplot(data=sub, x="group", y="duration_ms", hue="group", order=["HC", "MS"],
                    palette={"HC": "skyblue", "MS": "lightcoral"}, legend=False, cut=0)
    if n_outliers:
        plt.text(0.02, 0.98, f"{n_outliers} value(s) > {display_cap_ms:.0f} ms not shown",
                  transform=plt.gca().transAxes, fontsize=9, ha="left", va="top",
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))
    colors = ["red", "darkorange"]
    for line_ms, color in zip(ref_lines, colors):
        plt.axhline(line_ms, color=color, linestyle="--", linewidth=2, label=f"{line_ms} ms")

    # Welch's t-test is the featured/primary result (user decision 2026-09-27) - shown
    # in the title; the other two candidate tests stay in the corner box for reference.
    stats_text = (
        f"Mann-Whitney U: p={stats_result['mannwhitney_p']:.2e}\n"
        f"Permutation (mean diff): p={stats_result['permutation_p']:.4f}"
    )
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
              verticalalignment="top", horizontalalignment="right",
              bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))

    plt.title(f"{title}\nWelch's t-test: t={stats_result['welch_t']:.2f}, p={stats_result['welch_p']:.2e}",
              fontweight="bold")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.legend()
    plt.grid(True, alpha=0.3)
    annotate_demographics(demographics)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(annotation_method="threshold_based", run_dir=None, date_str=None, participant_whitelist=None):
    """date_str: which Stage 1 date-folder to read annotated CSVs from (None
    auto-resolves to the most recent existing one). run_dir: override where this run's
    own output goes (None uses the normal feature_analysis_base_dir default).
    participant_whitelist: restrict to just these participant names (None runs the
    full population)."""
    if run_dir is None:
        base_dir = pipeline_config.feature_analysis_base_dir(annotation_method)
        run_dir = resolve_run_dir("compute", base_dir=base_dir)
    out_dir = os.path.join(run_dir, "duration_analysis")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    df = build_duration_table(annotation_method, date_str=date_str, participant_whitelist=participant_whitelist)
    if df.empty:
        print("No duration records built - aborting.")
        return
    df = _merge_gender(df)
    csv_path = os.path.join(out_dir, "event_durations.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} event-duration records -> {csv_path}")

    participants_df = df.drop_duplicates("participant")
    demographics = demographics_text(participants_df)

    fix_all = df.loc[df["event_type"] == "fixation", "duration_ms"].to_numpy()
    sacc_all = df.loc[df["event_type"] == "saccade", "duration_ms"].to_numpy()

    plot_histogram(fix_all, FIXATION_REF_LINES_MS, "Fixation Duration - All Participants",
                    "Fixation Duration (ms)", os.path.join(out_dir, "fixation_duration_hist.png"), demographics,
                    display_cap_ms=FIXATION_DISPLAY_CAP_MS)
    plot_histogram(sacc_all, SACCADE_REF_LINES_MS, "Saccade Duration - All Participants",
                    "Saccade Duration (ms)", os.path.join(out_dir, "saccade_duration_hist.png"), demographics,
                    display_cap_ms=SACCADE_DISPLAY_CAP_MS)

    stats_rows = []
    for event_type, ref_lines, ylabel, display_cap in [
        ("fixation", FIXATION_REF_LINES_MS, "Fixation Duration (ms)", FIXATION_DISPLAY_CAP_MS),
        ("saccade", SACCADE_REF_LINES_MS, "Saccade Duration (ms)", SACCADE_DISPLAY_CAP_MS),
    ]:
        sub = df[df["event_type"] == event_type].copy()
        sub["group_2"] = sub["group"].replace({"pwMS": "MS"})
        ms_vals = sub.loc[sub["group_2"] == "MS", "duration_ms"].to_numpy()
        hc_vals = sub.loc[sub["group_2"] == "HC", "duration_ms"].to_numpy()
        if len(ms_vals) < 5 or len(hc_vals) < 5:
            print(f"Skipping {event_type} violin/test: not enough data (MS={len(ms_vals)}, HC={len(hc_vals)}).")
            continue
        result = run_group_tests(ms_vals, hc_vals)
        result["event_type"] = event_type
        stats_rows.append(result)
        plot_violin(df, event_type, ref_lines, f"{event_type.capitalize()} Duration - MS vs HC",
                    ylabel, os.path.join(out_dir, f"{event_type}_duration_violin.png"), result, demographics,
                    display_cap_ms=display_cap)
        print(f"{event_type}: {result}")

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(os.path.join(out_dir, "group_comparison_stats.csv"), index=False)

    print("duration_analysis complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] not in pipeline_config.ANNOTATION_METHODS:
            print(f"usage: python duration_analysis.py [{'|'.join(pipeline_config.ANNOTATION_METHODS)}]")
            sys.exit(1)
        main(sys.argv[1])
    else:
        main()

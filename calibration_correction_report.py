"""
Analysis report over the tables calibration_drift_qa.py produces.

Reads calibration_quality.csv, dominant_eye_check.csv and drift_correction_qa.csv (from
DEFAULT_CALIBRATION_QC_DIR) and answers three questions:

  1. How good was calibration to begin with? (summarize_calibration_quality)
  2. Did the dictionary-drift correction actually improve gaze quality? Before vs after.
     (summarize_drift_correction)
  3. Does it matter whether analysis uses Dom_Eye vs whichever eye Tobii actually
     calibrated best, and does that interact with the drift correction? A 2x2 comparison:
     {dominant eye, better-calibrated eye} x {uncorrected, drift-corrected}.
     (build_eye_comparison_table / summarize_eye_comparison)

Item 3 needs data that doesn't exist in any of the three input tables yet - the
"better-calibrated eye" pipeline's own gaze quality - so it re-runs the pipeline with
ParticipantGazeDataManager's new `analysis_eye` override (see participant_gaze_data_manager.py)
for participant/panels where dominant_matches_best is False. Rows where the dominant eye
already IS the better-calibrated eye are copied over rather than recomputed, since the two
pipelines are identical in that case.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from calibration_drift_qa import (
    DEFAULT_MAIN_DATA_PATH, DEFAULT_CALIBRATION_QC_DIR,
    CALIBRATION_MEASUREMENT_GLOSSARY, evaluate_drift_correction,
)
from participant_gaze_data_manager import ParticipantGazeDataManager

# Commonly used accuracy bands for Tobii-style validation reports (deg). Not a strict
# clinical cutoff - just a readable way to bucket the acc column from calibration_quality.csv.
CALIBRATION_ACCURACY_BANDS = [(0.5, "excellent (<=0.5 deg)"), (1.0, "acceptable (<=1.0 deg)"), (float("inf"), "poor (>1.0 deg)")]


def _band_accuracy(acc):
    if pd.isna(acc):
        return "unknown"
    for threshold, label in CALIBRATION_ACCURACY_BANDS:
        if acc <= threshold:
            return label
    return "unknown"


def load_qc_tables(calibration_qc_dir=DEFAULT_CALIBRATION_QC_DIR):
    calibration_df = pd.read_csv(os.path.join(calibration_qc_dir, "calibration_quality.csv"))
    dominant_eye_df = pd.read_csv(os.path.join(calibration_qc_dir, "dominant_eye_check.csv"))
    drift_qa_df = pd.read_csv(os.path.join(calibration_qc_dir, "drift_correction_qa.csv"))
    return calibration_df, dominant_eye_df, drift_qa_df


def load_whole_dictionary_table(calibration_qc_dir=DEFAULT_CALIBRATION_QC_DIR):
    """Separate from load_qc_tables() since whole_dictionary_drift_qa.csv is a newer, optional table."""
    path = os.path.join(calibration_qc_dir, "whole_dictionary_drift_qa.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ---------------------------------------------------------------------------------------
# 1. Initial calibration quality
# ---------------------------------------------------------------------------------------
def summarize_calibration_quality(calibration_df):
    valid = calibration_df[calibration_df["calibration_missing"] == False].copy()
    valid["best_eye_acc"] = valid[["left_acc", "right_acc"]].min(axis=1)
    valid["best_eye_band"] = valid["best_eye_acc"].apply(_band_accuracy)

    return {
        "n_participant_panels": len(calibration_df),
        "n_missing_calibration": int((calibration_df["calibration_missing"] == True).sum()),
        "pct_missing_calibration": 100.0 * (calibration_df["calibration_missing"] == True).mean(),
        "left_acc_mean": valid["left_acc"].mean(),
        "left_acc_median": valid["left_acc"].median(),
        "right_acc_mean": valid["right_acc"].mean(),
        "right_acc_median": valid["right_acc"].median(),
        "best_eye_accuracy_band_counts": valid["best_eye_band"].value_counts().to_dict(),
        "mean_acc_by_group": valid.groupby("group")[["left_acc", "right_acc"]].mean().round(3).to_dict(orient="index"),
    }


# ---------------------------------------------------------------------------------------
# 2. Gaze quality before vs after drift correction
# ---------------------------------------------------------------------------------------
def summarize_drift_correction(drift_qa_df):
    valid = drift_qa_df.dropna(subset=["mean_dist_before_px", "mean_dist_after_px"]).copy()
    valid["dist_improved"] = valid["mean_dist_after_px"] < valid["mean_dist_before_px"]
    valid["search_pct_improved"] = valid["pct_trials_with_dict_search_after"] > valid["pct_trials_with_dict_search_before"]

    summary = {
        "n_participant_panels": len(drift_qa_df),
        "n_usable": len(valid),
        "mean_dist_before_px": valid["mean_dist_before_px"].mean(),
        "mean_dist_after_px": valid["mean_dist_after_px"].mean(),
        "median_dist_before_px": valid["median_dist_before_px"].median(),
        "median_dist_after_px": valid["median_dist_after_px"].median(),
        "pct_rows_distance_improved": 100.0 * valid["dist_improved"].mean() if len(valid) else None,
        "pct_rows_search_detection_improved": 100.0 * valid["search_pct_improved"].mean() if len(valid) else None,
        "mean_pct_trials_with_search_before": valid["pct_trials_with_dict_search_before"].mean(),
        "mean_pct_trials_with_search_after": valid["pct_trials_with_dict_search_after"].mean(),
    }

    try:
        _, p_value = stats.wilcoxon(valid["mean_dist_before_px"], valid["mean_dist_after_px"])
        summary["wilcoxon_p_value_dist_before_vs_after"] = p_value
    except ValueError:
        summary["wilcoxon_p_value_dist_before_vs_after"] = None

    return summary


# ---------------------------------------------------------------------------------------
# 3. Dominant eye vs better-calibrated eye, with/without correction
# ---------------------------------------------------------------------------------------
def build_eye_comparison_table(dominant_eye_df, drift_qa_df, main_data_path=DEFAULT_MAIN_DATA_PATH, task="SDMT"):
    """
    One row per participant + panel: the already-computed dominant-eye pipeline quality
    (dom_*) next to the better-calibrated-eye pipeline quality (best_*). Only mismatched
    rows (dominant_matches_best == False) require rebuilding a ParticipantGazeDataManager
    with analysis_eye=better_eye - matched rows reuse the dominant-eye numbers directly.
    A single alternate-eye manager is reused across every panel of the same participant
    that needs it, instead of rebuilding it per panel.
    """
    dom_lookup = drift_qa_df.set_index(["participant", "panel"])
    rows = []

    for participant, participant_rows in dominant_eye_df.groupby("participant"):
        group = participant_rows["group"].iloc[0]
        subject_dir = os.path.join(main_data_path, group, participant)
        alt_managers = {}  # better_eye -> ParticipantGazeDataManager, built lazily

        for _, r in participant_rows.iterrows():
            panel = r["panel"]
            if (participant, panel) not in dom_lookup.index:
                continue
            dom_row = dom_lookup.loc[(participant, panel)]

            row = {
                "group": group, "participant": participant, "panel": panel,
                "dominant_eye": r["dominant_eye"], "better_eye": r["better_eye"],
                "dominant_matches_best": bool(r["dominant_matches_best"]),
                "dom_mean_dist_before_px": dom_row.get("mean_dist_before_px"),
                "dom_mean_dist_after_px": dom_row.get("mean_dist_after_px"),
                "dom_pct_search_before": dom_row.get("pct_trials_with_dict_search_before"),
                "dom_pct_search_after": dom_row.get("pct_trials_with_dict_search_after"),
            }

            if row["dominant_matches_best"]:
                row["best_mean_dist_before_px"] = row["dom_mean_dist_before_px"]
                row["best_mean_dist_after_px"] = row["dom_mean_dist_after_px"]
                row["best_pct_search_before"] = row["dom_pct_search_before"]
                row["best_pct_search_after"] = row["dom_pct_search_after"]
            else:
                better_eye = r["better_eye"]
                if better_eye not in alt_managers:
                    try:
                        alt_managers[better_eye] = ParticipantGazeDataManager(
                            subject_dir, main_data_path, task, group, analysis_eye=better_eye)
                    except Exception as e:
                        alt_managers[better_eye] = None
                        row["error"] = f"could not build analysis_eye={better_eye!r} manager: {e}"
                alt_manager = alt_managers[better_eye]
                if alt_manager is not None:
                    try:
                        best_result = evaluate_drift_correction(alt_manager, panel)
                        row["best_mean_dist_before_px"] = best_result["mean_dist_before_px"]
                        row["best_mean_dist_after_px"] = best_result["mean_dist_after_px"]
                        row["best_pct_search_before"] = best_result["pct_trials_with_dict_search_before"]
                        row["best_pct_search_after"] = best_result["pct_trials_with_dict_search_after"]
                    except Exception as e:
                        row["error"] = str(e)

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_eye_comparison(eye_comparison_df):
    df = eye_comparison_df.dropna(subset=["dom_mean_dist_before_px", "best_mean_dist_before_px"]).copy()
    mismatched = df[df["dominant_matches_best"] == False]

    def _pair(d, before_col, after_col):
        return {"mean_before": d[before_col].mean(), "mean_after": d[after_col].mean()}

    result = {
        "n_rows": len(df),
        "n_mismatched_rows": len(mismatched),
        "all_rows": {
            "dom_dist_px": _pair(df, "dom_mean_dist_before_px", "dom_mean_dist_after_px"),
            "best_dist_px": _pair(df, "best_mean_dist_before_px", "best_mean_dist_after_px"),
            "dom_pct_search": _pair(df, "dom_pct_search_before", "dom_pct_search_after"),
            "best_pct_search": _pair(df, "best_pct_search_before", "best_pct_search_after"),
        },
        "mismatched_rows_only": None,
    }

    if len(mismatched):
        result["mismatched_rows_only"] = {
            "dom_dist_px": _pair(mismatched, "dom_mean_dist_before_px", "dom_mean_dist_after_px"),
            "best_dist_px": _pair(mismatched, "best_mean_dist_before_px", "best_mean_dist_after_px"),
            "pct_mismatched_where_better_eye_helps_uncorrected":
                100.0 * (mismatched["best_mean_dist_before_px"] < mismatched["dom_mean_dist_before_px"]).mean(),
            "pct_mismatched_where_better_eye_helps_corrected":
                100.0 * (mismatched["best_mean_dist_after_px"] < mismatched["dom_mean_dist_after_px"]).mean(),
        }

    return result


# ---------------------------------------------------------------------------------------
# Accuracy-gap histograms: dominant vs best-calibrated eye, and right vs left eye
# ---------------------------------------------------------------------------------------
def compute_eye_accuracy_diffs(dominant_eye_df):
    """
    dominant_minus_best_acc (deg, >=0): how much worse the dominant eye's calibration is
    than whichever eye actually calibrated best - 0 wherever they're the same eye. Answers
    "what does sticking with Dom_Eye cost you, accuracy-wise".
    right_minus_left_acc (deg, signed): right-eye accuracy minus left-eye accuracy across
    the cohort, regardless of dominance - shows whether one eye is systematically
    better/worse calibrated than the other.
    """
    df = dominant_eye_df.copy()
    dom_is_left = df["dominant_eye"].str.lower() == "l"
    dom_acc = np.where(dom_is_left, df["left_acc"], df["right_acc"])
    best_acc = df[["left_acc", "right_acc"]].min(axis=1)
    df["dominant_minus_best_acc"] = dom_acc - best_acc
    df["right_minus_left_acc"] = df["right_acc"] - df["left_acc"]
    return df


def _plot_diff_histogram(values, title, xlabel, save_path=None, bins=30):
    values = pd.Series(values).dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="skyblue", edgecolor="k")
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel("Count (participant-panels)")
    plt.title(f"{title}\nmean={values.mean():.3f}, median={values.median():.3f}, n={len(values)}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved histogram to {save_path}")
    plt.show()
    plt.close()


def plot_eye_accuracy_histograms(dominant_eye_df, output_dir=DEFAULT_CALIBRATION_QC_DIR):
    """Renders (and saves) both requested histograms; returns the per-row diff table."""
    df = compute_eye_accuracy_diffs(dominant_eye_df)
    os.makedirs(output_dir, exist_ok=True)

    _plot_diff_histogram(
        df["dominant_minus_best_acc"],
        "Dominant eye accuracy minus best-calibrated eye accuracy",
        "Dominant acc - best acc (deg)",
        save_path=os.path.join(output_dir, "hist_dominant_minus_best_acc.png"),
    )
    _plot_diff_histogram(
        df["right_minus_left_acc"],
        "Right eye accuracy minus left eye accuracy",
        "Right acc - left acc (deg)",
        save_path=os.path.join(output_dir, "hist_right_minus_left_acc.png"),
    )
    return df


# ---------------------------------------------------------------------------------------
# Whole-dictionary drift correction: correction-size histogram + gap-zone before/after
# ---------------------------------------------------------------------------------------
def plot_whole_dictionary_drift_histogram(whole_dict_df, output_dir=DEFAULT_CALIBRATION_QC_DIR):
    """Histogram of the whole-dictionary vertical correction (dy) across all participant-panels."""
    os.makedirs(output_dir, exist_ok=True)
    _plot_diff_histogram(
        whole_dict_df["dy"],
        "Whole-dictionary drift correction\n(outlier-cleaned mean fixation y vs dictionary midline)",
        "dy = mean dictionary-fixation y (outlier-cleaned) - midline y (px)",
        save_path=os.path.join(output_dir, "hist_whole_dictionary_dy.png"),
    )


def plot_gap_zone_fixation_histograms(whole_dict_df, output_dir=DEFAULT_CALIBRATION_QC_DIR, bins=30):
    """
    Side-by-side before/after histograms of how many fixations per participant-panel fell
    in the dictionary/grid gap zone (see count_dictionary_grid_gap_fixations in
    calibration_drift_qa.py) - the zone SearchFinder's own threshold would otherwise miss.
    """
    os.makedirs(output_dir, exist_ok=True)
    before = whole_dict_df["n_gap_zone_fixations_before_correction"].dropna()
    after = whole_dict_df["n_gap_zone_fixations_after_correction"].dropna()
    shared_bins = np.histogram_bin_edges(pd.concat([before, after]), bins=bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, values, label in zip(axes, [before, after], ["Before correction", "After correction"]):
        ax.hist(values, bins=shared_bins, color="skyblue", edgecolor="k")
        ax.axvline(values.mean(), color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{label}\nmean={values.mean():.2f}, median={values.median():.1f}, n={len(values)}")
        ax.set_xlabel("# fixations in dictionary/grid gap zone")
    axes[0].set_ylabel("Count (participant-panels)")
    fig.suptitle("Fixations in the SearchFinder-missed dictionary/grid gap zone, before vs after whole-dictionary correction")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "hist_gap_zone_fixations_before_after.png")
    plt.savefig(save_path, dpi=200)
    print(f"Saved histogram to {save_path}")
    plt.show()
    plt.close()


def plot_negative_y_histogram(whole_dict_df, output_dir=DEFAULT_CALIBRATION_QC_DIR):
    """
    Histogram of how many fixations per participant-panel got pushed to y<0 (off the top
    of the screen entirely) after the whole-dictionary correction - overshoot, not just
    recentering. See count_negative_y_fixations in calibration_drift_qa.py.
    """
    os.makedirs(output_dir, exist_ok=True)
    _plot_diff_histogram(
        whole_dict_df["n_negative_y_fixations_after"],
        "Fixations pushed off-screen (y < 0) after whole-dictionary correction",
        "# fixations with corrected y < 0",
        save_path=os.path.join(output_dir, "hist_negative_y_after_correction.png"),
    )


# ---------------------------------------------------------------------------------------
# Compare two whole-dictionary reference lines (dictionary midline vs second-row center)
# against each other and against the uncorrected data.
# ---------------------------------------------------------------------------------------
def _merge_gap_zone_methods(midline_df, second_row_df, per_trial_df=None):
    """
    Merges the "before" (original) column plus one "after" column per method, on
    participant+panel. Returns (merged_df, series) where series is a list of
    (values, label, color) tuples ready to plot - "Original" first, then one per method.
    """
    merged = midline_df[["participant", "panel", "n_gap_zone_fixations_before_correction",
                          "n_gap_zone_fixations_after_correction"]].rename(
        columns={"n_gap_zone_fixations_after_correction": "midline_after"})
    merged = merged.merge(
        second_row_df[["participant", "panel", "n_gap_zone_fixations_after_correction"]].rename(
            columns={"n_gap_zone_fixations_after_correction": "second_row_after"}),
        on=["participant", "panel"],
    )
    if per_trial_df is not None:
        merged = merged.merge(
            per_trial_df[["participant", "panel", "n_gap_zone_fixations_after_correction"]].rename(
                columns={"n_gap_zone_fixations_after_correction": "per_trial_after"}),
            on=["participant", "panel"],
        )
    merged = merged.dropna()

    series = [
        (merged["n_gap_zone_fixations_before_correction"], "Original (uncorrected)", "gray"),
        (merged["midline_after"], "Corrected - midline reference", "skyblue"),
        (merged["second_row_after"], "Corrected - second-row-center reference", "salmon"),
    ]
    if per_trial_df is not None:
        series.append((merged["per_trial_after"], "Corrected - per-trial method", "mediumseagreen"))
    return merged, series


def plot_gap_zone_reference_comparison_histogram(midline_df, second_row_df, per_trial_df=None,
                                                   output_dir=DEFAULT_CALIBRATION_QC_DIR, bins=30):
    """
    One overlaid histogram (different colors per series) of dictionary/grid gap-zone
    fixation counts per participant-panel: the original uncorrected data, after
    midline-referenced correction, after second-row-center-referenced correction, and
    (if per_trial_df is given) after the per-trial triggering-symbol-anchored correction.
    """
    os.makedirs(output_dir, exist_ok=True)
    merged, series = _merge_gap_zone_methods(midline_df, second_row_df, per_trial_df)
    shared_bins = np.histogram_bin_edges(pd.concat([v for v, _, _ in series]), bins=bins)

    plt.figure(figsize=(9, 6))
    for values, label, color in series:
        plt.hist(values, bins=shared_bins, color=color, edgecolor="k", alpha=0.5,
                  label=f"{label} (mean={values.mean():.2f}, n={len(values)})")
    plt.xlabel("# fixations in dictionary/grid gap zone")
    plt.ylabel("Count (participant-panels)")
    plt.title("Gap-zone fixation counts: original vs midline vs second-row vs per-trial")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "hist_gap_zone_reference_comparison.png")
    plt.savefig(save_path, dpi=200)
    print(f"Saved histogram to {save_path}")
    plt.show()
    plt.close()
    return merged


def plot_gap_zone_reference_comparison_histogram_side_by_side(midline_df, second_row_df, per_trial_df=None,
                                                                 output_dir=DEFAULT_CALIBRATION_QC_DIR, bins=30):
    """
    Same comparison as plot_gap_zone_reference_comparison_histogram(), but as side-by-side
    subplots (one per series) sharing bins/y-axis, instead of one overlaid plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    merged, series = _merge_gap_zone_methods(midline_df, second_row_df, per_trial_df)
    shared_bins = np.histogram_bin_edges(pd.concat([v for v, _, _ in series]), bins=bins)

    fig, axes = plt.subplots(1, len(series), figsize=(4 * len(series) + 2, 5), sharex=True, sharey=True)
    for ax, (values, label, color) in zip(axes, series):
        ax.hist(values, bins=shared_bins, color=color, edgecolor="k")
        ax.axvline(values.mean(), color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{label}\nmean={values.mean():.2f}, median={values.median():.1f}, n={len(values)}")
        ax.set_xlabel("# fixations in dictionary/grid gap zone")
    axes[0].set_ylabel("Count (participant-panels)")
    fig.suptitle("Gap-zone fixation counts: original vs midline vs second-row vs per-trial")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "hist_gap_zone_reference_comparison_side_by_side.png")
    plt.savefig(save_path, dpi=200)
    print(f"Saved histogram to {save_path}")
    plt.show()
    plt.close()
    return merged


def plot_negative_y_reference_comparison_histogram(midline_df, second_row_df, per_trial_df=None,
                                                      output_dir=DEFAULT_CALIBRATION_QC_DIR, bins=30):
    """
    One overlaid histogram (different colors per series) of how many fixations per
    participant-panel got pushed to y<0 after correction, comparing the midline reference,
    the second-row-center reference, and (if per_trial_df is given) the per-trial method.
    """
    os.makedirs(output_dir, exist_ok=True)
    merged = midline_df[["participant", "panel", "n_negative_y_fixations_after"]].rename(
        columns={"n_negative_y_fixations_after": "midline_after"})
    merged = merged.merge(
        second_row_df[["participant", "panel", "n_negative_y_fixations_after"]].rename(
            columns={"n_negative_y_fixations_after": "second_row_after"}),
        on=["participant", "panel"],
    )
    if per_trial_df is not None:
        merged = merged.merge(
            per_trial_df[["participant", "panel", "n_negative_y_fixations_after"]].rename(
                columns={"n_negative_y_fixations_after": "per_trial_after"}),
            on=["participant", "panel"],
        )
    merged = merged.dropna()

    series = [
        (merged["midline_after"], "Corrected - midline reference", "skyblue"),
        (merged["second_row_after"], "Corrected - second-row-center reference", "salmon"),
    ]
    if per_trial_df is not None:
        series.append((merged["per_trial_after"], "Corrected - per-trial method", "mediumseagreen"))
    shared_bins = np.histogram_bin_edges(pd.concat([v for v, _, _ in series]), bins=bins)

    plt.figure(figsize=(9, 6))
    for values, label, color in series:
        plt.hist(values, bins=shared_bins, color=color, edgecolor="k", alpha=0.5,
                  label=f"{label} (n panels affected={int((values > 0).sum())}, total={int(values.sum())})")
    plt.xlabel("# fixations with corrected y < 0")
    plt.ylabel("Count (participant-panels)")
    plt.title("Fixations pushed off-screen (y<0) after correction: midline vs second-row vs per-trial")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "hist_negative_y_reference_comparison.png")
    plt.savefig(save_path, dpi=200)
    print(f"Saved histogram to {save_path}")
    plt.show()
    plt.close()
    return merged


def summarize_whole_dictionary_drift(whole_dict_df):
    valid = whole_dict_df.dropna(subset=["dy"]).copy()
    gap = valid.dropna(subset=["n_gap_zone_fixations_before_correction", "n_gap_zone_fixations_after_correction"])
    gap_improved = gap["n_gap_zone_fixations_after_correction"] < gap["n_gap_zone_fixations_before_correction"]

    neg_after = valid["n_negative_y_fixations_after"].fillna(0)
    neg_before = valid["n_negative_y_fixations_before"].fillna(0)

    return {
        "n_participant_panels": len(whole_dict_df),
        "n_usable": len(valid),
        "dy_mean": valid["dy"].mean(),
        "dy_median": valid["dy"].median(),
        "mean_gap_zone_fixations_before": gap["n_gap_zone_fixations_before_correction"].mean(),
        "mean_gap_zone_fixations_after": gap["n_gap_zone_fixations_after_correction"].mean(),
        "median_gap_zone_fixations_before": gap["n_gap_zone_fixations_before_correction"].median(),
        "median_gap_zone_fixations_after": gap["n_gap_zone_fixations_after_correction"].median(),
        "pct_panels_gap_zone_improved": 100.0 * gap_improved.mean() if len(gap) else None,
        "n_panels_with_negative_y_after": int((neg_after > 0).sum()),
        "pct_panels_with_negative_y_after": 100.0 * (neg_after > 0).mean() if len(valid) else None,
        "total_negative_y_fixations_after": int(neg_after.sum()),
        "n_panels_with_negative_y_before": int((neg_before > 0).sum()),
    }


# ---------------------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------------------
def _format_summary_block(title, summary):
    lines = [title, "-" * len(title)]
    if summary is None:
        lines.append("(skipped)")
    else:
        for key, value in summary.items():
            lines.append(f"{key}: {value}")
    lines.append("")
    return "\n".join(lines)


def write_report(output_dir, calibration_summary, drift_summary, eye_comparison_summary, whole_dict_summary=None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "analysis_summary.txt")
    with open(path, "w") as f:
        f.write("Calibration / drift-correction analysis summary\n")
        f.write("=================================================\n\n")
        f.write("Metric glossary (see CALIBRATION_MEASUREMENT_GLOSSARY in calibration_drift_qa.py):\n")
        for key, explanation in CALIBRATION_MEASUREMENT_GLOSSARY.items():
            f.write(f"  {key}: {explanation}\n")
        f.write("\n")
        f.write(_format_summary_block("1. Initial calibration quality", calibration_summary))
        f.write(_format_summary_block("2. Gaze quality before vs after drift correction", drift_summary))
        f.write(_format_summary_block("3. Dominant eye vs better-calibrated eye", eye_comparison_summary))
        f.write(_format_summary_block("4. Whole-dictionary drift correction (gap zone + y<0 overshoot)", whole_dict_summary))
    return path


def build_full_report(main_data_path=DEFAULT_MAIN_DATA_PATH, calibration_qc_dir=DEFAULT_CALIBRATION_QC_DIR, task="SDMT"):
    calibration_df, dominant_eye_df, drift_qa_df = load_qc_tables(calibration_qc_dir)

    calibration_summary = summarize_calibration_quality(calibration_df)
    drift_summary = summarize_drift_correction(drift_qa_df)
    plot_eye_accuracy_histograms(dominant_eye_df, output_dir=calibration_qc_dir)

    whole_dict_df = load_whole_dictionary_table(calibration_qc_dir)
    if whole_dict_df is not None:
        whole_dict_summary = summarize_whole_dictionary_drift(whole_dict_df)
        plot_whole_dictionary_drift_histogram(whole_dict_df, output_dir=calibration_qc_dir)
        plot_gap_zone_fixation_histograms(whole_dict_df, output_dir=calibration_qc_dir)
        plot_negative_y_histogram(whole_dict_df, output_dir=calibration_qc_dir)
    else:
        whole_dict_summary = None
        print("whole_dictionary_drift_qa.csv not found - skipping whole-dictionary analysis "
              "(run calibration_drift_qa.build_whole_dictionary_drift_table() first)")

    # Dominant eye vs better-calibrated eye comparison - commented out for now (expensive
    # re-sweep: rebuilds a ParticipantGazeDataManager per mismatched participant).
    # eye_comparison_df = build_eye_comparison_table(dominant_eye_df, drift_qa_df, main_data_path, task)
    # eye_comparison_df.to_csv(os.path.join(calibration_qc_dir, "eye_comparison_qa.csv"), index=False)
    # eye_comparison_summary = summarize_eye_comparison(eye_comparison_df)
    eye_comparison_df = None
    eye_comparison_summary = None

    report_path = write_report(calibration_qc_dir, calibration_summary, drift_summary,
                                eye_comparison_summary, whole_dict_summary)
    print(f"Report written to {report_path}")

    return {
        "calibration_summary": calibration_summary,
        "drift_summary": drift_summary,
        "whole_dict_summary": whole_dict_summary,
        "eye_comparison_summary": eye_comparison_summary,
        "eye_comparison_df": eye_comparison_df,
        "report_path": report_path,
    }


if __name__ == "__main__":
    build_full_report()

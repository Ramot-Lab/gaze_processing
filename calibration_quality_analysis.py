"""
Calibration-quality analysis: per-calibration-event metadata, eye-selection fallback
provenance, group comparisons, threshold-sensitivity analysis, and an interactive
threshold-exploration dashboard. See /Volumes/Noam_M/calibration_qc/README.md.

The atomic unit for every analysis here is the CALIBRATION EVENT (= one participant-day =
one raw run .mat file), not the panel-row: the 3 panels sharing a mat file carry identical
calibration values, so treating panels as independent observations would pseudo-replicate
every count/statistic ~3x. build_calibration_events_table() collapses to one row per event.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from constants import (
    KEY_CALIBRATION_INFO, KEY_EYE_SELECTION_REASON, KEY_ANALYSIS_EYE, KEY_AUDIO_DATA,
    KEY_RECORDING_DATE,
)
from participant_gaze_data_manager import ParticipantGazeDataManager, CALIBRATION_METRIC_KEYS

DEFAULT_MAIN_DATA_PATH = "/Volumes/Noam_M/Results/Behavior"
DEFAULT_OUTPUT_ROOT = "/Volumes/Noam_M/calibration_qc"


def _safe_savefig(path, dpi=200):
    """
    Overwriting a file on this network mount fails in two different, seemingly
    inconsistent ways depending on the file's current lock state: sometimes a plain
    overwrite errors out and only works after deleting first; sometimes the reverse - the
    delete itself fails ("Resource busy") while a plain overwrite succeeds fine. Try plain
    overwrite first (the common case), and only delete-then-retry if that fails.
    """
    try:
        plt.savefig(path, dpi=dpi)
    except OSError:
        if os.path.exists(path):
            os.remove(path)
        plt.savefig(path, dpi=dpi)


def _iter_subject_dirs(main_data_path, groups=("HC", "pwMS")):
    for group in groups:
        for subject_dir in sorted(glob.glob(os.path.join(main_data_path, group, "*"))):
            if not os.path.isdir(subject_dir):
                continue
            # DONTUSE is a manual "exclude this person" flag unrelated to data quality.
            if "DONTUSE" in os.path.basename(subject_dir).upper():
                continue
            yield group, subject_dir


def _missing_reason_bucket(error_text):
    """Coarse, stable bucket for a load_errors entry's free-text error - used as
    eye_selection_reason for events that never got an eye assigned at all."""
    if not error_text:
        return "excluded_preprocessing_error"
    text = error_text.lower()
    if "nan" in text:
        if "would pass" in text:
            return "excluded_nan_threshold_rescuable"
        if "both eyes" in text:
            return "excluded_nan_threshold_both_eyes"
        return "excluded_nan_threshold"
    if "no gaze samples" in text:
        return "excluded_no_gaze_data"
    if "duplicate" in text:
        return "excluded_duplicate_file"
    if "expecting matrix" in text or "could not read file" in text:
        return "excluded_corrupted_file"
    if "dom_eye" in text:
        return "excluded_missing_dom_label_and_bad_data"
    return "excluded_preprocessing_error"


def build_calibration_events_table(main_data_path=DEFAULT_MAIN_DATA_PATH, task="SDMT", groups=("HC", "pwMS")):
    """
    One row per calibration event. Columns:
      group, participant, recording_date, calibration_missing, missing_reason,
      eye_used, eye_selection_reason, n_panels, n_panels_missing_audio,
      dominant_eye_raw, calibration_no, validation_no,
      {left,right}_{acc,accX,accY,std,rms,data_loss}, used_acc

    eye_selection_reason values:
      dominant                            - Dom_Eye label valid, used as-is
      fallback_missing_dom_label          - Dom_Eye missing/invalid, fell back to the
                                             better-calibrated eye for this event
      explicit_override                   - analysis_eye was explicitly forced (not used
                                             by this sweep, but supported by the pipeline)
      excluded_nan_threshold_rescuable      - selected eye failed the NaN-ratio check but the
                                             OTHER eye would have passed (no automatic
                                             rescue is performed - no accuracy threshold has
                                             been agreed on for that, see README.md)
      excluded_nan_threshold_both_eyes     - both eyes failed the NaN-ratio check - not
                                             recoverable by an eye swap
      excluded_nan_threshold               - NaN failure where the other eye's ratio isn't
                                             available for comparison (rare)
      excluded_no_gaze_data                - eye tracker recorded 0 samples this session
                                             (both eyes share one sample timeline, so this
                                             always affects both eyes equally)
      excluded_duplicate_file              - a duplicate/partial save of another event
      excluded_corrupted_file              - the .mat file itself failed to parse
      excluded_missing_dom_label_and_bad_data - Dom_Eye invalid AND no calibration message
                                             to fall back on
      excluded_participant_load_error      - participant-level failure (e.g. bad path)
    """
    rows = []
    for group, subject_dir in _iter_subject_dirs(main_data_path, groups):
        if task not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)

        try:
            subject_data = ParticipantGazeDataManager(subject_dir, main_data_path, task, group)
        except Exception as e:
            rows.append({
                "group": group, "participant": participant, "recording_date": None,
                "calibration_missing": True, "missing_reason": str(e),
                "eye_used": None, "eye_selection_reason": "excluded_participant_load_error",
                "n_panels": 0, "n_panels_missing_audio": None,
            })
            continue

        # Failed calibration events - one row per event so a bad day never hides whether
        # this participant's OTHER day was fine (see ParticipantGazeDataManager.load_errors).
        for load_error in subject_data.load_errors:
            rows.append({
                "group": group, "participant": subject_data.name,
                "recording_date": load_error.get("recording_date") or load_error.get("file"),
                "calibration_missing": True, "missing_reason": load_error["error"],
                "eye_used": None, "eye_selection_reason": _missing_reason_bucket(load_error["error"]),
                "n_panels": 0, "n_panels_missing_audio": None,
            })

        # Successful calibration events - the (up to 3) panels sharing a mat file carry
        # identical calibration_info/eye_used/eye_selection_reason/recording_date; group by
        # recording_date to collapse them to one row per event.
        by_date = {}
        for panel, info in subject_data.matched_data.items():
            by_date.setdefault(info.get(KEY_RECORDING_DATE), []).append((panel, info))

        for recording_date, panel_infos in by_date.items():
            _, first_info = panel_infos[0]
            calibration_info = first_info.get(KEY_CALIBRATION_INFO)
            n_missing_audio = sum(1 for _, info in panel_infos if info.get(KEY_AUDIO_DATA) is None)

            row = {
                "group": group, "participant": subject_data.name, "recording_date": recording_date,
                "eye_used": first_info.get(KEY_ANALYSIS_EYE),
                "eye_selection_reason": first_info.get(KEY_EYE_SELECTION_REASON),
                "n_panels": len(panel_infos),
                "n_panels_missing_audio": n_missing_audio,
            }
            if calibration_info is None:
                row["calibration_missing"] = True
                row["missing_reason"] = "no calibration Data Quality message found"
            else:
                row["calibration_missing"] = False
                row["missing_reason"] = None
                row["dominant_eye_raw"] = calibration_info.get("dominant_eye")
                row["calibration_no"] = calibration_info["calibration_no"]
                row["validation_no"] = calibration_info["validation_no"]
                for eye in ("left", "right"):
                    average = calibration_info[eye]["average"] or {}
                    for key in CALIBRATION_METRIC_KEYS:
                        row[f"{eye}_{key}"] = average.get(key)
            rows.append(row)

    df = pd.DataFrame(rows)
    if "left_acc" in df.columns and "right_acc" in df.columns:
        df["used_acc"] = np.where(df["eye_used"] == "l", df["left_acc"],
                                   np.where(df["eye_used"] == "r", df["right_acc"], np.nan))
    return df


# ---------------------------------------------------------------------------------------
# Task 4: HC vs pwMS calibration-accuracy comparison
# ---------------------------------------------------------------------------------------
def aggregate_participant_accuracy(events_df):
    """
    One row per participant: mean used_acc across that participant's valid (calibration
    present, eye successfully assigned) events. Aggregating to the participant avoids
    pseudoreplication in the group comparison - a participant with 2 valid events
    contributes ONE data point, not two.
    """
    valid = events_df[(events_df["calibration_missing"] == False) & events_df["used_acc"].notna()]
    return valid.groupby(["group", "participant"], as_index=False)["used_acc"].mean()


def aggregate_participant_metric(events_df, value_col):
    """Generalizes aggregate_participant_accuracy to any column: one row per participant,
    mean of value_col across that participant's valid events."""
    valid = events_df[(events_df["calibration_missing"] == False) & events_df[value_col].notna()]
    return valid.groupby(["group", "participant"], as_index=False)[value_col].mean()


def _group_stats_text(hc_vals, ms_vals):
    """'HC: mu=X sigma=Y (n=N) / pwMS: ... / Total: ...' for annotating a plot with all
    three versions of mean/SD at once (per Noam's request - shown on every relevant plot)."""
    all_vals = pd.concat([pd.Series(hc_vals), pd.Series(ms_vals)])

    def fmt(vals, label):
        vals = pd.Series(vals).dropna()
        if len(vals) == 0:
            return f"{label}: n=0"
        return f"{label}: μ={vals.mean():.3f} σ={vals.std(ddof=1):.3f} (n={len(vals)})"

    return "\n".join([fmt(hc_vals, "HC"), fmt(ms_vals, "pwMS"), fmt(all_vals, "Total")])


def _annotate_histogram_outliers(ax, values, participants, color, z=2.0):
    """
    Labels every point more than z SD from the mean of `values` with its participant name
    (from the aligned `participants` values), placed near the top of the axis at that
    point's x-position (rotated, so it reads like a flag pointing down at its bar). Ranks
    are staggered across a few vertical levels so nearby outliers don't overlap.
    """
    values = pd.Series(values).reset_index(drop=True)
    participants = pd.Series(participants).reset_index(drop=True)
    mean, std = values.mean(), values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return mean, std

    mask = (values - mean).abs() > z * std
    outliers = pd.DataFrame({"value": values[mask], "participant": participants[mask]}).sort_values("value")
    if not outliers.empty:
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        for rank, (_, row) in enumerate(outliers.iterrows()):
            y_frac = 0.95 - (rank % 4) * 0.12
            ax.annotate(row["participant"], (row["value"], ymin + span * y_frac), rotation=90,
                        fontsize=7, color=color, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75))
    return mean, std


def run_group_ttest(events_df):
    """
    Independent-samples t-test (Student's, equal variance assumed) comparing mean
    calibration accuracy between HC and pwMS, one value per participant (see
    aggregate_participant_accuracy). Returns t-statistic, df, p-value, group descriptives,
    and Cohen's d (pooled-SD effect size).
    """
    agg = aggregate_participant_accuracy(events_df)
    hc = agg.loc[agg["group"] == "HC", "used_acc"].dropna()
    ms = agg.loc[agg["group"] == "pwMS", "used_acc"].dropna()

    t_stat, p_value = stats.ttest_ind(hc, ms, equal_var=True)
    dof = len(hc) + len(ms) - 2
    pooled_std = np.sqrt(((len(hc) - 1) * hc.var(ddof=1) + (len(ms) - 1) * ms.var(ddof=1)) / dof)
    cohens_d = (hc.mean() - ms.mean()) / pooled_std if pooled_std > 0 else np.nan

    return {
        "n_HC": len(hc), "n_pwMS": len(ms),
        "mean_HC": hc.mean(), "sd_HC": hc.std(ddof=1),
        "mean_pwMS": ms.mean(), "sd_pwMS": ms.std(ddof=1),
        "t_statistic": t_stat, "degrees_of_freedom": dof, "p_value": p_value,
        "cohens_d": cohens_d,
    }, hc, ms


def plot_group_accuracy_histogram(hc, ms, ttest_result, output_dir):
    """Overlaid HC/pwMS accuracy histograms with the t-test result annotated on the figure."""
    os.makedirs(output_dir, exist_ok=True)
    shared_bins = np.histogram_bin_edges(pd.concat([hc, ms]), bins=20)

    plt.figure(figsize=(9, 6))
    plt.hist(hc, bins=shared_bins, color="skyblue", edgecolor="k", alpha=0.6,
              label=f"HC (n={ttest_result['n_HC']}, mean={ttest_result['mean_HC']:.3f})")
    plt.hist(ms, bins=shared_bins, color="salmon", edgecolor="k", alpha=0.6,
              label=f"pwMS (n={ttest_result['n_pwMS']}, mean={ttest_result['mean_pwMS']:.3f})")
    plt.xlabel("Mean calibration accuracy per participant (deg)")
    plt.ylabel("Count (participants)")
    plt.title("Calibration accuracy: HC vs pwMS")
    stats_text = (
        f"t({ttest_result['degrees_of_freedom']}) = {ttest_result['t_statistic']:.3f}, "
        f"p = {ttest_result['p_value']:.4f}\n"
        f"Cohen's d = {ttest_result['cohens_d']:.3f}"
    )
    plt.gca().text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                    ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    plt.gca().text(0.02, 0.98, _group_stats_text(hc, ms), transform=plt.gca().transAxes,
                    ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))
    plt.legend(loc="upper left", bbox_to_anchor=(0.0, 0.78))
    plt.tight_layout()

    save_path = os.path.join(output_dir, "ttest_histogram_HC_vs_MS.png")
    _safe_savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")
    return save_path


def run_group_missing_chisq(events_df):
    """
    Chi-square test of independence: is calibration_missing rate associated with group
    (HC vs pwMS)? Computed at the calibration-EVENT level (the atomic unit throughout this
    module) - each calibration attempt is treated as one trial, success or failure.
    """
    contingency = pd.crosstab(events_df["group"], events_df["calibration_missing"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    rates = events_df.groupby("group")["calibration_missing"].mean()
    return {
        "chi2_statistic": chi2, "degrees_of_freedom": dof, "p_value": p_value,
        "pct_missing_HC": 100 * rates.get("HC", np.nan),
        "pct_missing_pwMS": 100 * rates.get("pwMS", np.nan),
        "n_events_HC": int((events_df["group"] == "HC").sum()),
        "n_events_pwMS": int((events_df["group"] == "pwMS").sum()),
    }


# ---------------------------------------------------------------------------------------
# Task 5: profile all four quality metrics (both eyes, by group) on deduplicated events
# ---------------------------------------------------------------------------------------
QUALITY_METRICS = ["acc", "std", "rms", "data_loss"]
# Rough literature reference bands (deg) for context only - NOT an operative cutoff.
LITERATURE_REFERENCE_BANDS = {"acc": (0.5, 1.0), "std": (0.1, 0.2)}


def add_used_eye_metric_columns(events_df):
    """Adds used_{metric} columns (the value for whichever eye was actually used) for all
    four quality metrics, generalizing the used_acc column built in build_calibration_events_table."""
    df = events_df.copy()
    for metric in QUALITY_METRICS:
        left_col, right_col = f"left_{metric}", f"right_{metric}"
        if left_col in df.columns and right_col in df.columns:
            df[f"used_{metric}"] = np.where(df["eye_used"] == "l", df[left_col],
                                             np.where(df["eye_used"] == "r", df[right_col], np.nan))
    return df


def plot_eye_asymmetry_histogram(events_df, metric, output_dir):
    """
    Histogram of (right eye value - left eye value) per calibration event, for one
    property, pooled across both groups (this is about left/right eye asymmetry, not a
    group comparison). Marks the mean and +/-2SD, and labels any event more than 2SD from
    the mean with its participant's name.
    """
    os.makedirs(output_dir, exist_ok=True)
    left_col, right_col = f"left_{metric}", f"right_{metric}"
    valid = events_df[events_df["calibration_missing"] == False].dropna(subset=[left_col, right_col])
    diff = valid[right_col] - valid[left_col]
    participants = valid["participant"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.histogram_bin_edges(diff, bins=25)
    ax.hist(diff, bins=bins, color="mediumpurple", edgecolor="k", alpha=0.75)

    mean, std = _annotate_histogram_outliers(ax, diff, participants, "black")
    ax.axvline(mean, color="black", linestyle="-", linewidth=1.3, label=f"mean={mean:.3f}")
    if std > 0:
        ax.axvline(mean + 2 * std, color="black", linestyle="--", linewidth=1, label="mean ± 2SD")
        ax.axvline(mean - 2 * std, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, label="zero (no asymmetry)")

    hc_diff = diff[valid["group"] == "HC"]
    ms_diff = diff[valid["group"] == "pwMS"]
    ax.text(0.02, 0.98, _group_stats_text(hc_diff, ms_diff), transform=ax.transAxes,
             ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

    ax.set_xlabel(f"right {metric} - left {metric}")
    ax.set_ylabel("Count (calibration events)")
    ax.set_title(f"Eye asymmetry: {metric} (right - left), all participants (n={len(diff)})")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"eye_asymmetry_{metric}.png")
    _safe_savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")
    return save_path


def profile_quality_metrics(events_df):
    """Descriptive stats for each metric x eye x group, on calibration-present events."""
    valid = events_df[events_df["calibration_missing"] == False]
    rows = []
    for eye in ("left", "right"):
        for metric in QUALITY_METRICS:
            col = f"{eye}_{metric}"
            if col not in valid.columns:
                continue
            for group in ("HC", "pwMS"):
                vals = valid.loc[valid["group"] == group, col].dropna()
                rows.append({
                    "eye": eye, "metric": metric, "group": group, "n": len(vals),
                    "mean": vals.mean(), "sd": vals.std(), "median": vals.median(),
                    "min": vals.min() if len(vals) else np.nan,
                    "max": vals.max() if len(vals) else np.nan,
                })
    return pd.DataFrame(rows)


def run_mean_between_eyes_ttest(events_df, metric):
    """
    Independent-samples t-test comparing HC vs pwMS on the mean-between-both-eyes value
    for one property (avg of left+right), computed PER CALIBRATION EVENT - no
    participant-level aggregation, so a participant with 2 valid events contributes 2
    points (explicit choice, see calibration_quality_analysis README).
    """
    left_col, right_col = f"left_{metric}", f"right_{metric}"
    valid = events_df[events_df["calibration_missing"] == False].dropna(subset=[left_col, right_col]).copy()
    valid["mean_between_eyes"] = (valid[left_col] + valid[right_col]) / 2

    hc = valid.loc[valid["group"] == "HC", "mean_between_eyes"]
    ms = valid.loc[valid["group"] == "pwMS", "mean_between_eyes"]
    t_stat, p_value = stats.ttest_ind(hc, ms, equal_var=True)
    dof = len(hc) + len(ms) - 2
    return {
        "metric": metric, "n_HC": len(hc), "n_pwMS": len(ms),
        "mean_HC": hc.mean(), "sd_HC": hc.std(ddof=1),
        "mean_pwMS": ms.mean(), "sd_pwMS": ms.std(ddof=1),
        "t_statistic": t_stat, "degrees_of_freedom": dof, "p_value": p_value,
    }


def plot_metric_distribution(events_df, metric, output_dir):
    """
    Side-by-side (left eye / right eye) HC-vs-pwMS overlaid histograms for one metric.
    Each group gets its OWN mean/+-2SD dashed lines (computed separately per group, not
    pooled) and any event more than 2SD from its group's own mean is labeled with its
    participant's name. The HC-vs-pwMS t-test on the mean-between-both-eyes value
    (run_mean_between_eyes_ttest, per calibration event) is annotated on the figure.
    Literature reference lines shown where available (context only, not a cutoff).
    """
    os.makedirs(output_dir, exist_ok=True)
    valid = events_df[events_df["calibration_missing"] == False]
    ttest_result = run_mean_between_eyes_ttest(events_df, metric)
    group_colors = {"HC": "skyblue", "pwMS": "salmon"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, eye in zip(axes, ("left", "right")):
        col = f"{eye}_{metric}"
        hc_vals = valid.loc[valid["group"] == "HC", col].dropna()
        ms_vals = valid.loc[valid["group"] == "pwMS", col].dropna()
        if len(hc_vals) == 0 and len(ms_vals) == 0:
            continue
        bins = np.histogram_bin_edges(pd.concat([hc_vals, ms_vals]), bins=20)
        for group, vals in (("HC", hc_vals), ("pwMS", ms_vals)):
            color = group_colors[group]
            ax.hist(vals, bins=bins, alpha=0.5, color=color, edgecolor="k", label=f"{group} (n={len(vals)})")
            if len(vals) > 1:
                participants = valid.loc[vals.index, "participant"]
                mean, std = _annotate_histogram_outliers(ax, vals, participants, color)
                ax.axvline(mean, color=color, linestyle="-", linewidth=1.5)
                if std > 0:
                    ax.axvline(mean + 2 * std, color=color, linestyle="--", linewidth=1)
                    ax.axvline(mean - 2 * std, color=color, linestyle="--", linewidth=1)
        if metric in LITERATURE_REFERENCE_BANDS:
            lo, hi = LITERATURE_REFERENCE_BANDS[metric]
            ax.axvline(lo, color="gray", linestyle=":", linewidth=1)
            ax.axvline(hi, color="gray", linestyle=":", linewidth=1, label=f"literature range ({lo}-{hi})")
        ax.set_title(f"{eye} eye (solid=mean, dashed=±2SD, per group)")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count (events)")
        ax.legend(fontsize=8, loc="upper right")
        ax.text(0.02, 0.98, _group_stats_text(hc_vals, ms_vals), transform=ax.transAxes,
                 ha="left", va="top", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

    fig.suptitle(f"{metric} distribution by group (deduplicated calibration events)", y=0.98, fontsize=14)
    stats_text = (
        f"Mean-between-eyes t-test (HC vs pwMS, per event): "
        f"t({ttest_result['degrees_of_freedom']}) = {ttest_result['t_statistic']:.3f}, "
        f"p = {ttest_result['p_value']:.4f}"
    )
    fig.text(0.5, 0.90, stats_text, ha="center", fontsize=10,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    plt.tight_layout(rect=[0, 0, 1, 0.86])

    save_path = os.path.join(output_dir, f"metric_distributions_{metric}.png")
    _safe_savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")
    return save_path, ttest_result


def compute_metric_correlations(events_df):
    """
    Pairwise Pearson correlations between the four quality metrics, using the value from
    whichever eye was actually used per event (see add_used_eye_metric_columns) - answers
    whether sessions that fail on accuracy also tend to fail on precision/data loss, or
    whether these are largely independent failure modes.
    """
    df = add_used_eye_metric_columns(events_df)
    valid = df[df["calibration_missing"] == False]
    cols = [f"used_{m}" for m in QUALITY_METRICS if f"used_{m}" in valid.columns]
    return valid[cols].corr()


def plot_accuracy_vs_dataloss(events_df, output_dir):
    """
    Scatter of accuracy vs data_loss, one point per calibration event, using the actually-
    used eye for both axes (consistent with every other group-comparison plot/dashboard).
    Marks the pooled mean +/-1SD for both axes (dashed reference lines) and reports the
    Pearson correlation.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = add_used_eye_metric_columns(events_df)
    valid = df[df["calibration_missing"] == False].dropna(subset=["used_acc", "used_data_loss"])

    r, p = stats.pearsonr(valid["used_acc"], valid["used_data_loss"])
    acc_mean, acc_std = valid["used_acc"].mean(), valid["used_acc"].std(ddof=1)
    dl_mean, dl_std = valid["used_data_loss"].mean(), valid["used_data_loss"].std(ddof=1)

    plt.figure(figsize=(9, 7))
    colors = valid["group"].map({"HC": "skyblue", "pwMS": "salmon"})
    plt.scatter(valid["used_acc"], valid["used_data_loss"], c=colors, edgecolor="k", alpha=0.7, s=40)
    for label, color in [("HC", "skyblue"), ("pwMS", "salmon")]:
        plt.scatter([], [], c=color, edgecolor="k", label=label)

    plt.axvline(acc_mean, color="black", linestyle="-", linewidth=1, label=f"mean acc={acc_mean:.2f}")
    plt.axvline(acc_mean + acc_std, color="black", linestyle="--", linewidth=1)
    plt.axvline(acc_mean - acc_std, color="black", linestyle="--", linewidth=1, label="acc mean ± 1SD")
    plt.axhline(dl_mean, color="dimgray", linestyle="-", linewidth=1, label=f"mean data_loss={dl_mean:.2f}")
    plt.axhline(dl_mean + dl_std, color="dimgray", linestyle="--", linewidth=1)
    plt.axhline(dl_mean - dl_std, color="dimgray", linestyle="--", linewidth=1, label="data_loss mean ± 1SD")

    plt.xlabel("Accuracy (deg, used eye)")
    plt.ylabel("Data loss (%, used eye)")
    plt.title("Accuracy vs data loss (used eye, mean ± SD marked)")
    stats_text = f"Pearson r = {r:.3f}\np = {p:.4g}\nn = {len(valid)}"
    plt.gca().text(0.98, 0.97, stats_text, transform=plt.gca().transAxes, ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    group_stats_text = (
        "Accuracy:\n" + _group_stats_text(valid.loc[valid["group"] == "HC", "used_acc"],
                                           valid.loc[valid["group"] == "pwMS", "used_acc"])
        + "\nData loss:\n" + _group_stats_text(valid.loc[valid["group"] == "HC", "used_data_loss"],
                                                valid.loc[valid["group"] == "pwMS", "used_data_loss"])
    )
    plt.gca().text(0.02, 0.98, group_stats_text, transform=plt.gca().transAxes, ha="left", va="top", fontsize=7,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "accuracy_vs_dataloss_scatter.png")
    _safe_savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")
    return save_path, {"pearson_r": r, "p_value": p, "n": len(valid)}


def plot_pooled_metric_distribution(events_df, metric, output_dir):
    """
    One histogram per property, ALL participants pooled together (no group split) - one
    point per participant: the mean of their used-eye value across their valid events
    (matching aggregate_participant_accuracy's convention). Marks the pooled mean +/-2SD
    and labels any participant beyond that with their name.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = add_used_eye_metric_columns(events_df)
    value_col = f"used_{metric}"
    agg = aggregate_participant_metric(df, value_col)

    fig, ax = plt.subplots(figsize=(9, 6))
    values = agg[value_col]
    participants = agg["participant"]
    bins = np.histogram_bin_edges(values, bins=25)
    ax.hist(values, bins=bins, color="mediumseagreen", edgecolor="k", alpha=0.75)

    mean, std = _annotate_histogram_outliers(ax, values, participants, "black")
    ax.axvline(mean, color="black", linestyle="-", linewidth=1.3, label=f"mean={mean:.3f}")
    if std > 0:
        ax.axvline(mean + 2 * std, color="black", linestyle="--", linewidth=1, label="mean ± 2SD")
        ax.axvline(mean - 2 * std, color="black", linestyle="--", linewidth=1)

    hc_vals = agg.loc[agg["group"] == "HC", value_col]
    ms_vals = agg.loc[agg["group"] == "pwMS", value_col]
    ax.text(0.02, 0.98, _group_stats_text(hc_vals, ms_vals), transform=ax.transAxes,
             ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

    ax.set_xlabel(f"{metric} (used eye, mean per participant)")
    ax.set_ylabel("Count (participants)")
    ax.set_title(f"{metric}: all participants pooled, HC + pwMS (n={len(agg)})")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"pooled_{metric}_distribution.png")
    _safe_savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")
    return save_path


# ---------------------------------------------------------------------------------------
# Task 6: threshold sensitivity ("impact") analysis
# ---------------------------------------------------------------------------------------
DEFAULT_THRESHOLD_CANDIDATES = {
    "acc": [1.0, 1.5, 2.0, 3.0],
    "std": [0.2, 0.3, 0.5],
    "rms": [0.15, 0.2, 0.3],
    "data_loss": [5, 10, 20],
}


def threshold_sensitivity_table(events_df, thresholds=None):
    """
    For each (metric, cutoff) candidate, how many calibration events and how many
    PARTICIPANTS would be excluded, split by group. "Events excluded" = this event's
    used-eye metric exceeds the cutoff. "Participants excluded" = participants left with
    ZERO usable events at all once this cutoff is applied ON TOP of the existing hard
    exclusions (calibration_missing) - i.e. the participant would be fully lost from the
    cohort, not just missing one of several events.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_CANDIDATES

    df = add_used_eye_metric_columns(events_df)
    valid = df[df["calibration_missing"] == False]
    all_participants_by_group = df.groupby("group")["participant"].unique().apply(set).to_dict()

    rows = []
    for metric, cutoffs in thresholds.items():
        col = f"used_{metric}"
        if col not in valid.columns:
            continue
        for cutoff in cutoffs:
            survives = valid[valid[col] <= cutoff]
            excluded_events = valid[valid[col] > cutoff]
            for group, all_participants in list(all_participants_by_group.items()) + [("ALL", set(df["participant"]))]:
                group_all = all_participants
                group_surviving = set(survives.loc[survives["group"] == group, "participant"]) if group != "ALL" \
                    else set(survives["participant"])
                group_excluded_events = excluded_events if group == "ALL" else excluded_events[excluded_events["group"] == group]
                rows.append({
                    "metric": metric, "cutoff": cutoff, "group": group,
                    "n_events_excluded": len(group_excluded_events),
                    "n_participants_fully_excluded": len(group_all - group_surviving),
                    "n_participants_total": len(group_all),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Caching + Task 7: interactive dashboard
# ---------------------------------------------------------------------------------------
_DASHBOARD_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_dashboard_template.html")


def get_or_build_events_table(main_data_path=DEFAULT_MAIN_DATA_PATH, output_root=DEFAULT_OUTPUT_ROOT, force=False):
    """
    Returns the calibration events table, reusing the cached CSV
    (data/calibration_events_clean.csv) if it already exists instead of re-sweeping every
    participant's raw .mat files (~35-40 min for the full cohort). Pass force=True to
    rebuild from raw data regardless (e.g. after a pipeline fix).
    """
    csv_path = os.path.join(output_root, "data", "calibration_events_clean.csv")
    if not force and os.path.exists(csv_path):
        print(f"Using cached events table: {csv_path}")
        return pd.read_csv(csv_path)

    df = build_calibration_events_table(main_data_path=main_data_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Built and cached events table ({len(df)} events): {csv_path}")
    return df


def render_dashboard_html(events_df, output_path):
    """
    Fills calibration_dashboard_template.html with the current events table (only the
    columns the dashboard needs) and writes the self-contained result to output_path.
    """
    df = add_used_eye_metric_columns(events_df)
    cols = ["group", "participant", "recording_date", "calibration_missing", "missing_reason",
            "eye_selection_reason", "used_acc", "used_std", "used_rms", "used_data_loss"]
    slim = df[cols].copy()
    slim["calibration_missing"] = slim["calibration_missing"].astype(bool)
    slim = slim.replace({np.nan: None})
    records = slim.to_dict(orient="records")

    with open(_DASHBOARD_TEMPLATE_PATH) as f:
        template = f.read()
    html = template.replace("__EVENTS_JSON__", json.dumps(records))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w") as f:
            f.write(html)
    except OSError:
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, "w") as f:
            f.write(html)
    print(f"Saved dashboard: {output_path}")
    return output_path


def run_full_analysis(main_data_path=DEFAULT_MAIN_DATA_PATH, output_root=DEFAULT_OUTPUT_ROOT, force_rebuild=False):
    """
    End-to-end: Tasks 3-7. Uses the cached events table unless force_rebuild=True. Writes
    every stats CSV / plot / the dashboard under output_root, matching the layout in
    README.md.
    """
    data_dir = os.path.join(output_root, "data")
    stats_dir = os.path.join(output_root, "stats")
    plots_dir = os.path.join(output_root, "plots")
    dashboard_dir = os.path.join(output_root, "dashboard")
    for d in (data_dir, stats_dir, plots_dir, dashboard_dir):
        os.makedirs(d, exist_ok=True)

    events_df = get_or_build_events_table(main_data_path, output_root, force=force_rebuild)

    ttest_result, hc, ms = run_group_ttest(events_df)
    pd.DataFrame([ttest_result]).to_csv(os.path.join(stats_dir, "group_ttest_accuracy.csv"), index=False)
    plot_group_accuracy_histogram(hc, ms, ttest_result, plots_dir)

    chisq_result = run_group_missing_chisq(events_df)
    pd.DataFrame([chisq_result]).to_csv(os.path.join(stats_dir, "group_missing_chisq.csv"), index=False)

    profile_df = profile_quality_metrics(events_df)
    profile_df.to_csv(os.path.join(stats_dir, "metric_profiles_by_group_eye.csv"), index=False)

    mean_between_eyes_ttests = []
    for metric in QUALITY_METRICS:
        _, mbe_ttest = plot_metric_distribution(events_df, metric, plots_dir)
        mean_between_eyes_ttests.append(mbe_ttest)
        plot_eye_asymmetry_histogram(events_df, metric, plots_dir)
        plot_pooled_metric_distribution(events_df, metric, plots_dir)
    mean_between_eyes_ttest_df = pd.DataFrame(mean_between_eyes_ttests)
    mean_between_eyes_ttest_df.to_csv(os.path.join(stats_dir, "mean_between_eyes_ttests.csv"), index=False)

    _, acc_vs_dataloss_stats = plot_accuracy_vs_dataloss(events_df, plots_dir)
    pd.DataFrame([acc_vs_dataloss_stats]).to_csv(os.path.join(stats_dir, "accuracy_vs_dataloss_correlation.csv"), index=False)

    corr_df = compute_metric_correlations(events_df)
    corr_df.to_csv(os.path.join(stats_dir, "metric_correlations.csv"))

    impact_df = threshold_sensitivity_table(events_df)
    impact_df.to_csv(os.path.join(stats_dir, "threshold_impact_table.csv"), index=False)

    render_dashboard_html(events_df, os.path.join(dashboard_dir, "calibration_threshold_explorer.html"))

    return {
        "events_df": events_df, "ttest_result": ttest_result, "chisq_result": chisq_result,
        "profile_df": profile_df, "corr_df": corr_df, "impact_df": impact_df,
        "mean_between_eyes_ttest_df": mean_between_eyes_ttest_df,
        "acc_vs_dataloss_stats": acc_vs_dataloss_stats,
    }


if __name__ == "__main__":
    run_full_analysis()

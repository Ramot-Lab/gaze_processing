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
# Report assembly
# ---------------------------------------------------------------------------------------
def _format_summary_block(title, summary):
    lines = [title, "-" * len(title)]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    return "\n".join(lines)


def write_report(output_dir, calibration_summary, drift_summary, eye_comparison_summary):
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
    return path


def build_full_report(main_data_path=DEFAULT_MAIN_DATA_PATH, calibration_qc_dir=DEFAULT_CALIBRATION_QC_DIR, task="SDMT"):
    calibration_df, dominant_eye_df, drift_qa_df = load_qc_tables(calibration_qc_dir)

    calibration_summary = summarize_calibration_quality(calibration_df)
    drift_summary = summarize_drift_correction(drift_qa_df)

    eye_comparison_df = build_eye_comparison_table(dominant_eye_df, drift_qa_df, main_data_path, task)
    eye_comparison_df.to_csv(os.path.join(calibration_qc_dir, "eye_comparison_qa.csv"), index=False)
    eye_comparison_summary = summarize_eye_comparison(eye_comparison_df)

    report_path = write_report(calibration_qc_dir, calibration_summary, drift_summary, eye_comparison_summary)
    print(f"Report written to {report_path}")

    return {
        "calibration_summary": calibration_summary,
        "drift_summary": drift_summary,
        "eye_comparison_summary": eye_comparison_summary,
        "eye_comparison_df": eye_comparison_df,
        "report_path": report_path,
    }


if __name__ == "__main__":
    build_full_report()

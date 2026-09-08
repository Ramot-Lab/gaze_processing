"""
Three per-panel diagnostics, computed directly from PRE-interpolation, PRE-outlier-smoothing
raw gaze data (mirrors prepare_gaze_data_for_preprocessing's eye-selection decision, but reads
straight off the raw .mat arrays) - NOT matched_data[panel][KEY_TOBII_DATA], which has already
been outlier-smoothed + NaN-interpolated by ParticipantGazeDataManager.clean_outliers() and
would hide exactly what's being measured here:

  1. dist_center_first5 - distance from screen center (0.5, 0.5) of the mean of the first 5
     chronological samples in the panel
  2. pct_negative_y - % of a panel's samples with y < 0 (i.e. negative-y values RETAINED after
     the out-of-range NaN policy - values below OUT_OF_RANGE_Y_BOUNDS[0] are already NaN)
  3. nan_ratio_pct - NaN ratio per panel (same definition used for the eye-selection/exclusion
     NaN-ratio check)

Only panels that pass the current accuracy+NaN-ratio eye-selection policy are included (same
panel set ParticipantGazeDataManager would load).
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration_drift_qa import _iter_subject_dirs
from constants import MAX_VALID_NAN_VALUES
from exclusion_policy import ACCURACY_EXCLUSION_THRESHOLD_DEG
from participant_gaze_data_manager import (
    ParticipantGazeDataManager, extract_last_calibration_message, parse_calibration_quality_message,
    _apply_out_of_range_policy,
)

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc"


def sweep():
    dummy = ParticipantGazeDataManager.__new__(ParticipantGazeDataManager)
    dummy.task_validation_filter_param = "run"
    dummy.task = "SDMT"

    rows = []
    for group, subject_dir in _iter_subject_dirs(MAIN_DATA_PATH, ("HC", "pwMS")):
        if "SDMT" not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)
        task_dir = os.path.join(subject_dir, "SDMT")
        mat_files = glob.glob(os.path.join(task_dir, "*.mat"))
        mat_files = [f for f in mat_files if "run" in os.path.split(f)[1].lower()]
        try:
            loaded_mats, _ = dummy._load_and_dedupe_run_files(mat_files)
        except Exception as e:
            print(f"  FAILED dedup {participant}: {e}")
            continue

        for mat in loaded_mats:
            try:
                messages = mat["messages"]
                left_gaze = mat["data"].gaze.left.gazePoint.onDisplayArea
                right_gaze = mat["data"].gaze.right.gazePoint.onDisplayArea
                tobi_ts = mat["data"].gaze.systemTimeStamp
                if np.size(tobi_ts) == 0:
                    continue
                left_gaze, right_gaze = _apply_out_of_range_policy(left_gaze, right_gaze)

                panel_indices, break_indices = dummy.break_mat_into_pannels(messages)
                if len(panel_indices) != 3 or len(break_indices) != 3:
                    continue
                panel_start_times = [messages[i][0] for i in panel_indices]
                break_start_times = [messages[i][0] for i in break_indices]

                found = extract_last_calibration_message(messages)
                calib_parsed = parse_calibration_quality_message(found[1]) if found is not None else None
                acc = {}
                for eye, key in (("l", "left"), ("r", "right")):
                    average = (calib_parsed[key]["average"] if calib_parsed else None) or {}
                    acc[eye] = average.get("acc")
                acc_ok = {eye: (acc[eye] is not None and acc[eye] < ACCURACY_EXCLUSION_THRESHOLD_DEG)
                          for eye in ("l", "r")}
                if not acc_ok["l"] and not acc_ok["r"]:
                    continue

                task_data = mat["task_data"].__dict__
                codes = [(name[-2:]).replace("_", "").lower() for name in list(task_data.keys())[1::2]]

                gaze_by_eye = {"l": left_gaze, "r": right_gaze}

                for i in range(3):
                    indices = np.where((tobi_ts > panel_start_times[i]) & (tobi_ts < break_start_times[i]))[0]
                    if len(indices) == 0:
                        continue
                    candidates = []
                    for eye in ("l", "r"):
                        if not acc_ok[eye]:
                            continue
                        g = gaze_by_eye[eye]
                        panel_x = g[0, indices]
                        panel_y = g[1, indices]
                        panel_data = np.stack([panel_x, panel_y], axis=1)
                        nan_ratio = (np.count_nonzero(np.isnan(panel_data)) // 2) / len(panel_data)
                        if nan_ratio < MAX_VALID_NAN_VALUES:
                            candidates.append((eye, acc[eye], nan_ratio, panel_x, panel_y))
                    if not candidates:
                        continue
                    candidates.sort(key=lambda c: (c[1], c[2]))
                    eye, eye_acc, nan_ratio, panel_x, panel_y = candidates[0]

                    panel_code = codes[i] if i < len(codes) else f"panel_{i + 1}"
                    n = len(panel_x)
                    pct_negative_y = 100.0 * np.sum(panel_y < 0) / n
                    first5_x = np.nanmean(panel_x[:5])
                    first5_y = np.nanmean(panel_y[:5])
                    dist_center = float(np.hypot(first5_x - 0.5, first5_y - 0.5))

                    rows.append({
                        "group": group, "participant": participant, "panel": panel_code, "eye": eye,
                        "n_samples": n, "nan_ratio_pct": nan_ratio * 100,
                        "pct_negative_y": pct_negative_y,
                        "first5_mean_x": first5_x, "first5_mean_y": first5_y,
                        "dist_center_first5": dist_center,
                    })
            except Exception as e:
                print(f"  FAILED {participant}: {e}")
                continue

    return pd.DataFrame(rows)


def _plot_with_sd_lines(df, value_col, title, xlabel, save_prefix, color):
    sub = df.dropna(subset=[value_col])
    values = sub[value_col]
    mean = values.mean()
    sd = values.std()

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.hist(values, bins=60, color=color, edgecolor="k", alpha=0.75)

    for k in (-3, -2, -1, 0, 1, 2, 3):
        x = mean + k * sd
        if values.min() <= x <= values.max():
            line_color = "black" if k == 0 else ("goldenrod" if abs(k) == 1 else ("orangered" if abs(k) == 2 else "darkred"))
            label = "mean" if k == 0 else f"{'+' if k > 0 else ''}{k} SD"
            ax.axvline(x, color=line_color, linestyle="--", linewidth=1)
            ax.text(x, ax.get_ylim()[1] * 0.97, label, rotation=90, va="top", ha="right", fontsize=7, color=line_color)

    ax.set_title(f"{title}\nmean={mean:.4f}, sd={sd:.4f}, n={len(values)}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (panels)")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{save_prefix}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("saved", save_path)

    threshold = mean + 2 * sd
    outliers = sub[sub[value_col] > threshold].sort_values(value_col, ascending=False)
    outliers_path = os.path.join(OUTPUT_DIR, f"{save_prefix}_outliers_above_mean2sd.csv")
    outliers.to_csv(outliers_path, index=False)
    print(f"{len(outliers)} panels above mean+2sd ({threshold:.4f}) -> {outliers_path}")
    print(outliers[["group", "participant", "panel", value_col]].to_string(index=False))
    return mean, sd, outliers


def main():
    df = sweep()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "panel_diagnostics.csv"), index=False)
    print(f"{len(df)} panels analyzed, saved to panel_diagnostics.csv")

    _plot_with_sd_lines(
        df, "dist_center_first5",
        "Distance of first-5-sample mean location from screen center (0.5, 0.5)",
        "distance (normalized units)", "hist_dist_center_first5", "mediumseagreen",
    )
    _plot_with_sd_lines(
        df, "pct_negative_y",
        "% of panel samples with negative y (retained after out-of-range NaN policy)",
        "% negative-y samples", "hist_pct_negative_y", "cornflowerblue",
    )
    _plot_with_sd_lines(
        df, "nan_ratio_pct",
        "NaN ratio per panel (all participants pooled)",
        "% NaN samples", "hist_nan_ratio_pct", "salmon",
    )
    print("ALL DONE")


if __name__ == "__main__":
    main()

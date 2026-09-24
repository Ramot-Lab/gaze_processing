"""
Per participant-panel:
  pct_time_in_roi     - % of ALL samples (fixation+saccade, i.e. true "time") whose
                        pixel-space position falls inside ANY ROI, using the SAME
                        shape/factor conventions trial_manager.py actually uses:
                        dictionary ROIs (idx<18) -> shape="square", factor=1 (the primary
                        check at trial_manager.py:291); search-grid ROIs (idx>=18) ->
                        shape="circle", factor=2 (trial_manager.py:79/90)
  pct_fixation_dictionary - % of FIXATION-classified samples with y < DICTIONARY_BOUNDARY_RATIO
  pct_fixation_text       - % of FIXATION-classified samples with y >= DICTIONARY_BOUNDARY_RATIO
                        (these two sum to 100% of fixation samples)

Population: every participant except DONTUSE (Tobii_Sucks KEPT); eye = best-calibrated per
event (lower accuracy, regardless of its value - no accuracy threshold), falling back to
Dom_Eye if no calibration message exists for that event. NO NaN-ratio exclusion at all -
MAX_VALID_NAN_VALUES is monkey-patched to >1 for this sweep so every panel that loads is
kept regardless of its NaN ratio (the module still enforces a 10% gate by default, even on
the explicit_override/analysis_eye path, so this is the only way to fully disable it without
touching the shared pipeline file).

ROI detection (RoiFinder, cv2 contour finding on the panel image) is deterministic per PANEL
CODE, not per participant - panel images are shared/fixed, so each of the ~6 codes is only
computed once and reused for every participant.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd

import participant_gaze_data_manager as pgdm
pgdm.MAX_VALID_NAN_VALUES = 1.1  # disable the NaN-ratio gate for this sweep (see module docstring)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration_drift_qa import DICTIONARY_BOUNDARY_RATIO
from constants import FIXATION_CSV_KEY_FIXATION, FIXATION_IDX
from participant_gaze_data_manager import (
    ParticipantGazeDataManager, extract_last_calibration_message, parse_calibration_quality_message,
)
from RoiFinder import RoiFinder
from utils import prepare_image_and_gaze

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc/roi_time_analysis"


def _iter_dontuse_only(main_data_path, groups=("HC", "pwMS")):
    for group in groups:
        for subject_dir in sorted(glob.glob(os.path.join(main_data_path, group, "*"))):
            if not os.path.isdir(subject_dir):
                continue
            if "DONTUSE" in os.path.basename(subject_dir).upper():
                continue
            yield group, subject_dir


def _best_eye_per_event(dummy, subject_dir):
    """recording_date -> 'l'/'r': best-calibrated eye regardless of accuracy value, falling
    back to Dom_Eye if no calibration message exists for that event at all."""
    task_dir = os.path.join(subject_dir, "SDMT")
    if not os.path.isdir(task_dir):
        return {}
    mat_files = glob.glob(os.path.join(task_dir, "*.mat"))
    mat_files = [f for f in mat_files if "run" in os.path.split(f)[1].lower()]
    try:
        loaded_mats, _ = dummy._load_and_dedupe_run_files(mat_files)
    except Exception:
        return {}
    result = {}
    for mat in loaded_mats:
        found = extract_last_calibration_message(mat["messages"])
        calib = parse_calibration_quality_message(found[1]) if found is not None else None
        l_acc = (calib["left"]["average"] or {}).get("acc") if calib else None
        r_acc = (calib["right"]["average"] or {}).get("acc") if calib else None
        if l_acc is None and r_acc is None:
            dom_eye = mat.get("Dom_Eye")
            eye = dom_eye.lower() if isinstance(dom_eye, str) and dom_eye.lower() in ("l", "r") else None
        elif r_acc is None or (l_acc is not None and l_acc <= r_acc):
            eye = "l"
        else:
            eye = "r"
        try:
            date = dummy.get_creation_time(mat)
        except Exception:
            date = None
        result[date] = eye
    return result


def _get_rois(panel_code, roi_cache, sd, panel):
    if panel_code not in roi_cache:
        img = sd.get_panel_img(panel)
        dummy_annotated = pd.DataFrame({"eye_horizontal": [0.5], "eye_vertical": [0.5]})
        resized_img, _ = prepare_image_and_gaze(img, dummy_annotated)
        roi_cache[panel_code] = RoiFinder(panel_code, resized_img).rois
    return roi_cache[panel_code]


def _pct_inside_any_roi(px, py, rois):
    inside = np.zeros(len(px), dtype=bool)
    for r in rois:
        cx, cy = r.center
        if r.idx < 18:  # dictionary - square, factor=1 (trial_manager.py:291)
            rad = r.radius
            inside |= (px >= cx - rad) & (px <= cx + rad) & (py >= cy - rad) & (py <= cy + rad)
        else:  # search-grid - circle, factor=2 (trial_manager.py:79/90)
            rad2 = r.radius * 2
            inside |= (px - cx) ** 2 + (py - cy) ** 2 <= rad2 ** 2
    return 100.0 * np.mean(inside)


def process_panel(sd, panel, roi_cache):
    annotated = sd.annotate_gaze_events(panel)
    img = sd.get_panel_img(panel)
    resized_img, scaled = prepare_image_and_gaze(img, annotated)
    px = scaled["eye_horizontal"].to_numpy()
    py = scaled["eye_vertical"].to_numpy()
    rois = _get_rois(panel, roi_cache, sd, panel)
    pct_in_roi = _pct_inside_any_roi(px, py, rois)

    evt = annotated[FIXATION_CSV_KEY_FIXATION].to_numpy()
    fix_mask = evt == FIXATION_IDX
    y_norm = annotated["eye_vertical"].to_numpy()
    n_fix = int(fix_mask.sum())
    if n_fix > 0:
        pct_fix_dict = 100.0 * np.mean(y_norm[fix_mask] < DICTIONARY_BOUNDARY_RATIO)
        pct_fix_text = 100.0 - pct_fix_dict
    else:
        pct_fix_dict, pct_fix_text = np.nan, np.nan
    return pct_in_roi, pct_fix_dict, pct_fix_text, n_fix


def sweep():
    dummy = ParticipantGazeDataManager.__new__(ParticipantGazeDataManager)
    dummy.task_validation_filter_param = "run"
    dummy.task = "SDMT"
    roi_cache = {}

    rows = []
    n_participants = 0
    for group, subject_dir in _iter_dontuse_only(MAIN_DATA_PATH, ("HC", "pwMS")):
        if "SDMT" not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)
        best_eye = _best_eye_per_event(dummy, subject_dir)
        needed_eyes = {e for e in best_eye.values() if e is not None}
        if not needed_eyes:
            continue
        n_participants += 1

        for eye in needed_eyes:
            try:
                sd = ParticipantGazeDataManager(subject_dir, MAIN_DATA_PATH, "SDMT", group, analysis_eye=eye)
            except Exception as e:
                print(f"  FAILED {participant} eye={eye}: {e}")
                continue
            for panel in list(sd.matched_data.keys()):
                info = sd.matched_data[panel]
                date = info["recording_date"]
                if best_eye.get(date) != eye:
                    continue
                try:
                    pct_in_roi, pct_fix_dict, pct_fix_text, n_fix = process_panel(sd, panel, roi_cache)
                except Exception as e:
                    print(f"  FAILED {participant}/{panel} eye={eye}: {e}")
                    continue
                rows.append({"group": group, "participant": participant, "panel": panel, "eye": eye,
                             "n_fixation_samples": n_fix, "pct_time_in_roi": pct_in_roi,
                             "pct_fixation_dictionary": pct_fix_dict, "pct_fixation_text": pct_fix_text})

        if n_participants % 15 == 0:
            print(f"... {n_participants} participants done, {len(rows)} panels so far")

    return pd.DataFrame(rows)


def _plot_hist_with_outliers(df, metric_col, title, xlabel, save_path, color, bins=50, both_tails=False):
    sub = df.dropna(subset=[metric_col])
    values = sub[metric_col]
    mean = values.mean()
    sd = values.std()
    hi = mean + 2 * sd
    lo = mean - 2 * sd

    if both_tails:
        outliers = sub[(sub[metric_col] > hi) | (sub[metric_col] < lo)].copy()
        outliers["direction"] = np.where(outliers[metric_col] > hi, "above", "below")
    else:
        outliers = sub[sub[metric_col] > hi].copy()
        outliers["direction"] = "above"
    outliers = outliers.sort_values(metric_col, ascending=False)
    labels = [f"{r.participant}/{r.panel} ({getattr(r, metric_col):.2f}, {r.direction})" for r in outliers.itertuples()]

    n_cols = 3 if len(labels) > 15 else (2 if len(labels) > 6 else 1)
    n_rows = max(1, -(-len(labels) // n_cols)) if labels else 1
    text_height = max(0.7, 0.20 * n_rows + 0.4)

    fig = plt.figure(figsize=(12, 6.5 + text_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.5, text_height])
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    ax.hist(values, bins=bins, color=color, edgecolor="k", alpha=0.75)
    line_specs = [(0, "-", "black", "mean"), (1, "--", "goldenrod", "+1 SD"),
                  (2, "--", "orangered", "+2 SD"), (3, "--", "darkred", "+3 SD"),
                  (-1, "--", "goldenrod", "-1 SD"), (-2, "--", "orangered", "-2 SD"), (-3, "--", "darkred", "-3 SD")]
    for k, lstyle, lcolor, lbl in line_specs:
        x = mean + k * sd
        if values.min() <= x <= values.max():
            ax.axvline(x, color=lcolor, linestyle=lstyle, linewidth=1)
            ax.text(x, ax.get_ylim()[1] * 0.97, lbl, rotation=90, va="top", ha="right", fontsize=7, color=lcolor)

    ax.set_title(f"{title}\nmean={mean:.3f}, sd={sd:.3f}, n={len(values)}, n_outliers={len(labels)}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (panels)")

    if labels:
        col_texts = ["\n".join(labels[i::n_cols]) for i in range(n_cols)]
        for i, col_text in enumerate(col_texts):
            ax_text.text(0.02 + i * (0.98 / n_cols), 0.95, col_text, transform=ax_text.transAxes,
                         fontsize=7, va="top", ha="left", family="monospace")
        ax_text.set_title(f"participant/panel outliers (n={len(labels)}):", fontsize=9, loc="left")
    else:
        ax_text.text(0.02, 0.5, "No outliers", fontsize=9, transform=ax_text.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"saved {save_path} (n_outliers={len(labels)})")
    return outliers


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = sweep()
    csv_path = os.path.join(OUTPUT_DIR, "roi_time_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"DONE: {len(df)} panels saved to {csv_path}")

    o1 = _plot_hist_with_outliers(df, "pct_time_in_roi",
                                   "% of samples inside any ROI (dictionary: square/factor1, grid: circle/factor2)",
                                   "% time inside an ROI", os.path.join(OUTPUT_DIR, "hist_pct_time_in_roi.png"), "teal")
    o2 = _plot_hist_with_outliers(df, "pct_fixation_dictionary",
                                   "% of fixation samples in the dictionary area (y < DICTIONARY_BOUNDARY_RATIO)",
                                   "% fixations in dictionary area", os.path.join(OUTPUT_DIR, "hist_pct_fixation_dictionary.png"), "mediumpurple")
    o3 = _plot_hist_with_outliers(df, "pct_fixation_text",
                                   "% of fixation samples in the text/search-grid area (y >= DICTIONARY_BOUNDARY_RATIO)",
                                   "% fixations in text area", os.path.join(OUTPUT_DIR, "hist_pct_fixation_text.png"), "darkgoldenrod")
    o1.to_csv(os.path.join(OUTPUT_DIR, "outliers_pct_time_in_roi.csv"), index=False)
    o2.to_csv(os.path.join(OUTPUT_DIR, "outliers_pct_fixation_dictionary.csv"), index=False)
    o3.to_csv(os.path.join(OUTPUT_DIR, "outliers_pct_fixation_text.csv"), index=False)
    print("ALL DONE")

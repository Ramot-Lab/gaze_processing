"""
Computes, per participant-panel, from the ALREADY-SAVED raw-sample parquet (no re-sweep of
the .mat files):
  pct_fixations_in_roi    - % of FIXATION-labeled samples inside ANY ROI, same shape/factor conventions
                            trial_manager.py actually uses: dictionary ROIs (idx<18) ->
                            square, factor=1 (line 291); search-grid ROIs (idx>=18) ->
                            circle, factor=2 (lines 79/90). NaN (blink) samples never count
                            as "inside" (comparisons with NaN are False), which is arguably
                            more honest than the real pipeline's interpolated straight-line
                            fill-in for the same gap.
  pct_fixation_dictionary  - % of FIXATION-labeled samples (evt column, already in the
                            parquet) with y < DICTIONARY_BOUNDARY_RATIO
  pct_fixation_text        - % of FIXATION-labeled samples with y >= DICTIONARY_BOUNDARY_RATIO

Population = whichever parquet is passed in (default: excluded_by_acc_and_nan, i.e. the
actual current pipeline: accuracy<2.5deg + NaN-ratio<10%, Tobii_Sucks excluded).

RoiFinder (cv2 contour detection) is deterministic per PANEL CODE - panel images are shared
across participants, so each of the ~6 codes is computed once and cached.
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration_drift_qa import DICTIONARY_BOUNDARY_RATIO, _panel_image_path
from constants import SCREEN_SIZE, FIXATION_IDX
from RoiFinder import RoiFinder

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
BASE_DIR = "/Volumes/ramot/Noam_M/calibration_qc/space_coverage_analysis"
DEFAULT_PARQUET = os.path.join(BASE_DIR, "excluded_by_acc_and_nan", "raw_samples_with_fixation_labels.parquet")
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc/roi_time_analysis"


def _get_rois(panel_code, roi_cache):
    if panel_code not in roi_cache:
        img = cv2.imread(_panel_image_path(MAIN_DATA_PATH, "SDMT", panel_code))
        h, w = img.shape[:2]
        scale = SCREEN_SIZE[0] / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, SCREEN_SIZE[0]))
        roi_cache[panel_code] = RoiFinder(panel_code, resized).rois
    return roi_cache[panel_code]


def _pct_inside_any_roi(px, py, rois):
    inside = np.zeros(len(px), dtype=bool)
    for r in rois:
        cx, cy = r.center
        if r.idx < 18:
            rad = r.radius
            inside |= (px >= cx - rad) & (px <= cx + rad) & (py >= cy - rad) & (py <= cy + rad)
        else:
            rad2 = r.radius * 2
            inside |= (px - cx) ** 2 + (py - cy) ** 2 <= rad2 ** 2
    return 100.0 * np.mean(inside)


def compute(parquet_path):
    df = pd.read_parquet(parquet_path)
    screen_h, screen_w = SCREEN_SIZE
    roi_cache = {}
    offset_cache = {}  # panel_code -> x_offset (image dims are fixed per panel code)
    rows = []
    for (group, participant, panel, eye), sub in df.groupby(["group", "participant", "panel", "eye"], sort=False):
        x = sub["x"].to_numpy(dtype=np.float64)
        y = sub["y"].to_numpy(dtype=np.float64)
        evt = sub["evt"].to_numpy()

        # dictionary/text split - directly on normalized y, fixation-labeled samples only
        fix_mask = evt == FIXATION_IDX
        n_fix = int(fix_mask.sum())
        if n_fix > 0:
            y_fix = y[fix_mask]
            valid_fix = ~np.isnan(y_fix)
            pct_fix_dict = 100.0 * np.mean(y_fix[valid_fix] < DICTIONARY_BOUNDARY_RATIO) if valid_fix.any() else np.nan
            pct_fix_text = 100.0 - pct_fix_dict if not np.isnan(pct_fix_dict) else np.nan
        else:
            pct_fix_dict, pct_fix_text = np.nan, np.nan

        # ROI containment - pixel space, same scale/offset prepare_image_and_gaze uses.
        # Image dims (hence x_offset) are fixed per panel code - cache instead of
        # re-reading the image from the network mount for every participant.
        if panel not in offset_cache:
            img_path = _panel_image_path(MAIN_DATA_PATH, "SDMT", panel)
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            scale = screen_h / img_h
            new_w = int(img_w * scale)
            offset_cache[panel] = (screen_w - new_w) / 2
        x_offset = offset_cache[panel]
        px = x * screen_w - x_offset
        py = y * screen_h
        rois = _get_rois(panel, roi_cache)
        # % of FIXATION samples inside an ROI (not all samples - saccades excluded)
        pct_in_roi = _pct_inside_any_roi(px[fix_mask], py[fix_mask], rois) if n_fix > 0 else np.nan

        rows.append({"group": group, "participant": participant, "panel": panel, "eye": eye,
                     "n_samples": len(sub), "n_fixation_samples": n_fix,
                     "pct_fixations_in_roi": pct_in_roi,
                     "pct_fixation_dictionary": pct_fix_dict, "pct_fixation_text": pct_fix_text})

    return pd.DataFrame(rows)


def _plot_hist_with_outliers(df, metric_col, title, xlabel, save_path, color, bins=50):
    sub = df.dropna(subset=[metric_col])
    values = sub[metric_col]
    mean = values.mean()
    sd = values.std()
    hi, lo = mean + 2 * sd, mean - 2 * sd

    outliers = sub[(sub[metric_col] > hi) | (sub[metric_col] < lo)].copy()
    outliers["direction"] = np.where(outliers[metric_col] > hi, "above", "below")
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
        xv = mean + k * sd
        if values.min() <= xv <= values.max():
            ax.axvline(xv, color=lcolor, linestyle=lstyle, linewidth=1)
            ax.text(xv, ax.get_ylim()[1] * 0.97, lbl, rotation=90, va="top", ha="right", fontsize=7, color=lcolor)

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
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PARQUET
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = compute(parquet_path)
    csv_path = os.path.join(OUTPUT_DIR, "roi_time_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"DONE: {len(df)} panels (from {parquet_path}) saved to {csv_path}")

    o1 = _plot_hist_with_outliers(df, "pct_fixations_in_roi",
                                   "% of fixation samples inside any ROI (dictionary: square/factor1, grid: circle/factor2)",
                                   "% fixations inside an ROI", os.path.join(OUTPUT_DIR, "hist_pct_fixations_in_roi.png"), "teal")
    o2 = _plot_hist_with_outliers(df, "pct_fixation_dictionary",
                                   "% of fixation samples in the dictionary area (y < DICTIONARY_BOUNDARY_RATIO)",
                                   "% fixations in dictionary area", os.path.join(OUTPUT_DIR, "hist_pct_fixation_dictionary.png"), "mediumpurple")
    o3 = _plot_hist_with_outliers(df, "pct_fixation_text",
                                   "% of fixation samples in the text/search-grid area (y >= DICTIONARY_BOUNDARY_RATIO)",
                                   "% fixations in text area", os.path.join(OUTPUT_DIR, "hist_pct_fixation_text.png"), "darkgoldenrod")
    o1.to_csv(os.path.join(OUTPUT_DIR, "outliers_pct_fixations_in_roi.csv"), index=False)
    o2.to_csv(os.path.join(OUTPUT_DIR, "outliers_pct_fixation_dictionary.csv"), index=False)
    o3.to_csv(os.path.join(OUTPUT_DIR, "outliers_pct_fixation_text.csv"), index=False)
    print("ALL DONE")

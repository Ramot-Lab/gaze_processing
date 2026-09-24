"""
Heatmaps (ALL samples, fixation+saccade combined, off-screen included) for every panel of
every participant that appeared in EITHER the pct_fixations_in_roi or the
pct_fixation_dictionary/text outlier lists (build_roi_time_from_parquet.py) - not just their
flagged panel(s), all of their panels, so the flagged ones can be seen in context next to
their own unflagged panels.

Saved organized as outlier_heatmaps/<participant>/<panel>.png. Each title states whether
THIS SPECIFIC panel is an outlier, and of what (ROI / dictionary-text / both / neither).
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration_drift_qa import _panel_image_path
from constants import SCREEN_SIZE

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
RAW_PARQUET = "/Volumes/ramot/Noam_M/calibration_qc/space_coverage_analysis/excluded_by_acc_and_nan/raw_samples_with_fixation_labels.parquet"
ROI_ANALYSIS_DIR = "/Volumes/ramot/Noam_M/calibration_qc/roi_time_analysis"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc/outlier_heatmaps"


def _outlier_label(participant, panel, roi_set, dict_set):
    tags = []
    if (participant, panel) in roi_set:
        tags.append("ROI outlier")
    if (participant, panel) in dict_set:
        tags.append("dictionary/text outlier")
    return ", ".join(tags) if tags else "not an outlier"


def plot_one_heatmap(sub_df, participant, panel, outlier_label, save_path, margin_frac=0.08, bins=250):
    eye = sub_df["eye"].iloc[0]
    x = sub_df["x"].to_numpy()
    y = sub_df["y"].to_numpy()
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n_total = len(sub_df)
    n_valid = len(x)

    screen_h, screen_w = SCREEN_SIZE
    img = cv2.imread(_panel_image_path(MAIN_DATA_PATH, "SDMT", panel))
    img_h, img_w = img.shape[:2]
    scale = screen_h / img_h
    new_w = int(img_w * scale)
    resized_img = cv2.resize(img, (new_w, screen_h))
    x_offset = (screen_w - new_w) / 2

    pixel_x = x * screen_w - x_offset
    pixel_y = y * screen_h

    pad_left = max(0, -pixel_x.min()) * (1 + margin_frac)
    pad_right = max(0, pixel_x.max() - new_w) * (1 + margin_frac)
    pad_top = max(0, -pixel_y.min()) * (1 + margin_frac)
    pad_bottom = max(0, pixel_y.max() - screen_h) * (1 + margin_frac)
    pad_left, pad_right, pad_top, pad_bottom = (int(np.ceil(p)) for p in (pad_left, pad_right, pad_top, pad_bottom))

    canvas_w = screen_w + pad_left + pad_right
    canvas_h = screen_h + pad_top + pad_bottom
    canvas_x = pixel_x + pad_left
    canvas_y = pixel_y + pad_top

    base_canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)
    base_canvas[pad_top:pad_top + screen_h, pad_left:pad_left + new_w] = resized_img
    base_canvas_rgb = cv2.cvtColor(base_canvas, cv2.COLOR_BGR2RGB)

    heatmap, _, _ = np.histogram2d(canvas_x, canvas_y, bins=bins, range=[[0, canvas_w], [0, canvas_h]])
    heatmap = heatmap.T
    heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)

    fig, ax = plt.subplots(figsize=(canvas_w / 200, canvas_h / 200))
    ax.imshow(base_canvas_rgb, extent=[0, canvas_w, canvas_h, 0])
    im = ax.imshow(heatmap_masked, extent=[0, canvas_w, canvas_h, 0], cmap="inferno",
                   alpha=0.75, norm=matplotlib.colors.LogNorm())
    ax.add_patch(plt.Rectangle((pad_left, pad_top), new_w, screen_h, fill=False,
                                edgecolor="cyan", linewidth=2))
    plt.colorbar(im, ax=ax, label="sample count (log scale)")
    ax.set_title(f"{participant}/{panel} (eye={eye}) - {outlier_label}\n"
                 f"n_samples={n_total} (n_valid={n_valid}, {n_total - n_valid} blink/NaN excluded), "
                 f"cyan box = true screen bounds")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    roi_out = pd.read_csv(os.path.join(ROI_ANALYSIS_DIR, "outliers_pct_fixations_in_roi.csv"))
    dic_out = pd.read_csv(os.path.join(ROI_ANALYSIS_DIR, "outliers_pct_fixation_dictionary.csv"))
    roi_set = set(zip(roi_out.participant, roi_out.panel))
    dic_set = set(zip(dic_out.participant, dic_out.panel))
    participants = sorted(set(roi_out.participant) | set(dic_out.participant))
    print(f"{len(participants)} participants: {participants}")

    df = pd.read_parquet(RAW_PARQUET, filters=[("participant", "in", participants)])
    print(f"loaded {len(df)} raw samples for these participants")

    n_saved = 0
    for (participant, panel), sub in df.groupby(["participant", "panel"], sort=True):
        outlier_label = _outlier_label(participant, panel, roi_set, dic_set)
        save_path = os.path.join(OUTPUT_DIR, participant, f"{panel}.png")
        try:
            plot_one_heatmap(sub, participant, panel, outlier_label, save_path)
            n_saved += 1
        except Exception as e:
            print(f"  FAILED {participant}/{panel}: {e}")
        if n_saved % 20 == 0:
            print(f"... {n_saved} heatmaps saved")

    print(f"ALL DONE: {n_saved} heatmaps saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

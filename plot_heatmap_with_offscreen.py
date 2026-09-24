"""
2D density heatmap for one participant/panel, INCLUDING off-screen samples - the canvas is
padded around the panel image (sized to the actual data extent, not a fixed guess) so
excursions outside [0,1]x[0,1] are visible instead of being clipped by a histogram range or
falling outside the plotted image, the same problem render_padded_gaze_video.py solved for
video. Reads the raw (untouched, native-NaN-still-present) per-sample table already saved
by build_raw_space_coverage_sweep.py - no re-sweep of the .mat files needed.
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
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc"


def plot_heatmap(participant, panel, save_path, margin_frac=0.08, bins=250):
    df = pd.read_parquet(RAW_PARQUET, filters=[("participant", "=", participant), ("panel", "=", panel)])
    if df.empty:
        raise ValueError(f"No raw samples found for {participant}/{panel} in {RAW_PARQUET}")
    eye = df["eye"].iloc[0]
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n_total = len(df)
    n_valid = len(x)

    screen_h, screen_w = SCREEN_SIZE
    img = cv2.imread(_panel_image_path(MAIN_DATA_PATH, "SDMT", panel))
    img_h, img_w = img.shape[:2]
    scale = screen_h / img_h
    new_w = int(img_w * scale)
    resized_img = cv2.resize(img, (new_w, screen_h))
    x_offset = (screen_w - new_w) / 2

    # Raw (unclipped) pixel positions - can legitimately be negative or beyond the screen.
    pixel_x = x * screen_w - x_offset
    pixel_y = y * screen_h

    # Pad the canvas to the ACTUAL data extent (plus a margin), not a fixed guess.
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

    heatmap, xedges, yedges = np.histogram2d(canvas_x, canvas_y, bins=bins,
                                              range=[[0, canvas_w], [0, canvas_h]])
    heatmap = heatmap.T  # histogram2d returns (x_bins, y_bins); imshow wants (rows=y, cols=x)
    heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)

    fig, ax = plt.subplots(figsize=(canvas_w / 200, canvas_h / 200))
    ax.imshow(base_canvas_rgb, extent=[0, canvas_w, canvas_h, 0])
    im = ax.imshow(heatmap_masked, extent=[0, canvas_w, canvas_h, 0], cmap="inferno",
                   alpha=0.75, norm=matplotlib.colors.LogNorm())
    # true screen boundary
    ax.add_patch(plt.Rectangle((pad_left, pad_top), new_w, screen_h, fill=False,
                                edgecolor="cyan", linewidth=2))
    plt.colorbar(im, ax=ax, label="sample count (log scale)")
    ax.set_title(f"{participant}/{panel} (eye={eye}) - gaze density including off-screen samples\n"
                 f"n_samples={n_total} (n_valid={n_valid}, {n_total - n_valid} blink/NaN excluded), "
                 f"cyan box = true screen bounds")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved {save_path}")


if __name__ == "__main__":
    plot_heatmap("GS739", "l4", os.path.join(OUTPUT_DIR, "heatmap_GS739_l4_offscreen.png"))

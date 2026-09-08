"""
Reads the pooled tables build_raw_xy_histograms.py already saved (raw_fixation_samples_xy.parquet,
fixation_blocks_xy.parquet - full-panel, as-is, no NaN cleaning) and plots histograms of ONLY
the out-of-[0,1]-range values, per axis, samples vs fixation blocks - a zoomed-in complement
to hist_x/y_samples_vs_blocks_asis.png, which can't show this tail at linear scale.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc"

sample_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "raw_fixation_samples_xy.parquet"))
block_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "fixation_blocks_xy.parquet"))


def _out_of_range(series):
    return series[(series < 0) | (series > 1)]


def _plot_pair(values_samples, values_blocks, axis_name, color, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, values, label in [(axes[0], values_samples, "raw samples"), (axes[1], values_blocks, "fixation blocks")]:
        ax.hist(values, bins=100, color=color, edgecolor="k", alpha=0.75)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.axvline(1, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{axis_name} - {label} (n={len(values)})")
        ax.set_xlabel(f"eye_{'horizontal' if axis_name == 'X' else 'vertical'} (out-of-[0,1]-range values only)")
    axes[0].set_ylabel("count")
    fig.suptitle(f"{axis_name}: OUT-OF-RANGE-ONLY values - raw samples vs fixation blocks (full panel, as-is)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print("saved", save_path)
    plt.close()


x_samples_oor = _out_of_range(sample_df["x"])
x_blocks_oor = _out_of_range(block_df["x"])
y_samples_oor = _out_of_range(sample_df["y"])
y_blocks_oor = _out_of_range(block_df["y"])

print(f"x out-of-range: {len(x_samples_oor)}/{len(sample_df)} samples, {len(x_blocks_oor)}/{len(block_df)} blocks")
print(f"y out-of-range: {len(y_samples_oor)}/{len(sample_df)} samples, {len(y_blocks_oor)}/{len(block_df)} blocks")

_plot_pair(x_samples_oor, x_blocks_oor, "X", "steelblue", os.path.join(OUTPUT_DIR, "hist_x_out_of_range_only.png"))
_plot_pair(y_samples_oor, y_blocks_oor, "Y", "darkorange", os.path.join(OUTPUT_DIR, "hist_y_out_of_range_only.png"))

print("DONE")

"""
Reads the two space-coverage sweep tables (build_raw_space_coverage_sweep.py) and plots,
for each of the 5 metrics x 2 population variants, a histogram with mean/+-1/2/3 SD lines
and the participant/panel names of every point above mean+2SD written directly on the image.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = "/Volumes/ramot/Noam_M/calibration_qc/space_coverage_analysis"
RAW_DIR = os.path.join(BASE_DIR, "raw")
EXCLUDED_DIR = os.path.join(BASE_DIR, "excluded_by_acc_and_nan")

METRICS = [
    ("pct_out_of_range", "% of raw samples outside [0,1] (x or y)", "% out-of-range samples", "steelblue", "hist_pct_out_of_range.png"),
    ("space_coverage_x", "X-axis space coverage, ALL samples (max-min span; x<0 -> NaN, no upper cap)", "% of screen width covered", "mediumseagreen", "hist_space_coverage_x.png"),
    ("space_coverage_y", "Y-axis space coverage, ALL samples (max-min span; y<0 -> NaN, no upper cap)", "% of screen height covered", "darkorange", "hist_space_coverage_y.png"),
    ("space_coverage_x_fixation_only", "X-axis space coverage, FIXATION SAMPLES ONLY (max-min span; x<0 -> NaN, no upper cap)", "% of screen width covered", "seagreen", "hist_space_coverage_x_fixation_only.png"),
    ("space_coverage_y_fixation_only", "Y-axis space coverage, FIXATION SAMPLES ONLY (max-min span; y<0 -> NaN, no upper cap)", "% of screen height covered", "chocolate", "hist_space_coverage_y_fixation_only.png"),
    ("pct_fixation", "% of panel samples classified as fixation", "% fixation samples", "cornflowerblue", "hist_pct_fixation.png"),
    ("pct_saccade", "% of panel samples classified as saccade", "% saccade samples", "salmon", "hist_pct_saccade.png"),
]


def _plot_metric_with_outliers(df, metric_col, title, xlabel, save_path, color, bins=50):
    sub = df.dropna(subset=[metric_col])
    values = sub[metric_col]
    mean = values.mean()
    sd = values.std()

    outliers = sub[sub[metric_col] > mean + 2 * sd].sort_values(metric_col, ascending=False)
    labels = [f"{r.participant}/{r.panel} ({getattr(r, metric_col):.2f})" for r in outliers.itertuples()]

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
                  (-1, ":", "goldenrod", None), (-2, ":", "orangered", None), (-3, ":", "darkred", None)]
    for k, lstyle, lcolor, lbl in line_specs:
        x = mean + k * sd
        if values.min() <= x <= values.max():
            ax.axvline(x, color=lcolor, linestyle=lstyle, linewidth=1)
            if lbl:
                ax.text(x, ax.get_ylim()[1] * 0.97, lbl, rotation=90, va="top", ha="right", fontsize=7, color=lcolor)

    ax.set_title(f"{title}\nmean={mean:.3f}, sd={sd:.3f}, n={len(values)}, n_above_mean+2sd={len(labels)}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (panels)")

    if labels:
        col_texts = ["\n".join(labels[i::n_cols]) for i in range(n_cols)]
        for i, col_text in enumerate(col_texts):
            ax_text.text(0.02 + i * (0.98 / n_cols), 0.95, col_text, transform=ax_text.transAxes,
                         fontsize=7, va="top", ha="left", family="monospace")
        ax_text.set_title(f"participant/panel above mean+2SD (n={len(labels)}):", fontsize=9, loc="left")
    else:
        ax_text.text(0.02, 0.5, "No panels above mean+2SD", fontsize=9, transform=ax_text.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"saved {save_path} (n_above_2sd={len(labels)})")


def make_all_plots(csv_path, out_dir, variant_label):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {variant_label}: {len(df)} panels ===")
    for metric_col, title, xlabel, color, filename in METRICS:
        _plot_metric_with_outliers(df, metric_col, f"{variant_label} - {title}", xlabel,
                                    os.path.join(out_dir, filename), color)


if __name__ == "__main__":
    make_all_plots(os.path.join(RAW_DIR, "space_coverage_data.csv"), RAW_DIR,
                    "NaN-ratio-only exclusion, Tobii_Sucks kept, best-calibrated eye")
    make_all_plots(os.path.join(EXCLUDED_DIR, "space_coverage_data.csv"), EXCLUDED_DIR,
                    "Regular exclusion (accuracy<2.5deg + NaN-ratio<10%, Tobii_Sucks excluded)")
    print("ALL DONE")

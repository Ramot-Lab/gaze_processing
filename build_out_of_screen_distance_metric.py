"""
Replaces the earlier per-axis max-min "space coverage" metric with a joint x/y distance-to-
screen metric, computed per raw sample:

  1. dx = distance outside [0,1] in x (0 if inside), dy = same for y
  2. distance = sqrt(dx^2 + dy^2) - Euclidean, so a corner excursion (both axes out) is
     scored higher than a single-axis excursion of the same per-axis magnitude
  3. despike: an excursion only counts if it persists >=3 consecutive samples (600Hz - a
     1-2 sample spike is tracker noise, not a real gaze shift beyond the screen)
  4. winsorize: capped at 2.0 (2x the normalized screen width) before aggregating, so one
     wildly corrupted sample can't dominate a panel's average

Per participant-panel:
  pct_out_of_bounds   - % of samples with (despiked) distance > 0 - frequency
  severity_median     - median distance among the out-of-bounds samples only
  severity_p90        - 90th percentile of the same - severity, tail-sensitive version
  mean_distance       - mean distance across ALL samples (in-bounds contribute 0) - the
                        main combined frequency x severity metric, this is what gets
                        histogrammed

NOTE: x and y each stay in their own axis's normalized [0,1] scale (matching every other
metric in this project) - since the screen isn't square (1920x1080), 0.1 of x-distance and
0.1 of y-distance aren't the same physical distance. This treats them as commensurate for
the purpose of one combined number; flag if you want it aspect-ratio-corrected instead.

Reads the raw per-sample tables already saved by build_raw_space_coverage_sweep.py - no
re-sweep of the .mat files needed.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = "/Volumes/ramot/Noam_M/calibration_qc/space_coverage_analysis"
RAW_DIR = os.path.join(BASE_DIR, "raw")
EXCLUDED_DIR = os.path.join(BASE_DIR, "excluded_by_acc_and_nan")

DESPIKE_MIN_RUN = 3
WINSOR_CAP = 2.0


def _distance_to_screen(x, y):
    dx = np.maximum(0.0, np.maximum(-x, x - 1.0))
    dy = np.maximum(0.0, np.maximum(-y, y - 1.0))
    return np.sqrt(dx ** 2 + dy ** 2)


def _despike(distance, min_run=DESPIKE_MIN_RUN):
    """Zeroes out runs of consecutive out-of-bounds samples shorter than min_run."""
    is_out = (distance > 0).astype(np.int8)
    if not is_out.any():
        return distance
    padded = np.concatenate(([0], is_out, [0]))
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]  # exclusive
    despiked = distance.copy()
    short = (ends - starts) < min_run
    for s, e in zip(starts[short], ends[short]):
        despiked[s:e] = 0.0
    return despiked


def _panel_distance_metrics(x, y):
    # Native NaN samples (blinks/tracking loss - see the blink discussion earlier this
    # session, these were never touched by this raw extraction) have no defined on/off-
    # screen distance - exclude them upfront rather than letting NaN silently propagate
    # through np.maximum/np.mean and poison the whole panel's average.
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan

    distance = _distance_to_screen(x, y)
    distance = np.minimum(distance, WINSOR_CAP)
    distance = _despike(distance)

    out_mask = distance > 0
    pct_out_of_bounds = 100.0 * np.mean(out_mask)
    if out_mask.any():
        severity_median = float(np.median(distance[out_mask]))
        severity_p90 = float(np.percentile(distance[out_mask], 90))
    else:
        severity_median = 0.0
        severity_p90 = 0.0
    mean_distance = float(np.mean(distance))
    return pct_out_of_bounds, severity_median, severity_p90, mean_distance


def compute_for_variant(raw_parquet_path):
    df = pd.read_parquet(raw_parquet_path, columns=["group", "participant", "panel", "eye", "x", "y"])
    rows = []
    for (group, participant, panel, eye), sub in df.groupby(["group", "participant", "panel", "eye"], sort=False):
        x = sub["x"].to_numpy(dtype=np.float64)
        y = sub["y"].to_numpy(dtype=np.float64)
        n_nan = int((np.isnan(x) | np.isnan(y)).sum())
        pct_oob, sev_med, sev_p90, mean_dist = _panel_distance_metrics(x, y)
        rows.append({"group": group, "participant": participant, "panel": panel, "eye": eye,
                     "n_samples": len(sub), "n_nan_excluded": n_nan, "pct_out_of_bounds": pct_oob,
                     "severity_median": sev_med, "severity_p90": sev_p90, "mean_distance": mean_dist})
    return pd.DataFrame(rows)


def _plot_metric_both_tails(df, metric_col, title, xlabel, save_path, color, bins=60):
    sub = df.dropna(subset=[metric_col])
    values = sub[metric_col]
    mean = values.mean()
    sd = values.std()
    lo, hi = mean - 2 * sd, mean + 2 * sd

    outliers = sub[(sub[metric_col] > hi) | (sub[metric_col] < lo)].copy()
    outliers["direction"] = np.where(outliers[metric_col] > hi, "above", "below")
    outliers = outliers.sort_values(metric_col, ascending=False)
    labels = [f"{r.participant}/{r.panel} ({getattr(r, metric_col):.4f}, {r.direction})" for r in outliers.itertuples()]

    n_cols = 3 if len(labels) > 15 else (2 if len(labels) > 6 else 1)
    n_rows = max(1, -(-len(labels) // n_cols)) if labels else 1
    text_height = max(0.7, 0.20 * n_rows + 0.4)

    fig = plt.figure(figsize=(12, 6.5 + text_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.5, text_height])
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    ax.hist(values, bins=bins, color=color, edgecolor="k", alpha=0.75)
    line_specs = [(0, "-", "black", "mean"),
                  (1, "--", "goldenrod", "+1 SD"), (2, "--", "orangered", "+2 SD"), (3, "--", "darkred", "+3 SD"),
                  (-1, "--", "goldenrod", "-1 SD"), (-2, "--", "orangered", "-2 SD"), (-3, "--", "darkred", "-3 SD")]
    for k, lstyle, lcolor, lbl in line_specs:
        x = mean + k * sd
        if values.min() <= x <= values.max():
            ax.axvline(x, color=lcolor, linestyle=lstyle, linewidth=1)
            ax.text(x, ax.get_ylim()[1] * 0.97, lbl, rotation=90, va="top", ha="right", fontsize=7, color=lcolor)

    ax.set_title(f"{title}\nmean={mean:.4f}, sd={sd:.4f}, n={len(values)}, n_outside_mean+-2sd={len(labels)}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (panels)")

    if labels:
        col_texts = ["\n".join(labels[i::n_cols]) for i in range(n_cols)]
        for i, col_text in enumerate(col_texts):
            ax_text.text(0.02 + i * (0.98 / n_cols), 0.95, col_text, transform=ax_text.transAxes,
                         fontsize=7, va="top", ha="left", family="monospace")
        ax_text.set_title(f"participant/panel outside mean+-2SD (n={len(labels)}):", fontsize=9, loc="left")
    else:
        ax_text.text(0.02, 0.5, "No panels outside mean+-2SD", fontsize=9, transform=ax_text.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"saved {save_path} (n_outliers={len(labels)})")
    return outliers


def run_variant(out_dir, variant_label):
    raw_parquet = os.path.join(out_dir, "raw_samples_with_fixation_labels.parquet")
    df = compute_for_variant(raw_parquet)
    csv_path = os.path.join(out_dir, "out_of_screen_distance_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"=== {variant_label}: {len(df)} panels, saved to {csv_path} ===")

    outliers = _plot_metric_both_tails(
        df, "mean_distance",
        f"{variant_label} - joint x/y out-of-screen distance (despiked, winsorized, mean per panel)",
        "mean distance outside [0,1]x[0,1] (normalized units)",
        os.path.join(out_dir, "hist_out_of_screen_distance.png"), "slateblue",
    )
    outliers.to_csv(os.path.join(out_dir, "out_of_screen_distance_outliers.csv"), index=False)


if __name__ == "__main__":
    run_variant(RAW_DIR, "NaN-ratio-only exclusion, Tobii_Sucks kept, best-calibrated eye")
    run_variant(EXCLUDED_DIR, "Regular exclusion (accuracy<2.5deg + NaN-ratio<10%, Tobii_Sucks excluded)")
    print("ALL DONE")

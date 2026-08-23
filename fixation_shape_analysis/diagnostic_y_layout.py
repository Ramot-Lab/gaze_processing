"""
One-off diagnostic: histogram of ALL pooled y-fixation values (across every included
participant-panel unit) with the panel's dictionary-layout landmarks overlaid as dashed
lines, so the empirical y-distribution can be checked by eye against the known layout:

  1. top border of the dictionary box
  2. center of the symbol row
  3. the divider line between the symbol row and the digit/number row
  4. center of the digit/number row
  5. bottom border of the digit/number row

Landmark ratios are imported directly from calibration_drift_qa.py (same measured pixel
constants used there) rather than re-derived, so this stays a single source of truth.

Run with `python -m fixation_shape_analysis.diagnostic_y_layout` any time
step0_fixation_data_included.csv changes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from calibration_drift_qa import (
    _DICTIONARY_TOP_RATIO, DICTIONARY_MIDLINE_RATIO, DICTIONARY_SECOND_ROW_MID_RATIO,
    _DICTIONARY_ROW2_BOTTOM_RATIO,
)

from . import config

_SYMBOL_ROW_CENTER_RATIO = (_DICTIONARY_TOP_RATIO + DICTIONARY_MIDLINE_RATIO) / 2

LAYOUT_LANDMARKS = [
    ("dictionary top border", _DICTIONARY_TOP_RATIO, "tab:red"),
    ("symbol row center", _SYMBOL_ROW_CENTER_RATIO, "tab:orange"),
    ("symbol/digit row divider", DICTIONARY_MIDLINE_RATIO, "black"),
    ("digit row center", DICTIONARY_SECOND_ROW_MID_RATIO, "tab:green"),
    ("digit row bottom border", _DICTIONARY_ROW2_BOTTOM_RATIO, "tab:purple"),
]


def plot_y_distribution_with_layout(fixation_df, save_path, bins=200):
    y = fixation_df["y"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(y, bins=bins, color="lightgray", edgecolor="white")

    for label, ratio, color in LAYOUT_LANDMARKS:
        ax.axvline(ratio, color=color, linestyle="--", linewidth=1.3, label=f"{label} ({ratio:.4f})")

    # A handful of calibration-drift outliers stretch the raw range way past the dictionary
    # itself (see the raw-data caveat in summary.md) - zoom the view to where the histogram
    # bars and landmarks actually are, without dropping any data from the histogram itself.
    landmark_ratios = [ratio for _, ratio, _ in LAYOUT_LANDMARKS]
    lo = min(landmark_ratios[0], y.quantile(0.005))
    hi = max(landmark_ratios[-1], y.quantile(0.995))
    pad = 0.1 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)

    ax.set_xlabel("y position (normalized, 0 = top of screen)")
    ax.set_ylabel("Count (fixations)")
    ax.set_title(f"Pooled y-fixation distribution vs. dictionary layout landmarks (n={len(fixation_df)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_diagnostic_y_layout():
    config.ensure_output_dirs()
    fixation_df = pd.read_csv(config.table_path("step0_fixation_data_included.csv"))
    save_path = config.group_plot_path("diagnostic_y_distribution_layout.png")
    plot_y_distribution_with_layout(fixation_df, save_path)
    print(f"Wrote {save_path}")
    return save_path


if __name__ == "__main__":
    run_diagnostic_y_layout()

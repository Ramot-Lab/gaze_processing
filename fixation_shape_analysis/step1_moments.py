"""
Step 1 - basic moments (mean, median, sd, skewness, excess kurtosis) per participant-panel
unit, computed separately for the x and y fixation coordinate.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from . import config


def compute_moments_for_axis(fixation_df, axis):
    rows = []
    for (participant_id, trial_id, unit_id), group in fixation_df.groupby(["participant_id", "trial_id", "unit_id"]):
        values = group[axis].to_numpy()
        rows.append({
            "participant_id": participant_id,
            "trial_id": trial_id,
            "unit_id": unit_id,
            "axis": axis,
            "n_fixations": len(values),
            "mean": values.mean(),
            "median": pd.Series(values).median(),
            "sd": values.std(ddof=1),
            "skewness": stats.skew(values),
            "kurtosis": stats.kurtosis(values, fisher=True),
        })
    return pd.DataFrame(rows)


def compute_moments(fixation_df, axes=config.AXES):
    return pd.concat([compute_moments_for_axis(fixation_df, axis) for axis in axes], ignore_index=True)


def plot_skew_kurtosis_scatter(moments_df, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"x": "tab:blue", "y": "tab:orange"}
    for axis, group in moments_df.groupby("axis"):
        ax.scatter(group["skewness"], group["kurtosis"], alpha=0.6, s=25,
                   label=f"{axis}-axis", color=colors.get(axis))
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Skewness")
    ax.set_ylabel("Excess kurtosis (Fisher)")
    ax.set_title("Skewness vs. kurtosis per participant-panel, by axis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_step1(fixation_df, axes=config.AXES):
    config.ensure_output_dirs()

    moments_df = compute_moments(fixation_df, axes=axes)
    moments_df.to_csv(config.table_path("step1_movment_distribution_analysis.csv"), index=False)

    plot_skew_kurtosis_scatter(moments_df, config.group_plot_path("step1_skew_kurtosis_scatter.png"))

    print(f"Step 1: computed moments for {len(moments_df)} participant-panel/axis rows.")
    return moments_df


if __name__ == "__main__":
    from .step0_load_data import run_step0
    included_df, _ = run_step0()
    run_step1(included_df)

"""
fxsp panel-1 fixation plots, from the tables saccade_latency_analysis.main writes
(participant_latency_summary.csv, fxsp_fixation_bcea_per_evt_all_partici.csv):

  1. fixation_sample_percentage_hist.png - over all participants, % of panel-1 samples
     gazeNET labeled fixation (evt == FIXATION_IDX), out of all panel-1 samples
     (interpolated ones included - they are labeled too).
  2. per_participant_fixation_samples/<participant>.png - per participant, over the
     fixations that went into BCEA: histogram of n_samples_in_bcea (top) and of
     n_interpolated_samples_in_fixation (bottom). Two panels of one figure, since the
     counts are on very different scales (hundreds vs. 0-5 samples).
  3. sd_n_samples_in_bcea_hist.png - over all participants, the SD (across that
     participant's BCEA fixations) of n_samples_in_bcea.
  4. interpolated_pct_in_bcea_fixations_hist.png - over all participants, the % of
     interpolated samples among all samples of that participant's BCEA fixations
     (pooled: sum of interpolated / sum of samples).

Usage:
  python plot_fxsp_fixation_fraction.py --dir <folder with the two CSVs>
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_pipeline import demographics_text, annotate_demographics
from saccade_latency_analysis import SAMPLING_RATE_HZ

BAR_COLOR = "#2a78d6"      # dataviz reference palette, categorical slot 1
BAR_COLOR_2 = "#eb6834"    # slot 2
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e5e5e2"

SUMMARY_CSV = "participant_latency_summary.csv"
BCEA_CSV = "fxsp_fixation_bcea_per_evt_all_partici.csv"


def load_gender():
    from exclusion_policy import DEFAULT_TOBII_SUCKS_XLSX
    try:
        demo = pd.read_excel(DEFAULT_TOBII_SUCKS_XLSX)[["Patient_ID", "Gender"]]
        return demo.rename(columns={"Patient_ID": "participant"}).drop_duplicates("participant")
    except Exception as e:
        print(f"Gender lookup failed ({e}) - sex will show as unknown")
        return pd.DataFrame(columns=["participant", "Gender"])


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _demographics_upper_right(text, ax):
    """feature_pipeline.annotate_demographics only places boxes on the left, where the
    median label of a right-skewed histogram sits."""
    ax.text(0.99, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray"))


def _median_line(ax, value, label):
    ax.axvline(value, color=TEXT_SECONDARY, linestyle="--", linewidth=1.5)
    ax.text(value, ax.get_ylim()[1] * 0.97, f"  {label}", color=TEXT_SECONDARY, va="top", ha="left", fontsize=9)


def plot_fixation_fraction_hist(df_summary, out_path):
    pct = df_summary["fixation_sample_fraction"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pct, bins=np.arange(np.floor(pct.min()), np.ceil(pct.max()) + 1, 1),
            color=BAR_COLOR, edgecolor="white", linewidth=2)
    _median_line(ax, np.median(pct), f"median {np.median(pct):.1f}%")
    ax.set_xlabel("Fixation samples (% of all fxsp panel-1 samples)")
    ax.set_ylabel("Participants")
    ax.set_title("fxsp panel 1: share of samples labeled fixation\n(gazeNET, 600 Hz, no inference noise, adjacent fixations merged)")
    _style(ax)
    annotate_demographics(demographics_text(df_summary), ax=ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}  (N={len(pct)}, median {np.median(pct):.1f}%)")


def plot_participant_fixation_samples(df_fix, participant_row, out_path):
    """df_fix: this participant's BCEA fixations; participant_row: 1-row frame with
    group/Gender for the demographics box."""
    n = df_fix["n_samples_in_bcea"].to_numpy()
    k = df_fix["n_interpolated_samples_in_fixation"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    ax1.hist(n, bins=30, color=BAR_COLOR, edgecolor="white", linewidth=2)
    _median_line(ax1, np.median(n), f"median {np.median(n):.0f} samples ({np.median(n) * 1000 / SAMPLING_RATE_HZ:.0f} ms)")
    ax1.set_xlabel("Samples in the fixation (all used in BCEA)")
    ax1.set_ylabel("Fixations")
    _style(ax1)

    step = max(1, int(np.ceil((k.max() + 1) / 40)))  # <= ~40 bins, integer-aligned
    ax2.hist(k, bins=np.arange(-0.5, k.max() + step + 0.5, step), color=BAR_COLOR_2,
             edgecolor="white", linewidth=1 if step == 1 and k.max() > 40 else 2)
    ax2.set_xlabel("Interpolated samples in the fixation" + (f" ({step}-sample bins)" if step > 1 else ""))
    ax2.set_yscale("log")  # the 0 bin dominates; log shows the rare high counts
    ax2.set_ylabel("Fixations (log scale)")
    _style(ax2)
    ax2.text(0.99, 0.95, f"{(k == 0).mean() * 100:.0f}% of fixations have none\n"
                         f"{k.sum() / n.sum() * 100:.1f}% of BCEA samples interpolated",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9, color=TEXT_SECONDARY)

    p = participant_row["participant"].iloc[0]
    fig.suptitle(f"{p} - fxsp panel 1 fixations used in BCEA (n = {len(n)} fixations)")
    _demographics_upper_right(demographics_text(participant_row), ax1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sd_n_samples_hist(df_fix, df_people, out_path):
    sd = df_fix.groupby("participant")["n_samples_in_bcea"].std(ddof=1).dropna()
    people = df_people[df_people["participant"].isin(sd.index)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sd.to_numpy(), bins=25, color=BAR_COLOR, edgecolor="white", linewidth=2)
    _median_line(ax, sd.median(), f"median {sd.median():.0f} samples ({sd.median() * 1000 / SAMPLING_RATE_HZ:.0f} ms)")
    ax.set_xlabel("SD of samples per BCEA fixation, within participant")
    ax.set_ylabel("Participants")
    ax.set_title("fxsp panel 1: variability of fixation length (samples used in BCEA)")
    _style(ax)
    _demographics_upper_right(demographics_text(people), ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}  (N={len(sd)})")


def plot_interpolated_pct_hist(df_fix, df_people, out_path):
    sums = df_fix.groupby("participant")[["n_interpolated_samples_in_fixation", "n_samples_in_bcea"]].sum()
    pct = sums["n_interpolated_samples_in_fixation"] / sums["n_samples_in_bcea"] * 100
    people = df_people[df_people["participant"].isin(pct.index)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pct.to_numpy(), bins=25, color=BAR_COLOR, edgecolor="white", linewidth=2)
    _median_line(ax, pct.median(), f"median {pct.median():.2f}%")
    ax.set_xlabel("Interpolated samples (% of all samples in the participant's BCEA fixations)")
    ax.set_ylabel("Participants")
    ax.set_title("fxsp panel 1: interpolated share of the data used in BCEA\n"
                 "(fixations with > 5 consecutive or >= 15% interpolated samples excluded)")
    _style(ax)
    _demographics_upper_right(demographics_text(people), ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}  (N={len(pct)}, median {pct.median():.2f}%, max {pct.max():.2f}%)")


def main(folder):
    gender = load_gender()
    df_summary = pd.read_csv(os.path.join(folder, SUMMARY_CSV)).merge(gender, on="participant", how="left")
    df_fix = pd.read_csv(os.path.join(folder, BCEA_CSV))
    people = df_summary[["participant", "group", "Gender"]]

    plot_fixation_fraction_hist(df_summary, os.path.join(folder, "fixation_sample_percentage_hist.png"))
    plot_sd_n_samples_hist(df_fix, people, os.path.join(folder, "sd_n_samples_in_bcea_hist.png"))
    plot_interpolated_pct_hist(df_fix, people, os.path.join(folder, "interpolated_pct_in_bcea_fixations_hist.png"))

    per_dir = os.path.join(folder, "per_participant_fixation_samples")
    os.makedirs(per_dir, exist_ok=True)
    for p, grp in df_fix.groupby("participant"):
        plot_participant_fixation_samples(grp, people[people["participant"] == p], os.path.join(per_dir, f"{p}.png"))
    print(f"Saved {df_fix['participant'].nunique()} per-participant plots -> {per_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="folder holding the two CSVs; plots are written there")
    main(parser.parse_args().dir)

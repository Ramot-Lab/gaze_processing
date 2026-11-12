"""
Standalone diagnostic (decision 2026-11-04): for every feature that has both a mean and
an std column in the already-saved panel/participant feature tables, plots mean vs. std
as a scatterplot colored by score - one at panel level (every participant x panel row)
and one at participant level (one row per participant). Reads the existing
panel_features.csv/participant_features.csv directly; does not recompute any feature.

Not wired into feature_pipeline.py's main() - a one-off diagnostic folder, run manually:
    python3 build_mean_vs_std_scatterplots.py
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from feature_pipeline import demographics_text, EXCLUDED_PARTICIPANTS



def find_mean_std_pairs(df):
    """Every (base_name, mean_col, std_col) triple present in df. Handles the one
    irregularly-named pair (mean_typicality/typicality_std - prefix instead of suffix)
    as a special case."""
    cols = set(df.columns)
    pairs = []
    for col in df.columns:
        if col.endswith("_mean"):
            base = col[: -len("_mean")]
            std_col = base + "_std"
            if std_col in cols:
                pairs.append((base, col, std_col))
    if "mean_typicality" in cols and "typicality_std" in cols:
        pairs.append(("typicality", "mean_typicality", "typicality_std"))
    return pairs


def panel_demographics_text(df_panels_subset):
    """demographics_text() counts rows, which at panel level would count panels, not
    participants, for the HC/pwMS/F/M split - misleading (one participant contributes
    ~6 rows). Report N panels separately, then the real participant-level split."""
    n_panels = len(df_panels_subset)
    unique_participants = df_panels_subset.drop_duplicates("participant")
    return f"N panels={n_panels}\n{demographics_text(unique_participants)}"


def _stars(p_val):
    return "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."


def annotate_demographics_below(text, ax):
    """Places the N/HC-pwMS/sex demographics box below the axes, outside the plotted
    data area entirely (decision 2026-11-04) - unlike feature_pipeline's own
    annotate_demographics, which draws inside the axes corner and can sit on top of
    data points. clip_on=False lets it render outside the axes box; bbox_inches="tight"
    on savefig then expands the saved image to include it."""
    if not text:
        return
    ax.text(0.5, -0.16, text, transform=ax.transAxes, fontsize=9, ha="center", va="top",
             clip_on=False, bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))


def plot_mean_vs_std(df, mean_col, std_col, score_col, title, out_path, demographics):
    valid = df[[mean_col, std_col]].dropna()
    if len(valid) < 3:
        print(f"Skipping {title}: only {len(valid)} rows with both mean and std.")
        return

    r_val, p_val = pearsonr(valid[mean_col], valid[std_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    has_score = score_col in df.columns and df[score_col].notna().any()
    if has_score:
        scatter = ax.scatter(df[mean_col], df[std_col], c=df[score_col], cmap="viridis",
                              s=60, edgecolors="black", alpha=0.8)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(score_col, rotation=270, labelpad=15)
    else:
        ax.scatter(df[mean_col], df[std_col], color="teal", s=60, edgecolors="black", alpha=0.8)

    if valid[mean_col].std() > 0:
        z = np.polyfit(valid[mean_col], valid[std_col], 1)
        p_line = np.poly1d(z)
        x_vals = np.linspace(valid[mean_col].min(), valid[mean_col].max(), 100)
        ax.plot(x_vals, p_line(x_vals), "k--", alpha=0.6, linewidth=1.5)

    ax.set_title(f"{title}\nr = {r_val:.3f} ({_stars(p_val)}), p = {p_val:.3g}", pad=15)
    ax.set_xlabel(mean_col)
    ax.set_ylabel(std_col)
    ax.grid(True, alpha=0.3)
    annotate_demographics_below(demographics, ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path} (N={len(valid)}, r={r_val:.3f}, p={p_val:.3g})")


def main(run_dir=None, excluded=EXCLUDED_PARTICIPANTS):
    # Default (decision 2026-11-11): the newest consolidated run, not a hard-coded date.
    if run_dir is None:
        import pipeline_config as pc
        run_dir = pc.latest_feature_tables_dir()
    df_panels = pd.read_csv(os.path.join(run_dir, "panel_features.csv"))
    df_participants = pd.read_csv(os.path.join(run_dir, "participant_features.csv"))

    df_panels = df_panels[~df_panels["participant"].isin(excluded)].copy()
    df_participants = df_participants[~df_participants["participant"].isin(excluded)].copy()

    # panel_features.csv has no group/Gender columns of its own - bring them in from the
    # participant table just so the demographics box can report a real HC/pwMS/F/M split
    # (not needed for the plotted values themselves).
    df_panels = df_panels.merge(df_participants[["participant", "group", "Gender"]],
                                 on="participant", how="left")

    out_root = os.path.join(run_dir, "mean_vs_std_scatterplots")
    panel_dir = os.path.join(out_root, "panel_level")
    participant_dir = os.path.join(out_root, "participant_level")
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(participant_dir, exist_ok=True)

    panel_demo = panel_demographics_text(df_panels)
    participant_demo = demographics_text(df_participants)

    panel_pairs = find_mean_std_pairs(df_panels)
    print(f"Panel-level mean/std pairs ({len(panel_pairs)}): {[p[0] for p in panel_pairs]}")
    for base, mean_col, std_col in panel_pairs:
        plot_mean_vs_std(
            df_panels, mean_col, std_col, "score",
            f"{base}: mean vs std - Panel Level (N panels={len(df_panels)})",
            os.path.join(panel_dir, f"{base}_mean_vs_std.png"),
            demographics=panel_demo,
        )

    participant_pairs = find_mean_std_pairs(df_participants)
    print(f"\nParticipant-level mean/std pairs ({len(participant_pairs)}): {[p[0] for p in participant_pairs]}")
    for base, mean_col, std_col in participant_pairs:
        plot_mean_vs_std(
            df_participants, mean_col, std_col, "score",
            f"{base}: mean vs std - Participant Level (N={len(df_participants)})",
            os.path.join(participant_dir, f"{base}_mean_vs_std.png"),
            demographics=participant_demo,
        )

    print(f"\nDone. Output under: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None,
                         help="Feature-analysis run folder containing panel_features.csv/participant_features.csv "
                              "(default: the newest consolidated run)")
    args = parser.parse_args()
    main(run_dir=args.run_dir)

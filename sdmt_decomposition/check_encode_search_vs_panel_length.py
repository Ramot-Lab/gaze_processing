"""
Do the existing trial-level encode + search times exceed the real panel length?
(Noam, 2026-11-17)

Per participant x panel (all 683 panels in the 12_11_26 panel_features.csv):
  sum_encode_s = sum of time_in_relevant_fixations_ms over the panel's trials (existing column)
  sum_search_s = sum of trial_in_search_durations_ms over the panel's trials (existing column)
  panel_s      = real panel length = last - first t of the saved annotated CSV. This equals
                 the raw Tobii panel length exactly (verified on 5 panels in
                 test_sample_level_time_budget.py: same sample count, 0 ms difference).
Percentages are of panel_s, not of a nominal 90 s.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

# The share is sometimes mounted as the Noam_M folder itself (/Volumes/Noam_M).
_DIRECT_MOUNT = "/Volumes/Noam_M"
if os.path.isdir(os.path.join(_DIRECT_MOUNT, "Results", "Behavior", "processing_results")):
    pipeline_config.noam_m_root = lambda: _DIRECT_MOUNT

SOURCE_DATE = "12_11_26"
ANNOT_DATE = "11_11_26"
METHOD = "model_based"
ROOT = os.path.join(pipeline_config.noam_m_root(), SOURCE_DATE)
CSV_DIR = os.path.join(ROOT, "analysis_results", "features", "csvs")
OUT_DIR = os.path.join(ROOT, "analysis_results", "sdmt_decomposition_16_11_26", "check_encode_search_vs_panel_length")


def panel_length_s(group, pid, panel):
    path = pipeline_config.annotated_csv_path(METHOD, group, pid, panel, date_str=ANNOT_DATE)
    t = pd.read_csv(path, usecols=["t"])["t"].to_numpy(dtype=float)
    return (t.max() - t.min()) / 1e6, len(t)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trials = pd.read_csv(os.path.join(CSV_DIR, "trial_features.csv"), dtype={"panel": str})
    groups = pd.read_csv(os.path.join(CSV_DIR, "participant_features.csv"))[["participant", "group"]]
    scores = pd.read_csv(os.path.join(ROOT, "behavior_scores", "all_scores.csv"), dtype={"Panel": str}).rename(
        columns={"Participant": "participant", "Panel": "panel", "SDMT_Score": "S_panel"})

    df = (trials.groupby(["participant", "panel"])
          .agg(n_trials=("trial_idx", "size"),
               sum_encode_s=("time_in_relevant_fixations_ms", lambda x: x.sum() / 1000),
               sum_search_s=("trial_in_search_durations_ms", lambda x: x.sum() / 1000))
          .reset_index())
    df = df.merge(groups, on="participant", how="left", validate="m:1")
    df = df.merge(scores[["participant", "panel", "S_panel"]], on=["participant", "panel"], how="left", validate="1:1")

    lengths = []
    for r in df.itertuples():
        folder_group = "pwMS" if r.group == "MS" else r.group
        lengths.append(panel_length_s(folder_group, r.participant, r.panel))
    df["panel_s"], df["n_samples"] = zip(*lengths)

    df["encode_pct"] = 100 * df["sum_encode_s"] / df["panel_s"]
    df["search_pct"] = 100 * df["sum_search_s"] / df["panel_s"]
    df["encode_plus_search_pct"] = df["encode_pct"] + df["search_pct"]
    df["exceeds_panel"] = df["encode_plus_search_pct"] > 100
    df.to_csv(os.path.join(OUT_DIR, "encode_search_vs_panel_length.csv"), index=False)

    print(f"panels: {len(df)}; panel length s: min {df.panel_s.min():.2f}, median {df.panel_s.median():.2f}, "
          f"max {df.panel_s.max():.2f}")
    print(f"encode+search > panel length: {df.exceeds_panel.sum()}/{len(df)} panels "
          f"({100 * df.exceeds_panel.mean():.1f}%), {df[df.exceeds_panel].participant.nunique()} participants")
    print(df.groupby("group")[["encode_pct", "search_pct", "encode_plus_search_pct"]].describe(
        percentiles=[.25, .5, .75]).T.round(1).to_string())
    print(df.groupby("group")["exceeds_panel"].agg(["sum", "size"]).to_string())
    print("out:", OUT_DIR)


if __name__ == "__main__":
    main()

"""
Diagnostic scan (decision 2026-09-27, prompted by the ER635/l3 anomaly found in
duration_analysis.py): flags any participant/panel where the raw (x, y) gaze position
stays EXACTLY constant for more than a threshold duration - a frozen/stuck tracker
signal, not a real fixation (a genuine fixation still has microsaccadic jitter; exact
repeated floats for seconds at a time means the tracker stopped updating).

This is a SCAN ONLY for now - reports what's out there so the scope of the problem
(hopefully just ER635/l3) is known before deciding how to wire it into
run_preprocessing.py as a real exclusion criterion. Nothing is excluded yet.

Usage: python check_frozen_gaze.py [threshold_based|model_based]
"""
import os

import pandas as pd

import pipeline_config
from run_preprocessing import iter_all_subject_dirs
from signal_quality_checks import (FREEZE_THRESHOLD_MS, TIMESTAMP_GAP_THRESHOLD_MS,
                                    find_frozen_blocks, find_timestamp_gaps)

PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]


def run_timestamp_gap_scan(annotation_method="model_based", threshold_ms=TIMESTAMP_GAP_THRESHOLD_MS):
    """Only ever looks at participant/panels that PASSED Stage 1 (load_annotated_csv
    succeeds) - anyone Stage 1 already excluded is out of scope here by construction,
    same as find_frozen_blocks' scan."""
    flagged_rows = []
    n_panels_checked = 0
    for group, subject_dir in iter_all_subject_dirs(pipeline_config.main_data_path()):
        participant = os.path.basename(subject_dir)
        for panel in PANELS:
            try:
                df = pipeline_config.load_annotated_csv(annotation_method, group, participant, panel, corrected=False)
            except FileNotFoundError:
                continue
            n_panels_checked += 1
            gaps = find_timestamp_gaps(df, threshold_ms)
            for _, row in gaps.iterrows():
                flagged_rows.append({
                    "participant": participant, "group": group, "panel": panel,
                    "t_before": row["t_before"], "t_after": row["t_after"], "gap_ms": row["gap_ms"],
                })

    print(f"Checked {n_panels_checked} panels that passed Stage 1 preprocessing.")
    df_flagged = pd.DataFrame(flagged_rows)
    if df_flagged.empty:
        print(f"No timestamp gaps > {threshold_ms:.0f} ms found among participants who passed preprocessing.")
        return df_flagged

    print(f"Flagged {len(df_flagged)} gap(s) across {df_flagged['participant'].nunique()} "
          f"participant(s) that otherwise passed preprocessing:")
    print(df_flagged.sort_values("gap_ms", ascending=False).to_string(index=False))

    out_dir = pipeline_config.feature_analysis_base_dir(annotation_method)
    out_path = os.path.join(out_dir, "timestamp_gap_scan.csv")
    os.makedirs(out_dir, exist_ok=True)
    df_flagged.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df_flagged


def run(annotation_method="model_based", threshold_ms=FREEZE_THRESHOLD_MS):
    flagged_rows = []
    n_panels_checked = 0
    for group, subject_dir in iter_all_subject_dirs(pipeline_config.main_data_path()):
        participant = os.path.basename(subject_dir)
        for panel in PANELS:
            try:
                df = pipeline_config.load_annotated_csv(annotation_method, group, participant, panel, corrected=False)
            except FileNotFoundError:
                continue
            n_panels_checked += 1
            blocks = find_frozen_blocks(df, threshold_ms)
            for _, row in blocks.iterrows():
                flagged_rows.append({
                    "participant": participant, "group": group, "panel": panel,
                    "t_min": row["t_min"], "t_max": row["t_max"], "duration_ms": row["duration_ms"],
                    "x": row["x"], "y": row["y"], "n_samples": row["n_samples"],
                })

    print(f"Checked {n_panels_checked} panels across the full population.")
    df_flagged = pd.DataFrame(flagged_rows)
    if df_flagged.empty:
        print("No frozen-gaze blocks found above threshold.")
        return df_flagged

    print(f"Flagged {len(df_flagged)} block(s) across {df_flagged['participant'].nunique()} participant(s):")
    print(df_flagged.sort_values("duration_ms", ascending=False).to_string(index=False))

    out_dir = pipeline_config.feature_analysis_base_dir(annotation_method)
    out_path = os.path.join(out_dir, "frozen_gaze_scan.csv")
    os.makedirs(out_dir, exist_ok=True)
    df_flagged.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df_flagged


if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "model_based"
    if method not in pipeline_config.ANNOTATION_METHODS:
        print(f"usage: python check_frozen_gaze.py [{'|'.join(pipeline_config.ANNOTATION_METHODS)}]")
        sys.exit(1)
    run(method)

"""
Precomputes and saves the whole-dictionary-drift-corrected CSV (decision 2026-09-22)
next to every raw annotated CSV Stage 1 has already saved, for one annotation method -
so Stage 2/3 callers requesting pipeline_config.load_annotated_csv(..., corrected=True)
get the saved file instead of paying the correction's cost on every read.

Leaves Stage 1 (run_preprocessing.py) itself untouched, per instruction - this is a
separate, later step, not part of preprocessing.

Usage: python build_corrected_annotated_csvs.py <threshold_based|model_based> [date_str]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from calibration_drift_qa import correct_annotated_csv_whole_dictionary
from pipeline_config import ANNOTATION_METHODS, annotated_csv_path, annotated_gaze_dir


def run(method, date_str=None):
    stage1_dir = annotated_gaze_dir(method, date_str)
    raw_paths = sorted(glob.glob(os.path.join(stage1_dir, "*", "*", "panel_*_annotated_gaze_data.csv")))
    print(f"found {len(raw_paths)} raw annotated CSVs under {stage1_dir}")

    n_written, n_skipped, n_failed = 0, 0, 0
    for raw_path in raw_paths:
        # <stage1_dir>/<group>/<participant>/panel_<panel>_annotated_gaze_data.csv
        participant_dir = os.path.dirname(raw_path)
        fname = os.path.basename(raw_path)
        panel = fname[len("panel_"):-len("_annotated_gaze_data.csv")]
        group = os.path.basename(os.path.dirname(participant_dir))
        participant = os.path.basename(participant_dir)

        corrected_path = annotated_csv_path(method, group, participant, panel, date_str, corrected=True)
        if os.path.exists(corrected_path):
            n_skipped += 1
            continue

        try:
            raw = pd.read_csv(raw_path)
            corrected, dy, n_fixations = correct_annotated_csv_whole_dictionary(raw)
            corrected.to_csv(corrected_path, index=False)
            n_written += 1
            print(f"  {participant}/{panel}: dy={dy:.5f} (from {n_fixations} fixations) -> {os.path.basename(corrected_path)}")
        except Exception as e:
            n_failed += 1
            print(f"  FAILED {participant}/{panel}: {type(e).__name__}: {e}")

    print(f"\ndone. method={method} written={n_written} already_existed={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ANNOTATION_METHODS:
        print(f"usage: python build_corrected_annotated_csvs.py [{'|'.join(ANNOTATION_METHODS)}] [date_str]")
        sys.exit(1)
    method_arg = sys.argv[1]
    date_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run(method_arg, date_arg)

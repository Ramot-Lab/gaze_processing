"""
Sweeps every valid participant under the CURRENT preprocessing pipeline (per-panel
accuracy+NaN-ratio eye selection; out-of-range values NaN'd per OUT_OF_RANGE_X_BOUNDS/
OUT_OF_RANGE_Y_BOUNDS via OUT_OF_RANGE_VALUES_METHOD="only_extreme_values" - see
constants.py) and reports every exclusion, always at PANEL granularity (one row per
excluded panel, with the actual short code e.g. "i1"/"l4" - never the presentation-order
index), at whatever level it was decided:
  - "participant": the whole participant failed to load (e.g. bad path) - no panel identity
                    available, panel/accuracy/NaN columns blank
  - "event":  every panel of one calibration event failed (accuracy, or every panel's NaN
              ratio) - expanded from ParticipantGazeDataManager.load_errors'
              panel_details, one row per panel of that event
  - "infra":  a duplicate/partial save of another event, or a corrupted .mat file - no
              panel-level accuracy/NaN data available (the file didn't fully parse, or was
              skipped before that point)
  - "panel":  one panel was dropped even though its event otherwise loaded fine
              (ParticipantGazeDataManager.panel_load_errors)

Columns: group, participant, panel, scope, recording_date, reason, l_acc, r_acc,
l_nan_pct, r_nan_pct, other_errors (secondary criterion also failing - see
participant_gaze_data_manager._other_errors_note).
"""
import os

import pandas as pd

from calibration_drift_qa import _iter_subject_dirs
from participant_gaze_data_manager import ParticipantGazeDataManager

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc"

BLANK_METRICS = {"l_acc": None, "r_acc": None, "l_nan_pct": None, "r_nan_pct": None, "other_errors": ""}


def main():
    rows = []
    n_participants_loaded = 0
    n_panels_loaded = 0

    for group, subject_dir in _iter_subject_dirs(MAIN_DATA_PATH, ("HC", "pwMS")):
        if "SDMT" not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)
        try:
            sd = ParticipantGazeDataManager(subject_dir, MAIN_DATA_PATH, "SDMT", group)
        except Exception as e:
            rows.append({"group": group, "participant": participant, "panel": None, "scope": "participant",
                         "recording_date": None, "reason": str(e), **BLANK_METRICS})
            continue

        n_participants_loaded += 1
        n_panels_loaded += len(sd.matched_data)

        for err in sd.load_errors:
            panel_details = err.get("panel_details")
            if panel_details:
                for pd_ in panel_details:
                    rows.append({
                        "group": group, "participant": participant, "panel": pd_["panel"], "scope": "event",
                        "recording_date": err.get("recording_date"), "reason": pd_["error"],
                        "l_acc": pd_["l_acc"], "r_acc": pd_["r_acc"],
                        "l_nan_pct": pd_["l_nan_pct"], "r_nan_pct": pd_["r_nan_pct"],
                        "other_errors": pd_["other_errors"],
                    })
            else:
                # Duplicate-file / corrupted-file infra issue - no panel identity or
                # accuracy/NaN data available (the file never got that far).
                rows.append({"group": group, "participant": participant, "panel": None, "scope": "infra",
                             "recording_date": err.get("recording_date") or err.get("file"),
                             "reason": err["error"], **BLANK_METRICS})

        for err in sd.panel_load_errors:
            rows.append({
                "group": group, "participant": participant, "panel": err.get("panel"), "scope": "panel",
                "recording_date": err.get("recording_date"), "reason": err["error"],
                "l_acc": err.get("l_acc"), "r_acc": err.get("r_acc"),
                "l_nan_pct": err.get("l_nan_pct"), "r_nan_pct": err.get("r_nan_pct"),
                "other_errors": err.get("other_errors", ""),
            })

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "exclusion_report.csv")
    df.to_csv(save_path, index=False)

    print(f"{n_participants_loaded} participants loaded, {n_panels_loaded} panels loaded successfully")
    print(f"{len(df)} exclusion rows saved to {save_path}")
    if len(df):
        print(df["scope"].value_counts().to_string())
        print()
        print(df.to_string())


if __name__ == "__main__":
    main()

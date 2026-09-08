"""
Sweeps every valid participant under the CURRENT preprocessing pipeline (per-panel
accuracy+NaN-ratio eye selection; out-of-range values NaN'd per OUT_OF_RANGE_X_BOUNDS/
OUT_OF_RANGE_Y_BOUNDS via OUT_OF_RANGE_VALUES_METHOD="only_extreme_values" - see
constants.py) and reports every exclusion, at whatever granularity it happened:
  - "participant": the whole participant failed to load (e.g. bad path)
  - "event": one calibration event (mat file, up to 3 panels) failed entirely
             (ParticipantGazeDataManager.load_errors)
  - "panel":  one panel was dropped even though its event otherwise loaded fine
             (ParticipantGazeDataManager.panel_load_errors)

Note: a "panel" scope row's reason refers to the panel by its presentation-order index
(panel 1/2/3 within that event), not its short code (e.g. "l3") - that mapping only happens
later in group_task_info() and isn't reconstructed here.
"""
import os

import pandas as pd

from calibration_drift_qa import _iter_subject_dirs
from participant_gaze_data_manager import ParticipantGazeDataManager

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc"


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
            rows.append({"group": group, "participant": participant, "scope": "participant",
                         "recording_date": None, "reason": str(e)})
            continue

        n_participants_loaded += 1
        n_panels_loaded += len(sd.matched_data)

        for err in sd.load_errors:
            rows.append({"group": group, "participant": participant, "scope": "event",
                         "recording_date": err.get("recording_date") or err.get("file"),
                         "reason": err["error"]})
        for err in sd.panel_load_errors:
            rows.append({"group": group, "participant": participant, "scope": "panel",
                         "recording_date": err.get("recording_date"), "reason": err["error"]})

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

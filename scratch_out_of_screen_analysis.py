"""
One-off diagnostic (not yet part of the permanent pipeline): for every calibration event's
raw gaze stream, BEFORE any cleaning, how many samples are outside the normalized [0,1]
screen (x<0, x>1, y<0, y>1)? Informs the final decision on decision 2 (negative->NaN) and
whether the same treatment should extend to >1 values.
"""
import glob
import os

import numpy as np
import pandas as pd

from calibration_drift_qa import _iter_subject_dirs
from participant_gaze_data_manager import ParticipantGazeDataManager

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
OUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc"


def _panel_presentation_indices(messages, gaze_timestamps, dummy):
    panel_indices, break_indices = dummy.break_mat_into_pannels(messages)
    if len(panel_indices) != 3 or len(break_indices) != 3:
        return None
    panel_start_times = [messages[i][0] for i in panel_indices]
    break_start_times = [messages[i][0] for i in break_indices]
    indices_by_panel = []
    for i in range(3):
        idx = np.where((gaze_timestamps > panel_start_times[i]) & (gaze_timestamps < break_start_times[i]))[0]
        indices_by_panel.append(idx)
    return indices_by_panel


def main():
    rows = []
    x_out_of_range_values = []  # (value, participant, panel, eye)
    y_out_of_range_values = []

    dummy = ParticipantGazeDataManager.__new__(ParticipantGazeDataManager)
    dummy.task_validation_filter_param = "run"
    dummy.task = "SDMT"

    for group, subject_dir in _iter_subject_dirs(MAIN_DATA_PATH, ("HC", "pwMS")):
        participant = os.path.basename(subject_dir)
        task_dir = os.path.join(subject_dir, "SDMT")
        if not os.path.isdir(task_dir):
            continue
        mat_files = glob.glob(os.path.join(task_dir, "*.mat"))
        mat_files = [f for f in mat_files if "run" in os.path.split(f)[1].lower()]
        try:
            loaded_mats, _ = dummy._load_and_dedupe_run_files(mat_files)
        except Exception as e:
            print(f"  FAILED dedup {participant}: {e}")
            continue

        for mat in loaded_mats:
            try:
                messages = mat["messages"]
                gaze_ts = mat["data"].gaze.systemTimeStamp
                if np.size(gaze_ts) == 0:
                    continue
                indices_by_panel = _panel_presentation_indices(messages, gaze_ts, dummy)
                if indices_by_panel is None:
                    continue
                panel_names = ["a", "b", "c"]  # placeholder; real panel code resolved below via task_data
                # resolve real panel codes the same way group_task_info does
                task_data = mat["task_data"].__dict__
                codes = []
                for i, task_name in enumerate(list(task_data.keys())[1::2]):
                    codes.append((task_name[-2:]).replace("_", "").lower())

                for eye_label, gaze_attr in (("l", "left"), ("r", "right")):
                    gaze = getattr(mat["data"].gaze, gaze_attr).gazePoint.onDisplayArea
                    for i in range(min(3, len(codes))):
                        indices = indices_by_panel[i]
                        if len(indices) == 0:
                            continue
                        panel_code = codes[i]
                        x = gaze[0, indices]
                        y = gaze[1, indices]
                        n = len(x)
                        x_below0 = x < 0
                        x_above1 = x > 1
                        y_below0 = y < 0
                        y_above1 = y > 1
                        bad = x_below0 | x_above1 | y_below0 | y_above1
                        rows.append({
                            "group": group, "participant": participant, "panel": panel_code, "eye": eye_label,
                            "n_samples": n,
                            "n_x_below_0": int(x_below0.sum()), "n_x_above_1": int(x_above1.sum()),
                            "n_y_below_0": int(y_below0.sum()), "n_y_above_1": int(y_above1.sum()),
                            "n_out_of_screen": int(bad.sum()),
                            "pct_out_of_screen": 100.0 * bad.sum() / n if n else np.nan,
                        })
                        # capture the PAIRED (x,y) for every bad sample (not just the
                        # offending axis) so a true 2D scatter is possible afterward
                        x_bad_mask = x_below0 | x_above1
                        y_bad_mask = y_below0 | y_above1
                        if x_bad_mask.any():
                            for xv, yv in zip(x[x_bad_mask], y[x_bad_mask]):
                                x_out_of_range_values.append((xv, yv, participant, panel_code, eye_label))
                        if y_bad_mask.any():
                            for xv, yv in zip(x[y_bad_mask], y[y_bad_mask]):
                                y_out_of_range_values.append((xv, yv, participant, panel_code, eye_label))
            except Exception as e:
                print(f"  FAILED {participant}: {e}")
                continue

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(OUT_DIR, "out_of_screen_stats.csv"), index=False)
    print(f"saved out_of_screen_stats.csv ({len(stats_df)} rows)")

    x_df = pd.DataFrame(x_out_of_range_values, columns=["x", "y", "participant", "panel", "eye"])
    y_df = pd.DataFrame(y_out_of_range_values, columns=["x", "y", "participant", "panel", "eye"])
    x_df.to_csv(os.path.join(OUT_DIR, "out_of_screen_x_values.csv"), index=False)
    y_df.to_csv(os.path.join(OUT_DIR, "out_of_screen_y_values.csv"), index=False)
    print(f"saved out_of_screen_x_values.csv ({len(x_df)} rows), out_of_screen_y_values.csv ({len(y_df)} rows)")


if __name__ == "__main__":
    main()

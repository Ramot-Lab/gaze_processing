"""
For the participants whose old (Dom_Eye, no-accuracy-threshold, non-isolated) batch video
run produced zero videos, render videos for whichever of their EXCLUDED panels (per
exclusion_report.csv) failed purely on the accuracy<2deg criterion while still having a
usable eye (NaN ratio < 10%) - i.e. panels worth looking at despite being excluded today.
Uses the analysis_eye override so the loader accepts that panel regardless of accuracy
(explicit_override still enforces the NaN-ratio gate, which we've already confirmed passes).

Rendering itself is visualize_data.show_running_video_60fps (Noam's function), unchanged.
"""
import os

import pandas as pd

from calibration_drift_qa import _iter_subject_dirs
from participant_gaze_data_manager import ParticipantGazeDataManager
from visualize_data import show_running_video_60fps

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
EXCLUSION_REPORT = "/Volumes/ramot/Noam_M/calibration_qc/exclusion_report.csv"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc/videos/excluded_low_nan"

TARGET_PARTICIPANTS = [
    "AD732", "KS130", "KT158", "SM144", "YR187", "AP344",
    "BR876", "GR333", "GT239", "SE378", "SM964", "ST888",
]


def _pick_eye(row):
    l_ok = pd.notna(row.l_nan_pct) and row.l_nan_pct < 10
    r_ok = pd.notna(row.r_nan_pct) and row.r_nan_pct < 10
    if l_ok and r_ok:
        return "l" if row.l_nan_pct <= row.r_nan_pct else "r"
    if l_ok:
        return "l"
    if r_ok:
        return "r"
    return None


def main():
    df = pd.read_csv(EXCLUSION_REPORT)
    sub = df[df.participant.isin(TARGET_PARTICIPANTS) & df.panel.notna()].copy()
    sub["eye"] = sub.apply(_pick_eye, axis=1)
    qualifying = sub[sub["eye"].notna()]
    print(f"{len(qualifying)} excluded panels qualify (NaN<10% on at least one eye, "
          f"accuracy ignored):")
    print(qualifying[["group", "participant", "panel", "l_nan_pct", "r_nan_pct", "eye"]].to_string())

    # participant dir lookup (need the FULL path for ParticipantGazeDataManager's self.group
    # to resolve correctly - see render_padded_gaze_video.py's earlier bug/fix)
    dir_by_participant = {}
    for group, subject_dir in _iter_subject_dirs(MAIN_DATA_PATH, ("HC", "pwMS")):
        dir_by_participant[os.path.basename(subject_dir)] = (group, subject_dir)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for participant, part_rows in qualifying.groupby("participant"):
        if participant not in dir_by_participant:
            print(f"  SKIP {participant}: not found via _iter_subject_dirs (Tobii_Sucks/DONTUSE excluded?)")
            continue
        group, subject_dir = dir_by_participant[participant]
        for eye, eye_rows in part_rows.groupby("eye"):
            try:
                sd = ParticipantGazeDataManager(subject_dir, MAIN_DATA_PATH, "SDMT", group, analysis_eye=eye)
            except Exception as e:
                print(f"  FAILED to load {participant} with analysis_eye={eye}: {e}")
                continue
            for _, row in eye_rows.iterrows():
                panel = row["panel"]
                if panel not in sd.matched_data:
                    print(f"  SKIP {participant}/{panel} (eye={eye}): not in matched_data "
                          f"after forcing eye (unexpected)")
                    continue
                try:
                    show_running_video_60fps(sd, panel, OUTPUT_DIR, target_fps=60)
                except Exception as e:
                    print(f"  FAILED video {participant}/{panel} (eye={eye}): {e}")

    print("ALL DONE")


if __name__ == "__main__":
    main()

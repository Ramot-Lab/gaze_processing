"""
Calibration quality QA + dictionary-region drift correction for SDMT recordings.

Background
----------
Each SDMT run (one .mat file, 3 panels) starts with a single Tobii calibration +
validation. That validation has a per-participant/per-panel spatial bias: gaze in the
dictionary (upper) area of the image drifts away from its true position, and the size of
that drift is not known ahead of time. This breaks `SearchFinder`, which classifies a
fixation as "in the dictionary" using a fixed y-threshold (`dict_ratio=0.175`) - if drift
pushes a dictionary fixation below that line, it is silently dropped from `Search`
grouping, which then corrupts everything built on top of it (trial extraction, sequence
analysis, etc).

This module:
  1. Parses the calibration/validation quality report Tobii writes into `messages`
     (see `participant_gaze_data_manager.parse_calibration_quality_message`) and builds a
     readable QA table over all participants/panels.
  2. Checks whether `Dom_Eye` (the eye actually used for analysis, see
     `ParticipantGazeDataManager.prepare_gaze_data_for_preprocessing`) is the eye Tobii
     calibrated best.
  3. Estimates the dictionary-region drift per participant/panel from the data itself
     (using the known triggering_symbol -> dictionary-number ROI mapping) and builds a
     `gaze_correction` closure that `TrialManager` can re-run with.
  4. Quantifies whether the correction actually helped (distance-to-target and dictionary
     Search-detection rate, before vs after).
  5. Renders the drift-corrected gaze path over the panel image as a video, given just a
     participant's name (see `create_gaze_correction_video`) - reusing the per-panel
     annotated gaze CSV the pipeline already saves to disk and the drift estimates already
     written by `build_drift_qa_table` (drift_correction_qa.csv), in the same rendering
     style as `visualize_data.show_running_video_60fps`. Deliberately does not construct a
     `ParticipantGazeDataManager`, which is the expensive part of the normal pipeline.
"""

import glob
import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from constants import (
    KEY_CALIBRATION_INFO, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V,
    FIXATION_CSV_KEY_FIXATION, FIXATION_VALID_STATUS, TIME_STAMP,
    SECONDS_TO_MICROSECOND_FACTOR, SACCADE_IDX, VIDEO_CODEC, FIXATION_COLOR,
    SACCADE_COLOR, RADIUS, THICKNESS, SCREEN_SIZE,
)
from participant_gaze_data_manager import ParticipantGazeDataManager, CALIBRATION_METRIC_KEYS
from trial_manager import TrialManager
from utils import prepare_image_and_gaze

# Fraction of image height down to the line separating the dictionary (symbols/numbers)
# from the search-grid text below it - measured directly on the original panel image
# (870/3861 px, top-left origin). Every panel image shares this exact same layout, so this
# is a fixed constant, not something to estimate per participant/panel.
DICTIONARY_BOUNDARY_RATIO = 870 / 3861

DEFAULT_MAIN_DATA_PATH = "/Volumes/Noam_M/Results/Behavior"
DEFAULT_CALIBRATION_QC_DIR = "/Volumes/Noam_M/calibration_qc"


# ---------------------------------------------------------------------------------------
# What each calibration measurement means (plain-language glossary, degrees unless noted)
# ---------------------------------------------------------------------------------------
CALIBRATION_MEASUREMENT_GLOSSARY = {
    "acc": "Accuracy (deg): mean angular error between the gaze estimate and the true "
           "validation target, averaged over that point's samples. Lower is better - this "
           "is the headline 'how good was calibration' number.",
    "accX": "Accuracy, horizontal component (deg). Splitting acc into accX/accY shows "
            "whether the error leans horizontal or vertical.",
    "accY": "Accuracy, vertical component (deg).",
    "std": "Precision - STD (deg): spread of gaze samples around their own mean while "
           "fixating the point (noise/instability), not an offset from the target.",
    "rms": "Precision - RMS (deg): root-mean-square sample-to-sample distance (jitter "
           "between consecutive samples). A second, distinct noise measure from STD.",
    "data_loss": "Data loss (%%): fraction of samples with no valid gaze data while "
                 "fixating this validation point.",
}

CALIBRATION_TABLE_COLUMN_NOTES = (
    "Each row is one participant + panel. Left/right eye columns are the *average* row of "
    "that eye's validation report (mean over all validation points). 'calibration_no' / "
    "'validation_no' identify which (possibly re-taken) calibration/validation this is - "
    "see CALIBRATION_MEASUREMENT_GLOSSARY for what each metric means."
)


def _iter_subject_dirs(main_data_path, groups):
    for group in groups:
        for subject_dir in sorted(glob.glob(os.path.join(main_data_path, group, "*"))):
            if os.path.isdir(subject_dir):
                yield group, subject_dir


def _panel_image_path(main_data_path, task, panel):
    """Same suffix-match convention ParticipantGazeDataManager.group_task_info uses."""
    matches = glob.glob(os.path.join(main_data_path, "panels_images", task, "*.jpg"))
    matches = [m for m in matches if m.endswith(f"_{panel}.jpg")]
    if not matches:
        raise FileNotFoundError(f"No panel image found for task={task!r} panel={panel!r}")
    return matches[0]


def write_glossary(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "calibration_measurement_glossary.txt")
    with open(path, "w") as f:
        f.write(CALIBRATION_TABLE_COLUMN_NOTES + "\n\n")
        for key, explanation in CALIBRATION_MEASUREMENT_GLOSSARY.items():
            f.write(f"{key}: {explanation}\n")
    return path


# ---------------------------------------------------------------------------------------
# Item 1: calibration quality table
# ---------------------------------------------------------------------------------------
def build_calibration_table(main_data_path, task="SDMT", groups=("HC", "pwMS")):
    """One row per participant + panel with that run's calibration/validation quality."""
    rows = []
    for group, subject_dir in _iter_subject_dirs(main_data_path, groups):
        if task not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)
        try:
            subject_data = ParticipantGazeDataManager(subject_dir, main_data_path, task, group)
        except Exception as e:
            rows.append({"group": group, "participant": participant, "panel": None,
                         "calibration_missing": True, "error": str(e)})
            continue

        for panel in subject_data.matched_data:
            row = {
                "group": group,
                "participant": subject_data.name,
                "panel": panel,
                "dominant_eye": subject_data.dom_Eye,
            }
            calibration_info = subject_data.matched_data[panel].get(KEY_CALIBRATION_INFO)
            if calibration_info is None:
                row["calibration_missing"] = True
                rows.append(row)
                continue

            row["calibration_missing"] = False
            row["calibration_no"] = calibration_info["calibration_no"]
            row["validation_no"] = calibration_info["validation_no"]
            for eye in ("left", "right"):
                average = calibration_info[eye]["average"] or {}
                for key in CALIBRATION_METRIC_KEYS:
                    row[f"{eye}_{key}"] = average.get(key)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Item 2: is the dominant eye the best-calibrated eye?
# ---------------------------------------------------------------------------------------
def build_dominant_eye_table(calibration_df):
    """Derived from build_calibration_table(): does Dom_Eye match the better-calibrated eye?"""
    df = calibration_df[calibration_df["calibration_missing"] == False].copy()
    df["better_eye"] = np.where(df["left_acc"] <= df["right_acc"], "l", "r")
    df["accuracy_gap_deg"] = (df["left_acc"] - df["right_acc"]).abs()
    df["dominant_matches_best"] = df["better_eye"] == df["dominant_eye"].str.lower()
    columns = ["group", "participant", "panel", "dominant_eye", "better_eye",
               "left_acc", "right_acc", "accuracy_gap_deg", "dominant_matches_best"]
    return df[columns].reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Item 3: dictionary-region drift estimation & correction
# ---------------------------------------------------------------------------------------
@dataclass
class DriftEstimate:
    dx: float
    dy: float
    dx_median: float
    dy_median: float
    boundary_y: float
    n_trials_used: int
    per_trial_deltas: list = field(default_factory=list)  # [(trial_idx, dx, dy), ...]


_CACHED_BOUNDARY_Y = None


def get_dictionary_boundary_y():
    """
    Every panel image shares the exact same layout, so the dictionary/search-grid boundary
    is a fixed fraction of screen height (DICTIONARY_BOUNDARY_RATIO), not something to
    estimate per participant/panel. Computed once and cached for the rest of the run.
    """
    global _CACHED_BOUNDARY_Y
    if _CACHED_BOUNDARY_Y is None:
        _CACHED_BOUNDARY_Y = DICTIONARY_BOUNDARY_RATIO * SCREEN_SIZE[0]
        print(f"Dictionary/search-grid boundary_y = {_CACHED_BOUNDARY_Y:.2f}px "
              f"(fixed ratio {DICTIONARY_BOUNDARY_RATIO:.4f} of screen height, "
              f"same for all participants/panels)")
    return _CACHED_BOUNDARY_Y


def _valid_trials(tm: TrialManager):
    valid = []
    for trial in tm.trials:
        symbol = trial.triggering_symbol
        if symbol is None or symbol.value is None:
            continue
        start, end = trial.start_time, trial.end_time
        if start is None or end is None:
            continue
        if (isinstance(start, float) and math.isnan(start)) or (isinstance(end, float) and math.isnan(end)):
            continue
        valid.append(trial)
    return valid


def estimate_dictionary_drift(tm: TrialManager) -> DriftEstimate:
    """
    For each trial, the number the participant was looking for is known exactly
    (triggering_symbol.value -> dictionary "numbers" row ROI at idx = value + 8, see
    PanelSymbols: row 0-8 is the symbol/key row, row 9-17 is the digit row, both laid out
    in the same fixed 1-9 column order). The anchor fixation is the last raw fixation
    (independent of SearchFinder's Search grouping) within the trial's time window that
    falls above the dictionary/grid boundary. The mean offset of anchor fixations from
    their target ROI center is the empirical dictionary drift.
    """
    boundary_y = get_dictionary_boundary_y()
    roi_by_idx = {r.idx: r for r in tm.rois}

    deltas = []
    for trial in _valid_trials(tm):
        target_roi = roi_by_idx.get(trial.triggering_symbol.value + 8)
        if target_roi is None:
            continue

        anchor = None
        for fixation in tm.fixations:
            if trial.start_time <= fixation.start_time <= trial.end_time and fixation.position[1] < boundary_y:
                anchor = fixation  # tm.fixations is chronological -> last match wins
        if anchor is None:
            continue

        dx = anchor.position[0] - target_roi.center[0]
        dy = anchor.position[1] - target_roi.center[1]
        deltas.append((trial.idx, dx, dy))

    if not deltas:
        return DriftEstimate(dx=0.0, dy=0.0, dx_median=0.0, dy_median=0.0,
                              boundary_y=boundary_y, n_trials_used=0, per_trial_deltas=[])

    dxs = [d[1] for d in deltas]
    dys = [d[2] for d in deltas]
    return DriftEstimate(
        dx=float(np.mean(dxs)), dy=float(np.mean(dys)),
        dx_median=float(np.median(dxs)), dy_median=float(np.median(dys)),
        boundary_y=boundary_y, n_trials_used=len(deltas), per_trial_deltas=deltas,
    )


def build_drift_correction_fn(dx, dy, boundary_y):
    """gaze_correction closure for TrialManager: shift only rows above the dictionary boundary."""
    def _correct(annotated_data):
        corrected = annotated_data.copy()
        mask = corrected[FIXATION_CSV_KEY_EYE_V] < boundary_y
        corrected.loc[mask, FIXATION_CSV_KEY_EYE_H] -= dx
        corrected.loc[mask, FIXATION_CSV_KEY_EYE_V] -= dy
        return corrected
    return _correct


def get_drift_corrected_trial_manager(subject_data, panel):
    """
    Runs TrialManager once (uncorrected) to estimate the dictionary drift, then runs it a
    second time with that drift subtracted from the upper/dictionary-region gaze samples.
    Returns (raw_trial_manager, corrected_trial_manager, drift_estimate).
    """
    tm_raw = TrialManager(subject_data, panel)
    drift = estimate_dictionary_drift(tm_raw)
    correction_fn = build_drift_correction_fn(drift.dx, drift.dy, drift.boundary_y)
    tm_corrected = TrialManager(subject_data, panel, gaze_correction=correction_fn)
    return tm_raw, tm_corrected, drift


# ---------------------------------------------------------------------------------------
# Item 4: before/after QA
# ---------------------------------------------------------------------------------------
def _pct_trials_with_dictionary_search(tm: TrialManager):
    valid = _valid_trials(tm)
    if not valid:
        return None
    return 100.0 * sum(1 for t in valid if t.searches) / len(valid)


def evaluate_drift_correction(subject_data, panel):
    """Compares anchor-to-target distance and dictionary Search-detection rate, before vs after."""
    tm_raw, tm_corrected, drift_before = get_drift_corrected_trial_manager(subject_data, panel)
    drift_after = estimate_dictionary_drift(tm_corrected)

    dists_before = [math.hypot(dx, dy) for _, dx, dy in drift_before.per_trial_deltas]
    dists_after = [math.hypot(dx, dy) for _, dx, dy in drift_after.per_trial_deltas]

    return {
        "n_trials_used": drift_before.n_trials_used,
        "dx_before": drift_before.dx, "dy_before": drift_before.dy,
        "dx_after": drift_after.dx, "dy_after": drift_after.dy,
        "boundary_y": drift_before.boundary_y,
        "mean_dist_before_px": float(np.mean(dists_before)) if dists_before else None,
        "median_dist_before_px": float(np.median(dists_before)) if dists_before else None,
        "mean_dist_after_px": float(np.mean(dists_after)) if dists_after else None,
        "median_dist_after_px": float(np.median(dists_after)) if dists_after else None,
        "pct_trials_with_dict_search_before": _pct_trials_with_dictionary_search(tm_raw),
        "pct_trials_with_dict_search_after": _pct_trials_with_dictionary_search(tm_corrected),
    }


def build_drift_qa_table(main_data_path, task="SDMT", groups=("HC", "pwMS")):
    """One row per participant + panel comparing pre/post drift-correction quality."""
    rows = []
    for group, subject_dir in _iter_subject_dirs(main_data_path, groups):
        if task not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)
        try:
            subject_data = ParticipantGazeDataManager(subject_dir, main_data_path, task, group)
        except Exception as e:
            rows.append({"group": group, "participant": participant, "panel": None, "error": str(e)})
            continue

        for panel in subject_data.matched_data:
            row = {"group": group, "participant": subject_data.name, "panel": panel}
            try:
                row.update(evaluate_drift_correction(subject_data, panel))
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Item 5: video of the drift-corrected gaze, without rebuilding a ParticipantGazeDataManager
# ---------------------------------------------------------------------------------------
def _load_saved_annotated_gaze(main_data_path, participant, panel):
    """
    Reads the per-panel gaze CSV the pipeline already saves at
    processing_results/{participant}/task_{panel}_fixation.csv (written as a side effect
    of ParticipantGazeDataManager.annotate_gaze_events -> utils.generate_fixations_threshold_based).
    Columns: t, eye_horizontal, eye_vertical (normalized [0,1]), status, evt. Reusing this
    avoids re-loading the participant's raw .mat data just to render a QA video.
    """
    csv_path = os.path.join(main_data_path, "processing_results", participant, f"task_{panel}_fixation.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No saved annotated gaze CSV at {csv_path}. Run the normal pipeline (e.g. "
            f"TrialManager) at least once for this participant/panel first."
        )
    columns = [TIME_STAMP, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_VALID_STATUS, FIXATION_CSV_KEY_FIXATION]
    return pd.read_csv(csv_path)[columns]


def render_drift_corrected_gaze_video(main_data_path, participant, panel, dx, dy, boundary_y,
                                       task="SDMT", output_dir=None, target_fps=60):
    """
    Renders the drift-corrected gaze path over the panel image. Only x/y in the dictionary
    (upper) region are shifted - by build_drift_correction_fn, the exact same closure the
    real pipeline uses - everything else (t, evt) is untouched, matching how the fix is
    meant to apply to the saved CSV.
    """
    annotated = _load_saved_annotated_gaze(main_data_path, participant, panel)
    img = cv2.imread(_panel_image_path(main_data_path, task, panel))
    img_resized, gaze_scaled = prepare_image_and_gaze(img, annotated)
    gaze_corrected = build_drift_correction_fn(dx, dy, boundary_y)(gaze_scaled)

    eye_x = gaze_corrected[FIXATION_CSV_KEY_EYE_H].to_numpy()
    eye_y = gaze_corrected[FIXATION_CSV_KEY_EYE_V].to_numpy()
    fixation = gaze_corrected[FIXATION_CSV_KEY_FIXATION].to_numpy()
    times = gaze_corrected[TIME_STAMP].to_numpy() / SECONDS_TO_MICROSECOND_FACTOR

    output_dir = output_dir or os.path.join(DEFAULT_CALIBRATION_QC_DIR, "videos")
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{participant}_{panel}_drift_corrected.mp4")

    img_height, img_width = img_resized.shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), target_fps, (img_width, img_height))

    duration = times[-1]
    n_frames = max(1, int(duration * target_fps))
    gaze_idx = 0
    for frame_time in np.linspace(0, duration, n_frames):
        while gaze_idx + 1 < len(times) and times[gaze_idx + 1] <= frame_time:
            gaze_idx += 1

        frame = img_resized.copy()
        x, y = eye_x[gaze_idx], eye_y[gaze_idx]
        if not (np.isnan(x) or np.isnan(y)):
            color = SACCADE_COLOR if fixation[gaze_idx] == SACCADE_IDX else FIXATION_COLOR
            cv2.circle(frame, (int(x), int(y)), radius=RADIUS, color=color, thickness=THICKNESS)
        writer.write(frame)

    writer.release()
    print(f"Drift-corrected gaze video saved to {video_path}")
    return video_path


def create_gaze_correction_video(participant, panel=None, main_data_path=DEFAULT_MAIN_DATA_PATH,
                                  task="SDMT", calibration_qc_dir=DEFAULT_CALIBRATION_QC_DIR,
                                  output_dir=None, target_fps=60):
    """
    End-to-end entry point: participant name in, drift-corrected gaze video(s) out. Looks
    up the drift already computed by build_drift_qa_table() in drift_correction_qa.csv
    instead of recomputing it, so this never has to construct a ParticipantGazeDataManager.
    If panel is None, renders one video per panel available for this participant.
    """
    drift_qa_path = os.path.join(calibration_qc_dir, "drift_correction_qa.csv")
    if not os.path.exists(drift_qa_path):
        raise FileNotFoundError(
            f"{drift_qa_path} not found - run build_drift_qa_table() (or this module's "
            f"__main__ block) at least once so drift estimates are available."
        )

    drift_qa_df = pd.read_csv(drift_qa_path)
    rows = drift_qa_df[drift_qa_df["participant"] == participant]
    if panel is not None:
        rows = rows[rows["panel"] == panel]
    if rows.empty:
        raise ValueError(f"No drift estimate found for participant={participant!r} panel={panel!r} in {drift_qa_path}")

    video_paths = []
    for _, row in rows.iterrows():
        if pd.isna(row.get("dx_before")): # or pd.isna(row.get("boundary_y")):
            print(f"Skipping {participant}/{row['panel']}: no usable drift estimate "
                  f"(n_trials_used={row.get('n_trials_used')}).")
            continue
        video_paths.append(render_drift_corrected_gaze_video(
            main_data_path, participant, row["panel"],
            dx=row["dx_before"], dy=row["dy_before"], boundary_y=_CACHED_BOUNDARY_Y,
            task=task, output_dir=output_dir, target_fps=target_fps,
        ))
    return video_paths


if __name__ == "__main__":
    MAIN_DATA_PATH = DEFAULT_MAIN_DATA_PATH
    OUTPUT_DIR = DEFAULT_CALIBRATION_QC_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    get_dictionary_boundary_y()  # computed once up front, cached + printed

    # Set a participant (and optionally a panel) here to render its drift-corrected gaze
    # video(s) instead of rebuilding the QA tables below.
    VIDEO_PARTICIPANT = "LT157"  # e.g. "AG562"; leave None to run the QA sweep instead
    VIDEO_PANEL = "i1"  # e.g. "l3"; None renders every panel available for the participant

    if VIDEO_PARTICIPANT is not None:
        create_gaze_correction_video(VIDEO_PARTICIPANT, panel=VIDEO_PANEL,
                                      main_data_path=MAIN_DATA_PATH, calibration_qc_dir=OUTPUT_DIR)
        sys.exit(0)

    # glossary_path = write_glossary(OUTPUT_DIR)
    # print(f"Glossary written to {glossary_path}")

    # calibration_df = build_calibration_table(MAIN_DATA_PATH)
    # calibration_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_quality.csv"), index=False)
    # print(f"Calibration quality table: {len(calibration_df)} rows")

    # dominant_eye_df = build_dominant_eye_table(calibration_df)
    # dominant_eye_df.to_csv(os.path.join(OUTPUT_DIR, "dominant_eye_check.csv"), index=False)
    # if len(dominant_eye_df):
    #     pct_match = 100.0 * dominant_eye_df["dominant_matches_best"].mean()
    #     print(f"Dominant eye matches best-calibrated eye in {pct_match:.1f}% of "
    #           f"{len(dominant_eye_df)} participant-panels")

    # drift_qa_df = build_drift_qa_table(MAIN_DATA_PATH)
    # drift_qa_df.to_csv(os.path.join(OUTPUT_DIR, "drift_correction_qa.csv"), index=False)
    # print(f"Drift correction QA table: {len(drift_qa_df)} rows")

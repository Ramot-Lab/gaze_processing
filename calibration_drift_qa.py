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
import random
import re
import sys
from dataclasses import dataclass, field

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import (
    KEY_CALIBRATION_INFO, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V,
    FIXATION_CSV_KEY_FIXATION, FIXATION_VALID_STATUS, TIME_STAMP,
    SECONDS_TO_MICROSECOND_FACTOR, FIXATION_IDX, SACCADE_IDX, VIDEO_CODEC, FIXATION_COLOR,
    SACCADE_COLOR, RADIUS, THICKNESS, SCREEN_SIZE,
)
from exclusion_policy import load_tobii_sucks_excluded_participants
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
    tobii_sucks_excluded = load_tobii_sucks_excluded_participants()
    for group in groups:
        for subject_dir in sorted(glob.glob(os.path.join(main_data_path, group, "*"))):
            if not os.path.isdir(subject_dir):
                continue
            name = os.path.basename(subject_dir)
            # DONTUSE is a manual "exclude this person" flag unrelated to data quality -
            # skip entirely rather than let it show up as a calibration/load error.
            if "DONTUSE" in name.upper():
                continue
            # Tobii_Sucks==YES in the behavioral summary table (decision 2026-08-26) -
            # broader/authoritative superset of the DONTUSE folder-suffix flag.
            if name in tobii_sucks_excluded:
                continue
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

        # Each entry is one calibration event (mat file) that failed to load/process -
        # logged independently so a bad day doesn't hide whether the participant's OTHER
        # day was fine (see ParticipantGazeDataManager.load_errors).
        for load_error in subject_data.load_errors:
            rows.append({"group": group, "participant": subject_data.name, "panel": None,
                         "calibration_missing": True, "error": load_error["error"],
                         "recording_date": load_error.get("recording_date") or load_error.get("file")})

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


def count_negative_y_fixations(tm: TrialManager):
    """
    Fixations whose (corrected) y position is < 0, i.e. pushed above the top edge of the
    screen entirely - a sign the correction overshot rather than just recentering the
    dictionary-region drift. Works on any TrialManager, corrected or not (should be ~0 on
    an uncorrected one, since raw gaze rarely reports off-screen positions).
    """
    return sum(1 for f in tm.fixations if f.position[1] < 0)


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
        "n_negative_y_fixations_before": count_negative_y_fixations(tm_raw),
        "n_negative_y_fixations_after": count_negative_y_fixations(tm_corrected),
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
# Item 6: whole-dictionary mean-fixation drift correction (a coarser alternative to
# estimate_dictionary_drift above) + gap-zone fixation counts before/after
# ---------------------------------------------------------------------------------------

# Ratio y-location of the line separating the dictionary's symbol row (top) from its
# number row (bottom) - measured the same way as DICTIONARY_BOUNDARY_RATIO: detecting the
# horizontal grid line in the original panel image. Identical across all 6 panel images
# (line spans rows 286-291 out of image height 3867, midpoint 288.5).
DICTIONARY_MIDLINE_RATIO = 288.5 / 3867

# Ratio y-location of the dictionary box's own top border (the line above the symbol row),
# found with the same horizontal-line detection - identical across all 6 panels (line spans
# rows 4-8, midpoint 6). Used to derive the symbol row's own center (top border to midline
# midpoint) for row-spacing-unit conversions elsewhere.
_DICTIONARY_TOP_RATIO = 6 / 3867

# Ratio y-location of the second dictionary row's own lower border (the line below it,
# before the gap down to the dictionary/grid boundary), found with the same
# horizontal-line detection - identical across all 6 panels (line spans rows 571-573,
# midpoint 572). Used only to derive DICTIONARY_SECOND_ROW_MID_RATIO below.
_DICTIONARY_ROW2_BOTTOM_RATIO = 572 / 3867

# Ratio y-location of the vertical center of the second (lower) dictionary row - the
# midpoint between the symbol/number midline and that row's own lower border, as an
# alternative reference line to DICTIONARY_MIDLINE_RATIO for the whole-dictionary drift
# correction (see estimate_whole_dictionary_drift_from_csv's reference_ratio param).
DICTIONARY_SECOND_ROW_MID_RATIO = (DICTIONARY_MIDLINE_RATIO + _DICTIONARY_ROW2_BOTTOM_RATIO) / 2

# SearchFinder's own dict_ratio default (see SearchFinder.find) - the y-threshold it
# actually uses today to decide whether a fixation counts as "in the dictionary" at all.
# Fixations between this and get_dictionary_boundary_y() are in the gap where drift can
# cause SearchFinder to miss them even though they really are dictionary fixations.
SEARCHFINDER_DICT_RATIO = 0.175

_CACHED_MIDLINE_Y = None


def get_dictionary_midline_y():
    """
    The line separating the dictionary's symbol row from its number row - same
    fixed-layout reasoning as get_dictionary_boundary_y(). This is the "should be centered
    here" reference line for the whole-dictionary mean-fixation drift method below.
    """
    global _CACHED_MIDLINE_Y
    if _CACHED_MIDLINE_Y is None:
        _CACHED_MIDLINE_Y = DICTIONARY_MIDLINE_RATIO * SCREEN_SIZE[0]
        print(f"Dictionary midline_y = {_CACHED_MIDLINE_Y:.2f}px "
              f"(fixed ratio {DICTIONARY_MIDLINE_RATIO:.4f} of screen height, "
              f"same for all participants/panels)")
    return _CACHED_MIDLINE_Y


@dataclass
class WholeDictionaryDriftEstimate:
    dy: float  # the correction: mean_y_after_cleaning - midline_y
    mean_y_before_cleaning: float
    mean_y_after_cleaning: float
    n_fixations_before_cleaning: int
    n_fixations_after_cleaning: int
    boundary_y: float
    midline_y: float


def estimate_whole_dictionary_drift(tm: TrialManager, outlier_z_thresh=3.0) -> WholeDictionaryDriftEstimate:
    """
    Coarser alternative to estimate_dictionary_drift(): instead of anchoring on per-trial
    triggering-symbol targets, pool every fixation above the dictionary/grid boundary
    across the whole panel, drop y-outliers (more than outlier_z_thresh standard
    deviations from the mean), and compare the mean y of what's left to the line that
    should split the dictionary's two rows evenly (get_dictionary_midline_y()). The signed
    gap is a single per-participant/panel vertical correction - use with
    build_drift_correction_fn(dx=0, dy=result.dy, boundary_y=result.boundary_y).
    """
    boundary_y = get_dictionary_boundary_y()
    midline_y = get_dictionary_midline_y()

    dict_ys = np.array([f.position[1] for f in tm.fixations if f.position[1] < boundary_y])
    n_before = len(dict_ys)
    if n_before == 0:
        return WholeDictionaryDriftEstimate(
            dy=0.0, mean_y_before_cleaning=float("nan"), mean_y_after_cleaning=float("nan"),
            n_fixations_before_cleaning=0, n_fixations_after_cleaning=0,
            boundary_y=boundary_y, midline_y=midline_y,
        )

    mean_before = float(dict_ys.mean())
    std_before = dict_ys.std()
    cleaned = dict_ys[np.abs(dict_ys - mean_before) <= outlier_z_thresh * std_before] if std_before > 0 else dict_ys
    mean_after = float(cleaned.mean()) if len(cleaned) else mean_before

    return WholeDictionaryDriftEstimate(
        dy=mean_after - midline_y,
        mean_y_before_cleaning=mean_before, mean_y_after_cleaning=mean_after,
        n_fixations_before_cleaning=n_before, n_fixations_after_cleaning=len(cleaned),
        boundary_y=boundary_y, midline_y=midline_y,
    )


def get_whole_dictionary_corrected_trial_manager(subject_data, panel):
    """Same shape as get_drift_corrected_trial_manager(), using estimate_whole_dictionary_drift()."""
    tm_raw = TrialManager(subject_data, panel)
    estimate = estimate_whole_dictionary_drift(tm_raw)
    correction_fn = build_drift_correction_fn(dx=0.0, dy=estimate.dy, boundary_y=estimate.boundary_y)
    tm_corrected = TrialManager(subject_data, panel, gaze_correction=correction_fn)
    return tm_raw, tm_corrected, estimate


def count_dictionary_grid_gap_fixations(tm: TrialManager):
    """
    Number of fixations in the y-band SearchFinder itself doesn't currently treat as
    "in the dictionary" (below its own dict_ratio threshold) but that the measured true
    dictionary/grid boundary says still belong to the dictionary. Fixations here are
    exactly the ones drift can cause SearchFinder to silently drop from a Search.
    """
    lower = SEARCHFINDER_DICT_RATIO * SCREEN_SIZE[0]
    upper = get_dictionary_boundary_y()
    return sum(1 for f in tm.fixations if lower <= f.position[1] < upper)


def build_whole_dictionary_drift_table(main_data_path, task="SDMT", groups=("HC", "pwMS")):
    """
    One row per participant + panel: the whole-dictionary drift estimate, plus the
    dictionary/grid gap-zone fixation count before vs after applying that correction.
    """
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
                tm_raw, tm_corrected, estimate = get_whole_dictionary_corrected_trial_manager(subject_data, panel)
                row.update({
                    "dy": estimate.dy,
                    "mean_y_before_cleaning": estimate.mean_y_before_cleaning,
                    "mean_y_after_cleaning": estimate.mean_y_after_cleaning,
                    "n_fixations_before_cleaning": estimate.n_fixations_before_cleaning,
                    "n_fixations_after_cleaning": estimate.n_fixations_after_cleaning,
                    "n_gap_zone_fixations_before_correction": count_dictionary_grid_gap_fixations(tm_raw),
                    "n_gap_zone_fixations_after_correction": count_dictionary_grid_gap_fixations(tm_corrected),
                    "n_negative_y_fixations_before": count_negative_y_fixations(tm_raw),
                    "n_negative_y_fixations_after": count_negative_y_fixations(tm_corrected),
                })
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Item 6b: same whole-dictionary drift analysis, but reading the already-saved annotated
# gaze CSVs directly - no ParticipantGazeDataManager/TrialManager, ~instant per panel.
#
# The CSVs (processing_results/{participant}/task_{panel}_fixation[_{eye}].csv) hold
# eye_horizontal/eye_vertical in NORMALIZED [0,1] coordinates - i.e. before
# utils.prepare_image_and_gaze's screen-space scaling. That scaling multiplies
# eye_vertical by SCREEN_SIZE[0] with no y-offset (only x gets an offset, for horizontal
# centering), so DICTIONARY_BOUNDARY_RATIO / DICTIONARY_MIDLINE_RATIO / SEARCHFINDER_DICT_RATIO
# can be compared directly against normalized y with no conversion - the ratios ARE the
# normalized-space thresholds.
# ---------------------------------------------------------------------------------------
_SAVED_FIXATION_CSV_RE = re.compile(r"^task_(.+?)_fixation(?:_[lr])?\.csv$")


def _iter_saved_gaze_panels(main_data_path, groups=("HC", "pwMS")):
    """(group, participant, panel) for every participant+panel with a saved gaze CSV on disk."""
    tobii_sucks_excluded = load_tobii_sucks_excluded_participants()
    participant_to_group = {}
    for group in groups:
        for subject_dir in glob.glob(os.path.join(main_data_path, group, "*")):
            participant_to_group[os.path.basename(subject_dir)] = group

    processing_results_dir = os.path.join(main_data_path, "processing_results")
    for participant_dir in sorted(glob.glob(os.path.join(processing_results_dir, "*"))):
        participant = os.path.basename(participant_dir)
        group = participant_to_group.get(participant)
        if group is None:
            continue
        # Tobii_Sucks==YES participants may still have stale CSVs on disk from before this
        # exclusion existed - skip them here too, not just in _iter_subject_dirs, so they
        # don't leak into the lightweight from_csv tables.
        if participant in tobii_sucks_excluded:
            continue

        panels = set()
        for csv_path in glob.glob(os.path.join(participant_dir, "task_*_fixation*.csv")):
            match = _SAVED_FIXATION_CSV_RE.match(os.path.basename(csv_path))
            if match:
                panels.add(match.group(1))

        for panel in sorted(panels):
            yield group, participant, panel


def _extract_fixation_block_ys(annotated):
    """
    Groups the per-sample annotated gaze CSV into fixation BLOCKS, mirroring
    FixationHandler.process_fixations exactly: walk the rows in order, accumulate
    consecutive FIXATION_IDX samples, and close the current block when a SACCADE_IDX
    sample is hit (or at the end of the recording if it ends mid-fixation). Returns one
    y-value per fixation block (the mean eye_vertical over that block's samples) - i.e.
    counts fixation EVENTS, not raw samples, matching how tm.fixations works everywhere
    else in this file.
    """
    evt = annotated[FIXATION_CSV_KEY_FIXATION].to_numpy()
    y = annotated[FIXATION_CSV_KEY_EYE_V].to_numpy()

    block_means = []
    current = []
    for e, yy in zip(evt, y):
        if e == FIXATION_IDX:
            current.append(yy)
        elif e == SACCADE_IDX and current:
            block_means.append(np.mean(current))
            current = []
    if current:
        block_means.append(np.mean(current))

    return np.array(block_means)


def _extract_fixation_block_xy(annotated):
    """
    Same block-grouping as _extract_fixation_block_ys, but returns both coordinates: one
    (mean eye_horizontal, mean eye_vertical) pair per fixation block, in chronological
    order. Normalized [0,1] coordinates, same space as the saved CSV.
    """
    evt = annotated[FIXATION_CSV_KEY_FIXATION].to_numpy()
    x = annotated[FIXATION_CSV_KEY_EYE_H].to_numpy()
    y = annotated[FIXATION_CSV_KEY_EYE_V].to_numpy()

    block_x, block_y = [], []
    current_x, current_y = [], []
    for e, xx, yy in zip(evt, x, y):
        if e == FIXATION_IDX:
            current_x.append(xx)
            current_y.append(yy)
        elif e == SACCADE_IDX and current_x:
            block_x.append(np.mean(current_x))
            block_y.append(np.mean(current_y))
            current_x, current_y = [], []
    if current_x:
        block_x.append(np.mean(current_x))
        block_y.append(np.mean(current_y))

    return np.array(block_x), np.array(block_y)


# ---------------------------------------------------------------------------------------
# Raw (uncorrected) fixation x/y inspection - exploratory diagnostic ahead of drift-fix work.
# Two unit systems, both anchored to lines measured directly on the panel image (same
# horizontal/vertical line-detection method used for DICTIONARY_BOUNDARY_RATIO etc.),
# identical across all 6 panels:
#   - Y: row-spacing units. 0 = dictionary's own top border, 1 unit = the spacing between
#     the symbol row's center and the number row's center.
#   - X: column-width units. 0..9 = the 10 measured vertical grid lines bounding the 9
#     symbol/digit columns (column N sits between gridline N-1 and gridline N).
# ---------------------------------------------------------------------------------------
_PANEL_IMG_HEIGHT_PX = 3867
_PANEL_IMG_WIDTH_PX = 3386

# Raw image pixel x-positions of the 10 vertical grid lines separating the 9 symbol/digit
# columns, measured directly (vertical-line detection), identical across all 6 panels.
_DICTIONARY_COLUMN_GRIDLINES_PX = np.array(
    [419.0, 702.0, 984.0, 1267.0, 1550.5, 1833.0, 2115.0, 2398.0, 2682.0, 2964.0]
)


def _dictionary_column_gridlines_normalized():
    """
    The 10 column-boundary gridlines, converted from raw image pixels into the same
    normalized [0,1] coordinate space as the saved gaze CSVs, using the exact scale+offset
    from utils.prepare_image_and_gaze (image resized so its height matches the screen
    height, then horizontally centered within the screen width).
    """
    scale = SCREEN_SIZE[0] / _PANEL_IMG_HEIGHT_PX
    new_width = int(_PANEL_IMG_WIDTH_PX * scale)
    x_offset = (SCREEN_SIZE[1] - new_width) / 2
    return (_DICTIONARY_COLUMN_GRIDLINES_PX * scale + x_offset) / SCREEN_SIZE[1]


_RAW_XY_HISTOGRAM_BIN_EDGES = np.arange(-0.25, 9.25 + 0.01, 0.5)


def plot_raw_fixation_xy_histograms(main_data_path=DEFAULT_MAIN_DATA_PATH, pairs=None, groups=("HC", "pwMS"),
                                     output_dir=None, eye=None):
    """
    For each (participant, panel), plots the RAW (uncorrected) fixation blocks above the
    dictionary/grid boundary as a side-by-side Y histogram (row-spacing units) and X
    histogram (column-width units, labeled 1-9 under each symbol column) - both using the
    same 19-bin, 0.5-width, half-integer-offset scheme.

    pairs: iterable of (participant, panel) tuples to restrict to. If None, runs over every
    (participant, panel) with a saved gaze CSV (_iter_saved_gaze_panels over all groups).

    Saves one PNG per pair to output_dir (default: <DEFAULT_CALIBRATION_QC_DIR>/raw_xy_histograms).
    Returns the list of saved file paths.
    """
    output_dir = output_dir or os.path.join(DEFAULT_CALIBRATION_QC_DIR, "raw_xy_histograms")
    os.makedirs(output_dir, exist_ok=True)

    if pairs is None:
        pairs = [(participant, panel) for _, participant, panel in _iter_saved_gaze_panels(main_data_path, groups)]

    row1_center_ratio = (_DICTIONARY_TOP_RATIO + DICTIONARY_MIDLINE_RATIO) / 2
    row2_center_ratio = DICTIONARY_SECOND_ROW_MID_RATIO

    gridlines_x = _dictionary_column_gridlines_normalized()
    col_zero_x = gridlines_x[0]
    col_unit = np.diff(gridlines_x).mean()

    saved_paths = []
    for participant, panel in pairs:
        try:
            annotated = _load_saved_annotated_gaze(main_data_path, participant, panel, eye=eye)
            x, y = _extract_fixation_block_xy(annotated)
            mask = y < DICTIONARY_BOUNDARY_RATIO
            x_dict, y_dict = x[mask], y[mask]
            if len(x_dict) == 0:
                print(f"Skipping {participant}/{panel}: no fixation blocks above dictionary boundary")
                continue

            x_units = (x_dict - col_zero_x) / col_unit

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes[0].hist(y_dict, bins=30, color="skyblue", edgecolor="k")
            axes[0].axvline(row1_center_ratio, color="black", linestyle="--", linewidth=1,
                             label="symbol row center")
            axes[0].axvline(row2_center_ratio, color="green", linestyle="--", linewidth=1,
                             label="number row center")
            axes[0].set_xlabel("y position (normalized, 0 = top of screen)")
            axes[0].set_ylabel("Count (fixation blocks)")
            axes[0].set_title(f"Y histogram - all fixations above text (n={len(y_dict)})")
            axes[0].legend(fontsize=8)

            axes[1].hist(x_units, bins=_RAW_XY_HISTOGRAM_BIN_EDGES, color="salmon", edgecolor="k")
            for i in range(10):
                axes[1].axvline(i, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
            axes[1].set_xticks(np.arange(0, 9.5, 0.5))
            axes[1].set_xlabel("x position (column-width units, 0..9 = grid lines)")
            axes[1].set_ylabel("Count (fixation blocks)")
            axes[1].set_title(f"X histogram (n={len(x_units)})")
            # label each symbol column (1-9) just inside the plot, above the x-axis
            ymax = axes[1].get_ylim()[1]
            for symbol_idx in range(1, 10):
                axes[1].text(symbol_idx - 0.5, ymax * 0.02, str(symbol_idx),
                             ha="center", va="bottom", fontsize=9, color="dimgray")

            fig.suptitle(f"{participant} / panel {panel} - raw (uncorrected) fixations above dictionary boundary")
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"hist_raw_xy_{participant}_{panel}.png")
            if os.path.exists(save_path):
                # overwriting in place errors out on this SMB mount (OSError 89/9) -
                # delete-then-write avoids it.
                os.remove(save_path)
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
            saved_paths.append(save_path)
            print(f"Saved {save_path}")
        except Exception as e:
            print(f"FAILED {participant}/{panel}: {e}")

    return saved_paths


def plot_raw_fixation_xy_histograms_random_sample(main_data_path=DEFAULT_MAIN_DATA_PATH, n=10, groups=("HC", "pwMS"),
                                                    output_dir=None):
    """
    Convenience wrapper: picks n random participants (each with one random panel among
    their saved panels) and runs plot_raw_fixation_xy_histograms on just those pairs.
    """
    by_participant = {}
    for _, participant, panel in _iter_saved_gaze_panels(main_data_path, groups):
        by_participant.setdefault(participant, []).append(panel)

    chosen_participants = random.sample(list(by_participant.keys()), min(n, len(by_participant)))
    pairs = [(p, random.choice(by_participant[p])) for p in chosen_participants]
    print(f"Randomly selected pairs: {pairs}")
    return plot_raw_fixation_xy_histograms(main_data_path=main_data_path, pairs=pairs, groups=groups,
                                            output_dir=output_dir)


def estimate_whole_dictionary_drift_from_csv(main_data_path, participant, panel, eye=None, outlier_z_thresh=3.0,
                                              reference_ratio=None):
    """
    Reads the saved annotated gaze CSV directly and reproduces
    estimate_whole_dictionary_drift() without rebuilding a TrialManager: group samples into
    fixation blocks (_extract_fixation_block_ys), pool every block whose mean y is above the
    dictionary/grid boundary, drop y-outliers beyond outlier_z_thresh standard deviations,
    and compare the cleaned mean to a reference line - by default the dictionary's
    row-divider midline (DICTIONARY_MIDLINE_RATIO), or pass
    reference_ratio=DICTIONARY_SECOND_ROW_MID_RATIO to instead compare against the vertical
    center of the second (lower) dictionary row. Returns None if there are no
    dictionary-area fixation blocks at all (nothing to estimate from).
    """
    if reference_ratio is None:
        reference_ratio = DICTIONARY_MIDLINE_RATIO

    annotated = _load_saved_annotated_gaze(main_data_path, participant, panel, eye=eye)
    y = _extract_fixation_block_ys(annotated)

    dict_y = y[y < DICTIONARY_BOUNDARY_RATIO]
    if len(dict_y) == 0:
        return None

    mean_before = float(dict_y.mean())
    std_before = dict_y.std()
    cleaned = dict_y[np.abs(dict_y - mean_before) <= outlier_z_thresh * std_before] if std_before > 0 else dict_y
    mean_after = float(cleaned.mean()) if len(cleaned) else mean_before
    dy_ratio = mean_after - reference_ratio

    return {
        "y": y, "dy_ratio": dy_ratio,
        "mean_y_before_cleaning": mean_before, "mean_y_after_cleaning": mean_after,
        "n_fixations_before_cleaning": len(dict_y), "n_fixations_after_cleaning": len(cleaned),
    }


def _count_gap_and_negative_ratio(y, dy_ratio=0.0):
    """y: normalized fixation y values. dy_ratio>0 shifts dictionary-area (y<boundary) points by it."""
    y = y.copy()
    shift_mask = y < DICTIONARY_BOUNDARY_RATIO
    y[shift_mask] = y[shift_mask] - dy_ratio
    n_gap = int(((y >= SEARCHFINDER_DICT_RATIO) & (y < DICTIONARY_BOUNDARY_RATIO)).sum())
    n_negative = int((y < 0).sum())
    return n_gap, n_negative


def build_whole_dictionary_drift_table_from_csv(main_data_path=DEFAULT_MAIN_DATA_PATH, groups=("HC", "pwMS"), outlier_z_thresh=3.0,
                                                 reference_ratio=None):
    """
    Lightweight replacement for build_whole_dictionary_drift_table(): reads the already-saved
    annotated gaze CSVs directly instead of rebuilding a ParticipantGazeDataManager/TrialManager
    per participant/panel. Same output schema, so it's a drop-in for the calibration_correction_report
    plotting/summary functions (plot_whole_dictionary_drift_histogram, plot_gap_zone_fixation_histograms,
    plot_negative_y_histogram, summarize_whole_dictionary_drift). reference_ratio is forwarded to
    estimate_whole_dictionary_drift_from_csv (default: DICTIONARY_MIDLINE_RATIO).
    """
    rows = []
    for group, participant, panel in _iter_saved_gaze_panels(main_data_path, groups):
        row = {"group": group, "participant": participant, "panel": panel}
        try:
            estimate = estimate_whole_dictionary_drift_from_csv(main_data_path, participant, panel,
                                                                 outlier_z_thresh=outlier_z_thresh,
                                                                 reference_ratio=reference_ratio)
            if estimate is None:
                row["error"] = "no fixations above the dictionary/grid boundary"
                rows.append(row)
                continue

            n_gap_before, n_neg_before = _count_gap_and_negative_ratio(estimate["y"], dy_ratio=0.0)
            n_gap_after, n_neg_after = _count_gap_and_negative_ratio(estimate["y"], dy_ratio=estimate["dy_ratio"])

            row.update({
                "dy": estimate["dy_ratio"] * SCREEN_SIZE[0],
                "mean_y_before_cleaning": estimate["mean_y_before_cleaning"] * SCREEN_SIZE[0],
                "mean_y_after_cleaning": estimate["mean_y_after_cleaning"] * SCREEN_SIZE[0],
                "n_fixations_before_cleaning": estimate["n_fixations_before_cleaning"],
                "n_fixations_after_cleaning": estimate["n_fixations_after_cleaning"],
                "n_gap_zone_fixations_before_correction": n_gap_before,
                "n_gap_zone_fixations_after_correction": n_gap_after,
                "n_negative_y_fixations_before": n_neg_before,
                "n_negative_y_fixations_after": n_neg_after,
            })
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    return pd.DataFrame(rows)


def build_per_trial_gap_negative_table_from_csv(main_data_path=DEFAULT_MAIN_DATA_PATH,
                                                 calibration_qc_dir=DEFAULT_CALIBRATION_QC_DIR):
    """
    Gap-zone/negative-y comparison for the PER-TRIAL drift-correction method (item 3/4),
    computed the same lightweight way as build_whole_dictionary_drift_table_from_csv - no
    second TrialManager rebuild needed. The per-trial method's dy was already estimated by
    the expensive TrialManager-based sweep (drift_correction_qa.csv, built by
    build_drift_qa_table); this just re-applies that same y-shift to the saved annotated
    CSV and re-counts fixation blocks, exactly like the whole-dictionary methods do, so all
    three methods become directly comparable on the same gap-zone/negative-y metrics. Only
    dy matters here (dx shifts x, not y, so it doesn't affect these y-only metrics).
    """
    drift_qa_path = os.path.join(calibration_qc_dir, "drift_correction_qa.csv")
    drift_qa_df = pd.read_csv(drift_qa_path)

    rows = []
    for _, r in drift_qa_df.iterrows():
        row = {"group": r["group"], "participant": r["participant"], "panel": r["panel"]}
        if pd.isna(r.get("dy_before")):
            row["error"] = r["error"] if "error" in r and pd.notna(r["error"]) else "no per-trial drift estimate"
            rows.append(row)
            continue
        try:
            annotated = _load_saved_annotated_gaze(main_data_path, r["participant"], r["panel"])
            y = _extract_fixation_block_ys(annotated)
            dy_ratio = r["dy_before"] / SCREEN_SIZE[0]

            n_gap_before, n_neg_before = _count_gap_and_negative_ratio(y, dy_ratio=0.0)
            n_gap_after, n_neg_after = _count_gap_and_negative_ratio(y, dy_ratio=dy_ratio)

            row.update({
                "dy": r["dy_before"],
                "n_gap_zone_fixations_before_correction": n_gap_before,
                "n_gap_zone_fixations_after_correction": n_gap_after,
                "n_negative_y_fixations_before": n_neg_before,
                "n_negative_y_fixations_after": n_neg_after,
            })
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Item 5: video of the drift-corrected gaze, without rebuilding a ParticipantGazeDataManager
# ---------------------------------------------------------------------------------------
def _load_saved_annotated_gaze(main_data_path, participant, panel, eye=None):
    """
    Reads the per-panel gaze CSV the pipeline already saves at
    processing_results/{participant}/task_{panel}_fixation_{eye}.csv (written as a side
    effect of ParticipantGazeDataManager.annotate_gaze_events ->
    utils.generate_fixations_threshold_based - the eye suffix matches
    matched_data[panel][KEY_ANALYSIS_EYE], which is per-panel, not a single value for the
    whole participant/event). Falls back to the pre-eye-suffix legacy filename
    (task_{panel}_fixation.csv), and if no eye was requested, to whichever single
    eye-suffixed cache exists, so already-generated caches keep working.
    Columns: t, eye_horizontal, eye_vertical (normalized [0,1]), status, evt. Reusing this
    avoids re-loading the participant's raw .mat data just to render a QA video.
    """
    base_dir = os.path.join(main_data_path, "processing_results", participant)
    candidates = []
    if eye is not None:
        candidates.append(os.path.join(base_dir, f"task_{panel}_fixation_{eye.lower()}.csv"))
    candidates.append(os.path.join(base_dir, f"task_{panel}_fixation.csv"))
    csv_path = next((p for p in candidates if os.path.exists(p)), None)

    if csv_path is None and eye is None:
        matches = sorted(glob.glob(os.path.join(base_dir, f"task_{panel}_fixation_*.csv")))
        csv_path = matches[0] if matches else None

    if csv_path is None:
        raise FileNotFoundError(
            f"No saved annotated gaze CSV found for participant={participant!r} panel={panel!r} "
            f"eye={eye!r} under {base_dir}. Run the normal pipeline (e.g. TrialManager) at "
            f"least once for this participant/panel/eye first."
        )
    columns = [TIME_STAMP, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_VALID_STATUS, FIXATION_CSV_KEY_FIXATION]
    return pd.read_csv(csv_path)[columns]


def render_drift_corrected_gaze_video(main_data_path, participant, panel, dx, dy, boundary_y,
                                       task="SDMT", output_dir=None, target_fps=60, eye=None):
    """
    Renders the drift-corrected gaze path over the panel image. Only x/y in the dictionary
    (upper) region are shifted - by build_drift_correction_fn, the exact same closure the
    real pipeline uses - everything else (t, evt) is untouched, matching how the fix is
    meant to apply to the saved CSV.
    """
    annotated = _load_saved_annotated_gaze(main_data_path, participant, panel, eye=eye)
    img = cv2.imread(_panel_image_path(main_data_path, task, panel))
    img_resized, gaze_scaled = prepare_image_and_gaze(img, annotated)
    gaze_corrected = build_drift_correction_fn(dx, dy, boundary_y)(gaze_scaled)

    eye_x = gaze_corrected[FIXATION_CSV_KEY_EYE_H].to_numpy()
    eye_y = gaze_corrected[FIXATION_CSV_KEY_EYE_V].to_numpy()
    fixation = gaze_corrected[FIXATION_CSV_KEY_FIXATION].to_numpy()
    times = gaze_corrected[TIME_STAMP].to_numpy() / SECONDS_TO_MICROSECOND_FACTOR

    output_dir = output_dir or os.path.join(DEFAULT_CALIBRATION_QC_DIR, "videos")
    os.makedirs(output_dir, exist_ok=True)
    eye_suffix = f"_{eye.lower()}" if eye else ""
    video_path = os.path.join(output_dir, f"{participant}_{panel}{eye_suffix}_drift_corrected.mp4")

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
    VIDEO_PARTICIPANT = "LD965"  # e.g. "AG562"; leave None to run the QA sweep instead
    VIDEO_PANEL = "a3"  # e.g. "l3"; None renders every panel available for the participant

    if VIDEO_PARTICIPANT is not None:
        create_gaze_correction_video(VIDEO_PARTICIPANT, panel=VIDEO_PANEL,
                                      main_data_path=MAIN_DATA_PATH, calibration_qc_dir=OUTPUT_DIR)
        sys.exit(0)

    glossary_path = write_glossary(OUTPUT_DIR)
    print(f"Glossary written to {glossary_path}")

    calibration_df = build_calibration_table(MAIN_DATA_PATH)
    calibration_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_quality.csv"), index=False)
    print(f"Calibration quality table: {len(calibration_df)} rows")

    dominant_eye_df = build_dominant_eye_table(calibration_df)
    dominant_eye_df.to_csv(os.path.join(OUTPUT_DIR, "dominant_eye_check.csv"), index=False)
    if len(dominant_eye_df):
        pct_match = 100.0 * dominant_eye_df["dominant_matches_best"].mean()
        print(f"Dominant eye matches best-calibrated eye in {pct_match:.1f}% of "
              f"{len(dominant_eye_df)} participant-panels")

    drift_qa_df = build_drift_qa_table(MAIN_DATA_PATH)
    drift_qa_df.to_csv(os.path.join(OUTPUT_DIR, "drift_correction_qa.csv"), index=False)
    print(f"Drift correction QA table: {len(drift_qa_df)} rows")
    if drift_qa_df["n_negative_y_fixations_after"].fillna(0).gt(0).any():
        n_overshoot = int(drift_qa_df["n_negative_y_fixations_after"].fillna(0).gt(0).sum())
        print(f"WARNING: {n_overshoot} panels have fixations pushed to y<0 (off-screen) by the per-trial correction")

    whole_dict_df = build_whole_dictionary_drift_table(MAIN_DATA_PATH)
    whole_dict_df.to_csv(os.path.join(OUTPUT_DIR, "whole_dictionary_drift_qa.csv"), index=False)
    print(f"Whole-dictionary drift table: {len(whole_dict_df)} rows")
    if whole_dict_df["n_negative_y_fixations_after"].fillna(0).gt(0).any():
        n_overshoot = int(whole_dict_df["n_negative_y_fixations_after"].fillna(0).gt(0).sum())
        print(f"WARNING: {n_overshoot} panels have fixations pushed to y<0 (off-screen) by the whole-dictionary correction")

"""
Per-panel sweep for two population/eye-selection variants, each producing 5 metrics:

  1. pct_out_of_range   - % of ALL raw samples (fixation+saccade combined) with x<0, x>1,
                          y<0, or y>1 (computed on the untouched raw mat values, before any
                          NaN substitution)
  2. space_coverage_x   - max-min span of x among raw samples, AFTER substituting x<0 with
                          NaN (no upper-bound cap - a genuine/corrupted x>1 excursion still
                          extends the span past 1.0)
  3. space_coverage_y   - same, for y
  4. pct_fixation       - % of the panel's samples classified FIXATION_IDX by the normal
                          threshold-based annotate_gaze_events pipeline
  5. pct_saccade        - % classified SACCADE_IDX

Variant "nan_only_keep_tobii": population = every participant except DONTUSE (Tobii_Sucks
KEPT); eye = whichever eye has the better (lower) calibration accuracy for that EVENT,
regardless of the accuracy value itself (no accuracy threshold at all); a panel is included
iff that eye's NaN ratio for that panel is < 10% (uses ParticipantGazeDataManager's existing
explicit_override/analysis_eye path, which already enforces exactly this).

Variant "regular": the actual current pipeline as-is - Tobii_Sucks AND DONTUSE excluded,
per-panel joint accuracy<2.5deg + NaN-ratio<10% eye selection (ParticipantGazeDataManager
with no override).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from calibration_drift_qa import _iter_subject_dirs
from constants import FIXATION_CSV_KEY_FIXATION, FIXATION_IDX, SACCADE_IDX, KEY_ANALYSIS_EYE
from participant_gaze_data_manager import (
    ParticipantGazeDataManager, extract_last_calibration_message, parse_calibration_quality_message,
)

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
BASE_DIR = "/Volumes/ramot/Noam_M/calibration_qc/space_coverage_analysis"
RAW_DIR = os.path.join(BASE_DIR, "raw")                        # nan_only_keep_tobii variant
EXCLUDED_DIR = os.path.join(BASE_DIR, "excluded_by_acc_and_nan")  # regular (current pipeline) variant


def _iter_dontuse_only(main_data_path, groups=("HC", "pwMS")):
    """Same as calibration_drift_qa._iter_subject_dirs but WITHOUT the Tobii_Sucks filter -
    only DONTUSE-suffixed folders are skipped."""
    for group in groups:
        for subject_dir in sorted(glob.glob(os.path.join(main_data_path, group, "*"))):
            if not os.path.isdir(subject_dir):
                continue
            if "DONTUSE" in os.path.basename(subject_dir).upper():
                continue
            yield group, subject_dir


def _best_eye_per_event(dummy, subject_dir):
    """recording_date -> 'l'/'r' (whichever has lower accuracy, regardless of its value) or
    None if no calibration message could be parsed for that event."""
    task_dir = os.path.join(subject_dir, "SDMT")
    if not os.path.isdir(task_dir):
        return {}
    mat_files = glob.glob(os.path.join(task_dir, "*.mat"))
    mat_files = [f for f in mat_files if "run" in os.path.split(f)[1].lower()]
    try:
        loaded_mats, _ = dummy._load_and_dedupe_run_files(mat_files)
    except Exception:
        return {}
    result = {}
    for mat in loaded_mats:
        found = extract_last_calibration_message(mat["messages"])
        calib = parse_calibration_quality_message(found[1]) if found is not None else None
        l_acc = (calib["left"]["average"] or {}).get("acc") if calib else None
        r_acc = (calib["right"]["average"] or {}).get("acc") if calib else None
        if l_acc is None and r_acc is None:
            eye = None
        elif r_acc is None or (l_acc is not None and l_acc <= r_acc):
            eye = "l"
        else:
            eye = "r"
        try:
            date = dummy.get_creation_time(mat)
        except Exception:
            date = None
        result[date] = eye
    return result


def _raw_panel_xy(dummy, subject_dir, group, panel, eye, target_date):
    """Direct re-extraction of TRULY raw (untouched, no out-of-range policy applied) x/y for
    one participant/panel/eye, matched by recording_date - mirrors the approach already
    verified in render_padded_gaze_video.py / build_panel_diagnostics.py."""
    task_dir = os.path.join(subject_dir, "SDMT")
    mat_files = glob.glob(os.path.join(task_dir, "*.mat"))
    mat_files = [f for f in mat_files if "run" in os.path.split(f)[1].lower()]
    loaded_mats, _ = dummy._load_and_dedupe_run_files(mat_files)
    for mat in loaded_mats:
        try:
            date = dummy.get_creation_time(mat)
        except Exception:
            date = None
        if date != target_date:
            continue
        messages = mat["messages"]
        left_gaze = mat["data"].gaze.left.gazePoint.onDisplayArea
        right_gaze = mat["data"].gaze.right.gazePoint.onDisplayArea
        tobi_ts = mat["data"].gaze.systemTimeStamp
        gaze = right_gaze if eye == "r" else left_gaze

        panel_indices, break_indices = dummy.break_mat_into_pannels(messages)
        panel_start_times = [messages[i][0] for i in panel_indices]
        break_start_times = [messages[i][0] for i in break_indices]
        task_data = mat["task_data"].__dict__
        codes = [(n[-2:]).replace("_", "").lower() for n in list(task_data.keys())[1::2]]
        for i in range(3):
            if i >= len(codes) or codes[i] != panel:
                continue
            indices = np.where((tobi_ts > panel_start_times[i]) & (tobi_ts < break_start_times[i]))[0]
            return gaze[0, indices], gaze[1, indices]
    return None, None


def _space_coverage_pct(x, y):
    """max-min span (as a % of the [0,1] screen extent) after substituting negative values
    with NaN - no upper-bound cap, so a genuine/corrupted >1 excursion still extends it."""
    x_clean = x.copy()
    x_clean[x_clean < 0] = np.nan
    y_clean = y.copy()
    y_clean[y_clean < 0] = np.nan
    space_x = 100.0 * float(np.nanmax(x_clean) - np.nanmin(x_clean))
    space_y = 100.0 * float(np.nanmax(y_clean) - np.nanmin(y_clean))
    return space_x, space_y


def _panel_metrics(raw_x, raw_y, annotated):
    pct_out_of_range = 100.0 * np.mean((raw_x < 0) | (raw_x > 1) | (raw_y < 0) | (raw_y > 1))
    space_x, space_y = _space_coverage_pct(raw_x, raw_y)

    evt = annotated[FIXATION_CSV_KEY_FIXATION].to_numpy()
    n = len(evt)
    fix_mask = evt == FIXATION_IDX
    pct_fixation = 100.0 * np.sum(fix_mask) / n
    pct_saccade = 100.0 * np.sum(evt == SACCADE_IDX) / n

    # Same span computation, restricted to fixation-labeled samples only (raw_x/raw_y are
    # index-aligned with annotated - verified: both cover the same panel time window with
    # identical sample count/order).
    if fix_mask.any():
        space_x_fix, space_y_fix = _space_coverage_pct(raw_x[fix_mask], raw_y[fix_mask])
    else:
        space_x_fix, space_y_fix = np.nan, np.nan

    return pct_out_of_range, space_x, space_y, pct_fixation, pct_saccade, space_x_fix, space_y_fix


def _raw_sample_rows(group, participant, panel, eye, raw_x, raw_y, annotated):
    """Per-sample (x, y, evt) rows for this panel - saved alongside the aggregate metrics so
    a FUTURE metric redefinition (e.g. "only saccades", "only within some x/y band") can be
    computed straight from the saved table instead of re-sweeping every participant again."""
    evt = annotated[FIXATION_CSV_KEY_FIXATION].to_numpy().astype(np.int8)
    return pd.DataFrame({
        "group": group, "participant": participant, "panel": panel, "eye": eye,
        "x": raw_x.astype(np.float32), "y": raw_y.astype(np.float32), "evt": evt,
    })


def sweep_nan_only_keep_tobii():
    dummy = ParticipantGazeDataManager.__new__(ParticipantGazeDataManager)
    dummy.task_validation_filter_param = "run"
    dummy.task = "SDMT"

    rows = []
    raw_chunks = []
    n_participants = 0
    for group, subject_dir in _iter_dontuse_only(MAIN_DATA_PATH, ("HC", "pwMS")):
        if "SDMT" not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)
        best_eye = _best_eye_per_event(dummy, subject_dir)
        needed_eyes = {e for e in best_eye.values() if e is not None}
        if not needed_eyes:
            continue
        n_participants += 1

        for eye in needed_eyes:
            try:
                sd = ParticipantGazeDataManager(subject_dir, MAIN_DATA_PATH, "SDMT", group, analysis_eye=eye)
            except Exception as e:
                print(f"  FAILED {participant} eye={eye}: {e}")
                continue
            for panel in list(sd.matched_data.keys()):
                info = sd.matched_data[panel]
                date = info["recording_date"]
                if best_eye.get(date) != eye:
                    continue  # this event's best eye is the OTHER eye - handled in that pass
                try:
                    annotated = sd.annotate_gaze_events(panel)
                    raw_x, raw_y = _raw_panel_xy(dummy, subject_dir, group, panel, eye, date)
                    if raw_x is None:
                        raise ValueError("raw re-extraction failed to match event")
                    metrics = _panel_metrics(raw_x, raw_y, annotated)
                    raw_chunks.append(_raw_sample_rows(group, participant, panel, eye, raw_x, raw_y, annotated))
                except Exception as e:
                    print(f"  FAILED {participant}/{panel} eye={eye}: {e}")
                    continue
                rows.append({"group": group, "participant": participant, "panel": panel, "eye": eye,
                             "pct_out_of_range": metrics[0], "space_coverage_x": metrics[1],
                             "space_coverage_y": metrics[2], "pct_fixation": metrics[3], "pct_saccade": metrics[4],
                             "space_coverage_x_fixation_only": metrics[5], "space_coverage_y_fixation_only": metrics[6]})

        if n_participants % 15 == 0:
            print(f"... {n_participants} participants done, {len(rows)} panels so far")

    return pd.DataFrame(rows), pd.concat(raw_chunks, ignore_index=True) if raw_chunks else pd.DataFrame()


def sweep_regular():
    rows = []
    raw_chunks = []
    n_participants = 0
    dummy = ParticipantGazeDataManager.__new__(ParticipantGazeDataManager)
    dummy.task_validation_filter_param = "run"
    dummy.task = "SDMT"

    for group, subject_dir in _iter_subject_dirs(MAIN_DATA_PATH, ("HC", "pwMS")):
        if "SDMT" not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)
        try:
            sd = ParticipantGazeDataManager(subject_dir, MAIN_DATA_PATH, "SDMT", group)
        except Exception as e:
            print(f"  FAILED {participant}: {e}")
            continue
        n_participants += 1

        for panel in list(sd.matched_data.keys()):
            info = sd.matched_data[panel]
            eye = info[KEY_ANALYSIS_EYE]
            date = info["recording_date"]
            try:
                annotated = sd.annotate_gaze_events(panel)
                raw_x, raw_y = _raw_panel_xy(dummy, subject_dir, group, panel, eye, date)
                if raw_x is None:
                    raise ValueError("raw re-extraction failed to match event")
                metrics = _panel_metrics(raw_x, raw_y, annotated)
                raw_chunks.append(_raw_sample_rows(group, participant, panel, eye, raw_x, raw_y, annotated))
            except Exception as e:
                print(f"  FAILED {participant}/{panel}: {e}")
                continue
            rows.append({"group": group, "participant": participant, "panel": panel, "eye": eye,
                         "pct_out_of_range": metrics[0], "space_coverage_x": metrics[1],
                         "space_coverage_y": metrics[2], "pct_fixation": metrics[3], "pct_saccade": metrics[4],
                         "space_coverage_x_fixation_only": metrics[5], "space_coverage_y_fixation_only": metrics[6]})

        if n_participants % 15 == 0:
            print(f"... {n_participants} participants done, {len(rows)} panels so far")

    return pd.DataFrame(rows), pd.concat(raw_chunks, ignore_index=True) if raw_chunks else pd.DataFrame()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "nan_only_keep_tobii"
    if mode == "nan_only_keep_tobii":
        df, raw_df = sweep_nan_only_keep_tobii()
        out_dir = RAW_DIR
    elif mode == "regular":
        df, raw_df = sweep_regular()
        out_dir = EXCLUDED_DIR
    else:
        raise ValueError(f"unknown mode {mode!r}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "space_coverage_data.csv")
    df.to_csv(out_path, index=False)
    print(f"DONE: {len(df)} panels saved to {out_path}")

    raw_path = os.path.join(out_dir, "raw_samples_with_fixation_labels.parquet")
    raw_df.to_parquet(raw_path)
    print(f"DONE: {len(raw_df)} raw samples (x, y, evt per participant/panel/eye) saved to {raw_path} "
          f"- future metric changes can be computed from this without re-sweeping")

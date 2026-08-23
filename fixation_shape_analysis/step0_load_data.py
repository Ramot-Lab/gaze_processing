"""
Step 0 - build the fixation-level table this whole pipeline runs on.

Input on disk (per participant, per panel/trial): a per-SAMPLE gaze CSV written by the main
processing pipeline (participant_gaze_data_manager.generate_fixations_threshold_based),
columns [t, eye_horizontal, eye_vertical, status, evt] with evt==1 fixation / evt==2 saccade
samples at 600Hz, normalized [0,1] screen coordinates.

This step groups consecutive evt==1 samples into fixation BLOCKS (identical logic to
FixationHandler.process_fixations / calibration_drift_qa._extract_fixation_block_xy) and
collapses each block to one row: participant_id, trial_id, timestamp (block start t),
x (mean eye_horizontal), y (mean eye_vertical). That one-row-per-fixation table is what
steps 1-4 consume.

Per config.FILTER_ABOVE_DICTIONARY_BOUNDARY, blocks are also restricted to the dictionary
(symbols/numbers) area above the digit-sequence text - i.e. y < DICTIONARY_BOUNDARY_RATIO,
the same cutoff calibration_drift_qa.py uses - before anything else runs, so every step
downstream only ever sees "above the text area" fixations.

Re-run this alone with `python -m fixation_shape_analysis.step0_load_data` any time the
underlying gaze CSVs change; everything downstream just reads its output CSVs.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from calibration_drift_qa import DICTIONARY_BOUNDARY_RATIO

from . import config

_FIXATION_CSV_RE = re.compile(r"^task_(.+?)_fixation\.csv$")


def _iter_fixation_csv_paths(main_data_path):
    """(participant_id, trial_id, csv_path) for every dominant-eye fixation CSV on disk."""
    processing_results_dir = os.path.join(main_data_path, config.PROCESSING_RESULTS_DIRNAME)
    for participant_dir in sorted(glob.glob(os.path.join(processing_results_dir, "*"))):
        participant_id = os.path.basename(participant_dir)
        if "DONTUSE" in participant_id.upper():
            continue
        for csv_path in sorted(glob.glob(os.path.join(participant_dir, "task_*_fixation.csv"))):
            match = _FIXATION_CSV_RE.match(os.path.basename(csv_path))
            if match:
                yield participant_id, match.group(1), csv_path


def _extract_fixation_blocks(samples_df):
    """
    Group consecutive valid evt==FIXATION_IDX samples into blocks, closing a block on the
    next evt==SACCADE_IDX sample (or at end of recording). Returns one row per block:
    timestamp = block's first sample time, x/y = mean eye_horizontal/eye_vertical over the block.
    """
    valid = samples_df[samples_df[config.COL_STATUS] != 0]
    t = valid[config.COL_TIME].to_numpy()
    x = valid[config.COL_EYE_H].to_numpy()
    y = valid[config.COL_EYE_V].to_numpy()
    evt = valid[config.COL_EVT].to_numpy()

    rows = []
    cur_t, cur_x, cur_y = [], [], []
    for tt, xx, yy, e in zip(t, x, y, evt):
        if e == config.FIXATION_IDX:
            cur_t.append(tt)
            cur_x.append(xx)
            cur_y.append(yy)
        elif e == config.SACCADE_IDX and cur_t:
            rows.append((cur_t[0], np.mean(cur_x), np.mean(cur_y)))
            cur_t, cur_x, cur_y = [], [], []
    if cur_t:
        rows.append((cur_t[0], np.mean(cur_x), np.mean(cur_y)))

    return pd.DataFrame(rows, columns=["timestamp", "x", "y"])


def build_raw_fixation_table(main_data_path=config.DEFAULT_MAIN_DATA_PATH, limit_units=None,
                              filter_above_dictionary_boundary=config.FILTER_ABOVE_DICTIONARY_BOUNDARY):
    """
    Scans every participant/trial fixation CSV under main_data_path and returns the combined
    fixation-level table (participant_id, trial_id, timestamp, x, y), unfiltered by fixation
    count (that's filter_by_min_fixations below).

    If filter_above_dictionary_boundary (default True, see config.FILTER_ABOVE_DICTIONARY_BOUNDARY),
    blocks below the dictionary/text boundary (y >= DICTIONARY_BOUNDARY_RATIO - i.e. fixations on
    the symbol-sequence search text, not the symbol/digit dictionary) are dropped here, before any
    other step sees them.

    limit_units: optional cap on the number of (participant_id, trial_id) units scanned, for
    quick smoke-testing the pipeline before a full run.
    """
    pairs = list(_iter_fixation_csv_paths(main_data_path))
    if limit_units is not None:
        pairs = pairs[:limit_units]

    frames = []
    n_blocks_total = 0
    for participant_id, trial_id, csv_path in pairs:
        samples_df = pd.read_csv(csv_path, index_col=0)
        blocks = _extract_fixation_blocks(samples_df)
        if blocks.empty:
            continue

        n_blocks_total += len(blocks)
        if filter_above_dictionary_boundary:
            blocks = blocks[blocks["y"] < DICTIONARY_BOUNDARY_RATIO]
        if blocks.empty:
            continue

        blocks.insert(0, "trial_id", trial_id)
        blocks.insert(0, "participant_id", participant_id)
        frames.append(blocks)

    if not frames:
        raise RuntimeError(f"No fixation CSVs found under {main_data_path}")

    fixation_df = pd.concat(frames, ignore_index=True)
    fixation_df["unit_id"] = fixation_df["participant_id"] + "_" + fixation_df["trial_id"]

    if filter_above_dictionary_boundary:
        n_kept = len(fixation_df)
        print(f"Step 0: restricted to fixations above the dictionary/text boundary "
              f"(y < {DICTIONARY_BOUNDARY_RATIO:.4f}) - {n_kept}/{n_blocks_total} fixation blocks kept "
              f"({100 * n_kept / n_blocks_total:.1f}%).")

    return fixation_df


def filter_by_min_fixations(fixation_df, min_fixations=config.MIN_FIXATIONS_PER_UNIT):
    """
    Excludes participant-panel units with fewer than min_fixations fixation rows (too few for
    stable skewness/kurtosis/dip-test/GMM estimation). Returns (included_df, excluded_summary_df).
    """
    counts = fixation_df.groupby(["participant_id", "trial_id", "unit_id"]).size().rename("n_fixations").reset_index()
    counts["excluded"] = counts["n_fixations"] < min_fixations
    counts["reason"] = np.where(
        counts["excluded"],
        f"n_fixations < {min_fixations} (min_fixations_per_unit)",
        "",
    )

    kept_units = set(counts.loc[~counts["excluded"], "unit_id"])
    included_df = fixation_df[fixation_df["unit_id"].isin(kept_units)].reset_index(drop=True)
    excluded_summary_df = counts.sort_values(["participant_id", "trial_id"]).reset_index(drop=True)
    return included_df, excluded_summary_df


def run_step0(main_data_path=config.DEFAULT_MAIN_DATA_PATH, min_fixations=config.MIN_FIXATIONS_PER_UNIT,
              limit_units=None, filter_above_dictionary_boundary=config.FILTER_ABOVE_DICTIONARY_BOUNDARY):
    config.ensure_output_dirs()

    raw_df = build_raw_fixation_table(main_data_path, limit_units=limit_units,
                                       filter_above_dictionary_boundary=filter_above_dictionary_boundary)
    included_df, excluded_summary_df = filter_by_min_fixations(raw_df, min_fixations=min_fixations)

    raw_df.to_csv(config.table_path("step0_fixation_data_raw.csv"), index=False)
    included_df.to_csv(config.table_path("step0_fixation_data_included.csv"), index=False)
    excluded_summary_df.to_csv(config.table_path("step0_unit_fixation_counts.csv"), index=False)

    n_units_total = excluded_summary_df["unit_id"].nunique()
    n_units_excluded = int(excluded_summary_df["excluded"].sum())
    n_participants_total = raw_df["participant_id"].nunique()
    n_participants_remaining = included_df["participant_id"].nunique() if len(included_df) else 0

    print(f"Step 0: {n_units_total} participant-panel units found "
          f"({n_participants_total} participants), "
          f"{n_units_excluded} excluded for < {min_fixations} fixations, "
          f"{n_units_total - n_units_excluded} kept "
          f"({n_participants_remaining} participants remaining).")

    return included_df, excluded_summary_df


if __name__ == "__main__":
    run_step0()

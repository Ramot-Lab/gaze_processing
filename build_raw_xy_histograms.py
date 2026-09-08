"""
Sweeps every valid participant/panel under the CURRENT preprocessing pipeline (per-panel
accuracy+NaN-ratio eye selection, decision 2026-09-07; out-of-range values left as-is -
OUT_OF_RANGE_VALUES_METHOD="as_is" in constants.py, nothing converted to NaN) and pools:
  - sample-level x/y: every raw per-sample (fixation-labeled, evt==FIXATION_IDX) row's
    eye_horizontal/eye_vertical value
  - fixation-block-level x/y: one (mean x, mean y) per fixation BLOCK (consecutive
    same-classification samples between saccades), via calibration_drift_qa._extract_fixation_block_xy

Uses ParticipantGazeDataManager.annotate_gaze_events() directly (not TrialManager) since
only fixation labels are needed here, not ROI/Search/Trial extraction - cheaper per panel.

Saves two tables (group, participant, panel, x, y) as parquet so this sweep never needs to
be rerun just to slice/replot the pooled data differently - and 4 histograms (samples vs
fixation blocks, x and y).
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from participant_gaze_data_manager import ParticipantGazeDataManager
from calibration_drift_qa import _iter_subject_dirs, _extract_fixation_block_xy
from constants import FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_CSV_KEY_FIXATION, FIXATION_IDX

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc"
LOCAL_BACKUP_DIR = "/private/tmp/claude-501/-Users-noammizrachi-Documents-Msc-Ramot-lab-data-analysis-gaze-processing/5f2311ea-90b7-4b85-b25a-715d97b7d8df/scratchpad"


def _safe_write(write_fn, network_path, local_backup_path=None):
    """The network mount has dropped mid-sweep before - write to local disk first (always
    succeeds), then try to also place the same file on the network qc folder."""
    if local_backup_path:
        write_fn(local_backup_path)
    try:
        os.makedirs(os.path.dirname(network_path), exist_ok=True)
        write_fn(network_path)
        print(f"saved {network_path}")
    except OSError as e:
        print(f"WARNING: could not write {network_path} ({e}); local backup at {local_backup_path} is intact")


def main():
    sample_rows = []  # (group, participant, panel, x, y) - one row per fixation-labeled sample
    block_rows = []   # (group, participant, panel, x, y) - one row per fixation block

    n_participants = 0
    n_panels = 0
    n_panel_errors = 0
    t_start = time.time()

    for group, subject_dir in _iter_subject_dirs(MAIN_DATA_PATH, ("HC", "pwMS")):
        if "SDMT" not in os.listdir(subject_dir):
            continue
        participant = os.path.basename(subject_dir)
        try:
            sd = ParticipantGazeDataManager(subject_dir, MAIN_DATA_PATH, "SDMT", group)
        except Exception as e:
            print(f"skip participant {participant}: {e}")
            continue
        n_participants += 1

        for panel in list(sd.matched_data.keys()):
            try:
                df = sd.annotate_gaze_events(panel)
            except Exception as e:
                n_panel_errors += 1
                print(f"  {participant}/{panel} failed: {e}")
                continue
            n_panels += 1

            fix = df[df[FIXATION_CSV_KEY_FIXATION] == FIXATION_IDX]
            sample_rows.append(pd.DataFrame({
                "group": group, "participant": participant, "panel": panel,
                "x": fix[FIXATION_CSV_KEY_EYE_H].to_numpy(),
                "y": fix[FIXATION_CSV_KEY_EYE_V].to_numpy(),
            }))

            bx, by = _extract_fixation_block_xy(df)
            block_rows.append(pd.DataFrame({
                "group": group, "participant": participant, "panel": panel, "x": bx, "y": by,
            }))

        if n_participants % 15 == 0:
            elapsed = time.time() - t_start
            print(f"... {n_participants} participants, {n_panels} panels done, "
                  f"{n_panel_errors} panel errors, {elapsed:.0f}s elapsed")

    print(f"DONE sweeping: {n_participants} participants, {n_panels} panels, "
          f"{n_panel_errors} panel errors, {time.time()-t_start:.0f}s total")

    sample_df = pd.concat(sample_rows, ignore_index=True)
    block_df = pd.concat(block_rows, ignore_index=True)

    print(f"sample_df: {len(sample_df)} rows, x range [{sample_df['x'].min():.4f}, {sample_df['x'].max():.4f}], "
          f"y range [{sample_df['y'].min():.4f}, {sample_df['y'].max():.4f}]")
    print(f"block_df: {len(block_df)} rows, x range [{block_df['x'].min():.4f}, {block_df['x'].max():.4f}], "
          f"y range [{block_df['y'].min():.4f}, {block_df['y'].max():.4f}]")

    _safe_write(sample_df.to_parquet, os.path.join(OUTPUT_DIR, "raw_fixation_samples_xy.parquet"),
                os.path.join(LOCAL_BACKUP_DIR, "raw_fixation_samples_xy.parquet"))
    _safe_write(block_df.to_parquet, os.path.join(OUTPUT_DIR, "fixation_blocks_xy.parquet"),
                os.path.join(LOCAL_BACKUP_DIR, "fixation_blocks_xy.parquet"))

    def _plot_pair(values_samples, values_blocks, axis_name, color, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, values, label in [(axes[0], values_samples, "raw samples"), (axes[1], values_blocks, "fixation blocks")]:
            ax.hist(values, bins=150, color=color, edgecolor="k", alpha=0.75)
            ax.axvline(0, color="red", linestyle="--", linewidth=1)
            ax.axvline(1, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{axis_name} - {label} (n={len(values)})")
            ax.set_xlabel(f"eye_{'horizontal' if axis_name == 'X' else 'vertical'} (normalized, as-is - no NaN cleaning)")
        axes[0].set_ylabel("count")
        fig.suptitle(f"{axis_name}: fixation-labeled raw samples vs fixation blocks - current preprocessing, all values as-is")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print("saved", save_path)
        plt.close()

    _plot_pair(sample_df["x"], block_df["x"], "X", "steelblue", os.path.join(OUTPUT_DIR, "hist_x_samples_vs_blocks_asis.png"))
    _plot_pair(sample_df["y"], block_df["y"], "Y", "darkorange", os.path.join(OUTPUT_DIR, "hist_y_samples_vs_blocks_asis.png"))

    print("ALL DONE")


if __name__ == "__main__":
    main()

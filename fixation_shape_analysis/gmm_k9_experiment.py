"""
One-off follow-up to Step 3: the main pipeline caps the GMM search at k=1..3 (per the
original spec), so its "multimodal" label just means "k=3 won," not "found 3 modes" - it
can never surface the ~9 modes you'd expect in x if fixations cluster on the SDMT
dictionary's 9 symbol/digit columns. This reruns Step 3 with the search extended to
k=1..9 for both axes, so k=3 is no longer an artificial ceiling and we can actually see
whether x wants ~9 components and y still prefers ~2-3.

Reuses the already-computed Step 0/1 tables (fixation data + moments) from the main run -
only Step 3's GMM fit is redone, with the higher cap, since dip test/BC/moments don't
depend on k.

Output kept separate from the main pipeline's outputs/ (different k, not comparable
row-for-row) under calibration_qc/, alongside the other exploratory gaze-distribution
diagnostics already there (e.g. raw_xy_histograms/).

Run with `python -m fixation_shape_analysis.gmm_k9_experiment`.
"""

import os

import pandas as pd

from . import config
from .step3_gmm import compute_gmm_characterization, plot_gmm_diagnostic

MAX_COMPONENTS = 9

# calibration_qc already holds the other exploratory gaze-distribution diagnostics
# (raw_xy_histograms/, etc.) - same "/Volumes/Noam_M/..." share this machine mounts under
# "/Volumes/ramot/Noam_M/...", see config.DEFAULT_MAIN_DATA_PATH's comment.
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc/gmm_k9_diagnostics"
OUTPUT_TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def run_gmm_k9_experiment(max_components=MAX_COMPONENTS, make_plots=True):
    os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    fixation_df = pd.read_csv(config.table_path("step0_fixation_data_included.csv"))
    moments_df = pd.read_csv(config.table_path("step1_movment_distribution_analysis.csv"))

    gmm_df = compute_gmm_characterization(fixation_df, moments_df, max_components=max_components)

    if make_plots:
        for _, gmm_row in gmm_df.iterrows():
            values = fixation_df.loc[
                (fixation_df["participant_id"] == gmm_row["participant_id"]) &
                (fixation_df["trial_id"] == gmm_row["trial_id"]), gmm_row["axis"]
            ].to_numpy()
            filename = f"{gmm_row['participant_id']}_{gmm_row['trial_id']}_{gmm_row['axis']}_gmm_fit_k{max_components}.png"
            plot_gmm_diagnostic(values, gmm_row, os.path.join(OUTPUT_PLOTS_DIR, filename))

    gmm_df = gmm_df.drop(columns=["_model"])
    table_path = os.path.join(OUTPUT_TABLES_DIR, f"gmm_characterization_k{max_components}.csv")
    gmm_df.to_csv(table_path, index=False)

    print(f"GMM k=1..{max_components} experiment: fit {len(gmm_df)} participant-panel/axis rows.\n"
          f"Selected-k distribution:\n{gmm_df.groupby(['axis', 'selected_k']).size()}\n"
          f"Wrote {table_path}")
    return gmm_df


if __name__ == "__main__":
    run_gmm_k9_experiment()

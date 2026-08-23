"""
Shared configuration for the fixation-shape-analysis pipeline (steps 0-4).

Each step module can be re-run independently as long as the CSVs the earlier steps wrote
under OUTPUT_TABLES_DIR are still on disk - see run_pipeline.py for the full order.
"""

import os

# ---------------------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------------------
# Same tree calibration_drift_qa.DEFAULT_MAIN_DATA_PATH points at, just resolved to how this
# machine actually has the share mounted (/Volumes/Noam_M/... isn't mounted here, but the
# same folder appears under /Volumes/ramot/Noam_M/...).
DEFAULT_MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
PROCESSING_RESULTS_DIRNAME = "processing_results"

# Only the no-suffix fixation CSVs (dominant eye, the standard analysis pipeline's output).
# task_{trial_id}_fixation_l.csv / _r.csv are eye-override reruns for QA and are intentionally
# excluded here.
FIXATION_CSV_SUFFIX = "_fixation.csv"

# Per-sample CSV column names (see FixationHandler.py / constants.py)
COL_TIME = "t"
COL_EYE_H = "eye_horizontal"
COL_EYE_V = "eye_vertical"
COL_STATUS = "status"
COL_EVT = "evt"
FIXATION_IDX = 1
SACCADE_IDX = 2

# Only analyze fixations above the text/search-grid area, i.e. in the dictionary
# (symbols/numbers) region of the panel - not the digit-sequence text below it. Same
# y < DICTIONARY_BOUNDARY_RATIO cutoff calibration_drift_qa.py uses (imported from there in
# step0_load_data.py so the boundary stays a single source of truth).
FILTER_ABOVE_DICTIONARY_BOUNDARY = True

# ---------------------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------------------
BASE_OUTPUT_DIR = "/Volumes/ramot/Noam_M/fix_coordinates_analysis"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "outputs")
OUTPUT_TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
OUTPUT_PLOTS_PER_PARTICIPANT_DIR = os.path.join(OUTPUT_PLOTS_DIR, "per_participant")
OUTPUT_PLOTS_GROUP_DIR = os.path.join(OUTPUT_DIR, "plots", "group")
OUTPUT_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary.md")


def ensure_output_dirs():
    for d in (OUTPUT_TABLES_DIR, OUTPUT_PLOTS_GROUP_DIR, OUTPUT_PLOTS_PER_PARTICIPANT_DIR):
        os.makedirs(d, exist_ok=True)


def table_path(filename):
    return os.path.join(OUTPUT_TABLES_DIR, filename)


def group_plot_path(filename):
    return os.path.join(OUTPUT_PLOTS_GROUP_DIR, filename)


def per_participant_plot_path(filename):
    return os.path.join(OUTPUT_PLOTS_PER_PARTICIPANT_DIR, filename)


# ---------------------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------------------
# Unit of analysis is (participant_id, trial_id) - i.e. one participant-panel "session" -
# not the participant pooled across panels. See unit_id in step0_load_data.py.
MIN_FIXATIONS_PER_UNIT = 30  # configurable exclusion threshold (Step 0)

AXES = ("x", "y")  # run steps 1-4 independently for each, then a 3rd "both" pass in step 4 only

BIMODALITY_COEFFICIENT_THRESHOLD = 0.555
DIP_TEST_ALPHA = 0.05
ASHMANS_D_BIMODAL_THRESHOLD = 2.0
CLUSTER_WEIGHT_SYMMETRY_THRESHOLD = 0.15

GMM_MAX_COMPONENTS = 3
GMM_N_INIT = 5
GMM_RANDOM_STATE = 0

KMEANS_K_RANGE = range(2, 7)  # 2..6 inclusive
CLUSTERING_RANDOM_STATE = 0

KDE_GRID_POINTS = 512
RANDOM_STATE = 0

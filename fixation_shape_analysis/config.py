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


# A separate branch next to outputs/, not a variant of the main pipeline's own results:
# validates the main pipeline's BIC-based k selection (Step 3) against McLachlan's (1987)
# bootstrap likelihood-ratio test for the number of mixture components - see
# lrt_vs_bic_check.py for the full explanation and citations.
LRT_VS_BIC_DIR = os.path.join(BASE_OUTPUT_DIR, "LRT_vs_BIC_check")
LRT_VS_BIC_TABLES_DIR = os.path.join(LRT_VS_BIC_DIR, "tables")
LRT_VS_BIC_PLOTS_GROUP_DIR = os.path.join(LRT_VS_BIC_DIR, "plots", "group")
LRT_VS_BIC_PLOTS_PER_PARTICIPANT_DIR = os.path.join(LRT_VS_BIC_DIR, "plots", "per_participant")
LRT_VS_BIC_SUMMARY_PATH = os.path.join(LRT_VS_BIC_DIR, "summary.md")


def ensure_lrt_vs_bic_dirs():
    for d in (LRT_VS_BIC_TABLES_DIR, LRT_VS_BIC_PLOTS_GROUP_DIR, LRT_VS_BIC_PLOTS_PER_PARTICIPANT_DIR):
        os.makedirs(d, exist_ok=True)


def lrt_vs_bic_table_path(filename):
    return os.path.join(LRT_VS_BIC_TABLES_DIR, filename)


def lrt_vs_bic_group_plot_path(filename):
    return os.path.join(LRT_VS_BIC_PLOTS_GROUP_DIR, filename)


def lrt_vs_bic_per_participant_plot_path(filename):
    return os.path.join(LRT_VS_BIC_PLOTS_PER_PARTICIPANT_DIR, filename)


# ---------------------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------------------
# Unit of analysis is (participant_id, trial_id) - i.e. one participant-panel "session" -
# not the participant pooled across panels. See unit_id in step0_load_data.py.
MIN_FIXATIONS_PER_UNIT = 30  # configurable exclusion threshold (Step 0)

AXES = ("x", "y")  # run steps 1-4 independently for each - no combined x+y pass

BIMODALITY_COEFFICIENT_THRESHOLD = 0.555
DIP_TEST_ALPHA = 0.05
ASHMANS_D_BIMODAL_THRESHOLD = 2.0
CLUSTER_WEIGHT_SYMMETRY_THRESHOLD = 0.15

# Component search range differs per axis: x has 9 dictionary columns to potentially
# resolve, y only 2 rows, so a shared k=1..3 cap under-fits x (found via an earlier
# exploratory k=1..9 comparison run) while being plenty for y.
GMM_MAX_COMPONENTS_BY_AXIS = {"x": 9, "y": 3}
GMM_N_INIT = 10
GMM_RANDOM_STATE = 0

# Step 3 k-selection stability check: how often does refitting on a bootstrap resample of
# a unit's own fixations pick the same k as the main fit? B=100 is within the commonly-cited
# range for bootstrap standard-error/model-selection-stability estimates (Efron & Tibshirani,
# "An Introduction to the Bootstrap", 1993: ~50-200 replicates for a standard error; Monti et
# al., "Consensus Clustering", 2003: ~100-500 for resampling-based cluster/model-order
# stability). n_init is lower for the bootstrap refits than for the main fit - each replicate
# only needs a "good enough" fit since 100 of them already average out initialization noise,
# following the same logic as McLachlan's (1987) bootstrap test for the number of mixture
# components, which uses cheaper per-replicate fits than the definitive one being tested.
GMM_BOOTSTRAP_N_RESAMPLES = 100
GMM_BOOTSTRAP_N_INIT = 3

# --- LRT-vs-BIC check (lrt_vs_bic_check.py) ---
# Sequential parametric-bootstrap likelihood-ratio test for the number of mixture
# components, per McLachlan, G.J. (1987), "On bootstrapping the likelihood ratio test
# statistic for the number of components in a normal mixture", Journal of the Royal
# Statistical Society: Series C (Applied Statistics), 36(3), 318-324. See
# lrt_vs_bic_check.py's module docstring for the full method explanation.
# Same B/n_init reasoning as GMM_BOOTSTRAP_N_RESAMPLES/N_INIT above (Efron & Tibshirani
# 1993; Monti et al. 2003; McLachlan 1987) - full n_init on the two real-data fits per
# k-vs-k+1 transition, cheaper n_init on the many simulated-data refits.
LRT_N_RESAMPLES = 100
LRT_N_INIT = 3
LRT_ALPHA = 0.05  # reject H0 (k components suffice) if bootstrap p-value < this

KMEANS_K_RANGE = range(2, 7)  # 2..6 inclusive
CLUSTERING_RANDOM_STATE = 0

KDE_GRID_POINTS = 512
RANDOM_STATE = 0

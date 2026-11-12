"""
Cache builder for live_board.py (decision 2026-11-09). Reads existing pipeline outputs
under Noam_M and computes ONLY what was never saved as numbers - the split-half
reliability distributions (the pipeline's run_*_reliability steps saved only PNGs) - so
the board loads instantly. Everything else (participant table, symbol-count trajectory
stability) is read straight from its existing output, not recomputed.

Reliability = feature_pipeline.calculate_reliability unchanged (the pipeline's own
split-half definition: per participant, randomly split their values into two halves of
L, correlate half-A vs half-B means across participants, RELIABILITY_REPETITIONS times),
at L = min_count // 2 - the same L as the pipeline's saved reliability histograms.
Per feature, the cache keeps the full r distribution and the two half-mean vectors of
two example repetitions (the ones nearest the median and the 5th percentile of r).

Sources and levels, mirroring feature_pipeline.main's reliability steps:
  - trial level  : trial_features.csv, MIN_TRIALS_RELIABILITY  (-> <feature>_mean columns)
  - panel level  : panel_features.csv, MIN_PANELS_RELIABILITY  (fixation_bcea excluded,
                   as in run_panel_reliability)
  - saccade latency : compiled_latencies.csv, MIN_SACCADE_LATENCY_RELIABILITY
  - PSO duration    : duration_analysis.build_pso_duration_table  (--with-pso; reads
                      every annotated CSV, slow)
  - fixation BCEA   : the run's saved fixation_bcea_per_fixation.csv (written by
                      feature_pipeline.main since 2026-11-10); skipped if the run predates it -
                      the board then shows that run's saved reliability PNGs instead
A source whose cache file already exists is skipped unless --force.

Usage: python live_board_cache.py [--with-pso] [--force]
"""
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

import pipeline_config as pc
import feature_pipeline as fp

NOAM_M = pc.noam_m_root()

# The run the board shows (decision 2026-11-12 layout): a <Noam_M>/<DD_MM_YY>/ folder.
# LIVE_BOARD_RUN_ROOT pins one - run_full_pipeline.py sets it so the cache it builds comes
# from the run it just wrote; otherwise the newest date folder with feature tables.
RUN_ROOT = os.environ.get("LIVE_BOARD_RUN_ROOT") or pc.latest_date_root()
RESULTS_ROOT = os.path.join(RUN_ROOT, "analysis_results")
FEATURE_RUN_DIR = os.path.join(RESULTS_ROOT, "features")
FEATURE_TABLES_DIR = fp.tables_dir(FEATURE_RUN_DIR)
BOARD_ROOT = os.path.join(RUN_ROOT, "live_board")  # the board's files live with its run
CACHE_DIR = os.path.join(BOARD_ROOT, "cache")
EXPORT_DIR = os.path.join(BOARD_ROOT, "exports")
PARTICIPANT_FEATURES_CSV = os.path.join(FEATURE_TABLES_DIR, "participant_features.csv")
TRIAL_FEATURES_CSV = os.path.join(FEATURE_TABLES_DIR, "trial_features.csv")
PANEL_FEATURES_CSV = os.path.join(FEATURE_TABLES_DIR, "panel_features.csv")
# The run's own copy of the saccade-latency table - no fallback to another run's table.
SACCADE_LATENCY_CSV = fp.saccade_latency_csv_path(FEATURE_RUN_DIR)
FIXATION_BCEA_TABLE_CSV = fp.fixation_bcea_table_path(FEATURE_RUN_DIR)
SAVED_FIXATION_BCEA_PLOTS_DIR = os.path.join(FEATURE_RUN_DIR, "reliability", "per_event", "fixation_bcea",
                                             "fixation_bcea")
# Markov mean-matrix PCA outputs shown on the board: with / without the ENTER-EXIT states
# (decision 2026-11-12), each as plain transition probabilities and normalized by number
# of transitions. Value = (label, short label, the variant's folder under
# analysis_results/markov = markov_core.AnalysisConfig.folder_name).
MARKOV_VARIANTS = {
    "probabilities": ("With ENTER/EXIT · transition probabilities", "w/ ENTER-EXIT · prob",
                      os.path.join("w_exit_enter", "probabilities")),
    "n_transitions": ("With ENTER/EXIT · normalized by n transitions", "w/ ENTER-EXIT · n-trans",
                      os.path.join("w_exit_enter", "n_transitions")),
    "wo_probabilities": ("Digits only (no ENTER/EXIT) · transition probabilities", "digits · prob",
                         os.path.join("wo_exit_enter", "probabilities")),
    "wo_n_transitions": ("Digits only (no ENTER/EXIT) · normalized by n transitions", "digits · n-trans",
                         os.path.join("wo_exit_enter", "n_transitions")),
}
MARKOV_RUN_ROOT = RUN_ROOT


def markov_dir(variant):
    """<run>/analysis_results/markov/<variant>/pca - markov_analyzer's output_dir."""
    return os.path.join(RESULTS_ROOT, "markov", MARKOV_VARIANTS[variant][2], "pca")


def available_markov_variants():
    """Variants this run actually produced (a run only has the ones main.py configured)."""
    return [v for v in MARKOV_VARIANTS
            if os.path.exists(os.path.join(markov_dir(v), "panel_projection", "panel_projection_scores.csv"))]


# Names the user gives each variant's PCs on the board (decision 2026-11-12) - stored with
# the run, since a PC's meaning belongs to that run's fit.
MARKOV_PC_NAMES_JSON = os.path.join(BOARD_ROOT, "markov_pc_names.json")


SYMBOL_COUNT_STABILITY_DIR = os.path.join(RESULTS_ROOT, "symbol_count_over_panels")
ANNOTATION_METHOD = "model_based"

RELIABILITY_SEED = 20261109
EXAMPLE_PERCENTILES = {"median": 50, "p05": 5}


def reliability_cache_path(level):
    return os.path.join(CACHE_DIR, f"reliability_{level}.json")


def _reliability_record(df, feature, min_count, level, source, participant_col):
    participants, matrix = fp.build_reliability_matrix(df, feature, min_count)
    l_size = min_count // 2
    if participants is None or l_size < 1:
        return None
    np.random.seed(RELIABILITY_SEED)  # calculate_reliability draws from np.random's global state
    r, halves = fp.calculate_reliability(matrix, l_size, fp.RELIABILITY_REPETITIONS, min_count, feature,
                                         plot=False, return_details=True)
    valid = r[~np.isnan(r)]
    if len(valid) == 0:
        return None
    examples = {}
    for name, q in EXAMPLE_PERCENTILES.items():
        target = np.percentile(valid, q)
        it = int(np.nanargmin(np.abs(r - target)))
        examples[name] = {"iteration": it, "r": float(r[it]),
                          "half_a": [float(v) for v in halves[it, 0]], "half_b": [float(v) for v in halves[it, 1]]}
    print(f"  [{level}] {feature}: N={len(participants)}, L={l_size}, mean r={valid.mean():.3f}")
    return {"feature": feature, "participant_column": participant_col, "level": level, "source": source,
            "min_count": min_count, "l_size": l_size, "repetitions": fp.RELIABILITY_REPETITIONS,
            "seed": RELIABILITY_SEED, "participants": list(participants),
            "r": [None if np.isnan(v) else float(v) for v in r], "examples": examples}


def _write(level, records):
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {"built": date.today().isoformat(), "records": [x for x in records if x is not None]}
    with open(reliability_cache_path(level), "w") as f:
        json.dump(payload, f)
    print(f"Saved {len(payload['records'])} {level} reliability records -> {reliability_cache_path(level)}")


def _excluded(df):
    return df[~df["participant"].isin(fp.EXCLUDED_PARTICIPANTS)]


def build_trial_level():
    df = _excluded(pd.read_csv(TRIAL_FEATURES_CSV))
    feats = [c for c in df.columns if c not in fp.ID_COLS_TRIAL]
    _write("trial", [_reliability_record(df, f, fp.MIN_TRIALS_RELIABILITY, "trial", TRIAL_FEATURES_CSV,
                                         "mean_typicality" if f == "typicality" else f"{f}_mean") for f in feats])


def build_panel_level():
    df = _excluded(pd.read_csv(PANEL_FEATURES_CSV))
    exclude = fp.ID_COLS_PANEL | {"fixation_bcea_mean", "fixation_bcea_std"}  # as run_panel_reliability
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    # Panel column -> participant-table column it aggregates into (participant-level
    # values are the mean over panels; slope/corr_to_mean carry a _mean suffix there).
    rename = {"first5_last5_slope": "first5_last5_slope_mean", "symbols_corr_to_mean": "symbols_corr_to_mean_mean"}
    _write("panel", [_reliability_record(df, f, fp.MIN_PANELS_RELIABILITY, "panel", PANEL_FEATURES_CSV,
                                         rename.get(f, f)) for f in feats])


def build_saccade_latency():
    if not os.path.exists(SACCADE_LATENCY_CSV):
        print(f"No saccade latency table at {SACCADE_LATENCY_CSV} - not cached.")
        return
    df = _excluded(pd.read_csv(SACCADE_LATENCY_CSV))
    _write("saccade_latency", [_reliability_record(df, "saccade_latency_ms", fp.MIN_SACCADE_LATENCY_RELIABILITY,
                                                   "saccade_latency", SACCADE_LATENCY_CSV, "saccade_latency_ms_mean")])


def markov_reliability_level(variant):
    return f"markov_{os.path.basename(MARKOV_RUN_ROOT)}_{variant}"


def build_markov_pc(variant):
    """Panel-level split-half reliability (the pipeline's own definition, as for every
    other panel-level feature) of each participant's per-panel PC scores, where those
    scores are markov_analyzer.run_panel_projection's projections of each single panel
    onto the MEAN-MATRIX PCs - read from its saved CSV, nothing refit."""
    path = os.path.join(markov_dir(variant), "panel_projection", "panel_projection_scores.csv")
    df = _excluded(pd.read_csv(path).rename(columns={"Participant": "participant"}))
    level = markov_reliability_level(variant)
    _write(level, [_reliability_record(df, pc, fp.MIN_PANELS_RELIABILITY, level, path, f"markov_{variant}_{pc}")
                   for pc in ("PC1", "PC2", "PC3")])


def build_pso_duration():
    from duration_analysis import build_pso_duration_table
    df = _excluded(build_pso_duration_table(ANNOTATION_METHOD))
    _write("pso_duration", [_reliability_record(df, "pso_duration_ms", fp.MIN_PSO_DURATION_RELIABILITY,
                                                "pso_duration", f"build_pso_duration_table({ANNOTATION_METHOD})",
                                                "pso_duration_ms_mean")])


def build_event_durations():
    """Fixation and saccade duration (decision 2026-11-12) - from the run's saved raw
    per-event table, the same split-half as the pipeline's run_event_duration_reliability."""
    path = fp.event_durations_table_path(FEATURE_RUN_DIR)
    df = _excluded(pd.read_csv(path))
    records = []
    for event_type, feature in fp.EVENT_DURATION_FEATURES.items():
        d = df[df["event_type"] == event_type].rename(columns={"duration_ms": feature})
        records.append(_reliability_record(d, feature, fp.MIN_PSO_DURATION_RELIABILITY, "event_durations", path,
                                           f"{feature}_mean"))
    _write("event_durations", records)


def build_fixation_bcea():
    df = _excluded(pd.read_csv(FIXATION_BCEA_TABLE_CSV))
    _write("fixation_bcea", [_reliability_record(df, "fixation_bcea", fp.MIN_FIXATION_BCEA_RELIABILITY,
                                                 "fixation_bcea", FIXATION_BCEA_TABLE_CSV, "fixation_bcea_mean")])


def main(argv):
    force = "--force" in argv
    steps = [("trial", build_trial_level), ("panel", build_panel_level), ("saccade_latency", build_saccade_latency)]
    if "--with-pso" in argv:
        steps.append(("pso_duration", build_pso_duration))
    for variant in available_markov_variants():
        steps.append((markov_reliability_level(variant), lambda v=variant: build_markov_pc(v)))
    if os.path.exists(fp.event_durations_table_path(FEATURE_RUN_DIR)):
        steps.append(("event_durations", build_event_durations))
    if os.path.exists(FIXATION_BCEA_TABLE_CSV):
        steps.append(("fixation_bcea", build_fixation_bcea))
    else:
        print(f"No saved per-fixation BCEA table at {FIXATION_BCEA_TABLE_CSV} - fixation BCEA reliability "
              f"not cached (board shows the saved plots from {SAVED_FIXATION_BCEA_PLOTS_DIR}).")
    for level, fn in steps:
        if os.path.exists(reliability_cache_path(level)) and not force:
            print(f"Cache exists, skipping {level}: {reliability_cache_path(level)}")
            continue
        print(f"Building {level} reliability cache...")
        fn()


if __name__ == "__main__":
    main(sys.argv[1:])

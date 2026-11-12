"""
Short alternative to run_full_pipeline.py (decision 2026-11-12): takes a finished run saved
in the OLD layout (<Noam_M>/markov_features_results_<date>/), moves its outputs into the
new dated layout (<Noam_M>/<stage1 date>/, see run_full_pipeline.run), and computes only
what that run is missing - instead of recomputing everything.

Steps:
  1. Move: scores, exclusion summary, Markov w_exit_enter (probabilities + n_transitions),
     feature tables, non-SDMT tables, reliability, group t-tests, key-entry, symbol count,
     PCA, and the old correlation scatterplots. Anything without a place in the new layout
     (symbol_search_analysis, duration_analysis, partial_correlation, symbols_corr_to_mean,
     per-panel Markov heatmaps, old heatmap PNGs) stays in the old folder.
  2. Markov: asymmetry histograms for the moved variants; the two wo_exit_enter variants
     computed from the moved trial_stats.
  3. Features: typicality_slope (from the trial table), fixation/saccade duration (one scan
     of the Stage 1 CSVs), Markov_PC1-3 re-merged from w_exit_enter/n_transitions,
     demographics re-merged from the new summary table (copied into non_SDMT/demographics).
     Reliability for the new features; group t-tests redone (cheap).
  4. Correlations: all heatmaps redrawn; scatterplots only for significant pairs that
     involve a new or changed column (the rest were moved from the old run).
  5. Live board cache rebuilt for the dated folder.

Usage:
    python complete_existing_run.py --dry-run     # print the moves, change nothing
    python complete_existing_run.py               # do it
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

import pipeline_config as pc

OLD_RUN = os.path.join(pc.noam_m_root(), "markov_features_results_11_11_26")
DATE = "11_11_26"  # the Stage 1 run that OLD_RUN analyzed (SDMT_annotated_gaze_<DATE>_model)
METHOD = "model_based"

# The date folder holding that Stage 1 run (12_11_26 - named after the analysis day)
ROOT = pc.stage1_date_root(METHOD, DATE)
RESULTS = os.path.join(ROOT, "analysis_results")
MARKOV = os.path.join(RESULTS, "markov")
FEATURES = os.path.join(RESULTS, "features")
OLD_MARKOV = os.path.join(OLD_RUN, "markov_analysis")
OLD_FEAT = os.path.join(OLD_RUN, "feature_analysis")

OLD_MARKOV_VARIANTS = {"analysis_NO_repeats_ALL_states": "probabilities",
                       "analysis_NO_repeats_ALL_states_NORMALIZED_BY_N_TRANSITIONS": "n_transitions"}
# old plots/mean_matrix sub-folder -> new <variant>/pca sub-folder (heatmaps/ stays behind)
OLD_PCA_SUBDIRS = {"pca": "pc_scatter", "weights": "weights", "variance": "variance", "separation": "separation",
                   "consistency_within_vs_between": "consistency", "consistency": "consistency",
                   "pca_panel_projection": "panel_projection", "squared_cosines": "squared_cos"}


def move_plan():
    """(source, destination) pairs; a destination is the new full path of that item."""
    plan = [(os.path.join(OLD_RUN, "behavior_scores", "all_scores.csv"),
             os.path.join(ROOT, "behavior_scores", "all_scores.csv")),
            (os.path.join(OLD_RUN, "analysis_participant_exclusion_summary.csv"),
             os.path.join(ROOT, "analysis_participant_exclusion_summary.csv"))]

    for old_name, new_type in OLD_MARKOV_VARIANTS.items():
        src, dst = os.path.join(OLD_MARKOV, old_name), os.path.join(MARKOV, "w_exit_enter", new_type)
        plan += [(os.path.join(src, "markov_matrices"), os.path.join(dst, "matrices")),
                 (os.path.join(src, "digit_asymmetry"), os.path.join(dst, "asymmetry")),
                 (os.path.join(src, "exclusion_log.csv"), os.path.join(dst, "exclusion_log.csv"))]
        for old_sub, new_sub in OLD_PCA_SUBDIRS.items():
            plan.append((os.path.join(src, "plots", "mean_matrix", old_sub), os.path.join(dst, "pca", new_sub)))

    csvs = os.path.join(FEATURES, "csvs")
    for f in ("trial_features.csv", "panel_features.csv", "participant_features.csv", "exclusion_log.csv"):
        plan.append((os.path.join(OLD_FEAT, f), os.path.join(csvs, f)))
    plan += [(os.path.join(OLD_FEAT, "fixation_bcea_per_fixation.csv"),
              os.path.join(FEATURES, "non_SDMT", "fixation_bcea", "fixation_bcea_per_fixation.csv")),
             (os.path.join(OLD_FEAT, "saccade_latency"), os.path.join(FEATURES, "non_SDMT", "saccade_latency")),
             (os.path.join(OLD_FEAT, "reliability", "trial_level"), os.path.join(FEATURES, "reliability", "trial_level")),
             (os.path.join(OLD_FEAT, "reliability", "panel_level"), os.path.join(FEATURES, "reliability", "panel_level")),
             (os.path.join(OLD_FEAT, "pca_pareto"), os.path.join(FEATURES, "PCA")),
             (os.path.join(OLD_FEAT, "symbol_count_stability"), os.path.join(RESULTS, "symbol_count_over_panels"))]
    for ev in ("saccade_latency", "pso_duration", "fixation_bcea"):
        plan.append((os.path.join(OLD_FEAT, "reliability", ev), os.path.join(FEATURES, "reliability", "per_event", ev)))

    # group t-tests: ranked grids + stats.csv, and every boxplot into one boxplots/ folder
    gc = os.path.join(OLD_FEAT, "group_comparison")
    tt = os.path.join(FEATURES, "group_t_tests")
    if os.path.isdir(gc):
        for f in os.listdir(gc):
            if f.endswith((".png", ".csv")):
                plan.append((os.path.join(gc, f), os.path.join(tt, f)))
    for box_src in (os.path.join(gc, "signal_features"), os.path.join(OLD_FEAT, "key_entry_strategy", "group_comparison")):
        if os.path.isdir(box_src):
            for f in os.listdir(box_src):
                plan.append((os.path.join(box_src, f), os.path.join(tt, "boxplots", f)))

    ke = os.path.join(OLD_FEAT, "key_entry_strategy")
    if os.path.isdir(ke):
        for f in os.listdir(ke):
            if f != "group_comparison":
                plan.append((os.path.join(ke, f), os.path.join(RESULTS, "key_entry_strategy", f)))

    # old scatterplots: all_features = the full set, by_domain = the reduced set
    for subset in ("all", "MS", "HC"):
        plan += [(os.path.join(OLD_FEAT, "correlations", "all_features", subset, "scatterplots"),
                  os.path.join(FEATURES, "correlations", subset, "full", "scatterplots")),
                 (os.path.join(OLD_FEAT, "correlations", "by_domain", subset, "scatterplots"),
                  os.path.join(FEATURES, "correlations", subset, "reduced", "scatterplots"))]
    return plan


def do_moves(dry_run):
    moved = skipped = 0
    for src, dst in move_plan():
        if not os.path.exists(src):
            continue
        if os.path.exists(dst):
            print(f"  SKIP (already exists): {dst}")
            skipped += 1
            continue
        print(f"  {os.path.relpath(src, OLD_RUN)}  ->  {os.path.relpath(dst, ROOT)}")
        if not dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        moved += 1
    print(f"{'Would move' if dry_run else 'Moved'} {moved} items ({skipped} skipped).")


def markov_step():
    import main as markov_main
    from markov_core import AnalysisConfig
    from markov_loader import DataManager

    common = dict(with_repeats=False, use_mean_matrix=True, annotation_method=METHOD, date_str=DATE,
                  output_base_path=MARKOV, scores_dir=os.path.join(ROOT, "behavior_scores"))
    # asymmetry histogram for the two moved variants (the old run saved only the CSVs)
    for matrix_type in ("conditional", "normalized_by_n_transitions"):
        cfg = AnalysisConfig(only_1_to_9=False, from_scratch=False, matrix_type=matrix_type, **common)
        per_participant = os.path.join(cfg.main_output_path, "asymmetry", "digit_asymmetry_per_participant.csv")
        if os.path.exists(per_participant):
            DataManager(cfg)._plot_digit_asymmetry_histogram(pd.read_csv(per_participant),
                                                            os.path.dirname(per_participant))
    # the two new wo_exit_enter variants, from the moved trial_stats
    source = AnalysisConfig(only_1_to_9=False, from_scratch=False, matrix_type="conditional", **common).main_output_path
    for matrix_type in ("conditional", "normalized_by_n_transitions"):
        cfg = AnalysisConfig(only_1_to_9=True, from_scratch=True, matrix_type=matrix_type,
                             trial_stats_source_dir=source, **common)
        markov_main.run_task(cfg, exclusion_summary_path=os.path.join(ROOT, "analysis_participant_exclusion_summary.csv"))


def features_step():
    import feature_pipeline as fp
    from markov_core import AnalysisConfig

    df_trials, df_panels, df_old = fp.load_tables(FEATURES)
    df = df_old.copy()

    # typicality_slope (panel level -> participant mean), from the saved trial table
    df_panels = fp.compute_typicality_slope(df_trials, df_panels.drop(columns=["typicality_slope"], errors="ignore"))
    slope = df_panels.groupby("participant")["typicality_slope"].mean().rename("typicality_slope_mean").reset_index()
    df = df.drop(columns=["typicality_slope_mean"], errors="ignore").merge(slope, on="participant", how="left")

    # fixation / saccade duration (one scan of the Stage 1 CSVs)
    df_durations = fp.build_and_save_event_durations(METHOD, FEATURES, date_str=DATE)
    df = fp.merge_event_durations(df.drop(columns=[c for c in df.columns if c.startswith(("fixation_duration_ms",
                                                                                            "saccade_duration_ms"))]),
                                  df_durations)

    # Markov_PC1-3 from w_exit_enter/n_transitions, as in feature_pipeline.main
    pca_dir = AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=True, from_scratch=False,
                             annotation_method=METHOD, matrix_type="normalized_by_n_transitions",
                             output_base_path=MARKOV).plot_output_path
    df = fp.merge_markov_pca_scores(df.drop(columns=[c for c in df.columns if c.startswith("Markov_PC")]),
                                    METHOD, pca_dir=pca_dir)

    # demographics from the new summary table
    df = fp.add_encoded_demographics(fp.merge_ido_table(df))
    fp.copy_demographics_table(FEATURES)
    fp.save_tables(df_trials, df_panels, df, FEATURES)

    # columns that are new, or whose values changed, vs. the old participant table
    new_or_changed = set()
    for col in df.columns:
        if col not in df_old.columns:
            new_or_changed.add(col)
            continue
        a = df.set_index("participant")[col]
        b = df_old.set_index("participant")[col].reindex(a.index)
        if not (a.equals(b) or (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
                                and np.allclose(a.astype(float), b.astype(float), equal_nan=True))):
            new_or_changed.add(col)
    print(f"New or changed columns: {sorted(new_or_changed)}")

    # reliability for the new features
    fp.evaluate_feature_reliability(df_panels, "typicality_slope", fp.MIN_PANELS_RELIABILITY,
                                    os.path.join(FEATURES, "reliability", "panel_level"), df_participants=df)
    fp.run_event_duration_reliability(FEATURES, df_participants=df)

    # group t-tests (all features, cheap) + boxplots of the new duration features
    fp.run_group_comparison(df, FEATURES)
    box_dir = os.path.join(fp.group_t_tests_dir(FEATURES), "boxplots")
    demographics = fp.demographics_text(df[~df["participant"].isin(fp.EXCLUDED_PARTICIPANTS)])
    for feature, ylabel in (("fixation_duration_ms_mean", "Fixation Duration (ms)"),
                            ("saccade_duration_ms_mean", "Saccade Duration (ms)")):
        fp.plot_group_boxplot_with_ttest(df, feature, ylabel, os.path.join(box_dir, f"{feature}_boxplot.png"),
                                         demographics=demographics)

    # all heatmaps; scatterplots only for pairs with a new/changed column
    fp.run_correlation_heatmaps(df, FEATURES, scatter_only_for=new_or_changed)


def board_step():
    os.environ["LIVE_BOARD_RUN_ROOT"] = ROOT
    import live_board_cache
    live_board_cache.main(["--force", "--with-pso"])


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    dry = "--dry-run" in sys.argv[1:]
    print(f"Old run: {OLD_RUN}\nNew root: {ROOT}\n")
    do_moves(dry)
    if dry:
        sys.exit(0)
    for name, fn in (("Markov", markov_step), ("Features", features_step), ("Live board cache", board_step)):
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}", flush=True)
        fn()
    print("\nDone.")

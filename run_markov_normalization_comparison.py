"""
Markov analysis under all three transition-matrix normalizations (markov_core
AnalysisConfig.matrix_type), each followed by the feature-correlation heatmap with that
variant's Markov PC1-3 merged in (full heatmap + reduced, domain-colored heatmaps
ordered by domain and by correlation-block clustering):
    conditional                 - P(B | A), the original row-stochastic matrix
    normalized_by_score         - count(A->B) / the panel's SDMT score
    normalized_by_n_transitions - count(A->B) / all transitions in the panel

Nothing upstream is recomputed:
  * Markov matrices are rebuilt from SOURCE_RUN's saved trial_stats CSVs (their
    Sequence/Cleaned_Sequence columns - identical counts to the live computation,
    verified panel-by-panel against SOURCE_RUN's saved conditional matrices).
  * The feature table is SOURCE_RUN's participant_features.csv; only its Markov_PC1-3
    columns are replaced, per variant.

Usage:
    python run_markov_normalization_comparison.py                     # everything
    python run_markov_normalization_comparison.py --correlations-only # redraw the
        correlation heatmaps from each variant's already-saved participant_features.csv
    python run_markov_normalization_comparison.py --panel-projection-only # project every
        participant x panel matrix into each variant's mean-matrix PCA space, from the
        already-saved matrices (MarkovAnalyzer.run_panel_projection)
    python run_markov_normalization_comparison.py --squared-cosines-only # how well
        PC1-3 explain each participant's mean matrix (MarkovAnalyzer.run_squared_cosines)
    python run_markov_normalization_comparison.py --asymmetry-only # symbol->symbol
        transition asymmetry per panel/participant (DataManager.save_digit_asymmetry)
"""
import os
import shutil
import sys

import pandas as pd

import pipeline_config as pc
import feature_pipeline as fp
import main as markov_main
from markov_core import AnalysisConfig
from markov_loader import DataManager
from markov_analyzer import MarkovAnalyzer

ANNOTATION_METHOD = "model_based"
SOURCE_RUN = os.path.join(pc.noam_m_root(), "markov_features_results_27_09_26")
OUT_ROOT = os.path.join(pc.noam_m_root(), "markov_feature_analysis_02_11_26")

# matrix_type -> feature_analysis sub-folder name
VARIANTS = {
    "conditional": "regular",
    # "normalized_by_score": "normalized_by_score",  # off (decision 2026-11-11)
    "normalized_by_n_transitions": "normalized_by_n_transitions",
}


def run_correlations(df, run_dir):
    # Full feature set -> correlations/<subset>/; reduced feature set with domain-colored
    # feature names, ordered by domain -> correlations/by_domain/<subset>/ and by
    # correlation-block clustering -> correlations/by_cluster/<subset>/.
    fp.run_correlation_analysis(df, run_dir)
    fp.run_domain_grouped_correlation_analysis(df, run_dir)
    fp.run_clustered_correlation_analysis(df, run_dir)


def run_correlations_only():
    for feature_folder in VARIANTS.values():
        run_dir = os.path.join(OUT_ROOT, "feature_analysis", feature_folder)
        print(f"\n=== Correlations: {feature_folder} ===")
        df = pd.read_csv(os.path.join(run_dir, "participant_features.csv"))
        run_correlations(df, run_dir)


def _load_saved_variant(matrix_type):
    """DataManager over a variant's already-saved matrices/trial stats in OUT_ROOT,
    with the same >= MIN_PANELS_FOR_MARKOV filter the full run applies."""
    cfg = AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=True, from_scratch=False,
                         annotation_method=ANNOTATION_METHOD, output_base_path=os.path.join(OUT_ROOT, "markov_analysis"),
                         matrix_type=matrix_type, legacy_layout=True)
    loader = DataManager(cfg)
    loader.load_participants_and_scores()
    loader.load_or_compute_matrices(from_scratch=False)
    loader.filter_by_min_panels()
    return cfg, loader


def run_panel_projection_only():
    for matrix_type in VARIANTS:
        cfg, loader = _load_saved_variant(matrix_type)
        print(f"\n=== Panel projection: {cfg.folder_name} ===")
        MarkovAnalyzer(loader.get_flat_participants(), cfg).run_panel_projection()


def run_squared_cosines_only():
    for matrix_type in VARIANTS:
        cfg, loader = _load_saved_variant(matrix_type)
        print(f"\n=== Squared cosines: {cfg.folder_name} ===")
        MarkovAnalyzer(loader.get_flat_participants(), cfg).run_squared_cosines()


def run_asymmetry_only():
    for matrix_type in VARIANTS:
        cfg, loader = _load_saved_variant(matrix_type)
        print(f"\n=== Digit asymmetry: {cfg.folder_name} ===")
        loader.save_digit_asymmetry()


def run():
    src_markov = os.path.join(SOURCE_RUN, "markov_analysis")
    markov_base = os.path.join(OUT_ROOT, "markov_analysis")
    exclusion_summary_path = os.path.join(OUT_ROOT, "analysis_participant_exclusion_summary.csv")

    # DataManager reads scores from <output_base_path>/behavior_scores/all_scores.csv
    # (falling back to a slow per-participant WAV-filename scan over the share).
    scores_dst = os.path.join(markov_base, "behavior_scores", "all_scores.csv")
    os.makedirs(os.path.dirname(scores_dst), exist_ok=True)
    shutil.copy(os.path.join(SOURCE_RUN, "behavior_scores", "all_scores.csv"), scores_dst)

    src_features = pd.read_csv(os.path.join(SOURCE_RUN, "feature_analysis", "participant_features.csv"))
    src_features = src_features.drop(columns=[c for c in src_features.columns if c.startswith("Markov_PC")])

    for matrix_type, feature_folder in VARIANTS.items():
        cfg = AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=True, from_scratch=True,
                             annotation_method=ANNOTATION_METHOD, output_base_path=markov_base,
                             matrix_type=matrix_type, legacy_layout=True)
        # Every variant (conditional included) reads the SAME source run's trial stats.
        cfg.trial_stats_source_dir = os.path.join(src_markov, AnalysisConfig(
            with_repeats=False, only_1_to_9=False, use_mean_matrix=True, from_scratch=False,
            annotation_method=ANNOTATION_METHOD, output_base_path=src_markov, legacy_layout=True).folder_name)
        markov_main.run_task(cfg, exclusion_summary_path=exclusion_summary_path)

        run_dir = os.path.join(OUT_ROOT, "feature_analysis", feature_folder)
        os.makedirs(run_dir, exist_ok=True)
        df = fp.merge_markov_pca_scores(src_features.copy(), ANNOTATION_METHOD,
                                        pca_dir=cfg.plot_output_path)
        df.to_csv(os.path.join(run_dir, "participant_features.csv"), index=False)
        run_correlations(df, run_dir)


if __name__ == "__main__":
    if "--correlations-only" in sys.argv[1:]:
        run_correlations_only()
    elif "--panel-projection-only" in sys.argv[1:]:
        run_panel_projection_only()
    elif "--squared-cosines-only" in sys.argv[1:]:
        run_squared_cosines_only()
    elif "--asymmetry-only" in sys.argv[1:]:
        run_asymmetry_only()
    else:
        run()

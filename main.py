import sys

from markov_matrix_visualizer import MarkovVisualizer
from behavioral_analyzer import BehavioralAnalyzer
from markov_core import AnalysisConfig
from markov_loader import DataManager
from markov_analyzer import MarkovAnalyzer
import pipeline_config
from pipeline_config import ANNOTATION_METHODS

def run_pipeline(annotation_method="threshold_based", date_str=None, output_base_path=None,
                  participant_whitelist=None, exclusion_summary_path=None):
    """date_str: which Stage 1 date-folder to read annotated CSVs from (None
    auto-resolves to the most recent existing one). output_base_path: override where
    this run's own output goes (None uses the normal preliminary_results_<method>
    default) - e.g. a folder shared with the same day's feature-analysis run
    (pipeline_config.consolidated_results_dir). participant_whitelist: restrict to just
    these participant names (None runs the full population). exclusion_summary_path:
    where the shared cross-stage exclusion summary CSV lives (None uses the old global
    default - see pipeline_config.append_exclusion_summary_row)."""

    tasks = [
        # #----------------  WITHOUT REPEATS  ----------------#
        # 1.  with Enter, Exit + Mean Matrix (decision 2026-09-27: analyze each
        # participant's mean transition matrix, not every panel as its own row)
        AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=True, from_scratch=True,
                       annotation_method=annotation_method, date_str=date_str, output_base_path=output_base_path),

        # 1b/1c. Same as 1, but instead of row-conditional P(B | A):
        #   1b "normalized_by_score":         count(A->B) / the panel's SDMT score
        #   1c "normalized_by_n_transitions": count(A->B) / all transitions in the panel
        #                                     (matrix sums to 1)
        # Built from task 1's freshly saved trial_stats CSVs (no second pass over Stage
        # 1's annotated gaze), so they must stay AFTER task 1. Each lands in its own
        # analysis_..._NORMALIZED_BY_SCORE / analysis_..._NORMALIZED_BY_N_TRANSITIONS folder.
        AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=True, from_scratch=True,
                       annotation_method=annotation_method, date_str=date_str, output_base_path=output_base_path,
                       matrix_type="normalized_by_score"),
        AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=True, from_scratch=True,
                       annotation_method=annotation_method, date_str=date_str, output_base_path=output_base_path,
                       matrix_type="normalized_by_n_transitions"),

        # # 2. with Enter, Exit + Mean Matrix
        # AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=True, from_scratch=False),
        
        # # # 3. Only 1-9 States + All Panels
        # AnalysisConfig(with_repeats=False, only_1_to_9=True, use_mean_matrix=False, from_scratch=False),
    
        # # # 3. Only 1-9 States + Mean Matrix
        # # AnalysisConfig(with_repeats=False, only_1_to_9=True, use_mean_matrix=True, from_scratch=False),
    

        # # #----------------  WITH REPEATS  ----------------#
        # # # 1.  with Enter, Exit + All Panels
        # AnalysisConfig(with_repeats=True, only_1_to_9=False, use_mean_matrix=False, from_scratch=False),
        
        # # # 2.  with Enter, Exit + Mean Matrix
        # # AnalysisConfig(with_repeats=True, only_1_to_9=False, use_mean_matrix=True, from_scratch=False),
        
        # # # 3.  1-9 + All Panels
        # AnalysisConfig(with_repeats=True, only_1_to_9=True, use_mean_matrix=False, from_scratch=False),
    
        # # 3. 1-9 + Mean Matrix
        # AnalysisConfig(with_repeats=True, only_1_to_9=True, use_mean_matrix=True, from_scratch=False),

        #----------------  CONCATENATED PANELS  ----------------#

        # #------- WITH REPEATS -------#
        # 1.  with Enter, Exit + Concatenated Panels
        # AnalysisConfig(with_repeats=True, only_1_to_9=False, use_mean_matrix=False, from_scratch=False, concat_panels=True),

        # 2.  1-9 + Concatenated Panels
        # AnalysisConfig(with_repeats=True, only_1_to_9=True, use_mean_matrix=False, from_scratch=False, concat_panels=True),

        # #------- WITHOUT REPEATS -------#
        # # 3.  with Enter, Exit + Concatenated Panels
        # AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=False, from_scratch=False, concat_panels=True),

        # # 4.  1-9 + Concatenated Panels
        # AnalysisConfig(with_repeats=False, only_1_to_9=True, use_mean_matrix=False, from_scratch=False, concat_panels=True),

    
    ]

    # Non-conditional tasks with no explicit trial_stats source read the conditional task with the
    # same repeats/states settings from this same run (listed earlier in tasks).
    for config in tasks:
        if config.matrix_type != "conditional" and config.from_scratch and config.trial_stats_source_dir is None:
            config.trial_stats_source_dir = AnalysisConfig(
                with_repeats=config.with_repeats, only_1_to_9=config.only_1_to_9,
                use_mean_matrix=config.use_mean_matrix, from_scratch=False,
                annotation_method=config.annotation_method, date_str=config.date_str,
                raw_behavior_path=config.raw_behavior_path,
                output_base_path=config.output_base_path).main_output_path

    for config in tasks:
        run_task(config, participant_whitelist=participant_whitelist,
                 exclusion_summary_path=exclusion_summary_path)


def run_task(config, participant_whitelist=None, exclusion_summary_path=None):
    """One AnalysisConfig's full Markov run: load/compute matrices, exclusion
    accounting, PCA, consistency analysis, per-panel heatmaps."""
    print(f"\n{'='*50}")
    print(f"STARTING: {config.folder_name} - {'Mean Matrix' if config.use_mean_matrix else 'All Panels'} - {'Concatenated Panels' if config.concat_panels else ''} - matrix_type={config.matrix_type}")
    print(f"{'='*50}")
    
    #--------------- Load ---------------#
    loader = DataManager(config)
    loader.load_participants_and_scores()
    if participant_whitelist is not None:
        for group in list(loader.participants.keys()):
            loader.participants[group] = {
                name: p for name, p in loader.participants[group].items() if name in participant_whitelist
            }
        print(f"Restricted to whitelist: { {g: list(ps.keys()) for g, ps in loader.participants.items()} }")
    loader.load_or_compute_matrices(from_scratch=config.from_scratch)

    #--------------- Exclusion accounting + <4-panel filter ---------------#
    loader.filter_by_min_panels()
    if config.from_scratch:
        loader.save_exclusion_log()
    # compute_exclusion_summary scans the FULL eligible population - meaningless
    # (and misleading in the shared summary table) when restricted to a whitelist,
    # since everyone else would show up as a false "technical" exclusion.
    if participant_whitelist is None:
        summary_stats = loader.compute_exclusion_summary()
        pipeline_config.append_exclusion_summary_row(
            f"Markov ({config.folder_name})", config.annotation_method, summary_stats,
            out_path=exclusion_summary_path)

    # Symbol->symbol (1-9 only) transition asymmetry per panel, count-based - same
    # values for every matrix_type (computed from trial stats, not the saved matrix).
    loader.save_digit_asymmetry()

    #--------------- Get Data ---------------#
    participants = loader.get_flat_participants()
    print(f"Participants: {len(participants)}")
    
    #--------------- Markov Analysis ---------------#
    # print("\n--- Running Markov Analysis ---")

    analyzer = MarkovAnalyzer(participants, config)

    # A. PCA Analysis (Includes the Colored Plots loop: scatter, weights heatmap, elbow)
    analyzer.run_pca()

    # A2. Every individual participant x panel matrix projected into the mean-matrix PCA
    # space above (no-op outside mean-matrix mode)
    analyzer.run_panel_projection()

    # A3. Squared cosines: how much of each participant's mean matrix PC1/PC2/PC3 explain
    analyzer.run_squared_cosines()

    # B. Consistency Analysis (Violin + Permutation)
    analyzer.run_consistency_analysis(n_permutations=1000)

    # C. Per-participant/panel matrix visualization (heatmap only - the network
    # graph visualization was removed per decision 2026-09-27)
    viz = MarkovVisualizer(output_dir=config.plot_output_path, matrix_type=config.matrix_type)
    for p in participants:
        for panel in p.matrices.keys():
            viz.plot_heatmap(p, panel)


    # ---------------------------------------------------------
    # 2. RUN BEHAVIORAL ANALYSIS
    # ---------------------------------------------------------
    # print("\n--- Running Behavioral Analysis ---")
    # behav_analyzer = BehavioralAnalyzer(participants, config)
    
    # # A. POPULATE DATA (Master Loader)
    # # force_recompute=True -> Forces calculation of 'Is_One_Fix', 'Is_Two_Fix' etc.
    # behav_analyzer.populate_trial_stats(force_recompute=False)
    # # Plot Global Symbols (Mean across panels)
    # behav_analyzer.plot_metric_global_average(metric="Num_Symbols", title_suffix="Symbols")

    # # Plot Global Fixations (Mean across panels)
    # behav_analyzer.plot_metric_global_average(metric="Num_Fixations", title_suffix="Fixations")
    
    # # B. GENERATE VISUALIZATIONS
    
    # # # Task 2: Line Plots (Mean + Shaded Error Bar)
    # # behav_analyzer.plot_metric_across_trials(metric="Num_Fixations", title_suffix="Fixations")
    # # behav_analyzer.plot_metric_across_trials(metric="Num_Symbols", title_suffix="Symbols (Sequence Length)")
    
    # # Task 3: Bar Plots (Zero/One/Two items)
    # behav_analyzer.compute_and_plot_trial_types()
    # behav_analyzer.compute_and_plot_unique_symbols_distribution()
    # # Task 4: Correlations with Score
    # # behav_analyzer.analyze_correlations_with_score()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] not in ANNOTATION_METHODS:
            print(f"usage: python main.py [{'|'.join(ANNOTATION_METHODS)}]")
            sys.exit(1)
        run_pipeline(sys.argv[1])
    else:
        run_pipeline()

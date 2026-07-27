# from markov_matrix_visualizer import MarkovVisualizer
from behavioral_analyzer import BehavioralAnalyzer
from markov_core import AnalysisConfig
from markov_loader import DataManager
from markov_analyzer import MarkovAnalyzer

def run_pipeline():
    
    tasks = [
        # #----------------  WITHOUT REPEATS  ----------------#
        # 1.  with Enter, Exit + All Panels
        AnalysisConfig(with_repeats=False, only_1_to_9=False, use_mean_matrix=False, from_scratch=False),
        
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

    for config in tasks:
        print(f"\n{'='*50}")
        print(f"STARTING: {config.folder_name} - {'Mean Matrix' if config.use_mean_matrix else 'All Panels'} - {'Concatenated Panels' if config.concat_panels else ''}")
        print(f"{'='*50}")
        
        #--------------- Load ---------------#
        loader = DataManager(config)
        loader.load_participants_and_scores()
        loader.load_or_compute_matrices(from_scratch=config.from_scratch)
        
        #--------------- Get Data ---------------#
        participants = loader.get_flat_participants()
        print(f"Participants: {len(participants)}")
        
        #--------------- Markov Analysis ---------------#
        # print("\n--- Running Markov Analysis ---")

        analyzer = MarkovAnalyzer(participants, config)
        
        # A. PCA Analysis (Includes the Colored Plots loop)
        # analyzer.run_pca()
        
        # B. Consistency Analysis (Violin + Permutation)
        analyzer.run_consistency_analysis(n_permutations=1000)
        # #3. Visualize results (Optional)
        # viz = MarkovVisualizer(output_dir=config.full_output_path)

        # for p in participants:
        #     for panel in p.matrices.keys():
        #         viz.plot_heatmap(p, panel)
                # viz.plot_graph(p, panel)


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
    run_pipeline()

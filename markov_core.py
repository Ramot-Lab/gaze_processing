import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import warnings

import pipeline_config

# AnalysisConfig.matrix_type -> output folder suffix ("" keeps the original folder names).
MATRIX_TYPE_SUFFIXES = {"conditional": "", "normalized_by_n_transitions": "_NORMALIZED_BY_N_TRANSITIONS", "normalized_by_score": "_NORMALIZED_BY_SCORE"}
# Nested layout (decision 2026-11-12): <base>/<w_exit_enter|wo_exit_enter>/<matrix type>/
MATRIX_TYPE_FOLDERS = {"conditional": "probabilities", "normalized_by_n_transitions": "n_transitions",
                       "normalized_by_score": "normalized_by_score"}

@dataclass
class AnalysisConfig:
    with_repeats: bool
    only_1_to_9: bool
    use_mean_matrix: bool
    from_scratch: bool
    concat_panels: bool = False

    # Which Stage-1 annotation method this analysis reads from
    # (pipeline_config.ANNOTATION_METHODS) - drives both which annotated CSVs get read
    # and which preliminary_results_<method> folder output lands in.
    annotation_method: str = "threshold_based"

    # Which Stage 1 date-folder to read annotated CSVs from (decision 2026-09-27) - None
    # auto-resolves to the most recent existing folder for annotation_method
    # (pipeline_config.latest_annotated_gaze_date), same as before this field existed.
    # Pin explicitly (e.g. "26_09_26") when you want to be certain which Stage 1 run is
    # being read, independent of what folders happen to exist on a given day.
    date_str: str = None

    # None means "derive from pipeline_config using annotation_method" - resolved in
    # __post_init__ rather than as a dataclass default, since the value depends on
    # another field (annotation_method) and on the live network mount.
    raw_behavior_path: str = None
    output_base_path: str = None

    # What each matrix cell holds:
    #   "conditional"                 - row-stochastic P(B | A) = count(A->B) / count(A->*)
    #                                   (original).
    #   "normalized_by_n_transitions" - count(A->B) / number of all transitions in the
    #                                   panel; the whole matrix sums to 1
    #                                   (GazeMarkovModel.get_joint_probability_matrix).
    #   "normalized_by_score"         - count(A->B) / the panel's SDMT score; not summing
    #                                   to 1 (GazeMarkovModel.get_per_score_matrix).
    # Each non-conditional type gets its own folder (folder_name suffix) so it never
    # overwrites the conditional run's matrices/plots.
    matrix_type: str = "conditional"

    # If set, from_scratch builds matrices from an EARLIER run's saved
    # trial_stats_<p>_panel_<panel>.csv files (their Sequence/Cleaned_Sequence columns)
    # instead of re-reading Stage 1's annotated gaze CSVs. Point it at that run's
    # analysis folder (the one containing markov_matrices/ and exclusion_log.csv, i.e.
    # its AnalysisConfig.main_output_path). with_repeats/only_1_to_9 are applied at
    # count time, so any earlier run's trial_stats serves every variant.
    trial_stats_source_dir: str = None

    # Where all_scores.csv is read from / written to; None -> <output_base_path>/behavior_scores.
    scores_dir: str = None

    # True keeps the pre-2026-11-12 flat folder names (analysis_NO_repeats_ALL_states...,
    # markov_matrices/, digit_asymmetry/, plots/mean_matrix/) - for scripts that read
    # runs saved in that layout (run_markov_normalization_comparison.py).
    legacy_layout: bool = False

    def __post_init__(self):
        if self.matrix_type not in MATRIX_TYPE_SUFFIXES:
            raise ValueError(f"matrix_type must be one of {list(MATRIX_TYPE_SUFFIXES)}, got {self.matrix_type!r}")
        if self.raw_behavior_path is None:
            self.raw_behavior_path = pipeline_config.main_data_path()
        if self.output_base_path is None:
            self.output_base_path = pipeline_config.preliminary_results_dir(self.annotation_method)

    @property
    def folder_name(self):
        """Path of this variant under output_base_path (also its label in printouts and
        in the exclusion summary). New layout: w_exit_enter = all states incl. ENTER/EXIT,
        wo_exit_enter = digits 1-9 only; "_with_repeats" appended when repeats are kept."""
        if self.legacy_layout:
            repeat_str = "WITH_repeats" if self.with_repeats else "NO_repeats"
            state_str = "ONLY_1_9" if self.only_1_to_9 else "ALL_states"
            return f"analysis_{repeat_str}_{state_str}{MATRIX_TYPE_SUFFIXES[self.matrix_type]}"
        states = "wo_exit_enter" if self.only_1_to_9 else "w_exit_enter"
        if self.with_repeats:
            states += "_with_repeats"
        return os.path.join(states, MATRIX_TYPE_FOLDERS[self.matrix_type])

    @property
    def matrices_dirname(self):
        return "markov_matrices" if self.legacy_layout else "matrices"

    @property
    def asymmetry_dirname(self):
        return "digit_asymmetry" if self.legacy_layout else "asymmetry"

    @property
    def main_output_path(self):
        return os.path.join(self.output_base_path, self.folder_name)

    @property
    def plot_output_path(self):
        """MarkovAnalyzer's output dir: <variant>/pca (mean-matrix mode) in the new layout."""
        if self.use_mean_matrix:
            sub_folder = "mean_matrix"
        else:
            if self.concat_panels:
                sub_folder = "concat_panels"
            else:
                sub_folder = "all_panels"
        if self.legacy_layout:
            return os.path.join(self.main_output_path, "plots", sub_folder)
        return os.path.join(self.main_output_path, "pca" if self.use_mean_matrix else f"pca_{sub_folder}")


class Participant:
    def __init__(self, name, group):
        self.name = name
        self.group = group
        
        self.matrices = {}       # {panel: DataFrame}
        self.trial_data = {}
        self.mean_matrix = None
        
        # Structure: self.panel_features[panel_name][feature_name] = value
        self.panel_features = defaultdict(dict) 

    def add_matrix(self, panel, df):
        self.matrices[panel] = df

    def add_panel_feature(self, panel, key, value):
        """
        Add any metric here: Score, Dispersion, Entropy, etc.
        e.g. p.add_panel_feature('l4', 'dispersion', 0.5)
        """
        self.panel_features[panel][key] = value

    def add_trial_data(self, panel, df):
        """
        Stores trial-by-trial statistics (fixation counts, symbol counts).
        """
        self.trial_data[panel] = df

    # Wrapper for backward compatibility (Loader uses this)

    # ====== Specific Feature Adders ======
    def add_score(self, panel, score):
        self.add_panel_feature(panel, "SDMT_Score", score)

    def add_1_fixation_shearch_percentage(self, panel, percentage):
        self.add_panel_feature(panel, "1_Fixation_Search_Percentage", percentage)

    def add_dispersion(self, panel, dispersion):
        self.add_panel_feature(panel, "Dispersion", dispersion)

    # def add_entropy(self, panel, entropy):
    #     self.add_panel_feature(panel, "Entropy", entropy)

    # ====================================

    # =========== get features ===========
    def get_score(self, panel):
        return self.panel_features.get(panel, {}).get("SDMT_Score", np.nan)
    
    def get_dispersion(self, panel):
        return self.panel_features.get(panel, {}).get("Dispersion", np.nan)
    
    # ====================================

    def compute_mean_matrix(self):
        if not self.matrices: return
        raw_mats = [m.values for m in self.matrices.values()]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_vals = np.nanmean(raw_mats, axis=0)
            
        ref = list(self.matrices.values())[0]
        self.mean_matrix = pd.DataFrame(mean_vals, index=ref.index, columns=ref.columns)

    def get_features_for_panel(self, panel):
        """Returns dict of features for a specific panel."""
        return self.panel_features.get(panel, {})

    def get_mean_features(self):
        """
        Automatically averages ALL numeric features across panels.
        Returns: {'SDMT_Score': 55.5, 'Dispersion': 0.32, ...}
        """
        # 1. Collect all unique keys
        all_keys = set()
        for f_dict in self.panel_features.values():
            all_keys.update(f_dict.keys())
            
        # 2. Compute Mean for each key
        mean_feats = {}
        for key in all_keys:
            values = []
            for f_dict in self.panel_features.values():
                val = f_dict.get(key, np.nan)
                # Only collect numbers
                if isinstance(val, (int, float)) and not np.isnan(val):
                    values.append(val)
            
            if values:
                mean_feats[key] = np.mean(values)
            else:
                mean_feats[key] = np.nan
                
        return mean_feats
    
    def get_mean_score(self):
        mean_feats = self.get_mean_features()
        return mean_feats.get("SDMT_Score", np.nan)
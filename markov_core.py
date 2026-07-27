import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import warnings

@dataclass
class AnalysisConfig:
    with_repeats: bool      
    only_1_to_9: bool       
    use_mean_matrix: bool   
    from_scratch: bool
    concat_panels: bool = False

    raw_behavior_path: str = "/Volumes/ramot/Noam_M/Results/Behavior"
    output_base_path: str = "/Volumes/ramot/Noam_M/preliminary_results"
    
    @property
    def folder_name(self):
        repeat_str = "WITH_repeats" if self.with_repeats else "NO_repeats"
        state_str = "ONLY_1_9" if self.only_1_to_9 else "ALL_states"
        return f"analysis_{repeat_str}_{state_str}"

    @property
    def main_output_path(self):
        return os.path.join(self.output_base_path, self.folder_name)

    @property
    def plot_output_path(self):
        if self.use_mean_matrix:
            sub_folder = "mean_matrix"
        else:
            if self.concat_panels:
                sub_folder = "concat_panels"
            else:
                sub_folder = "all_panels"
        return os.path.join(self.main_output_path, "plots", sub_folder)


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
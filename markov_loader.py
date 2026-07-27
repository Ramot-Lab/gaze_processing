import os
import re
import pandas as pd
import numpy as np
from markov_core import Participant, AnalysisConfig

# Ensure we import the new Logic Class
from gaze_markov_model import GazeMarkovModel
from trial_manager import TrialManager
from participant_gaze_data_manager import ParticipantGazeDataManager

PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]

class DataManager:
    def __init__(self, config: AnalysisConfig):
        self.cfg = config
        self.participants = {"HC": {}, "pwMS": {}}
        
        self.matrix_save_dir = os.path.join(self.cfg.main_output_path, "markov_matrices")
        os.makedirs(self.matrix_save_dir, exist_ok=True)

        self.score_save_dir = os.path.join(self.cfg.output_base_path, "behavior_scores")
        os.makedirs(self.score_save_dir, exist_ok=True)

    def _is_matrix_valid(self, df):
        """
        Gatekeeper Function:
        Returns False if matrix is scientifically useless.
        Criteria:
        1. Empty or None
        2. Not enough data points (all NaNs)
        3. Zero Variance (Flat matrix)
        4. Binary Artifacts (Contains ONLY 0s and 1s)
        """
        if df is None or df.empty:
            return False

        # Flatten to 1D array of values, ignoring NaNs
        values = df.values.flatten()
        mask = ~np.isnan(values)
        
        # 1. Check if we have enough data points (at least 2 valid numbers)
        if np.sum(mask) < 2:
            return False
            
        clean_vals = values[mask]
        
        # 2. Check for Zero Variance (Flat matrix)
        if np.std(clean_vals) < 1e-9:
            return False
            
        # 3. Check for Binary Artifacts (Only 0s and 1s)
        # We check if all unique values are close to 0 or 1
        unique_vals = np.unique(clean_vals)
        is_binary = np.all([np.isclose(x, 0) or np.isclose(x, 1) for x in unique_vals])
        
        if is_binary:
            return False
            
        return True

    def load_participants_and_scores(self):
        print("--- Scanning for Participants & Scores ---")
        for group in ["HC", "pwMS"]:
            g_path = os.path.join(self.cfg.raw_behavior_path, group)
            if not os.path.exists(g_path): continue
            
            for p_name in os.listdir(g_path):
                if os.path.isdir(os.path.join(g_path, p_name)):
                    self.participants[group][p_name] = Participant(p_name, group)
                    self._load_scores_for_participant(group, p_name)

    def _load_scores_for_participant(self, group, p_name):
        p_obj : Participant = self.participants[group][p_name]
        sdmt_path = os.path.join(self.cfg.raw_behavior_path, group, p_name, "SDMT")
        scores_csv_path = os.path.join(self.score_save_dir, "all_scores.csv")

        # 1. Try loading from Master CSV first
        if os.path.exists(scores_csv_path):
            try:
                df = pd.read_csv(scores_csv_path)
                p_scores = df[(df["Group"] == group) & (df["Participant"] == p_name)]
                if not p_scores.empty:
                    for _, row in p_scores.iterrows():
                        panel_name = str(row["Panel"]).strip().lower()
                        # Wrapper handles storage into 'panel_features'
                        p_obj.add_score(panel_name, row["SDMT_Score"])
                    return 
            except Exception as e:
                print(f"Warning: Could not read scores from CSV for {p_name}: {e}")

        # 2. Fallback: Parse WAV files
        if not os.path.exists(sdmt_path): return

        for f in os.listdir(sdmt_path):
            if f.endswith(".wav"):
                pm = re.search(r"img_test_(.+?)_strikes", f)
                sm = re.search(r"_strikes_(\d+)", f)
                if pm and sm:
                    clean_panel = pm.group(1).strip().lower()
                    score = int(sm.group(1))
                    p_obj.add_score(clean_panel, score)

    def save_score_summary_table(self):
        records = []
        for group in self.participants:
            for p_name, p_obj in self.participants[group].items():
                for panel in PANELS:
                    score = p_obj.get_score(panel)
                    
                    records.append({
                        "Group": group,
                        "Participant": p_name,
                        "Panel": panel,
                        "SDMT_Score": score
                    })
        
        df = pd.DataFrame(records)
        save_path = os.path.join(self.score_save_dir, "all_scores.csv")
        df.to_csv(save_path, index=False)
        print(f"✓ Score table saved/updated: {save_path}")


    # def load_or_compute_matrices(self, from_scratch=False):
    #     """
    #     OPTIMIZED: Loads raw data only once per participant.
    #     """
    #     print(f"--- Processing Matrices for: {self.cfg.folder_name} ---")
        
    #     # Iterate Participants
    #     for group in ["HC", "pwMS"]:
    #         # Use list(items) to safely iterate
    #         for p_name, p_obj in list(self.participants[group].items()):
                
    #             user_folder = os.path.join(self.matrix_save_dir, group, p_name)
    #             os.makedirs(user_folder, exist_ok=True)
                
    #             # Check which panels are missing
    #             missing_panels = []
    #             for panel in PANELS:
    #                 csv_path = os.path.join(user_folder, f"matrix_{p_name}_{panel}.csv")
    #                 if from_scratch or not os.path.exists(csv_path):
    #                     missing_panels.append(panel)
    #                 else:
    #                     # Load existing
    #                     try:
    #                         df = pd.read_csv(csv_path, index_col=0)
    #                         p_obj.add_matrix(panel, df)
    #                     except:
    #                         missing_panels.append(panel)

    #             # OPTIMIZATION: Only load raw data if we actually need to compute something
    #             if missing_panels:
    #                 try:
    #                     # Load Raw Data ONCE
    #                     print(f"Loading raw data for {p_name}...") 
    #                     p_data = ParticipantGazeDataManager(p_name, self.cfg.raw_behavior_path, "SDMT", group)
                        
    #                     # Process all missing panels with this one object
    #                     for panel in missing_panels:
    #                         df = self._compute_matrix_with_loaded_data(p_data, panel)
    #                         if df is not None:
    #                             csv_path = os.path.join(user_folder, f"matrix_{p_name}_{panel}.csv")
    #                             df.to_csv(csv_path)
    #                             p_obj.add_matrix(panel, df)
                                
    #                 except Exception as e:
    #                     print(f"Error loading raw data for {p_name}: {e}")

    #             p_obj.compute_mean_matrix()

    # def load_or_compute_matrices(self, from_scratch=False):
    #     print(f"--- Processing Matrices & Features for: {self.cfg.folder_name} ---")
        
    #     for group in self.participants.keys():
    #         for p_name, p_obj in list(self.participants[group].items()):
    #             print(f" - loading information for {p_name} - ")
                
    #             user_folder = os.path.join(self.matrix_save_dir, group, p_name)
    #             os.makedirs(user_folder, exist_ok=True)
                
    #             # 1. Load Existing Matrices
    #             missing_panels = []
    #             for panel in PANELS:
    #                 csv_path = os.path.join(user_folder, f"markov_matrix_{p_name}_panel_{panel}.csv")
    #                 if from_scratch or not os.path.exists(csv_path):
    #                     missing_panels.append(panel)
    #                 else:
    #                     try:
    #                         df = pd.read_csv(csv_path, index_col=0)
    #                         if self._is_matrix_valid(df):
    #                             p_obj.add_matrix(panel, df)
    #                         else:
    #                             print(f" matrix invalid for panel {panel}, {p_name}")
    #                     except Exception as e:
    #                         print(f"Warning: Could not load matrix for {p_name} panel {panel}: {e}")
                
    #             # 2. Compute Extra Features & Missing Matrices
    #             try:
    #                 p_data = ParticipantGazeDataManager(p_name, self.cfg.raw_behavior_path, "SDMT", group)
                    
    #                 # A. Compute Missing Matrices
    #                 for panel in missing_panels:
    #                     df = self._compute_matrix_with_loaded_data(p_data, panel)
    #                     if df is not None:
    #                         csv_path = os.path.join(user_folder, f"markov_matrix_{p_name}_panel_{panel}.csv")
    #                         df.to_csv(csv_path)
    #                         p_obj.add_matrix(panel, df)
                    
                    # B. Compute Extra Features (Dispersion, Duration, etc.)
                    # for panel in PANELS:
                    #     # Only compute for panels we actually have data for
                    #     if panel not in p_obj.matrices and panel not in missing_panels: 
                    #         continue
                        
                        # dur = self._calculate_duration(p_data, panel)
                        # p_obj.add_panel_feature(panel, "Duration", dur)
                        
                        # disp = self._calculate_dispersion(p_data, panel)
                        # p_obj.add_panel_feature(panel, "Dispersion", disp)

                # except Exception as e:
                #     pass

                # p_obj.compute_mean_matrix()

    def load_or_compute_matrices(self, from_scratch=False):
        """
        Simplified Logic:
        - If from_scratch=True: IGNORE disk. Compute ALL matrices fresh. Save them.
        - If from_scratch=False: READ disk. Do NOT compute anything missing.
        """
        print(f"--- Processing Matrices & Features for: {self.cfg.folder_name} ---")
        
        for group in self.participants.keys():
            for p_name, p_obj in list(self.participants[group].items()):
                print(f"- loading information for {p_name} - ")
                
                user_folder = os.path.join(self.matrix_save_dir, group, p_name)
                os.makedirs(user_folder, exist_ok=True)
                
                # =========================================================
                # MODE 1: COMPUTE FRESH (Ignore existing files)
                # =========================================================
                if from_scratch:
                    try:
                        # Initialize raw data reader
                        p_data = ParticipantGazeDataManager(p_name, self.cfg.raw_behavior_path, "SDMT", group)
                        
                        for panel in PANELS:
                            # 1. Compute
                            result = self._compute_matrix_with_loaded_data(p_data, panel)
                            
                            # 2. Validate & Save
                            if result is not None:
                                matrix_df, trial_stats_df = result

                                csv_path = os.path.join(user_folder, f"markov_matrix_{p_name}_panel_{panel}.csv")
                                matrix_df.to_csv(csv_path)
                                p_obj.add_matrix(panel, matrix_df)

                                if trial_stats_df is not None:
                                    trial_path = os.path.join(user_folder, f"trial_stats_{p_name}_panel_{panel}.csv")
                                    trial_stats_df.to_csv(trial_path, index=False)
                                    p_obj.add_trial_data(panel, trial_stats_df)

                    except Exception as e:
                        print(f"Error computing fresh data for {p_name}: {e}")
                        pass

                # =========================================================
                # MODE 2: LOAD EXISTING ONLY (Do not compute missing)
                # =========================================================
                else:
                    for panel in PANELS:
                        # 1. Load Matrix
                        mat_path = os.path.join(user_folder, f"markov_matrix_{p_name}_panel_{panel}.csv")
                        if os.path.exists(mat_path):
                            try:
                                df = pd.read_csv(mat_path, index_col=0)
                                if self._is_matrix_valid(df):
                                    p_obj.add_matrix(panel, df)
                            except: pass
                        
                        # 2. Load Trial Stats
                        trial_path = os.path.join(user_folder, f"trial_stats_{p_name}_panel_{panel}.csv")
                        if os.path.exists(trial_path):
                            try:
                                t_df = pd.read_csv(trial_path)
                                p_obj.add_trial_data(panel, t_df)
                            except: pass


                            # # ===== Compute Extra Features =====
                            # dur = self._calculate_duration(p_data, panel)
                            # p_obj.add_panel_feature(panel, "Duration", dur)
                            
                            # disp = self._calculate_dispersion(p_data, panel)
                            # p_obj.add_panel_feature(panel, "Dispersion", disp)
                # Finally, compute Mean Matrix if applicable
                p_obj.compute_mean_matrix()

    def _compute_matrix_with_loaded_data(self, p_data, panel):
        try:
            trial_mgr = TrialManager(p_data, panel)
            trial_df_stats = trial_mgr.add_trial_statistics_dataframe()
            gaze_model = GazeMarkovModel(trial_mgr.trials, only_keys=self.cfg.only_1_to_9)
            raw_df = gaze_model.get_probability_matrix(clean=not self.cfg.with_repeats)
            if self._is_matrix_valid(raw_df) is False:
                return None
            return raw_df, trial_df_stats
        except Exception as e:
            print(f"Error computing matrix for panel {panel}, participant {p_data.participant_name}: {e}")
            print(trial_df_stats if 'trial_df_stats' in locals() else "No trial stats available.")
            return e

    # def _calculate_duration(self, p_data, panel):
    #     try:
    #         df = p_data.get_trial_data(panel) 
    #         if df is None or df.empty: return np.nan
    #         return (df['time'].iloc[-1] - df['time'].iloc[0]) / 1000.0 
    #     except:
    #         return np.nan

    # def _calculate_dispersion(self, p_data, panel):
    #     try:
    #         df = p_data.get_trial_data(panel)
    #         if df is None or df.empty: return np.nan
    #         x = df['x'].dropna()
    #         y = df['y'].dropna()
    #         if len(x) < 2: return np.nan
    #         return (np.std(x) + np.std(y)) / 2.0
    #     except:
    #         return np.nan

    def get_flat_participants(self):
        all_p = []
        for g in self.participants:
            all_p.extend(self.participants[g].values())
        return [p for p in all_p if p.mean_matrix is not None]
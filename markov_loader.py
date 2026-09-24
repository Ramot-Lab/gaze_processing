import os
import re
import pandas as pd
import numpy as np
from markov_core import Participant, AnalysisConfig

# Ensure we import the new Logic Class
from gaze_markov_model import GazeMarkovModel
from trial_manager import TrialManager
from participant_gaze_data_manager import ParticipantGazeDataManager
import pipeline_config

PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]

class DataManager:
    def __init__(self, config: AnalysisConfig):
        self.cfg = config
        self.participants = {"HC": {}, "pwMS": {}}

        self.matrix_save_dir = os.path.join(self.cfg.main_output_path, "markov_matrices")
        os.makedirs(self.matrix_save_dir, exist_ok=True)

        self.score_save_dir = os.path.join(self.cfg.output_base_path, "behavior_scores")
        os.makedirs(self.score_save_dir, exist_ok=True)

        # One row per participant/panel dropped during load_or_compute_matrices - saved to
        # exclusion_log.csv the same way Stage 1 logs its own drops, so both stages'
        # exclusions are comparable across methods.
        self.exclusions = []

    def _log_exclusion(self, participant, group, panel, reason):
        self.exclusions.append({"participant": participant, "group": group, "panel": panel, "reason": reason})

    def save_exclusion_log(self):
        out_path = os.path.join(pipeline_config.markov_output_dir(self.cfg.annotation_method), "exclusion_log.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(self.exclusions).to_csv(out_path, index=False)
        print(f"Saved {len(self.exclusions)} exclusion rows to {out_path}")

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
                        # Full path, not bare p_name - required for sd.group/self.name to
                        # come out right inside ParticipantGazeDataManager.
                        subject_dir = os.path.join(self.cfg.raw_behavior_path, group, p_name)
                        p_data = ParticipantGazeDataManager(subject_dir, self.cfg.raw_behavior_path, "SDMT", group)
                    except Exception as e:
                        print(f"Error loading participant data for {p_name}: {e}")
                        self._log_exclusion(p_name, group, None, f"participant_load_error: {type(e).__name__}: {e}")
                        continue

                    # One panel's failure must not abort the participant's other panels -
                    # each panel gets its own try/except, not one shared around the loop.
                    for panel in PANELS:
                        try:
                            result, fail_reason = self._compute_matrix_with_loaded_data(p_data, panel, group=group)
                        except Exception as e:
                            result, fail_reason = None, f"unexpected: {type(e).__name__}: {e}"

                        if result is None:
                            self._log_exclusion(p_name, group, panel, fail_reason or "unknown")
                            continue

                        matrix_df, trial_stats_df = result
                        csv_path = os.path.join(user_folder, f"markov_matrix_{p_name}_panel_{panel}.csv")
                        matrix_df.to_csv(csv_path)
                        p_obj.add_matrix(panel, matrix_df)

                        if trial_stats_df is not None:
                            trial_path = os.path.join(user_folder, f"trial_stats_{p_name}_panel_{panel}.csv")
                            trial_stats_df.to_csv(trial_path, index=False)
                            p_obj.add_trial_data(panel, trial_stats_df)

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

    def _compute_matrix_with_loaded_data(self, p_data, panel, group=None):
        """
        p_data is still a live ParticipantGazeDataManager - TrialManager needs it for
        message/press timing (PanelMessages) and the panel image, neither of which is
        part of the saved gaze-annotation CSV. What this DOES skip is re-running
        annotation live (annotate_gaze_events): the gaze+evt data is loaded straight from
        Stage 1's saved CSV for self.cfg.annotation_method instead, which matters a lot
        for model_based (skips re-running inference here). If Stage 1 excluded this
        participant/panel, that's reported as-is (prefixed "preprocessing_exclusion:")
        instead of falling back to live annotation and re-discovering the same gap as a
        fresh, confusingly-labeled failure. Only falls back to live annotation if Stage 1
        genuinely has no record of this participant/panel at all (e.g. it hasn't been run
        for this method/date yet), so this can still work before or without Stage 1.
        """
        try:
            group = group or p_data.group
            try:
                # corrected=True (decision 2026-09-22): Markov chain analysis's ROI
                # matching is y-value-dependent (dictionary-area search sequencing), so it
                # always reads the whole-dictionary-drift-corrected gaze, not raw.
                annotated_data = pipeline_config.load_annotated_csv(
                    self.cfg.annotation_method, group, p_data.name, panel, corrected=True)
            except FileNotFoundError:
                stage1_reason = pipeline_config.stage1_exclusion_reason(
                    self.cfg.annotation_method, p_data.name, panel)
                if stage1_reason is not None:
                    return None, f"preprocessing_exclusion: {stage1_reason}"
                print(f"  no Stage-1 CSV for {p_data.name}/{panel}/{self.cfg.annotation_method} - "
                      f"falling back to live annotation")
                annotated_data = None

            trial_mgr = TrialManager(p_data, panel, annotated_data=annotated_data,
                                      annotation_method=self.cfg.annotation_method)
            trial_df_stats = trial_mgr.add_trial_statistics_dataframe()
            gaze_model = GazeMarkovModel(trial_mgr.trials, only_keys=self.cfg.only_1_to_9)
            raw_df = gaze_model.get_probability_matrix(clean=not self.cfg.with_repeats)
            if self._is_matrix_valid(raw_df) is False:
                return None, "invalid/empty transition matrix"
            return (raw_df, trial_df_stats), None
        except Exception as e:
            print(f"Error computing matrix for panel {panel}, participant {p_data.name}: {e}")
            print(trial_df_stats if 'trial_df_stats' in locals() else "No trial stats available.")
            return None, f"{type(e).__name__}: {e}"

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
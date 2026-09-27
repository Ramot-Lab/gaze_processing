import os
import re
import pandas as pd
import numpy as np
from markov_core import Participant, AnalysisConfig

# Ensure we import the new Logic Class
from gaze_markov_model import GazeMarkovModel
from trial_manager import TrialManager
from participant_gaze_data_manager import ParticipantGazeDataManager
from run_preprocessing import iter_all_subject_dirs
from exclusion_policy import load_manually_excluded_participants
import pipeline_config

PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]

# Decision 2026-09-27: a participant with fewer than this many valid per-panel transition
# matrices isn't a reliable basis for a mean matrix - excluded from Markov analysis
# entirely (kept out of get_flat_participants() via filter_by_min_panels), same as any
# other Stage 1 drop, but logged with its own reason so it's traceable.
MIN_PANELS_FOR_MARKOV = 4

class DataManager:
    def __init__(self, config: AnalysisConfig):
        self.cfg = config
        self.participants = {"HC": {}, "pwMS": {}}

        self.matrix_save_dir = os.path.join(self.cfg.main_output_path, "markov_matrices")
        os.makedirs(self.matrix_save_dir, exist_ok=True)

        # Nothing ever writes here (save_score_summary_table is dead code - never
        # called) - _load_scores_for_participant only ever READS from this path
        # (falling back to WAV-file parsing if it's missing), so the folder doesn't
        # need to be pre-created (decision 2026-09-27).
        self.score_save_dir = os.path.join(self.cfg.output_base_path, "behavior_scores")

        # One row per participant/panel dropped during load_or_compute_matrices - saved to
        # exclusion_log.csv the same way Stage 1 logs its own drops, so both stages'
        # exclusions are comparable across methods.
        self.exclusions = []

    def _log_exclusion(self, participant, group, panel, reason):
        self.exclusions.append({"participant": participant, "group": group, "panel": panel, "reason": reason})

    def save_exclusion_log(self):
        # self.cfg.main_output_path (derived from output_base_path) - NOT
        # pipeline_config.markov_output_dir(), which always recomputes the global
        # default location and silently ignores any output_base_path/date_str override
        # (decision 2026-09-27, found via a consolidated/test run's exclusion log
        # landing in the wrong, global folder instead of the intended one).
        out_path = os.path.join(self.cfg.main_output_path, "exclusion_log.csv")
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
        part of the saved gaze-annotation CSV. This NEVER re-runs annotation live
        (annotate_gaze_events): the gaze+evt data is loaded straight from Stage 1's
        saved CSV for self.cfg.annotation_method. If Stage 1 excluded this
        participant/panel, that's reported as-is (prefixed "preprocessing_exclusion:").
        If Stage 1 has no record of this participant/panel at all - genuinely missing,
        not excluded - that's ALSO just reported and skipped (decision 2026-09-27: Stage
        2/3 only ever reads Stage 1's saved output, live re-annotation here is disabled
        by policy; recomputing model_based inference on demand is exactly the slow,
        unattended-unfriendly cost Stage 1 exists to pay once, up front)."""
        try:
            group = group or p_data.group
            try:
                # corrected=True (decision 2026-09-22): Markov chain analysis's ROI
                # matching is y-value-dependent (dictionary-area search sequencing), so it
                # always reads the whole-dictionary-drift-corrected gaze, not raw.
                annotated_data = pipeline_config.load_annotated_csv(
                    self.cfg.annotation_method, group, p_data.name, panel,
                    date_str=self.cfg.date_str, corrected=True)
            except FileNotFoundError:
                stage1_reason = pipeline_config.stage1_exclusion_reason(
                    self.cfg.annotation_method, p_data.name, panel, date_str=self.cfg.date_str)
                if stage1_reason is not None:
                    return None, f"preprocessing_exclusion: {stage1_reason}"
                return None, (f"no Stage 1 annotated CSV found for {p_data.name}/{panel}/"
                              f"{self.cfg.annotation_method} and no exclusion reason logged - "
                              f"live re-annotation is disabled here by policy")

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

    def filter_by_min_panels(self, min_panels=MIN_PANELS_FOR_MARKOV):
        """Drop participants with 1..min_panels-1 valid panel matrices from the analysis
        population - some data, but not enough to trust a mean matrix built from it.
        Participants with 0 panels are already excluded elsewhere (compute_mean_matrix
        leaves mean_matrix=None, and get_flat_participants filters on that), so this only
        needs to catch the partial case. Call after load_or_compute_matrices, before
        save_exclusion_log so this shows up in the same exclusion_log.csv."""
        for group in self.participants:
            for p_name, p_obj in list(self.participants[group].items()):
                n_panels = len(p_obj.matrices)
                if 0 < n_panels < min_panels:
                    self._log_exclusion(
                        p_name, group, None,
                        f"only {n_panels} valid panel matrice(s) for Markov analysis "
                        f"(< {min_panels} required)")
                    del self.participants[group][p_name]

    def compute_exclusion_summary(self, min_panels=MIN_PANELS_FOR_MARKOV):
        """One row of participant accounting for this analysis, categorized the way the
        user wants to see it broken down: hardcoded (exclusion_policy's manual list),
        panel_count (had some data but fewer than min_panels valid matrices), technical
        (everything else - Stage 1's accuracy/NaN/saccade/never-visits-.../corrupted/
        load-error drops, zero panels for any reason). Uses run_preprocessing's own
        eligible-participant universe (excludes DONTUSE) as the denominator, and must be
        called BEFORE filter_by_min_panels (needs the pre-filter panel counts). Also
        reports the group/sex breakdown of the INCLUDED population specifically (same
        source table as markov_analyzer.py's demographics annotation), so this one row
        doubles as the same demographic accounting shown on every plot."""
        import pandas as pd
        from exclusion_policy import DEFAULT_TOBII_SUCKS_XLSX
        try:
            demo = pd.read_excel(DEFAULT_TOBII_SUCKS_XLSX)[["Patient_ID", "Gender"]]
            gender_map = demo.set_index("Patient_ID")["Gender"].astype(str).str.strip().str.upper().str[:1].to_dict()
        except Exception:
            gender_map = {}

        manual = load_manually_excluded_participants()
        n_included = n_hardcoded = n_panel_count = n_technical = 0
        n_hc = n_ms = n_female = n_male = 0
        for group, subject_dir in iter_all_subject_dirs(self.cfg.raw_behavior_path):
            p_name = os.path.basename(subject_dir)
            p_obj = self.participants.get(group, {}).get(p_name)
            n_panels = len(p_obj.matrices) if p_obj is not None else 0
            if p_name in manual:
                n_hardcoded += 1
                continue
            if n_panels >= min_panels:
                n_included += 1
                n_hc += group == "HC"
                n_ms += group != "HC"
                gender = gender_map.get(p_name, "")
                n_female += gender == "F"
                n_male += gender == "M"
            elif n_panels > 0:
                n_panel_count += 1
            else:
                n_technical += 1
        return {
            "n_total_eligible": n_included + n_technical + n_panel_count + n_hardcoded,
            "n_included": n_included,
            "n_excluded_technical": n_technical,
            "n_excluded_panel_count": n_panel_count,
            "n_excluded_hardcoded": n_hardcoded,
            "n_hc": n_hc,
            "n_ms": n_ms,
            "n_female": n_female,
            "n_male": n_male,
        }

    def get_flat_participants(self):
        all_p = []
        for g in self.participants:
            all_p.extend(self.participants[g].values())
        return [p for p in all_p if p.mean_matrix is not None]
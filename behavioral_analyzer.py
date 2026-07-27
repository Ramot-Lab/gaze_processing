import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from participant_gaze_data_manager import ParticipantGazeDataManager
from trial_manager import TrialManager

PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]

class BehavioralAnalyzer:
    def __init__(self, participants, config):
        self.participants = participants
        self.config = config
        # --- NEW FOLDER NAME ---
        self.output_dir = os.path.join(config.main_output_path, "behavioral_poster")
        
        # Sub-folders
        self.plot_dir = os.path.join(self.output_dir, "plots_per_trial")
        self.stats_dir = os.path.join(self.output_dir, "trial_type_stats")
        self.corr_dir = os.path.join(self.output_dir, "correlations")
        
        for d in [self.plot_dir, self.stats_dir, self.corr_dir]:
            os.makedirs(d, exist_ok=True)

    # =========================================================================
    # MASTER DATA LOADER
    # =========================================================================
    def populate_trial_stats(self, force_recompute=False):
        """
        Master load function:
        1. Loads Cached Summary Stats (Fast) -> Populates p.panel_features
        2. Loads/Computes Granular Trial Data (Slow) -> Populates p.trial_data
        """
        print("\n--- Populating Behavioral Data ---")
        
        # 1. FAST LOAD: Try Loading Global Summary Table first
        csv_path = os.path.join(self.stats_dir, "all_trial_type_stats.csv")
        if os.path.exists(csv_path) and not force_recompute:
            try:
                print(f"   Loading cached summary stats from: {csv_path}")
                df_all = pd.read_csv(csv_path)
                self._restore_features_from_df(df_all)
            except Exception as e:
                print(f"   Warning: Failed to load summary cache ({e})")

        # 2. GRANULAR LOAD: Scan participants for trial-by-trial data
        print("   Checking granular trial data...")
        for p in self.participants:
            existing_panels = set(p.trial_data.keys())
            missing_panels = [pan for pan in PANELS if pan not in existing_panels]
            
            if not missing_panels and not force_recompute:
                continue

            print(f"   Computing granular stats for {p.name} (Missing: {missing_panels})...")
            
            try:
                # Initialize Raw Data Manager (Heavy IO)
                p_data = ParticipantGazeDataManager(p.name, self.config.raw_behavior_path, "SDMT", p.group)
                
                for panel in PANELS:
                    if panel in p.trial_data and not force_recompute:
                        continue
                        
                    try:
                        tm = TrialManager(p_data, panel)
                        
                        stats_data = []
                        for i, trial in enumerate(tm.trials):
                            n_fix = 0
                            n_sym = 0
                            n_unique = 0
                            
                            if hasattr(trial, 'searches') and trial.searches:
                                for s in trial.searches:
                                    # --- YOUR SPECIFIC LOGIC PRESERVED ---
                                    seq = s.sequence
                                    cleaned_seq = s.cleaned_sequence
                                    if cleaned_seq:
                                        n_sym += len(cleaned_seq)
                                        n_unique += len(set(cleaned_seq))
                                    if seq:
                                        n_fix += len(seq)
                                    # -------------------------------------

                            stats_data.append({
                                "Trial_Index": i,
                                "Num_Fixations": n_fix,
                                "Num_Symbols": n_sym,
                                "Unique_Symbols_Count": n_unique,
                                "Is_Zero_Fix": n_fix == 0,
                                "Is_One_Fix": n_fix == 1,
                                "Is_Two_Fix": n_fix == 2,
                                "Is_Zero_Symbol": n_sym == 0,
                                "Is_One_Symbol": n_sym == 1,
                                "Is_Two_Symbols": n_sym == 2
                            })
                        
                        if stats_data:
                            df_stats = pd.DataFrame(stats_data)
                            p.add_trial_data(panel, df_stats)
                            
                    except Exception:
                        pass
            except Exception as e_p:
                print(f"   Error loading raw data for {p.name}: {e_p}")

    # =========================================================================
    # TASK 2: Line Plots
    # =========================================================================
    def plot_metric_across_trials(self, metric="Num_Fixations", title_suffix="Fixations"):
        print(f"--- Plotting {title_suffix} per Trial (Mean + Error) ---")
        panels = self._get_valid_panels()
        
        for panel in panels:
            all_records = []
            for p in self.participants:
                if panel in p.trial_data:
                    df = p.trial_data[panel].copy()
                    if metric in df.columns:
                        df["Group"] = p.group
                        df["Participant"] = p.name
                        all_records.append(df[["Trial_Index", metric, "Group", "Participant"]])
            
            if not all_records: continue
            
            df_panel = pd.concat(all_records, ignore_index=True)
            
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=df_panel, x="Trial_Index", y=metric, hue="Group", 
                         palette={"HC": "blue", "pwMS": "red"}, 
                         errorbar=('ci', 95)) 
            
            plt.title(f"{title_suffix} per Trial - Panel {panel} (Mean ± 95% CI)")
            plt.xlabel("Trial Number")
            plt.ylabel(title_suffix)
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.plot_dir, f"lineplot_GroupMean_{metric}_{panel}.png")
            plt.savefig(save_path)
            plt.close()

    # =========================================================================
    # TASK 3: Statistics (Compute if missing, then Plot)
    # =========================================================================
    def compute_and_plot_trial_types(self):
        """
        1. Checks if data was loaded in 'populate'.
        2. If not, Computes summary from granular data & Saves.
        3. Plots results.
        """
        print("--- Generating Trial Type Statistics ---")
        
        csv_path = os.path.join(self.stats_dir, "all_trial_type_stats.csv")
        long_data = []

        # Config: (Category, Column, Numeric_Level, Readable_Label)
        config_map = [
            ("Fixations", "Is_Zero_Fix",    "0", "Zero Fixations"),
            ("Fixations", "Is_One_Fix",     "1", "One Fixation"),
            ("Fixations", "Is_Two_Fix",     "2", "Two Fixations"),
            ("Symbols",   "Is_Zero_Symbol", "0", "Zero Symbols"),
            ("Symbols",   "Is_One_Symbol",  "1", "One Symbol"),
            ("Symbols",   "Is_Two_Symbols", "2", "Two Symbols"),
            ()
        ]

        # Scan participants (Data should be in memory now, either from CSV load or Granular load)
        has_data = False
        
        for p in self.participants:
            # METHOD A: Extract from Granular (Preferred if available)
            if p.trial_data:
                has_data = True
                for panel, df in p.trial_data.items():
                    if df.empty: continue
                    
                    # Means
                    if "Num_Fixations" in df.columns:
                        val = df["Num_Fixations"].mean()
                        p.add_panel_feature(panel, "Mean_Fixations", val)
                        long_data.append(self._make_record(p, panel, "Fixations", "Mean", "All", "Average", val))
                    if "Num_Symbols" in df.columns:
                        val = df["Num_Symbols"].mean()
                        p.add_panel_feature(panel, "Mean_Symbols", val)
                        long_data.append(self._make_record(p, panel, "Symbols", "Mean", "All", "Average", val))

                    # Trial Types
                    for category, col_name, level, label in config_map:
                        if col_name in df.columns:
                            pct = df[col_name].mean() * 100
                            count = df[col_name].sum()
                            
                            # Add to Features (for correlations)
                            p.add_panel_feature(panel, f"Pct_{'Zero' if level=='0' else 'One' if level=='1' else 'Two'}_{category[:3]}", pct)
                            
                            # Add to Plotting List
                            long_data.append(self._make_record(p, panel, category, "Percentage", level, label, pct))
                            long_data.append(self._make_record(p, panel, category, "Count", level, label, count))
            
        if not has_data and os.path.exists(csv_path):
             # If granular data missing but CSV exists, just read CSV for plotting
             df_all = pd.read_csv(csv_path)
        else:
            df_all = pd.DataFrame(long_data)
            # Save if we just computed it
            if not df_all.empty:
                df_all.to_csv(csv_path, index=False)
                print(f"   [Saved] Complete statistics table to: {csv_path}")

        if df_all.empty: return

        self._generate_stat_plots(df_all)
    def compute_and_plot_unique_symbols_distribution(self):
        print("--- Computing Unique Symbols Distribution ---")
        
        long_data = []
        possible_counts = list(range(0, 10)) 

        for p in self.participants:
            for panel, df in p.trial_data.items():
                if df.empty or "Unique_Symbols_Count" not in df.columns: 
                    continue
                
                total_trials = len(df)
                if total_trials == 0: continue

                # Count how many trials had exactly X unique symbols
                counts = df["Unique_Symbols_Count"].value_counts()

                for n_unique in possible_counts:
                    count_val = counts.get(n_unique, 0)
                    pct_val = (count_val / total_trials) * 100

                    # 1. Record Percentage
                    long_data.append({
                        "Participant": p.name, "Group": p.group, "Panel": panel,
                        "Measure": "Percentage",
                        "N_Unique_Symbols": n_unique,
                        "Value": pct_val
                    })

                    # 2. Record Count
                    long_data.append({
                        "Participant": p.name, "Group": p.group, "Panel": panel,
                        "Measure": "Count",
                        "N_Unique_Symbols": n_unique,
                        "Value": count_val
                    })

        df_all = pd.DataFrame(long_data)
        if df_all.empty: return

        # --- SAVE THE TABLE TO CSV ---
        csv_path = os.path.join(self.stats_dir, "unique_symbols_distribution.csv")
        df_all.to_csv(csv_path, index=False)
        print(f"   [Saved] Unique symbols distribution table to: {csv_path}")

        # --- Plotting Helper (Poster Style) ---
        def plot_dist(data, title_prefix, filename_prefix):
            # Apply Poster Context
            sns.set_context("poster", font_scale=1.2)
            sns.set_style("whitegrid")

            for measure in ["Percentage", "Count"]:
                subset = data[data["Measure"] == measure]
                if subset.empty: continue

                # Bigger Figure
                plt.figure(figsize=(14, 10))
                
                sns.barplot(data=subset, x="N_Unique_Symbols", y="Value", hue="Group",
                            palette={"HC": "blue", "pwMS": "red"}, 
                            errorbar=('se', 1), capsize=0.1)
                
                # Big Title & Labels
                plt.title(f"{title_prefix}: Unique Symbols Sampled ({measure})", fontsize=35, fontweight='bold', pad=30)
                plt.xlabel("Number of Unique Symbols in Search", fontsize=30, fontweight='bold', labelpad=20)
                plt.ylabel(f"{measure} of Trials", fontsize=30, fontweight='bold', labelpad=20)
                
                # Big Ticks
                plt.xticks(fontsize=28, fontweight='bold')
                plt.yticks(fontsize=25)
                plt.grid(True, axis='y', alpha=0.3)
                
                # Big Legend
                plt.legend(title="Group", fontsize=22, title_fontsize=24, loc='upper right')
                
                # Clean Layout
                sns.despine(trim=True)
                plt.tight_layout()
                
                fname = f"{filename_prefix}_UniqueSymbols_{measure}.png"
                plt.savefig(os.path.join(self.stats_dir, fname), dpi=300)
                plt.close()

        # 1. Plot Per Panel
        panels = self._get_valid_panels()
        for panel in panels:
            panel_data = df_all[df_all["Panel"] == panel]
            plot_dist(panel_data, f"Panel {panel}", f"barplot_Panel_{panel}")

        # 2. Plot Global Average
        df_avg = df_all.groupby(["Participant", "Group", "Measure", "N_Unique_Symbols"])["Value"].mean().reset_index()
        plot_dist(df_avg, "Average Across All Panels", "barplot_Global_Average")
    # def compute_and_plot_unique_symbols_distribution(self):
    #     print("--- Computing Unique Symbols Distribution ---")
        
    #     long_data = []
    #     # UPDATED: Range now starts from 0 to include "No Search" trials
    #     possible_counts = list(range(0, 10)) 

    #     for p in self.participants:
    #         for panel, df in p.trial_data.items():
    #             if df.empty or "Unique_Symbols_Count" not in df.columns: 
    #                 continue
                
    #             total_trials = len(df)
    #             if total_trials == 0: continue

    #             # Count how many trials had exactly X unique symbols
    #             counts = df["Unique_Symbols_Count"].value_counts()

    #             for n_unique in possible_counts:
    #                 count_val = counts.get(n_unique, 0)
    #                 pct_val = (count_val / total_trials) * 100

    #                 # 1. Record Percentage
    #                 long_data.append({
    #                     "Participant": p.name, "Group": p.group, "Panel": panel,
    #                     "Measure": "Percentage",
    #                     "N_Unique_Symbols": n_unique,
    #                     "Value": pct_val
    #                 })

    #                 # 2. Record Count
    #                 long_data.append({
    #                     "Participant": p.name, "Group": p.group, "Panel": panel,
    #                     "Measure": "Count",
    #                     "N_Unique_Symbols": n_unique,
    #                     "Value": count_val
    #                 })

    #     df_all = pd.DataFrame(long_data)
    #     if df_all.empty: return

    #     # --- SAVE THE TABLE TO CSV ---
    #     csv_path = os.path.join(self.stats_dir, "unique_symbols_distribution.csv")
    #     df_all.to_csv(csv_path, index=False)
    #     print(f"   [Saved] Unique symbols distribution table to: {csv_path}")

    #     # --- Plotting Helper ---
    #     def plot_dist(data, title_prefix, filename_prefix):
    #         for measure in ["Percentage", "Count"]:
    #             subset = data[data["Measure"] == measure]
    #             if subset.empty: continue

    #             plt.figure(figsize=(10, 6))
                
    #             sns.barplot(data=subset, x="N_Unique_Symbols", y="Value", hue="Group",
    #                         palette={"HC": "blue", "pwMS": "red"}, 
    #                         errorbar=('se', 1), capsize=0.1)
                
    #             plt.title(f"{title_prefix}: Unique Symbols Sampled ({measure})")
    #             plt.xlabel("Number of Unique Symbols in Search")
    #             plt.ylabel(f"{measure} of Trials")
    #             plt.grid(True, axis='y', alpha=0.3)
                
    #             fname = f"{filename_prefix}_UniqueSymbols_{measure}.png"
    #             plt.savefig(os.path.join(self.stats_dir, fname))
    #             plt.close()

    #     # 1. Plot Per Panel
    #     panels = self._get_valid_panels()
    #     for panel in panels:
    #         panel_data = df_all[df_all["Panel"] == panel]
    #         plot_dist(panel_data, f"Panel {panel}", f"barplot_Panel_{panel}")

    #     # 2. Plot Global Average
    #     df_avg = df_all.groupby(["Participant", "Group", "Measure", "N_Unique_Symbols"])["Value"].mean().reset_index()
    #     plot_dist(df_avg, "Average Across All Panels", "barplot_Global_Average")

    def plot_metric_global_average(self, metric="Num_Symbols", title_suffix="Symbols"):
        print(f"--- Plotting Global Average {title_suffix} (Mean ± SEM) with 80% Cutoff ---")
        
        all_records = []
        
        # 1. Collect data from ALL panels for ALL participants
        for p in self.participants:
            for panel, df in p.trial_data.items():
                if metric in df.columns:
                    # Extract the columns we need
                    sub = df[["Trial_Index", metric]].copy()
                    sub["Group"] = p.group
                    sub["Participant"] = p.name
                    all_records.append(sub)
        
        if not all_records: 
            print("   No data found for plotting.")
            return
        
        # 2. Combine into one giant table
        df_global = pd.concat(all_records, ignore_index=True)

        # --- 3. APPLY 80% PARTICIPATION CUTOFF ---
        # A. Calculate total unique participants per group
        total_per_group = df_global.groupby("Group")["Participant"].nunique()
        
        # B. Calculate participation count per Trial_Index per Group
        # We assume a participant is "present" in a trial if they have at least one data point (from any panel)
        trial_counts = df_global.groupby(["Trial_Index", "Group"])["Participant"].nunique()

        # C. Find the first trial where < 80% remain
        cutoff_trial = df_global["Trial_Index"].max() + 1 # Default to end of data

        for trial in sorted(df_global["Trial_Index"].unique()):
            groups_ok = True
            for group in total_per_group.index:
                # How many people in this group have data for this trial?
                count = trial_counts.get((trial, group), 0)
                threshold = total_per_group[group] * 0.8
                
                if count < threshold:
                    groups_ok = False
                    break
            
            if not groups_ok:
                cutoff_trial = trial
                print(f"   [Cutoff Applied] Stopping at Trial {cutoff_trial} (Participation < 80%)")
                break
        
        # D. Filter the data
        df_global = df_global[df_global["Trial_Index"] < cutoff_trial]

        if df_global.empty:
            print("   Error: Cutoff removed all data.")
            return

        # --- 4. PLOT ---
        plt.figure(figsize=(12, 8))
        
        sns.lineplot(data=df_global, x="Trial_Index", y=metric, hue="Group", 
                     palette={"HC": "blue", "pwMS": "red"}, 
                     errorbar='se') 
        
        plt.title(f"{title_suffix} per Trial - Average Across All Panels (Mean ± SEM)\n(Trials with >80% participation)")
        plt.xlabel("Trial Number")
        plt.ylabel(title_suffix)
        plt.grid(True, alpha=0.3)
        
        # 5. Save with "Global" in the filename
        save_path = os.path.join(self.plot_dir, f"lineplot_GlobalMean_{metric}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"   [Saved] Global plot to: {save_path}")

    # =========================================================================
    # TASK 4: Correlations
    # =========================================================================
    def analyze_correlations_with_score(self):
        print("--- Analyzing Correlations with Score ---")
        panels = self._get_valid_panels()
        
        features_to_corr = [
            "Pct_Zero_Fix", "Pct_One_Fix", "Pct_Two_Fix",
            "Pct_Zero_Sym", "Pct_One_Sym", "Pct_Two_Sym", 
            "Mean_Fixations", "Mean_Symbols"
        ]
        
        summary_results = []

        for panel in panels:
            for feat in features_to_corr:
                scores = []
                values = []
                groups = []
                
                for p in self.participants:
                    s = p.get_score(panel)
                    v = p.get_features_for_panel(panel).get(feat, np.nan)
                    
                    if not np.isnan(s) and not np.isnan(v):
                        scores.append(s)
                        values.append(v)
                        groups.append(p.group)
                
                if len(scores) < 5: continue
                
                r, p_val = pearsonr(values, scores)
                summary_results.append({"Panel": panel, "Feature": feat, "R": r, "P": p_val})
                
                plt.figure(figsize=(6, 6))
                sns.scatterplot(x=values, y=scores, hue=groups, palette={"HC": "blue", "pwMS": "red"})
                sns.regplot(x=values, y=scores, scatter=False, color="gray", line_kws={"linestyle": "--"})
                plt.title(f"{panel}: {feat} vs Score\nr={r:.2f}, p={p_val:.4f}")
                plt.xlabel(feat)
                plt.ylabel("SDMT Score")
                plt.tight_layout()
                plt.savefig(os.path.join(self.corr_dir, f"corr_{panel}_{feat}.png"))
                plt.close()

        pd.DataFrame(summary_results).to_csv(os.path.join(self.corr_dir, "correlation_summary.csv"), index=False)

    

    # --- HELPERS ---

    def _get_valid_panels(self):
        panels = set()
        for p in self.participants:
            panels.update(p.trial_data.keys()) 
        return sorted(list(panels))

    def _make_record(self, p, panel, cat, measure, level, cond, val):
        return {
            "Participant": p.name, "Group": p.group, "Panel": panel,
            "Category": cat, "Measure": measure, 
            "Level": level,      
            "Condition": cond,   
            "Value": val
        }

    def _restore_features_from_df(self, df):
        p_map = {p.name: p for p in self.participants}
        for _, row in df.iterrows():
            p = p_map.get(row["Participant"])
            if not p: continue
            
            panel = row["Panel"]
            cat = row["Category"]
            measure = row["Measure"]
            level = str(row["Level"])
            val = row["Value"]
            
            if measure == "Mean":
                p.add_panel_feature(panel, f"Mean_{cat}", val)
            elif measure == "Percentage":
                level_name = 'Zero' if level=='0' else 'One' if level=='1' else 'Two'
                p.add_panel_feature(panel, f"Pct_{level_name}_{cat[:3]}", val)

    # def _generate_stat_plots(self, df_all):
    #     def plot_subset(data, title_prefix, filename_prefix):
    #         for category in ["Fixations", "Symbols"]:
    #             for measure in ["Percentage", "Count"]:
    #                 subset = data[
    #                     (data["Category"] == category) & 
    #                     (data["Measure"] == measure) & 
    #                     (data["Level"].isin(["0", "1", "2"]))
    #                 ].copy()
                    
    #                 if subset.empty: continue
    #                 subset.sort_values("Level", inplace=True)

    #                 plt.figure(figsize=(8, 6))
    #                 sns.barplot(data=subset, x="Condition", y="Value", hue="Group",
    #                             palette={"HC": "blue", "pwMS": "red"}, 
    #                             errorbar=('se', 1), capsize=0.1)
                    
    #                 plt.title(f"{title_prefix}: {category} ({measure})")
    #                 plt.xlabel("")
    #                 plt.ylabel(f"{measure} of Trials")
    #                 plt.grid(True, axis='y', alpha=0.3)
                    
    #                 fname = f"{filename_prefix}_{category}_{measure}.png"
    #                 plt.savefig(os.path.join(self.stats_dir, fname))
    #                 plt.close()

    #     panels = self._get_valid_panels()
    #     if not panels and "Panel" in df_all.columns:
    #         panels = sorted(df_all["Panel"].unique())

    #     for panel in panels:
    #         panel_data = df_all[df_all["Panel"] == panel]
    #         plot_subset(panel_data, f"Panel {panel}", f"barplot_Panel_{panel}")

    #     df_avg = df_all.groupby(["Participant", "Group", "Category", "Measure", "Level", "Condition"])["Value"].mean().reset_index()
    #     plot_subset(df_avg, "Average Across All Panels", "barplot_Global_Average")

    def _generate_stat_plots(self, df_all):
        def plot_subset(data, title_prefix, filename_prefix):
            # --- POSTER STYLE SETTINGS ---
            sns.set_context("poster", font_scale=1.2)
            sns.set_style("whitegrid")
            
            for category in ["Fixations", "Symbols"]:
                for measure in ["Percentage", "Count"]:
                    subset = data[
                        (data["Category"] == category) & 
                        (data["Measure"] == measure) & 
                        (data["Level"].isin(["0", "1", "2"]))
                    ].copy()
                    
                    if subset.empty: continue
                    subset.sort_values("Level", inplace=True)

                    # Bigger Figure for Poster
                    plt.figure(figsize=(12, 10))
                    
                    # Bar Plot
                    sns.barplot(data=subset, x="Condition", y="Value", hue="Group",
                                palette={"HC": "blue", "pwMS": "red"}, 
                                errorbar=('se', 1), capsize=0.1)
                    
                    # Big Title and Labels
                    plt.title(f"{title_prefix}: {category} ({measure})", fontsize=35, fontweight='bold', pad=25)
                    plt.xlabel("", fontsize=0) # Hide X label if self-explanatory
                    plt.ylabel(f"{measure} of Trials", fontsize=30, fontweight='bold', labelpad=20)
                    
                    # Big Ticks
                    plt.xticks(fontsize=28, fontweight='bold')
                    plt.yticks(fontsize=25)
                    plt.grid(True, axis='y', alpha=0.3)
                    
                    # Big Legend
                    plt.legend(title="Group", fontsize=22, title_fontsize=24, loc='upper right')
                    
                    # Clean Look
                    sns.despine(trim=True)
                    plt.tight_layout()
                    
                    fname = f"{filename_prefix}_{category}_{measure}.png"
                    plt.savefig(os.path.join(self.stats_dir, fname), dpi=300)
                    plt.close()

        panels = self._get_valid_panels()
        if not panels and "Panel" in df_all.columns:
            panels = sorted(df_all["Panel"].unique())

        for panel in panels:
            panel_data = df_all[df_all["Panel"] == panel]
            plot_subset(panel_data, f"Panel {panel}", f"barplot_Panel_{panel}")

        df_avg = df_all.groupby(["Participant", "Group", "Category", "Measure", "Level", "Condition"])["Value"].mean().reset_index()
        plot_subset(df_avg, "Average Across All Panels", "barplot_Global_Average")
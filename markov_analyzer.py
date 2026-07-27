import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from markov_core import AnalysisConfig, Participant

class MarkovAnalyzer:
    def __init__(self, participants_list: list[Participant], config: AnalysisConfig):
        self.participants = participants_list 
        self.config = config

        # --- FOLDER RENAMING ---
        self.output_dir = config.plot_output_path
        
        self.pca_dir = os.path.join(self.output_dir, "pca")   
        self.weights_dir = os.path.join(self.output_dir, "weights")
        self.variance_dir = os.path.join(self.output_dir, "variance")
        self.consistency_dir = os.path.join(self.output_dir, "consistency")
        self.separation_dir = os.path.join(self.output_dir, "separation")
        self.fix_corr_dir = os.path.join(self.output_dir, "fixation_corr")
        
        for d in [self.output_dir, self.pca_dir, self.weights_dir, 
                  self.variance_dir, self.consistency_dir, self.separation_dir, self.fix_corr_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.global_states = self._get_global_states()
        
        self.special_names = {
            ('ENTER', 'ENTER'): "Non-Symbol Fixations",
            ('ENTER', 'EXIT'): "Skipped Search",
            ('EXIT', 'EXIT'): "Correct Search",
            ('EXIT', 'ENTER'): "Uncertainty"
        }

    # --- HELPERS ---
    def _get_global_states(self):
        states = set()
        for p in self.participants:
            if p.mean_matrix is not None:
                states.update(p.mean_matrix.index.astype(str))
                states.update(p.mean_matrix.columns.astype(str))
        return sorted([s for s in states if s.lower() not in ['nan', 'none']])

    def _get_feature_names(self):
        names = []
        for r in self.global_states:
            for c in self.global_states:
                if (r, c) in self.special_names:
                    names.append(self.special_names[(r, c)])
                else:
                    names.append(f"({r}, {c})")
        return np.array(names)
    
    def _get_feature_tuple(self, feature_name: str):
        """
        Inverse of _get_feature_names.
        Converts a feature string (e.g., "Non-Symbol Fixations" or "(1, 2)") 
        back into a tuple (u, v).
        Returns (None, None) if parsing fails.
        """
        # 1. Reverse lookup for special names
        # Maps "Non-Symbol Fixations" -> ('ENTER', 'ENTER')
        name_to_coords = {v: k for k, v in self.special_names.items()}

        # Case A: Feature is a Special Name
        if feature_name in name_to_coords:
            return name_to_coords[feature_name]

        # Case B: Standard "(u, v)" string
        try:
            # Remove parens/quotes and split: "(1, 2)" -> "1", "2"
            clean_feat = feature_name.replace("(", "").replace(")", "").replace("'", "")
            parts = [x.strip() for x in clean_feat.split(",")]
            
            if len(parts) == 2:
                return parts[0], parts[1]
        except Exception:
            raise( f"Failed to parse feature name: {feature_name}")
            
        return None, None

    def _align_vector(self, df, fill_value=np.nan):
        if df is None: return None
        df = df.copy()
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        df = df.reindex(index=self.global_states, columns=self.global_states, fill_value=fill_value)
        return df.values.flatten()
    
    def _get_valid_panels(self):
        panels = set()
        for p in self.participants:
            panels.update(p.matrices.keys())
        return sorted(list(panels))

    def _safe_correlation(self, vec_a, vec_b):
        """
        Robust correlation that handles NaNs and non-numeric types safely.
        """
        # 1. Force conversion to numeric (Fixes the TypeError)
        vec_a = pd.to_numeric(vec_a, errors='coerce')
        vec_b = pd.to_numeric(vec_b, errors='coerce')
        
        # 2. Create Mask for valid numbers
        mask = ~np.isnan(vec_a) & ~np.isnan(vec_b)
        
        # 3. Check if we have enough data to correlate
        if np.sum(mask) < 2:
            return 0.0  # Not enough overlap
            
        a_clean = vec_a[mask]
        b_clean = vec_b[mask]
        
        # 4. Check for variance (cannot correlate flat lines)
        if np.std(a_clean) == 0 or np.std(b_clean) == 0:
            return 0.0
            
        return np.corrcoef(a_clean, b_clean)[0, 1]
    
    def _build_and_save_feature_table(self, save=True):
        """
        Constructs the feature table.
        Saves CSV with NaNs for missing data.
        Returns X with NaNs filled with 0.0 for PCA.
        """
        # Determine Mode
        if self.config.concat_panels:
            mode_name = "Concatenated_Panels"
        elif self.config.use_mean_matrix:
            mode_name = "Mean_Matrix"
        else:
            mode_name = "All_Panels"
            
        print(f"--- Building Data Table: {mode_name} ---")

        vectors = []
        metadata_list = []
        valid_panels = self._get_valid_panels()
        n_features = len(self.global_states) ** 2

        # ---------------------------------------------------------
        # FORMAT 1: CONCATENATED
        # ---------------------------------------------------------
        if self.config.concat_panels:
            print(f"Structure: One row per Participant. Columns stacked by panels {valid_panels}.")
            
            # 1. Generate concatenated feature names
            base_feats = self._get_feature_names()
            feature_names = []
            for panel in valid_panels:
                for feat in base_feats:
                    # Naming convention: "l3_(1, 2)"
                    feature_names.append(f"{panel}_{feat}")
            feature_names = np.array(feature_names)

            # 2. Build Stacked Vectors
            for p in self.participants:
                p_vecs = []
                # We assume the participant is valid if they have ANY data in ANY panel
                has_any_data = False
                
                for panel in valid_panels:
                    mat = p.matrices.get(panel)
                    # Use NaN for CSV saving purposes
                    vec = self._align_vector(mat, fill_value=np.nan)
                    
                    if vec is None:
                        # Entire panel missing -> Block of NaNs
                        vec = np.full(n_features, np.nan)
                    else:
                        has_any_data = True
                        
                    p_vecs.append(vec)
                
                # Only add participant if they exist (have at least one panel of data)
                if has_any_data:
                    full_vec = np.concatenate(p_vecs)
                    vectors.append(full_vec)
                    score = p.get_mean_score()
                    metadata_list.append({
                        "Participant": p.name, 
                        "Group": p.group, 
                        "Panel": "Concat", 
                        "SDMT_Score": score
                    })

        # ---------------------------------------------------------
        # FORMAT 2: STANDARD
        # ---------------------------------------------------------
        else:
            print(f"Structure: One row per {'Participant' if self.config.use_mean_matrix else 'Panel Instance'}.")
            feature_names = self._get_feature_names()
            
            for p in self.participants:
                if self.config.use_mean_matrix:
                    score = p.get_mean_score()
                    # Use NaN for CSV
                    vec = self._align_vector(p.mean_matrix, fill_value=np.nan)
                    if vec is not None:
                        vectors.append(vec)
                        metadata_list.append({
                            "Participant": p.name, 
                            "Group": p.group, 
                            "Panel": "Mean", 
                            "SDMT_Score" : score
                        })
                else:
                    for panel, matrix in p.matrices.items():
                        # Use NaN for CSV
                        vec = self._align_vector(matrix, fill_value=np.nan)
                        if vec is not None:
                            vectors.append(vec)
                            p_score = p.get_score(panel)
                            metadata_list.append({
                                "Participant": p.name, 
                                "Group": p.group, 
                                "Panel": panel, 
                                "SDMT_Score": p_score
                            })

        # Create Arrays
        X_with_nans = np.array(vectors) # This X has NaNs
        df_metadata = pd.DataFrame(metadata_list)

        # ---------------------------------------------------------
        # SAVE TO CSV (PRESERVING NANS)
        # ---------------------------------------------------------
        if save and X_with_nans.size > 0:
            try:
                df_features = pd.DataFrame(X_with_nans, columns=feature_names)
                df_full = pd.concat([df_metadata, df_features], axis=1)
                
                csv_filename = f"feature_table_{mode_name}.csv"
                save_path = os.path.join(self.separation_dir, csv_filename)
                
                df_full.to_csv(save_path, index=False)
                print(f"Saved feature table to: {save_path}")
            except Exception as e:
                print(f"Warning: Could not save CSV table. Error: {e}")
        
        # ---------------------------------------------------------
        # PREPARE FOR PCA (IMPUTATION)
        # ---------------------------------------------------------

        # 1. Identify columns that are NOT all NaN (Valid columns)
        # We cannot calculate a mean for a column if it is 100% empty.
        valid_col_mask = ~np.isnan(X_with_nans).all(axis=0)
        
        # 2. Filter Data and Feature Names
        X_filtered = X_with_nans[:, valid_col_mask]
        feature_names_filtered = feature_names[valid_col_mask]
        
        # 3. Impute remaining NaNs with Column Mean
        col_means = np.nanmean(X_filtered, axis=0)
        
        # Find indices where NaN still exists (in valid columns)
        inds = np.where(np.isnan(X_filtered))
        
        # Replace those NaNs with the mean of that specific column
        X_filtered[inds] = np.take(col_means, inds[1])
        
        # Return the clean data and the updated feature names
        return X_filtered, df_metadata, feature_names_filtered
    
        return X_for_pca, df_metadata, feature_names

    # =========================================================================
    # PART 1: PCA & FEATURE ANALYSIS
    # =========================================================================
    def run_pca(self, n_components = None, save=True):
        # 1. BUILD OR FETCH DATA
        X, df_metadata, feature_names = self._build_and_save_feature_table(save=save)
        
        if X.size == 0:
            print("No data available for PCA.")
            return

        # 2. PCA CLEANING
        X_var = np.var(X, axis=0)
        valid_mask = X_var > 1e-9 
        
        X_clean = X[:, valid_mask]
        feature_names_clean = feature_names[valid_mask]
        
        print(f"Features: {X.shape[1]} -> {X_clean.shape[1]} (Dropped constant features)")
        
        if len(X_clean) < 3:
            print("Not enough data.")
            return

        # 3. RUN PCA
        mode_name = "Concatenated Panels" if self.config.concat_panels else ("Mean Matrix" if self.config.use_mean_matrix else "All Panels")
        
        max_comps = min(len(X_clean), X_clean.shape[1], 20) if n_components is None else n_components
        pca = PCA(n_components=max_comps)
        X_scaled = StandardScaler().fit_transform(X_clean)
        coords = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_

        for i in range(coords.shape[1]):
            df_metadata[f"PC{i+1}"] = coords[:, i]

        # 4. GROUP SEPARATION
        top_separators = self._analyze_group_separation(df_metadata)
        
        # 5. PLOTTING STRATEGY
        pcs_to_plot = []
        
        # A. Add Top Separators
        if len(top_separators) >= 2:
            p1_name, _ = top_separators[0]
            p2_name, _ = top_separators[1]
            idx1 = int(p1_name.replace("PC", "")) - 1
            idx2 = int(p2_name.replace("PC", "")) - 1
            pcs_to_plot.append((idx1, idx2))
            
        # B. Add Standard Pairs (PC1 vs PC2, PC1 vs PC3)
        standard_pairs = list(itertools.combinations(range(min(3, coords.shape[1])), 2))
        for pair in standard_pairs:
            if pair not in pcs_to_plot:
                pcs_to_plot.append(pair)

        # 6. GENERATE PLOTS
        for (idx1, idx2) in pcs_to_plot:
            print(f"Generating Plots: PC{idx1+1} vs PC{idx2+1}...")
            
            # Scatters
            self._plot_pca_scatter(coords, df_metadata["Group"].values, df_metadata["Participant"].values,
                                   explained_var, mode_name, col_name="Group", 
                                   pc_x=idx1, pc_y=idx2, is_numeric=False)
            
            self._plot_pca_scatter(coords, df_metadata["SDMT_Score"].values, df_metadata["Participant"].values,
                                   explained_var, mode_name, col_name="SDMT_Score", 
                                   pc_x=idx1, pc_y=idx2, is_numeric=True)
            
            # Weights
            self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=idx1)
            self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=idx2)

        # 7. ELBOW PLOT
        self._plot_elbow(X_clean, mode_name)
# =========================================================================
    # PART 1: PCA & FEATURE ANALYSIS
    # =========================================================================
    def run_pca(self, n_components = None, save=True):
        # 1. BUILD OR FETCH DATA
        X, df_metadata, feature_names = self._build_and_save_feature_table(save=save)
        
        if X.size == 0:
            print("No data available for PCA.")
            return

        # 2. PCA CLEANING
        X_var = np.var(X, axis=0)
        valid_mask = X_var > 1e-9 
        
        X_clean = X[:, valid_mask]
        feature_names_clean = feature_names[valid_mask]
        
        print(f"Features: {X.shape[1]} -> {X_clean.shape[1]} (Dropped constant features)")
        
        if len(X_clean) < 3:
            print("Not enough data.")
            return

        # 3. RUN PCA
        mode_name = "Concatenated Panels" if self.config.concat_panels else ("Mean Matrix" if self.config.use_mean_matrix else "All Panels")
        
        max_comps = min(len(X_clean), X_clean.shape[1], 20) if n_components is None else n_components
        pca = PCA(n_components=max_comps)
        X_scaled = StandardScaler().fit_transform(X_clean)
        coords = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_

        for i in range(coords.shape[1]):
            df_metadata[f"PC{i+1}"] = coords[:, i]

        # 4. GROUP SEPARATION
        top_separators = self._analyze_group_separation(df_metadata)
        
        # 5. PLOTTING STRATEGY
        pcs_to_plot = []
        
        # A. Add Top Separators (The logic that caused the error)
        if len(top_separators) >= 2:
            # Removed trailing comma and ensured variable names match
            p1_name, p1_val = top_separators[0] 
            p2_name, p2_val = top_separators[1]
            
            # Extract numbers from strings like "PC10" -> 9 (index)
            idx1 = int(p1_name.replace("PC", "")) - 1
            idx2 = int(p2_name.replace("PC", "")) - 1
            pcs_to_plot.append((idx1, idx2))
            
        # B. Add Standard Pairs (PC1 vs PC2, PC1 vs PC3) if not already added
        standard_pairs = list(itertools.combinations(range(min(3, coords.shape[1])), 2))
        for pair in standard_pairs:
            if pair not in pcs_to_plot:
                pcs_to_plot.append(pair)

        # 6. GENERATE PLOTS
        for (idx1, idx2) in pcs_to_plot:
            print(f"Generating Plots: PC{idx1+1} vs PC{idx2+1}...")
            
            # Scatters
            # Pass 1: Colored by Group (Categorical)
            self._plot_pca_scatter(coords, df_metadata["Group"].values, df_metadata["Participant"].values,
                                   explained_var, mode_name, col_name="Group", 
                                   pc_x=idx1, pc_y=idx2, is_numeric=False)
            
            # Pass 2: Colored by SDMT Score (Numeric)
            self._plot_pca_scatter(coords, df_metadata["SDMT_Score"].values, df_metadata["Participant"].values,
                                   explained_var, mode_name, col_name="SDMT_Score", 
                                   pc_x=idx1, pc_y=idx2, is_numeric=True)
            
            # Weights
            self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=idx1)
            self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=idx2)

        # 7. ELBOW PLOT
        self._plot_elbow(X_clean, mode_name)
    # def run_pca(self, n_components=3):
    #     mode_name = "Mean Matrix" if self.config.use_mean_matrix else "All Panels"
    #     print(f"--- Running Analysis: {mode_name} ---")
        
    #     # 1. DATA COLLECTION
    #     vectors = []
    #     metadata_list = [] 
        
    #     for p in self.participants:
    #         if mode_name == "Mean Matrix":
    #             score = p.get_mean_score()
    #             vec = self._align_vector(p.mean_matrix)
    #             if vec is not None:
    #                 vectors.append(vec)
    #                 meta = {"Participant": p.name, "Group": p.group, "SDMT_Score": score}
    #                 meta.update(p.get_mean_features())
    #                 metadata_list.append(meta)
    #         else:
    #             for panel, matrix in p.matrices.items():
    #                 vec = self._align_vector(matrix)
    #                 if vec is not None:
    #                     vectors.append(vec)
    #                     p_score = p.get_score(panel)

    #                     meta = {"Participant": f"{p.name}_{panel}", "Group": p.group, 
    #                             "Panel": panel, "SDMT_Score": p_score}
    #                     meta.update(p.get_features_for_panel(panel))
    #                     metadata_list.append(meta)

    #     X = np.array(vectors)
    #     X = np.nan_to_num(X, nan=0.0)

    #     df_metadata = pd.DataFrame(metadata_list)
    #     feature_names = self._get_feature_names()
        
    #     if X.size == 0:
    #         print("No data available for PCA.")
    #         return

    #     X_var = np.var(X, axis=0)
    #     valid_mask = X_var > 1e-9 
        
    #     X_clean = X[:, valid_mask]
    #     feature_names_clean = feature_names[valid_mask]
        
    #     print(f"Features: {X.shape[1]} -> {X_clean.shape[1]} (Dropped constant features)")
    #     print(f"doropped features were: {feature_names[~valid_mask]}")
        
    #     if len(X_clean) < 3:
    #         print("Not enough data.")
    #         return

    #     # 3. RUN PCA
    #     max_comps = min(len(X_clean), X_clean.shape[1], 20)
    #     pca = PCA(n_components=max_comps)
    #     X_scaled = StandardScaler().fit_transform(X_clean)
    #     coords = pca.fit_transform(X_scaled)
    #     explained_var = pca.explained_variance_ratio_

    #     for i in range(coords.shape[1]):
    #         df_metadata[f"PC{i+1}"] = coords[:, i]

    #     # 4. GROUP SEPARATION (Find Best Separators)
    #     top_separators = self._analyze_group_separation(df_metadata)
        
    #     # --- PLOT BEST SEPARATORS ---
    #     if len(top_separators) >= 2:
    #         pc_best_1, p1 = top_separators[0], 
    #         pc_best_2, p2 = top_separators[1] # e.g. "PC9" , 0.00012
            
    #         # Extract indices (PC1 -> index 0)
    #         idx1 = int(pc_best_1.replace("PC", "")) - 1
    #         idx2 = int(pc_best_2.replace("PC", "")) - 1
            
    #         print(f"--- Generating Focused Plots for Best Separators: {pc_best_1} vs {pc_best_2} ---")
            
    #         # Plot 1: Colored by Group
    #         self._plot_pca_scatter(coords, df_metadata["Group"].values, df_metadata["Participant"].values,
    #                                explained_var, mode_name, col_name="Group_Separation", 
    #                                pc_x=idx1, pc_y=idx2, is_numeric=False)
            
    #         # Plot 2: Colored by Score
    #         self._plot_pca_scatter(coords, df_metadata["SDMT_Score"].values, df_metadata["Participant"].values,
    #                                explained_var, mode_name, col_name="Score_Separation", 
    #                                pc_x=idx1, pc_y=idx2, is_numeric=True)
            

    #         # Weight plots for these specific PCs
    #         self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=idx1)
    #         self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=idx2)

    #     # 5. STANDARD PCA PLOTS (Top 3)
    #     coloring_cols = [c for c in df_metadata.columns if c not in ["Participant", "SDMT_Score"] and not c.startswith("PC")]
    #     n_pcs_to_plot = min(3, coords.shape[1])
    #     pc_pairs = list(itertools.combinations(range(n_pcs_to_plot), 2))
        
    #     for col in coloring_cols:
    #         is_numeric = pd.api.types.is_numeric_dtype(df_metadata[col])
    #         values = df_metadata[col].values
    #         for (px, py) in pc_pairs:
    #             self._plot_pca_scatter(coords, values, df_metadata["Participant"].values, 
    #                                    explained_var, mode_name, col_name=col, pc_x=px, pc_y=py, is_numeric=is_numeric)

    #     # Standard weights (PC1-3)
    #     for i in range(min(3, len(pca.components_))):
    #         self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=i)
            
    #     self._plot_elbow(X_clean, mode_name)

    def run_pca_and_pareto(self, n_components=3):
        # 1. Determine Mode
        if self.config.concat_panels:
            mode_name = "Concatenated Panels"
        elif self.config.use_mean_matrix:
            mode_name = "Mean Matrix"
        else:
            mode_name = "All Panels"

        print(f"--- Running PCA: {mode_name} ---")
        
        vectors = []
        metadata_list = [] 
        valid_panels = self._get_valid_panels()

        # ---------------------------------------------------------
        # TABLE CONSTRUCTION (The only part that changes logic)
        # ---------------------------------------------------------
        if mode_name == "Concatenated Panels":
            print(f"Stacking panels: {valid_panels}")
            
            # Generate feature names for the giant vector: "PanelName_Feature"
            base_feats = self._get_feature_names()
            feature_names = []
            for panel in valid_panels:
                for feat in base_feats:
                    feature_names.append(f"{panel}_{feat}")
            feature_names = np.array(feature_names)

            # Build Vectors
            for p in self.participants:
                p_vecs = []
                # Ensure every participant has the same panels in same order
                for panel in valid_panels:
                    mat = p.matrices.get(panel)
                    vec = self._align_vector(mat)
                    
                    if vec is None:
                        # Fill missing panel with zeros (neutral for PCA)
                        vec = np.zeros(len(self.global_states) ** 2)
                    
                    p_vecs.append(vec)
                
                # Check if participant has at least some data
                if any(np.sum(v) > 0 for v in p_vecs):
                    full_vec = np.concatenate(p_vecs)
                    vectors.append(full_vec)
                    # Metadata for coloring (Use mean score)
                    score = p.get_mean_score()
                    metadata_list.append({"Participant": p.name, "Group": p.group, "SDMT_Score": score})

        else:
            # Standard Mode (Mean or All Panels)
            feature_names = self._get_feature_names()
            
            for p in self.participants:
                if mode_name == "Mean Matrix":
                    score = p.get_mean_score()
                    vec = self._align_vector(p.mean_matrix)
                    if vec is not None:
                        vectors.append(vec)
                        metadata_list.append({"Participant": p.name, "Group": p.group, "SDMT_Score": score})
                else:
                    for panel, matrix in p.matrices.items():
                        vec = self._align_vector(matrix)
                        if vec is not None:
                            vectors.append(vec)
                            p_score = p.get_score(panel)
                            metadata_list.append({"Participant": f"{p.name}_{panel}", "Group": p.group, 
                                                  "Panel": panel, "SDMT_Score": p_score})

        # ---------------------------------------------------------
        # PCA EXECUTION (Same Math)
        # ---------------------------------------------------------
        X = np.array(vectors)
        X = np.nan_to_num(X, nan=0.0)
        
        if X.size == 0:
            print("No data available for PCA.")
            return

        # Drop constant features (zero variance)
        X_var = np.var(X, axis=0)
        valid_mask = X_var > 1e-9 
        X_clean = X[:, valid_mask]
        feature_names_clean = feature_names[valid_mask]
        
        print(f"Input Shape: {X.shape}. Features kept: {X_clean.shape[1]}")

        # Metadata DataFrame
        df_metadata = pd.DataFrame(metadata_list)
        
        # Run PCA
        max_comps = min(len(X_clean), X_clean.shape[1], 20)
        pca = PCA(n_components=max_comps)
        X_scaled = StandardScaler().fit_transform(X_clean)
        coords = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_

        # Add PCs to Metadata
        for i in range(coords.shape[1]):
            df_metadata[f"PC{i+1}"] = coords[:, i]

        # ---------------------------------------------------------
        # PLOTTING
        # ---------------------------------------------------------
        
        # 1. Elbow Plot
        self._plot_elbow(X_clean, mode_name)

        # 2. Group Separation Analysis
        top_separators = self._analyze_group_separation(df_metadata)
        
        # 3. Plot Scatter & Weights
        # Define which PCs to plot (Top 2 separators OR PC1 vs PC2)
        pcs_to_plot = []
        if len(top_separators) >= 2:
            # Get indices of best separating PCs
            p1 = int(top_separators[0][0].replace("PC", "")) - 1
            p2 = int(top_separators[1][0].replace("PC", "")) - 1
            pcs_to_plot.append((p1, p2))
        
        # Always add standard PC1 vs PC2
        if (0, 1) not in pcs_to_plot:
            pcs_to_plot.append((0, 1))

        for (idx1, idx2) in pcs_to_plot:
            print(f"Generating plots for PC{idx1+1} vs PC{idx2+1}...")
            
            # Scatters
            self._plot_pca_scatter(coords, df_metadata["Group"].values, df_metadata["Participant"].values,
                                   explained_var, mode_name, col_name="Group", pc_x=idx1, pc_y=idx2, is_numeric=False)
            self._plot_pca_scatter(coords, df_metadata["SDMT_Score"].values, df_metadata["Participant"].values,
                                   explained_var, mode_name, col_name="SDMT_Score", pc_x=idx1, pc_y=idx2, is_numeric=True)

            # Weights (The visualization logic handles the concatenation)
            self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=idx1)
            self._plot_pca_weights(pca, feature_names_clean, mode_name, target_pc_idx=idx2)

    # def _plot_pca_scatter(self, coords, values, names, explained_var, mode_name, col_name, pc_x=0, pc_y=1, is_numeric=True):
    #     plt.figure(figsize=(10, 8))
    #     x_vals = coords[:, pc_x]
    #     y_vals = coords[:, pc_y]
        
    #     # Handle NaN values
    #     if is_numeric:
    #         # Explicitly checking for NaN in numeric array
    #         # "values" might be object type if it has NaNs, coerce to float
    #         values = values.astype(float)
    #         mask = ~np.isnan(values)
    #     else:
    #         # For categorical, None check
    #         mask = np.array([v is not None for v in values])

    #     if np.sum(mask) == 0: 
    #         plt.close(); return

    #     plot_x = x_vals[mask]
    #     plot_y = y_vals[mask]
    #     plot_vals = values[mask]
    #     plot_names = names[mask]

    #     if is_numeric:
    #         sc = plt.scatter(plot_x, plot_y, c=plot_vals, cmap='viridis', s=60, edgecolors='k', alpha=0.7)
    #         plt.colorbar(sc, label=col_name)
    #     else:
    #         unique_cats = np.unique(plot_vals.astype(str))
    #         palette = sns.color_palette("Set1", n_colors=len(unique_cats))
    #         for i, cat in enumerate(unique_cats):
    #             cat_mask = (plot_vals == cat)
    #             plt.scatter(plot_x[cat_mask], plot_y[cat_mask], label=cat, color=palette[i], s=60, edgecolors='k', alpha=0.7)
    #         plt.legend(title=col_name)

    #     # Pareto Hull & Labels
    #     if len(plot_x) > 3:
    #         try:
    #             points_2d = np.column_stack((plot_x, plot_y))
    #             hull = ConvexHull(points_2d)
    #             for simplex in hull.simplices:
    #                 plt.plot(points_2d[simplex, 0], points_2d[simplex, 1], 'r--', lw=1, alpha=0.5)
                
    #             # Archetypes
    #             arch_idx = hull.vertices
    #             plt.scatter(points_2d[arch_idx, 0], points_2d[arch_idx, 1], s=120, facecolors='none', edgecolors='red')
                
    #             # --- NEW: Add Labels to Hull Vertices ---
    #             for idx in arch_idx:
    #                 label_txt = str(plot_names[idx])
    #                 # Offset text slightly
    #                 plt.text(points_2d[idx, 0], points_2d[idx, 1], label_txt, 
    #                          fontsize=8, fontweight='bold', color='darkred',
    #                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    #         except: pass

    #     plt.xlabel(f"PC{pc_x+1} ({explained_var[pc_x]:.1%} var)")
    #     plt.ylabel(f"PC{pc_y+1} ({explained_var[pc_y]:.1%} var)")
    #     plt.title(f"{mode_name}: PC{pc_x+1} vs PC{pc_y+1}\nColored by {col_name}")
    #     plt.grid(True, alpha=0.3)
    #     fname = f"PCA_PC{pc_x+1}_PC{pc_y+1}_by_{col_name}.png".replace(" ", "_")
    #     plt.savefig(os.path.join(self.pca_dir, fname))
    #     plt.close()


    def _plot_pca_scatter(self, coords, color_values, labels, explained_var, mode_name, 
                          col_name, pc_x=0, pc_y=1, is_numeric=False):
        
        # --- POSTER STYLE SETTINGS ---
        sns.set_context("poster", font_scale=1.2) # Big fonts
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 10))

        # 1. SETUP DATA
        x_vals = coords[:, pc_x]
        y_vals = coords[:, pc_y]
        
        # 2. PLOTTING
        if is_numeric:
            # --- CONTINUOUS COLOR (SDMT Score) ---
            # Remove NaNs for plotting
            mask = ~np.isnan(color_values.astype(float))
            sc = plt.scatter(x_vals[mask], y_vals[mask], c=color_values[mask], 
                             cmap='viridis', s=200, edgecolors='k', alpha=0.9) # s=200 for big points
            cbar = plt.colorbar(sc)
            cbar.set_label(col_name, fontsize=25, fontweight='bold', labelpad=15)
            cbar.ax.tick_params(labelsize=20)
            
            title_text = f"PCA by {col_name}"
        else:
            # --- CATEGORICAL COLOR (Group) ---
            # Define specific colors if needed, or let seaborn handle it
            unique_cats = np.unique(color_values)
            palette = {"HC": "blue", "pwMS": "red"} if set(unique_cats).issubset({"HC", "pwMS", "MS"}) else None
            
            sns.scatterplot(x=x_vals, y=y_vals, hue=color_values, palette=palette, 
                            s=200, edgecolor='k', alpha=0.9) # s=200
            plt.legend(title=col_name, fontsize=20, title_fontsize=22, loc='best')
            title_text = f"PCA by {col_name}"

        # 3. LABELS & TITLES (BIG FONTS)
        var_x = explained_var[pc_x] * 100
        var_y = explained_var[pc_y] * 100
        
        plt.xlabel(f"PC{pc_x+1} ({var_x:.1f}%)", fontsize=30, fontweight='bold', labelpad=15)
        plt.ylabel(f"PC{pc_y+1} ({var_y:.1f}%)", fontsize=30, fontweight='bold', labelpad=15)
        plt.title(f"{title_text}\n({mode_name})", fontsize=35, fontweight='bold', pad=25)
        
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        
        # Remove top/right spines
        sns.despine(trim=True)
        plt.tight_layout()
        
        # 4. SAVE
        # Sanitize filename
        safe_col = "".join([c for c in col_name if c.isalnum() or c in (' ', '_')]).strip()
        fname = f"pca_{mode_name.replace(' ', '_')}_PC{pc_x+1}_PC{pc_y+1}_{safe_col}.png"
        save_path = os.path.join(self.pca_dir, fname)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   Saved PCA plot: {fname}")

    def _analyze_group_separation(self, df):
        print("--- Analyzing Group Separation ---")
        save_path = os.path.join(self.separation_dir, "pca_scores_and_features.csv")
        df.to_csv(save_path, index=False)
        
        groups = df["Group"].unique()
        if len(groups) != 2: return []

        g1, g2 = groups[0], groups[1]
        pc_cols = [c for c in df.columns if c.startswith("PC")]
        
        mw_results = []
        for col in pc_cols:
            v1 = df[df["Group"] == g1][col].dropna()
            v2 = df[df["Group"] == g2][col].dropna()
            if len(v1) > 0 and len(v2) > 0:
                stat, p_val = mannwhitneyu(v1, v2, alternative='two-sided')
                mw_results.append((col, p_val))

        mw_results.sort(key=lambda x: x[1])
        print("Top 3 Separating PCs:")
        for n, p in mw_results[:3]: print(f"  {n}: p={p:.5f}")
        return mw_results

    # =========================================================================
    # PART 2: CONSISTENCY ANALYSIS
    # =========================================================================

    def run_consistency_analysis(self, n_permutations=1000):
        print("\n=== Running Consistency Analysis (Violin & Permutation) ===")
        
        valid_panels = self._get_valid_panels()
        if not valid_panels:
            print("Warning: No panels found. Skipping consistency analysis.")
            return
        
        print(f"Detected Panels: {valid_panels}")

        pools = {panel: [] for panel in valid_panels}
        valid_participants = [p for p in self.participants if p.matrices]
        
        for p in valid_participants:
            for panel in valid_panels:
                mat = p.matrices.get(panel)
                vec = self._align_vector(mat)
                pools[panel].append(vec) 

        # self._run_violin_analysis(valid_participants, valid_panels)
        self._run_permutation_test(pools, valid_panels, n_permutations)

    def _run_violin_analysis(self, participants, valid_panels):
        print("--- Generating Violin Plot (Within vs Between) ---")
        
        within_vals_plot = []
        between_vals_plot = []
        
        flat_data = {}
        for p in participants:
            for panel in valid_panels:
                mat = p.matrices.get(panel)
                if mat is not None:
                    flat_data[(p.name, panel)] = self._align_vector(mat)

        # A. Within
        for p in participants:
            p_vecs = []
            p_panels = []
            for panel in valid_panels:
                if (p.name, panel) in flat_data:
                    p_vecs.append(flat_data[(p.name, panel)])
                    p_panels.append(panel)
            
            if len(p_vecs) > 1:
                for i in range(len(p_vecs)):
                    for j in range(i + 1, len(p_vecs)):
                        r = self._safe_correlation(p_vecs[i], p_vecs[j])
                        if not np.isnan(r):
                            # === PRINT PERFECT MATCHES ===
                            if r >= 0.99:
                                print(f"PERFECT MATCH (Within): {p.name} [{p_panels[i]}] vs [{p_panels[j]}] r={r:.3f}")
                            # =============================
                            within_vals_plot.append(r)

        # B. Between
        keys = list(flat_data.keys())
        n_pairs = min(5000, len(keys) * 10)
        attempts = 0
        while len(between_vals_plot) < n_pairs and attempts < n_pairs * 5:
            attempts += 1
            idx1, idx2 = np.random.choice(len(keys), 2, replace=False)
            name1, panel1 = keys[idx1]
            name2, panel2 = keys[idx2]
            if name1 != name2 and panel1 != panel2:
                r = self._safe_correlation(flat_data[keys[idx1]], flat_data[keys[idx2]])
                if not np.isnan(r):
                    # === PRINT PERFECT MATCHES ===
                    if r >= 0.99:
                        print(f"PERFECT MATCH (Between): {name1} [{panel1}] vs {name2} [{panel2}] r={r:.3f}")
                    # =============================
                    between_vals_plot.append(r)

        
        if len(within_vals_plot) == 0 or len(between_vals_plot) == 0:
            print("Warning: Not enough data for Violin.")
            return

        # --- PLOTTING ---
        data = [np.array(within_vals_plot), np.array(between_vals_plot)]
        mu_w, sd_w, n_w = np.mean(data[0]), np.std(data[0]), len(data[0])
        mu_b, sd_b, n_b = np.mean(data[1]), np.std(data[1]), len(data[1])

        plt.figure(figsize=(9, 7))
        parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        colors = ['blue', 'red']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.5)
            
        plt.errorbar([1, 2], [mu_w, mu_b], yerr=[sd_w, sd_b], fmt='o', color='k', label='Mean ± SD')
        
        # --- NEW: STATS TEXT BOX ---
        stats_text = (
            f"WITHIN:\nN = {n_w}\nMean = {mu_w:.3f}\nSD = {sd_w:.3f}\n\n"
            f"BETWEEN:\nN = {n_b}\nMean = {mu_b:.3f}\nSD = {sd_b:.3f}"
        )
        # Position box to the right
        plt.text(2.6, np.mean([mu_w, mu_b]), stats_text, fontsize=10, 
                 verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        plt.xticks([1, 2], ['Within-Subject', 'Between-Subject'])
        plt.ylabel("Correlation (r)")
        plt.title("Consistency: Within vs Between Participants")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.5, 3.5) # Expand x-axis to fit the text box
        plt.tight_layout()
        plt.savefig(os.path.join(self.consistency_dir, "violin_within_vs_between.png"))
        plt.close()

    def _run_permutation_test(self, pools, valid_panels, n_permutations):
        print(f"--- Running Permutation Test ({n_permutations} iters) ---")
        if not pools or not valid_panels: return

        n_subs = len(pools[valid_panels[0]])
        
        def calc_mean_stat(current_pool_dict):
            individual_means = []
            for i in range(n_subs):
                sub_vecs = []
                for panel in valid_panels:
                    if i < len(current_pool_dict[panel]):
                        v = current_pool_dict[panel][i]
                        if v is not None:
                            sub_vecs.append(v)
                if len(sub_vecs) < 2: continue
                rs = []
                for a in range(len(sub_vecs)):
                    for b in range(a + 1, len(sub_vecs)):
                        r = self._safe_correlation(sub_vecs[a], sub_vecs[b])
                        if not np.isnan(r):
                            rs.append(r)
                if rs: individual_means.append(np.mean(rs))
            
            if not individual_means: return np.nan
            return np.nanmean(individual_means)

        obs_mean = calc_mean_stat(pools)
        if np.isnan(obs_mean): return

        print(f"Observed Mean: {obs_mean:.4f}")
        
        null_means = []
        np_pools = {k: np.array(v, dtype=object) for k, v in pools.items()}
        
        for _ in tqdm(range(n_permutations)):
            shuffled = {}
            for panel in valid_panels:
                arr = np_pools[panel].copy()
                np.random.shuffle(arr)
                shuffled[panel] = arr
            null_means.append(calc_mean_stat(shuffled))
            
        valid_nulls = np.array([x for x in null_means if not np.isnan(x)])
        
        if len(valid_nulls) == 0: return

        n_better = np.sum(valid_nulls >= obs_mean)
        p_val = (n_better + 1) / (len(valid_nulls) + 1)
        print(f"Permutation P-Value: {p_val:.5f}")
        
        # --- PLOTTING (POSTER STYLE) ---
        mu_null = np.mean(valid_nulls)
        sd_null = np.std(valid_nulls)
        n_null = len(valid_nulls)

        # 1. Set Context for Big Fonts
        sns.set_context("poster", font_scale=1.2)
        sns.set_style("whitegrid")
        
        # 2. Big Figure
        plt.figure(figsize=(12, 10))
        
        # 3. Plot
        plt.hist(valid_nulls, bins=50, color='gray', alpha=0.5, density=True, label='Null Distribution')
        plt.axvline(obs_mean, color='red', linestyle='--', linewidth=5, label=f'Observed') # Thicker line
        
        # --- STATS TEXT BOX (BIGGER) ---
        stats_text = (
            f"Observed Mean: {obs_mean:.4f}\n"
            f"Null Mean: {mu_null:.4f}\n"
            f"Null SD: {sd_null:.4f}\n"
            f"N Permutations: {n_null}\n"
            f"p-value: {p_val:.5f}"
        )
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=22, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

        # 4. Labels & Titles
        plt.title("Permutation Test: Within-Subject Consistency", fontsize=35, fontweight='bold', pad=25)
        plt.xlabel("Mean Correlation", fontsize=30, fontweight='bold', labelpad=20)
        plt.ylabel("Density", fontsize=30, fontweight='bold', labelpad=20)
        
        # 5. Ticks & Legend
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(loc='upper right', fontsize=22)
        
        # 6. Save
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(self.consistency_dir, "permutation_test_histogram.png"), dpi=300)
        plt.close()
        
    # def _run_permutation_test(self, pools, valid_panels, n_permutations):
    #     print(f"--- Running Permutation Test ({n_permutations} iters) ---")
    #     if not pools or not valid_panels: return

    #     n_subs = len(pools[valid_panels[0]])
        
    #     def calc_mean_stat(current_pool_dict):
    #         individual_means = []
    #         for i in range(n_subs):
    #             sub_vecs = []
    #             for panel in valid_panels:
    #                 if i < len(current_pool_dict[panel]):
    #                     v = current_pool_dict[panel][i]
    #                     if v is not None:
    #                         sub_vecs.append(v)
    #             if len(sub_vecs) < 2: continue
    #             rs = []
    #             for a in range(len(sub_vecs)):
    #                 for b in range(a + 1, len(sub_vecs)):
    #                     r = self._safe_correlation(sub_vecs[a], sub_vecs[b])
    #                     if not np.isnan(r):
    #                         rs.append(r)
    #             if rs: individual_means.append(np.mean(rs))
            
    #         if not individual_means: return np.nan
    #         return np.nanmean(individual_means)

    #     obs_mean = calc_mean_stat(pools)
    #     if np.isnan(obs_mean): return

    #     print(f"Observed Mean: {obs_mean:.4f}")
        
    #     null_means = []
    #     np_pools = {k: np.array(v, dtype=object) for k, v in pools.items()}
        
    #     for _ in tqdm(range(n_permutations)):
    #         shuffled = {}
    #         for panel in valid_panels:
    #             arr = np_pools[panel].copy()
    #             np.random.shuffle(arr)
    #             shuffled[panel] = arr
    #         null_means.append(calc_mean_stat(shuffled))
            
    #     valid_nulls = np.array([x for x in null_means if not np.isnan(x)])
        
    #     if len(valid_nulls) == 0: return

    #     n_better = np.sum(valid_nulls >= obs_mean)
    #     p_val = (n_better + 1) / (len(valid_nulls) + 1)
    #     print(f"Permutation P-Value: {p_val:.5f}")
        
    #     # --- PLOTTING ---
    #     mu_null = np.mean(valid_nulls)
    #     sd_null = np.std(valid_nulls)
    #     n_null = len(valid_nulls)

    #     plt.figure(figsize=(8, 6))
    #     plt.hist(valid_nulls, bins=50, color='gray', alpha=0.5, density=True, label='Null Distribution')
    #     plt.axvline(obs_mean, color='red', linestyle='--', linewidth=2, label=f'Observed')
        
    #     # --- NEW: STATS TEXT BOX ---
    #     stats_text = (
    #         f"Observed Mean: {obs_mean:.4f}\n"
    #         f"Null Mean: {mu_null:.4f}\n"
    #         f"Null SD: {sd_null:.4f}\n"
    #         f"N Permutations: {n_null}\n"
    #         f"p-value: {p_val:.5f}"
    #     )
        
    #     plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
    #              verticalalignment='top', fontsize=10,
    #              bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    #     plt.title("Permutation Test: Within-Subject Consistency")
    #     plt.xlabel("Mean Correlation")
    #     plt.legend(loc='upper right')
    #     plt.savefig(os.path.join(self.consistency_dir, "permutation_test_histogram.png"))
    #     plt.close()


    # def _plot_pca_weights(self, pca, feature_names, mode_name, target_pc_idx=0):
    #     """
    #     Plots PCA weights as a heatmap over the transition matrix grid.
    #     """
    #     pc_num = target_pc_idx + 1
    #     comp = pca.components_[target_pc_idx]
        
    #     # 1. Define Matrix Layout
    #     states = ['ENTER'] + [str(i) for i in range(1, 10)] + ['EXIT']
        
    #     # Initialize grid with NaN (NaNs will be transparent/black in heatmap)
    #     matrix_df = pd.DataFrame(np.nan, index=states, columns=states)
        
    #     # 2. Parse Features and Fill Grid using the new helper
    #     for feature, weight in zip(feature_names, comp):
    #         # --- NEW CLEAN CALL ---
    #         u, v = self._get_feature_tuple(feature)
    #         # ----------------------
            
    #         if u in states and v in states:
    #             matrix_df.loc[u, v] = weight

    #     # 3. Plot Heatmap (Same as before)
    #     plt.figure(figsize=(10, 8))
        
    #     max_abs = max(abs(comp.min()), abs(comp.max()))
    #     ax = plt.gca()
    #     ax.set_facecolor('black') # Missing values appear black
        
    #     sns.heatmap(matrix_df, 
    #                 annot=True,
    #                 fmt=".2f",
    #                 cmap="RdBu",
    #                 center=0,
    #                 vmin=-max_abs,
    #                 vmax=max_abs,
    #                 square=True,
    #                 linewidths=0.5,
    #                 linecolor='gray',
    #                 cbar_kws={"label": "PCA Weight"})
        
    #     plt.title(f"PC{pc_num} Weights Heatmap ({mode_name})")
    #     plt.tight_layout()
        
    #     save_path = os.path.join(self.weights_dir, f"heatmap_weights_PC{pc_num}.png")
    #     plt.savefig(save_path)
    #     plt.close()

    def _plot_pca_weights(self, pca, feature_names, mode_name, target_pc_idx=0):
        """
        Plots PCA weights as a heatmap over the transition matrix grid.
        (Poster Style: Large text and annotations)
        """
        pc_num = target_pc_idx + 1
        comp = pca.components_[target_pc_idx]
        
        # 1. Define Matrix Layout
        states = ['ENTER'] + [str(i) for i in range(1, 10)] + ['EXIT']
        
        # Initialize grid with NaN (NaNs will be transparent/black in heatmap)
        matrix_df = pd.DataFrame(np.nan, index=states, columns=states)
        
        # 2. Parse Features and Fill Grid
        for feature, weight in zip(feature_names, comp):
            u, v = self._get_feature_tuple(feature)
            if u in states and v in states:
                matrix_df.loc[u, v] = weight

        # 3. Plot Heatmap (Poster Style)
        # Scale up all fonts globally
        sns.set_context("poster", font_scale=1.2)
        
        # Make the figure larger
        plt.figure(figsize=(14, 12))
        
        max_abs = max(abs(comp.min()), abs(comp.max()))
        ax = plt.gca()
        ax.set_facecolor('black') # Missing values appear black
        
        # Draw Heatmap
        sns.heatmap(matrix_df, 
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu",
                    center=0,
                    vmin=-max_abs,
                    vmax=max_abs,
                    square=True,
                    linewidths=1.0,  # Thicker grid lines
                    linecolor='gray',
                    # Bigger numbers inside the boxes
                    annot_kws={"size": 18, "weight": "bold"}, 
                    # Colorbar settings passed here, but we can fine-tune below
                    cbar=False) 
        
        # Manually add Colorbar to control size and font
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        cbar = plt.colorbar(ax.collections[0], cax=cax)
        
        # Colorbar Font Formatting
        cbar.set_label(f"PCA Weight (PC{pc_num})", fontsize=30, fontweight='bold', labelpad=20)
        cbar.ax.tick_params(labelsize=24)

        # Axis Label Formatting
        ax.set_title(f"PC{pc_num} Weights Heatmap\n({mode_name})", fontsize=40, fontweight='bold', pad=30)
        
        # Ticks
        ax.tick_params(axis='x', labelsize=24, rotation=45)
        ax.tick_params(axis='y', labelsize=24, rotation=0)

        plt.tight_layout()
        
        save_path = os.path.join(self.weights_dir, f"heatmap_weights_PC{pc_num}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   Saved Weights Heatmap: {save_path}")

    def _plot_elbow(self, X, mode_name):
        print("--- Generating Elbow Plot ---")
        max_pcs = min(100, min(X.shape))
        pca_full = PCA(n_components=max_pcs)
        pca_full.fit(StandardScaler().fit_transform(X))
        evr = pca_full.explained_variance_ratio_
        cum_var = np.cumsum(evr)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(evr)+1), cum_var, 'r-o')
        plt.axhline(0.9, color='g', linestyle='--')
        plt.title(f"Elbow Plot - {mode_name}")
        plt.savefig(os.path.join(self.variance_dir, "elbow_explained_variance.png"))
        plt.close()

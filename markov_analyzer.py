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
from exclusion_policy import DEFAULT_TOBII_SUCKS_XLSX

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

        for d in [self.output_dir, self.pca_dir, self.weights_dir,
                  self.variance_dir, self.consistency_dir, self.separation_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.global_states = self._get_global_states()
        self.demographics_text = self._build_demographics_text()
        
        self.special_names = {
            ('ENTER', 'ENTER'): "Non-Symbol Fixations",
            ('ENTER', 'EXIT'): "Skipped Search",
            ('EXIT', 'EXIT'): "Correct Search",
            ('EXIT', 'ENTER'): "Uncertainty"
        }

    # --- HELPERS ---
    def _build_demographics_text(self):
        """N, group split, and sex split for THIS analysis's actual participant
        population (post-exclusion) - so every population-level plot can show who's
        actually in it, not just the raw counts. Gender comes from the same behavioral
        summary table feature_pipeline.py already uses for demographics."""
        try:
            demo = pd.read_excel(DEFAULT_TOBII_SUCKS_XLSX)[["Patient_ID", "Gender"]]
            gender_map = demo.set_index("Patient_ID")["Gender"].astype(str).str.strip().str.upper().str[:1].to_dict()
        except Exception:
            gender_map = {}

        n = len(self.participants)
        n_hc = sum(1 for p in self.participants if p.group == "HC")
        n_ms = n - n_hc
        n_panels = sum(len(p.matrices) for p in self.participants)
        genders = [gender_map.get(p.name, "") for p in self.participants]
        n_f = sum(1 for g in genders if g == "F")
        n_m = sum(1 for g in genders if g == "M")
        n_unknown = n - n_f - n_m
        text = f"N={n} (HC={n_hc}, pwMS={n_ms})\nN panels={n_panels}\nSex: F={n_f}, M={n_m}"
        if n_unknown:
            text += f", unknown={n_unknown}"
        return text

    def _annotate_demographics(self, ax=None, loc="lower left", on_figure=False):
        """Adds self.demographics_text as a small text box - on the given/current axes
        by default, or in a figure-level corner (on_figure=True) for plots where every
        axes position is already full of real content (e.g. a heatmap)."""
        xy = (0.01, 0.01) if "lower" in loc else (0.01, 0.99)
        va = "bottom" if "lower" in loc else "top"
        target = plt.gcf() if on_figure else (ax or plt.gca())
        transform = plt.gcf().transFigure if on_figure else target.transAxes
        target.text(xy[0], xy[1], self.demographics_text, transform=transform,
                     fontsize=10, verticalalignment=va,
                     bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray'))

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
    def _fit_pca(self, n_components=None, save=True):
        """Builds the feature table and fits StandardScaler + PCA on it - the one fit
        both run_pca and run_panel_projection use, so a projection is always into
        exactly the space run_pca plots. Returns None if there isn't enough data."""
        # 1. BUILD OR FETCH DATA
        X, df_metadata, feature_names = self._build_and_save_feature_table(save=save)
        
        if X.size == 0:
            print("No data available for PCA.")
            return None

        # 2. PCA CLEANING
        X_var = np.var(X, axis=0)
        valid_mask = X_var > 1e-9 
        
        X_clean = X[:, valid_mask]
        feature_names_clean = feature_names[valid_mask]
        
        print(f"Features: {X.shape[1]} -> {X_clean.shape[1]} (Dropped constant features)")
        
        if len(X_clean) < 3:
            print("Not enough data.")
            return None

        # 3. RUN PCA
        max_comps = min(len(X_clean), X_clean.shape[1], 20) if n_components is None else n_components
        pca = PCA(n_components=max_comps)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        coords = pca.fit_transform(X_scaled)
        return {
            "pca": pca, "scaler": scaler, "coords": coords, "df_metadata": df_metadata,
            "X_clean": X_clean,
            # Columns kept by _build_and_save_feature_table (not all-NaN), the training
            # column means it imputed remaining NaNs with, and which of those columns
            # survived the constant-feature filter - everything a new vector needs to go
            # through the identical preprocessing before pca.transform.
            "feature_names": feature_names, "col_means": X.mean(axis=0),
            "valid_mask": valid_mask, "feature_names_clean": feature_names_clean,
        }

    def run_pca(self, n_components = None, save=True):
        fit = self._fit_pca(n_components=n_components, save=save)
        if fit is None:
            return
        pca, coords, df_metadata = fit["pca"], fit["coords"], fit["df_metadata"]
        X_clean, feature_names_clean = fit["X_clean"], fit["feature_names_clean"]
        explained_var = pca.explained_variance_ratio_
        mode_name = "Concatenated Panels" if self.config.concat_panels else ("Mean Matrix" if self.config.use_mean_matrix else "All Panels")

        for i in range(coords.shape[1]):
            df_metadata[f"PC{i+1}"] = coords[:, i]

        # 4. GROUP SEPARATION (still computed/saved for reference, no longer used to
        # pick which PCs to plot - decision 2026-09-27: always plot every PC1-3 pairwise
        # combination, a fixed and predictable set, rather than sometimes substituting
        # in whichever PC pair happens to separate groups best)
        self._analyze_group_separation(df_metadata)

        # 5. PLOTTING STRATEGY: always all PC1-3 pairwise combinations (PC1-2, PC1-3,
        # PC2-3) - fixed regardless of how many components exist or how they separate.
        pcs_to_plot = list(itertools.combinations(range(min(3, coords.shape[1])), 2))

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

    # =========================================================================
    # PART 1b: PER-PANEL PROJECTION INTO THE MEAN-MATRIX PCA SPACE
    # =========================================================================
    def run_panel_projection(self, n_pcs=3):
        """Fits the mean-matrix PCA exactly as run_pca does (one row per participant,
        _fit_pca), then projects every individual participant x panel matrix into that
        SAME space - same feature columns, same training-mean imputation, same
        StandardScaler and PCA weights; nothing is refit on the panels. Shows how a
        participant's single panels scatter around their mean-matrix position and
        whether the mean-matrix axes still order score/group at the single-panel level.
        Plots (PC1-3 pairwise + 3D), each colored by panel SDMT score and by group, once
        with each participant's mean-matrix point overlaid and once without, into
        <plot_output_path>/pca_panel_projection/{with_mean_overlay,no_overlay}/."""
        if not self.config.use_mean_matrix or self.config.concat_panels:
            print("run_panel_projection only applies to mean-matrix mode - skipping.")
            return
        fit = self._fit_pca(save=False)
        if fit is None:
            return
        pca, scaler = fit["pca"], fit["scaler"]
        n_pcs = min(n_pcs, pca.n_components_)
        all_feature_names = self._get_feature_names()
        keep_cols = np.isin(all_feature_names, fit["feature_names"])

        def project(matrix):
            vec = self._align_vector(matrix, fill_value=np.nan)[keep_cols]
            vec = np.where(np.isnan(vec), fit["col_means"], vec)
            return pca.transform(scaler.transform(vec[fit["valid_mask"]][None, :]))[0, :n_pcs]

        pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]
        panel_rows = []
        for p in self.participants:
            for panel, matrix in p.matrices.items():
                panel_rows.append({"Participant": p.name, "Group": p.group, "Panel": panel,
                                   "SDMT_Score": p.get_score(panel),
                                   **dict(zip(pc_cols, project(matrix)))})
        df_panels = pd.DataFrame(panel_rows)
        df_means = fit["df_metadata"][["Participant", "Group", "SDMT_Score"]].copy()
        for i, c in enumerate(pc_cols):
            df_means[c] = fit["coords"][:, i]

        out_root = os.path.join(self.output_dir, "pca_panel_projection")
        os.makedirs(out_root, exist_ok=True)
        df_panels.to_csv(os.path.join(out_root, "panel_projection_scores.csv"), index=False)
        df_means.to_csv(os.path.join(out_root, "mean_matrix_scores.csv"), index=False)
        print(f"Projected {len(df_panels)} panels from {len(df_means)} participants into the mean-matrix PCA space.")

        explained = pca.explained_variance_ratio_
        for overlay in (True, False):
            out_dir = os.path.join(out_root, "with_mean_overlay" if overlay else "no_overlay")
            os.makedirs(out_dir, exist_ok=True)
            for color_by in ("SDMT_Score", "Group"):
                for i, j in itertools.combinations(range(n_pcs), 2):
                    self._plot_projection_2d(df_panels, df_means if overlay else None, i, j, color_by,
                                             explained, out_dir)
                if n_pcs >= 3:
                    self._plot_projection_3d(df_panels, df_means if overlay else None, color_by,
                                             explained, out_dir)

    _GROUP_COLORS = {"HC": "blue", "pwMS": "red", "MS": "red"}

    # =========================================================================
    # PART 1c: SQUARED COSINES - HOW WELL EACH PC EXPLAINS EACH PARTICIPANT
    # =========================================================================
    def run_squared_cosines(self, n_pcs=3):
        """Squared cosine ("quality of representation") of each participant's mean
        matrix with each PC - Abdi & Williams (2010), 'Principal component analysis',
        WIREs Computational Statistics 2(4):433-459, section on contributions and
        squared cosines of observations. z = the participant's standardized mean-matrix
        vector (the exact PCA input, i.e. how they deviate from the average
        participant); cos2_k = score_k^2 / ||z||^2 = share of that deviation PC k
        explains. ||z|| is taken over the FULL standardized vector (not just the
        retained PCs), so cos2 summed over ALL possible PCs is 1 and
        rest = 1 - sum(cos2_PC1..n_pcs) is genuinely behaviour outside PC1..n_pcs.
        ||z|| itself = how atypical the participant is overall. Signed cosines are
        saved too (negative = the opposite pole of that PC). Mean-matrix mode only."""
        if not self.config.use_mean_matrix or self.config.concat_panels:
            print("run_squared_cosines only applies to mean-matrix mode - skipping.")
            return
        fit = self._fit_pca(save=False)
        if fit is None:
            return
        n_pcs = min(n_pcs, fit["pca"].n_components_)
        z = fit["scaler"].transform(fit["X_clean"])
        z_norm_sq = (z ** 2).sum(axis=1)
        scores = fit["coords"][:, :n_pcs]

        df = fit["df_metadata"][["Participant", "Group", "SDMT_Score"]].copy()
        pc_names = [f"PC{k + 1}" for k in range(n_pcs)]
        for k, name in enumerate(pc_names):
            df[name] = scores[:, k]
            df[f"cos_{name}"] = scores[:, k] / np.sqrt(z_norm_sq)
            df[f"cos2_{name}"] = scores[:, k] ** 2 / z_norm_sq
        cos2_cols = [f"cos2_{n}" for n in pc_names]
        df[f"cos2_PC1_{n_pcs}_sum"] = df[cos2_cols].sum(axis=1)
        df["cos2_rest"] = 1 - df[f"cos2_PC1_{n_pcs}_sum"]
        df["dominant_PC"] = df[cos2_cols].idxmax(axis=1).str.replace("cos2_", "")
        df["z_norm"] = np.sqrt(z_norm_sq)

        out_dir = os.path.join(self.output_dir, "squared_cosines")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "squared_cosines_per_participant.csv"), index=False)
        print(f"Squared cosines for {len(df)} participants: mean share PC1..{n_pcs} = "
              f"{df[cos2_cols].mean().round(3).to_dict()}, mean rest = {df['cos2_rest'].mean():.3f}")
        self._plot_squared_cosines(df, pc_names, fit["pca"].explained_variance_ratio_, out_dir)
        return df

    def _plot_squared_cosines(self, df, pc_names, explained, out_dir):
        """One stacked bar per participant (PC1..n shares + rest), sorted by mean SDMT
        score, with a group strip (HC blue / pwMS red) under the bars."""
        d = df.sort_values("SDMT_Score").reset_index(drop=True)
        x = np.arange(len(d))
        colors = ["#4C72B0", "#DD8452", "#55A868"][:len(pc_names)] + ["#D0D0D0"]
        fig, (ax, ax_grp) = plt.subplots(2, 1, figsize=(max(14, len(d) * 0.16), 7), sharex=True,
                                         gridspec_kw={"height_ratios": [12, 1], "hspace": 0.05})
        bottom = np.zeros(len(d))
        for name, color in zip(pc_names, colors):
            vals = d[f"cos2_{name}"].values
            k = int(name[2:]) - 1
            ax.bar(x, vals, bottom=bottom, color=color, width=0.85,
                   label=f"{name} (explains {explained[k] * 100:.1f}% of total variance)")
            bottom += vals
        ax.bar(x, d["cos2_rest"].values, bottom=bottom, color=colors[-1], width=0.85, label="Rest (outside PC1-3)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of participant's deviation explained (cos²)")
        ax.set_title("How well PC1-3 explain each participant's mean matrix (squared cosines)\n"
                     f"sorted by mean SDMT score (low -> high) | {self.config.folder_name}", fontsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        ax_grp.bar(x, np.ones(len(d)), width=1.0,
                   color=[self._GROUP_COLORS.get(g, "gray") for g in d["Group"]])
        ax_grp.set_yticks([])
        ax_grp.set_ylabel("Group", rotation=0, ha="right", va="center", fontsize=9)
        ax_grp.set_xticks(x)
        ax_grp.set_xticklabels([f"{s:.0f}" if pd.notna(s) else "" for s in d["SDMT_Score"]], rotation=90, fontsize=6)
        ax_grp.set_xlabel("Participant (label = mean SDMT score)")
        ax_grp.set_xlim(-0.6, len(d) - 0.4)
        from matplotlib.patches import Patch
        ax_grp.legend(handles=[Patch(color="blue", label="HC"), Patch(color="red", label="pwMS")],
                      loc="upper left", bbox_to_anchor=(1.005, 1.5), fontsize=9)
        self._annotate_demographics(on_figure=True)
        fig.savefig(os.path.join(out_dir, "squared_cosines_stacked_by_score.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _projection_title(self, color_by, overlay):
        what = "panel SDMT score" if color_by == "SDMT_Score" else "group"
        sub = "panels (small) + participant mean matrix (large)" if overlay else "individual panels"
        return f"Panels projected into mean-matrix PCA space - colored by {what}\n{sub} | {self.config.folder_name}"

    def _plot_projection_2d(self, df_panels, df_means, i, j, color_by, explained, out_dir):
        x, y = f"PC{i + 1}", f"PC{j + 1}"
        fig, ax = plt.subplots(figsize=(9, 7))
        panel_alpha = 0.45 if df_means is not None else 0.8
        if color_by == "SDMT_Score":
            vals = pd.concat([df_panels["SDMT_Score"]] + ([df_means["SDMT_Score"]] if df_means is not None else []))
            norm = plt.Normalize(np.nanmin(vals), np.nanmax(vals))
            sc = ax.scatter(df_panels[x], df_panels[y], c=df_panels["SDMT_Score"], cmap="viridis", norm=norm,
                            s=22, alpha=panel_alpha, edgecolors="none")
            if df_means is not None:
                ax.scatter(df_means[x], df_means[y], c=df_means["SDMT_Score"], cmap="viridis", norm=norm,
                           s=110, marker="D", edgecolors="black", linewidths=0.8)
            fig.colorbar(sc, ax=ax).set_label("SDMT score (panel; mean for overlay)")
        else:
            for group, g in df_panels.groupby("Group"):
                ax.scatter(g[x], g[y], color=self._GROUP_COLORS.get(group, "gray"), s=22, alpha=panel_alpha,
                           edgecolors="none", label=f"{group} panel")
            if df_means is not None:
                for group, g in df_means.groupby("Group"):
                    ax.scatter(g[x], g[y], color=self._GROUP_COLORS.get(group, "gray"), s=110, marker="D",
                               edgecolors="black", linewidths=0.8, label=f"{group} mean matrix")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel(f"{x} ({explained[i] * 100:.1f}%)")
        ax.set_ylabel(f"{y} ({explained[j] * 100:.1f}%)")
        ax.set_title(self._projection_title(color_by, df_means is not None), fontsize=11)
        ax.grid(True, alpha=0.3)
        self._annotate_demographics(ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"projection_{x}_{y}_{color_by}.png"), dpi=200)
        plt.close(fig)

    def _plot_projection_3d(self, df_panels, df_means, color_by, explained, out_dir):
        """Static PNG (fixed viewing angle) + interactive plotly HTML (rotatable) of PC1-3."""
        axis_labels = [f"PC{k + 1} ({explained[k] * 100:.1f}%)" for k in range(3)]
        title = self._projection_title(color_by, df_means is not None)
        panel_alpha = 0.45 if df_means is not None else 0.8

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        if color_by == "SDMT_Score":
            vals = pd.concat([df_panels["SDMT_Score"]] + ([df_means["SDMT_Score"]] if df_means is not None else []))
            norm = plt.Normalize(np.nanmin(vals), np.nanmax(vals))
            sc = ax.scatter(df_panels["PC1"], df_panels["PC2"], df_panels["PC3"], c=df_panels["SDMT_Score"],
                            cmap="viridis", norm=norm, s=14, alpha=panel_alpha)
            if df_means is not None:
                ax.scatter(df_means["PC1"], df_means["PC2"], df_means["PC3"], c=df_means["SDMT_Score"],
                           cmap="viridis", norm=norm, s=80, marker="D", edgecolors="black", linewidths=0.6)
            fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1).set_label("SDMT score")
        else:
            for group, g in df_panels.groupby("Group"):
                ax.scatter(g["PC1"], g["PC2"], g["PC3"], color=self._GROUP_COLORS.get(group, "gray"), s=14,
                           alpha=panel_alpha, label=f"{group} panel")
            if df_means is not None:
                for group, g in df_means.groupby("Group"):
                    ax.scatter(g["PC1"], g["PC2"], g["PC3"], color=self._GROUP_COLORS.get(group, "gray"), s=80,
                               marker="D", edgecolors="black", linewidths=0.6, label=f"{group} mean matrix")
            ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1]); ax.set_zlabel(axis_labels[2])
        ax.set_title(title, fontsize=10)
        self._annotate_demographics(on_figure=True)
        fig.savefig(os.path.join(out_dir, f"projection_3D_PC1_PC2_PC3_{color_by}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        try:
            import plotly.graph_objects as go
        except ImportError:
            print("plotly not installed - skipping interactive 3D HTML.")
            return
        traces = []

        def add(df, size, symbol, name_suffix, opacity, line_width):
            hover = (df["Participant"] + (" / " + df["Panel"] if "Panel" in df.columns else " / mean")
                     + "<br>score=" + df["SDMT_Score"].round(1).astype(str))
            if color_by == "SDMT_Score":
                traces.append(go.Scatter3d(
                    x=df["PC1"], y=df["PC2"], z=df["PC3"], mode="markers", text=hover, hoverinfo="text",
                    name=name_suffix, opacity=opacity,
                    marker=dict(size=size, symbol=symbol, color=df["SDMT_Score"], colorscale="Viridis",
                                cmin=vals.min(), cmax=vals.max(), line=dict(width=line_width, color="black"),
                                colorbar=dict(title="SDMT score") if name_suffix == "panels" else None)))
            else:
                for group, g in df.groupby("Group"):
                    traces.append(go.Scatter3d(
                        x=g["PC1"], y=g["PC2"], z=g["PC3"], mode="markers", text=hover[g.index], hoverinfo="text",
                        name=f"{group} {name_suffix}", opacity=opacity,
                        marker=dict(size=size, symbol=symbol, color=self._GROUP_COLORS.get(group, "gray"),
                                    line=dict(width=line_width, color="black"))))

        vals = pd.concat([df_panels["SDMT_Score"]] + ([df_means["SDMT_Score"]] if df_means is not None else []))
        add(df_panels, 3, "circle", "panels", panel_alpha, 0)
        if df_means is not None:
            add(df_means, 7, "diamond", "mean matrix", 1.0, 1)
        demo = self.demographics_text.replace("\n", "<br>")
        layout = go.Layout(title=title.replace("\n", "<br>"),
                           scene=dict(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1], zaxis_title=axis_labels[2]),
                           annotations=[dict(text=demo, showarrow=False, xref="paper", yref="paper", x=0, y=0,
                                             align="left", bordercolor="gray", borderwidth=1, bgcolor="white")])
        go.Figure(data=traces, layout=layout).write_html(
            os.path.join(out_dir, f"projection_3D_PC1_PC2_PC3_{color_by}.html"), include_plotlyjs="cdn")

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


    def _plot_pca_scatter(self, coords, color_values, labels, explained_var, mode_name, 
                          col_name, pc_x=0, pc_y=1, is_numeric=False):
        
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 6))

        # 1. SETUP DATA
        x_vals = coords[:, pc_x]
        y_vals = coords[:, pc_y]

        # 2. PLOTTING
        if is_numeric:
            # --- CONTINUOUS COLOR (SDMT Score) ---
            # Remove NaNs for plotting
            mask = ~np.isnan(color_values.astype(float))
            sc = plt.scatter(x_vals[mask], y_vals[mask], c=color_values[mask],
                             cmap='viridis', s=60, edgecolors='k', alpha=0.9)
            cbar = plt.colorbar(sc)
            cbar.set_label(col_name, fontsize=11, labelpad=10)

            title_text = f"PCA by {col_name}"
        else:
            # --- CATEGORICAL COLOR (Group) ---
            # red=MS/pwMS, blue=HC (decision 2026-09-27)
            unique_cats = np.unique(color_values)
            palette = {"HC": "blue", "pwMS": "red", "MS": "red"} if set(unique_cats).issubset({"HC", "pwMS", "MS"}) else None

            sns.scatterplot(x=x_vals, y=y_vals, hue=color_values, palette=palette,
                            s=60, edgecolor='k', alpha=0.9)
            plt.legend(title=col_name, loc='best')
            title_text = f"PCA by {col_name}"

        # 3. LABELS & TITLES
        var_x = explained_var[pc_x] * 100
        var_y = explained_var[pc_y] * 100

        plt.xlabel(f"PC{pc_x+1} ({var_x:.1f}%)")
        plt.ylabel(f"PC{pc_y+1} ({var_y:.1f}%)")
        plt.title(f"{title_text}\n({mode_name})")

        # Remove top/right spines
        sns.despine(trim=True)
        self._annotate_demographics()
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

        self._run_violin_analysis(valid_participants, valid_panels)
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

        # Formal test, decision 2026-09-27: is within-subject consistency actually
        # higher than between-subject, not just descriptively (mean/SD) different?
        # Mann-Whitney U (same nonparametric choice as _analyze_group_separation) since
        # correlation coefficients aren't reliably normal, especially near +/-1.
        u_stat, u_pval = mannwhitneyu(data[0], data[1], alternative='greater')

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
            f"BETWEEN:\nN = {n_b}\nMean = {mu_b:.3f}\nSD = {sd_b:.3f}\n\n"
            f"Mann-Whitney U (within > between):\nU = {u_stat:.1f}\np = {u_pval:.2e}"
        )
        # Position box to the right
        plt.text(2.6, np.mean([mu_w, mu_b]), stats_text, fontsize=10, 
                 verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        self._annotate_demographics()
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
        
        # --- PLOTTING ---
        mu_null = np.mean(valid_nulls)
        sd_null = np.std(valid_nulls)
        n_null = len(valid_nulls)

        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 6))

        plt.hist(valid_nulls, bins=50, color='gray', alpha=0.5, density=True, label='Null Distribution')
        plt.axvline(obs_mean, color='red', linestyle='--', linewidth=2, label='Observed')

        stats_text = (
            f"Observed Mean: {obs_mean:.4f}\n"
            f"Null Mean: {mu_null:.4f}\n"
            f"Null SD: {sd_null:.4f}\n"
            f"N Permutations: {n_null}\n"
            f"p-value: {p_val:.5f}"
        )

        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
        plt.text(0.95, 0.95, self.demographics_text, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray'))

        plt.title("Permutation Test: Within-Subject Consistency")
        plt.xlabel("Mean Correlation")
        plt.ylabel("Density")
        plt.legend(loc='upper right')

        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(self.consistency_dir, "permutation_test_histogram.png"), dpi=300)
        plt.close()

    def _plot_pca_weights(self, pca, feature_names, mode_name, target_pc_idx=0):
        """Plots PCA weights as a heatmap over the transition matrix grid. cmap
        "RdBu_r" (decision 2026-09-27): positive weights red, negative blue."""
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

        plt.figure(figsize=(9, 7.5))

        max_abs = max(abs(comp.min()), abs(comp.max()))
        ax = plt.gca()
        ax.set_facecolor('black') # Missing values appear black

        sns.heatmap(matrix_df,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    center=0,
                    vmin=-max_abs,
                    vmax=max_abs,
                    square=True,
                    linewidths=1.0,
                    linecolor='gray',
                    annot_kws={"size": 7},
                    cbar_kws={"label": f"PCA Weight (PC{pc_num})"})

        ax.set_title(f"PC{pc_num} Weights Heatmap\n({mode_name})")
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

        self._annotate_demographics(on_figure=True, loc="upper left")
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
        self._annotate_demographics()
        plt.savefig(os.path.join(self.variance_dir, "elbow_explained_variance.png"))
        plt.close()

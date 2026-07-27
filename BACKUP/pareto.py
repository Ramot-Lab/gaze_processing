import itertools
import re
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.preprocessing import StandardScaler # Fixed import (StandardScaler is in preprocessing, not discriminant_analysis)

def find_score(group, p_name, panel):
    sdmt_dir = os.path.join('/Volumes/ramot/Noam_M/Results/Behavior', group, p_name, "SDMT")
    if not os.path.isdir(sdmt_dir):
        # print(f"SDMT directory not found for {p_name} in group {group}.")
        return None
    
    for fname in os.listdir(sdmt_dir):
        if fname.endswith(".wav"):
            panel_match = re.search(r"img_test_(.+?)_strikes", fname)
            score_match = re.search(r"_strikes_(\d+)", fname)
            
            if panel_match and score_match:
                file_panel = panel_match.group(1)
                # Ensure exact panel match
                if file_panel == panel:
                    return int(score_match.group(1)) # Return immediately when found
    
    return None # Return None if loop finishes with no match

class ParetoFrontAnalyzer:
    
    def __init__(self, n_archetypes=4,
                with_repeats=False,
                 output_path="/Volumes/ramot/Noam_M/preliminary_results/pareto_analysis"):
        
        self.folder_name = "with repeats" if with_repeats else "no repeats"
        self.output_path = os.path.join(output_path, self.folder_name)
        markov_matrices_path = f"/Volumes/ramot/Noam_M/preliminary_results/{self.folder_name}/markov_matrices"

        # 1. Load Dictionary
        data_dict = self._load_all_markov_matrices(markov_matrices_path)
        
        # 2. Flatten AND Create Metadata simultaneously to ensure order
        df, self.metadata = self.flatten_markov_data(data_dict)

        self.data = df.values
        self.index = df.index
        self.n_archetypes = n_archetypes

        self.pca = None 
        self.pca_data = self.compute_pca()
        self.hull = self.compute_pareto_front()
        self.archetypes = self.extract_archetypes()

    # ---------------------------------------------------------
    @staticmethod
    def _load_all_markov_matrices(markov_matrices_path):
        """
        Loads data into a nested dictionary. 
        Stores (DataFrame, Score) tuples so they stay together.
        """
        data = {}
        
        if not os.path.exists(markov_matrices_path):
            print(f"Path not found: {markov_matrices_path}")
            return data

        for group in os.listdir(markov_matrices_path):
            group_path = os.path.join(markov_matrices_path, group)
            if not os.path.isdir(group_path):
                continue

            data[group] = {}

            for participant in os.listdir(group_path):
                p_path = os.path.join(group_path, participant)
                if not os.path.isdir(p_path):
                    continue

                data[group][participant] = {}

                for file in os.listdir(p_path):
                    if file.startswith("markov_matrix") and file.endswith(".csv"):
                        # Extract panel
                        try:
                            panel = file.split("_panel_")[1].replace(".csv", "")
                        except IndexError:
                            continue
                            
                        file_path = os.path.join(p_path, file)
                        
                        # Find score immediately
                        score = find_score(group, participant, panel)
                        
                        # Load Matrix
                        try:
                            df = pd.read_csv(file_path, index_col=0)
                            # Store both matrix and score together
                            data[group][participant][panel] = {
                                'matrix': df,
                                'score': score
                            }
                        except Exception as e:
                            print(f"Error loading {file}: {e}")

        return data

    # ---------------------------------------------------------
    def get_scores(self):
        # Extract scores safely from the aligned metadata
        scores = [m['score'] if m['score'] is not None else np.nan for m in self.metadata]
        return np.array(scores)

    # ---------------------------------------------------------
    @staticmethod
    def flatten_markov_data(data_dict):
        """
        Flattens the data and creates the metadata list in the EXACT same loop.
        This guarantees that rows[i] corresponds to metadata[i].
        """
        rows = []
        index = []
        metadata_list = [] # List of dicts for easier access
        columns = None

        # Iterate strictly through the dictionary
        for group in data_dict:
            for participant in data_dict[group]:
                for panel, content in data_dict[group][participant].items():
                    
                    matrix = content['matrix']
                    score = ['score']
                    
                    flat = matrix.values.flatten()
                    
                    # Set columns once based on the first matrix found
                    if columns is None:
                        n = matrix.shape[0]
                        columns = [f"({i},{j})" for i in range(n) for j in range(n)]

                    # Append Data
                    rows.append(flat)
                    index.append(f"{participant}_panel_{panel}")
                    
                    # Append Metadata exactly here
                    metadata_list.append({
                        'group': group,
                        'participant': participant,
                        'panel': panel,
                        'score': score
                    })

        return pd.DataFrame(rows, index=index, columns=columns), metadata_list

    # ---------------------------------------------------------
    def compute_pca(self):
        """Reduce to 3D for visualization + Pareto front."""
        self.pca = PCA(n_components=3) # Save instance to self.pca
        return self.pca.fit_transform(self.data)

    # ---------------------------------------------------------
    def compute_pareto_front(self):
        if self.pca_data is None:
            self.compute_pca()
        return ConvexHull(self.pca_data)
         
    # ---------------------------------------------------------
    @staticmethod
    def simplex_volume(points):
        k = points.shape[0]
        base = points[1:] - points[0]
        gram = base @ base.T
        det = np.linalg.det(gram)
        det = max(det, 0)
        return np.sqrt(det) / math.factorial(k - 1)

    # ---------------------------------------------------------
    def _max_volume_simplex(self, X, k):
        n = X.shape[0]
        if k > n:
            return X # Return all points if fewer than k
        
        best_volume = -1.0
        best_simplex = None

        # Optimization: If n is huge, combinations might freeze. 
        # For typical dataset sizes it is fine.
        for idxs in itertools.combinations(range(n), k):
            pts = X[list(idxs)]
            vol = self.simplex_volume(pts)
            if vol > best_volume:
                best_volume = vol
                best_simplex = pts

        return best_simplex

    # ---------------------------------------------------------
    def extract_archetypes(self):
        if self.hull is None:
            self.compute_pareto_front()
        hull_points = self.pca_data[self.hull.vertices]
        return self._max_volume_simplex(hull_points, self.n_archetypes)

# ============================================================
#                     VISUALIZATION
# ============================================================

    def save_3d_rotation(self, filename="pareto_rotation.mp4"):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.pca_data[:, 0], self.pca_data[:, 1], self.pca_data[:, 2], s=20, alpha=0.5)

        if self.archetypes is not None:
            A = self.archetypes
            for i in range(len(A)):
                for j in range(i+1, len(A)):
                    ax.plot([A[i,0], A[j,0]],
                            [A[i,1], A[j,1]],
                            [A[i,2], A[j,2]],
                            linewidth=3, color='r')

        def rotate(angle):
            ax.view_init(azim=angle)

        ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2))
        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, filename)
        ani.save(path, fps=20)
        plt.close()
        print(f"Saved rotating animation to {path}")

    # ---------------------------------------------------------
    def plot_pca_elbow(self, max_components=10, file_name="elbow.png"):
        n_col = min(max_components, self.data.shape[1], self.data.shape[0])
        pca_temp = PCA(n_components=n_col)
        pca_temp.fit(self.data)
        
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, n_col + 1),
                 pca_temp.explained_variance_ratio_,
                 marker="o")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Elbow Plot")
        plt.grid(True)

        os.makedirs(self.output_path, exist_ok=True)
        save_path = os.path.join(self.output_path, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    def save_pc_scatterplots(self, labels=None, save_prefix=None):
        """
        Unified function to save PC scatter plots.
        If labels are provided, points are colored accordingly.
        If no labels are provided, points are generic blue.
        """
        # Ensure we have PCA data
        if self.pca_data is None:
            self.compute_pca()
        pcs = self.pca_data

        pc_pairs = [(0,1), (0,2), (1,2)]
        os.makedirs(self.output_path, exist_ok=True)

        for i, j in pc_pairs:
            plt.figure(figsize=(6,5))

            if labels is None:
                plt.scatter(pcs[:,i], pcs[:,j], alpha=0.7)
            else:
                # Handle coloring
                labels = np.array(labels)
                
                # Check for NaNs in labels (scores might be NaN)
                valid_mask = ~pd.isnull(labels)
                plot_pcs = pcs[valid_mask]
                plot_labels = labels[valid_mask]
                
                # Check if categorical (strings/few ints) or continuous (many floats)
                unique_labels = np.unique(plot_labels)
                is_categorical = False
                if len(unique_labels) < 10 or isinstance(plot_labels[0], str):
                    is_categorical = True

                if is_categorical:
                    # Discrete colors
                    import matplotlib.cm as cm
                    colors = cm.get_cmap('tab10', len(unique_labels))
                    for k, ul in enumerate(unique_labels):
                        idx = plot_labels == ul
                        plt.scatter(plot_pcs[idx,i], plot_pcs[idx,j], 
                                    alpha=0.7, label=str(ul))
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # Continuous colors (heatmap style for scores)
                    sc = plt.scatter(plot_pcs[:,i], plot_pcs[:,j], 
                                c=plot_labels, cmap='viridis', alpha=0.7)
                    plt.colorbar(sc, label="Value")

            plt.xlabel(f"PC{i+1}")
            plt.ylabel(f"PC{j+1}")
            plt.title(f"PC{i+1} vs PC{j+1}")
            plt.grid(True)
            plt.tight_layout()

            # Construct filename
            prefix_str = f"{save_prefix}_" if save_prefix else ""
            filename = f"{prefix_str}pca_PC{i+1}_vs_PC{j+1}.png"
            path = os.path.join(self.output_path, filename)
            plt.savefig(path, dpi=300)
            plt.close()

        print(f"Saved PC scatter plots to {self.output_path}")

    # ---------------------------------------------------------
    def kmeans_clustering(self, labels=None, n_clusters=2, save_prefix="kmeans"):
        """
        Perform KMeans. 
        Note: 'labels' arg is for COMPARISON (e.g. true labels), 
        not for training (KMeans is unsupervised).
        """
        
        # Default labels from metadata if none provided
        if labels is None:
            labels = [m['group'] for m in self.metadata]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data)

        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # 1. Plot colored by the NEW Clusters
        self.save_pc_scatterplots(labels=cluster_labels, save_prefix=f"{save_prefix}_clusters")

        # 2. Plot colored by the TRUE labels (e.g. Group or Panel) for comparison
        if labels is not None:
            self.save_pc_scatterplots(labels=labels, save_prefix=f"{save_prefix}_truth")

        return cluster_labels

    def plot_pc_weights(self):
        if self.pca is None:
            self.compute_pca()
            
        components = self.pca.components_
        plt.figure(figsize=(10,6))
        for i in range(components.shape[0]):
            plt.plot(components[i], label=f'PC{i+1} weights')
        plt.xlabel('Feature Index')
        plt.ylabel('Weight')
        plt.title('PCA Component Weights')
        plt.legend()
        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, "pca_component_weights.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Ensure this path actually exists on your machine
    analyzer = ParetoFrontAnalyzer(n_archetypes=4, with_repeats=False)
    
    # Check if data loaded
    if analyzer.data.shape[0] == 0:
        print("No data loaded! Check paths.")
    else:
        analyzer.plot_pca_elbow()
        analyzer.plot_pc_weights()
        
        # Save generic PC plots (no color)
        analyzer.save_pc_scatterplots(save_prefix="basic")
        
        analyzer.save_3d_rotation(filename="pareto_no_repeats.mp4")
        
        # Clustering
        cluster_labels = analyzer.kmeans_clustering(n_clusters=2, save_prefix="kmeans")
        
        # Scores plotting
        scores_labels = analyzer.get_scores()
        analyzer.save_pc_scatterplots(labels=scores_labels, save_prefix="scores")
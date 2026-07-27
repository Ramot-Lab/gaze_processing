import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import os

class MarkovVisualizer:
    """
    Handles plotting of existing matrices stored in Participant objects.
    """
    def __init__(self, output_dir=None):
        self.output_dir = output_dir

    def plot_heatmap(self, participant, panel, save=True):
        """
        Plots the transition matrix for a specific participant and panel.
        """
        if panel not in participant.matrices:
            print(f"Warning: Panel {panel} not found for {participant.name}")
            return

        # Get the matrix (it already has NaNs from the Model class)
        df = participant.matrices[panel]
        
        plt.figure(figsize=(10, 8))
        
        # Setup Colormap: NaNs = Gray, Probabilities = Blue-Green
        cmap = sns.color_palette("YlGnBu", as_cmap=True)
        cmap.set_bad("gray", 0.3)
        
        sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Transitions: {participant.name} ({panel})")
        plt.ylabel("From State")
        plt.xlabel("To State")
        
        if save and self.output_dir:
            save_path = os.path.join(self.output_dir, "heatmaps", participant.group, participant.name)
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"heatmap_{panel}.png"))
            plt.close()
        else:
            plt.show()

    def plot_graph(self, participant, panel, threshold=0.11, margin=0.15, save=True):
        """
        Plots a network graph of the transitions.
        """
        if panel not in participant.matrices:
            return

        df = participant.matrices[panel]
        
        # Graph logic cannot handle NaNs, fill with 0 for edges
        df_clean = df.fillna(0)
        
        G = nx.DiGraph()
        for u in df_clean.index:
            for v in df_clean.columns:
                w = df_clean.loc[u, v]
                if w >= threshold:
                    G.add_edge(u, v, weight=w)

        # Plotting
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw Nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', edgecolors='k')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

        # Draw Edges (Custom Arrows)
        for u, v, d in G.edges(data=True):
            if u not in pos or v not in pos: continue
            
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx, dy = x2 - x1, y2 - y1
            
            # Shorten arrows
            new_x1 = x1 + margin * dx
            new_y1 = y1 + margin * dy
            new_x2 = x2 - margin * dx
            new_y2 = y2 - margin * dy
            
            # Calculate angle for text
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < -90 or angle > 90: angle += 180

            plt.arrow(new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1,
                      length_includes_head=True, head_width=0.05, head_length=0.08,
                      fc='k', ec='k', linewidth=1 + (d['weight'] * 3), alpha=0.7)
            
            # Label
            plt.text((new_x1 + new_x2)/2, (new_y1 + new_y2)/2, f"{d['weight']:.2f}",
                     rotation=angle, fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

        plt.title(f"Transition Graph: {participant.name} ({panel})")
        plt.axis('off')

        if save and self.output_dir:
            save_path = os.path.join(self.output_dir, "graphs", participant.group, participant.name)
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"graph_{panel}.png"))
            plt.close()
        else:
            plt.show()
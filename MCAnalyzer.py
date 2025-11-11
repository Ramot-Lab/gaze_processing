from collections import defaultdict

import pandas as pd

from Trial import Trial

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class MCAnalyzer():

    def __init__(self, trials: list[Trial]):
        self.trials = trials
        self.transitions = self._analyze_transitions()

    def _analyze_transitions(self) -> dict[tuple[str, str], int]:
        transitions = defaultdict(int)

        for trial in self.trials:
            searches = [s for s in trial.searches if s.sequence]
            if not searches:
                continue

            for si, search in enumerate(searches):
                seq = [str(s.value) for s in search.sequence.values() if s]
                if not seq:
                    continue

                # Enter transition
                transitions[("ENTER", seq[0])] += 1

                # Internal transitions
                for a, b in zip(seq[:-1], seq[1:]):
                    transitions[(a, b)] += 1

                # Exit transition
                transitions[(seq[-1], "EXIT")] += 1

                # Between-search transition: EXIT of current → ENTER of next
                if si < len(searches) - 1:
                    transitions[("EXIT", "ENTER")] += 1
                else:
                    # Last search in trial goes to EXIT
                    transitions[("EXIT", "EXIT")] += 1

        return dict(transitions)


    def _get_transition_table(self) -> pd.DataFrame:
        """go from dict to dataframe -> rows = From, columns = To"""
        states = ["ENTER", "1", "2", "3", "4", "5", "6", "7", "8", "9", "EXIT"]

        transition_counts = pd.DataFrame(0, index=states, columns=states)

        for (from_state, to_state), count in self.transitions.items():
            transition_counts.loc[from_state, to_state] += count
        return transition_counts

    def get_transition_probabilities(self, nodes_to_drop = None) -> pd.DataFrame:
        """
        Get transition probabilities as a DataFrame.
        Rows = FROM state, Columns = TO state.
        """

        transition_counts = self._get_transition_table()
        if nodes_to_drop:
            transition_counts = transition_counts.drop(index=nodes_to_drop, columns=nodes_to_drop, errors='ignore')
        transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
        return transition_probabilities

    def stationary_distribution(self, tol=1e-12):
        """
        Compute the stationary distribution of a Markov chain

        tol : float ; Numerical tolerance for eigenvalue comparison.

        Returns:
        pi : np.ndarray ; vector of shape (n,) of the stationary distribution
        """

        #FIXME: not sure what is the meaning of it.
        
        P = self.get_transition_probabilities().to_numpy()
        eigvals, eigvecs = np.linalg.eig(P.T)

        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigvals - 1))
        pi = np.real(eigvecs[:, idx])

        # Normalize so that it sums to 1
        pi = pi / np.sum(pi)

        # Ensure all values are non-negative
        pi = np.where(pi < tol, 0, pi)

        return pi


    def plot_transition_graph(self, threshold: float = 1/9, margin: float = 0.15):
        """
        Visualize gaze transition graph with shorter arrows that don't overlap nodes.
        
        Parameters:
            threshold: float - minimum probability to show an edge
            margin: float - fraction of distance to shorten the arrow at both ends
        """


        prob_matrix = self.get_transition_probabilities()
        G = nx.DiGraph()

        for node in prob_matrix.index:
            G.add_node(node)

        for from_node in prob_matrix.index:
            for to_node in prob_matrix.columns:
                prob = prob_matrix.loc[from_node, to_node]
                if prob >= threshold:
                    G.add_edge(from_node, to_node, weight=prob)

        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')

        # Draw edges manually with shortened lines
        for u, v, d in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            # Flip text if upside down
            if angle < -90 or angle > 90:
                angle += 180

            # shorten both ends
            new_x1 = x1 + margin * dx
            new_y1 = y1 + margin * dy
            new_x2 = x2 - margin * dx
            new_y2 = y2 - margin * dy

            plt.arrow(new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1,
                    length_includes_head=True,
                    head_width=0.06, head_length=0.08,
                    fc='k', ec='k',
                    linewidth=d['weight']*5)
            plt.text((new_x1 + new_x2) / 2, (new_y1 + new_y2) / 2,
                    f"{d['weight']:.2f}",
                    rotation=angle,
                    fontsize=7,
                    ha='center',
                    va='center',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        plt.axis('off')
        plt.title("Gaze Transition Graph")
        plt.show()


    def plot_transition_graph_no_enter_exit(self, threshold: float = 1/9, margin: float = 0.15):
        """
        Visualize gaze transition graph (without ENTER/EXIT) where each connectivity component
        is shown separately and spaced apart. Edges are weighted by transition probability.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        prob_matrix = self.get_transition_probabilities(nodes_to_drop=["ENTER", "EXIT"])

        nodes = prob_matrix.index.tolist()

        # Build directed graph
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node)

        for from_node in nodes:
            for to_node in nodes:
                prob = prob_matrix.loc[from_node, to_node]
                if prob >= threshold:
                    G.add_edge(from_node, to_node, weight=prob)

        # If no edges left, just stop
        if not G.edges:
            print("No edges above threshold.")
            return

        # Get connected components (weakly connected for directed graph)
        components = list(nx.weakly_connected_components(G))
        pos = {}

        # Layout each component separately with spacing
        offset_x = 0
        for comp in components:
            subG = G.subgraph(comp)
            sub_pos = nx.spring_layout(subG, seed=42)
            # Offset component positions so they don’t overlap
            for k in sub_pos:
                sub_pos[k][0] += offset_x
            pos.update(sub_pos)
            offset_x += 2.5  # space between components

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

        # Draw shortened arrows with probability labels
        for u, v, d in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))

            new_x1 = x1 + margin * dx
            new_y1 = y1 + margin * dy
            new_x2 = x2 - margin * dx
            new_y2 = y2 - margin * dy

            plt.arrow(new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1,
                    length_includes_head=True,
                    head_width=0.06, head_length=0.08,
                    fc='k', ec='k', alpha=0.7,
                    linewidth=d['weight'] * 5)

            # Probability label near middle of arrow
            plt.text((new_x1 + new_x2) / 2, (new_y1 + new_y2) / 2,
                    f"{d['weight']:.2f}",
                    rotation=angle,
                    fontsize=7,
                    ha='center',
                    va='center',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.axis('off')
        plt.title("Gaze Transition Graph (No ENTER/EXIT, Connectivity Components)")
        plt.show()
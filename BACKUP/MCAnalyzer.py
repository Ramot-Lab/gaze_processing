from collections import defaultdict
import os
import pandas as pd
from SDMT_Search_Processor.SearchFinder import Search  
from BACKUP.Trial import Trial
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class MCAnalyzer():

    def __init__(self, trials: list[Trial], only_keys: bool = True):

        self.trials = trials
        self.only_keys = only_keys
        if only_keys:
            self.transiotions = self._analyze_only_states_transitions(clean = False)
            self.cleaned_transition = self._analyze_only_states_transitions(clean = True)
        else:
            self.transitions = self._analyze_all_transitions(clean = False)
            self.cleaned_transition = self._analyze_all_transitions(clean = True)

    def _analyze_only_states_transitions(self, clean : bool) -> dict[tuple[str, str], int]:
        transitions = defaultdict(int)

        for trial in self.trials:
            if not clean:
                searches :list[Search] = [s for s in trial.searches if s.sequence]
            else:
                searches :list[Search] = [s for s in trial.searches if s.cleaned_sequence]
            if searches == []:
                continue
                
            for search in searches:
                if not clean:
                    seq = [str(s.value) for s in search.sequence.values() if s]
                else:
                    seq = [str(s.value) for s in search.cleaned_sequence.values() if s]
                if seq == []:
                    continue
                for a, b in zip(seq[:-1], seq[1:]):
                    transitions[(a, b)] += 1
        return dict(transitions)

    def _analyze_all_transitions(self, clean : bool) -> dict[tuple[str, str], int]:
        transitions = defaultdict(int)

        for trial in self.trials:
            if not clean:
                searches :list[Search] = [s for s in trial.searches if s.sequence]
            else:
                searches :list[Search] = [s for s in trial.searches if s.cleaned_sequence]
            if searches == []:
                transitions[("ENTER", "EXIT")] += 1 # haven't even gone up to the dictionary area
                continue

            for si, search in enumerate(searches):
                if not clean:
                    seq = [str(s.value) for s in search.sequence.values() if s]
                else:
                    seq = [str(s.value) for s in search.cleaned_sequence.values() if s]
                if seq == []: # all fixations are aoutside ROIs meaning [NONE, NONE, NONE...]
                    transitions[("ENTER", "ENTER")] += 1
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

    def _get_transition_table(self, clean : bool) -> pd.DataFrame:
        """go from dict to dataframe -> rows = From, columns = To"""
        if self.only_keys:
            states = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        else:
            states = ["ENTER", "1", "2", "3", "4", "5", "6", "7", "8", "9", "EXIT"]

        transition_counts = pd.DataFrame(0, index=states, columns=states)
        transitions = self.cleaned_transition if clean else self.transitions
        for (from_state, to_state), count in transitions.items():
            transition_counts.loc[from_state, to_state] += count
        return transition_counts

    def get_transition_probabilities(self, clean : bool) -> pd.DataFrame:
        """
        Get transition probabilities as a DataFrame.
        Rows = FROM state, Columns = TO state.
        """

        transition_counts = self._get_transition_table(clean = clean)
        transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
        return transition_probabilities

    def stationary_distribution(self, nodes_to_drop = None, tol=1e-12):
        """
        Compute the stationary distribution of a Markov chain
        nodes_to_drop : list[str] ; List of nodes to exclude from the computation - 
                                    probabilities will be normalized before stationary dist. is calculated.
        tol : float ; Numerical tolerance for eigenvalue comparison.

        Returns:
        pi : np.ndarray ; vector of shape (n,) of the stationary distribution
        """

        #FIXME: not sure what is the meaning of it.

        P = self.get_transition_probabilities()
        if nodes_to_drop:
            indices_to_keep = [i for i, state in enumerate(P.index) if state not in nodes_to_drop]
            P = P.iloc[(indices_to_keep, indices_to_keep)].to_numpy()
            # Normalize rows again after dropping nodes - each state has to have an added up probability of 1
            P = P / P.sum(axis=1, keepdims=True)
        else:
            P = P.to_numpy()
        eigvals, eigvecs = np.linalg.eig(P.T)

        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigvals - 1))
        pi = np.real(eigvecs[:, idx])

        # Normalize so that it sums to 1
        pi = pi / np.sum(pi)

        # Ensure all values are non-negative
        pi = np.where(pi < tol, 0, pi)

        return pi

    def build_directed_graph(self, threshold = 0) -> nx.DiGraph:
        """
        Build a directed graph from the transition probabilities.
        """

        prob_matrix = self.get_transition_probabilities()
        G = nx.DiGraph()

        for from_node in prob_matrix.index:
            for to_node in prob_matrix.columns:
                prob = prob_matrix.loc[from_node, to_node]
                if prob >= threshold:
                    G.add_edge(from_node, to_node, weight=prob)
        return G

    def plot_transition_graph(self, threshold: float = 1/9, margin: float = 0.15):
        """
        Visualize gaze transition graph with shorter arrows that don't overlap nodes.
        
        Parameters:
            threshold: float - minimum probability to show an edge
            margin: float - fraction of distance to shorten the arrow at both ends
        """


        G = self.build_directed_graph(threshold=threshold)

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

    def _most_probable_l_len_paths(self, path_length: int, n: int = 5) -> list[tuple[list[str], float]]:
        """
        Compute the top-n most probable simple paths of EXACT given length
        (in number of nodes) from ENTER to EXIT.

        path_length : int
            Number of nodes in the path (ENTER,...,EXIT). 
            So path_length = 5 → sequences like ENTER → 2 → 7 → 9 → EXIT (5 nodes).

        n : int
            Number of most probable paths to return.

        Returns: list of (path, probability)
        """

        G = self.build_directed_graph()

        start_node = "ENTER"
        end_node = "EXIT"

        # Get all simple paths up to given length
        all_paths = list(nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=path_length))

        # Keep only paths EXACTLY matching the requested length
        exact_paths = [p for p in all_paths if len(p) == path_length]

        path_probs = []
        for path in exact_paths:
            prob = 1.0
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                w = G[u][v].get('weight', 0.0)
                prob *= w
            path_probs.append((path, prob))

        # Sort by probability (descending)
        path_probs.sort(key=lambda x: x[1], reverse=True)

        # Return top-n
        return [(p, round(prob, 5)) for p, prob in path_probs[:n]]

    def most_probable_paths(self, n: int = 3, max_length: int = 5) -> list[tuple[list[str], float]]:
        """
        Public method to get the most probable paths.
        """
        most_probable_paths = {}
        for l in range(3, max_length + 1):
            most_probable_paths[l] = self._most_probable_l_len_paths(path_length=l, n=n)
        return most_probable_paths

    def create_transition_probabilities_heatmap(self, p_name: str, panel: str, output_path: str = None):
        p = self.get_transition_probabilities()
        if output_path is None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(p, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title(f"Transition Probabilities for Participant {p_name} on Panel {panel}")
            plt.xlabel("To Symbol")
            plt.ylabel("From Symbol")
            plt.show()
        else:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(
            output_path,
            f"transition_probabilities_heatmap_{p_name}_panel{panel}.png"
        )
            plt.figure(figsize=(10, 8))
            sns.heatmap(p, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title(f"Transition Probabilities for Participant {p_name} on Panel {panel}")
            plt.xlabel("To Symbol")
            plt.ylabel("From Symbol")
            plt.savefig(output_file)
            plt.close()

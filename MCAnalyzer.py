# from typing import List, Dict, Any, Optional
# from pydtmc import MarkovChain
# import numpy as np
# from FixationHandler import Fixation
# from RoiFinder import ROI, RoiFinder
# from SearchFinder import Search, SearchFinder
import networkx as nx
from collections import defaultdict

from SearchFinder import Search


# class MCAnalyzer:
#     def __init__(self, rois: List[ROI], search_finder: SearchFinder, use_nearest: bool = True):
#         """
#         Parameters
#         ----------
#         rois : list
#             A list of ROI objects (or ROI identifiers).
#         use_nearest : bool
#             If True, assign a fixation to the nearest ROI if it doesn't fall inside one.
#         """
#         self.rois = rois
#         self.use_nearest = use_nearest
#         self.markov_chain: Optional[MarkovChain] = None
#         # self.search_finder = search_finder
#         self.roi_sequences = search_finder.get_roi_sequences(searches=search_finder.find())



# ### what did i ddoooo
#     def build_markov_chain(self):
#         """
#         Build a Markov chain model from stored ROI sequences.
#         """
#         if not self.roi_sequences:
#             raise ValueError("No ROI sequences found. First run search_finder.get_roi_sequences().")

#         # Flatten transitions
#         transitions = []
#         for idx, seq in self.roi_sequences.items():
#             for a, b in zip(seq[:-1].symbol, seq[1:].symbol):
#                 transitions.append((a, b))

#         # Build transition matrix
#         n = 9  # ROIs are labeled from 1 to 9
#         matrix = np.zeros((n, n))
#         for a, b in transitions:
#             matrix[a, b] += 1

#         # Normalize to probabilities
#         for i in range(n):
#             row_sum = matrix[i, :].sum()
#             if row_sum > 0:
#                 matrix[i, :] /= row_sum

#         # Create pydtmc chain
#         self.markov_chain = MarkovChain(matrix, [str(i) for i in range(n)])
#         return self.markov_chain
    



def build_symbol_transition_graph(searches: list[Search]):
    """
    Build a weighted directed graph of symbol-key transitions (Markov chain style).

    Parameters
    ----------
    searches : list[Search]
        Each Search contains a list of Fixation objects already mapped to symbols.

    Returns
    -------
    G : networkx.DiGraph
        Weighted directed graph with transition probabilities.
    """

    transition_counts = defaultdict(lambda: defaultdict(int))

    for search in searches[2:]:
        # extract the sequence of symbols visited
        sequence = [fix.mapped_symbol for fix in search.fixations if hasattr(fix, 'mapped_symbol') and fix.mapped_symbol is not None]

        if not sequence:
            continue

        # Add ENTER -> first symbol
        transition_counts["ENTER_SEARCH"][sequence[0]] += 1

        # Add within-search transitions
        for i in range(len(sequence) - 1):
            a, b = sequence[i], sequence[i + 1]
            transition_counts[a][b] += 1

        # Add last symbol -> EXIT
        transition_counts[sequence[-1]]["EXIT_SEARCH"] += 1

    # Convert counts to probabilities
    G = nx.DiGraph()
    for src, targets in transition_counts.items():
        total = sum(targets.values())
        for dst, count in targets.items():
            prob = count / total
            G.add_edge(src, dst, weight=prob)

    return G



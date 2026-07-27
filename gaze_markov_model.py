from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx

class GazeMarkovModel:
    """
    Handles the mathematical modeling of Gaze Transitions.
    """
    def __init__(self, trials: list, only_keys: bool = True):
        self.trials = trials
        self.only_keys = only_keys
        
        # Calculate transitions immediately
        self.transitions = self._count_transitions(use_cleaned_seq=False)
        self.clean_transitions = self._count_transitions(use_cleaned_seq=True)

    def _count_transitions(self, use_cleaned_seq: bool) -> dict:
        """
        Counts A->B transitions.
        """
        counts = defaultdict(int)
        
        for trial in self.trials:
            # 1. Select the searches
            if use_cleaned_seq:
                searches = [s for s in trial.searches if s.cleaned_sequence]
            else:
                searches = [s for s in trial.searches if s.sequence]

            # 2. Handle Empty Trial
            if not searches:
                counts[("ENTER", "EXIT")] += 1
                continue

            # 3. Process each Search
            for si, search in enumerate(searches):
                if use_cleaned_seq:
                    source = search.cleaned_sequence
                else:
                    source = search.sequence
                
                seq = [str(s.value) for s in source.values() if s]
                
                # Handle Empty Search : there are fixations in search but not no symbols (ENTER -> ENTER)
                if not seq:
                    counts[("ENTER", "ENTER")] += 1
                    continue

                # --- FULL TRANSITION LOGIC ---
                
                # A. Start: ENTER -> First ROI
                counts[("ENTER", seq[0])] += 1
                
                # B. Internal: ROI -> ROI
                for a, b in zip(seq[:-1], seq[1:]):
                    counts[(a, b)] += 1
                
                # C. End: Last ROI -> EXIT
                counts[(seq[-1], "EXIT")] += 1
                
                # D. Between Searches or End of Trial
                if si < len(searches) - 1:
                    counts[("EXIT", "ENTER")] += 1
                else:
                    counts[("EXIT", "EXIT")] += 1
                        
        return dict(counts)

    def get_probability_matrix(self, clean: bool) -> pd.DataFrame:
        """Returns the Row-Stochastic Transition Matrix (with NaNs)."""
        # Define states
        if self.only_keys:
            states = [str(i) for i in range(1, 10)]
        else:
            states = ["ENTER"] + [str(i) for i in range(1, 10)] + ["EXIT"]

        # 1. Build Count DataFrame (Initialize with 0 because we need to count)
        df_counts = pd.DataFrame(0, index=states, columns=states)
        source_dict = self.clean_transitions if clean else self.transitions
        
        for (u, v), c in source_dict.items():
            if u in states and v in states:
                df_counts.loc[u, v] += c
        
        # 2. Normalize (Count -> Probability)
        # Rows with sum=0 (unvisited states) will become 0.
        probs = df_counts.div(df_counts.sum(axis=1), axis=0).fillna(0)

        # 3. Apply Structural NaNs = Impossible Transitions
        # A. Diagonal Logic
        if clean:
            # Only remove diagonal for 1-9. We preserve ENTER->ENTER and EXIT->EXIT.
            targets = [str(i) for i in range(1, 10)]
            for t in targets:
                if t in probs.index and t in probs.columns:
                    probs.loc[t, t] = np.nan
        
        # B. Model Logic (Digits -> ENTER, EXIT -> Digits)
        if not self.only_keys:
            digits = [str(i) for i in range(1, 10)]
            # Cannot go back to start
            probs.loc[digits, "ENTER"] = np.nan
            # EXIT can only go to ENTER or EXIT, never to a digit
            probs.loc["EXIT", digits] = np.nan
        
        return probs
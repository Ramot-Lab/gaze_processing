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
        self.transitions = self._count_transitions(self._trials_to_seqs(use_cleaned_seq=False))
        self.clean_transitions = self._count_transitions(self._trials_to_seqs(use_cleaned_seq=True))

    @classmethod
    def from_trial_stats(cls, trial_stats_df: pd.DataFrame, only_keys: bool = True):
        """Rebuilds the model from a saved trial_stats_<p>_panel_<panel>.csv
        (TrialManager.add_trial_statistics_dataframe) instead of live Trial objects - its
        Sequence/Cleaned_Sequence columns hold exactly the per-search state sequences
        _count_transitions consumes ("|" within a search, ";" between searches, "None"
        for a fixation with no symbol), so the resulting counts are identical to the
        live computation without re-reading Stage 1's annotated gaze."""
        model = cls.__new__(cls)
        model.trials = None
        model.only_keys = only_keys
        model.transitions = cls._count_transitions(cls._parse_seq_column(trial_stats_df["Sequence"]))
        model.clean_transitions = cls._count_transitions(cls._parse_seq_column(trial_stats_df["Cleaned_Sequence"]))
        return model

    @staticmethod
    def _parse_seq_column(col) -> list:
        """trial_stats string column -> [trial][search][symbol str]. An empty cell (NaN
        once read back by pandas) is a trial with no non-empty searches."""
        trials = []
        for cell in col:
            if not isinstance(cell, str) or cell == "":
                trials.append([])
                continue
            trials.append([[v for v in search.split("|") if v != "None"]
                           for search in cell.split(";")])
        return trials

    def _trials_to_seqs(self, use_cleaned_seq: bool) -> list:
        """Live Trial objects -> [trial][search][symbol str], same shape as
        _parse_seq_column."""
        trials = []
        for trial in self.trials:
            if use_cleaned_seq:
                searches = [s for s in trial.searches if s.cleaned_sequence]
            else:
                searches = [s for s in trial.searches if s.sequence]
            trials.append([
                [str(sym.value) for sym in (s.cleaned_sequence if use_cleaned_seq else s.sequence).values() if sym]
                for s in searches
            ])
        return trials

    @staticmethod
    def _count_transitions(trial_seqs: list) -> dict:
        """
        Counts A->B transitions.
        trial_seqs: [trial][search][symbol str], only non-empty searches included.
        """
        counts = defaultdict(int)
        
        for searches in trial_seqs:
            # 1. Handle Empty Trial
            if not searches:
                counts[("ENTER", "EXIT")] += 1
                continue

            # 2. Process each Search
            for si, seq in enumerate(searches):
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

    def get_count_matrix(self, clean: bool) -> pd.DataFrame:
        """Raw A->B transition counts over this model's state set (no structural NaNs)."""
        # Define states
        if self.only_keys:
            states = [str(i) for i in range(1, 10)]
        else:
            states = ["ENTER"] + [str(i) for i in range(1, 10)] + ["EXIT"]

        # Initialize with 0 because we need to count
        df_counts = pd.DataFrame(0, index=states, columns=states)
        source_dict = self.clean_transitions if clean else self.transitions
        
        for (u, v), c in source_dict.items():
            if u in states and v in states:
                df_counts.loc[u, v] += c
        return df_counts

    def get_probability_matrix(self, clean: bool) -> pd.DataFrame:
        """Returns the Row-Stochastic Transition Matrix (with NaNs): P(B | A) =
        count(A->B) / count(A->anything)."""
        df_counts = self.get_count_matrix(clean)
        
        # Normalize (Count -> Probability)
        # Rows with sum=0 (unvisited states) will become 0.
        probs = df_counts.div(df_counts.sum(axis=1), axis=0).fillna(0)
        return self._apply_structural_nans(probs, clean)

    def get_joint_probability_matrix(self, clean: bool) -> pd.DataFrame:
        """Returns the joint transition-frequency matrix (with NaNs): P(A->B) =
        count(A->B) / total count of all transitions in the panel's matrix, so the WHOLE
        matrix (not each row) sums to 1. Unlike the row-stochastic version this keeps
        how often each source state is visited instead of conditioning it away.
        The denominator only counts transitions between the states in this matrix
        (e.g. with only_keys, ENTER/EXIT transitions are not part of the total).
        Same structural NaNs as get_probability_matrix (those cells always count 0)."""
        df_counts = self.get_count_matrix(clean)
        total = df_counts.values.sum()
        joint = df_counts / total if total > 0 else df_counts.astype(float)
        return self._apply_structural_nans(joint, clean)

    def get_per_score_matrix(self, clean: bool, score: float) -> pd.DataFrame:
        """Returns count(A->B) / the panel's SDMT score (with NaNs) - transitions per
        correctly-answered item. Not normalized to sum to 1, so it keeps HOW MUCH
        searching happened, scaled by task performance. Note the score is used as given,
        even though the first practice trials are dropped upstream (so the panel has
        score - 2 analyzed trials).
        Same structural NaNs as get_probability_matrix (those cells always count 0)."""
        df_counts = self.get_count_matrix(clean)
        return self._apply_structural_nans(df_counts / float(score), clean)

    def get_digit_asymmetry(self, clean: bool):
        """Total asymmetry of symbol->symbol (1-9 -> 1-9) transitions, ignoring every
        transition that involves ENTER/EXIT (those are one-way by design).
        J = digit-block counts / total digit->digit transitions (the joint matrix
        restricted to 1-9, re-normalized to sum to 1); asymmetry = 1/2 * sum|J - J^T|
        = sum over unordered pairs {a,b} of |J(a->b) - J(b->a)| - the fraction of
        transitions that are net one-directional: 0 = every a->b matched by b->a
        (detailed balance, pure back-and-forth), 1 = fully one-way. Count-based on
        purpose: on the row-conditional P(B | A) matrix, P(a->b) != P(b->a) even with
        perfectly balanced traffic. NOTE: biased upward at low counts (even a
        symmetric process gives |C_ab - C_ba| > 0 by chance), so n_digit_transitions
        is returned alongside it for control. Returns (asymmetry, n_digit_transitions)."""
        digits = [str(i) for i in range(1, 10)]
        counts = self.get_count_matrix(clean).loc[digits, digits].values.astype(float)
        n = counts.sum()
        if n == 0:
            return np.nan, 0
        joint = counts / n
        return 0.5 * np.abs(joint - joint.T).sum(), int(n)

    def _apply_structural_nans(self, probs: pd.DataFrame, clean: bool) -> pd.DataFrame:
        """Apply Structural NaNs = Impossible Transitions"""
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

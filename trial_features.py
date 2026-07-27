import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from PanelMessages import Press
from SearchFinder import Search
from trial import Trial
from RoiFinder import ROI


class TrialFeatures:
    """
    Responsible for extracting low and mid-level statistical features 
    from the trials and their associated searches/fixations.
    All features are computed and stored upon instantiation.
    """
    def __init__(self, trials: list[Trial], rois: list[ROI], presses: list[Press], searches: list[Search]):
        self.trials = trials
        self.rois = rois
        self.presses = presses
        self.searches = searches

        # Pre-compute and store all features (now mapped to trial.idx)
        self.dispersions_global_bcea = self._extract_dispersions(type="BCEA")
        self.dispersions_global_rms = self._extract_dispersions(type="RMS")
        self.fixation_counts_per_trial = self._extract_fixation_counts_per_trial()
        self.symbol_counts_per_trial = self._extract_symbol_counts_per_trial()
        self.trial_in_search_durations = self._extract_trial_in_search_durations()
        self.time_in_relevant_fixations = self._extract_time_in_relevant_fixations()
        self.num_of_searches_per_trial = self._extract_num_of_searches_per_trial()
        self.revisits_per_trial = self._revisits_per_trial()
        
        # Unpacking tuple of dictionaries properly
        self.unique_counts_per_trial, self.percent_n_unique_symbols = self._extract_percent_N_unique_symbols_in_trial()
        self.learning_slopes_per_trial = self._extract_learning_slopes()
        

    
    def _calculate_bcea(self, positions, p=0.68):
        """ output units: pixels^2 """
        if len(positions) < 2:
            return 0
        x, y = zip(*positions)
        if len(x) < 2 or len(x) != len(y):
            return np.nan
                
        std_x = np.std(x, ddof=1)
        std_y = np.std(y, ddof=1)
        
        if std_x == 0 or std_y == 0:
            return 0.0
            
        rho, _ = pearsonr(x, y)
        k = -np.log(1 - p)
        return 2 * np.pi * k * std_x * std_y * np.sqrt(1 - rho**2)
    
    def _calculate_rms(self, positions):
        """ output units: pixels """
        if len(positions) == 0:
            return 0
        x, y = zip(*positions)
        mean_x, mean_y = np.mean(x), np.mean(y)
        return np.sqrt(np.mean([(px - mean_x)**2 + (py - mean_y)**2 for px, py in positions]))

    def _extract_dispersions(self, type="BCEA"):
        dispersions = {}
        for trial in self.trials:
            if trial.relevant_fixations:
                # Flatten the 'x' and 'y' coordinates from all relevant DataFrames into a single list of tuples
                positions = []
                for f in trial.relevant_fixations:
                    # Extract pairs and extend the list
                    coords = list(zip(f.microsaccades['x'], f.microsaccades['y']))
                    positions.extend(coords)
                
                if type == "BCEA":
                    dispersions[trial.idx] = self._calculate_bcea(positions)
                elif type == "RMS":
                    dispersions[trial.idx] = self._calculate_rms(positions)
                else:
                    raise ValueError(f"Unknown dispersion type: {type}")
            else:
                dispersions[trial.idx] = 0  
        return dispersions

    def _extract_fixation_counts_per_trial(self):
        return {trial.idx: (sum(len(search.fixations) for search in trial.searches) if trial.searches else 0) for trial in self.trials}

    def _extract_symbol_counts_per_trial(self):
        return {trial.idx: (sum(len(search.cleaned_sequence) for search in trial.searches) if trial.searches else 0) for trial in self.trials}
    
    def _extract_trial_in_search_durations(self):
        durations = {}
        for trial in self.trials:
            trial_duration = 0
            if trial.searches:
                for search in trial.searches:
                    trial_duration += search.duration()
            durations[trial.idx] = trial_duration
        return durations

    def _extract_time_in_relevant_fixations(self):
        time_in_rel_fix = {}
        for trial in self.trials:
            time_in_fixations = 0
            if trial.relevant_fixations:
                start = trial.relevant_fixations[0].start_time
                end = trial.relevant_fixations[-1].end_time
                time_in_fixations = end - start
            time_in_rel_fix[trial.idx] = time_in_fixations
        return time_in_rel_fix
        
    def _extract_num_of_searches_per_trial(self):
        return {trial.idx: len(trial.searches) if trial.searches is not None else 0 for trial in self.trials}
    

    def _extract_percent_N_unique_symbols_in_trial(self):
        unique_counts = {}
        for trial in self.trials:
            unique_symbols = set()
            if trial.searches:
                for search in trial.searches:
                    unique_symbols.update(search.cleaned_sequence)
            unique_counts[trial.idx] = len(unique_symbols)
        
        total_trials = len(self.trials)
        if total_trials == 0:
            return {}, {}
            
        percentages = {}
        counts_list = list(unique_counts.values())
        
        for num_unique_symbols in set(counts_list):
            percentages[num_unique_symbols] = (counts_list.count(num_unique_symbols) / total_trials) * 100
            
        return unique_counts, percentages


    def _extract_learning_slopes(self):
        #this function calculates the slope between the first 5 trials number of symbols in search to the last 5 number of symbols in search
        slopes = {}
        for trial in self.trials:
            if trial.searches and len(trial.searches) >= 10:
                first_half = trial.searches[:5]
                second_half = trial.searches[-5:]
                
                first_half_counts = [len(search.cleaned_sequence) for search in first_half]
                second_half_counts = [len(search.cleaned_sequence) for search in second_half]
                
                if len(first_half_counts) > 1 and len(second_half_counts) > 1:
                    slope, _ = pearsonr(range(5), first_half_counts)
                    slopes[trial.idx] = slope
                else:
                    slopes[trial.idx] = np.nan
            else:
                slopes[trial.idx] = np.nan

    def _revisits_per_trial(self):
        """
        Calculates value-based revisits per trial by resolving the 
        dictionary structure of cleaned_sequence.
        """
        revisits = {}
        
        for trial in self.trials:
            if hasattr(trial, 'searches') and trial.searches:
                flattened_values = []
                
                for search in trial.searches:
                    # CRITICAL FIX: Extract values from the dictionary instead of keys
                    if isinstance(search.cleaned_sequence, dict):
                        sequence_source = search.cleaned_sequence.values()
                    else:
                        sequence_source = search.cleaned_sequence
                    
                    for sym in sequence_source:
                        # Extract the primitive numeric value (1-9) from the Symbol object
                        val = sym.value if hasattr(sym, 'value') else sym
                        flattened_values.append(val)
                
                # Value-based calculation: Total items minus unique identities
                revisit_count = len(flattened_values) - len(set(flattened_values))
                revisits[trial.idx] = revisit_count
            else:
                revisits[trial.idx] = 0
                
        return revisits

    def _extract_inter_press_durations(self):
        durations = []
        for i in range(1, len(self.presses)):
            durations.append(self.presses[i].time - self.presses[i-1].time)
        return durations

    def _extract_search_to_press_gaps(self):
        gaps = []
        for search in self.searches:
            presses_inside = [p for p in self.presses if search.start_time <= p.time <= search.end_time]
            if presses_inside:
                continue
            next_presses = [p for p in self.presses if p.time > search.end_time]
            if next_presses:
                gaps.append(next_presses[0].time - search.end_time)
        return gaps

    def plot_iti_vs_search_time(self):
        """
        Plots Inter-Trial Interval vs. In-Search Time using the pre-computed dict attributes.
        """
        # Align data perfectly using trial.idx
        valid_trials = [t.idx for t in self.trials]
        
        itis = np.array([self.inter_trial_intervals.get(idx, np.nan) for idx in valid_trials])
        search_times = np.array([self.trial_in_search_durations.get(idx, np.nan) for idx in valid_trials])
        trial_indices = np.array(valid_trials)
        
        valid_mask = ~np.isnan(itis)
        x = itis[valid_mask] / 1e6
        y = search_times[valid_mask] / 1e6
        c = trial_indices[valid_mask]
        
        if len(x) < 2:
            print("Insufficient valid trials to compute regression.")
            return
            
        r, p = pearsonr(x, y)
        
        plt.figure(figsize=(9, 6))
        scatter = plt.scatter(x, y, c=c, cmap='viridis', alpha=0.8, edgecolor='k', s=50, zorder=2)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Trial Number', fontweight='bold')
        
        sns.regplot(x=x, y=y, scatter=False, color='darkred', line_kws={'zorder': 1})
        
        plt.title("Inter-Trial Interval vs. In-Search Time", fontweight='bold')
        plt.xlabel("Inter-Trial Interval (seconds)", fontweight='bold')
        plt.ylabel("In-Search Time (seconds)", fontweight='bold')
        
        stats_text = f"r = {r:.3f}\np < 0.001" if p < 0.001 else f"r = {r:.3f}\np = {p:.3f}"
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white", alpha=0.9), zorder=3)
        
        sns.despine()
        plt.tight_layout()
        plt.show()

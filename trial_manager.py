import pandas as pd
from FixationHandler import FixationHandler
from PanelMessages import MessageInfo, PanelMessages, Press
from PanelSymbols import PanelSymbols
from RoiFinder import RoiFinder
from SearchFinder import SearchFinder
from trial import Trial
from utils import prepare_image_and_gaze
from participant_gaze_data_manager import ParticipantGazeDataManager
from trial_features import TrialFeatures


class TrialManager:
    def __init__(self, subject_data: ParticipantGazeDataManager, panel: str, gaze_correction=None):
        # --- Data Loading ---
        panel_messages = PanelMessages(panel, subject_data)
        message_info: MessageInfo = panel_messages.message_info
        annotated_data = subject_data.annotate_gaze_events(panel, "threshold_based")
        img = subject_data.get_panel_img(panel)

        # --- Attributes ---
        self.img_resized, annotated_data = prepare_image_and_gaze(img, annotated_data)
        if gaze_correction is not None:
            annotated_data = gaze_correction(annotated_data)
        self.subject_name = subject_data.name
        self.panel = panel
        self.presses: list[Press] = message_info.presses
        self.symbols = PanelSymbols.get_panel_symbols(panel)
        self.rois = RoiFinder(panel, self.img_resized).rois
        self.fixations = FixationHandler(annotated_data).fixations
        self.searches = SearchFinder(all_fixations=self.fixations).searches
        
        # --- Core Processing ---
        self.trials = []
        self._extract_trials()
        self._add_search_sequance()
        
        # --- Component Initialization ---
        # 1. Feature Extraction (Scientific Logic)
        self.features = TrialFeatures(self.trials, self.rois, self.presses, self.searches)
        

    def _extract_trials(self):
        """
        Extract trials bounding the end of Trial N strictly by the exact start time 
        of the first fixation on Target N+1. No fallbacks are used.
        """
        start_idx = 2  
        end_idx = len(self.presses) - 2 
        
        # Initialize as None so the first trial finds its own physiological start
        current_trial_start = None

        for i, press in enumerate(self.presses[start_idx:end_idx], start=start_idx):
            symbol_index = press.idx + 18  
            triggering_symbol = next((s for s in self.symbols if s.idx == symbol_index), None)
            roi = next((r for r in self.rois if r.idx == symbol_index), None)

            if roi is None:
                raise ValueError(f"No ROI found with idx {triggering_symbol.idx}")            
            
            # Identify the NEXT ROI to establish the future boundary
            next_symbol_index = self.presses[i+1].idx + 18
            next_roi = next((r for r in self.rois if r.idx == next_symbol_index), None)

            # 1. Searches in Current Trial
            searches_in_trial = [
                s for s in self.searches 
                if s.start_time <= press.time <= s.end_time or 
                (press.time < s.start_time and s.end_time < self.presses[i+1].time)
            ]

            # 2. Fixations on Current Target ROI
            window_start = current_trial_start if current_trial_start is not None else press.time
            window_end = self.presses[i+1].time
            
            relevant_fixations_in_roi = [
                f for f in self.fixations
                if window_start <= f.start_time <= window_end and roi.contains(f.position, 2)
            ]

            # 3. Find the First Fixation on the NEXT ROI
            # Look ahead up to the press AFTER next to ensure we catch it even if Ido's RT was slow
            window_end_next = self.presses[i+2].time if (i + 2) < len(self.presses) else self.presses[-1].time
            
            next_roi_fixations = []
            if next_roi:
                next_roi_fixations = [
                    f for f in self.fixations
                    if window_start <= f.start_time <= window_end_next and next_roi.contains(f.position, 2)
                ]

            # 4. Determine Trial Start
            event_starts = []
            if searches_in_trial:
                event_starts.append(searches_in_trial[0].start_time)
            if relevant_fixations_in_roi:
                event_starts.append(relevant_fixations_in_roi[0].start_time)

            trial_start = min(event_starts) if event_starts and current_trial_start is None else current_trial_start

            # 5. Determine Trial End (Strict Cognitive Boundary)
            # No fallbacks. If they didn't fixate on the next ROI, the end time is undefined.
            trial_end = next_roi_fixations[0].start_time if next_roi_fixations else None

            # 6. Handle Missing Data and Validate Temporal Flow
            if trial_start is None or trial_end is None:
                print(f"Warning: Insufficient physiological data for Trial {press.idx + 1}. Assigning NaN bounds.")
                trial_start = float('nan')
                trial_end = float('nan')
                
                # CRITICAL: Reset the temporal chain using the system press so the next trial does not inherit NaN
                current_trial_start = self.presses[i+1].time
            else:
                # Failsafe: Prevent algorithmic noise from creating negative durations
                trial_end = max(trial_start, trial_end)
                
                # Pass the boundary forward to guarantee chronological continuity
                current_trial_start = trial_end

            # Create trial object
            trial = Trial(
                idx=press.idx + 1, 
                triggering_symbol=triggering_symbol,
                relevant_fixations=relevant_fixations_in_roi,
                searches=searches_in_trial,
                start_time=trial_start, 
                end_time=trial_end
            )
            self.trials.append(trial)
            
        return self.trials
    # def _extract_trials(self):
    #     """
    #     Extract trials based on physiological gaze boundaries.
    #     If a trial lacks physiological data, it is assigned NaN boundaries, 
    #     and the temporal chain is reset using the system keypress.
    #     """
    #     start_idx = 2  
    #     end_idx = len(self.presses) - 2 
        
    #     # Initialize as None so the first trial finds its own physiological start
    #     current_trial_start = None

    #     for i, press in enumerate(self.presses[start_idx:end_idx], start=start_idx):
    #         symbol_index = press.idx + 18  
    #         triggering_symbol = next((s for s in self.symbols if s.idx == symbol_index), None)
    #         roi = next((r for r in self.rois if r.idx == symbol_index), None)

    #         if roi is None:
    #             raise ValueError(f"No ROI found with idx {triggering_symbol.idx}")            
            
    #         # 1. Searches
    #         searches_in_trial = [
    #             s for s in self.searches 
    #             if s.start_time <= press.time <= s.end_time or 
    #             (press.time < s.start_time and s.end_time < self.presses[i+1].time)
    #         ]

    #         # 2. Fixations on Target ROI
    #         window_start = current_trial_start if current_trial_start is not None else press.time
    #         window_end = self.presses[i+1].time
            
    #         relevant_fixations_in_roi = [
    #             f for f in self.fixations
    #             if window_start <= f.start_time <= window_end and roi.contains(f.position, 2)
    #         ]

    #         triggering_fixation = relevant_fixations_in_roi[0] if relevant_fixations_in_roi else None

    #         # 3. Define Boundaries
    #         event_starts = []
    #         event_ends = []
            
    #         if searches_in_trial:
    #             event_starts.append(searches_in_trial[0].start_time)
    #             event_ends.append(searches_in_trial[-1].end_time)
    #         if relevant_fixations_in_roi:
    #             event_starts.append(relevant_fixations_in_roi[0].start_time)
    #             event_ends.append(relevant_fixations_in_roi[-1].end_time)

    #         if not event_starts or not event_ends:
    #             print(f"Warning: No physiological data found for Trial {press.idx + 1}. Assigning NaN bounds.")
    #             trial_start = float('nan')
    #             trial_end = float('nan')
                
    #             # CRITICAL: Reset the temporal chain using the system press so the next trial does not inherit NaN
    #             current_trial_start = self.presses[i+1].time
    #         else:
    #             # Start Time: Earliest physiological event for the first trial, otherwise contiguous from Trial N-1
    #             trial_start = min(event_starts) if current_trial_start is None else current_trial_start

    #             # End Time: Latest physiological interaction
    #             trial_end = max(event_ends)

    #             # Failsafe: Prevent algorithmic noise from creating negative durations
    #             trial_end = max(trial_start, trial_end)
                
    #             # Pass the physiological boundary forward to guarantee chronological continuity
    #             current_trial_start = trial_end

    #         # Create trial object
    #         trial = Trial(
    #             idx=press.idx + 1, 
    #             triggering_symbol=triggering_symbol,
    #             triggering_fixation=triggering_fixation,
    #             relevant_fixations=relevant_fixations_in_roi,
    #             searches=searches_in_trial,
    #             start_time=trial_start, 
    #             end_time=trial_end
    #         )
    #         self.trials.append(trial)
            
    #     return self.trials

    # def _extract_trials(self):
    #     """
    #     Extract trials based on presses and searches.
    #     Each trial starts with a press indicating the symbol of last trial solved.
    #     Returns a list of Trial objects.
    #     """

    #     start_idx = 2  #first 2 presses are a bit messy
    #     end_idx = len(self.presses) - 2 # last 2 presses are the ending and the task stop in the middle.
    #     # first press means the subject solved the first symbol (0th symbol) and last press is finishing with the panel
    #     for i, press in enumerate(self.presses[start_idx:end_idx], start=start_idx):
    #         symbol_index = press.idx + 18  # press 1 -> start of 2nd trial solving 2nd symbol -> symbol 19 (1 - based idx)
    #         triggering_symbol = next((s for s in self.symbols if s.idx == symbol_index), None)
    #         roi = next((r for r in self.rois if r.idx == symbol_index), None)

    #         if roi is None:
    #             raise ValueError(f"No ROI found with idx {triggering_symbol.idx}")            
            
    #         # Find the first search starting after this press
    #         searches_in_trial = [s for s in self.searches if s.start_time <= press.time <= s.end_time or (press.time < s.start_time and s.end_time < self.presses[i+1].time)]
    #         first_search_start = searches_in_trial[0].start_time if searches_in_trial else (press.time + self.presses[i+1].time)/2
    #         last_search_end = self.trials[-1].searches[-1].end_time if self.trials and self.trials[-1].searches else self.presses[i-1].time
    #         relevant_fixations = []
    #         if first_search_start and last_search_end:
    #             relevant_fixations = [
    #                 f for f in self.fixations
    #                 if last_search_end < f.start_time < first_search_start
    #             ]

    #         # Choose only relevant fixations in an elarged ROI of the symbol
    #         relevant_fixations_in_roi = []
    #         if relevant_fixations:
    #             relevant_fixations_in_roi = [
    #                 rf for rf in relevant_fixations
    #                 if roi.contains(rf.position, 2)
    #                 ]

    #         # coosing the last fixation before going out to search again
    #         if relevant_fixations_in_roi:
    #             triggering_fixation = relevant_fixations_in_roi[-1]
    #         else:
    #             triggering_fixation = None

    #         # trial start/end
    #         trial_start = searches_in_trial[0].start_time if searches_in_trial else press.time
    #         trial_end = searches_in_trial[-1].end_time if searches_in_trial else self.presses[i+1].time

    #         # Create trial object
    #         trial = Trial(
    #             idx=press.idx + 1, # trial i starts with press i - 1 to press i
    #             triggering_symbol=triggering_symbol,
    #             triggering_fixation= triggering_fixation,
    #             relevant_fixations=relevant_fixations_in_roi,
    #             searches=searches_in_trial,
    #             start_time=trial_start,
    #             end_time=trial_end
    #         )
    #         self.trials.append(trial)
        
            
    #     return self.trials

    def _add_search_sequance(self):
        """
        For each fixation in search, associate fixation with the most likely symbol.
        - First, try to match fixation center to an ROI.
        - If not found, check microsaccade coordinates within the fixation. roi with most microsaccade points wins.
        - If still not found, assign None.
        """
        rois_in_dict_area = [r for r in self.rois if r.idx < 18]

        for search in self.searches:
            fixation_symbol = {} # fixation:symbol

            for fixation in search.fixations:
                roi = next((r for r in rois_in_dict_area if r.contains(fixation.position, shape="square")), None)

                # If fixation center not in ROI, check microsaccades
                if roi is None:
                    roi_counts = {r: 0 for r in rois_in_dict_area}

                    for _, row in fixation.microsaccades.iterrows():
                        point = (row['x'], row['y'])
                        for r in rois_in_dict_area:
                            if r.contains(point, shape="square", factor = 1.5):
                                roi_counts[r] += 1

                    # Pick ROI with most microsaccades
                    roi = max(roi_counts, key=roi_counts.get)  #FIXME: what to do if there are more then one with max count?
                  
                    if roi_counts[roi] == 0:
                        roi = None  # no microsaccades in any ROI
                
                if roi:
                    fixation_symbol[fixation] = next((s for s in self.symbols if s.idx == roi.idx), None)
                else:
                    fixation_symbol[fixation] = None

            search.sequence = fixation_symbol

            # create cleaned sequance without repeats
            search.clean_sequence()

    # --- Convenience Accessors (Optional wrappers around Features) ---
    def get_trial_duration(self, trial_number: int) -> tuple:
        for trial in self.trials:
            if trial.idx == trial_number:
                return trial.get_trial_duration()
        raise ValueError(f"Trial number {trial_number} not found.")
    
    def add_trial_statistics_dataframe(self) -> pd.DataFrame:
        stats_records = []
        for i, trial in enumerate(self.trials):
            n_fixations = 0
            n_symbols = 0
            
            # Sum up stats from all searches in this trial
            if trial.searches:
                for s in trial.searches:
                    clean_seq = s.cleaned_sequence
                    if clean_seq:
                        n_symbols += len(clean_seq)
                    seq = s.sequence
                    if seq:
                        n_fixations += len(seq)

            stats_records.append({
                "Trial_Index": i,
                "Num_Fixations": n_fixations,
                "Num_Symbols": n_symbols,
                "Is_Zero_Fix": n_fixations == 0,
                "Is_One_Fix": n_fixations == 1,
                "Is_Two_Fix": n_fixations == 2,
                "Is_Zero_Symbol": n_symbols == 0,
                "Is_One_Symbol": n_symbols == 1,
                "Is_Two_Symbols": n_symbols == 2
            })
        
        trial_df = pd.DataFrame(stats_records)
        return trial_df

if __name__ == "__main__":
    main_path = "/Volumes/ramot/Noam_M/Results/Behavior"
    SUBJECT = "AG562" 
    PANEL = "l3"
    # PATH = "/Volumes/ramot/Noam_M/Results/Behavior"

    panels = ["0", "i1", "l4", "l3", "a5", "a3"]

    # 1. Init Data
    p_data = ParticipantGazeDataManager(SUBJECT, main_path, "SDMT", "pwMS")
    for panel in panels:
        try:
            print(f"--- Debugging {SUBJECT} on {panel} ---")

            tm = TrialManager(p_data, panel)
            
            tm.features.plot_iti_vs_search_time()
        except Exception as e:
            print(f"Error processing {SUBJECT} on {panel}: {e}")

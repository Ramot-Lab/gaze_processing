import os
import random
from matplotlib import pyplot as plt
import numpy as np
from FixationHandler import Fixation, FixationHandler
from PanelMessages import MessageInfo, PanelMessages, Press
from PanelSymbols import PanelSymbols, Symbol
from RoiFinder import RoiFinder
from SearchFinder import SearchFinder, Search
from constants import SECONDS_TO_MICROSECOND_FACTOR
from participant_gaze_data_manager import ParticipantGazeDataManager
from utils import prepare_image_and_gaze
import cv2


class Trial():
    def __init__(self, idx, triggering_symbol:Symbol, triggering_fixation:Fixation, relevant_fixations: list[Fixation], searches:list[Search], start_time, end_time):
        self.idx = idx
        self.triggering_symbol = triggering_symbol
        self.triggering_fixation = triggering_fixation
        self.relevant_fixations = relevant_fixations
        self.start_time = start_time
        self.end_time = end_time
        self.searches = searches

    def get_trial_duration(self):
        return (self.start_time, self.end_time)

class TrialManager():
    def __init__(self, subject_data: ParticipantGazeDataManager, panel, img):
        # --- Setup ---
        panel_messages = PanelMessages(panel, subject_data)
        message_info : MessageInfo = panel_messages.message_info
        annotated_data = subject_data.annotate_gaze_events('threshold_based', panel)
        self.img_resized, annotated_data = prepare_image_and_gaze(img, annotated_data)
        # --- Internal Use Only ---
        self.presses :list[Press] = message_info.presses  # list of Press objects with fields: idx, time
        self.symbols = PanelSymbols.get_panel_symbols(panel)
        self.rois = RoiFinder(panel, self.img_resized).rois
        self.fixations = FixationHandler(annotated_data).fixations
        self.searches = SearchFinder(all_fixations= self.fixations).searches
        #For external use
        self.trials = []  # list of Trial objects
        self._extract_trials()
        self._add_search_sequance()

    
    def _extract_trials(self):
        """
        Extract trials based on presses and searches.
        Each trial starts with a press indicating the symbol of last trial solved.
        Returns a list of Trial objects.
        """

        start_idx = 2  #first 2 presses are a bit messy
        end_idx = len(self.presses) - 2 # last 2 presses are the ending and the task stop in the middle.
        # first press means the subject solved the first symbol (0th symbol) and last press is finishing with the panel
        for i, press in enumerate(self.presses[start_idx:end_idx], start=start_idx):
            symbol_index = press.idx + 18  # press 1 -> start of 2nd trial solving 2nd symbol -> symbol 19 (1 - based idx)
            triggering_symbol = next((s for s in self.symbols if s.idx == symbol_index), None)
            roi = next((r for r in self.rois if r.idx == symbol_index), None)

            if roi is None:
                raise ValueError(f"No ROI found with idx {triggering_symbol.idx}")            
            
            # Find the first search starting after this press
            searches_in_trial = [s for s in self.searches if s.start_time <= press.time <= s.end_time or (press.time < s.start_time and s.end_time < self.presses[i+1].time)]
            first_search_start = searches_in_trial[0].start_time if searches_in_trial else (press.time + self.presses[i+1].time)/2
            last_search_end = self.trials[-1].searches[-1].end_time if self.trials and self.trials[-1].searches else self.presses[i-1].time
            relevant_fixations = []
            if first_search_start and last_search_end:
                relevant_fixations = [
                    f for f in self.fixations
                    if last_search_end < f.start_time < first_search_start
                ]

            # Choose only relevant fixations in an elarged ROI of the symbol
            relevant_fixations_in_roi = []
            if relevant_fixations:
                relevant_fixations_in_roi = [
                    rf for rf in relevant_fixations
                    if roi.contains(rf.position, roi.radius * 2)
                    ]

            # coosing the last fixation before going out to search again
            if relevant_fixations_in_roi:
                triggering_fixation = relevant_fixations_in_roi[-1]
            else:
                triggering_fixation = None

            # trial start/end
            trial_start = searches_in_trial[0].start_time if searches_in_trial else press.time
            trial_end = searches_in_trial[-1].end_time if searches_in_trial else self.presses[i+1].time

            # Create trial object
            trial = Trial(
                idx=press.idx + 1, # trial i starts with press i - 1 to press i
                triggering_symbol=triggering_symbol,
                triggering_fixation= triggering_fixation,
                relevant_fixations=relevant_fixations_in_roi,
                searches=searches_in_trial,
                start_time=trial_start,
                end_time=trial_end
            )
            self.trials.append(trial)
        
            
        return self.trials
    
    def get_trial_duration(self, trial_number: int) -> tuple:
        """
        Given a trial number (int),
        returns the start and end time of that trial as a tuple (start_time, end_time).
        """
        for trial in self.trials:
            if trial.idx == trial_number:
                return trial.get_trial_duration()
        raise ValueError(f"Trial number {trial_number} not found.")
    
    def get_inter_trial_times(self, trial_pairs: list[tuple[int, int]]) -> list[tuple]:
        """
        Given a list of trial number pairs [(t1, t2), ...],
        returns a list of tuples containing the end time of t1 and start time of t2 for each pair.

        for instance: if tiral_pairs = [(43, 46), ...] 
        i will get back [(end_time_trial_43, start_time_trial_46), ...]
        """
        times = []
        for t1, t2 in trial_pairs:
            end_t1 = self.trials[t1].end_time
            start_t2 = self.trials[t2].start_time
            times.append((end_t1, start_t2))
        return times
    
# #havent checked
    def trials_per_symbol(self, number: int) -> list[Trial]:
        """
        Given a number of the wanted trrigering symbol (1–9),
        returns the list of trials that where the triggered by the given symbol.
        """
        return [
            trial
            for trial in self.trials
            if trial.triggering_symbol and trial.triggering_symbol.value == number
        ]

    def compute_dists_and_angles(self):
        """
        Creates a list containing tuples of (distance, angle) of each triggering fixation to its symbol center.
        """
        distances = []
        angles = []
        for trial in self.trials:
            if trial.triggering_fixation and trial.triggering_symbol:
                roi = next((r for r in self.rois if r.idx == trial.triggering_symbol.idx), None)
                distance, angle = roi.dist_and_angle(trial.triggering_fixation.position)
                distances.append(distance)
                angles.append(angle)
            else:
                print(f"Trial {trial.idx}: Missing triggering fixation or symbol")
        return distances, angles


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
                            if r.contains(point, shape="square"):
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


# ---------------------------------------------------------------- #
#                    TRIALS ANALYSIS PLOTTING                      #
# ---------------------------------------------------------------- #

    def plot_searches_and_presses(self):
        """
        Plot a timeline of searches and presses in milliseconds at the bottom.
        Searches: horizontal red lines at y=1
        Presses: vertical dashed blue lines
        X-axis: show all relevant times (presses, search start/end) in ms
        """
        plt.figure(figsize=(12, 3))  # short height for bottom placement

        x_labels, x_positions = [], []

        # Plot searches as horizontal red lines at y=1
        for i, search in enumerate(self.searches):
            start = search.start_time / 1_000  # microseconds -> ms
            end = search.end_time / 1_000
            plt.plot([start, end], [1, 1], color='red', linewidth=2, 
                    label='Search' if i == 0 else "")
            # Add start and end times for x-axis labels
            x_labels.extend([f"{start:.0f}", f"{end:.0f}"])
            x_positions.extend([start, end])

        # Plot presses as vertical dashed blue lines
        for i, press in enumerate(self.presses):
            ts = press.time / 1_000  # microseconds -> ms
            plt.axvline(ts, color='blue', linestyle='--', linewidth=1, 
                        label='Press' if i == 0 else "")
            x_labels.append(f"{ts:.0f}")
            x_positions.append(ts)

        # Labels, title, and aesthetics
        plt.xlabel("Time (ms)")
        plt.yticks([])  # hide y-axis ticks
        plt.ylim(0, 1.5)  # compress y-axis near bottom
        plt.title("Searches and Presses Timeline (ms)")
        plt.grid(True, axis='x', linestyle=':', alpha=0.7)

        # Show legend and x-ticks
        plt.legend(loc='upper right')
        plt.xticks(x_positions, x_labels, rotation=90, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_search_sequences_on_image(self):
        """
        Plot all searches on the image with fixations.
        Each fixation is plotted as a dot, and the symbol index is shown in green nearby.
        """
        img_copy = self.img_resized.copy()  # make a copy to draw on

        for search in self.searches:
            for fix in search.fixations:
                fx, fy = map(int, fix.position)
                # Draw fixation as a small red dot
                cv2.circle(img_copy, (fx, fy), radius=5, color=(0, 0, 255), thickness=-1)  # Red in BGR
                
                text = str(search.idx)
                cv2.putText(
                    img_copy,
                    text,
                    (fx + 5, fy - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(17, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

        # Show the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img_copy)
        plt.axis('off')
        plt.title("Fixations and Symbol Sequence")
        plt.show()

    def duration_between_press_histogram(self):
        durations = []
        for i in range(1, len(self.presses)):
            duration = (self.presses[i].time - self.presses[i-1].time) / 1_000
            durations.append(duration)

        plt.figure(figsize=(8, 5))
        plt.hist(durations, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("Duration between presses (ms)")
        plt.ylabel("Count")
        plt.title("Histogram of Durations Between Presses")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        print(f"Computed {len(durations)} durations. Mean = {np.mean(durations):.2f}s, Median = {np.median(durations):.2f}s")

    def gap_between_start_search_and_closest_press_histogram(self):
        """
        For each search, find the closest press in time (before or after)
        and plot a histogram of the time gaps in milliseconds.
        """

        # Get arrays of times in microseconds
        press_times = np.array([p.time for p in self.presses])
        search_starts = np.array([s.start_time for s in self.searches[3:]])

        # Compute closest press for each search
        gaps = []
        for s_time in search_starts:
            closest_press_time = press_times[np.argmin(np.abs(press_times - s_time))]
            gaps.append(s_time - closest_press_time)

        # Convert to milliseconds
        gaps_ms = np.array(gaps) / 1_000

        # Plot histogram
        plt.figure(figsize=(8, 4))
        plt.hist(gaps_ms, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel("Time gap to closest press (ms)")
        plt.ylabel("Number of searches")
        plt.title("Histogram of closest press gaps per search")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_trials_and_searches(self):
        plt.figure(figsize=(12,5))
        x_labels, x_positions = [], []

        # Plot searches in red
        for search in self.searches:
            start, end, n_fixations = search.start_time, search.end_time, len(search.fixations)
            start, end = start/SECONDS_TO_MICROSECOND_FACTOR, end/SECONDS_TO_MICROSECOND_FACTOR
            plt.plot([start, end], [n_fixations, n_fixations], color='red', linewidth=2, label='Search' if search == self.searches[0] else "")
            x_labels.extend([f"{start:.2f}", f"{end:.2f}"])
            x_positions.extend([start, end])

        # Add presses as vertical dashed blue lines
        for press in self.presses:
            ts = press.time / SECONDS_TO_MICROSECOND_FACTOR
            plt.axvline(ts, color='blue', linestyle='--', linewidth=1, label='Press' if press.idx==0 else "")

        # Add trial start/end lines
        for trial in self.trials:
            st = trial.start_time / SECONDS_TO_MICROSECOND_FACTOR
            et = trial.end_time / SECONDS_TO_MICROSECOND_FACTOR
            plt.axvline(st, color='purple', linestyle='--', linewidth=1, label='Trial start' if trial.idx==0 else "")
            plt.axvline(et, color='orange', linestyle='--', linewidth=1, label='Trial end' if trial.idx==0 else "")

        plt.xlabel("Time (s)")
        plt.ylabel("Number of fixations in search")
        plt.title("Search fixations and trials per search")
        plt.grid(True)
        plt.xticks(x_positions, x_labels, rotation=90, ha='right')

        # Add legend
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def find_time_gap_search_press(self):

        max_diff = 0
        for search in self.searches[:3]:
            search_start = search.start_time
            search_end = search.end_time

            # Find presses inside this search
            relevant_presses = [p for p in self.presses if search_start <= p.time <= search_end]
            if not relevant_presses:
                continue

            # Compute time differences from search start
            diffs = [- search_start + p.time for p in relevant_presses]

            # Update max_diff
            max_diff = max(max_diff, max(diffs))
        return max_diff
    
    def plot_search_to_press_gaps(self):
        """
        Plot a histogram of time gaps between the end of each search and the next press,
        considering only searches that have no press inside their time span.

        Example timeline:
            press[i]  —  start(search[j]) ... end(search[j])  —  press[i+1]

        The gap is (press[i+1].time - search[j].end_time).
        """

        gaps = []
        for search in self.searches:
            # Check if any press occurs inside this search
            presses_inside = [
                p for p in self.presses if search.start_time <= p.time <= search.end_time
            ]
            if presses_inside:
                continue  # skip searches that already have a press inside

            # Find the next press after this search
            next_presses = [p for p in self.presses if p.time > search.end_time]
            if not next_presses:
                continue  # no press after this search (e.g., last one)
            next_press = next_presses[0]

            # Compute time gap in seconds
            gap = next_press.time - search.end_time
            gaps.append(gap/SECONDS_TO_MICROSECOND_FACTOR)

        if not gaps:
            print("No valid search-to-press gaps found.")
            return

        # --- Plot histogram ---
        plt.figure(figsize=(8, 5))
        plt.hist(gaps, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("Time gap (s)")
        plt.ylabel("Count")
        plt.title("Gap between search end and next press (no press inside search)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        print(f"Computed {len(gaps)} gaps. Mean = {np.mean(gaps):.2f}s, Median = {np.median(gaps):.2f}s")

    def plot_relevant_fixations_on_image(self): 
        """
        Plot all relevant fixations from all trials on the image.
        Relevant fixations are shown in different colors - dots connected by lines (within each trial).
        Triggering fixations are shown as red dots, labeled by their symbol index.
        """
        img_copy = self.img_resized.copy()

        for trial in self.trials:
            # Assign a random color for this trial (BGR format)
            color = tuple(random.randint(50, 255) for _ in range(3))

            # Get fixation positions as integer tuples
            points = [tuple(map(int, fix.position)) for fix in trial.relevant_fixations]

            # Draw fixation points
            for p in points:
                cv2.circle(img_copy, p, 5, color, -1)

            # Draw trajectory lines
            for p1, p2 in zip(points, points[1:]):
                cv2.line(img_copy, p1, p2, color, 2)

            # Draw triggering fixation (red)
            if trial.triggering_fixation:
                fx, fy = map(int, trial.triggering_fixation.position)
                cv2.circle(img_copy, (fx, fy), 7, (255, 0, 0), -1)
                cv2.putText(
                    img_copy,
                    str(trial.triggering_symbol.idx),
                    (fx + 10, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        plt.figure(figsize=(10, 10))
        plt.imshow(img_copy)
        plt.axis('off')
        plt.title("Relevant Fixations (diff color per trial) and Triggering Fixations (red)")
        plt.show()

    def plot_scatter_distance_angle(self):
        "plots a scatter plot of distances and angles of triggering fixations to symbol centers"
        distances, angles = self.compute_dists_and_angles()
        plt.figure(figsize=(8, 6))
        plt.scatter(angles, distances, color='blue', alpha=0.7)
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Distance to Symbol Center (pixels)")
        plt.title("Triggering Fixation Distances and Angles to Symbol Centers")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_distance_histogram(self):
        distances, _ = self.compute_dists_and_angles()
        plt.figure(figsize=(8, 5))
        plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("Distance to Symbol Center (pixels)")
        plt.ylabel("Count")
        plt.title("Histogram of Triggering Fixation Distances to Symbol Centers")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    def plot_angle_histogram(self):
        _, angles = self.compute_dists_and_angles()
        plt.figure(figsize=(8, 5))
        plt.hist(angles, bins=20, color='lightgreen', edgecolor='black')
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Count")
        plt.title("Histogram of Triggering Fixation Angles to Symbol Centers")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    


class TrialManagerDebugger:
    def __init__(self, trial_manager: TrialManager, output_path="./debug_videos"):
        # self.trial_manager = trial_manager
        self.img = trial_manager.img_resized
        self.fixations = trial_manager.fixations
        self.trials = trial_manager.trials
        # self.searches = trial_manager.searches
        self.rois = trial_manager.rois
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def make_debug_video(self, target_fps=60):
        """
        Create a gaze video showing gaze path (blue) and accumulating triggering fixations (red).
        """
        img_height, img_width, _ = self.img.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_file = os.path.join(self.output_path, "debug_trials.mp4")
        out = cv2.VideoWriter(output_file, fourcc, target_fps, (img_width, img_height))

        # --- Prepare gaze timeline ---
        gaze_times = np.array([f.start_time for f in self.fixations])
        gaze_positions = np.array([f.position for f in self.fixations])
        gaze_times_sec = gaze_times / 1_000_000  # microseconds → seconds
        duration = gaze_times_sec[-1]
        n_frames = int(duration * target_fps)
        frame_times = np.linspace(0, duration, n_frames)

        # --- Prepare persistent triggering fixations ---
        persistent_red = []  # [(x, y, roi_idx, search_idx)]

        # Build lookup of triggering fixations from trials
        triggering_fixations = []
        for trial in self.trials:
            if trial.triggering_fixation:
                fx, fy = map(int, trial.triggering_fixation.position)
                roi_idx = trial.triggering_symbol.idx if trial.triggering_symbol else -1
                search_idx = trial.searches[0].idx if trial.searches else -1
                triggering_fixations.append((trial.triggering_fixation.start_time / 1_000_000, fx, fy, roi_idx, search_idx))

        # --- Render video ---
        gaze_idx = 0
        for ft in frame_times:
            # progress gaze
            while gaze_idx + 1 < len(gaze_times_sec) and gaze_times_sec[gaze_idx + 1] <= ft:
                gaze_idx += 1

            fx, fy = map(int, gaze_positions[gaze_idx])
            frame = self.img.copy()

            # draw persistent red fixations that already occurred
            for (_, x, y, roi_idx, search_idx) in persistent_red:
                cv2.circle(frame, (x, y), 7, (255, 0, 0), -1)  # red dot
                cv2.putText(frame, f"R{roi_idx}/S{search_idx}", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

            # add new trigger fixations when time passes them
            new_triggers = [t for t in triggering_fixations if t[0] <= ft and t not in persistent_red]
            persistent_red.extend(new_triggers)

            # draw current gaze (blue)
            cv2.circle(frame, (fx, fy), 5, (0, 0, 255), -1)  # blue dot

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"✅ Debug video saved: {output_file}")

    def save_final_triggering_fixations_img(self):
        """
        Create and save an image showing only the accumulated triggering fixations
        (red dots with orange text labels), as in the final frame of the debug video.
        """
        # Start with a clean copy of the base image
        frame = self.img.copy()

        # Loop through all trials and draw the red dots + text
        for trial in self.trials:
            if trial.triggering_fixation is None:
                continue

            fx, fy = map(int, trial.triggering_fixation.position)
            roi_idx = trial.triggering_symbol.idx if trial.triggering_symbol else -1
            search_idx = trial.searches[0].idx if trial.searches else -1

            # Red dot (same as video)
            cv2.circle(frame, (fx, fy), 7, (255, 0, 0), -1)  # BGR = red

            # Bold orange label
            label = f"R{roi_idx}/S{search_idx}"
            cv2.putText(frame, label, (fx - 8, fy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        # Save the image
        output_file = os.path.join(self.output_path, "final_triggering_fixations_1.png")
        cv2.imwrite(output_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Final triggering fixation image saved: {output_file}")

    def none_in_search_symbols(self):
        """
        For each search, count how many times symbol 'None' appears in its fixation-symbol sequence.
        Return a list of counts per search.
        """
        counts = {}
        for trial in self.trials:
            for search in trial.searches:
                count_nun = sum(1 for symbol in search.sequence.values() if symbol == None)
                counts[search.idx] = count_nun
        return counts


if __name__ == "__main__":
    participant_data = ParticipantGazeDataManager("AG562", "/Volumes/ramot/Noam_M/Results/Behavior", "SDMT", "pwMS")
    panel = '0'
    img = plt.imread('/Volumes/ramot/Noam_M/Results/Behavior/panels_images/SDMT/combined_testable_0.jpg')

    trial_maneger = TrialManager(participant_data, panel, img)

    trial_maneger.plot_searches_and_presses()
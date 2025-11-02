import os
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


class Trail():
    def __init__(self, idx, triggering_symbol:Symbol, triggering_fixation:Fixation, searches:list[Search], start_time, end_time):
        self.idx = idx
        self.triggering_symbol = triggering_symbol
        self.triggering_fixation = triggering_fixation
        self.start_time = start_time
        self.end_time = end_time
        self.searches = searches

class TrailManager():
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
        self.trails = []  # list of Trail objects
        self._extract_trails()

    
    def _extract_trails(self):

        small_offset = self.find_time_gap_search_press() + 1_000  # 1 ms buffer
        start_idx = 2  
        end_idx = len(self.presses) - 1  
        # first press means the subject solved the first symbol (0th symbol) and last press is finishing with the panel
        for i, press in enumerate(self.presses[start_idx:end_idx], start=start_idx):
            if i >= len(self.presses) - 1 | press.idx < 2:
                break
        
            symbol_index = press.idx + 18 - 1  # press 1 -> symbol 18 (0-based idx)
            # print (f"Press idx: {press.idx}, symbol index: {symbol_index}, total symbols: {len(self.symbols)}, total rois: {len(self.rois)}, total presses: {len(self.presses)}")
            triggering_symbol = next((s for s in self.symbols if s.idx == symbol_index), None)
            roi = next((r for r in self.rois if r.idx == symbol_index), None)

            if roi is None:
                raise ValueError(f"No ROI found with idx {triggering_symbol.idx}")            
            
            # Find the first search starting after this press
            searches_in_trail = [s for s in self.searches if s.start_time >= press.time - small_offset and s.start_time < self.presses[i+1].time] # a little before press till next press
            first_search_start = searches_in_trail[0].start_time if searches_in_trail else None
            # Find triggering fixation in time window [press_ts - small_offset, first_search_start]
            if first_search_start:
                relevant_fixations = [
                    f for f in self.fixations
                    if (press.time - small_offset) <= f.start_time <= first_search_start
                ]
            # Choose fixation closest to the symbol/ROI position
            if relevant_fixations:
                # distance = Euclidean distance between fixation and symbol/ROI
                def fixation_distance(fix: Fixation):
                    rx, ry = roi.center
                    fx, fy = fix.position
                    return ((fx - rx)**2 + (fy - ry)**2)**0.5

                triggering_fixation = min(relevant_fixations, key=fixation_distance)
            else:
                triggering_fixation = None

            # Trail start/end
            trail_start = searches_in_trail[0].start_time if searches_in_trail else press.time
            trail_end = searches_in_trail[-1].end_time if searches_in_trail else self.presses[i+1].time

            # Create Trail object
            trail = Trail(
                idx=press.idx,
                triggering_symbol=triggering_symbol,
                triggering_fixation=triggering_fixation,
                searches=searches_in_trail,
                start_time=trail_start,
                end_time=trail_end
            )
            self.trails.append(trail)
        


        #     print(f"Trail {press.idx}: Symbol {triggering_symbol.idx} at {roi.center}, "
        #           f"Fixation at {triggering_fixation.position if triggering_fixation else None}, "
        #           f"Searches: {len(searches_in_trail)}, "
        #           f"Time: {trail_start}-{trail_end}")
        #     print("------------------------------------")
        # print("amount of trails with amount of searches=0:", len([t for t in self.trails if len(t.searches)==0]))
            
        return self.trails
    

    def trails_per_symbol(self, number: int) -> list[Trail]:
        """
        Given a number (1–9),
        returns the list of searches where the triggering symbol was the given symbol.
        """
        return [
            trail
            for trail in self.trails
            if trail.triggering_symbol and trail.triggering_symbol.value == number
        ]
        
    def plot_trails_and_searches(self):
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

        # Add trail start/end lines
        for trail in self.trails:
            st = trail.start_time / SECONDS_TO_MICROSECOND_FACTOR
            et = trail.end_time / SECONDS_TO_MICROSECOND_FACTOR
            plt.axvline(st, color='purple', linestyle='--', linewidth=1, label='Trail start' if trail.idx==0 else "")
            plt.axvline(et, color='orange', linestyle='--', linewidth=1, label='Trail end' if trail.idx==0 else "")

        plt.xlabel("Time (s)")
        plt.ylabel("Number of fixations in search")
        plt.title("Search fixations and trails per search")
        plt.grid(True)
        plt.xticks(x_positions, x_labels, rotation=90, ha='right')

        # Add legend
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def find_time_gap_search_press(self):

        max_diff = 0
        for search in self.searches:
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


class TrailManagerDebugger:
    def __init__(self, trail_manager: TrailManager, output_path="./debug_videos"):
        # self.trail_manager = trail_manager
        self.img = trail_manager.img_resized
        self.fixations = trail_manager.fixations
        self.trails = trail_manager.trails
        # self.searches = trail_manager.searches
        self.rois = trail_manager.rois
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def make_debug_video(self, target_fps=60):
        """
        Create a gaze video showing gaze path (blue) and accumulating triggering fixations (red).
        """
        img_height, img_width, _ = self.img.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_file = os.path.join(self.output_path, "debug_trails.mp4")
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

        # Build lookup of triggering fixations from trails
        triggering_fixations = []
        for trail in self.trails:
            if trail.triggering_fixation:
                fx, fy = map(int, trail.triggering_fixation.position)
                roi_idx = trail.triggering_symbol.idx if trail.triggering_symbol else -1
                search_idx = trail.searches[0].idx if trail.searches else -1
                triggering_fixations.append((trail.triggering_fixation.start_time / 1_000_000, fx, fy, roi_idx, search_idx))

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

        # Loop through all trails and draw the red dots + text
        for trail in self.trails:
            if trail.triggering_fixation is None:
                continue

            fx, fy = map(int, trail.triggering_fixation.position)
            roi_idx = trail.triggering_symbol.idx if trail.triggering_symbol else -1
            search_idx = trail.searches[0].idx if trail.searches else -1

            # Red dot (same as video)
            cv2.circle(frame, (fx, fy), 7, (255, 0, 0), -1)  # BGR = red

            # Bold orange label
            label = f"R{roi_idx}/S{search_idx}"
            cv2.putText(frame, label, (fx - 8, fy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        # Save the image
        output_file = os.path.join(self.output_path, "final_triggering_fixations.png")
        cv2.imwrite(output_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Final triggering fixation image saved: {output_file}")



participant_data = ParticipantGazeDataManager("AG562", "/Volumes/ramot/Noam_M/Results/Behavior", "SDMT", "pwMS")
panel = '0'
img = plt.imread('/Volumes/ramot/Noam_M/Results/Behavior/panels_images/SDMT/combined_testable_0.jpg')

trail_maneger = TrailManager(participant_data, panel, img)

debugger = TrailManagerDebugger(trail_maneger)

debugger.save_final_triggering_fixations_img()
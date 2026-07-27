import os
import random
import cv2

from matplotlib import pyplot as plt

from constants import SECONDS_TO_MICROSECOND_FACTOR
from trial_manager import TrialManager


class TrialVisualizer:
    """
    Responsible for all matplotlib visualizations based on TrialManager data.
    """
    def __init__(self, manager: TrialManager):
        self.manager = manager
        self.features = manager.features # Access the features class
        self.output_path = f"/Volumes/ramot/Noam_M/preliminary_results/{manager.subject_name}/{manager.panel}"
        os.makedirs(self.output_path, exist_ok=True)

    def plot_searches_and_presses(self):
        plt.figure(figsize=(12, 3))
        x_labels, x_positions = [], []

        # Plot searches
        for i, search in enumerate(self.manager.searches):
            start = search.start_time / 1_000
            end = search.end_time / 1_000
            plt.plot([start, end], [1, 1], color='red', linewidth=2, label='Search' if i == 0 else "")
            x_labels.extend([f"{start:.0f}", f"{end:.0f}"])
            x_positions.extend([start, end])

        # Plot presses
        for i, press in enumerate(self.manager.presses):
            ts = press.time / 1_000
            plt.axvline(ts, color='blue', linestyle='--', linewidth=1, label='Press' if i == 0 else "")
            x_labels.append(f"{ts:.0f}")
            x_positions.append(ts)

        plt.yticks([])
        plt.ylim(0, 1.5)
        plt.title("Searches and Presses Timeline (ms)")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_spatial_histograms(self):
        dists, angles = self.features.get_spatial_deviations()
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 5))
        
        # Distance Hist
        ax[0].hist(dists, bins=20, color='skyblue', edgecolor='black')
        ax[0].set_title("Triggering Fixation Distances")
        ax[0].set_xlabel("Pixels")
        
        # Angle Hist
        ax[1].hist(angles, bins=20, color='lightgreen', edgecolor='black')
        ax[1].set_title("Triggering Fixation Angles")
        ax[1].set_xlabel("Degrees")
        
        plt.tight_layout()
        plt.show()

    def plot_fixations_per_trial(self):
        counts = self.features.get_fixation_counts_per_trial()
        self._plot_per_trial_metric(counts, "Number of Fixations", "fixations_per_trial")

    def plot_symbols_per_trial(self):
        counts = self.features.get_symbol_counts_per_trial()
        self._plot_per_trial_metric(counts, "Number of Symbols", "symbols_per_trial")

    def _plot_per_trial_metric(self, data, y_label, filename_suffix):
        """Helper to plot metrics that are one-per-trial with time indicators."""
        midpoint_times = []
        for trial in self.manager.trials:
            midpoint = (trial.start_time + trial.end_time) / (2 * SECONDS_TO_MICROSECOND_FACTOR)
            midpoint_times.append(midpoint)

        plt.figure(figsize=(14, 6))
        plt.plot(midpoint_times, data, marker='o', linewidth=2)
        
        # Draw trial boundaries
        for trial in self.manager.trials:
            start = trial.start_time / SECONDS_TO_MICROSECOND_FACTOR
            end = trial.end_time / SECONDS_TO_MICROSECOND_FACTOR
            plt.axvline(start, linestyle='--', linewidth=0.8, color='blue')
            plt.axvline(end, linestyle='--', linewidth=0.8, color='red')
            
            # Label trial index
            plt.text((start+end)/2, max(data), str(trial.idx), ha='center', va='bottom', fontsize=8)

        plt.title(f"{y_label}; Subject {self.manager.subject_name}")
        plt.ylabel(y_label)
        plt.xlabel("Time (s)")
        
        output_file = os.path.join(self.output_path, f"{filename_suffix}_{self.manager.subject_name}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Saved: {output_file}")

    def plot_relevant_fixations_on_image(self): 
        """
        Plot all relevant fixations from all trials on the image.
        Relevant fixations are shown in different colors - dots connected by lines (within each trial).
        Triggering fixations are shown as red dots, labeled by their symbol index.
        """
        img_copy = self.manager.img_resized.copy()

        for trial in self.manager.trials:
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
            # #add relevant ROIs with twice radius of roi
            # for fix in trial.relevant_fixations:
            #     roi = next((r for r in self.manager.rois if r.contains(fix.position, factor = 2,  shape="circle")), None)
            #     if roi:
            #         cv2.circle(img_copy, roi.center, int(roi.radius*2), color, 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(img_copy)
        plt.axis('off')
        plt.title("Relevant Fixations (diff color per trial) and Triggering Fixations (red)")
        plt.show()


if __name__ == "__main__":
    from participant_gaze_data_manager import ParticipantGazeDataManager

    main_path = "/Volumes/ramot/Noam_M/Results/Behavior"
    SUBJECT = "NN111" 
    PANEL = "l3"
    PATH = "/Volumes/ramot/Noam_M/Results/Behavior"


    print(f"--- Debugging {SUBJECT} on {PANEL} ---")
    
    # 1. Init Data
    p_data = ParticipantGazeDataManager(SUBJECT, PATH, "SDMT", "HC")
    
    # 2. Init Manager
    tm = TrialManager(p_data, PANEL)
    
    # 3. Check Data
    tv = TrialVisualizer(tm)
    tv.plot_relevant_fixations_on_image()


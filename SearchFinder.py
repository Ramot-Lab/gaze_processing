from constants import *
from participant_gaze_data_manager import ParticipantGazeDataManager
from utils import prepare_image_and_gaze
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from RoiFinder import RoiFinder, ROI
from PanelSymbols import Symbol, PanelSymbols
from FixationHandler import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Search:
    def __init__(self,idx: int, start_time, end_time, fixations: list[Fixation]):
        self.idx = idx
        self.start_time = start_time
        self.end_time = end_time
        self.fixations = fixations  # list of fixation objects
        # self.fixation_before_search : Fixation | None = None

    def duration(self):
        return self.end_time - self.start_time

class SearchFinder:
    def __init__(self, all_fixations: list[Fixation]):
        self.all_fixations = all_fixations
        self.searches = self.find()


    def find(self, dict_ratio=0.175, text_ratio=0.22) -> list[Search]:
        H, W = SCREEN_SIZE
        dict_thresh = H * dict_ratio
        text_thresh = H * text_ratio

        searches = []
        in_search = False
        current_search_fixations = []
        last_fixation_before_search = None
        search_idx = 0

        for fixation in self.all_fixations:
            # Use mean y to decide if it's part of the search region
            y_mean = fixation.position[1]

            if not in_search and y_mean < dict_thresh:
                # Search starts
                in_search = True
                current_search_fixations = [fixation]
                search_idx += 1

            elif in_search:
                if y_mean < dict_thresh:
                    # Continue search
                    current_search_fixations.append(fixation)
                else:
                    # Search ends
                    search = Search(
                        idx=search_idx,
                        start_time=current_search_fixations[0].start_time,
                        end_time=current_search_fixations[-1].end_time,
                        fixations=current_search_fixations
                    )

                    searches.append(search)
                    in_search = False
                    current_search_fixations = []

        # Catch last search if it ends at the last fixation
        if in_search and current_search_fixations:
            search = Search(
                idx=search_idx,
                start_time=current_search_fixations[0].start_time,
                end_time=current_search_fixations[-1].end_time,
                fixations=current_search_fixations
            )

            searches.append(search)

        self.searches = searches
        return searches


        
    # def get_roi_sequences(self, searches: list[Search]):
    #     """
    #     Convert fixation data in searches into sequences of ROI objects.
    #     """

    #     roi_sequences = {}
    #     if not searches:
    #         raise ValueError("No searches provided to extract ROI sequences. - first run SearchFinder.find(img_shape)")
        
    #     for search in searches:
    #         roi_seq = []
    #         for fixation in search.fixations:
    #             roi = self._map_fixation_to_roi(fixation)
    #             if roi is not None:
    #                 roi_seq.append(roi)
    #         roi_sequences[search.idx] = roi_seq

    #     return roi_sequences
    
    # def _map_fixation_to_roi(self, fixation):
    #     """
    #     Assign a fixation to an ROI index (0..len(rois)-1).
    #     Falls back to nearest ROI if not inside any.
    #     """
    #     if self.rois:
    #          for roi in self.rois:
    #             if roi.contains(fixation):
    #                 return roi
        # # fallback: nearest ROI
        # nearest_roi = self.roi_finder.nearest_roi(fixation.position[0], fixation.position[1])
        # return 'nearest', nearest_roi
    


class SearchVisualizer():
    def __init__(self, searches: list[Search], dispersions=None):
        self.searches = searches
        if dispersions is None:
            dispersions = self._compute_fixation_dispersions()
        else:
            self.dispersions = dispersions

    def combine_video_with_graph(self, video_path, output_path):
        output_path = os.path.join(output_path, f"{p_name}_{panel}_w_curser.mp4")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(8, 3))
        times_and_fixations = [
            (s.start_time / SECONDS_TO_MICROSECOND_FACTOR,
            s.end_time / SECONDS_TO_MICROSECOND_FACTOR,
            len(s.fixations))
            for s in self.searches
        ]

        for (start, end, n_fixations) in times_and_fixations:
            ax.plot([start, end], [n_fixations, n_fixations], color='blue', linewidth=2)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Number of fixations in search")
        ax.set_title("Fixations per Search")
        ax.grid(True)

        vline = ax.axvline(x=0, color='red', linestyle='--')
        canvas = FigureCanvas(fig)
        fig.tight_layout()

        # Prepare video writer
        graph_width, graph_height = 800, 300
        combined_width = max(width, graph_width)
        combined_height = height + graph_height

        time_to_first = self.time_to_first_fixation()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

        for frame_idx in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            vline.set_xdata([current_time])

            # Render the graph
            canvas.draw()
            graph_img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
            graph_img = graph_img[:, :, :3]
            graph_img = cv2.resize(graph_img, (combined_width, graph_height))

            # Create combined frame
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            combined_frame[:height, :width] = frame
            combined_frame[height:, :] = graph_img

            # ✅ Draw text directly on the combined frame (not just frame!)
            overlay_text = f"Time to 1st fixation: {time_to_first:.2f}s"
            cv2.putText(combined_frame, overlay_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

            out.write(combined_frame)

        cap.release()
        out.release()
        plt.close(fig)
        print(f"✅ Combined video saved to {output_path}")

    def _compute_fixation_dispersions(self):
        """
        Compute dispersion of gaze points within each fixation as RMS distance 
        from the mean (centroid) of all microfixations in that fixation.

        Args:
            searches (list): output of self.find()

        Returns:
            dispersions (list): RMS dispersion (pixels) per fixation
        """
        dispersions = []

        for search in self.searches:
            for fixation in search.fixations:
                points = np.array(fixation.microsaccades)  # shape (N, 3)
                if points.shape[0] < 2:
                    continue  # skip single-point fixations, dispersion=0

                x_vals = points[:,1]
                y_vals = points[:,2]
                # compute centroid
                centroid_x = x_vals.mean()
                centroid_y = y_vals.mean()
                # Euclidean distance of each point from centroid
                distances = np.sqrt((x_vals - centroid_x)**2 + (y_vals - centroid_y)**2)
                # RMS distance
                dispersion = np.sqrt(np.mean(distances**2))
                dispersions.append(dispersion)
        self.dispersions = dispersions
        return dispersions

    def _plot_fixations_per_search(self, p_name: str, panel: str, save_dir: str):

        plt.figure(figsize=(12,5))
        x_labels, x_positions = [], []
        for search in self.searches:
            start, end, n_fixations = search.start_time, search.end_time, len(search.fixations)
            start, end = start/SECONDS_TO_MICROSECOND_FACTOR, end/SECONDS_TO_MICROSECOND_FACTOR
            plt.plot([start, end], [n_fixations, n_fixations], color='blue', linewidth=2)
            x_labels.extend([f"{start:.2f}", f"{end:.2f}"])
            x_positions.extend([start, end])

        plt.xlabel("Time (s)")
        plt.ylabel("Number of fixations in search")
        plt.title(f"Search fixations per search - Subject {p_name}, Panel {panel}")
        plt.grid(True)
        plt.xticks(x_positions, x_labels, rotation=90, ha='right')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{p_name}_{panel}_fixations_per_search.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_dispersion_histogram(self, p_name: str, panel: str, save_dir: str):
        plt.figure(figsize=(8,5))
        plt.hist(self.dispersions, bins=20, color='skyblue', edgecolor='k')
        plt.xlabel("Gaze dispersion in single fixations (pixels)")
        plt.ylabel("Count")
        plt.title(f"Fixation dispersions - Subject {p_name}, Panel {panel}")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{p_name}_{panel}_dispersion_hist.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_search_durations(self, p_name: str, panel: str, save_dir: str):
        durations = [(s.end_time - s.start_time) / SECONDS_TO_MICROSECOND_FACTOR for s in self.searches]

        plt.figure(figsize=(10,5))
        plt.plot(range(1, len(durations)+1), durations, linestyle="-", linewidth=2, color="royalblue")
        plt.xlabel("Search Index")
        plt.ylabel("Duration (s)")
        plt.title(f"Search Durations - {p_name}, Panel {panel}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{p_name}_{panel}_search_durations.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def save_all_plots(self, p_name: str, panel: str, output_path: str):
        """
        Generate and save all plots for a subject + panel into a dedicated folder.
        """
        # create folder for patient inside output_path
        patient_folder = os.path.join(output_path, p_name)
        os.makedirs(patient_folder, exist_ok=True)

        # 1. Fixations per search
        self._plot_fixations_per_search(p_name, panel, patient_folder)

        # 2. Dispersion histogram
        self._plot_dispersion_histogram(p_name, panel, patient_folder)

        # 3. Search durations
        self._plot_search_durations(p_name, panel, patient_folder)

        print(f"✅ All plots saved in {patient_folder}")
    #FIXME: image proportion!
    def see_searches_on_image(self, img, idxs_searches_to_show: list[int] = []):
        """
        Overlay search paths on the original image.

        Args:
            img (ndarray): Original panel image.
            idxs_searches_to_show (list): Indices of searches to visualize. If None, show all.
        """
        if idxs_searches_to_show == []:
            idxs_searches_to_show = list(range(10))  # Default to the first 10 searches

        searches_to_show = [self.searches[i] for i in idxs_searches_to_show]
        colors = plt.colormaps['tab20'](range(len(searches_to_show)))
        plt.figure(figsize=(8, 8))

        # Show the panel image with fixed proportions
        plt.imshow(img)#, aspect="equal")
        # plt.axis("off")

        for idx, search in enumerate(searches_to_show):
            # Get the fixation data for the search
            fixation_to_plt = [search.fixation_before_search] + search.fixations if search.fixation_before_search else search.fixations
            x_positions = [fix.position[0] for fix in fixation_to_plt]
            y_positions = [fix.position[1] for fix in fixation_to_plt]

            # Plot arrows between consecutive fixations
            for i in range(len(fixation_to_plt) - 1):
                plt.arrow(
                    x_positions[i], y_positions[i],
                    x_positions[i+1] -x_positions[i], y_positions[i+1] - y_positions[i],
                    color=colors[idx],
                    head_width=5, head_length=10, linewidth=1.5, alpha=0.8
                )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


    def time_to_first_fixation(self):
        """Compute time from table appearance (t=0) to first fixation on first symbol."""
        if not self.searches or not self.searches[0].fixations:
            return None  # no searches or no fixations available
        
        first_fixation = self.searches[0].fixations[0]
        time_to_first = first_fixation.start_time / SECONDS_TO_MICROSECOND_FACTOR
        return time_to_first



def draw_roi_with_symbol(image, rois, symbols, center_color=(0, 255, 0), circle_color=(255, 0, 0), text_color=(0, 0, 255)):        
    """
    Draw ROIs with center, circle, and matching symbol value.

    Parameters:
        image (np.ndarray): The image to draw on.
        rois (list[ROI]): List of ROI objects (with .center and .radius).
        symbols (list[Symbol]): List of Symbol objects (with .center and .value).
        color (tuple): BGR color for center dot and text.
        radius_color (tuple): BGR color for ROI circle.
    """
    img_copy = image.copy()
    if len(rois) != len(symbols):
        raise ValueError("Number of ROIs and symbols must match")

    for roi in rois:
        cx, cy = roi.center

        # Draw ROI center
        cv2.circle(img_copy, (int(cx), int(cy)), 4, center_color, -1)

        # Draw ROI circle
        cv2.circle(img_copy, (int(cx), int(cy)), int(roi.radius), circle_color, 2)

    # If there is a symbol for this ROI, write it
        sym = symbols[roi.idx]
        type_of_s = "K" if sym.type == "key" else "N" if sym.type == "number" else "S"
        label = f"{sym.value}_{type_of_s}"
        cv2.putText(img_copy,
                    label,
                    (int(cx) + 5, int(cy) - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    2)

        # Convert from BGR to RGB for correct display in matplotlib
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()    




if __name__=="__main__":
    p_name = "AG562"
    task = "SDMT"
    group = "pwMS"  # "HC" or "pwMS"
    panel = "0"
    panel_path = f'/Volumes/ramot/rotation_students/Noam_M/Results/Behavior/panels_images/SDMT/combined_testable_{panel}.jpg'
    data_path = "/Volumes/ramot/rotation_students/Noam_M/Results/Behavior"
    output_path = "/Volumes/ramot/rotation_students/Noam_M/visualized data/test_videos/"
    video_path = '/Volumes/ramot/rotation_students/Noam_M/visualized data/test_videos/AG562/participant_task_0_AG562.mp4'
    img = plt.imread(panel_path)
    
    # for p_name in p_names:
    subject_data = ParticipantGazeDataManager(p_name, data_path, "SDMT", group)

    from PanelMessages import PanelMessages
    panel_messages = PanelMessages(panel, subject_data)

    res = subject_data.annotate_gaze_events('threshold_based', panel)        
    img_resized, res = prepare_image_and_gaze(img, res)
    #all_fixations, rois, symbols, searches
    symbols = PanelSymbols.get_panel_symbols(panel)
    roifinder = RoiFinder(panel, img_resized)
    fixationhandler = FixationHandler(data = res)

    search_finder = SearchFinder(all_fixations= fixationhandler.fixations)
    search_visualizer = SearchVisualizer(search_finder.searches)
    # search_visualizer.combine_video_with_graph(video_path, output_path)
    search_visualizer.see_searches_on_image(img_resized, idxs_searches_to_show=[2, 3])
    # draw_roi_with_symbol(img_resized,roifinder.rois, symbols)











# now i want to upgrade my logic of searchFinder. i want it to have a triggering symbol logic.
# search usually starts with a fixation on the correct symbol in the text (this is determined by: 1. if there was a fixation on the symbol that its index is the same as press index from the message of panel = "panel_name" fron there we take messages_info.presses - the time stamp of the correct idx is 




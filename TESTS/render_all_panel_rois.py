"""
Sanity-check rendering of every panel's ROI boxes, straight from the live RoiFinder/ROI
code (2026-09-17 standardization: single elongation ratio, all-square boxes, exact
row/column tiling) - not a re-derivation in a one-off script. Uses RoiFinder.visualize_rois,
which draws exactly the boxes self.rois holds, so what's rendered is guaranteed to match
what ROI.contains() actually checks.

One representative participant (CM906, confirmed to have valid Stage-1 data for all 6
panels) supplies the resized panel image for each panel - panel images themselves are
shared across participants (main_data_path/panels_images/SDMT/*.jpg), only the resizing
step needs a real annotated_data sample to run through prepare_image_and_gaze.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from participant_gaze_data_manager import ParticipantGazeDataManager
from RoiFinder import RoiFinder
from utils import prepare_image_and_gaze
import pipeline_config

PARTICIPANT = "CM906"
GROUP = "pwMS"
PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]
OUT_DIR = "/Volumes/ramot/Noam_M/after_analysis_quality_checks_17_09_26/roi_renders"


def run():
    data_path = pipeline_config.main_data_path()
    subject_dir = os.path.join(data_path, GROUP, PARTICIPANT)
    sd = ParticipantGazeDataManager(subject_dir, data_path, "SDMT", GROUP)
    os.makedirs(OUT_DIR, exist_ok=True)

    for panel in PANELS:
        img = sd.get_panel_img(panel)
        annotated_data = pipeline_config.load_annotated_csv("threshold_based", GROUP, PARTICIPANT, panel)
        img_resized, _ = prepare_image_and_gaze(img, annotated_data)

        roi_finder = RoiFinder(panel, img_resized)
        n_rows = len(roi_finder.rows)
        row_sizes = [len(r) for r in roi_finder.rows]
        print(f"panel {panel}: {len(roi_finder.rois)} ROIs, {n_rows} rows, sizes={row_sizes}")

        save_path = os.path.join(OUT_DIR, f"roi_render_panel_{panel}.png")
        roi_finder.visualize_rois(save_path=save_path)


if __name__ == "__main__":
    run()

"""
Renders a 60fps gaze-overlay video for one participant/panel, using the SAME raw
(pre-interpolation, out-of-range-NaN'd) x/y data the panel diagnostics
(build_panel_diagnostics.py) are computed from - NOT matched_data[panel][KEY_TOBII_DATA],
which is outlier-smoothed + NaN-interpolated by ParticipantGazeDataManager.clean_outliers()
and would hide the very off-screen excursions this is meant to show.

The canvas is padded around the panel image so off-screen points (negative y, or beyond any
edge) are actually visible instead of being drawn outside the frame and clipped away.
Fixation/saccade coloring (evt) and per-frame timing come from the normal
ParticipantGazeDataManager.annotate_gaze_events() pipeline, which shares the same sample
ordering/length as the raw extraction (clean_outliers only replaces values, never drops
rows) - only x/y position is swapped out for the raw, uninterpolated version.
"""
import glob
import os

import cv2
import numpy as np

from calibration_drift_qa import _panel_image_path
from constants import (
    FIXATION_CSV_KEY_FIXATION, TIME_STAMP,
    SECONDS_TO_MICROSECOND_FACTOR, SACCADE_IDX, VIDEO_CODEC, FIXATION_COLOR,
    SACCADE_COLOR, RADIUS, THICKNESS, SCREEN_SIZE,
)
from participant_gaze_data_manager import ParticipantGazeDataManager, _apply_out_of_range_policy

MAIN_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/calibration_qc/videos"


def _raw_panel_xy(sd, panel, group):
    """Re-extracts this panel's x/y directly off the raw .mat data, for whichever eye
    matched_data says was actually used - same pipeline (out-of-range policy, eye-selection)
    as build_panel_diagnostics.py, but without clean_outliers' jump-smoothing/interpolation."""
    from constants import KEY_ANALYSIS_EYE
    eye = sd.matched_data[panel][KEY_ANALYSIS_EYE]

    # NOTE: sd.group is only correctly populated when ParticipantGazeDataManager is
    # constructed with a full directory path as participant_name (the convention used by
    # every sweep in this codebase) - here it's constructed with a bare name, so sd.group
    # would be '' ; use the explicit group argument instead.
    task_dir = os.path.join(sd.main_data_path, group, sd.name, sd.task)
    mat_files = glob.glob(os.path.join(task_dir, "*.mat"))
    mat_files = [f for f in mat_files if "run" in os.path.split(f)[1].lower()]
    loaded_mats, _ = sd._load_and_dedupe_run_files(mat_files)

    target_date = sd.matched_data[panel]["recording_date"]
    for mat in loaded_mats:
        if sd.get_creation_time(mat) != target_date:
            continue
        messages = mat["messages"]
        left_gaze = mat["data"].gaze.left.gazePoint.onDisplayArea
        right_gaze = mat["data"].gaze.right.gazePoint.onDisplayArea
        tobi_ts = mat["data"].gaze.systemTimeStamp
        left_gaze, right_gaze = _apply_out_of_range_policy(left_gaze, right_gaze)
        gaze = right_gaze if eye == "r" else left_gaze

        panel_indices, break_indices = sd.break_mat_into_pannels(messages)
        panel_start_times = [messages[i][0] for i in panel_indices]
        break_start_times = [messages[i][0] for i in break_indices]

        task_data = mat["task_data"].__dict__
        codes = [(name[-2:]).replace("_", "").lower() for name in list(task_data.keys())[1::2]]
        for i in range(3):
            if i >= len(codes) or codes[i] != panel:
                continue
            indices = np.where((tobi_ts > panel_start_times[i]) & (tobi_ts < break_start_times[i]))[0]
            return gaze[0, indices], gaze[1, indices], eye

    raise ValueError(f"Could not find raw data for {sd.name}/{panel} matching recording_date {target_date}")


def render_padded_gaze_video(participant, panel, group, main_data_path=MAIN_DATA_PATH,
                              output_dir=OUTPUT_DIR, target_fps=60,
                              pad_top=250, pad_bottom=50, pad_left=300, pad_right=300):
    sd = ParticipantGazeDataManager(participant, main_data_path, "SDMT", group)
    if panel not in sd.matched_data:
        raise ValueError(f"Panel {panel!r} not in matched_data for {participant} "
                          f"(available: {list(sd.matched_data.keys())})")

    annotated = sd.annotate_gaze_events(panel)  # evt + t labels, standard pipeline
    raw_x, raw_y, eye = _raw_panel_xy(sd, panel, group)
    if len(raw_x) != len(annotated):
        raise ValueError(f"Raw extraction ({len(raw_x)} samples) doesn't match "
                          f"annotate_gaze_events ({len(annotated)} samples) - alignment assumption broke")

    fixation = annotated[FIXATION_CSV_KEY_FIXATION].to_numpy()
    times = annotated[TIME_STAMP].to_numpy() / SECONDS_TO_MICROSECOND_FACTOR

    img = cv2.imread(_panel_image_path(main_data_path, "SDMT", panel))
    img_height_orig, img_width_orig = img.shape[:2]
    screen_height, screen_width = SCREEN_SIZE
    scale = screen_height / img_height_orig
    new_width = int(img_width_orig * scale)
    resized_img = cv2.resize(img, (new_width, screen_height))
    x_offset = (screen_width - new_width) / 2

    # Raw (unclipped) pixel positions - can legitimately be negative or beyond the image
    # edges, since raw_x/raw_y are normalized [0,1]-ish values before any canvas clipping.
    pixel_x = raw_x * screen_width - x_offset
    pixel_y = raw_y * screen_height

    canvas_width = screen_width + pad_left + pad_right
    canvas_height = screen_height + pad_top + pad_bottom
    canvas_x = pixel_x + pad_left
    canvas_y = pixel_y + pad_top

    base_canvas = np.full((canvas_height, canvas_width, 3), 40, dtype=np.uint8)  # dark gray padding
    base_canvas[pad_top:pad_top + screen_height, pad_left:pad_left + new_width] = resized_img
    # Mark the true screen boundary so it's clear what's "on-screen" vs padding.
    cv2.rectangle(base_canvas, (pad_left, pad_top), (pad_left + new_width - 1, pad_top + screen_height - 1),
                  color=(0, 255, 255), thickness=2)

    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{participant}_{panel}_{eye}_raw_padded.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), target_fps, (canvas_width, canvas_height))

    duration = times[-1]
    n_frames = max(1, int(duration * target_fps))
    gaze_idx = 0
    for frame_time in np.linspace(0, duration, n_frames):
        while gaze_idx + 1 < len(times) and times[gaze_idx + 1] <= frame_time:
            gaze_idx += 1

        frame = base_canvas.copy()
        x, y = canvas_x[gaze_idx], canvas_y[gaze_idx]
        if not (np.isnan(x) or np.isnan(y)):
            color = SACCADE_COLOR if fixation[gaze_idx] == SACCADE_IDX else FIXATION_COLOR
            cv2.circle(frame, (int(x), int(y)), radius=RADIUS, color=color, thickness=THICKNESS)
        writer.write(frame)

    writer.release()
    print(f"Saved {video_path} ({n_frames} frames @ {target_fps}fps, eye={eye}, "
          f"pct_negative_y={100*np.mean(raw_y < 0):.2f}%)")
    return video_path


if __name__ == "__main__":
    render_padded_gaze_video("GS739", "l4", "HC")

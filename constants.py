import math
import os
from enum import Enum

KEY_TOBII_DATA = "tobii_data"  # Key for Tobii eye movement data
KEY_TASK_PANEL_IMG = "task_panel_img"  # Key for the image of the task panel
KEY_PANEL_MESSAGES = "messages"  # Key for messages related to the task
KEY_CALIBRATION_INFO = "calibration_info"  # Key for the parsed calibration/validation quality report
KEY_EYE_SELECTION_REASON = "eye_selection_reason"  # Key for why this PANEL's analysis eye was chosen (best_calibrated/only_qualifying_eye/explicit_override) - see participant_gaze_data_manager.prepare_gaze_data_for_preprocessing
KEY_ANALYSIS_EYE = "analysis_eye"  # Key for which eye ("l"/"r") was actually used for this panel (can differ between panels of the same calibration event, since NaN ratio is per-panel)
KEY_AUDIO_DATA = "audio_data"  # Key for the audio data file path
KEY_STRIKE_SCORE = "strike_score"
KEY_REACTION_TIMES = "reaction_times"  # Key for the raw per-press reaction-time deltas (seconds), fallback for panels with no "press N" messages
KEY_RECORDING_DATE = "recording_date"
KEY_INTERPOLATED_MASK = "interpolated_mask"  # per-sample bool: missing / out-of-range / outlier sample filled by np.interp
TIME_STAMP = "t"
SIGNAL_IDX = "signal_index"
SENTENCE_BREAK = "sentence_breaks"
MAX_VALID_NAN_VALUES = 0.1 #10% of nan values in the eye tracking data is still valid

# --- Out-of-[0,1]-range gaze sample handling (2026-09-07 decision) ---------------------
# How to treat raw x/y gaze coordinates outside the normalized [0,1] screen area.
#   "as_is"              - leave every value exactly as recorded
#   "all_NaN"             - convert every out-of-range sample (on either side, <0 or >1) to
#                           NaN
#   "only_extreme_values" - convert to NaN only outside OUT_OF_RANGE_X_BOUNDS /
#                           OUT_OF_RANGE_Y_BOUNDS (normalized screen units); within bounds,
#                           left as-is
OUT_OF_RANGE_VALUES_METHOD = "only_extreme_values"

# Decided from the empirical out-of-range distributions (calibration_qc/hist_x_out_of_range_only.png,
# hist_y_out_of_range_only.png): X's out-of-range mass sits well clear of either edge on both
# sides (looks like real tracking failure, not noise), so the same 0.15 margin is tolerated on
# both sides; Y's out-of-range mass concentrates tightly just past the TOP edge (y<0, plausible
# noise) but its y>1 (bottom-edge overshoot) population did not show that same tight
# near-edge clustering, so no margin is tolerated there - any y>1 is treated as invalid.
# NOTE: as literally written, "-0.15<x<0.15" would exclude nearly all real on-screen data
# (which sits around x=0.5) - read as a 0.15 margin on EACH of x's own two edges (0 and 1),
# mirroring how the y margin was specified. Flag/correct if that reading is wrong.
OUT_OF_RANGE_X_BOUNDS = (0.0, 1.0)  # valid x range is exactly the panel/screen width, no margin either side
OUT_OF_RANGE_Y_BOUNDS = (-0.15, 1.0)   # (keep down to 0.15 past the top edge; NO margin past the bottom edge)


OLD_MIC_REPLACEMENT_DATE = "2024-07-01"

VIDEO_CODEC = 'mp4v'   # Codec for .mp4
FPS = 600              # Frames per second for video output
RADIUS = 6             # Radius for eye position marker
SACCADE_COLOR = (0, 0, 255)  # Color for eye marker (red in BGR format)
FIXATION_COLOR  = (255, 0, 0)  # Blue in OpenCV (BGR)
THICKNESS = 3          # Thickness of the circle around the eye position

# Heatmap settings
GAUSSIAN_SIGMA = 3  # Standard deviation for Gaussian filter
HEATMAP_ALPHA = 0.5  # Transparency for the heatmap overlay
HEATMAP_COLOR_MAP = 'jet'  # Color map for the heatmap

# Animation settings
MARKER_SIZE = 5  # Marker size for eye position in animation
ANIMATION_INTERVAL = 2  # Interval between frames in milliseconds
BLIT = True  # Blit flag for animation optimization

# Gaze heatmap movie settings
DECAY_FACTOR = 0.99  # Heatmap decay factor for gaze heatmap movie
OUTPUT_FPS = 600  # Frames per second for gaze heatmap movie
TOTAL_TIME = 90  # Total duration of the gaze movie in seconds
TOTAL_FRAMES = TOTAL_TIME * OUTPUT_FPS  # Total frames in the gaze movie
HEATMAP_ALPHA_MOVIE = 0.3  # Heatmap transparency for gaze movie
SECONDS_TO_MICROSECOND_FACTOR = 1_000_000  # Conversion factor from seconds to microseconds

# Audio settings
AUDIO_SAMPLE_RATE = 44100  # Default sample rate for audio

# Saccade finder related keys
KEY_SACCADE_START = "saccade_start"  # Key for saccade start indices
KEY_SACCADE_END = "saccade_end"  # Key for saccade end indices
KEY_FIXATION_START = "fixation_start"  # Key for fixation start indices
KEY_FIXATION_END = "fixation_end"  # Key for fixation end indices
KEY_VELOCITY = "velocity"  # Key for eye movement velocity data
KEY_AMPLITUDE = "amplitude"  # Key for saccade amplitude data
KEY_DURATION = "duration"  # Key for saccade duration data
KEY_PEAK_VELOCITY = "peak_velocity"  # Key for saccade peak velocity data

KEY_ANNOTATION_MODEL_BASE = 'threshold_based' # or ״model_based״, or "pymovements_based"
# String constants for messages and logs
VIDEO_SAVED_MESSAGE = "Video saved without audio at: {video_output_path}"
GAZE_MOVIE_PROGRESS = 'Processing data point {i} of {total} ({progress:.1f}%)'
GAZE_MOVIE_COMPLETE = 'Gaze heatmap movie creation completed.'
AUDIO_FRAME_RATE_MESSAGE = "audio frame rate: {frame_rate}"

# String constants for file paths and naming
HEATMAP_FILENAME_TEMPLATE = "task_{task_code}_heatmap_sub_{subject_name}.jpg"
VIDEO_FILENAME_TEMPLATE = "participant_task_{task_code}_{subject_name}.mp4"
VIDEO_OUTPUT_WITH_AUDIO = "participant_task_{task_code}_{subject_name}_audio.mp4"
GAZE_HEATMAP_VIDEO_FILENAME_TEMPLATE = "{task_code}_{subject_name}_gaze_heatmap_video.mp4"


FIXATION_CSV_KEY_EYE_H ="eye_horizontal"
FIXATION_CSV_KEY_EYE_V ="eye_vertical"
FIXATION_CSV_KEY_FIXATION ="evt"
FIXATION_VALID_STATUS = "status"
FIXATION_CSV_TIME = "t"
SACCADE_IDX = 2
FIXATION_IDX = 1
# gazeNET's 3rd event class (ModelPropertise.EVENTS / GazeModel/config.json's "events":
# [1,2,3]) - post-saccadic oscillation, per the gazeNET paper's 3-class scheme
# (Zemblys et al. 2018). Only ever produced by model_based annotation - threshold_based
# has no equivalent 3rd class.
PSO_IDX = 3

# "Merge adjacent fixations" post-processing of the gazeNET labels (decision 2026-11-10),
# applied in GazeModel/recording_data_runner.generate_fixation_model_based so SDMT and fxsp
# both get it. Two fixations are joined (the samples between them relabeled fixation) when
# the time between them is <= 75 ms AND the angle between their mean positions is <= 0.5 deg
# - the Tobii I-VT filter defaults (Olsen, 2012, "The Tobii I-VT Fixation Filter", Tobii
# Technology white paper; see also Komogortsev et al., 2010, IEEE Trans Biomed Eng
# 57:2635-2645). gazeNET itself has no such post-processing (Zemblys, Niehorster &
# Holmqvist, 2019), so a 1-2 sample "saccade" otherwise splits a fixation in two.
MERGE_ADJACENT_FIXATIONS = True
MERGE_MAX_TIME_BETWEEN_FIXATIONS_MS = 75
MERGE_MAX_ANGLE_BETWEEN_FIXATIONS_DEG = 0.5

# BCEA interpolation rules, shared by fxsp (saccade_latency_analysis) and SDMT
# (feature_pipeline) via saccade_latency_analysis.filled_in_rule.
# Filled-in (interpolated) samples (decision 2026-11-10): the tracker drops isolated
# samples constantly (fxsp: ~5% of panel-1 samples, 98.7% of filled-in samples inside fixations
# are recording gaps, mostly 1-sample stretches), and a straight line across 1-5 samples
# (<= 8.3 ms) between real samples is essentially the real position. A fixation is
# dropped only if it contains a filled-in stretch LONGER than this (a blink / track
# loss, where gaze was really not measured); otherwise all its samples, filled-in ones
# included, go into BCEA. Was: any filled-in sample dropped the fixation (kept 61% of
# fixation samples; this rule keeps ~93%).
BCEA_MAX_FILLED_IN_STRETCH_SAMPLES = 5
# ...AND it must have < 15% filled-in samples overall (decision 2026-11-10). The stretch
# rule alone lets through "flicker" - many short dropouts (e.g. NS960: 214 of 716 samples
# filled in, all in 1-5 sample stretches). Such data loss goes with poor tracking quality
# (Wass, Smith & Johnson, 2013, Behav Res Methods 45:229-250), and in this data BCEA rises
# with the filled-in share (>= 300 ms fixations, relative to the participant's median:
# 1.2x at 0%, 1.8x at 5-20%, 2.2x at 20-30%, 3.1x above 30%). The 15% cutoff is a
# data-driven choice, not a published standard; it removes ~3% of fixations.
BCEA_MAX_FILLED_IN_FIXATION_PCT = 15


# #fixatio saccade preprossesing constants
# MIN_FIXATION_POINTS = 90 #600*150/1000  for 150 ms at 600Hz
THRESHOLD_FOR_MARKOV_DIAGRAM = 1/11  # Minimum transition probability to include an edge in the Markov diagram

class ModelPropertise(Enum):
    # Relative to this file's own location (not a hardcoded path on one machine) so the
    # weights, checked into the repo under GazeModel/, resolve on any clone.
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GazeModel", "gazeNET_0004_00003750.pth.tar")
    EVENTS = [1,2,3]



PIXEL2METER = 0.000264583
ROW_SIZE = 8
SCREEN_SIZE = [1080,1920] #(height, width), (vertical, horizontal)
PIXEL2METER = 0.000264583  # Conversion factor for 96 DPI (pixels to meters)
screenDistance = 0.65  
deg_per_pixel = 2 * math.degrees(math.atan((PIXEL2METER / 2) / screenDistance))
# Measured lab screen (checked 2026-11-08): 60.4 cm diagonal, 1920x1080, active area
# 52.7 x 29.65 cm -> 0.02745 cm/px both axes (square pixels). PIXEL2METER above is the
# 96-DPI default (0.02646 cm/px, ~3.6% smaller) and is left unchanged because the SDMT
# pipeline (gazeNET degree input, threshold annotation) was built on it.
SCREEN_WIDTH_CM = 52.7
SCREEN_HEIGHT_CM = 29.65
deg_per_cm = 2 * math.degrees(math.atan(0.5 / (screenDistance * 100)))

panel_center_locations = [
        [537.0,296.0],
        [597.1428571428571,296.0],
        [657.2857142857143,296.0],
        [717.4285714285714,296.0],
        [777.5714285714286,296.0],
        [837.7142857142858,296.0],
        [897.8571428571429,296.0],
        [958.0,296.0],
        [1018.1428571428571,296.0],
        [1078.2857142857142,296.0],
        [1138.4285714285716,296.0],
        [1198.5714285714284,296.0],
        [1258.7142857142858,296.0],
        [1318.857142857143,296.0],
        [1379.0,296.0],
        [537.0,401.42857142857144],
        [597.1428571428571,401.42857142857144],
        [657.2857142857143,401.42857142857144],
        [717.4285714285714,401.42857142857144],
        [777.5714285714286,401.42857142857144],
        [837.7142857142858,401.42857142857144],
        [897.8571428571429,401.42857142857144],
        [958.0,401.42857142857144],
        [1018.1428571428571,401.42857142857144],
        [1078.2857142857142,401.42857142857144],
        [1138.4285714285716,401.42857142857144],
        [1198.5714285714284,401.42857142857144],
        [1258.7142857142858,401.42857142857144],
        [1318.857142857143,401.42857142857144],
        [1379.0,401.42857142857144],
        [537.0,506.8571428571429],
        [597.1428571428571,506.8571428571429],
        [657.2857142857143,506.8571428571429],
        [717.4285714285714,506.8571428571429],
        [777.5714285714286,506.8571428571429],
        [837.7142857142858,506.8571428571429],
        [897.8571428571429,506.8571428571429],
        [958.0,506.8571428571429],
        [1018.1428571428571,506.8571428571429],
        [1078.2857142857142,506.8571428571429],
        [1138.4285714285716,506.8571428571429],
        [1198.5714285714284,506.8571428571429],
        [1258.7142857142858,506.8571428571429],
        [1318.857142857143,506.8571428571429],
        [1379.0,506.8571428571429],
        [537.0,612.2857142857142],
        [597.1428571428571,612.2857142857142],
        [657.2857142857143,612.2857142857142],
        [717.4285714285714,612.2857142857142],
        [777.5714285714286,612.2857142857142],
        [837.7142857142858,612.2857142857142],
        [897.8571428571429,612.2857142857142],
        [958.0,612.2857142857142],
        [1018.1428571428571,612.2857142857142],
        [1078.2857142857142,612.2857142857142],
        [1138.4285714285716,612.2857142857142],
        [1198.5714285714284,612.2857142857142],
        [1258.7142857142858,612.2857142857142],
        [1318.857142857143,612.2857142857142],
        [1379.0,612.2857142857142],
        [537.0,717.7142857142858],
        [597.1428571428571,717.7142857142858],
        [657.2857142857143,717.7142857142858],
        [717.4285714285714,717.7142857142858],
        [777.5714285714286,717.7142857142858],
        [837.7142857142858,717.7142857142858],
        [897.8571428571429,717.7142857142858],
        [958.0,717.7142857142858],
        [1018.1428571428571,717.7142857142858],
        [1078.2857142857142,717.7142857142858],
        [1138.4285714285716,717.7142857142858],
        [1198.5714285714284,717.7142857142858],
        [1258.7142857142858,717.7142857142858],
        [1318.857142857143,717.7142857142858],
        [1379.0,717.7142857142858],
        [537.0,823.1428571428571],
        [597.1428571428571,823.1428571428571],
        [657.2857142857143,823.1428571428571],
        [717.4285714285714,823.1428571428571],
        [777.5714285714286,823.1428571428571],
        [837.7142857142858,823.1428571428571],
        [897.8571428571429,823.1428571428571],
        [958.0,823.1428571428571],
        [1018.1428571428571,823.1428571428571],
        [1078.2857142857142,823.1428571428571],
        [1138.4285714285716,823.1428571428571],
        [1198.5714285714284,823.1428571428571],
        [1258.7142857142858,823.1428571428571],
        [1318.857142857143,823.1428571428571],
        [1379.0,823.1428571428571],
        [537.0,928.5714285714286],
        [597.1428571428571,928.5714285714286],
        [657.2857142857143,928.5714285714286],
        [717.4285714285714,928.5714285714286],
        [777.5714285714286,928.5714285714286],
        [837.7142857142858,928.5714285714286],
        [897.8571428571429,928.5714285714286],
        [958.0,928.5714285714286],
        [1018.1428571428571,928.5714285714286],
        [1078.2857142857142,928.5714285714286],
        [1138.4285714285716,928.5714285714286],
        [1198.5714285714284,928.5714285714286],
        [1258.7142857142858,928.5714285714286],
        [1318.857142857143,928.5714285714286],
        [1379.0,928.5714285714286],
        [537.0,1034.0],
        [597.1428571428571,1034.0],
        [657.2857142857143,1034.0],
        [717.4285714285714,1034.0],
        [777.5714285714286,1034.0],
        [837.7142857142858,1034.0],
        [897.8571428571429,1034.0],
        [958.0,1034.0],
        [1018.1428571428571,1034.0],
        [1078.2857142857142,1034.0],
        [1138.4285714285716,1034.0],
        [1198.5714285714284,1034.0],
        [1258.7142857142858,1034.0],
        [1318.857142857143,1034.0],
        [1379.0,1034.0
        ]
    ]
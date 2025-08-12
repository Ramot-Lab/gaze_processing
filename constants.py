import math
from enum import Enum

KEY_TOBII_DATA = "tobii_data"  # Key for Tobii eye movement data
KEY_TASK_PANEL_IMG = "task_panel_img"  # Key for the image of the task panel
KEY_AUDIO_DATA = "audio_data"  # Key for the audio data file path
KEY_STRIKE_SCORE = "strike_score"
KEY_RECORDING_DATE = "recording_date"
TIME_STAMP = "t"
SIGNAL_IDX = "signal_index"
SENTENCE_BREAK = "sentence_breaks"
MAX_VALID_NAN_VALUES = 0.1 #10% of nan values in the eye tracking data is still valid


OLD_MIC_REPLACEMENT_DATE = "2024-07-01"

VIDEO_CODEC = 'mp4v'  # Codec for .mp4
FPS = 600 / 6  # Frames per second for video output
RADIUS = 6  # Radius for eye position marker
COLOR = (255, 0, 0)  # Color for eye marker (red in BGR format)
THICKNESS = 3  # Thickness of the circle around the eye position

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

# String constants for messages and logs
VIDEO_SAVED_MESSAGE = "Video saved without audio at: {video_output_path}"
GAZE_MOVIE_PROGRESS = 'Processing data point {i} of {total} ({progress:.1f}%)'
GAZE_MOVIE_COMPLETE = 'Gaze heatmap movie creation completed.'
AUDIO_FRAME_RATE_MESSAGE = "audio frame rate: {frame_rate}"

# String constants for file paths and naming
HEATMAP_FILENAME_TEMPLATE = "task_{task_code}_heatmap_sub_{subject_name}.jpg"
VIDEO_FILENAME_TEMPLATE = "participant_task_{task_code}_{subject_name}.mp4"
VIDEO_OUTPUT_WITH_AUDIO = "participant_task_{task_code}_{subject_name}_audio.mp4"

FIXATION_CSV_KEY_EYE_H ="eye_horizontal"
FIXATION_CSV_KEY_EYE_V ="eye_vertical"
FIXATION_CSV_KEY_FIXATION ="evt"
FIXATION_VALID_STATUS = "status"
SACCADE_IDX = 2
FIXATION_IDX = 1

class ModelPropertise(Enum):
    MODEL_PATH = "/Users/nitzankarby/Desktop/dev/Nitzan_K/MS_processing/GazeModel/gazeNET_0004_00003750.pth.tar"
    EVENTS = [1,2,3]



PIXEL2METER = 0.000264583
ROW_SIZE = 8
SCREEN_SIZE = [1080,1920] #(height, width), (vertical, horizontal)
PIXEL2METER = 0.000264583  # Conversion factor for 96 DPI (pixels to meters)
screenDistance = 0.65  
deg_per_pixel = 2 * math.degrees(math.atan((PIXEL2METER / 2) / screenDistance))

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
KEY_TOBII_DATA = "tobii_data"  # Key for Tobii eye movement data
KEY_TASK_PANEL_IMG = "task_panel_img"  # Key for the image of the task panel
KEY_AUDIO_DATA = "audio_data"  # Key for the audio data file path
KEY_STRIKE_SCORE = "strike_score"
KEY_RECORDING_DATE = "recording_date"
TIME_STAMP = "time_stamp"
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
FIXATION_CSV_KEY_FIXATION ="fixation"

panel_center_locations = [[537, 297],
                          [537, 402],
                          [537, 507],
                          [537, 612],
                          [537, 718],
                          [537, 823],
                          [537, 928],
                          [537, 1034],
                          [597, 297],
                          [597, 402],
                          [597, 507],
                          [597, 612],
                          [597, 718],
                          [597, 823],
                          [597, 928],
                          [597, 1034],
                          [657, 297],
                          [657, 402],
                          [657, 507],
                          [657, 612],
                          [657, 718],
                          [657, 823],
                          [657, 928],
                          [657, 1034],
                          [717, 297]]

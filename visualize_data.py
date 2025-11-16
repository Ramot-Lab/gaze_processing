import matplotlib.pyplot as plt
from participant_gaze_data_manager import ParticipantGazeDataManager
import matplotlib.animation as animation
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import wave
from matplotlib.animation import FFMpegWriter
from constants import *
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
import subprocess
from SearchFinder import SearchFinder, SearchVisualizer
from utils import prepare_image_and_gaze
import moviepy.config as mpy_config
from matplotlib import cm


mpy_config.change_settings({
    "FFMPEG_BINARY": "/opt/anaconda3/envs/gaze/bin/ffmpeg"
})


MARKER_SIZE = 10  # Set your marker size
ANIMATION_INTERVAL = 20  # Set your animation interval
BLIT = True  # Use blitting to optimize performance
evt_color_map = dict({
        0: 'gray',  #0. Undefined
        1: 'b',     #1. Fixation
        2: 'r',     #2. Saccade
        3: 'y',     #3. Post-saccadic oscillation
        4: 'm',     #4. Smooth pursuit
        5: 'k',     #5. Blink
        9: 'k',     #9. Other
    })


def show_running_video_live(eye_data, img_path):
    """
    Plots the movement of the eye across the image and the correlated audio live.
    """
    # Read the image
    img = plt.imread(img_path)
    # Extract the eye movement coordinates and fixation from the DataFrame
    eye_x = eye_data[FIXATION_CSV_KEY_EYE_H].values * SCREEN_SIZE[1]  # Scale x from [0,1] to [0, image width]
    eye_y = eye_data[FIXATION_CSV_KEY_EYE_V].values * SCREEN_SIZE[0]    # Scale y from [0,1] to [0, image height]
    fixation = eye_data[FIXATION_CSV_KEY_FIXATION].values  # Get fixation data

    # Create figure and axis for the video
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    # Eye movement plot
    eye_plot, = ax.plot([], [], 'o', markersize=MARKER_SIZE)

    # Update function for animation
    def update(i):
        # Change color based on fixation
        if fixation[i] == SACCADE_IDX:
            eye_plot.set_color('red')  # saccade  -> red
        else:
            eye_plot.set_color('blue')  # Fixation  -> blue
        
        eye_plot.set_data(eye_x[i], eye_y[i])  # Update eye position
        return eye_plot,

    # Set up the animation
    ani = animation.FuncAnimation(fig, update, frames=len(eye_x), interval=ANIMATION_INTERVAL, blit=BLIT)

    # Display the animation live
    plt.show()

def nan_helper(x):
    return np.isnan(x), lambda z: z.nonzero()[0]

def _get_image_and_gaze_data_and_rescale(subject, panel_code):
    # Get the data
    annotated_data = subject.annotate_gaze_events(KEY_ANNOTATION_MODEL_BASE, panel)
    matching_dictionary = subject.matched_data
    task_data = matching_dictionary[panel_code]
    img = cv2.imread(task_data[KEY_TASK_PANEL_IMG])
    resized_img, func_gaze_data_scaled = prepare_image_and_gaze(img, annotated_data)
    return resized_img, func_gaze_data_scaled

def create_gaze_heatmap_video(subject: ParticipantGazeDataManager, panel_code: str, output_path: str):
    """
    Create a silent gaze video overlayed on an image.
    Fixations -> blue heatmap, Saccades -> red heatmap.

    Parameters
    ----------
    subject : ParticipantGazeDataManager
        Subject data manager.
    panel_code : str like "0", "i4", "l1" etc.
    output_path : str
        Path to save the output video (.mp4).

    """
    
    # Load background image
    img, gaze_df_scaled = _get_image_and_gaze_data_and_rescale(subject, panel_code)
    height, width, _ = img.shape
    fps = FPS/10 # downsampled to 60 fps

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Downsample gaze data (600Hz -> 60Hz)
    gaze_ds = gaze_df_scaled.iloc[::10].reset_index(drop=True)
    # Limit to first 30 ms for testing
    gaze_ds = gaze_ds.iloc[:30]


    duration = len(gaze_df_scaled) / FPS

    # Normalize time to [0, 1] over 90s
    gaze_ds['time_norm'] = (gaze_ds[FIXATION_CSV_TIME] - gaze_ds[FIXATION_CSV_TIME].min()) / (duration * 1000)  # time in ms
    gaze_ds['time_norm'] = gaze_ds['time_norm'].clip(0, 1)

    # Generate colormaps
    cmap_fix = plt.cm.Blues
    cmap_sac = plt.cm.Reds

    # Frame loop
    total_frames = fps * duration
    for frame_idx in range(int(total_frames)):
        frame = img.copy()

        # Current time in ms
        current_time = frame_idx / fps * 1000

        # Select data up to current time
        current_points = gaze_ds[gaze_ds[FIXATION_CSV_TIME] <= current_time]

        for _, row in current_points.iterrows():
            x, y, t, label = int(row[FIXATION_CSV_KEY_EYE_H]), int(row[FIXATION_CSV_KEY_EYE_V]), row['time_norm'], row[FIXATION_CSV_KEY_FIXATION]
            if label == FIXATION_IDX:
                color = tuple(int(c * 255) for c in cmap_fix(t)[:3])
            else:
                color = tuple(int(c * 255) for c in cmap_sac(t)[:3])

            cv2.circle(frame, (x, y), 5, color, -1)

        video.write(frame)
        print(f"Processing frame {frame_idx + 1}/{total_frames}")


    video.release()
    print(f"Video saved to {output_path}")

#WORKING
def show_running_video(subject, panel_code, output_path):
    """
    Plots the movement of the eye across the image and the correlated audio, saving the final video with embedded audio.
    """

    img, func_gaze_data_scaled = _get_image_and_gaze_data_and_rescale(subject, panel_code)
    img_height, img_width, _ = img.shape

    eye_x = func_gaze_data_scaled[FIXATION_CSV_KEY_EYE_H]
    eye_y = func_gaze_data_scaled[FIXATION_CSV_KEY_EYE_V]
    fixation = func_gaze_data_scaled[FIXATION_CSV_KEY_FIXATION]

    # Set up OpenCV video writer
    video_output_path = os.path.join(output_path, VIDEO_FILENAME_TEMPLATE.format(task_code=panel_code, subject_name=subject.name))
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)  # Codec for .mp4
    out = cv2.VideoWriter(video_output_path, fourcc, FPS, (img_width, img_height))    

    # Render each frame and write to the video file
    for i in range(len(eye_x)):
        if fixation.iloc[i] == SACCADE_IDX:
            frame_color = SACCADE_COLOR
        else:
            frame_color = FIXATION_COLOR
        cur_img = cv2.circle(img.copy(), (int(eye_x[i]), int(eye_y[i])), radius=RADIUS, color=frame_color, thickness=THICKNESS)
        # Write frame to video file
        out.write(cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR))
        print(f"Processing frame {i + 1}/{len(eye_x)}")
    # Release the video writer
    out.release()
    print(VIDEO_SAVED_MESSAGE.format(video_output_path=video_output_path))

#better for human eye - 600 fps is not supported in all video players (like QuickTime in Mac)
def show_running_video_60fps(subject, panel_code, output_path, target_fps=60):
    """
    Generates a gaze video at a fixed FPS (e.g., 60 fps) without changing timeline.
    """
    img, func_gaze_data_scaled = _get_image_and_gaze_data_and_rescale(subject, panel_code)
    img_height, img_width, _ = img.shape

    eye_x = func_gaze_data_scaled[FIXATION_CSV_KEY_EYE_H].to_numpy()
    eye_y = func_gaze_data_scaled[FIXATION_CSV_KEY_EYE_V].to_numpy()
    fixation = func_gaze_data_scaled[FIXATION_CSV_KEY_FIXATION].to_numpy()
    times = func_gaze_data_scaled[FIXATION_CSV_TIME].to_numpy() / SECONDS_TO_MICROSECOND_FACTOR  # seconds

    # Set up OpenCV video writer
    video_output_path = os.path.join(output_path, VIDEO_FILENAME_TEMPLATE.format(task_code=panel_code, subject_name=subject.name))
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    out = cv2.VideoWriter(video_output_path, fourcc, target_fps, (img_width, img_height))

    # Compute frame times for target FPS
    duration = times[-1]  # last gaze timestamp in seconds
    n_frames = int(duration * target_fps)
    frame_times = np.linspace(0, duration, n_frames)

    gaze_idx = 0
    for ft in frame_times:
        # Find the last gaze sample <= current frame time
        while gaze_idx + 1 < len(times) and times[gaze_idx + 1] <= ft:
            gaze_idx += 1

        fx, fy = int(eye_x[gaze_idx]), int(eye_y[gaze_idx])
        color = SACCADE_COLOR if fixation[gaze_idx] == SACCADE_IDX else FIXATION_COLOR

        cur_img = cv2.circle(img.copy(), (fx, fy), radius=RADIUS, color=color, thickness=THICKNESS)
        out.write(cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"✅ Video saved at {target_fps} FPS: {video_output_path}")

def show_heatmap(subject_data: ParticipantGazeDataManager, task_code, output_path = "", show_plot = True):
    """
    Plots and saves a heat map of the eye movement for a given task.
    Saves the figure to <output_path> under the name "task_{task_code}_heatmap.jpg".
    """

    # Get the data
    matching_dictionary = subject_data.matched_data
    task_data = matching_dictionary[task_code]
    eye_data = task_data[KEY_TOBII_DATA]
    img = plt.imread(task_data[KEY_TASK_PANEL_IMG])

    eye_x = eye_data[:, 0] * SCREEN_SIZE[0]  # Scale x from [0,1] to [0, image width]
    eye_y = (1 - eye_data[:, 1]) * SCREEN_SIZE[1]  # Scale y from [0,1] to [0, image height]

    # Create a 2D histogram (heatmap) of the eye positions
    heatmap, _, _ = np.histogram2d(eye_x, eye_y, bins=[SCREEN_SIZE[0], SCREEN_SIZE[1]], range=[[0, SCREEN_SIZE[0]], [0, SCREEN_SIZE[1]]])
    img_height, img_width = SCREEN_SIZE
    # Apply Gaussian filter to smooth the heatmap
    heatmap = gaussian_filter(heatmap, sigma=GAUSSIAN_SIGMA)

    # Plot the heatmap on top of the task image
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, img_width, 0, img_height], alpha=0.8)
    ax.imshow(heatmap.T, extent=[0, img_width, 0, img_height], origin='lower', cmap=HEATMAP_COLOR_MAP, alpha=HEATMAP_ALPHA)

    if len(output_path) > 0:
        # Save the heatmap
        heatmap_output_path = os.path.join(output_path, HEATMAP_FILENAME_TEMPLATE.format(task_code=task_code, subject_name=subject_data.name))
        plt.savefig(heatmap_output_path)
    if show_plot:
        plt.show()

# def create_gaze_heatmap_movie(subject_data:ParticipantGazeDataManager, panel_code:str, output_path:str):
    """
    Creates a gaze heatmap movie showing where the subject looked on the image over time.
    """
    # #I added the resizing of image and gaze data scaling
    # img, gaze_data_scaled = prepare_image_and_gaze(img, annotated_data)

    # Get the data
    # matching_dictionary = subject_data.matched_data
    # task_data = matching_dictionary[panel_code]
    # eye_data = task_data[KEY_TOBII_DATA]
    # img = plt.imread(task_data[KEY_TASK_PANEL_IMG])


    img, eye_data = _get_image_and_gaze_data_and_rescale(subject_data, panel_code)

    total_frames = TOTAL_FRAMES  # Total number of frames for the movie
    writer = FFMpegWriter(fps=OUTPUT_FPS, codec = 'mpeg4')
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes

    heatmap = np.zeros((img.shape[0], img.shape[1]))  # Create an empty heatmap
    decay_factor = DECAY_FACTOR  # Slower decay for more persistent heatmap

    data_points = eye_data.shape[0]  # Number of data points
    frames_per_point = max(1, total_frames // data_points)  # Frames to show per data point

    # Setup heatmap plot
    h = ax.imshow(heatmap, cmap='hot', alpha=HEATMAP_ALPHA_MOVIE, interpolation='bilinear')

    # Setup scatter plot for gaze point
    scatter_obj = ax.scatter([], [], s=150, c='r', edgecolor='k')

    print('Creating gaze heatmap movie...')
    output_filename = os.path.join(output_path, GAZE_HEATMAP_VIDEO_FILENAME_TEMPLATE.format(task_code=panel_code, subject_name=subject_data.name))
    with writer.saving(fig, output_filename, dpi=100):
        frame_count = 0
        for i , row in eye_data.iterrows():
            heatmap *= decay_factor  # Decay heatmap over time

            x = int(np.round(row[FIXATION_CSV_KEY_EYE_H]))
            y = int(np.round(row[FIXATION_CSV_KEY_EYE_V]))


            # Check if coordinates are valid
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                heatmap[y, x] += 1  # Increment heatmap at gaze point

            # Update heatmap and scatter plot
            h.set_data(heatmap)
            scatter_obj.set_offsets([[x, y]])
            h.set_clim(0, np.max(heatmap))  # Adjust color limits dynamically
            print(f"Processing data point {i + 1}/{data_points}...")
            # Write frames for the current data point
            for _ in range(frames_per_point):
                writer.grab_frame()
                frame_count += 1
                if frame_count >= total_frames:
                    break

            # Progress feedback
            if i % (data_points // 10) == 0:
                print(GAZE_MOVIE_PROGRESS.format(i=i + 1, total=data_points, progress=(i + 1) / data_points * 100))

            if frame_count >= total_frames:
                break

    print(GAZE_MOVIE_COMPLETE)

# havent checked this function yet!
def create_gaze_course_image(img, annotated_data, output_filename, time_limit=10.0):
    """
    Creates a static image showing the course of gaze over the stimulus image.
    Fixations are shown as blue circles with a time gradient,
    and saccades are shown as red lines with a time gradient.
    
    Parameters
    ----------
    img : np.ndarray
        Background image (H x W x C).
    annotated_data : np.ndarray
        Gaze data, must include at least: [time, x, y, fixation_flag].
        fixation_flag: 1 for fixation, 0 for saccade.
    output_filename : str
        Path to save the resulting visualization (e.g. "gaze_course.png").
    time_limit : float
        Only include data up to this many seconds.
    """
    # Step 1: Resize + scale gaze points to image size
    img, gaze_data_scaled = prepare_image_and_gaze(img, annotated_data)

    # Step 2: Filter to first `time_limit` seconds
    gaze_data_scaled = gaze_data_scaled[gaze_data_scaled[:, FIXATION_CSV_TIME] <= time_limit * SECONDS_TO_MICROSECOND_FACTOR]
    times = gaze_data_scaled[:, FIXATION_CSV_TIME]
    xs = gaze_data_scaled[:, FIXATION_CSV_KEY_EYE_H]
    ys = gaze_data_scaled[:, FIXATION_CSV_KEY_EYE_V]
    fixation_flags = gaze_data_scaled[:, FIXATION_CSV_KEY_FIXATION]

    # Normalize times to [0, 1] for color gradients
    time_norm = (times - times.min()) / (times.max() - times.min() + 1e-9)

    # Step 3: Set up the figure
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # Step 4: Plot saccades (red gradient)
    for i in range(len(xs) - 1):
        if fixation_flags[i] == SACCADE_IDX:  # if saccade point
            color = plt.cm.Reds(time_norm[i])  # red gradient
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=color, linewidth=2)

    # Step 5: Plot fixations (blue gradient)
    ax.scatter(xs[fixation_flags == FIXATION_IDX], ys[fixation_flags == FIXATION_IDX],
               c=time_norm[fixation_flags == FIXATION_IDX], cmap='Blues',
               s=80, edgecolor='k', alpha=0.9, zorder=3)

    # Step 6: Save and show
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Gaze course image saved to {output_filename}")

#working but not pretty
def create_gaze_course_image_instant(img, annotated_data, time_limit=10.0):
    """
    Creates and displays a static image showing the course of gaze over the stimulus image.
    Fixations are shown as blue circles with a time gradient,
    and saccades are shown as red lines with a time gradient.

    Parameters
    ----------
    img : np.ndarray
        Background image (H x W x C).
    annotated_data : np.ndarray
        Gaze data, must include at least: [time, x, y, fixation_flag].
        fixation_flag: 1 for fixation, 0 for saccade.
    time_limit : float
        Only include data up to this many seconds.
    """
    # Step 1: Resize + scale gaze points to image size
    img, gaze_data_scaled = prepare_image_and_gaze(img, annotated_data)

    # Step 2: Filter to first `time_limit` seconds
    gaze_data_scaled = gaze_data_scaled[
        gaze_data_scaled[FIXATION_CSV_TIME] <= time_limit * SECONDS_TO_MICROSECOND_FACTOR
    ]
    times = gaze_data_scaled[FIXATION_CSV_TIME]
    xs = gaze_data_scaled[FIXATION_CSV_KEY_EYE_H]
    ys = gaze_data_scaled[FIXATION_CSV_KEY_EYE_V]
    fixation_flags = gaze_data_scaled[FIXATION_CSV_KEY_FIXATION]

    # Normalize times to [0, 1] for color gradients
    time_norm = (times - times.min()) / (times.max() - times.min() + 1e-9)

    # Step 3: Set up the figure
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # Step 4: Plot saccades (red gradient)
    for i in range(len(xs) - 1):
        if fixation_flags[i] == SACCADE_IDX:  # if saccade point
            color = plt.cm.Reds(time_norm[i])  # red gradient
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=color, linewidth=2)

    # Step 5: Plot fixations (blue gradient)
    ax.scatter(
        xs[fixation_flags == FIXATION_IDX],
        ys[fixation_flags == FIXATION_IDX],
        c=time_norm[fixation_flags == FIXATION_IDX],
        cmap='Blues',
        s=80,
        edgecolor='k',
        alpha=0.9,
        zorder=3,
    )

    # Step 6: Show immediately (no saving)
    plt.tight_layout()
    plt.show()

def plot_histogram(data, x_title, title, label = ''):
    plt.clf()
    plt.hist(data, label=label)
    plt.xlabel(x_title)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()

def plot_barplot(x_axis, y_axis, x_title, y_title, fig_title):
    plt.clf()
    plt.bar(x_axis,y_axis)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(fig_title)
    plt.show()
    plt.clf()

def plot_gaze_over_img(subject_data:ParticipantGazeDataManager, img_path, task_code, output_path):
    img = plt.imread(img_path)
    matching_dictionary = subject_data.matched_data
    task_data = matching_dictionary[task_code]
    eye_data = task_data[KEY_TOBII_DATA]

    resized_img, eye_x, eye_y = prepare_image_and_gaze(img, eye_data)

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(resized_img)
    ax.scatter(eye_x, eye_y, s=3, c="red")
    ax.set_title(f"Gaze overlay: {task_code} method 1")

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, f"{task_code}_gaze_overlay_with_scaling_fix.png")
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved gaze overlay plot1 to {out_file}")

    plt.show()
    plt.close(fig)

    #### OLD VERSION - WITHOUT RESIZING OF IMAGE TO FIT SCREEN SIZE

def plot_gaze_over_img2(subject_data:ParticipantGazeDataManager, img_path, task_code, output_path):
    img = plt.imread(img_path)
    matching_dictionary = subject_data.matched_data
    task_data = matching_dictionary[task_code]
    eye_data = task_data[KEY_TOBII_DATA]
    eye_x = eye_data[:, 0] * SCREEN_SIZE[1]  # Scale x from [0,1] to [0, image width]
    eye_y = (eye_data[:, 1]) * SCREEN_SIZE[0]  # Scale y from [0,1] to [0, image height]
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(eye_x, eye_y, s=3, c="red")
    ax.set_title(f"Gaze overlay: {task_code} method 2")

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, f"{task_code}_gaze_overlay_no_scaling.png")
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved gaze overlay plot2 to {out_file}")
        
    plt.show()
    plt.close(fig)

def plot_gaze_over_img_original(subject_data:ParticipantGazeDataManager, img_path, task_code):
    img = plt.imread(img_path)
    matching_dictionary = subject_data.matched_data
    task_data = matching_dictionary[task_code]
    eye_data = task_data[KEY_TOBII_DATA]
    eye_x = eye_data[:, 0] * SCREEN_SIZE[1]  # Scale x from [0,1] to [0, image width]
    eye_y = (eye_data[:, 1]) * SCREEN_SIZE[0]  # Scale y from [0,1] to [0, image height]
    plt.imshow(img)
    plt.scatter(eye_x, eye_y, s=3)
    plt.show()

def plot_gaze_over_img_by_time(annotated_data, img, times):
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Define colormaps for each time window
    colormaps = [cm.Blues, cm.Greens, cm.Reds, cm.Purples, cm.Oranges]

    for i, (start, end) in enumerate(times):
        # Filter relevant time span
        mask = (annotated_data[FIXATION_CSV_TIME] >= start) & (annotated_data[FIXATION_CSV_TIME] <= end)
        data = annotated_data.loc[mask]

        if data.empty:
            print(f"No data between {start}–{end}")
            continue

        eye_x = data[FIXATION_CSV_KEY_EYE_H].values
        eye_y = data[FIXATION_CSV_KEY_EYE_V].values
        time_vals = data[FIXATION_CSV_TIME].values

        # Normalize time within the window to get gradient colors
        norm = plt.Normalize(vmin=time_vals.min(), vmax=time_vals.max())
        colors = colormaps[i % len(colormaps)](norm(time_vals))

        ax.scatter(eye_x, eye_y, s=3, c=colors, alpha=0.7, edgecolors="none")

    ax.set_title("Gaze over image (time-colored)")
    ax.axis("off")

    plt.show()
    plt.close(fig)


def plot_gazeNet_fig(data, spath = None, save=False, show=True, title=None):
    '''Plots trial
    '''
    if show:
        plt.ioff()
    # if show:
    #     plt.ion()
    # else:
    #     plt.ioff()
    if 'x' in data.keys():
        horizontal, vertical, time = ('x', 'y', 't')
    else:
        horizontal, vertical, time = (FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, TIME_STAMP)
    fig = plt.figure(figsize=(10,6))
    ax00 = plt.subplot2grid((2, 2), (0, 0))
    ax10 = plt.subplot2grid((2, 2), (1, 0), sharex=ax00)
    ax01 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    ax00.plot(data[time], data[horizontal], '-')
    ax10.plot(data[time], data[vertical], '-')
    ax01.plot(data[horizontal], data[vertical], '-')
    for e, c in evt_color_map.items():
        mask = data['evt'] == e
        ax00.plot(data[time][mask], data[horizontal][mask], '.', color = c)
        ax10.plot(data[time][mask], data[vertical][mask], '.', color = c)
        ax01.plot(data[horizontal][mask], data[vertical][mask], '.', color = c)

    etdata_extent = np.nanmax([np.abs(data[horizontal]), np.abs(data[vertical])])+1

    ax00.axis([data[time].min(), data[time].max(), -etdata_extent, etdata_extent])
    ax10.axis([data[time].min(), data[time].max(), -etdata_extent, etdata_extent])
    ax01.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])

    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()

    plt.show()
    if save and not(spath is None):
        plt.savefig('%s.png' % (spath))
        plt.close()

#is it better then show_running_video()?
def create_panel_video(subject, panel, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Fetch panel data ----
    if panel not in subject.matched_data:
        print(f"[ERROR] Panel '{panel}' not found in subject data.")
        return None

    task_data = subject.matched_data[panel]
    img_path = task_data.get(KEY_TASK_PANEL_IMG)

    eye_data = subject.annotate_gaze_events('threshold_based', panel)

    if eye_data is None or len(eye_data) == 0:
        print("[ERROR] No gaze data found.")
        return None

    # ---- Load and resize panel image ----
    img_bgr = cv2.imread(img_path)
    img, eye_data_scaled = prepare_image_and_gaze(img_bgr, eye_data)
    eye_x = eye_data_scaled[FIXATION_CSV_KEY_EYE_H]
    eye_y = eye_data_scaled[FIXATION_CSV_KEY_EYE_V]
    fixation = eye_data_scaled[FIXATION_CSV_KEY_FIXATION]
    frame_height, frame_width = img.shape[:2]

    # ---- Compute video duration from gaze data ----
    gaze_duration = len(eye_x) / float(FPS)
    if gaze_duration <= 0:
        print("[ERROR] Non-positive gaze duration.")
        return None

    total_frames = int(gaze_duration * FPS)

    # ---- Write silent video ----
    video_path = os.path.join(output_dir, f"{panel}_gaze.mp4")
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (frame_width, frame_height))

    for frame_idx in range(total_frames):
        # Current time in seconds
        t = frame_idx / FPS
        gaze_idx = int(t * FPS)

        frame = img.copy()

        if 0 <= gaze_idx < len(eye_x):
            x = eye_x[gaze_idx]
            y = eye_y[gaze_idx]
            if fixation.iloc[gaze_idx] == SACCADE_IDX:
                color = SACCADE_COLOR
            else:
                color = FIXATION_COLOR
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(frame, (int(x), int(y)), 10, color, -1)

        writer.write(frame)

    writer.release()
    print(f"✅ Written silent gaze video: {video_path}")

    return video_path


def slow_down_video(input_path, output_folder, slow_factor= 2 ):
    """
    Slow down a video (including sound).

    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save slowed video.
        slow_factor (float): Factor by which to slow down video.
                             0.5 = half speed (slower),
                             2.0 = double speed (faster).
    """
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_folder, f"{name}_slowed{ext}")
    # Load video with moviepy
    clip = VideoFileClip(input_path)

    # Slow down video and audio
    slowed_clip = clip.fx(vfx.speedx, slow_factor)

    slowed_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        preset=None,
        threads=4,  
        ffmpeg_params=[]
    )
    clip.close()
    slowed_clip.close()
    print(f"✅ Slowed-down video saved to {output_path}")


if __name__=="__main__":
    p_name = "AG562"
    task = "SDMT"
    group = "pwMS"
    panel = "a3"
    panel_path = f'/Volumes/ramot/Noam_M/Results/Behavior/panels_images/SDMT/combined_testable_{panel}.jpg'
    data_path = "/Volumes/ramot/Noam_M/Results/Behavior"
    output_path = f'/Volumes/ramot/Noam_M/visualized_data/test_videos/'

    video_w_curser_path = '/Volumes/ramot/rotation_students/Noam_M/visualized_data/test_videos/AG562_curser_w_sound.mov'

    # slow_down_video(video_w_curser_path, output_path, slow_factor= 3 )
    subject_data= ParticipantGazeDataManager(p_name, data_path, task, group)
    # # gaze over time in space - and time:
    # res = subject_data.annotate_gaze_events('threshold_based', panel)
    # plot_gazeNet_fig(res, save=True) 
    # img_bgr = cv2.imread(panel_path)   
    # create_panel_video(subject_data, panel, output_path) # - WORKING

    # showing and saving a video = picture + gaze - ?
    show_running_video_60fps(subject_data, panel, output_path)


    # # showing heatmap of the subject gaze - over the image - WORKING
    # create_gaze_heatmap_video(subject_data, panel, output_path)
    
    
    # # showing the a video of gaze over the image - DID NOT WORK!!!!!!!!!
    # create_gaze_heatmap_movie(
    #     img = img_bgr,
    #     annotated_data=res,
    #     output_filename=os.path.join("/Volumes/ramot/rotation_students/Noam_M/Results/Behavior/processing_results/AA_TEST", f"{p_name}_{task}_panel_{panel}_gaze_heatmap_video.mp4"))
    
    # create_gaze_course_image_instant(img=img_bgr,annotated_data=res, time_limit=10.0)
    # plot gaze over image
    # plot_gaze_over_img(subject_data, panel_path , panel, output_path)
    # plot_gaze_over_img2(subject_data, panel_path , panel, output_path)
    #plot_gaze_over_img_original(subject_data, panel_path , panel)




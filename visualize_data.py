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


def show_running_video(subject, task_code, output_path):
    """
    Plots the movement of the eye across the image and the correlated audio, saving the final video with embedded audio.
    """

    # Get the data
    matching_dictionary = subject.matched_data
    task_data = matching_dictionary[task_code]
    eye_data = task_data[KEY_TOBII_DATA]
    img = cv2.imread(task_data[KEY_TASK_PANEL_IMG])

    # Extract the image dimensions
    img_height, img_width, _ = img.shape

    # Extract the eye movement coordinates and timestamps
    eye_x = eye_data[:, 0][::6] * SCREEN_SIZE[1]  # Scale x from [0,1] to [0, image width]
    eye_y = eye_data[:, 1][::6] * SCREEN_SIZE[0]  # Scale y from [0,1] to [0, image height]

    # Set up OpenCV video writer
    video_output_path = os.path.join(output_path, VIDEO_FILENAME_TEMPLATE.format(task_code=task_code, subject_name=subject.name))
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)  # Codec for .mp4
    fps = FPS  # Frames per second
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (img_width, img_height))    

    # Render each frame and write to the video file
    for i in range(len(eye_x)):
        cur_img = cv2.circle(img.copy(), (int(eye_x[i]), int(eye_y[i])), radius=RADIUS, color=COLOR, thickness=THICKNESS)
        # Write frame to video file
        out.write(cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR))

    # Release the video writer
    out.release()
    print(VIDEO_SAVED_MESSAGE.format(video_output_path=video_output_path))



def show_heatmap(subject_data, task_code, output_path = "", show_plot = True):
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


def create_gaze_heatmap_movie(img, func_gaze_data, output_filename):
    """
    Creates a gaze heatmap movie showing where the subject looked on the image over time.
    """
    total_frames = TOTAL_FRAMES  # Total number of frames for the movie
    writer = FFMpegWriter(fps=OUTPUT_FPS)
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes

    heatmap = np.zeros((img.shape[0], img.shape[1]))  # Create an empty heatmap
    decay_factor = DECAY_FACTOR  # Slower decay for more persistent heatmap

    data_points = func_gaze_data.shape[0]  # Number of data points
    frames_per_point = max(1, total_frames // data_points)  # Frames to show per data point

    # Setup heatmap plot
    h = ax.imshow(heatmap, cmap='hot', alpha=HEATMAP_ALPHA_MOVIE, interpolation='bilinear')

    # Setup scatter plot for gaze point
    scatter_obj = ax.scatter([], [], s=150, c='r', edgecolor='k')

    print('Creating gaze heatmap movie...')

    with writer.saving(fig, output_filename, dpi=100):
        frame_count = 0
        for i in range(data_points):
            heatmap *= decay_factor  # Decay heatmap over time
            
            x = int(np.round(func_gaze_data[i, 0]))  # Gaze X coordinate
            y = int(np.round(func_gaze_data[i, 1]))  # Gaze Y coordinate

            # Check if coordinates are valid
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                heatmap[y, x] += 1  # Increment heatmap at gaze point

            # Update heatmap and scatter plot
            h.set_data(heatmap)
            scatter_obj.set_offsets([[x, y]])
            h.set_clim(0, np.max(heatmap))  # Adjust color limits dynamically

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

def plot_gaze_over_img(subject_data:ParticipantGazeDataManager, img_path, task_code):
    img = plt.imread(img_path)
    matching_dictionary = subject_data.matched_data
    task_data = matching_dictionary[task_code]
    eye_data = task_data[KEY_TOBII_DATA]
    eye_x = eye_data[:, 0] * SCREEN_SIZE[1]  # Scale x from [0,1] to [0, image width]
    eye_y = (eye_data[:, 1]) * SCREEN_SIZE[0]  # Scale y from [0,1] to [0, image height]
    plt.imshow(img)
    plt.scatter(eye_x, eye_y, s=3)
    plt.show()

def plot_gazeNet_fig(data, spath = None, save=False, show=True, title=None):
    '''Plots trial
    '''
    if show:
        plt.ion()
    else:
        plt.ioff()
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

if __name__=="__main__":
    p_name = "RD707"
    task = "SDMT"
    group = "pwMS"
    panel = "i1"
    panel_path = "/Users/nitzankarby/Desktop/dev/Nitzan_K/data/panels_images/panel_a5.jpg"
    data_path = "/Volumes/labs/ramot/rotation_students/Nitzan_K/MS/Results/Behavior"
    subject_data= ParticipantGazeDataManager(p_name, data_path, "SDMT", group)
    fixation_data = subject_data.save_fixation_to_csv('l3')
    plot_gazeNet_fig(fixation_data)

    
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

def show_running_video_live(eye_data, img_path):
    """
    Plots the movement of the eye across the image and the correlated audio live.
    """
    # Read the image
    img = plt.imread(img_path)

    # Extract the image dimensions
    img_height, img_width, _ = img.shape

    # Extract the eye movement coordinates and fixation from the DataFrame
    eye_x = eye_data[FIXATION_CSV_KEY_EYE_H].values * img_width  # Scale x from [0,1] to [0, image width]
    eye_y = eye_data[FIXATION_CSV_KEY_EYE_V].values * img_height    # Scale y from [0,1] to [0, image height]
    fixation = eye_data[FIXATION_CSV_KEY_FIXATION].values  # Get fixation data

    # Create figure and axis for the video
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    # Eye movement plot
    eye_plot, = ax.plot([], [], 'o', markersize=MARKER_SIZE)

    # Update function for animation
    def update(i):
        # Change color based on fixation
        if fixation[i] == 0:
            eye_plot.set_color('red')  # Fixation 0 -> red
        else:
            eye_plot.set_color('blue')  # Fixation 1 -> blue
        
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
    eye_x = eye_data[:, 0][::6] * img_width  # Scale x from [0,1] to [0, image width]
    eye_y = eye_data[:, 1][::6] * img_height  # Scale y from [0,1] to [0, image height]

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



def show_heatmap(subject_data, task_code, output_path):
    """
    Plots and saves a heat map of the eye movement for a given task.
    Saves the figure to <output_path> under the name "task_{task_code}_heatmap.jpg".
    """

    # Get the data
    matching_dictionary = subject_data.matched_data
    task_data = matching_dictionary[task_code]
    eye_data = task_data[KEY_TOBII_DATA]
    img = plt.imread(task_data[KEY_TASK_PANEL_IMG])

    # Extract the image dimensions
    img_height, img_width, _ = img.shape

    eye_x = eye_data[:, 0] * img_width  # Scale x from [0,1] to [0, image width]
    eye_y = (1 - eye_data[:, 1]) * img_height  # Scale y from [0,1] to [0, image height]

    # Create a 2D histogram (heatmap) of the eye positions
    heatmap, _, _ = np.histogram2d(eye_x, eye_y, bins=[img_width, img_height], range=[[0, img_width], [0, img_height]])

    # Apply Gaussian filter to smooth the heatmap
    heatmap = gaussian_filter(heatmap, sigma=GAUSSIAN_SIGMA)

    # Plot the heatmap on top of the task image
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, img_width, 0, img_height], alpha=0.8)
    ax.imshow(heatmap.T, extent=[0, img_width, 0, img_height], origin='lower', cmap=HEATMAP_COLOR_MAP, alpha=HEATMAP_ALPHA)

    # Save the heatmap
    heatmap_output_path = os.path.join(output_path, HEATMAP_FILENAME_TEMPLATE.format(task_code=task_code, subject_name=subject_data.name))
    plt.savefig(heatmap_output_path)

    # plt.show()


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

def plot_histogram(data, x_title, title):
    plt.clf()
    plt.hist(data)
    plt.xlabel(x_title)
    plt.title(title)
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

if __name__=="__main__":
    from glob import glob
    
    p_name = "LA627"
    task = "SDMT"
    group = "pwMS"
    data_path = "/Users/nitzankarby/Desktop/dev/Nitzan_K/data"
            
    subject_AA562 = ParticipantGazeDataManager(p_name, data_path, task, group)
    show_running_video(subject_AA562, "l4", "/Users/nitzankarby/Desktop/dev/Nitzan_K/MS_processing/preprocessing_output")
    # show_running_video_live(subject_AA562.save_fixation_to_csv("l4"), "/Users/nitzankarby/Desktop/dev/Nitzan_K/data/panels_images/panel_l4.jpg")
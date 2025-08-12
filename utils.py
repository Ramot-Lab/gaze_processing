import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import os

def get_panel_edges(image) -> list:
    """
    Allows the user to manually select 4 points on the image and returns the 4 points as an array of [(x1, y1), ..., (x4, y4)]
    """
    points = []

    def click_callback(event, x, y, flags, param):
        """
        Callback for mouse events. Adds the selected point to the list.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Points", image)

    # Clone the image for refreshing if needed
    image_clone = image.copy()
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", click_callback)

    # Main loop to show the image and collect points
    while True:
        cv2.imshow("Select Points", image)
        key = cv2.waitKey(1) & 0xFF

        # Reset if user presses 'r'
        if key == ord('r'):
            points = []
            image = image_clone.copy()
            cv2.imshow("Select Points", image)

        # Confirm points if user presses 'c' and 4 points are selected
        elif key == ord('c') and len(points) == 4:
            break

        # Exit without saving points if user presses 'q'
        elif key == ord('q'):
            points = []
            break

    cv2.destroyAllWindows()
    return points

def plot_points_on_image(cv2_image, points):
    """
    Given an image and a set of points, plots red dots at each point on the image.
    Displays the image with the points marked.
    
    Parameters:
    - cv2_image: The OpenCV image on which to plot the points.
    - points: A list of (x, y) tuples representing points to plot on the image.
    """
    # Clone the image to avoid modifying the original
    image_with_dots = cv2_image.copy()

    # Plot each point as a red dot
    for (x, y) in points:
        cv2.circle(image_with_dots, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red dot with a radius of 5

    # Display the image
    cv2.imshow("Image with Points", image_with_dots)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def map_panel_into_dot_points(panel_image):
    json_path = "center_locations.json"
    panel = cv2.imread(panel_image)
    width, hight = panel.shape[:2]
    images_in_row = 15
    rows_in_panel = 8
    panel_edges = get_panel_edges(panel)
    dist_between_images_in_row = (panel_edges[1][0]-panel_edges[0][0])/14
    dist_between_rows_in_panel = (panel_edges[2][1] - panel_edges[0][1])/7
    center_locations = []
    for dot_j in range(rows_in_panel):
        for dot_i in range(images_in_row):
            center_locations.append((panel_edges[0][0] + (dist_between_images_in_row*dot_i),
                                        panel_edges[0][1] +  (dist_between_rows_in_panel*dot_j)))
    # plot_points_on_image(panel, center_locations)
    data = {"center_locations":center_locations}
    with open(json_path, "w") as f:
        json.dump(data, f, indent = 4)
    return center_locations

def read_fixaiton_data(fixation_path):
    """
    Reads fixation data from a specified file path and returns it as a pandas DataFrame.

    Parameters:
    - fixation_path: The file path to the fixation data, expected to be either in CSV or NPY format.

    Returns:
    - A pandas DataFrame containing the fixation data. The DataFrame will have columns: 
      TIME_STAMP, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_CSV_KEY_FIXATION.
    """
    if fixation_path.endswith(".csv"):
        data =  pd.read_csv(fixation_path)
        data = pd.DataFrame(data, columns = [TIME_STAMP, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_CSV_KEY_FIXATION])
    elif fixation_path.endswith(".npy"):
        data = np.load(fixation_path)
        time_values = data['t'].astype(float)
        vertical_values = data['y'].astype(float)
        horizontal_values = data['x'].astype(float)
        valid_values = data['status'].astype(bool)
        fixaiton_values = data['evt'].astype(int)
        fixaiton_values[fixaiton_values != FIXATION_IDX] = SACCADE_IDX
        data = np.column_stack((time_values, horizontal_values, vertical_values, valid_values, fixaiton_values))
        data = pd.DataFrame(data, columns = [TIME_STAMP, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, "status" ,FIXATION_CSV_KEY_FIXATION])
    return data

from typing import Tuple

def compair_fixation_results(
    fixation_csv_1_path_threshold: str,
    fixation_csv_2_path_ai_model: str,
    df_subset: int = -1
) -> Tuple[None]:
    """
    Compares two fixation results and plots them.

    Parameters:
    - fixation_csv_1_path_threshold: The file path to the first fixation data, expected to be in CSV format. (str)
    - fixation_csv_2_path_ai_model: The file path to the second fixation data, expected to be in CSV format. (str)
    - df_subset: The subset of the dataframes to analyze and plot. Default is to analyze the entire dataframe. (int)

    Returns:
    - None
    """
    if not (fixation_csv_1_path_threshold and fixation_csv_2_path_ai_model):
        raise ValueError("Both fixation_csv_1_path_threshold and fixation_csv_2_path_ai_model must be provided.")

    fixation_1 = read_fixaiton_data(fixation_csv_1_path_threshold)[:df_subset]
    fixation_2 = read_fixaiton_data(fixation_csv_2_path_ai_model)[: df_subset]
    fixation_1[FIXATION_CSV_KEY_FIXATION] = fixation_1[FIXATION_CSV_KEY_FIXATION].astype(int) #todo: make the data type int
    fixation_2[FIXATION_CSV_KEY_FIXATION] = fixation_2[FIXATION_CSV_KEY_FIXATION].astype(int)#todo: make the data type int
    # assert len(fixation_1) == len(fixation_2) , "Both dataframes should discribe the same panel and thereso have to be the same length"
    agreement = fixation_1[FIXATION_CSV_KEY_FIXATION] & fixation_2[FIXATION_CSV_KEY_FIXATION]
    # analyze agreement - plotting the data and drawing a bar of agreement and disagreement on the bottom
        # Create a new figure for bar representation
    
    # Create the figure and subplots
    fig, (ax_movements, ax_bars) = plt.subplots(
        2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True
    )
    
    # Plot horizontal and vertical movements
    ax_movements.plot(fixation_2[TIME_STAMP], fixation_2[FIXATION_CSV_KEY_EYE_H], label="Horizontal Movement", color='blue')
    ax_movements.plot(fixation_2[TIME_STAMP], fixation_2[FIXATION_CSV_KEY_EYE_V], label="Vertical Movement", color='red')
    ax_movements.set_ylabel("Eye Position")
    ax_movements.legend()
    ax_movements.set_title(f"Horizontal and Vertical Eye Movements fixation agreement percentage : {agreement.mean() * 100:.2f}%")
    
    # Bar height and positions for fixation results
    y_positions = [1, 2]  # Fixation 1 and Fixation 2 positions
    bar_height = 0.8

    # Create horizontal bars for fixation_1
    colors_1 = ['blue' if val == FIXATION_IDX else 'gray' for val in fixation_1[FIXATION_CSV_KEY_FIXATION]]
    ax_bars.barh(
        y_positions[0],
        fixation_1[TIME_STAMP].diff().fillna(0),
        left=fixation_1[TIME_STAMP],
        height=bar_height,
        color=colors_1,
        edgecolor='none',
        align='center',
    )

    # Create horizontal bars for fixation_2
    colors_2 = ['blue' if val == FIXATION_IDX else 'gray' for val in fixation_2[FIXATION_CSV_KEY_FIXATION]]
    ax_bars.barh(
        y_positions[1],
        fixation_2[TIME_STAMP].diff().fillna(0),
        left=fixation_2[TIME_STAMP],
        height=bar_height,
        color=colors_2,
        edgecolor='none',
        align='center',
    )

    # Add disagreement lines
    disagreement_idx = fixation_1[FIXATION_CSV_KEY_FIXATION] != fixation_2[FIXATION_CSV_KEY_FIXATION]
    for ts in fixation_1.loc[disagreement_idx, TIME_STAMP]:
        ax_bars.plot([ts, ts], y_positions, color='black', linestyle='-', linewidth=0.1)

    # Formatting for bar plot
    ax_bars.set_yticks(y_positions)
    ax_bars.set_yticklabels(['Fixation 1 - threshold', 'Fixation 2 - ai model'])
    ax_bars.set_xlabel("Time")
    ax_bars.set_title("Fixation Agreement and Disagreement")

    # Final layout adjustments
    plt.tight_layout()
    plt.show()


####### Fixation threshold finder ##########
def __fixation_finder(gaze_horizontal_deg, gaze_vertical_deg, sacc_parameters, tobii_fps=600):
    # Initialize parameters
    # 1. smooth the data points with 15 ms kernel size
    smoothing_window = int(sacc_parameters["smoothing_window"] / ((1/tobii_fps)*1000))
    smooth_kernel = np.ones(shape=(smoothing_window,)) / smoothing_window
    gaze_horizontal_deg = np.convolve(a=gaze_horizontal_deg, v = smooth_kernel, mode='same')
    gaze_vertical_deg = np.convolve(a=gaze_vertical_deg, v = smooth_kernel, mode='same')

    minimum_velocity_per_sample = sacc_parameters['saccade_min_velocity']
    maximum_angle_change = sacc_parameters['saccade_angle_threshold']

    # Tangential velocity, speed, and angle difference
    velocity_h = np.diff(gaze_horizontal_deg) # deg per (1/tobii_fps sec)
    velocity_v = np.diff(gaze_vertical_deg) # deg per  (1/tobii_fps sec)
    velocity_h_deg_sec = velocity_h * tobii_fps
    velocity_v_deg_sec = velocity_v * tobii_fps
    sample_speed_vec_per_sec = np.sqrt(velocity_h_deg_sec ** 2 + velocity_v_deg_sec ** 2)
    sample_difangle_vec_rad = np.diff(np.arctan2(np.deg2rad(gaze_vertical_deg), np.deg2rad(gaze_horizontal_deg)))
    sample_difangle_vec = np.rad2deg(np.abs(sample_difangle_vec_rad))
    movement = np.array([sample_speed_vec_per_sec, sample_difangle_vec]).T
    minimum_fixation_period = int(sacc_parameters["min_fixation_time"]*tobii_fps)
    potantial_saccade = np.where((movement[:,0] > minimum_velocity_per_sample) & (movement[:,1] < maximum_angle_change))
    saccade = np.zeros_like(gaze_horizontal_deg)
    saccade[potantial_saccade] = SACCADE_IDX
    result = []
    for i in range(len(saccade)):
        if sum(saccade[i: i+minimum_fixation_period]) == 0:
            result.append(FIXATION_IDX)
        else:
            result.append(SACCADE_IDX)
    return result


    
def generate_fixations_threshold_based(participants_data ,panel):
    fixation_array = _calculate_fixation(participants_data,panel)
    eyes_data = participants_data.matched_data[panel][KEY_TOBII_DATA]
    eyes_data[:,2] -= eyes_data[:,2][0]
    matched_data = np.concatenate((np.expand_dims(eyes_data[:,2], 1), np.expand_dims(eyes_data[:,0],1), np.expand_dims(eyes_data[:,1],1),
                                   np.expand_dims(np.ones_like(fixation_array),1),np.expand_dims(fixation_array,1)), axis=1)
    df_data = pd.DataFrame(matched_data, columns= [TIME_STAMP, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_VALID_STATUS,FIXATION_CSV_KEY_FIXATION])
    df_data.to_csv(os.path.join(participants_data.output_path, f"task_{panel}_fixation.csv"))
    return df_data

def _calculate_fixation(participants_data, panel):
    """
    returns a fixation points calculated from a given eye_data
    """
    tobii_fps = 600  # Frame rate of Tobii eye tracker

    # Saccade/Fixation parameters
    PIXEL2METER = 0.000264583  # Conversion factor for 96 DPI (pixels to meters)
    screenDistance = 0.65       # Distance from screen to participant (meters)

    sacc_parameters = {
        'saccade_min_amp': 0.08,          # Minimum amplitude of a saccade (degrees)
        'saccade_max_amp': 2,           # Maximum amplitude of a saccade (degrees)
        'saccade_min_velocity': 15,       # Minimum velocity of a saccade (degrees/sec)
        'saccade_peak_velocity': 150,      # Peak velocity of a saccade (degrees/sec)
        'saccade_min_duration': 0.009,       # Minimum duration of a saccade (seconds)
        'saccade_angle_threshold': 30.0, # Maximum angle within a saccade (degrees)
        'min_fixation_time' : 0.01, # Minimum time for a non-saccade movement to be considered as fixation
        'merge_overshoot': True,
        'overshoot_min_amp': 0.5,
        'merge_intrusions': 0,
        'intrusion_range': 300,
        "smoothing_window" : 15, # ms
        'intrusion_angle_threshold': 90.0
    }
    eye_data = participants_data.matched_data[panel][KEY_TOBII_DATA]
    img_data = participants_data.matched_data[panel][KEY_TASK_PANEL_IMG]
    gaze_horizontal_pix = eye_data[:, 0] * SCREEN_SIZE[1]
    gaze_vertical_pix = eye_data[:, 1] * SCREEN_SIZE[0]
    # Convert gaze positions from pixels to degrees
    mid_of_image = np.array(cv2.imread(img_data).shape[:2]) // 2
    horizontal_offset = gaze_horizontal_pix - mid_of_image[1]
    vertical_offset = gaze_vertical_pix - mid_of_image[0]
    gaze_horizontal_deg = np.degrees(np.arctan2(horizontal_offset * PIXEL2METER, screenDistance))
    gaze_vertical_deg = np.degrees(np.arctan2(vertical_offset * PIXEL2METER, screenDistance))

    # Call a function to detect saccades 
    panel_saccade_vec = __fixation_finder(
        gaze_horizontal_deg, gaze_vertical_deg, sacc_parameters, tobii_fps)
    return panel_saccade_vec

#####################################

def BoxMuller_gaussian(u1,u2):
  z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
  z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
  return z1,z2


if __name__=="__main__":
    threshold_based = '/Volumes/labs/ramot/rotation_students/Nitzan_K/MS/Results/Behavior/processing_results/RD707/task_l3_fixation.csv'
    model_based = '/Users/nitzankarby/Desktop/dev/Nitzan_K/MS_processing/data_for_training_gazeNet/RD707/l3_gaze_data.npy'
    compair_fixation_results(threshold_based, model_based, 3000)
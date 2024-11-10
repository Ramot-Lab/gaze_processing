from participant_gaze_data_manager import ParticipantGazeDataManager
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from constants import *
import cv2
from reliability_measurement import calculate_reliability, calculate_reliability_distribution
from visualize_data import *


KEYS_LOCATION = [0,0.142]
OUTPUT_PATH = "/Users/nitzankarby/Desktop/dev/Nitzan_K/data/processing_results"
MIN_VALID_SCORE = 20

def get_score(subject_data, plot):
    """return the subject STDM result"""
    results = {}
    for key in subject_data.matched_data.keys():
        results[key] = subject_data.matched_data[key]["strikes_score"]
    if plot:
        plt.bar(results.keys(), results.values())
    return results

def plot_data(declaration_time_map):
    names = list(declaration_time_map.keys())
    means = [declaration_time_map[name]["mean"] for name in names]
    stds = [declaration_time_map[name]["std"] for name in names]
    groups = [declaration_time_map[name]["group"] for name in names]

    # Set colors for groups
    colors = ['blue' if g == 'HC' else 'orange' for g in groups]

    # Sort data for plotting
    sorted_indices = np.argsort(groups)
    sorted_names = np.array(names)[sorted_indices]
    sorted_means = np.array(means)[sorted_indices]
    sorted_stds = np.array(stds)[sorted_indices]
    sorted_colors = np.array(colors)[sorted_indices]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_names, sorted_means, yerr=sorted_stds, color=sorted_colors, alpha=0.7)
    plt.xticks(rotation=45)
    plt.ylabel('Mean Declaration Time (ms)')
    plt.title('Mean Declaration Time by Participant Group')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend([ 'pwMS','HC'], loc='upper right', frameon=False)
    plt.tight_layout()
    plt.show()

def mean_declaration_time(data_path):
    declaration_time_map = {}
    task = "SDMT"
    for cur_group in ["pwMS", "HC"]:
        for subject_name in glob(os.path.join(data_path, cur_group, "*")):
            if not os.path.isdir(subject_name): continue
            if "SDMT" not in os.listdir(subject_name): continue
            try:
                cur= ParticipantGazeDataManager(subject_name, data_path, task,cur_group)
            except Exception as e:
                print(e)
                continue
            participant_score_list = []
            for panel_task in cur.matched_data.keys():
                panel_decliration_score = compute_declaration_time(cur, panel_task)       
                if len(panel_decliration_score) > 0:    
                    participant_score_list.append([np.median(panel_decliration_score), np.nanstd(panel_decliration_score)])
            participant_score_list = np.array(participant_score_list)
            if len(participant_score_list) > 0:
                declaration_time_map[cur.name] = {"mean" : np.median(participant_score_list[:,0]), 
                                                  "std" : np.median(participant_score_list[:,1]),
                                                  "group" : cur_group}

            
    plot_data(declaration_time_map)


def last_fixation_before_declaration(cur_subject_data, panel_name, correlated_data):
    audio_event = correlated_data['audio_event']
    audio_starts = correlated_data[audio_event == 1].index
    fixation = correlated_data.fixation
    fixation_event_starts = correlated_data[:-1][(np.array(fixation[1:]) - np.array(fixation[:-1])) == 1].index
    locations = []
    for audio_event in audio_starts:
        closest_fixation_loc = np.argmax(fixation_event_starts[fixation_event_starts < audio_event] - audio_event)
        closest_fixation = fixation_event_starts[closest_fixation_loc]
        locations.append(tuple(correlated_data[["eye_vertical_size", "eye_horizontal_size"]].iloc[closest_fixation]))
    #plot points on the panel image
    points_array = np.array(locations)
    panel_image = cv2.imread(cur_subject_data.matched_data[panel_name][KEY_TASK_PANEL_IMG])
    panel_image = cv2.imread(cur_subject_data.matched_data[panel_name][KEY_TASK_PANEL_IMG])
    img_height, img_width, _ = panel_image.shape  # Fix width and height order

    eye_x = points_array[:, 1] * img_width
    eye_y = ( points_array[:, 0]) * img_height

    # Draw fixations on the image
    for i, (x, y) in enumerate(zip(eye_x, eye_y)):
        # Draw the circle
        panel_image = cv2.circle(panel_image, (int(x), int(y)), color=(255, 0, 0), radius=1, thickness=4)
        # Add index number next to each dot
        plt.text(int(x) + 3, int(y), str(i + 1), color="red", fontsize=8)

    # Final display
    plt.imshow(panel_image)
    plt.axis("off")  # Optional: hide axes for cleaner display
    plt.show()


def validation_video_generator(cur_subject_data, panel_name, correlated_data, save_video=False):
    # Load the panel image
    panel_image_path = cur_subject_data.matched_data[panel_name][KEY_TASK_PANEL_IMG]
    panel_image = cv2.imread(panel_image_path)
    
    height, width, _ = panel_image.shape
    audio_event = correlated_data['audio_event']
    
    # Identify the start and end of audio events
    audio_starts = correlated_data[audio_event == 1].index
    audio_ends = correlated_data[audio_event == -1].index

    # Create a video writer if saving the video
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_file = 'eye_movement_video.avi'
        fps = 30  # Frames per second
        video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

    # Create a black screen frame
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate through each audio event
    for start, end in zip(audio_starts, audio_ends):
        # Select the segment of data for the current audio event
        segment = correlated_data.loc[start:end]

        # Create frames for the video
        for index, row in segment.iterrows():
            # Copy the panel image to create a frame
            frame = panel_image.copy()

            # Scale eye positions to the dimensions of the image
            eye_pos_x = int(row['eye_horizontal_size'] * width)
            eye_pos_y = int(row['eye_vertical_size'] * height)

            # Draw fixation point if fixation is active
            if row['fixation'] == 1:
                cv2.circle(frame, (eye_pos_x, eye_pos_y), 5, (0, 255, 0), -1)  # Green circle for fixation
            else:
                cv2.circle(frame, (eye_pos_x, eye_pos_y), 5, (0, 0, 255), -1)  # Red circle for non-fixation
            
            # Display the frame in real-time
            cv2.imshow('Eye Movement', frame)

            # Write the frame to the video if saving
            if save_video:
                video_writer.write(frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
                break

        # Display black screen as a separator
        for _ in range(30):  # Adjust duration (75 frames at 30 FPS is ~2.5 seconds)
            cv2.imshow('Eye Movement', black_frame)

            # Write the black frame to the video if saving
            if save_video:
                video_writer.write(black_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Release the video writer if it was created
    if video_writer:
        video_writer.release()

    # Close the display window
    cv2.destroyAllWindows()

    if save_video:
        print(f'Video saved as {video_file}')


def eye_location_in_declaration(correlated_data, panel_name, subject_name):
    audio_event = correlated_data['audio_event']
    
    # Identify the start and end of audio events
    audio_starts = correlated_data[audio_event == 1].index
    audio_ends = correlated_data[audio_event == -1].index

    # Prepare lists to store the results for plotting
    percent_looking_key_list = []
    percent_looking_board_list = []
    location_at_declaration_start_list = []
    time_at_fixation_list = []

    for start, end in zip(audio_starts, audio_ends):
        segment = correlated_data[start:end]
        eye_ver_seg = segment["eye_vertical_size"]

        # Calculate the percentages
        percent_looking_key = len(eye_ver_seg[eye_ver_seg < KEYS_LOCATION[1]]) / len(eye_ver_seg)
        percent_looking_board = 1 - percent_looking_key
        location_at_declaration_start = "up" if np.mean(eye_ver_seg[:30]) < KEYS_LOCATION[1] else "down"
        time_at_fixation = len(segment[segment["fixation"] == 0]) / len(segment)

        # Append results to lists
        percent_looking_key_list.append(percent_looking_key)
        percent_looking_board_list.append(percent_looking_board)
        location_at_declaration_start_list.append(location_at_declaration_start)
        time_at_fixation_list.append(time_at_fixation)
    avg_percent_looking_key = np.mean(percent_looking_key)
    avg_time_at_fixation = np.mean(time_at_fixation)
    return avg_percent_looking_key, avg_time_at_fixation

def plot_eye_tracking_data(percent_looking_key, percent_looking_board, location_at_declaration_start, time_at_fixation, panel_name, fig_location):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(panel_name)
    # Percent looking at key
    axs[0].plot(percent_looking_key, marker='o', color='blue')
    axs[0].set_title('Percent Looking at Key')
    axs[0].set_ylabel('Percent')
    axs[0].set_ylim(0, 1)
    axs[0].grid()

    # Location at declaration start
    axs[1].bar(range(len(location_at_declaration_start)), 
               [1 if loc == "up" else 0 for loc in location_at_declaration_start],
               color='green')
    axs[1].set_title('Location at Declaration Start (1 = Up, 0 = Down)')
    axs[1].set_ylabel('Location')
    axs[1].set_xticks(range(len(location_at_declaration_start)))
    axs[1].set_xticklabels(range(len(location_at_declaration_start)))
    axs[1].set_ylim(-0.5, 1.5)
    axs[1].grid()

    # Time spent in fixation
    axs[2].plot(time_at_fixation, marker='o', color='red')
    axs[2].set_title('Time Spent in Fixation')
    axs[2].set_ylabel('Percent')
    axs[2].set_ylim(0, 1)
    axs[2].grid()

    # Calculate and display averages
    avg_percent_looking_key = np.mean(percent_looking_key)
    avg_time_at_fixation = np.mean(time_at_fixation)

    # Add text annotation for averages
    plt.figtext(0.5, 0.95, f'Average Percent Looking at Key: {avg_percent_looking_key:.2%}', ha='center', fontsize=12)
    plt.figtext(0.5, 0.93, f'Average Time Spent in Fixation: {avg_time_at_fixation:.2%}', ha='center', fontsize=12)

    plt.savefig(os.path.join(fig_location, f"{panel_name}.png"))
    # plt.tight_layout()
    # plt.show()




def time_between_fixation_declaration(participant_data : ParticipantGazeDataManager):
    average_time_at_key = []
    average_time_at_fixation = []
    for panel_task in participant_data.matched_data.keys():
        print(f"PANEL TASK : {panel_task}")
        fixation_data = participant_data.save_fixation_to_csv(panel_task)
        audio_data = participant_data.compute_sentence_boundaries_wav(panel_task, save_csv=False, show_result=False, save_image_path = os.path.join(OUTPUT_PATH, f"{participant_data.name}"))
        if not 10 < len(audio_data) < 250:  continue
        correlated_data = participant_data.correlate_fixation_audio_in_time(fixation_data, audio_data)
        last_fixation_before_declaration(participant_data, panel_task, correlated_data)
        # validation_video_generator(participant_data, panel_task, correlated_data)
        cur_avg_percent_looking_key, cur_avg_time_at_fixation = eye_location_in_declaration(correlated_data, panel_task, participant_data.name)
        average_time_at_key.append(cur_avg_percent_looking_key)
        average_time_at_fixation.append(cur_avg_time_at_fixation)
    plt.clf()
    plt.bar(list(participant_data.matched_data.keys()), average_time_at_key)
    plt.title(f"{participant_data.name} average time spent on key")
    plt.ylabel("time spent on key (percent) out of total declaration time")
    plt.xlabel("panel name")
    plt.savefig(f"{participant_data.name}_average_time_at_key")
    plt.clf()
    plt.bar(list(participant_data.matched_data.keys()), average_time_at_fixation)
    plt.title(f"{participant_data.name} average time at fixation")
    plt.ylabel("time spent on fixation (percent) out of total declaration time")
    plt.xlabel("panel name")
    plt.savefig(f"{participant_data.name}_average_time_at_fixation")
    
def compute_declaration_time(cur_subject : ParticipantGazeDataManager, panel_name):
    computed_declaration_answer = cur_subject.compute_sentence_boundaries_wav(panel_name, False, False)
    panel_decliration_score = []
    for sentence_i in range(len(computed_declaration_answer)//2):
        panel_decliration_score.append(computed_declaration_answer["time_stamp"].iloc[sentence_i+1] - 
                                    computed_declaration_answer["time_stamp"].iloc[sentence_i])
    return panel_decliration_score

def calculate_reliability_of_declaration_time(data_path, task, minimal_declaration_count = 25):
    PANEL_AMOUNT = 6
    participants_results_matrix = []
    for cur_group in ["pwMS", "HC"]:
        for subject_name in glob(os.path.join(data_path, cur_group, "*")):
            if not os.path.isdir(subject_name): continue
            if "SDMT" not in os.listdir(subject_name): continue
            try:
                cur= ParticipantGazeDataManager(subject_name, data_path, task, cur_group)
            except:
                print(f"an error with processing subject : {subject_name}")
            #get time duration samples
            declaration_time_samples = [] #Flat vector of all participant's panel results
            number_of_panels = 0
            for panel_name in cur.matched_data.keys():
                panel_declaration_score = compute_declaration_time(cur, panel_name)
                if len(panel_declaration_score) > minimal_declaration_count:
                    number_of_panels += 1
                    declaration_time_samples.extend(panel_declaration_score)
            if number_of_panels == 6:
                participants_results_matrix.append(np.array(declaration_time_samples))
    distribution = calculate_reliability_distribution(participants_results_matrix, 39, 40, 10000, minimal_declaration_count*PANEL_AMOUNT)
    plot_barplot(list(range(20,60)) ,distribution, "L value","reliability value", "reliability distribution over different L values")
    

def distance_from_target_symbol_analysis(cur_data : ParticipantGazeDataManager):
    """
    computes the distance of the fixation point from the target location for participant's all panels 
    """
    results = []
    panel_path = cur_data.matched_data["0"][KEY_TASK_PANEL_IMG]
    vertical_size, horizontal_size= cv2.imread(panel_path).shape[:2]
    for panel_task_name in cur_data.matched_data.keys():
        score = cur_data.matched_data[panel_task_name][KEY_STRIKE_SCORE]
        audio_data = cur_data.compute_sentence_boundaries_wav(panel_task_name, save_csv=False, show_result=True)
        if not 0.8 <((len(audio_data)//2) / score) < 1.2 :
            print(len(audio_data)//2, score, f"{cur.name} {panel_task_name}")
            continue
        fixation_data = cur_data.save_fixation_to_csv(panel_task_name)
        correlated_data = cur_data.correlate_fixation_audio_in_time(fixation_data, audio_data)
        correlated_data[[FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V]] = correlated_data[[FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V]] * np.array([horizontal_size, vertical_size])
        audio_event = correlated_data['audio_event']
        audio_starts = correlated_data[audio_event == 1].index
        prior_event = 0
        #calculate the distance from the target figure when the subject last looked at it
        for declaration_i, start_index in enumerate(audio_starts):
            target_index = panel_center_locations[declaration_i]
            segment = correlated_data[(correlated_data.index >prior_event) & (correlated_data.index < start_index)]
            fixation_distance = get_closest_fixation_distance(segment, target_index)
            results.append(fixation_distance)
            prior_event = start_index
    return results

def get_closest_fixation_distance(correlated_data, target):
    target = np.array(target)
    segments_fixation = np.diff(correlated_data["fixation"]) 
    start = correlated_data[:-1][segments_fixation==1].index
    end = correlated_data[:-1][segments_fixation==-1].index
    best_dist = np.inf
    if correlated_data["fixation"].iloc[0] == 1:
        start = np.concatenate([[0],start])
    for s, e in zip(start, end):
        dist = np.linalg.norm((correlated_data[s:e][[FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V]].mean(axis=0).values - target))
        if dist < best_dist:
            best_dist = dist
    return best_dist



if __name__=="__main__":
    data_path = "/Users/nitzankarby/Desktop/dev/Nitzan_K/data"
    # mean_declaration_time(data_path)
    cur= ParticipantGazeDataManager("RD707", data_path, "SDMT", "pwMS")
    # distance_from_target_symbol_analysis(cur)
    # time_between_fixation_declaration(cur)
    task = "SDMT"
    calculate_reliability_of_declaration_time(data_path, task)
    # for cur_group in ["pwMS", "HC"]:
    #     print(f"############### {cur_group} ###############")
    #     for subject_name in glob(os.path.join(data_path, cur_group, "*")):
    #         if not os.path.isdir(subject_name): continue
    #         if "SDMT" not in os.listdir(subject_name): continue
    #         try:
    #             cur= ParticipantGazeDataManager(subject_name, data_path, task, "pwMS")
    #             time_between_fixation_declaration(cur)
    #     #         cur.save_fixation_to_csv("0")
    #     #         cur.compute_sentence_boundaries_wav("0", True)
    #     #         break
    #         except Exception as ex:
    #             print(subject_name)
    #             continue
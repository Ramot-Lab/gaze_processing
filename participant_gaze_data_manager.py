import scipy.io as scio
import numpy as np
import pandas as pd
import os
from glob import glob
import cv2
import librosa
from constants import *
from utils import *
import matplotlib.pyplot as plt
from datetime import datetime
import re

class ParticipantGazeDataManager:
    def __init__(self, participant_name, main_data_path, task = "SDMT", participant_group = "pwMS") -> None:
        group_path, self.name = os.path.split(participant_name)
        _,  self.group = os.path.split(group_path)
        tobii_data, task_png, audio_recordings = self.load_data(participant_name , main_data_path, task, participant_group)
        self.matched_data = {}
        self.main_data_path = main_data_path
        self.output_path = os.path.join(main_data_path, "processing_results", self.name)
        os.makedirs(self.output_path, exist_ok=True)
        for mat_file in tobii_data:
            self.task_data, self.messages, self.gaze_data, self.presentation_info, self.dom_Eye = self.prepare_gaze_data_for_preprocessing(mat_file)
            self.matched_data = {**self.matched_data, **self.group_task_info(self.gaze_data, task_png, audio_recordings, self.task_data, mat_file)}
        

    def get_creation_time(self, tobii_data):
        header = str(tobii_data["__header__"])
        # Extract the date part using regex
        expression_object = re.search(r'\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \d{4}', header)
        if expression_object is not None:
            date_str = expression_object.group()
            date_obj = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")
        else:
            date_obj = datetime.strptime(OLD_MIC_REPLACEMENT_DATE, "%Y-%m-%d")
        return date_obj

    def load_data(self, participant_code_name, main_data_path, task, participant_group):
        task_files_path = os.path.join(main_data_path, participant_group, participant_code_name, task)
        if not os.path.exists(task_files_path):
            raise f"no such path {task_files_path}"
        mat_files = glob(os.path.join(task_files_path, "*.mat"))
        mat_files = [file_path for file_path in mat_files if "run" in os.path.split(file_path)[1].lower()]
        task_png_files = glob(os.path.join(main_data_path, "panels_images", "*.jpg"))
        recording_files = glob(os.path.join(task_files_path, "*.wav"))
        return [scio.loadmat(mat, struct_as_record=False, squeeze_me=True) for mat in mat_files], task_png_files, recording_files
    

    def group_task_info(self, tobii_data_file, task_png, audio_recordings, task_data, mat_file):
        matched_data_files = {}
        for i, task_name in enumerate(list(task_data.keys())[1::2]):
            task_code = (task_name[-2:]).replace("_", "")
            task_code_lower = task_code.lower()
            png_img = [img for img in task_png if img.endswith(f"_{task_code_lower}.jpg")][0]
            audio_file = [audio for audio in audio_recordings if os.path.split(audio)[1].split("_")[2].lower() == task_code_lower][0]
            matched_data_files[task_code_lower] = {KEY_TOBII_DATA: self.clean_outliers(tobii_data_file[f"panel_{i+1}"]),
                                                   KEY_TASK_PANEL_IMG : png_img,
                                                   KEY_AUDIO_DATA :  audio_file,
                                                   KEY_STRIKE_SCORE : task_data[f"strikes_img_test_{task_code}"],
                                                   KEY_RECORDING_DATE : self.get_creation_time(mat_file)}
        return matched_data_files
            

        
    def prepare_gaze_data_for_preprocessing(self, data):
        # Extract left and right gaze data
        left_gaze = data['data'].gaze.left.gazePoint.onDisplayArea
        right_gaze = data['data'].gaze.right.gazePoint.onDisplayArea
        tobi_ts = data["data"].gaze.systemTimeStamp

        # Extract messages
        messages = data['messages']

        # Find indices for presentation times of each panel and break
        panel_indices = []
        break_indices = []
        for i, message in enumerate(messages):
            if message[1] in ['panel number 1', 'panel number 2', 'panel number 3']:
                panel_indices.append(i)
            elif message[1] in ['break panel number 1', 'break panel number 2', 'finished']:
                break_indices.append(i)

        # Ensure we found the expected number of indices
        assert len(panel_indices) == 3, f'Expected 3 panel indices, but found {len(panel_indices)}'
        assert len(break_indices) == 3, f'Expected 3 break indices, but found {len(break_indices)}'

        # Extract system timestamps for the start of each panel and break
        panel_start_times = [messages[i][0] for i in panel_indices]
        break_start_times = [messages[i][0] for i in break_indices]

        # Find the indices in the gaze data corresponding to each panel's presentation
        panel_presentation_indices = []
        for i in range(3):
            indices = np.where((data['data'].gaze.systemTimeStamp > panel_start_times[i]) & 
                            (data['data'].gaze.systemTimeStamp < break_start_times[i]))[0]
            panel_presentation_indices.append(indices)

        # Validate Dom_Eye
        Dom_Eye = data['Dom_Eye']
        assert Dom_Eye in ['r', 'l'], 'Dom_Eye must be either "r" or "l"'

        # Select gaze data for each panel based on the dominant eye
        gaze_data = {}
        for i in range(3):
            if Dom_Eye == 'r':
                panel_data = np.concatenate((right_gaze[:, panel_presentation_indices[i]].T, np.reshape(tobi_ts[panel_presentation_indices[i]], (-1,1))), axis=1)
                if (np.count_nonzero(np.isnan(panel_data))//2) / len(panel_data) < MAX_VALID_NAN_VALUES:
                    gaze_data[f'panel_{i+1}'] = panel_data
            else:
                panel_data = np.concatenate((left_gaze[:, panel_presentation_indices[i]].T, np.reshape(tobi_ts[panel_presentation_indices[i]], (-1,1))), axis=1)
                if (np.count_nonzero(np.isnan(panel_data))//2) / len(panel_data) < MAX_VALID_NAN_VALUES:
                    gaze_data[f'panel_{i+1}'] = panel_data

        # Calculate presentation durations
        presentation_info = {}
        for i in range(3):
            presentation_info[f'panel_{i+1}'] = {
                'start_time': panel_start_times[i],
                'end_time': break_start_times[i],
                'duration': break_start_times[i] - panel_start_times[i]
            }

        # Extract time_of_slides
        task_data = data['task_data'].__dict__
        return task_data, messages, gaze_data, presentation_info, Dom_Eye
    
    def compute_sentence_boundaries_wav(self, panel, save_csv = False, show_result = False, save_image_path = ""):
        audio_path = self.matched_data[panel]["audio_data"]
        signal, sr = librosa.load(audio_path, sr=None)
        signal_shape = len(signal)
        
        # Compute RMS energy
        frame_length = 4096*2
        hop_length = 1024*2
        rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
        median_signal = np.median(abs(signal))
        if median_signal > 0.01:
            print(f"old microphone date, impossible to process panel: {panel} subject: {self.name}, date: {self.matched_data[panel][KEY_RECORDING_DATE]}")
            return []
        # Normalize the RMS values
        rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

        # Detect speech based on a threshold
        threshold = np.median(rms_normalized)
        silence_indices = rms_normalized < threshold
        rms_normalized[silence_indices]= 0
        rms_normalized[rms_normalized!=0] = 1
        rms_normalized_borders = np.diff(rms_normalized) 

        # Translate frame indices to sample indices
        start_indices = np.where(rms_normalized_borders == 1)[0] * hop_length
        end_indices = np.where(rms_normalized_borders == -1)[0] * hop_length
        start_indices_cleaned = []
        end_indices_cleaned = []
        min_speaking_time = 0.4 * sr
        for s, e, in zip(start_indices, end_indices):
            if e - s < min_speaking_time:
                continue
            else:
                start_indices_cleaned.append(s)
                end_indices_cleaned.append(e)
        time_array = (np.arange(signal_shape) / sr) * 1000000  # translate to miliseconds
        time_array = np.round((time_array), decimals=5) 

        output_map = []
        for s, e in zip(start_indices_cleaned, end_indices_cleaned):
            output_map.append([time_array[s], s, 1])
            output_map.append([time_array[min(e, len(time_array)-1)], e, -1])

        out = pd.DataFrame(output_map, columns=[TIME_STAMP, SIGNAL_IDX, SENTENCE_BREAK])
        if save_csv:
            out.to_csv(os.path.join(self.output_path, f"task_{panel}_audio_preprocess.csv"))
        if show_result or len(save_image_path) > 0:
            plt.plot(signal)
            for i, j in zip(start_indices_cleaned, end_indices_cleaned):
                plt.vlines(x=i, ymin = -0.15, ymax=0.15, color = "g")
                plt.vlines(x=j, ymin = -0.15, ymax=0.15, color = "r")
            # Create a time array based on sample rate
            if show_result:
                plt.title(f"panel: {panel} subject: {self.name}, date: {self.matched_data[panel][KEY_RECORDING_DATE]}")
                plt.show()
            if len(save_image_path) > 0:
                plt.savefig(os.path.join(save_image_path, f"{panel}_separation.png"))

        return out

    
    def nan_helper(self, x):
        return np.isnan(x), lambda z: z.nonzero()[0]
    
    def interpulate_nan_values(self, eye):
        nans, eye_temp = self.nan_helper(eye)
        if sum(nans) == len(nans):
            raise Exception("only nan values detected")
        eye[nans] = np.interp(eye_temp(nans), eye_temp(~nans), eye[~nans])
        return eye
    
    def clean_outliers(self, eyes):
        eyes[:, 0] = self.clean_outliers_single_eye(eyes[:, 0])
        eyes[:, 1] = self.clean_outliers_single_eye(eyes[:, 1])
        return eyes

    def clean_outliers_single_eye(self, eye):
        eye_movment_l  = abs(eye[1:]-eye[:-1])
        eye_movment_r  = abs(eye[:-1]-eye[1:])
        outlier_cutoff = np.mean([np.nanmean(eye_movment_l), np.nanmean(eye_movment_r)]) + (2*np.mean([np.nanstd(eye_movment_l), np.nanstd(eye_movment_r)]))
        outlier_values_l = (eye_movment_l > outlier_cutoff)
        outlier_values_r = (eye_movment_r > outlier_cutoff)
        outlier_values = outlier_values_l & outlier_values_r
        outlier_values = np.concatenate(([False], outlier_values))
        eye[outlier_values] = None
        return self.interpulate_nan_values(eye)
    

    def fixation_finder(self, gaze_horizontal_deg, gaze_vertical_deg, sacc_parameters, tobii_fps=600):
        # Initialize parameters
        saccade_vec = np.zeros((3, int(len(gaze_horizontal_deg))))  # Max number of saccades = 10/s
        # 1. smooth the data points with 15 ms kernel size
        smoothing_window = int(sacc_parameters["smoothing_window"] / ((1/tobii_fps)*1000))
        smooth_kernel = np.ones(shape=(smoothing_window,)) / smoothing_window
        gaze_horizontal_deg = np.convolve(a=gaze_horizontal_deg, v = smooth_kernel, mode='same')
        gaze_vertical_deg = np.convolve(a=gaze_vertical_deg, v = smooth_kernel, mode='same')

        minimum_velocity_per_sample = sacc_parameters['saccade_min_velocity']
        maximum_angle_change = sacc_parameters['saccade_angle_threshold']
        overshoot_max_gap = int(10 * tobii_fps / 1000)  # 10ms
        minimum_sacade_time_sec = int(9 / ((1/tobii_fps)*1000)) + 1

        # Tangential velocity, speed, and angle difference
        velocity_h = np.diff(gaze_horizontal_deg) # deg per (1/tobii_fps sec)
        velocity_v = np.diff(gaze_vertical_deg) # deg per  (1/tobii_fps sec)
        velocity_h_deg_sec = velocity_h * tobii_fps
        velocity_v_deg_sec = velocity_v * tobii_fps
        sample_speed_vec_per_sec = np.sqrt(velocity_h_deg_sec ** 2 + velocity_v_deg_sec ** 2)
        sample_difangle_vec_rad = np.diff(np.arctan2(np.deg2rad(gaze_vertical_deg), np.deg2rad(gaze_horizontal_deg)))
        sample_difangle_vec = np.rad2deg(np.abs(sample_difangle_vec_rad))
        movement = np.array([sample_speed_vec_per_sec, sample_difangle_vec]).T
        potantial_saccade = np.where((movement[:,0] > minimum_velocity_per_sample) & (movement[:,1] < maximum_angle_change))
        saccade = np.zeros_like(gaze_horizontal_deg)
        saccade[potantial_saccade] = 1
        result = []
        for i in range(len(saccade)):
            if len(saccade[i: i+minimum_sacade_time_sec]) == sum(saccade[i: i+minimum_sacade_time_sec]):
                result.append(0)
            else:
                result.append(1)
        return result

    
        
    def save_fixation_to_csv(self, panel):
        fixation_array = self.calculate_fixation(panel)
        eyes_data = self.matched_data[panel][KEY_TOBII_DATA]
        eyes_data[:,2] -= eyes_data[:,2][0]
        matched_data = np.concatenate((np.expand_dims(eyes_data[:,2], 1), np.expand_dims(eyes_data[:,0],1), np.expand_dims(eyes_data[:,1],1), np.expand_dims(fixation_array,1)), axis=1)
        df_data = pd.DataFrame(matched_data, columns= [TIME_STAMP, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_CSV_KEY_FIXATION])
        df_data.to_csv(os.path.join(self.output_path, f"task_{panel}_fixation.csv"))
        return df_data

    def calculate_fixation(self, panel):
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
            'merge_overshoot': True,
            'overshoot_min_amp': 0.5,
            'merge_intrusions': 0,
            'intrusion_range': 300,
            "smoothing_window" : 15, # ms
            'intrusion_angle_threshold': 90.0
        }
        eye_data = self.matched_data[panel][KEY_TOBII_DATA]
        img_data = self.matched_data[panel][KEY_TASK_PANEL_IMG]
        gaze_horizontal_pix = eye_data[:, 0] * 1920
        gaze_vertical_pix = eye_data[:, 1] * 1080
        # Convert gaze positions from pixels to degrees
        mid_of_image = np.array(cv2.imread(img_data).shape[:2]) // 2
        horizontal_offset = gaze_horizontal_pix - mid_of_image[1]
        vertical_offset = gaze_vertical_pix - mid_of_image[0]
        gaze_horizontal_deg = np.degrees(np.arctan2(horizontal_offset * PIXEL2METER, screenDistance))
        gaze_vertical_deg = np.degrees(np.arctan2(vertical_offset * PIXEL2METER, screenDistance))

        # Call a function to detect saccades (Assumed function `saccade_workhorse_kd`)
        panel_saccade_vec = self.fixation_finder(
            gaze_horizontal_deg, gaze_vertical_deg, sacc_parameters, tobii_fps)
        return panel_saccade_vec


    def correlate_fixation_audio_in_time(self, fixation_df, audio_df):
        """
        matches the audio events onto the fixation events.
        returns a df of <time_ms, dom_eye_x, dom_eye_right, fixation, audio event> 
        """
        audio_to_concat = np.zeros((len(fixation_df),1))
        for idx, time_ms in enumerate(audio_df[TIME_STAMP]):
            event_time_idx = np.argmin(abs(fixation_df[TIME_STAMP] - time_ms))
            audio_to_concat[event_time_idx] = audio_df[SENTENCE_BREAK].iloc[idx]
        audio_to_concat_df = pd.DataFrame(audio_to_concat, columns=["audio_event"])
        return pd.concat((fixation_df, audio_to_concat_df), axis=1)
    
            
if __name__=="__main__":
    from visualize_data import show_running_video_live
    # panel_path = "/Users/nitzankarby/Desktop/dev/Nitzan_K/data/panels_images/panel_a5.jpg"
    # dots_on_panel_location = map_panel_into_dot_points(panel_path)
    # print(dots_on_panel_location)
    p_name = "YS875"
    task = "SDMT"
    group = "pwMS"
    # panel = "i1"
    data_path = "/Users/nitzankarby/Desktop/dev/Nitzan_K/data"
    subject_data = ParticipantGazeDataManager(p_name, data_path, task, group)
    # # fixation_data = subject_data.save_fixation_to_csv(panel)
    for panel in subject_data.matched_data.keys():
        audio_data = subject_data.compute_sentence_boundaries_wav(panel, save_csv=False, show_result=True)
    # # correlated_data = subject_data.correlate_fixation_audio_in_time(fixation_data, audio_data)
    # # data = pd.read_csv("/Users/nitzankarby/Desktop/dev/Nitzan_K/data/processing_results/AA562/task_0_fixation.csv")
    # img_path = subject_data.matched_data[panel][KEY_TASK_PANEL_IMG]
    # show_running_video_live(data, img_path)

import scipy.io as scio
import numpy as np
import pandas as pd
import os
from glob import glob
import torch
import librosa
from constants import *
from utils import *
import matplotlib.pyplot as plt
from datetime import datetime
import GazeModel.model as gazeNET_model
from GazeModel.recording_data_runner import RecordingDataRunner, generate_fixation_model_based
import re

class ParticipantGazeDataManager:
    def __init__(self, participant_name, main_data_path, task = "SDMT", participant_group = "pwMS", clean_gaze_data = True) -> None:
        group_path, self.name = os.path.split(participant_name)
        _,  self.group = os.path.split(group_path)
        self.task = task
        self.task_validation_filter_param = "run" if task == "SDMT" else "kd"
        tobii_data, task_png, audio_recordings = self.load_data(participant_name , main_data_path, task, participant_group)
        self.matched_data = {}
        self.main_data_path = main_data_path
        self.output_path = os.path.join(main_data_path, "processing_results", self.name)
        self.model = None
        os.makedirs(self.output_path, exist_ok=True)
        for mat_file in tobii_data:
            self.task_data, self.messages, self.gaze_data, self.presentation_info, self.dom_Eye = self.prepare_gaze_data_for_preprocessing(mat_file)
            self.matched_data = {**self.matched_data, **self.group_task_info(self.gaze_data, task_png, audio_recordings, self.task_data, mat_file, clean_gaze_data)}
        

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
        mat_files = [file_path for file_path in mat_files if self.task_validation_filter_param in os.path.split(file_path)[1].lower()]
        task_png_files = glob(os.path.join(main_data_path, "panels_images", task,"*.jpg"))
        recording_files = glob(os.path.join(task_files_path, "**","*.wav"), recursive=True)
        return [scio.loadmat(mat, struct_as_record=False, squeeze_me=True) for mat in mat_files], task_png_files, recording_files
    

    def group_task_info(self, tobii_data_file, task_png, audio_recordings, task_data, mat_file, clean_gaze_data):
        matched_data_files = {}
        audio_idx = 2 if self.task == "SDMT" else 1
        for i, task_name in enumerate(list(task_data.keys())[1::2]):
            if f"panel_{i+1}" not in tobii_data_file.keys(): continue
            task_code = (task_name[-2:]).replace("_", "")
            task_code_lower = task_code.lower()
            png_img = [img for img in task_png if img.endswith(f"_{task_code_lower}.jpg")][0]
            audio_files = [audio for audio in audio_recordings if task_code_lower in os.path.split(audio)[1].split("_")[audio_idx].lower()]
            if len(audio_files) == 0: continue
            audio_file = audio_files[0]
            preprocess_gaze_method = self.clean_outliers if clean_gaze_data else self.clean_outliers_no_interpolation
            matched_data_files[task_code_lower] = {KEY_TOBII_DATA: preprocess_gaze_method(tobii_data_file[f"panel_{i+1}"]),
                                                   KEY_TASK_PANEL_IMG : png_img,
                                                   KEY_AUDIO_DATA :  audio_file,
                                                   KEY_STRIKE_SCORE : 0 if list(task_data.keys())[0] == "dummy" else task_data[f"strikes_img_test_{task_code}"],
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
        panel_indices , break_indices = self.break_mat_into_pannels(messages)
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
        assert Dom_Eye.lower() in ['r', 'l'], 'Dom_Eye must be either "r" or "l"'

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
        if self.task == "KD":
            task_data = {"dummy": None, "1":None, "0":None,"2": None, "00":None, "3":None, "000":None} # TODO: how to get the real task results.
        elif self.task == "SDMT":
            task_data = data["task_data"].__dict__
        return task_data, messages, gaze_data, presentation_info, Dom_Eye
    
    def break_mat_into_pannels(self, mat_file_messages):
        panel_indices = []
        break_indices = []
        if self.task == "SDMT":
            panel_names = [f"panel number {i+1}" for i in range(3)]
            break_names = [f"break panel number {i+1}" for i in range(3)] + ["finished"]
        elif self.task == "KD":
            panel_names = [f"slide {(2*i)+1}" for i in range(3)]
            break_names = [f"slide {(2*i)+2}" for i in range(2)] + ["done with slides"]
        for i, message in enumerate(mat_file_messages):
            if message[1] in panel_names:
                panel_indices.append(i)
            elif message[1] in break_names:
                break_indices.append(i)
        return panel_indices, break_indices

    def compute_sentence_boundaries_wav(self, panel, save_csv=False, show_result=False, save_image_path=""):
        audio_path = self.matched_data[panel]["audio_data"]
        signal, sr = librosa.load(audio_path, sr=None)
        signal_shape = len(signal)
        
        # Compute RMS energy
        frame_length = 4096 * 2  # the window size within the average calculation
        hop_length = 1024*2  # step size of the windows
        rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Check microphone quality
        median_signal = np.median(abs(signal))
        if median_signal > 0.01:
            print(f"Old microphone data, cannot process panel: {panel} subject: {self.name}, date: {self.matched_data[panel][KEY_RECORDING_DATE]}")
            return []
        
        rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

        # Detect speech based on a threshold
        threshold = np.median(rms_normalized) + (np.std(rms_normalized) / 4)
        silence_indices = rms_normalized < threshold
        rms_normalized[silence_indices] = 0
        rms_normalized[rms_normalized != 0] = 1
        rms_normalized_borders = np.diff(rms_normalized)

        # Translate frame indices to sample indices
        start_indices = np.where(rms_normalized_borders == 1)[0] * hop_length
        end_indices = np.where(rms_normalized_borders == -1)[0] * hop_length

        # Handle missing last end index if the last frame is a speech frame
        if len(start_indices) > len(end_indices):
            end_indices = np.append(end_indices, signal_shape - 1)  # append last index of signal if unclosed

        min_speaking_time = 0.25 * sr # 0.25 * sec
        time_array = (np.arange(signal_shape) / sr) * 1000000  # translate to milliseconds
        time_array = np.round(time_array, decimals=5)

        output_map = []
        diff_array_start_end = end_indices - start_indices[:len(end_indices)]  # filter out short noises
        declaration_diff_array_filter = diff_array_start_end < min_speaking_time
        start_indices = start_indices[:len(end_indices)][~declaration_diff_array_filter]
        end_indices = end_indices[~declaration_diff_array_filter]

        for s, e in zip(start_indices, end_indices):
            output_map.append([time_array[s], s, 1])
            output_map.append([time_array[min(e, len(time_array) - 1)], e, -1])

        out = pd.DataFrame(output_map, columns=[TIME_STAMP, SIGNAL_IDX, SENTENCE_BREAK])
        
        if save_csv:
            out.to_csv(os.path.join(self.output_path, f"task_{panel}_audio_preprocess.csv"))
        if show_result or len(save_image_path) > 0:
            plt.plot(signal)
            for i, j in zip(start_indices, end_indices):
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
    
    def clean_outliers_no_interpolation(self,gaze_data):
        gaze_data[:, 0] = self.clean_outliers_single_eye(gaze_data[:, 0], False)
        gaze_data[:, 1] = self.clean_outliers_single_eye(gaze_data[:, 1], False)
        return gaze_data
        

    def clean_outliers(self, eyes):
        eyes[:, 0] = self.clean_outliers_single_eye(eyes[:, 0])
        eyes[:, 1] = self.clean_outliers_single_eye(eyes[:, 1])
        return eyes
    
    def clean_outliers_nan_removal(self, eyes):
        eyes = eyes[~np.isnan(eyes[:,0])] 
        return eyes

    def clean_outliers_single_eye(self, eye, interpolate = True):
        eye_movment_l  = abs(np.diff(eye))
        outlier_cutoff = (np.nanstd(eye_movment_l)*8) + np.nanmean(eye_movment_l)
        outlier_values_l = (eye_movment_l > outlier_cutoff)
        outlier_values = np.concatenate(([False], outlier_values_l))
        eye[outlier_values] = None
        if interpolate == False:
            return eye
        return self.interpulate_nan_values(eye)

    def annotate_gaze_events(self, annotation_method, panel):
        if annotation_method == "threshold_based":
            return generate_fixations_threshold_based(self, panel)
        elif annotation_method == "model_based":
            if self.model is None:
                self.load_model()
            return generate_fixation_model_based(self.matched_data[panel][KEY_TOBII_DATA], model = self.model)
        else:
            raise Exception(f"unknown annotation method: {annotation_method} choose from ['threshold_based', 'model_based']")

    def load_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_classes = len(ModelPropertise.EVENTS.value)
        self.model = gazeNET_model.gazeNET(num_classes)
        self.model, _ = gazeNET_model.load(self.model, ModelPropertise.MODEL_PATH.value)
        self.model = self.model.to(self.device)
        self.model.eval()

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
    p_name = "RD707"
    task = "SDMT"
    group = "pwMS"
    panel = "3"
    panel_path = "/Users/nitzankarby/Desktop/dev/Nitzan_K/data/panels_images/panel_a5.jpg"
    data_path = "/Volumes/labs/ramot/rotation_students/Nitzan_K/MS/Results/Behavior"
    subject_data= ParticipantGazeDataManager(p_name, data_path, "KD", group)
    print(subject_data.annotate_gaze_events("model_based", panel)[:20])

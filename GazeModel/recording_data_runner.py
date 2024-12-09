from GazeModel import model
import json
import numpy as np
from data_processing.data_loader import EMDataset, GazeDataLoader
import torch
import pandas as pd
from constants import *
from torch.autograd import Variable
import copy
from data_processing.training_npy_generator import Recoring2GazeNetProcessor


class RecordingDataRunner:
    def __init__(self, model ,config = "GazeModel/config.json"):
        with open(config, 'r') as f:
            self.config = json.load(f)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        #%% prepare model
        

    def run_inference(self, recording_data):
        """
        Runs inference on the provided recording data.

        Args:
            recording_data: Gaze recording data to run inference on. Should be a list of lists, where
                each sublist contains the data for a single recording, in the form of:
                [[horizontal_gaze, verical_gaze], ...]

        Returns:
            A list of lists, where each sublist contains the output of the model for a single recording.
            The output is of the form [[fixation, saccade, ...], ...].
        """
        test_loader, test_dataset = self._prepare_recording_data(recording_data)
        dataset_copy = copy.deepcopy(test_dataset.data)
        if test_loader is None:
            return None
        for data, _dataset in zip(test_loader, dataset_copy):
            inputs, _, _, target_sizes, _ = data
            inputs = Variable(inputs).contiguous()
            y = self.model(inputs)
            if self.device == "cuda":
                inputs = inputs.cuda()
                y_ = y_.cuda()
            outputs_split = [_y[:_l] for _y, _l in zip(y.data, target_sizes)]
            _dataset["evt"] = 0
            _dataset["evt"][1:] = np.argmax(outputs_split[0], axis=1)+1
        dataset_copy = pd.concat([pd.DataFrame(_d) for _d in dataset_copy]).reset_index(drop=True)
        _data = pd.DataFrame(recording_data)
        _data = _data.merge(dataset_copy, on='t', suffixes=('', '_pred'), how='left')
        _data['evt'] = _data['evt_pred'].replace({np.nan:0})

        return _data[["t","x_original","y_original","status","evt"]]
    
        
    def apply_prediction_to_recording_data(self, recording_data):
        """
        Applies inference results to the given recording data.

        This function uses the `run_inference` method to obtain predictions for the provided
        gaze recording data and updates the recording data in place with these predictions.

        Args:
            recording_data: A list of lists where each sublist contains the data for a single
                recording in the form of [[horizontal_gaze, vertical_gaze], ...].

        Returns:
            None if the inference results are None; otherwise, updates `recording_data` in place
            with the inference results, where each sublist is replaced with the corresponding
            prediction results.
        """
        recording_data = self.align_data_type(recording_data)
        recording_data = self.run_inference(recording_data)
        return recording_data

    def align_data_type(self, tobii_data):
        """
        tobii_data : a 2d array of (t,x,y,valid) by N array. raw signal from tobii. assumes x, y are in [0,1].
        """
        #check if tobii data have been processed or needs processing
        if tobii_data.dtype.names is not None: #if already processed
            reconstacted_data = tobii_data
        else: #if needs processing
            reconstacted_data_maker = Recoring2GazeNetProcessor(tobii_data)
            reconstacted_data = reconstacted_data_maker.get_processed_data()
        return reconstacted_data

    def _prepare_recording_data(self, recording_data):
        """
        recording_data : a 2d array of (t,x,y,valid) by N array. raw signal from tobii. assumes x, y are in degrees        
        """
        sample_size = recording_data.shape[0]
        if sample_size < self.config['seq_len']:
            return None
        else:
            test_dataset = EMDataset(config = self.config, gaze_data = [recording_data])
            test_loader = GazeDataLoader(test_dataset, batch_size=1,
                                        num_workers=1,
                                        shuffle=False)
            return test_loader, test_dataset

def generate_fixation_model_based(recording_data, model):
    recording_ranner = RecordingDataRunner(model)
    annotated_data = recording_ranner.apply_prediction_to_recording_data(recording_data)
    annotated_data.columns = ["t", FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_VALID_STATUS,FIXATION_CSV_KEY_FIXATION]
    return annotated_data
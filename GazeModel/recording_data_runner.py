from GazeModel import model
import json
import numpy as np
import os
from data_processing.data_loader import EMDataset, GazeDataLoader
import torch
import pandas as pd
from constants import *
from torch.autograd import Variable
import copy
from data_processing.training_npy_generator import Recoring2GazeNetProcessor

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class RecordingDataRunner:
    def __init__(self, model, config = DEFAULT_CONFIG_PATH):
        with open(config, 'r') as f:
            self.config = json.load(f)
        # config.json's "augment" adds random Gaussian noise (up to 0.495 deg RMS per
        # 201-sample window, data_loader.EventParser.parse_data) - a training-time
        # augmentation (Zemblys, Niehorster & Holmqvist, 2019, Behav Res Methods 51:840-864).
        # At inference it only corrupts the real signal, so it is off here (decision
        # 2026-11-10); config.json itself is unchanged, so training still augments.
        self.config['augment'] = False
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
        # EMDataset builds overlapping seq_len+1 windows (stride_step apart) so the diff-based
        # input has +1 sample of context; consecutive windows therefore share one raw sample
        # (previous window's last row == next window's row 0), which is always left at the
        # default evt=0 ("no prediction yet", see _dataset["evt"][1:]=... above). Concatenating
        # every window's rows thus produces one duplicate 't' per window boundary: a real
        # (nonzero) prediction from the earlier window and a placeholder 0 from the later one.
        # Left-merging _data against dataset_copy while those duplicates exist fans out into
        # extra output rows (one per match), silently inflating the returned length beyond
        # len(recording_data). Keep the real prediction over the 0 placeholder at each duplicated
        # timestamp before merging, so every input sample maps to exactly one output row.
        dataset_copy = (dataset_copy.sort_values("evt", ascending=False)
                         .drop_duplicates(subset="t", keep="first")
                         .sort_values("t")
                         .reset_index(drop=True))
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
                                        num_workers=0,
                                        shuffle=False)
            return test_loader, test_dataset

def merge_adjacent_fixations(evt, x_deg, y_deg, t_ms,
                             max_time_ms=MERGE_MAX_TIME_BETWEEN_FIXATIONS_MS,
                             max_angle_deg=MERGE_MAX_ANGLE_BETWEEN_FIXATIONS_DEG):
    """Tobii I-VT "merge adjacent fixations" (Olsen, 2012; constants.py): walking the
    fixations (runs of evt == FIXATION_IDX) in order, the next fixation is joined to the
    current one when the time from the current one's last sample to the next one's first
    sample is <= max_time_ms and the distance between their mean positions is
    <= max_angle_deg; the samples in between are relabeled FIXATION_IDX. A merged fixation's
    mean position is that of all its samples, and it can absorb further fixations.
    Returns a new evt array (same dtype)."""
    evt = np.asarray(evt).copy()
    is_fix = (evt == FIXATION_IDX).astype(np.int8)
    edges = np.diff(np.concatenate(([0], is_fix, [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)  # ends exclusive
    if len(starts) < 2:
        return evt
    x_deg, y_deg, t_ms = (np.asarray(a, dtype=float) for a in (x_deg, y_deg, t_ms))
    cur_end = ends[0]
    cur_sum = np.array([x_deg[starts[0]:ends[0]].sum(), y_deg[starts[0]:ends[0]].sum()])
    cur_n = ends[0] - starts[0]
    for s, e in zip(starts[1:], ends[1:]):
        nxt_sum = np.array([x_deg[s:e].sum(), y_deg[s:e].sum()])
        gap_ms = t_ms[s] - t_ms[cur_end - 1]
        dist = np.hypot(*(cur_sum / cur_n - nxt_sum / (e - s)))
        if gap_ms <= max_time_ms and dist <= max_angle_deg:
            evt[cur_end:s] = FIXATION_IDX
            cur_sum, cur_n = cur_sum + nxt_sum, cur_n + (e - s)
        else:
            cur_sum, cur_n = nxt_sum, e - s
        cur_end = e
    return evt


def normalized_to_deg(x_norm, y_norm):
    """Tobii normalized display coordinates -> degrees, on the measured screen and 65 cm
    viewing distance (constants.SCREEN_WIDTH_CM/HEIGHT_CM, deg_per_cm). Used for the merge
    angle only; the model's own input still uses deg_per_pixel, as it was trained."""
    return (np.asarray(x_norm, dtype=float) * SCREEN_WIDTH_CM * deg_per_cm,
            np.asarray(y_norm, dtype=float) * SCREEN_HEIGHT_CM * deg_per_cm)


def generate_fixation_model_based(recording_data, model, merge_adjacent=MERGE_ADJACENT_FIXATIONS):
    recording_ranner = RecordingDataRunner(model)
    annotated_data = recording_ranner.apply_prediction_to_recording_data(recording_data)
    annotated_data.columns = ["t", FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V, FIXATION_VALID_STATUS,FIXATION_CSV_KEY_FIXATION]
    if merge_adjacent:
        x_deg, y_deg = normalized_to_deg(annotated_data[FIXATION_CSV_KEY_EYE_H], annotated_data[FIXATION_CSV_KEY_EYE_V])
        # 't' is the Tobii system timestamp (us) relative to the first sample
        annotated_data[FIXATION_CSV_KEY_FIXATION] = merge_adjacent_fixations(
            annotated_data[FIXATION_CSV_KEY_FIXATION].to_numpy(), x_deg, y_deg,
            annotated_data["t"].to_numpy() / 1000)
    return annotated_data

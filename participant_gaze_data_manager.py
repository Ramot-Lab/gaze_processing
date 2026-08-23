from curses import KEY_MESSAGE
import hashlib
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

# --- Calibration/validation quality report parsing -------------------------------------
# Tobii writes one "CALIBRATION N Data Quality (computed from validation M):" message per
# calibration, right before the first task-start message. It contains a tab-separated
# table per eye with a row per validation point plus an "average" row. See
# CALIBRATION_MEASUREMENT_GLOSSARY (in calibration_drift_qa.py) for what each metric means.
CALIBRATION_HEADER_RE = re.compile(r"CALIBRATION (\d+) Data Quality \(computed from validation (\d+)\):")
CALIBRATION_POINT_LABEL_RE = re.compile(r"^(\d+)\s*@\s*\(([-\d.]+),\s*([-\d.]+)\)$")
CALIBRATION_METRIC_KEYS = ["acc", "accX", "accY", "std", "rms", "data_loss"]


def _parse_calibration_eye_block(block_text):
    """Parse the per-point + average rows for one eye out of a Data Quality message."""
    points = []
    average = None
    for raw_line in block_text.strip("\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        label = parts[0].strip()
        try:
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(values) != len(CALIBRATION_METRIC_KEYS):
            continue
        metrics = dict(zip(CALIBRATION_METRIC_KEYS, values))
        point_match = CALIBRATION_POINT_LABEL_RE.match(label)
        if point_match:
            metrics.update({
                "point_idx": int(point_match.group(1)),
                "screen_x": float(point_match.group(2)),
                "screen_y": float(point_match.group(3)),
            })
            points.append(metrics)
        elif label.lower() == "average":
            average = metrics
    return {"points": points, "average": average}


def parse_calibration_quality_message(text):
    """
    Parse a raw "CALIBRATION N Data Quality (computed from validation M): ..." message
    into {calibration_no, validation_no, left: {points, average}, right: {points, average}}.
    Returns None if the text doesn't match the expected format.
    """
    header_match = CALIBRATION_HEADER_RE.search(text)
    if header_match is None or "left eye:" not in text or "right eye:" not in text:
        return None
    left_text = text.split("left eye:", 1)[1].split("right eye:", 1)[0]
    right_text = text.split("right eye:", 1)[1]
    return {
        "calibration_no": int(header_match.group(1)),
        "validation_no": int(header_match.group(2)),
        "left": _parse_calibration_eye_block(left_text),
        "right": _parse_calibration_eye_block(right_text),
    }


def _better_calibrated_eye(calib_parsed):
    """
    'l'/'r' for whichever eye has the lower (better) calibration accuracy in a parsed
    calibration message, or None if calib_parsed is None or neither eye has an accuracy
    value to compare. Used to pick an analysis eye when Dom_Eye is missing/invalid.
    """
    if calib_parsed is None:
        return None
    l_acc = (calib_parsed["left"]["average"] or {}).get("acc")
    r_acc = (calib_parsed["right"]["average"] or {}).get("acc")
    if l_acc is None and r_acc is None:
        return None
    if r_acc is None or (l_acc is not None and l_acc <= r_acc):
        return "l"
    return "r"


def extract_last_calibration_message(messages):
    """
    Return (timestamp, raw_text) of the last message containing "Data Quality" that occurs
    before the first task-start message ("panel number ..." for SDMT, "slide ..." for KD),
    or None if no such message exists.
    """
    first_task_idx = None
    for i in range(len(messages)):
        text = str(messages[i][1])
        if text.startswith("panel number") or text.startswith("slide "):
            first_task_idx = i
            break
    candidates = messages[:first_task_idx] if first_task_idx is not None else messages
    last = None
    for ts, txt in candidates:
        if "Data Quality" in str(txt):
            last = (ts, str(txt))
    return last


class ParticipantGazeDataManager:
    def __init__(self, participant_name, main_data_path, task = "SDMT", participant_group = "pwMS", clean_gaze_data = True, analysis_eye = None) -> None:
        """
        analysis_eye: "l" or "r" to force which eye's gaze stream is used for analysis,
        overriding Dom_Eye. Defaults to None, meaning the recording's own Dom_Eye is used
        (existing/original behavior). self.dom_Eye always reflects the true dominant eye
        from the recording; self.analysis_eye reflects the eye actually used to build
        gaze_data/matched_data - the two differ only when analysis_eye is passed explicitly.
        """
        group_path, self.name = os.path.split(participant_name)
        _,  self.group = os.path.split(group_path)
        if "DONTUSE" in self.name.upper():
            raise ValueError(f"Participant {self.name!r} is flagged DONTUSE - excluded at data load.")
        self.task = task
        self.task_validation_filter_param = "run" if task == "SDMT" else "kd"
        tobii_data, task_png, audio_recordings, file_load_errors = self.load_data(participant_name , main_data_path, task, participant_group)
        self.matched_data = {}
        self.main_data_path = main_data_path
        self.output_path = os.path.join(main_data_path, "processing_results", self.name)
        self.model = None
        os.makedirs(self.output_path, exist_ok=True)
        # Each mat_file is one calibration event (one day's recording). A failure on one
        # must not prevent the others from being processed - previously an exception here
        # propagated out of __init__ and silently dropped every other calibration event for
        # this participant. file_load_errors carries failures from load_data() itself (a
        # mat file that didn't even parse); this loop appends per-event processing failures
        # to the same list so every calibration event's outcome (success or specific error)
        # is recorded independently.
        self.load_errors = list(file_load_errors)
        for mat_file in tobii_data:
            try:
                (self.task_data, self.messages, self.gaze_data, self.presentation_info,
                 self.dom_Eye, self.analysis_eye, self.eye_selection_reason) = self.prepare_gaze_data_for_preprocessing(mat_file, analysis_eye)
                self.matched_data = {**self.matched_data, **self.group_task_info(self.gaze_data, task_png, audio_recordings, self.task_data, mat_file, clean_gaze_data)}
            except Exception as e:
                try:
                    recording_date = self.get_creation_time(mat_file)
                except Exception:
                    recording_date = None
                self.load_errors.append({"recording_date": recording_date, "error": str(e)})

    def get_gaze_data_for_panel(self, panel: str):
        if panel not in self.matched_data:
            raise ValueError(f"Panel {panel} not found in matched data.")
        return self.matched_data[panel][KEY_TOBII_DATA]
    
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
            raise FileNotFoundError(f"no such path {task_files_path}")
        mat_files = glob(os.path.join(task_files_path, "*.mat"))
        mat_files = [file_path for file_path in mat_files if self.task_validation_filter_param in os.path.split(file_path)[1].lower()]
        task_png_files = glob(os.path.join(main_data_path, "panels_images", task,"*.jpg"))
        recording_files = glob(os.path.join(task_files_path, "**","*.wav"), recursive=True)

        loaded_mats, load_errors = self._load_and_dedupe_run_files(mat_files)
        return loaded_mats, task_png_files, recording_files, load_errors

    _RUN_GROUP_RE = re.compile(r'run[_]?(\d+)', re.IGNORECASE)

    def _run_group_key(self, file_path):
        """
        Best-effort grouping key for candidate run files that likely represent the same
        calibration event: the run number parsed out of the filename (case/underscore
        insensitive), if one is present. Files with no recognizable run number are treated
        as singletons (nothing to dedupe them against by name alone).
        """
        match = self._RUN_GROUP_RE.search(os.path.split(file_path)[1])
        return f"run{match.group(1)}" if match else os.path.split(file_path)[1]

    @staticmethod
    def _file_md5(file_path):
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _mat_completeness_score(mat_dict):
        """
        Higher = more complete/usable recording. Some participant folders contain exact or
        partial duplicate copies of the same session (identical content saved under a
        different filename, or a partial/truncated/empty re-save under a different date) -
        this is used to pick the genuine one by what's actually inside the file rather than
        by filename.
        """
        score = 0
        try:
            score += 1000 if np.size(mat_dict["data"].gaze.systemTimeStamp) > 0 else 0
        except Exception:
            pass
        try:
            score += 500 if extract_last_calibration_message(mat_dict["messages"]) is not None else 0
        except Exception:
            pass
        try:
            score += len(mat_dict["messages"])
        except Exception:
            pass
        try:
            score += 10 * len([k for k in mat_dict["task_data"].__dict__.keys() if not k.startswith("_")])
        except Exception:
            pass
        return score

    def _load_and_dedupe_run_files(self, mat_files):
        """
        Loads every candidate run file, but resolves duplicates by CONTENT rather than
        filename: identical files (byte-for-byte, even under different names/dates) collapse
        to one automatically; files that share a filename-derived run number but differ in
        content are resolved by keeping whichever has the higher completeness score (the
        others are typically partial/empty/truncated re-saves). One bad/corrupted file never
        prevents the participant's other calibration events from loading - each outcome
        (kept, or excluded as a duplicate/corrupt/etc, with the specific reason) is recorded
        in load_errors for anything not kept, so nothing is silently dropped.
        """
        # Stage 1: collapse exact byte-identical duplicates regardless of filename.
        by_hash = {}
        load_errors = []
        for file_path in mat_files:
            try:
                file_hash = self._file_md5(file_path)
            except Exception as e:
                load_errors.append({"file": file_path, "error": f"could not read file: {e}"})
                continue
            by_hash.setdefault(file_hash, []).append(file_path)

        unique_files = []
        for file_hash, paths in by_hash.items():
            paths_sorted = sorted(paths)
            unique_files.append(paths_sorted[0])
            for duplicate_path in paths_sorted[1:]:
                load_errors.append({
                    "file": duplicate_path,
                    "error": f"exact duplicate of {os.path.basename(paths_sorted[0])} (identical content) - excluded",
                })

        # Stage 2: group the remaining (content-distinct) files by filename-derived run
        # number, and resolve any group with >1 candidate by completeness score.
        groups = {}
        for file_path in unique_files:
            groups.setdefault(self._run_group_key(file_path), []).append(file_path)

        loaded_mats = []
        for group_key, candidates in groups.items():
            loaded = []
            for file_path in candidates:
                try:
                    loaded.append((file_path, scio.loadmat(file_path, struct_as_record=False, squeeze_me=True)))
                except Exception as e:
                    load_errors.append({"file": file_path, "error": str(e)})

            if not loaded:
                continue
            if len(loaded) == 1:
                loaded_mats.append(loaded[0][1])
                continue

            scored = sorted(
                ((self._mat_completeness_score(m), f, m) for f, m in loaded),
                key=lambda t: t[0], reverse=True,
            )
            best_score, best_file, best_mat = scored[0]
            loaded_mats.append(best_mat)
            for score, file_path, _ in scored[1:]:
                load_errors.append({
                    "file": file_path,
                    "error": f"duplicate/partial save of {os.path.basename(best_file)} (matched as "
                             f"{group_key}) - lower completeness score ({score} vs {best_score}), excluded",
                })

        return loaded_mats, load_errors


    def group_task_info(self, tobii_data_file, task_png, audio_recordings, task_data, mat_file, clean_gaze_data):
        matched_data_files = {}
        audio_idx = 2 if self.task == "SDMT" else 1
        calibration_info = self.build_calibration_info(self.messages)
        for i, task_name in enumerate(list(task_data.keys())[1::2]):
            if f"panel_{i+1}" not in tobii_data_file.keys(): continue
            task_code = (task_name[-2:]).replace("_", "")
            task_code_lower = task_code.lower()
            png_imgs = [img for img in task_png if img.endswith(f"_{task_code_lower}.jpg")]
            png_img = None if len(png_imgs) == 0 else png_imgs[0]
            # Stray non-conforming filenames (e.g. a leftover "SDMT_Sample.wav" template
            # that doesn't follow the "img_test_<PANEL>_strikes_<N>.wav" pattern) must not
            # crash panel matching - they just don't match any panel, same as a genuinely
            # missing recording (audio_file stays None -> no SDMT score for that panel,
            # gaze/calibration processing is unaffected).
            audio_files = [
                audio for audio in audio_recordings
                if len(os.path.split(audio)[1].split("_")) > audio_idx
                and task_code_lower in os.path.split(audio)[1].split("_")[audio_idx].lower()
            ]
            audio_file = None if len(audio_files) == 0 else audio_files[0]
            preprocess_gaze_method = self.clean_outliers if clean_gaze_data else self.clean_outliers_no_interpolation
            messages = self.messages_for_panel(i+1)
            matched_data_files[task_code_lower] = {KEY_TOBII_DATA: preprocess_gaze_method(tobii_data_file[f"panel_{i+1}"]),
                                                   KEY_TASK_PANEL_IMG : png_img,
                                                   KEY_PANEL_MESSAGES : messages,
                                                   KEY_CALIBRATION_INFO : calibration_info,
                                                   KEY_EYE_SELECTION_REASON : self.eye_selection_reason,
                                                   KEY_ANALYSIS_EYE : self.analysis_eye,
                                                   KEY_AUDIO_DATA :  audio_file,
                                                   KEY_STRIKE_SCORE : 0 if list(task_data.keys())[0] == "dummy" else task_data[f"strikes_img_test_{task_code}"],
                                                   KEY_REACTION_TIMES : None if list(task_data.keys())[0] == "dummy" else task_data.get(f"reaction_times_img_test_{task_code}"),
                                                   KEY_RECORDING_DATE : self.get_creation_time(mat_file)}
        return matched_data_files

    def build_calibration_info(self, messages):
        """
        Parse the calibration/validation quality report that applies to this run's panels
        (the last "Data Quality" message before the first panel starts). Returns None if no
        calibration message is found or it doesn't match the expected format, so a subject
        with a malformed/missing calibration message doesn't crash loading - callers should
        treat None as "calibration info unavailable" rather than skip the subject.
        """
        found = extract_last_calibration_message(messages)
        if found is None:
            return None
        timestamp, text = found
        parsed = parse_calibration_quality_message(text)
        if parsed is None:
            return None
        parsed["timestamp"] = timestamp
        parsed["dominant_eye"] = self.dom_Eye
        return parsed

    def messages_for_panel(self, panel_idx: int ): #1, 2 or 3
        all_messages = self.messages
        if not isinstance(all_messages, np.ndarray):
            all_messages = np.array(all_messages, dtype=object)

        # --- Find start of this panel ---
        start_match_idx = next(
            (i for i, (_, msg) in enumerate(all_messages) if f"panel number {panel_idx}" in msg),
            None
        )
        if start_match_idx is None:
            raise ValueError(f"No start message found for panel number {panel_idx}")

        start_time = all_messages[start_match_idx, 0]

        # --- Find end of this panel ---
        end_match_idx = next(
            (i for i, (_, msg) in enumerate(all_messages) if f"break panel number {panel_idx}" in msg),
            None
        )

        if end_match_idx is not None:
            end_time = all_messages[end_match_idx, 0]
        else:
            # if no break message, fall back to "finished"
            finished_idx = next(
                (i for i, (_, msg) in enumerate(all_messages) if "finished" in msg),
                None
            )
            if finished_idx is None:
                raise ValueError(f"No end or finished message found for panel number {panel_idx}")
            end_time = all_messages[finished_idx, 0]

        # --- Extract messages in time range ---
        timestamps = all_messages[:, 0].astype(float)
        mask = (timestamps >= float(start_time)) & (timestamps <= float(end_time))
        panel_messages = all_messages[mask]
        return panel_messages
    
    def get_panel_img(self, panel: str):
        img_path = self.matched_data[panel].get(KEY_TASK_PANEL_IMG, None)
        try:
            return plt.imread(img_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No image found at path: {img_path}")

    def prepare_gaze_data_for_preprocessing(self, data, analysis_eye=None):
        # Extract left and right gaze data
        left_gaze = data['data'].gaze.left.gazePoint.onDisplayArea
        right_gaze = data['data'].gaze.right.gazePoint.onDisplayArea
        tobi_ts = data["data"].gaze.systemTimeStamp

        # A session where the task ran (messages/presses look normal) but the eye tracker
        # streamed zero samples squeezes onDisplayArea/systemTimeStamp down to a 1-D empty
        # array, which then fails deep inside indexing below with a cryptic
        # "too many indices for array" - raise a clear, specific reason instead so it's
        # identifiable as "no gaze data recorded" rather than looking like a code bug.
        if np.size(tobi_ts) == 0:
            raise ValueError("No gaze samples recorded for this session (eye tracker produced 0 samples)")

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
        dom_eye_valid = isinstance(Dom_Eye, str) and Dom_Eye.lower() in ('r', 'l')

        # eye_for_analysis is the eye whose gaze stream actually gets used below - Dom_Eye
        # unless analysis_eye explicitly overrides it (see ParticipantGazeDataManager.__init__).
        # eye_selection_reason records WHY, for provenance (see KEY_EYE_SELECTION_REASON).
        if analysis_eye is not None:
            eye_for_analysis = analysis_eye.lower()
            eye_selection_reason = "explicit_override"
        elif dom_eye_valid:
            eye_for_analysis = Dom_Eye.lower()
            eye_selection_reason = "dominant"
        else:
            # Dom_Eye missing/invalid (e.g. "Dom_Eye must be either r or l") - rather than
            # discarding this event, fall back to whichever eye has the better-calibrated
            # accuracy for THIS calibration event's own Data Quality message.
            found = extract_last_calibration_message(messages)
            calib_parsed = parse_calibration_quality_message(found[1]) if found is not None else None
            fallback_eye = _better_calibrated_eye(calib_parsed)
            if fallback_eye is None:
                raise ValueError(
                    'Dom_Eye missing/invalid ("Dom_Eye must be either \"r\" or \"l\"") and no '
                    "calibration message available to fall back on - cannot pick an analysis eye"
                )
            eye_for_analysis = fallback_eye
            eye_selection_reason = "fallback_missing_dom_label"
        assert eye_for_analysis in ['r', 'l'], f'analysis_eye must be "r" or "l", got {analysis_eye!r}'

        # Select gaze data for each panel based on eye_for_analysis
        selected_gaze = right_gaze if eye_for_analysis == 'r' else left_gaze
        other_eye = 'l' if eye_for_analysis == 'r' else 'r'
        other_gaze = left_gaze if eye_for_analysis == 'r' else right_gaze

        def _nan_ratio(gaze_2d, indices):
            panel_data = np.concatenate((gaze_2d[:, indices].T, np.reshape(tobi_ts[indices], (-1, 1))), axis=1)
            return panel_data, (np.count_nonzero(np.isnan(panel_data)) // 2) / len(panel_data)

        gaze_data = {}
        for i in range(3):
            indices = panel_presentation_indices[i]
            panel_data, nan_ratio = _nan_ratio(selected_gaze, indices)
            if nan_ratio < MAX_VALID_NAN_VALUES:
                gaze_data[f'panel_{i+1}'] = panel_data
                continue

            # The selected eye failed - also check the OTHER eye for this same panel so the
            # error message says whether a rescue would even have been possible, rather than
            # only ever reporting the one eye that happened to be selected (see eye-fallback
            # discussion). This does NOT automatically switch eyes - no accuracy threshold
            # has been agreed on for that (see calibration_quality_analysis.py README).
            _, other_nan_ratio = _nan_ratio(other_gaze, indices)
            if other_nan_ratio < MAX_VALID_NAN_VALUES:
                raise Exception(
                    f"Too many NaN values in panel {i+1} for analysis eye '{eye_for_analysis}' "
                    f"(NaN ratio={nan_ratio:.1%}); other eye '{other_eye}' would pass "
                    f"(NaN ratio={other_nan_ratio:.1%})"
                )
            else:
                raise Exception(
                    f"Too many NaN values in panel {i+1} for BOTH eyes "
                    f"('{eye_for_analysis}': {nan_ratio:.1%}, '{other_eye}': {other_nan_ratio:.1%})"
                )


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
        return task_data, messages, gaze_data, presentation_info, Dom_Eye, eye_for_analysis, eye_selection_reason
    
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
        if audio_path is None:
            print(f"No audio data for panel: {panel} subject: {self.name}, date: {self.matched_data[panel][KEY_RECORDING_DATE]}")
            return []
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

    def annotate_gaze_events(self, panel : str,  annotation_method = 'threshold_based'):
        if annotation_method == "threshold_based":
            return generate_fixations_threshold_based(self, panel)
        elif annotation_method == "model_based":
            if self.model is None:
                self.load_model()
            return generate_fixation_model_based(self.matched_data[panel][KEY_TOBII_DATA], model = self.model)
        elif annotation_method == "pymovments_based":
            return generate_fixations_pymovements_based(self, panel)
        else:
            raise Exception(f"unknown annotation method: {annotation_method} choose from ['threshold_based', 'model_based', 'pymovments_based']")

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
    from visualize_data import show_running_video_live, plot_gazeNet_fig
    p_name = "RD707"
    task = "SDMT"
    group = "pwMS"
    panel = "3"
    data_path = "/Volumes/labs/ramot/rotation_students/Nitzan_K/MS/Results/Behavior"
    subject_data= ParticipantGazeDataManager(p_name, data_path, "KD", group)


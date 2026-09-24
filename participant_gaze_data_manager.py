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
from exclusion_policy import ACCURACY_EXCLUSION_THRESHOLD_DEG
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


def _apply_out_of_range_policy(left_gaze, right_gaze):
    """
    Applies constants.OUT_OF_RANGE_VALUES_METHOD to the raw x/y gaze arrays (shape (2, N):
    row 0 = x, row 1 = y). See constants.py for the current bounds and the reasoning
    (2026-09-07) behind treating x/y and the two sides of each axis differently.
    """
    if OUT_OF_RANGE_VALUES_METHOD == "as_is":
        return left_gaze, right_gaze

    left_gaze = left_gaze.copy()
    right_gaze = right_gaze.copy()
    if OUT_OF_RANGE_VALUES_METHOD == "all_NaN":
        left_gaze[(left_gaze < 0) | (left_gaze > 1)] = np.nan
        right_gaze[(right_gaze < 0) | (right_gaze > 1)] = np.nan
    elif OUT_OF_RANGE_VALUES_METHOD == "only_extreme_values":
        x_lo, x_hi = OUT_OF_RANGE_X_BOUNDS
        y_lo, y_hi = OUT_OF_RANGE_Y_BOUNDS
        for gaze in (left_gaze, right_gaze):
            gaze[0][(gaze[0] < x_lo) | (gaze[0] > x_hi)] = np.nan
            gaze[1][(gaze[1] < y_lo) | (gaze[1] > y_hi)] = np.nan
    else:
        raise ValueError(f"Unknown OUT_OF_RANGE_VALUES_METHOD: {OUT_OF_RANGE_VALUES_METHOD!r}")
    return left_gaze, right_gaze


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


class EventExcludedError(ValueError):
    """
    Raised when NO panel of a calibration event can be used at all (every panel failed the
    accuracy/NaN-ratio eye-selection policy). Carries panel_details - one dict per panel of
    this event with the full accuracy/NaN-ratio breakdown for both eyes - so callers can
    report exactly why, per panel, instead of only a summary string.
    """
    def __init__(self, message, panel_details):
        super().__init__(message)
        self.panel_details = panel_details


def _other_errors_note(l_acc, r_acc, l_nan_pct, r_nan_pct):
    """
    Secondary-criterion note for an excluded panel: flags an eye whose exclusion is
    "overdetermined" - i.e. it fails BOTH accuracy and NaN-ratio simultaneously, not just
    whichever one was the stated primary reason. Blank when no eye double-fails.
    """
    notes = []
    for label, acc, nan_pct in (("left", l_acc, l_nan_pct), ("right", r_acc, r_nan_pct)):
        acc_bad = acc is not None and acc >= ACCURACY_EXCLUSION_THRESHOLD_DEG
        nan_bad = nan_pct is not None and nan_pct >= MAX_VALID_NAN_VALUES * 100
        if acc_bad and nan_bad:
            notes.append(f"{label} eye also fails both criteria (acc={acc:.2f}, NaN={nan_pct:.1f}%)")
    return "; ".join(notes)


class ParticipantGazeDataManager:
    def __init__(self, participant_name, main_data_path, task = "SDMT", participant_group = "pwMS", clean_gaze_data = True, analysis_eye = None) -> None:
        """
        analysis_eye: "l" or "r" to force which eye's gaze stream is used for analysis, for
        every panel unconditionally. Defaults to None, meaning each panel picks its own eye
        via the accuracy+NaN-ratio policy in prepare_gaze_data_for_preprocessing (decision
        2026-09-07) - panels of the SAME calibration event can end up using different eyes,
        since NaN ratio is per-panel while accuracy is per-event. self.dom_Eye always
        reflects the true physiological dominant eye from the recording (audit/reference
        only, not used for eye selection); self.panel_eye_used/self.panel_eye_reason are
        dicts keyed 'panel_1'/'panel_2'/'panel_3' recording which eye was actually used per
        panel and why.
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
        # Panels dropped individually (the event otherwise loaded fine) - see
        # prepare_gaze_data_for_preprocessing's per-panel eye-selection/NaN-ratio policy.
        # Distinct from load_errors, which is always a WHOLE-event failure.
        self.panel_load_errors = []
        for mat_file in tobii_data:
            try:
                (self.task_data, self.messages, self.gaze_data, self.presentation_info,
                 self.dom_Eye, self.panel_eye_used, self.panel_eye_reason,
                 panel_failure_reasons) = self.prepare_gaze_data_for_preprocessing(mat_file, analysis_eye)
                self.matched_data = {**self.matched_data, **self.group_task_info(self.gaze_data, task_png, audio_recordings, self.task_data, mat_file, clean_gaze_data)}
                if panel_failure_reasons:
                    try:
                        recording_date = self.get_creation_time(mat_file)
                    except Exception:
                        recording_date = None
                    for reason in panel_failure_reasons:
                        self.panel_load_errors.append({"recording_date": recording_date, **reason})
            except EventExcludedError as e:
                # Whole-event failure with a full per-panel accuracy/NaN-ratio breakdown
                # (e.panel_details, one dict per panel of this event) - kept as an EXTRA key
                # on a single load_errors entry (rather than expanding to 3 rows here) so
                # existing one-row-per-event consumers (e.g. calibration_quality_analysis.py)
                # are unaffected; a report that wants the per-panel breakdown can read
                # panel_details itself (see build_exclusion_report.py).
                try:
                    recording_date = self.get_creation_time(mat_file)
                except Exception:
                    recording_date = None
                self.load_errors.append({"recording_date": recording_date, "error": str(e),
                                          "panel_details": e.panel_details})
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
                                                   KEY_EYE_SELECTION_REASON : self.panel_eye_reason.get(f"panel_{i+1}"),
                                                   KEY_ANALYSIS_EYE : self.panel_eye_used.get(f"panel_{i+1}"),
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

        # Out-of-[0,1]-range handling (2026-09-07): the right threshold for "tracker noise
        # near the edge" vs "real data loss" hasn't been decided yet, so this is a switch
        # (constants.OUT_OF_RANGE_VALUES_METHOD), not a hardcoded rule. Default is "as_is" -
        # out-of-range samples are left exactly as recorded, including for the NaN-ratio
        # check just below (a value like -0.03 or 1.4 is NOT NaN and does not count toward
        # MAX_VALID_NAN_VALUES under "as_is").
        left_gaze, right_gaze = _apply_out_of_range_policy(left_gaze, right_gaze)

        # Extract messages
        messages = data['messages']

        # Resolved early (moved up from the end of this function) so panel-level messages
        # below can name panels by their actual short code (e.g. "i1", "l4") instead of
        # their presentation-order index - same resolution group_task_info uses.
        if self.task == "KD":
            task_data = {"dummy": None, "1": None, "0": None, "2": None, "00": None, "3": None, "000": None}  # TODO: how to get the real task results.
        elif self.task == "SDMT":
            task_data = data["task_data"].__dict__
        panel_codes = [(name[-2:]).replace("_", "").lower() for name in list(task_data.keys())[1::2]]

        def _panel_label(i):
            return panel_codes[i] if i < len(panel_codes) else f"panel_{i + 1}"

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

        Dom_Eye = data['Dom_Eye']
        # A few recordings have Dom_Eye typed in Hebrew (keyboard-locale slip during the
        # experiment) instead of English 'r'/'l' - e.g. YO399's second event. Normalize
        # the known variants rather than losing that event's panels to an assertion error.
        HEBREW_EYE_ALIASES = {'ר': 'r', 'ך': 'l'}
        if Dom_Eye in HEBREW_EYE_ALIASES:
            Dom_Eye = HEBREW_EYE_ALIASES[Dom_Eye]

        # Decision 2026-09-22: default to the participant's own dominant eye,
        # unconditionally, for every panel - accuracy-based per-panel eye selection
        # (decision 2026-09-07, the big `else` block below) is disregarded for now. A
        # cross-participant check found the dominant eye is NOT reliably the
        # better-calibrated one (calibration accuracy in degrees and dominance were
        # mildly ANTI-correlated, p=9e-6 - see after_analysis_quality_checks_17_09_26/
        # eye_accuracy_dominant_vs_nondominant.png), so "best calibrated" was routinely
        # overriding genuine dominant-eye tracking. Reuses the existing explicit-override
        # branch just below (originally for a caller-forced eye) rather than duplicating
        # its NaN-ratio-per-panel logic - passing analysis_eye explicitly still overrides
        # this default. The accuracy-based branch is left in place, just unreachable by
        # default, in case this decision is revisited.
        _eye_was_explicit = analysis_eye is not None
        if analysis_eye is None:
            analysis_eye = Dom_Eye

        def _nan_ratio(gaze_2d, indices):
            panel_data = np.concatenate((gaze_2d[:, indices].T, np.reshape(tobi_ts[indices], (-1, 1))), axis=1)
            return panel_data, (np.count_nonzero(np.isnan(panel_data)) // 2) / len(panel_data)

        gaze_data = {}
        panel_eye_used = {}
        panel_eye_reason = {}
        panel_failure_reasons = []

        if analysis_eye is not None:
            # Explicit override: unconditionally use this eye - still subject to the
            # per-panel NaN-ratio gate, and a panel that fails it is simply dropped rather
            # than failing the whole event (same per-panel policy as below). No accuracy
            # check here - an explicit override is meant to bypass that by design (e.g.
            # for a caller deliberately testing one specific eye).
            #
            # But when this branch is running because analysis_eye defaulted to the
            # dominant eye (not an explicit override), accuracy exclusion is back
            # (2026-09-22 decision, restoring the 2026-09-07 accuracy gate): the dominant
            # eye is still used unconditionally for selection, but if its own calibration
            # accuracy doesn't clear the threshold, the whole event is excluded exactly as
            # the old per-eye-accuracy policy would have.
            forced_eye = analysis_eye.lower()
            assert forced_eye in ['r', 'l'], f'analysis_eye must be "r" or "l", got {analysis_eye!r}'

            if not _eye_was_explicit:
                found = extract_last_calibration_message(messages)
                calib_parsed = parse_calibration_quality_message(found[1]) if found is not None else None
                eye_key = 'left' if forced_eye == 'l' else 'right'
                average = (calib_parsed[eye_key]['average'] if calib_parsed else None) or {}
                dom_acc = average.get('acc')
                if dom_acc is None or dom_acc >= ACCURACY_EXCLUSION_THRESHOLD_DEG:
                    acc_txt = f"{dom_acc:.2f}" if dom_acc is not None else "unknown"
                    event_panels = ", ".join(_panel_label(i) for i in range(3))
                    panel_details = [{
                        "panel": _panel_label(i),
                        "error": f"panel {_panel_label(i)}: dominant eye ('{forced_eye}') calibration "
                                 f"accuracy={acc_txt} deg is not under {ACCURACY_EXCLUSION_THRESHOLD_DEG} deg",
                        "l_acc": dom_acc if forced_eye == 'l' else None,
                        "r_acc": dom_acc if forced_eye == 'r' else None,
                        "l_nan_pct": None, "r_nan_pct": None, "other_errors": "",
                    } for i in range(3)]
                    raise EventExcludedError(
                        f"Dominant eye ('{forced_eye}') calibration accuracy={acc_txt} deg is not under "
                        f"{ACCURACY_EXCLUSION_THRESHOLD_DEG} deg - event excluded (panels: {event_panels})",
                        panel_details,
                    )

            selected_gaze = right_gaze if forced_eye == 'r' else left_gaze
            for i in range(3):
                indices = panel_presentation_indices[i]
                panel_data, nan_ratio = _nan_ratio(selected_gaze, indices)
                if nan_ratio < MAX_VALID_NAN_VALUES:
                    gaze_data[f'panel_{i + 1}'] = panel_data
                    panel_eye_used[f'panel_{i + 1}'] = forced_eye
                    panel_eye_reason[f'panel_{i + 1}'] = "explicit_override" if _eye_was_explicit else "dominant_eye_default"
                else:
                    panel_failure_reasons.append({
                        "panel": _panel_label(i),
                        "error": f"panel {_panel_label(i)}: forced eye '{forced_eye}' NaN ratio={nan_ratio:.1%} "
                                 f"(>= {MAX_VALID_NAN_VALUES:.0%})",
                        "l_acc": None, "r_acc": None, "l_nan_pct": None, "r_nan_pct": None,
                        "other_errors": "",
                    })
        else:
            # Eye selection + exclusion policy (decision 2026-09-07): accuracy and NaN ratio
            # are evaluated jointly, per panel, instead of accuracy picking one eye for the
            # whole event and NaN ratio only gating that fixed choice afterward. Accuracy is
            # identical across all 3 panels of one event (one calibration report covers all
            # 3), but NaN ratio is computed per panel - so which eye ends up used, or
            # whether a panel is usable at all, CAN differ panel-to-panel within one event.
            #
            # An eye "qualifies" for a panel iff its calibration accuracy < 2 deg AND this
            # panel's NaN ratio for that eye is < 10%:
            #   - neither eye's accuracy is under 2 deg -> whole event excluded (accuracy is
            #     event-level, so this applies identically to all 3 panels)
            #   - exactly one eye has accuracy < 2 deg -> only that eye is ever a candidate;
            #     the panel is used if its NaN ratio is < 10%, dropped otherwise
            #   - both eyes have accuracy < 2 deg -> for this panel, use whichever qualifying
            #     eye has the better (lower) accuracy, tie-broken by lower NaN ratio; if only
            #     one of the two actually passes the NaN-ratio check for this panel, use that
            #     one (this is the "rescue" case - the worse-calibrated eye can still save a
            #     panel the better-calibrated eye can't)
            #   - neither eye passes the NaN-ratio check for this panel -> panel dropped
            found = extract_last_calibration_message(messages)
            calib_parsed = parse_calibration_quality_message(found[1]) if found is not None else None
            acc = {}
            for eye, key in (('l', 'left'), ('r', 'right')):
                average = (calib_parsed[key]['average'] if calib_parsed else None) or {}
                acc[eye] = average.get('acc')
            acc_ok = {eye: (acc[eye] is not None and acc[eye] < ACCURACY_EXCLUSION_THRESHOLD_DEG)
                      for eye in ('l', 'r')}

            # NaN ratio is computed for BOTH eyes on every panel unconditionally (not just
            # the eye(s) that pass accuracy) purely for reporting - so an excluded panel's
            # full accuracy/NaN-ratio picture is always available, even for the eye that was
            # never actually a candidate.
            gaze_by_eye = {'l': left_gaze, 'r': right_gaze}
            nan_pct = {'l': [None, None, None], 'r': [None, None, None]}
            panel_data_by_eye = {'l': [None, None, None], 'r': [None, None, None]}
            for i in range(3):
                indices = panel_presentation_indices[i]
                for eye in ('l', 'r'):
                    panel_data, ratio = _nan_ratio(gaze_by_eye[eye], indices)
                    nan_pct[eye][i] = ratio * 100
                    panel_data_by_eye[eye][i] = panel_data

            if not acc_ok['l'] and not acc_ok['r']:
                l_txt = f"{acc['l']:.2f}" if acc['l'] is not None else "unknown"
                r_txt = f"{acc['r']:.2f}" if acc['r'] is not None else "unknown"
                event_panels = ", ".join(_panel_label(i) for i in range(3))
                panel_details = [{
                    "panel": _panel_label(i),
                    "error": f"panel {_panel_label(i)}: neither eye's calibration accuracy is under "
                             f"{ACCURACY_EXCLUSION_THRESHOLD_DEG} deg (left={l_txt}, right={r_txt})",
                    "l_acc": acc['l'], "r_acc": acc['r'],
                    "l_nan_pct": nan_pct['l'][i], "r_nan_pct": nan_pct['r'][i],
                    "other_errors": _other_errors_note(acc['l'], acc['r'], nan_pct['l'][i], nan_pct['r'][i]),
                } for i in range(3)]
                raise EventExcludedError(
                    f"Neither eye's calibration accuracy is under {ACCURACY_EXCLUSION_THRESHOLD_DEG} "
                    f"deg (left={l_txt}, right={r_txt}) - event excluded (panels: {event_panels})",
                    panel_details,
                )

            for i in range(3):
                candidates = []
                for eye in ('l', 'r'):
                    if not acc_ok[eye]:
                        continue
                    if nan_pct[eye][i] / 100 < MAX_VALID_NAN_VALUES:
                        candidates.append((eye, acc[eye], nan_pct[eye][i], panel_data_by_eye[eye][i]))

                if not candidates:
                    panel_failure_reasons.append({
                        "panel": _panel_label(i),
                        "error": f"panel {_panel_label(i)}: no eye both has accuracy < {ACCURACY_EXCLUSION_THRESHOLD_DEG} "
                                 f"deg and NaN ratio < {MAX_VALID_NAN_VALUES:.0%}",
                        "l_acc": acc['l'], "r_acc": acc['r'],
                        "l_nan_pct": nan_pct['l'][i], "r_nan_pct": nan_pct['r'][i],
                        "other_errors": _other_errors_note(acc['l'], acc['r'], nan_pct['l'][i], nan_pct['r'][i]),
                    })
                    continue

                # Best accuracy wins; tie-break by lower NaN ratio (only matters in the rare
                # exact-accuracy-tie case).
                candidates.sort(key=lambda c: (c[1], c[2]))
                best_eye, _, _, panel_data = candidates[0]
                gaze_data[f'panel_{i + 1}'] = panel_data
                panel_eye_used[f'panel_{i + 1}'] = best_eye
                panel_eye_reason[f'panel_{i + 1}'] = (
                    "best_calibrated" if (acc_ok['l'] and acc_ok['r']) else "only_qualifying_eye"
                )

        if not gaze_data:
            raise EventExcludedError(
                "No panel passed eye-selection/NaN-ratio criteria: "
                + "; ".join(d["error"] for d in panel_failure_reasons),
                panel_failure_reasons,
            )
        if panel_failure_reasons:
            print(f"Warning: dropping panel(s) that failed eye-selection/NaN-ratio criteria "
                  f"for {self.name}: {'; '.join(d['error'] for d in panel_failure_reasons)}")

        # Calculate presentation durations
        presentation_info = {}
        for i in range(3):
            presentation_info[f'panel_{i+1}'] = {
                'start_time': panel_start_times[i],
                'end_time': break_start_times[i],
                'duration': break_start_times[i] - panel_start_times[i]
            }

        return task_data, messages, gaze_data, presentation_info, Dom_Eye, panel_eye_used, panel_eye_reason, panel_failure_reasons
    
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


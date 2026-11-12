"""
Saccade latency on the fxsp task (fixation part, "panel 1"), at the native 600 Hz with
gazeNET model-based annotation (decision 2026-11-04) - replaces the external pipeline
that produced .../saccade_latency/Maysan_test/compiled_latencies.csv
(rotation_students/Maysan/gaze_processing-Noam_Mizrachi/test_noam.py).

What is kept identical to test_noam.py, on purpose, so the two tables are comparable:
  - data extraction: same .mat selection (largest *fxsp*.mat), same
    panel split ('fixation trial 1' -> 'fixation task done'), dominant eye, the same
    NaN-ratio gate (< 0.2) and the same clean_outliers (mean + 8 SD jump -> NaN ->
    linear interpolation) - see load_fxsp_fixation_panel.
  - latency computation: per trial, count samples from trial onset up to and including
    the first saccade sample, latency = count / sampling rate; drop the last "trial"
    (it is the 'fixation task done' message, not a trial), keep 0.1 <= latency <= 0.5 s,
    skip participants left with < 2 trials - see compute_trial_latencies.
  - output table schema: compiled_latencies.csv with participant, trial_idx,
    saccade_latency_ms (NOTE: values are in SECONDS despite the column name - kept as
    is because feature_pipeline.merge_saccade_latency/run_saccade_latency_reliability
    read this exact column).

What changes:
  - input: Noam_M/Results/Behavior/{pwMS,HC}/<participant>/fxsp (every participant)
    instead of Maysan's subset copy; DONTUSE folders skipped.
  - no [::6] decimation - all samples at 600 Hz are used. Sampling at 100 Hz quantizes
    onset to 10 ms and biases onset-based measures by about half a sample interval to
    one sample interval (Andersson, Nystrom & Holmqvist, 2010, J Eye Mov Res 3(3)).
    The decimation was only needed because pymovements' I-DT requires integer,
    constant-interval timesteps (pymovements/events/detection/idt.py).
  - annotation by gazeNET (Zemblys, Niehorster & Holmqvist, 2019, Behav Res Methods
    51:840-864) - the same model + weights the SDMT pipeline uses (GazeModel/) -
    instead of pymovements I-DT (Salvucci & Goldberg, 2000). Saccade = evt ==
    SACCADE_IDX only; under I-DT every non-fixation sample counted as saccade, here
    PSO (evt 3) and unlabeled (evt 0) samples do not.

Also writes fxsp_fixation_bcea_per_evt_all_partici.csv: one row per fixation in the same
panel-1 annotations, BCEA in deg^2 and normalized units (see compute_fixation_bcea).

Usage:
  python saccade_latency_analysis.py                       # full population, today's run dir
  python saccade_latency_analysis.py --run-dir <dir>       # resume / write elsewhere
  python saccade_latency_analysis.py --participants AM367 KN566
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import scipy.io as scio

import pipeline_config
from constants import (SACCADE_IDX, FIXATION_IDX, PSO_IDX, SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM, deg_per_cm,
                       BCEA_MAX_FILLED_IN_STRETCH_SAMPLES, BCEA_MAX_FILLED_IN_FIXATION_PCT)

TASK = "fxsp"
GROUPS = ["pwMS", "HC"]

# fxsp panel 1 = the fixation/saccade part; panel 2 is smooth pursuit (Maysan's split).
PANEL_START_MESSAGE = "fixation trial 1"
PANEL_END_MESSAGE = "fixation task done"
PANEL_FALLBACK_END_MESSAGE = "finished"

SCREEN_W_PX, SCREEN_H_PX = 1920, 1080
SAMPLING_RATE_HZ = 600

MAX_PANEL_NAN_RATIO = 0.2  # test_noam.py / Maysan's ParticipantGazeDataManager gate
OUTLIER_SD_MULTIPLIER = 8  # clean_outliers_single_eye

LATENCY_MIN_S = 0.1
LATENCY_MAX_S = 0.5
MIN_TRIALS_PER_PARTICIPANT = 2

# Per-fixation BCEA on the same panel-1 annotations (decision 2026-11-08): a fixation is
# a run of consecutive evt == FIXATION_IDX samples (after merge-adjacent-fixations, see
# constants.MERGE_ADJACENT_FIXATIONS); kept only if >= 30 ms long.
FIXATION_MIN_DURATION_MS = 30
# Interpolation rules for BCEA fixations: constants.BCEA_MAX_FILLED_IN_STRETCH_SAMPLES and
# constants.BCEA_MAX_FILLED_IN_FIXATION_PCT (rationale and citations there), applied by
# filled_in_rule - shared with the SDMT fixation BCEA in feature_pipeline.
MAX_FILLED_IN_STRETCH_SAMPLES = BCEA_MAX_FILLED_IN_STRETCH_SAMPLES
MAX_FILLED_IN_FIXATION_PCT = BCEA_MAX_FILLED_IN_FIXATION_PCT
# BCEA reported in deg^2, the visual-angle unit of the fixation-stability literature
# (Steinman, 1965, JOSA 55:1158; Castet & Crossland, 2012, Seeing Perceiving 25:449),
# in cm^2 on the screen, and in Tobii normalized units (for comparison with
# fixation_bcea_per_evt_all_partici.csv). Normalized x is a fraction of screen width and
# y of screen height, so x_cm = x_norm * SCREEN_WIDTH_CM, y_cm = y_norm * SCREEN_HEIGHT_CM
# (measured screen, constants.py), and since BCEA ~ sd_x * sd_y (rho is scale-invariant)
# bcea_cm2 = bcea_norm * width_cm * height_cm exactly. deg^2 uses the measured cm and the
# 65 cm viewing distance (constants.deg_per_cm, small-angle, as deg_per_pixel), not the
# 96-DPI PIXEL2METER default.
CM2_PER_NORM_UNIT = SCREEN_WIDTH_CM * SCREEN_HEIGHT_CM
DEG2_PER_NORM_UNIT = CM2_PER_NORM_UNIT * deg_per_cm ** 2

EVENT_NAMES = {FIXATION_IDX: "fixation", SACCADE_IDX: "saccade", PSO_IDX: "pso", 0: "unlabeled"}
SACCADE_GAZE_TYPE = "saccade"

ANNOTATION_METHOD = "model_based"
OUTPUT_SUBDIR = "saccade_latency"


def fxsp_input_data_path():
    """The lab's full Behavior dataset (decision 2026-11-04 - all participants, not just
    the subset copied into rotation_students/Maysan/Results/data/my_dataset/raw that
    test_noam.py read). Same root the SDMT pipeline reads; re-resolved on each call since
    the share remounts as /Volumes/ramot-N."""
    return pipeline_config.main_data_path()


def _fxsp_dir(subject_path):
    """The participant's fxsp folder, whatever its case on disk (None if absent)."""
    return next((os.path.join(subject_path, x) for x in os.listdir(subject_path) if x.lower() == TASK), None)


def iter_fxsp_subject_dirs(input_data_path=None):
    """(group, subject_dir) for every participant with an fxsp folder (same filter as
    test_noam.py). DONTUSE folders are skipped, as in run_preprocessing.iter_all_subject_dirs."""
    input_data_path = input_data_path or fxsp_input_data_path()
    for group in GROUPS:
        for subject_path in sorted(glob.glob(os.path.join(input_data_path, group, "*"))):
            if not os.path.isdir(subject_path) or "DONTUSE" in os.path.basename(subject_path).upper():
                continue
            if _fxsp_dir(subject_path) is None:
                continue
            yield group, subject_path


# ---------------------------------------------------------------------------------------
# Extraction - a side-effect-free port of Maysan's ParticipantGazeDataManager for fxsp
# (load_data / prepare_gaze_data_for_preprocessing / get_messages_for_panel /
# clean_outliers). The SDMT ParticipantGazeDataManager in this repo asserts 3 panels and
# can't load fxsp; Maysan's also os.makedirs() into her raw folder on every load.
# ---------------------------------------------------------------------------------------
def _load_fxsp_mat(subject_path):
    task_dir = _fxsp_dir(subject_path)
    mat_files = [f for f in glob.glob(os.path.join(task_dir, "*.mat"))
                 if TASK in os.path.split(f)[1].lower()]  # filters out the logfile
    if not mat_files:
        raise FileNotFoundError(f"no *{TASK}*.mat in {task_dir}")
    mat_path = max(mat_files, key=os.path.getsize)  # as Maysan: largest file wins
    return scio.loadmat(mat_path, struct_as_record=False, squeeze_me=True), mat_path


# Dom_Eye typed with the Hebrew keyboard layout (the same slip ParticipantGazeDataManager
# normalizes for SDMT, e.g. YO399): the key for 'r' types 'ר', the key for 'l' types 'ך'.
HEBREW_EYE_ALIASES = {"ר": "r", "ך": "l"}
FALLBACK_EYE = "r"


def resolve_dominant_eye(raw_dom_eye):
    """fxsp only (SDMT's own handling is untouched): Hebrew-layout letters are mapped to
    r/l, case is ignored ('R' is the right eye), and when no usable dominant eye is
    recorded at all, fall back to FALLBACK_EYE (decision 2026-11-10) instead of skipping
    the participant. Returns (eye, fell_back)."""
    eye = str(raw_dom_eye).strip()
    eye = HEBREW_EYE_ALIASES.get(eye, eye).lower()
    if eye in ("r", "l"):
        return eye, False
    return FALLBACK_EYE, True


def _clean_outliers_single_eye(eye):
    """Returns (cleaned_eye, interpolated_mask) - the mask marks every sample that was
    missing in the recording or removed as an outlier jump, i.e. filled in by np.interp."""
    jumps = np.abs(np.diff(eye))
    cutoff = np.nanstd(jumps) * OUTLIER_SD_MULTIPLIER + np.nanmean(jumps)
    eye = eye.copy()
    eye[np.concatenate(([False], jumps > cutoff))] = np.nan
    nans = np.isnan(eye)
    if nans.all():
        raise ValueError("only nan values detected")
    idx = np.arange(len(eye))
    eye[nans] = np.interp(idx[nans], idx[~nans], eye[~nans])
    return eye, nans


def _panel_messages(messages):
    """All messages from the first one containing PANEL_START_MESSAGE to the first one
    containing PANEL_END_MESSAGE (fallback 'finished'), inclusive - each is one trial
    onset, the last being the end message itself."""
    messages = np.asarray(messages, dtype=object)
    start = next((m[0] for m in messages if PANEL_START_MESSAGE in m[1]), None)
    if start is None:
        raise ValueError(f"no '{PANEL_START_MESSAGE}' message")
    end = next((m[0] for m in messages if PANEL_END_MESSAGE in m[1]), None)
    if end is None:
        end = next((m[0] for m in messages if PANEL_FALLBACK_END_MESSAGE in m[1]), None)
    if end is None:
        raise ValueError(f"no '{PANEL_END_MESSAGE}' or '{PANEL_FALLBACK_END_MESSAGE}' message")
    ts = messages[:, 0].astype(float)
    return messages[(ts >= float(start)) & (ts <= float(end))]


def load_fxsp_fixation_panel(subject_path):
    """Returns dict(gaze=(N,3) [x_norm, y_norm, tobii_ts_us] cleaned + interpolated,
    messages=panel-1 messages, dom_eye, mat_path). Raises on any reason the participant
    would have been skipped by test_noam.py."""
    data, mat_path = _load_fxsp_mat(subject_path)
    messages = data["messages"]
    tobii_ts = data["data"].gaze.systemTimeStamp

    # break_mat_into_pannels: exact-match on message text, first occurrence wins
    starts = [m[0] for m in messages if m[1] == PANEL_START_MESSAGE]
    ends = [m[0] for m in messages if m[1] in (PANEL_END_MESSAGE, "smooth pursuit done")]
    if not starts or not ends:
        raise ValueError("fxsp panel start/end message not found")
    indices = np.where((tobii_ts > starts[0]) & (tobii_ts < ends[0]))[0]

    dom_eye, eye_fell_back = resolve_dominant_eye(data["Dom_Eye"])
    eye = data["data"].gaze.right if dom_eye == "r" else data["data"].gaze.left
    gaze_2d = eye.gazePoint.onDisplayArea
    panel = np.concatenate((gaze_2d[:, indices].T, tobii_ts[indices].reshape(-1, 1)), axis=1).astype(float)
    if len(panel) == 0:
        raise ValueError("no gaze samples inside fxsp panel 1")
    nan_ratio = (np.count_nonzero(np.isnan(panel)) // 2) / len(panel)
    if nan_ratio >= MAX_PANEL_NAN_RATIO:
        raise ValueError(f"panel 1 NaN ratio {nan_ratio:.3f} >= {MAX_PANEL_NAN_RATIO}")

    missing = np.isnan(panel[:, :2]).any(axis=1)  # NaN in the recording itself
    panel[:, 0], interp_x = _clean_outliers_single_eye(panel[:, 0])
    panel[:, 1], interp_y = _clean_outliers_single_eye(panel[:, 1])
    return {"gaze": panel, "interpolated": interp_x | interp_y, "missing_in_recording": missing,
            "messages": _panel_messages(messages),
            "dom_eye": dom_eye, "raw_dom_eye": str(data["Dom_Eye"]), "eye_fell_back": eye_fell_back,
            "mat_path": mat_path, "nan_ratio": nan_ratio}


# ---------------------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------------------
_MODEL = None


def _get_model():
    """gazeNET loaded once per process (same loader as ParticipantGazeDataManager.load_model)."""
    global _MODEL
    if _MODEL is None:
        import torch
        from GazeModel import model as gazeNET_model
        from constants import ModelPropertise
        net = gazeNET_model.gazeNET(len(ModelPropertise.EVENTS.value))
        net, _ = gazeNET_model.load(net, ModelPropertise.MODEL_PATH.value)
        net = net.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        net.eval()
        _MODEL = net
    return _MODEL


def annotate_model_based(panel_gaze):
    """Per-sample gazeNET event labels (0 unlabeled, 1 fixation, 2 saccade, 3 PSO),
    one per input row."""
    from GazeModel.recording_data_runner import generate_fixation_model_based
    annotated = generate_fixation_model_based(panel_gaze.copy(), model=_get_model())
    if len(annotated) != len(panel_gaze):
        raise RuntimeError(f"model returned {len(annotated)} rows for {len(panel_gaze)} samples")
    return annotated["evt"].to_numpy().astype(int)


def build_samples_df(panel_gaze, evt, sampling_rate_hz=SAMPLING_RATE_HZ, decimate=1):
    """Same layout as test_noam.py's samples_df (timestamp [ms], x_px, y_px, gaze_type)
    plus the raw evt code. decimate > 1 reproduces the old [::decimate] downsampling -
    used only by the validation in TESTS/validate_saccade_latency.py."""
    df = pd.DataFrame({
        "timestamp": (np.arange(len(panel_gaze)) * (1000 / sampling_rate_hz))[::decimate],
        "x_px": panel_gaze[::decimate, 0] * SCREEN_W_PX,
        "y_px": panel_gaze[::decimate, 1] * SCREEN_H_PX,
        "evt": evt[::decimate],
    })
    df["gaze_type"] = df["evt"].map(EVENT_NAMES).fillna("unlabeled")
    return df.dropna(subset=["timestamp", "x_px", "y_px"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------------------
def compute_trial_latencies(panel_messages, samples_df, sampling_rate_hz):
    """Vectorized, output-identical equivalent of Maysan's utils.saccade_latency_pm:
    for each panel message (trial onset, ms relative to the first message), take the
    first sample with timestamp >= onset, then the first saccade sample at or after it;
    latency = (number of samples from the first through the saccade sample, inclusive)
    / sampling rate, in seconds. Trials with no saccade before the end of the data get
    no entry, and keys are a running counter of trials WITH a saccade (not the message
    index) - both exactly as the original loop.

    Verified identical to the original loop on Maysan's own 100 Hz classified samples by
    TESTS/validate_saccade_latency.py."""
    ts = samples_df["timestamp"].to_numpy(dtype=float)
    is_saccade = (samples_df["gaze_type"] == SACCADE_GAZE_TYPE).to_numpy()
    saccade_idx = np.flatnonzero(is_saccade)
    base_ts = float(panel_messages[0][0] / 1000)
    latencies = {}
    for msg in panel_messages:
        onset = float(msg[0] / 1000) - base_ts
        first = np.searchsorted(ts, onset, side="left")
        k = np.searchsorted(saccade_idx, first, side="left")
        if k >= len(saccade_idx):
            break  # every later trial would also run off the end
        latencies[len(latencies)] = (saccade_idx[k] - first + 1) / sampling_rate_hz
    return latencies


def filter_latencies(latencies):
    """Drop the last entry (the panel-end message) and keep LATENCY_MIN_S..LATENCY_MAX_S."""
    keep = list(latencies.keys())[:-1]
    return {k: v for k, v in latencies.items() if k in keep and LATENCY_MIN_S <= v <= LATENCY_MAX_S}


def filled_in_rule(filled):
    """The BCEA interpolation rules (constants.BCEA_MAX_FILLED_IN_*), for one fixation.
    filled: per-sample bool, True where the sample was filled in by interpolation.
    Returns (reject, n_filled, longest_stretch); reject is None (keep), "long_stretch"
    (> MAX_FILLED_IN_STRETCH consecutive filled-in samples) or "high_pct"
    (>= MAX_FILLED_IN_FIXATION_PCT % of the fixation filled in)."""
    filled = np.asarray(filled, dtype=bool)
    n_filled = int(filled.sum())
    if n_filled == 0:
        return None, 0, 0
    fe = np.diff(np.concatenate(([0], filled.astype(np.int8), [0])))
    longest = int((np.flatnonzero(fe == -1) - np.flatnonzero(fe == 1)).max())
    if longest > MAX_FILLED_IN_STRETCH_SAMPLES:
        return "long_stretch", n_filled, longest
    if n_filled / len(filled) * 100 >= MAX_FILLED_IN_FIXATION_PCT:
        return "high_pct", n_filled, longest
    return None, n_filled, longest


def compute_fixation_bcea(samples_df, interpolated, panel_messages, sampling_rate_hz=SAMPLING_RATE_HZ):
    """One row per fixation (run of consecutive evt == FIXATION_IDX samples) in panel 1.
    BCEA uses feature_pipeline._bcea (the same formula/p as the SDMT fixation_bcea
    feature) on the normalized Tobii coordinates; bcea_deg2 is that value scaled - BCEA
    is proportional to sd_x * sd_y, and rho is scale-invariant, so the conversion is exact.
    Returns (kept_rows_df, counts_dict)."""
    from feature_pipeline import _bcea  # lazy: feature_pipeline imports this module

    if len(samples_df) != len(interpolated):
        raise RuntimeError(f"annotation has {len(samples_df)} rows but panel has {len(interpolated)} samples")
    is_fix = (samples_df["evt"] == FIXATION_IDX).to_numpy()
    edges = np.diff(np.concatenate(([0], is_fix.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)  # ends exclusive

    x = samples_df["x_px"].to_numpy() / SCREEN_W_PX
    y = samples_df["y_px"].to_numpy() / SCREEN_H_PX
    ts = samples_df["timestamp"].to_numpy()
    trial_onsets = np.array([m[0] for m in panel_messages], dtype=float) / 1000
    trial_onsets -= trial_onsets[0]
    min_samples = int(np.ceil(FIXATION_MIN_DURATION_MS * sampling_rate_hz / 1000))

    rows, n_short, n_interp, n_high_pct = [], 0, 0, 0
    for fix_idx, (s, e) in enumerate(zip(starts, ends)):
        if e - s < min_samples:
            n_short += 1
            continue
        filled = interpolated[s:e]
        reject, n_filled, longest_filled = filled_in_rule(filled)
        if reject == "long_stretch":
            n_interp += 1
            continue
        if reject == "high_pct":
            n_high_pct += 1
            continue
        bcea_norm = _bcea(list(zip(x[s:e], y[s:e])))
        rows.append({
            "fixation_idx": fix_idx,
            "trial_idx": int(np.searchsorted(trial_onsets, ts[s], side="right") - 1),
            "onset_ms": ts[s],
            "duration_ms": (e - s) * 1000 / sampling_rate_hz,
            "n_samples_in_bcea": e - s,  # every fixation sample, interpolated ones included
            "n_interpolated_samples_in_fixation": n_filled, "longest_filled_in_stretch": longest_filled,
            "mean_x_norm": x[s:e].mean(), "mean_y_norm": y[s:e].mean(),
            "std_x_norm": x[s:e].std(ddof=1), "std_y_norm": y[s:e].std(ddof=1),
            "mean_x_cm": x[s:e].mean() * SCREEN_WIDTH_CM, "mean_y_cm": y[s:e].mean() * SCREEN_HEIGHT_CM,
            "std_x_cm": x[s:e].std(ddof=1) * SCREEN_WIDTH_CM, "std_y_cm": y[s:e].std(ddof=1) * SCREEN_HEIGHT_CM,
            "bcea_norm": bcea_norm,
            "bcea_cm2": bcea_norm * CM2_PER_NORM_UNIT,
            "bcea_deg2": bcea_norm * DEG2_PER_NORM_UNIT,
        })
    counts = {"n_fixations_total": len(starts), "n_fixations_kept": len(rows),
              "n_fixations_too_short": n_short, "n_fixations_long_filled_in_gap": n_interp,
              "n_fixations_high_filled_in_pct": n_high_pct}
    return pd.DataFrame(rows), counts


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------
def annotated_cache_path(participant, mat_path, dom_eye, cache_dir=None):
    """The eye is part of the key (resolve_dominant_eye, 2026-11-10, changed which eye some
    participants get), and so is 'noaug': labels from before inference noise was turned
    off (GazeModel/recording_data_runner.py, 2026-11-10) are never reused."""
    cache_dir = cache_dir or pipeline_config.fxsp_annotation_cache_dir()
    mat_stem = os.path.splitext(os.path.basename(mat_path))[0]
    return os.path.join(cache_dir, f"{participant}__{mat_stem}__eye-{dom_eye}__panel1_model_600hz_noaug{_merge_tag()}.parquet")


def _merge_tag():
    """'_merged' when the merge-adjacent-fixations step (constants.MERGE_ADJACENT_FIXATIONS)
    is on, so merged and unmerged labels never share a cache file."""
    from constants import MERGE_ADJACENT_FIXATIONS
    return "_merged" if MERGE_ADJACENT_FIXATIONS else ""


def merge_cached_samples_df(samples_df):
    """Applies GazeModel's merge_adjacent_fixations to an already-annotated (unmerged)
    samples_df - same function and thresholds as at annotation time, so no re-annotation."""
    from GazeModel.recording_data_runner import merge_adjacent_fixations, normalized_to_deg
    x_deg, y_deg = normalized_to_deg(samples_df["x_px"] / SCREEN_W_PX, samples_df["y_px"] / SCREEN_H_PX)
    out = samples_df.copy()
    out["evt"] = merge_adjacent_fixations(out["evt"].to_numpy(), x_deg, y_deg, out["timestamp"].to_numpy())
    out["gaze_type"] = out["evt"].map(EVENT_NAMES).fillna("unlabeled")
    return out


def compiled_latencies_path(run_dir):
    return os.path.join(run_dir, OUTPUT_SUBDIR, "compiled_latencies.csv")


def fixation_bcea_path(run_dir):
    return os.path.join(run_dir, OUTPUT_SUBDIR, "fxsp_fixation_bcea_per_evt_all_partici.csv")


def process_participant(group, subject_path, overwrite_annotation=False, cache_dir=None):
    """Returns (latencies_dict_after_filtering, info_dict). gazeNET labels are cached in
    pipeline_config.fxsp_annotation_cache_dir() (shared across runs) so only
    participants never annotated before pay the ~40 s/participant model cost."""
    participant = os.path.basename(subject_path)
    panel = load_fxsp_fixation_panel(subject_path)
    cache = annotated_cache_path(participant, panel["mat_path"], panel["dom_eye"], cache_dir)
    unmerged = cache.replace("_noaug_merged.parquet", "_noaug.parquet")
    if not os.path.exists(cache) and unmerged != cache and os.path.exists(unmerged) and not overwrite_annotation:
        # the 2026-11-10 no-noise labels from before the merge step - merge, don't re-annotate
        merge_cached_samples_df(pd.read_parquet(unmerged)).to_parquet(cache, index=False)
    if os.path.exists(cache) and not overwrite_annotation:
        samples_df = pd.read_parquet(cache)
        if len(samples_df) != np.count_nonzero(~np.isnan(panel["gaze"][:, :2]).any(axis=1)):
            raise RuntimeError(f"cached annotation length mismatch for {participant} - rerun with --overwrite")
    else:
        evt = annotate_model_based(panel["gaze"])
        samples_df = build_samples_df(panel["gaze"], evt)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        samples_df.to_parquet(cache, index=False)

    raw = compute_trial_latencies(panel["messages"], samples_df, SAMPLING_RATE_HZ)
    latencies = filter_latencies(raw)
    info = {"participant": participant, "group": group, "dom_eye": panel["dom_eye"],
            "mat_file": os.path.basename(panel["mat_path"]), "panel_nan_ratio": panel["nan_ratio"],
            "eye_fell_back_to_r": panel["eye_fell_back"], "n_samples": len(samples_df), "n_trial_messages": len(panel["messages"]),
            "n_latencies_raw": len(raw), "n_latencies_kept": len(latencies),
            "saccade_sample_fraction": float((samples_df["gaze_type"] == SACCADE_GAZE_TYPE).mean()),
            "fixation_sample_fraction": float((samples_df["evt"] == FIXATION_IDX).mean()),
            "pso_sample_fraction": float((samples_df["evt"] == PSO_IDX).mean()),
            "unlabeled_sample_fraction": float((samples_df["evt"] == 0).mean()),
            "interpolated_sample_fraction": float(panel["interpolated"].mean())}

    fixations, fix_counts = compute_fixation_bcea(samples_df, panel["interpolated"], panel["messages"])
    fixations.insert(0, "group", group)
    fixations.insert(0, "participant", participant)
    info.update(fix_counts)
    return latencies, info, fixations


def build_latency_table(latency_pars):
    """compiled_latencies.csv rows - same schema as test_noam.py."""
    rows = [{"participant": p, "trial_idx": k, "saccade_latency_ms": v}
            for p, trials in latency_pars.items() for k, v in trials.items()]
    return pd.DataFrame(rows, columns=["participant", "trial_idx", "saccade_latency_ms"])


def main(run_dir=None, participant_whitelist=None, overwrite_annotation=False, input_data_path=None):
    """Called by feature_pipeline.main (compute mode) with its own run_dir, so each
    feature run gets its own <run_dir>/saccade_latency/compiled_latencies.csv; standalone,
    run_dir=None -> feature_analysis/<today> (same convention as duration_analysis.main).
    participant_whitelist: restrict to these participant names - only ever with a
    separate run_dir, so a subset never overwrites a full run's table.
    Returns the compiled_latencies.csv path."""
    if run_dir is None:
        from feature_pipeline import resolve_run_dir
        run_dir = resolve_run_dir("compute", base_dir=pipeline_config.feature_analysis_base_dir(ANNOTATION_METHOD))
    out_dir = os.path.join(run_dir, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    latency_pars, infos, skipped, fixation_tables = {}, [], [], []
    subjects = [(g, s) for g, s in iter_fxsp_subject_dirs(input_data_path)
                if participant_whitelist is None or os.path.basename(s) in participant_whitelist]
    for i, (group, subject_path) in enumerate(subjects, 1):
        participant = os.path.basename(subject_path)
        t0 = time.time()
        try:
            latencies, info, fixations = process_participant(group, subject_path, overwrite_annotation)
        except Exception as e:
            print(f"[{i}/{len(subjects)}] {group}/{participant}: SKIPPED - {e}")
            skipped.append({"participant": participant, "group": group, "reason": str(e)})
            continue
        infos.append(info)
        fixation_tables.append(fixations)
        # As test_noam.py: latency_pars is filled BEFORE the < MIN_TRIALS check, so such a
        # participant is still in compiled_latencies.csv - the check only drops them from
        # its per-participant plots/stats. Flagged in skipped_participants.csv here;
        # downstream reliability applies its own MIN_SACCADE_LATENCY_RELIABILITY.
        latency_pars[participant] = latencies
        if len(latencies) < MIN_TRIALS_PER_PARTICIPANT:
            skipped.append({"participant": participant, "group": group,
                            "reason": f"only {len(latencies)} valid latencies (< {MIN_TRIALS_PER_PARTICIPANT}); kept in table"})
        print(f"[{i}/{len(subjects)}] {group}/{participant}: {len(latencies)} trials, "
              f"mean {np.mean(list(latencies.values())) if latencies else np.nan:.4f} s ({time.time() - t0:.0f}s)")

    df_latencies = build_latency_table(latency_pars)
    csv_path = compiled_latencies_path(run_dir)
    df_latencies.to_csv(csv_path, index=False)
    if fixation_tables:
        pd.concat(fixation_tables, ignore_index=True).to_csv(fixation_bcea_path(run_dir), index=False)

    df_info = pd.DataFrame(infos)
    if not df_latencies.empty:
        stats = df_latencies.groupby("participant")["saccade_latency_ms"].agg(
            saccade_latency_s_mean="mean", saccade_latency_s_std="std").reset_index()
        df_info = df_info.merge(stats, on="participant", how="left")
    df_info.to_csv(os.path.join(out_dir, "participant_latency_summary.csv"), index=False)
    pd.DataFrame(skipped, columns=["participant", "group", "reason"]).to_csv(
        os.path.join(out_dir, "skipped_participants.csv"), index=False)

    print(f"\nSaved {len(df_latencies)} trial latencies for {len(latency_pars)} participants "
          f"-> {csv_path}  ({len(skipped)} skipped)")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=None, help="output run dir (default: feature_analysis/<today>)")
    parser.add_argument("--participants", nargs="+", default=None, help="restrict to these participant names")
    parser.add_argument("--overwrite", action="store_true", help="re-run gazeNET even if a cached annotation exists")
    args = parser.parse_args()
    main(run_dir=args.run_dir, participant_whitelist=set(args.participants) if args.participants else None,
         overwrite_annotation=args.overwrite)

"""
Sample-level time-budget sanity test (Noam, 2026-11-17).

For 5 random participant x panels that survived preprocessing (rows of the 12_11_26
panel_features.csv), independently of the trial/fixation machinery:

1. Raw panel length: last - first raw Tobii systemTimeStamp inside the panel window
   (ParticipantGazeDataManager slices the panel between its 'panel' and 'break'
   messages). The message-window length is reported next to it.
2. Sample-level "searches": maximal runs of consecutive annotated samples with
   y < dict_thresh (= H * DICTIONARY_BOUNDARY_RATIO, same constant SearchFinder.find uses;
   in normalized annotated coordinates this is simply y_norm < DICTIONARY_BOUNDARY_RATIO,
   since prepare_image_and_gaze maps y_px = y_norm * H). Run duration = t_last - t_first.
3. Everything that is not dictionary, split into buffer (dict <= y < text) and text
   (y >= text_thresh = H * TEXT_BOUNDARY_RATIO), same run logic.
4. Sample-level "encode": maximal runs of consecutive samples inside the same text ROI
   (RoiFinder ROI with idx >= 18 - the symbol/answer row, as TrialManager's
   symbol_index = press.idx + 18), same run logic.

Run duration as end - start drops one sample interval (~1.67 ms at 600 Hz) per run, plus
any timestamp gap; whatever the runs do not cover is reported as "unaccounted".

Input (Noam, 2026-11-17): the saved Stage 1 file panel_*_annotated_gaze_data.csv exactly
as it is - already annotated and already interpolated (no drift correction, no sample
removal). panel_*_interpolated_samples.csv is only read to report how many samples were
filled in.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

# The share is sometimes mounted as the Noam_M folder itself (/Volumes/Noam_M) rather than
# as /Volumes/ramot*/Noam_M - accept that layout too, for this script only.
_DIRECT_MOUNT = "/Volumes/Noam_M"
if os.path.isdir(os.path.join(_DIRECT_MOUNT, "Results", "Behavior", "processing_results")):
    pipeline_config.noam_m_root = lambda: _DIRECT_MOUNT

from pipeline_config import DICTIONARY_BOUNDARY_RATIO, TEXT_BOUNDARY_RATIO  # noqa: E402
from constants import KEY_TOBII_DATA, KEY_PANEL_MESSAGES, FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V  # noqa: E402
from participant_gaze_data_manager import ParticipantGazeDataManager  # noqa: E402
from utils import prepare_image_and_gaze  # noqa: E402
from RoiFinder import RoiFinder  # noqa: E402

N_PANELS = 5
RANDOM_SEED = 20261117
SOURCE_DATE = "12_11_26"
ANNOT_DATE = "11_11_26"
METHOD = "model_based"
FIRST_TEXT_ROI_IDX = 18

ROOT = os.path.join(pipeline_config.noam_m_root(), SOURCE_DATE)
PANEL_CSV = os.path.join(ROOT, "analysis_results", "features", "csvs", "panel_features.csv")
PARTICIPANT_CSV = os.path.join(ROOT, "analysis_results", "features", "csvs", "participant_features.csv")
OUT_DIR = os.path.join(ROOT, "analysis_results", "sdmt_decomposition_16_11_26", "test_sample_level_time_budget")


def run_durations(t, mask):
    """Sum and count of (t_last - t_first) over maximal runs of consecutive True in mask."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0.0, 0
    edges = np.diff(np.concatenate(([0], m.astype(int), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    return float(np.sum(t[ends] - t[starts])), len(starts)


def roi_runs(t, labels, valid):
    """Runs of consecutive samples inside the SAME text ROI (label >= 0)."""
    lab = np.where(valid, labels, -1)
    total, n = 0.0, 0
    i = 0
    while i < len(lab):
        if lab[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        total += t[j] - t[i]
        n += 1
        i = j + 1
    return total, n


def raw_panel_length(manager, panel):
    raw = manager.matched_data[panel][KEY_TOBII_DATA]
    ts = raw[:, 2].astype(float)
    msgs = manager.matched_data[panel][KEY_PANEL_MESSAGES]
    return (ts.max() - ts.min()) / 1000.0, (float(msgs[-1, 0]) - float(msgs[0, 0])) / 1000.0, len(ts)


def text_roi_labels(manager, panel, ann):
    img = manager.get_panel_img(panel)
    img_resized, aligned = prepare_image_and_gaze(img, ann)
    rois = [r for r in RoiFinder(panel, img_resized).rois if r.idx >= FIRST_TEXT_ROI_IDX]
    x = aligned[FIXATION_CSV_KEY_EYE_H].to_numpy()
    y = aligned[FIXATION_CSV_KEY_EYE_V].to_numpy()
    labels = np.full(len(x), -1)
    for r in rois:
        cx, cy = r.center
        inside = (x >= cx - r.x_half_left) & (x <= cx + r.x_half_right) & \
                 (y >= cy - r.y_half_up) & (y <= cy + r.y_half_down) & (labels < 0)
        labels[inside] = r.idx
    return labels


def analyse(pid, group, panel):
    manager = ParticipantGazeDataManager(os.path.join(pipeline_config.main_data_path(), group, pid),
                                         pipeline_config.main_data_path(), "SDMT", group)
    raw_len_ms, msg_len_ms, n_raw = raw_panel_length(manager, panel)
    ann = pipeline_config.load_annotated_csv(METHOD, group, pid, panel, date_str=ANNOT_DATE, corrected=False)
    interp_t = pipeline_config.load_interpolated_t(METHOD, group, pid, panel, date_str=ANNOT_DATE) or set()
    t = ann["t"].to_numpy(dtype=float)
    y = ann[FIXATION_CSV_KEY_EYE_V].to_numpy(dtype=float)
    is_interp = np.isin(np.round(t).astype(np.int64), np.fromiter(interp_t, dtype=np.int64)) if interp_t else np.zeros(len(t), bool)
    labels = text_roi_labels(manager, panel, ann)
    ann_len_ms = (t.max() - t.min()) / 1000.0

    rows = []
    for variant in ("saved_annotated_as_is",):
        valid = ~np.isnan(y)
        dict_m = valid & (y < DICTIONARY_BOUNDARY_RATIO)
        nondict_m = valid & (y >= DICTIONARY_BOUNDARY_RATIO)
        buffer_m = valid & (y >= DICTIONARY_BOUNDARY_RATIO) & (y < TEXT_BOUNDARY_RATIO)
        text_m = valid & (y >= TEXT_BOUNDARY_RATIO)
        offscreen_m = valid & ((y < 0) | (y > 1))
        search_ms, n_search = run_durations(t, dict_m)
        nondict_ms, n_nondict = run_durations(t, nondict_m)
        buffer_ms, _ = run_durations(t, buffer_m)
        text_ms, _ = run_durations(t, text_m)
        encode_ms, n_encode = roi_runs(t, labels, valid)
        rows.append({
            "participant": pid, "group": group, "panel": panel, "variant": variant,
            "raw_panel_ms": raw_len_ms, "message_window_ms": msg_len_ms, "annotated_span_ms": ann_len_ms,
            "n_raw_samples": n_raw, "n_annotated_samples": len(t),
            "n_interpolated_samples": int(is_interp.sum()),
            "interpolated_pct_samples": 100 * is_interp.mean(),
            "search_ms": search_ms / 1000, "n_searches": n_search,
            "nondict_ms": nondict_ms / 1000, "buffer_ms": buffer_ms / 1000, "text_ms": text_ms / 1000,
            "encode_ms": encode_ms / 1000, "n_encode_runs": n_encode,
            "offscreen_pct_samples": 100 * offscreen_m.mean(),
            "raw_minus_search_ms": raw_len_ms - search_ms / 1000,
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panels = pd.read_csv(PANEL_CSV, dtype={"panel": str})[["participant", "panel"]]
    groups = pd.read_csv(PARTICIPANT_CSV)[["participant", "group"]]
    panels = panels.merge(groups, on="participant", how="left", validate="m:1")
    panels["group"] = panels["group"].replace({"MS": "pwMS"})  # folder name on disk
    sample = panels.sample(N_PANELS, random_state=RANDOM_SEED)

    rows = []
    for _, r in sample.iterrows():
        print(f"--- {r.participant}/{r.panel} ({r.group})")
        rows += analyse(r.participant, r.group, r.panel)
    df = pd.DataFrame(rows)

    # percentages of the raw panel length
    for c in ["search_ms", "nondict_ms", "buffer_ms", "text_ms", "encode_ms"]:
        df[c.replace("_ms", "_pct")] = 100 * df[c] / df["raw_panel_ms"]
    df["unaccounted_pct"] = 100 - df["search_pct"] - df["nondict_pct"]
    df["nondict_vs_raw_minus_search_ms"] = df["nondict_ms"] - df["raw_minus_search_ms"]
    df.to_csv(os.path.join(OUT_DIR, "sample_level_time_budget.csv"), index=False)

    pd.set_option("display.width", 250)
    show = ["participant", "panel", "variant", "raw_panel_ms", "message_window_ms", "interpolated_pct_samples",
            "search_ms", "nondict_ms", "raw_minus_search_ms", "text_ms", "buffer_ms", "encode_ms"]
    print(df[show].round(0).to_string(index=False))
    pct = ["participant", "panel", "variant", "search_pct", "nondict_pct", "unaccounted_pct",
           "text_pct", "buffer_pct", "encode_pct", "n_searches", "n_encode_runs", "offscreen_pct_samples"]
    print(df[pct].round(2).to_string(index=False))
    print("out:", OUT_DIR)


if __name__ == "__main__":
    main()

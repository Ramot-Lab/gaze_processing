"""
Gaze-anchored, non-overlapping partition of SDMT panel time into per-trial components
(design agreed with Noam 2026-11-17).

Why: the existing time_in_relevant_fixations_ms is a first-to-last target-fixation SPAN
inside a press-bounded window, so it contains the trial's own dictionary visit, searches
assigned to neighbouring trials, other gaze, and time shared with adjacent trials; summed
over a panel it exceeds the panel length in 67% of panels. This module rebuilds the
time components from the SAME fixations / searches / ROIs / presses TrialManager uses,
without changing any existing feature.

1. Episodes. Every fixation that is not part of a SearchFinder search becomes T_r (inside
   text-row ROI r, ROI idx >= 18) or O (anything else); every search is one D episode.
2. Trial boundary b(r), gaze only: the first T_r fixation (after b(r-1), before the
   answer press of r+1) that is "committed" - followed by a D or a T_{r+1..r+2} before any
   T_{r-2..r-1}; ROI hits further than 2 targets away are stray (vertical misses) and
   ignored. A glance at r followed by
   a return to an earlier target is a preview and does not open trial r. If a later target
   commits before r does, r was never fixated: b(r) is undefined and trial r-1's window is
   extended to the next defined boundary (flag window_merged; the absorbed trial gets NaN
   and its time stays in the panel total exactly once).
3. Within window [b(r), b(r+1)) - disjoint by construction, so the parts sum exactly:
     encode_pre     = sum of T_r fixation durations before the first search starts
                      (no search -> all T_r time; a memory trial)
     encode_recheck = sum of T_r fixation durations between the first search's end and
                      the last search's start (only with >= 2 searches)
     encode_post    = sum of T_r fixation durations after the last search ends
     search         = sum of durations of searches starting in the window
     rest           = window - all of the above (saccades, other symbols, buffer, preview,
                      missing)
4. Presses are used only to select the same trial set as trial_features.csv (presses
   2 .. n-2, trial_idx = press.idx + 1) and as a QA check: answer_press_minus_last_search
   < 0 means the answer press for r came before r's last search ended (possible
   mis-assignment of a search to the wrong trial).

Panel totals are divided by S_panel (time per item, as 90/S) and by the real panel length.
Input: the same drift-corrected annotated CSVs the feature pipeline reads.
"""
import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

# The share is sometimes mounted as the Noam_M folder itself (/Volumes/Noam_M).
_DIRECT_MOUNT = "/Volumes/Noam_M"
if os.path.isdir(os.path.join(_DIRECT_MOUNT, "Results", "Behavior", "processing_results")):
    pipeline_config.noam_m_root = lambda: _DIRECT_MOUNT

SOURCE_DATE = "12_11_26"
ANNOT_DATE = "11_11_26"
METHOD = "model_based"
FIRST_TEXT_ROI_IDX = 18
ROI_OFFSET = 18          # TrialManager: symbol_index = press.idx + 18
US_PER_MS = 1000.0
N_WORKERS = 4
NEIGHBOURHOOD = 2     # targets r-2..r+2 count as task-relevant for boundary decisions
EXCLUDE_EDGE_PCT = 20.0  # Noam 2026-11-17: panel excluded from time-partition features above this % outside windows
MAX_MERGE = 10        # a window may absorb at most this many unfixated targets
TEST_PANELS = [("GR946", "l3"), ("ZA720", "0"), ("GR946", "0"), ("AG562", "a3"), ("GM552", "a5")]

ROOT = os.path.join(pipeline_config.noam_m_root(), SOURCE_DATE)
CSV_DIR = os.path.join(ROOT, "analysis_results", "features", "csvs")
BASE_OUT = os.path.join(ROOT, "analysis_results", "sdmt_decomposition_16_11_26")

PART_COLS = ["encode_pre", "encode_recheck", "encode_post", "search", "rest"]


# ---------------------------------------------------------------- episodes & boundaries
def build_episodes(tm):
    """Time-ordered list of (start, end, kind, label): kind 'D' (search), 'T' (text ROI
    `label`) or 'O'."""
    in_search = {id(f) for s in tm.searches for f in s.fixations}
    text_rois = [r for r in tm.rois if r.idx >= FIRST_TEXT_ROI_IDX]
    eps = [(s.start_time, s.end_time, "D", None) for s in tm.searches]
    for f in tm.fixations:
        if id(f) in in_search:
            continue
        label = next((r.idx for r in text_rois if r.contains(f.position)), None)
        eps.append((f.start_time, f.end_time, "T" if label is not None else "O", label))
    eps.sort(key=lambda e: e[0])
    return eps


def _committed(eps, i, near=None):
    """T episode i (target r) is committed if a D or a T_{r+1..r+near} follows before any
    T_{r-near..r-1}. Labels further than `near` from r are stray hits (e.g. a vertical
    miss into the neighbouring row) and are treated as O."""
    near = near or NEIGHBOURHOOD
    r = eps[i][3]
    for _, _, kind, label in eps[i + 1:]:
        if kind == "D" or (kind == "T" and r < label <= r + near):
            return True
        if kind == "T" and r - near <= label < r:
            return False
    return True  # end of panel


def find_boundaries(eps, targets, horizons):
    """{roi_idx: episode index of b(r) or None}, monotonic in time. b(r) is looked for only
    between b(r-1) and horizons[r] (the answer press of r+1 - the same look-ahead limit
    TrialManager._extract_trials uses), so one bad boundary cannot cascade."""
    bounds, pos = {}, 0
    for r in targets:
        found = None
        for i in range(pos, len(eps)):
            kind, label = eps[i][2], eps[i][3]
            if eps[i][0] > horizons[r]:
                break
            if kind != "T" or not (r - NEIGHBOURHOOD <= label <= r + NEIGHBOURHOOD):
                continue
            if label == r and _committed(eps, i):
                found = i
                break
            if label > r and _committed(eps, i):
                break  # a later target took over: r never fixated
        bounds[r] = found
        if found is not None:
            pos = found
    return bounds


def partition_window(eps, i0, i1, targets_in_window):
    """Disjoint time parts of window [eps[i0].start, eps[i1].start) (i1 None -> last episode end)."""
    ws, we = eps[i0][0], eps[i1][0]
    inside = [e for e in eps if ws <= e[0] < we]
    searches = [e for e in inside if e[2] == "D"]
    tfix = [e for e in inside if e[2] == "T" and e[3] in targets_in_window]
    out = {p: 0.0 for p in PART_COLS}
    out["search"] = sum(e[1] - e[0] for e in searches)
    if searches:
        first_start, first_end = searches[0][0], searches[0][1]
        last_start, last_end = searches[-1][0], searches[-1][1]
    for s, e, _, _ in tfix:
        d = e - s
        if not searches or e <= first_start:
            out["encode_pre"] += d
        elif s >= last_end:
            out["encode_post"] += d
        elif s >= first_end and e <= last_start:
            out["encode_recheck"] += d
        else:  # cannot overlap a search by construction; keep any edge case visible
            out["encode_recheck"] += d
    window = we - ws
    out["rest"] = window - sum(out[p] for p in PART_COLS if p != "rest")
    return ws, we, window, len(searches), (searches[-1][1] if searches else np.nan), out


# ---------------------------------------------------------------- one panel
def process_panel(pid, group, panel):
    from participant_gaze_data_manager import ParticipantGazeDataManager
    from trial_manager import TrialManager
    folder_group = "pwMS" if group == "MS" else group
    mgr = ParticipantGazeDataManager(os.path.join(pipeline_config.main_data_path(), folder_group, pid),
                                     pipeline_config.main_data_path(), "SDMT", folder_group)
    return partition_panel(mgr, pid, group, panel)


def partition_panel(mgr, pid, group, panel):
    from trial_manager import TrialManager
    folder_group = "pwMS" if group == "MS" else group
    ann = pipeline_config.load_annotated_csv(METHOD, folder_group, pid, panel, date_str=ANNOT_DATE, corrected=True)
    tm = TrialManager(mgr, panel, annotated_data=ann, annotation_method=METHOD)
    panel_len = float(ann["t"].max() - ann["t"].min())
    eps = build_episodes(tm)

    presses = tm.presses
    trial_press_idx = list(range(2, len(presses) - 2))           # same trial set as _extract_trials
    targets = [presses[i].idx + ROI_OFFSET for i in trial_press_idx]
    # boundaries for every analysed target plus a few following ones (window ends)
    extra = [targets[-1] + j for j in range(1, MAX_MERGE + 1)] if targets else []
    horizons = {r: presses[min(pi + 2, len(presses) - 1)].time for pi, r in zip(trial_press_idx, targets)}
    horizons.update({r: presses[-1].time for r in extra})
    bounds = find_boundaries(eps, targets + extra, horizons)

    rows = []
    for k, (pi, r) in enumerate(zip(trial_press_idx, targets)):
        row = {"participant": pid, "group": group, "panel": panel, "trial_idx": presses[pi].idx + 1,
               "target_roi": r, "target_fixated": bounds[r] is not None}
        if bounds[r] is None:
            row.update({"absorbed_into_previous": True, "window_merged": False})
            rows.append(row)
            continue
        nxt, covered = r + 1, [r]
        while nxt in bounds and bounds[nxt] is None and nxt <= r + MAX_MERGE:
            covered.append(nxt)
            nxt += 1
        i1 = bounds.get(nxt)
        if i1 is None:  # no later boundary within reach: leave the trial undefined
            row.update({"absorbed_into_previous": False, "window_merged": False, "window_undefined": True})
            rows.append(row)
            continue
        ws, we, window, n_s, last_search_end, parts = partition_window(eps, bounds[r], i1, set(covered))
        answer_press = presses[pi + 1].time
        row.update({"absorbed_into_previous": False, "window_merged": len(covered) > 1, "window_undefined": False,
                    "n_targets_in_window": len(covered), "window_start_us": ws, "window_end_us": we,
                    "window_ms": window / US_PER_MS, "n_searches_gaze": n_s,
                    "answer_press_minus_last_search_ms": (answer_press - last_search_end) / US_PER_MS
                    if n_s else np.nan,
                    **{f"{p}_ms": v / US_PER_MS for p, v in parts.items()}})
        rows.append(row)
    trials = pd.DataFrame(rows)
    trials["panel_length_ms"] = panel_len / US_PER_MS
    for k, v in {**coarse_fixation_time(tm), **row_alignment(tm)}.items():
        trials[k] = v

    window_sum = trials["window_ms"].sum() if "window_ms" in trials else 0.0
    if 100 * (1 - window_sum / (panel_len / US_PER_MS)) > EXCLUDE_EDGE_PCT:
        save_gaze_heatmap(tm, ann, pid, group, panel)
    return trials


# ---------------------------------------------------------------- coarse measures & QA
def coarse_fixation_time(tm):
    """Whole-panel, segmentation-free: total fixation duration inside any dictionary ROI
    (idx < 18) and inside any text-row symbol ROI (idx >= 18), regardless of target."""
    dict_rois = [r for r in tm.rois if r.idx < FIRST_TEXT_ROI_IDX]
    text_rois = [r for r in tm.rois if r.idx >= FIRST_TEXT_ROI_IDX]
    out = {"coarse_dict_roi_fix_ms": 0.0, "coarse_text_roi_fix_ms": 0.0, "coarse_all_fix_ms": 0.0}
    for f in tm.fixations:
        d = (f.end_time - f.start_time) / US_PER_MS
        out["coarse_all_fix_ms"] += d
        if any(r.contains(f.position) for r in dict_rois):
            out["coarse_dict_roi_fix_ms"] += d
        elif any(r.contains(f.position) for r in text_rois):
            out["coarse_text_roi_fix_ms"] += d
    return out


def row_alignment(tm):
    """Is text-area gaze on the row of the symbol currently being answered? For every
    non-search fixation inside a text ROI between press 2 and press n-2: row of the fixated
    ROI minus row of the expected target (target of the latest press). A systematic
    non-zero offset = vertical drift of the text-area gaze (e.g. GS739/i1, 2026-11-17)."""
    text_rois = [r for r in tm.rois if r.idx >= FIRST_TEXT_ROI_IDX]
    ys = np.array(sorted(r.center[1] for r in text_rois))
    row_centres = [ys[0]]
    for y in ys[1:]:
        if y - row_centres[-1] > 20:
            row_centres.append(y)
    row_of = {r.idx: int(np.argmin([abs(r.center[1] - c) for c in row_centres])) for r in text_rois}
    in_search = {id(f) for s in tm.searches for f in s.fixations}
    press_t = np.array([p.time for p in tm.presses])
    t0, t1 = tm.presses[2].time, tm.presses[-3].time
    diffs, times = [], []
    for f in tm.fixations:
        if id(f) in in_search or not (t0 <= f.start_time < t1):
            continue
        roi = next((r for r in text_rois if r.contains(f.position)), None)
        if roi is None:
            continue
        j = int(np.searchsorted(press_t, f.start_time, side="right") - 1)
        expected = tm.presses[j].idx + ROI_OFFSET
        if expected not in row_of:
            continue
        diffs.append(row_of[roi.idx] - row_of[expected])
        times.append(f.start_time)
    if not diffs:
        return {"rowcheck_n_fix": 0, "rowcheck_pct_on_expected_row": np.nan, "rowcheck_median_offset": np.nan,
                "rowcheck_median_offset_first_half": np.nan, "rowcheck_median_offset_second_half": np.nan}
    diffs, times = np.array(diffs), np.array(times)
    half = times < (t0 + t1) / 2
    return {"rowcheck_n_fix": len(diffs), "rowcheck_pct_on_expected_row": 100 * float(np.mean(diffs == 0)),
            "rowcheck_median_offset": float(np.median(diffs)),
            "rowcheck_median_offset_first_half": float(np.median(diffs[half])) if half.any() else np.nan,
            "rowcheck_median_offset_second_half": float(np.median(diffs[~half])) if (~half).any() else np.nan}


def save_gaze_heatmap(tm, ann, pid, group, panel):
    """Heatmap of ALL gaze samples over the panel image (log density), text-row ROIs drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    from utils import prepare_image_and_gaze
    from feature_pipeline import demographics_text, annotate_demographics
    from constants import FIXATION_CSV_KEY_EYE_H, FIXATION_CSV_KEY_EYE_V
    out_dir = os.path.join(BASE_OUT, "gaze_partition", "excluded_panel_heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    _, aligned = prepare_image_and_gaze(tm.img_resized, ann)  # image already at screen height
    x = aligned[FIXATION_CSV_KEY_EYE_H].to_numpy()
    y = aligned[FIXATION_CSV_KEY_EYE_V].to_numpy()
    h, w = tm.img_resized.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(tm.img_resized, extent=(0, w, h, 0))
    hist, xe, ye = np.histogram2d(x, y, bins=[np.linspace(0, w, 90), np.linspace(0, h, 110)])
    hist = np.ma.masked_equal(hist.T, 0)
    im = ax.imshow(hist, extent=(0, w, h, 0), cmap="viridis", norm=LogNorm(), alpha=0.65)
    for r in tm.rois:
        cx, cy = r.center
        ax.add_patch(Rectangle((cx - r.x_half_left, cy - r.y_half_up), r.x_half_left + r.x_half_right,
                               r.y_half_up + r.y_half_down, fill=False, lw=0.3, ec="tab:red"))
    off = np.mean((x < 0) | (x > w) | (y < 0) | (y > h)) * 100
    ax.set_xlim(-0.05 * w, 1.05 * w)
    ax.set_ylim(1.05 * h, -0.05 * h)
    ax.set_title(f"{pid}/{panel} - all gaze samples (log density); {off:.1f}% outside the panel")
    fig.colorbar(im, ax=ax, fraction=0.03, label="samples per bin")
    ax.axis("off")
    part = pd.read_csv(os.path.join(CSV_DIR, "participant_features.csv"))
    annotate_demographics(demographics_text(part[part["participant"] == pid]), ax=ax)
    fig.savefig(os.path.join(out_dir, f"heatmap_{pid}_{panel}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- aggregation
def panel_summary(trials, scores):
    t = trials[trials["window_ms"].notna()]
    ms_cols = [f"{p}_ms" for p in PART_COLS] + ["window_ms"]
    g = t.groupby(["participant", "group", "panel"])
    s = g[ms_cols].sum()
    s["panel_length_ms"] = g["panel_length_ms"].first()
    s["n_trials"] = trials.groupby(["participant", "group", "panel"]).size()
    s["n_absorbed"] = trials.groupby(["participant", "group", "panel"])["absorbed_into_previous"].sum()
    s["n_merged_windows"] = g["window_merged"].sum()
    s["n_undefined"] = trials.groupby(["participant", "group", "panel"])["window_undefined"].apply(lambda x: (x == True).sum())
    s["n_press_before_search_end"] = g["answer_press_minus_last_search_ms"].apply(lambda x: (x < 0).sum())
    panel_const = [c for c in trials.columns if c.startswith(("coarse_", "rowcheck_"))]
    s = s.join(trials.groupby(["participant", "group", "panel"])[panel_const].first())
    s = s.reset_index().merge(scores, on=["participant", "panel"], how="left", validate="1:1")
    s["edge_time_ms"] = s["panel_length_ms"] - s["window_ms"]
    for p in PART_COLS:
        s[f"time_T_{p}"] = s[f"{p}_ms"] / s["S_panel"]
        s[f"{p}_pct_of_panel"] = 100 * s[f"{p}_ms"] / s["panel_length_ms"]
    s["time_T_item"] = 90000.0 / s["S_panel"]
    s["time_T_window"] = s["window_ms"] / s["S_panel"]
    s["edge_pct_of_panel"] = 100 * s["edge_time_ms"] / s["panel_length_ms"]
    s["excluded_time_partition"] = s["edge_pct_of_panel"] > EXCLUDE_EDGE_PCT
    # items whose time lies inside the analysed windows (a merged window holds >1 item)
    s["n_items_in_windows"] = t.groupby(["participant", "group", "panel"])["n_targets_in_window"].sum().reindex(
        pd.MultiIndex.from_frame(s[["participant", "group", "panel"]])).to_numpy()
    for p in PART_COLS:
        s[f"time_Ttrial_{p}"] = s[f"{p}_ms"] / s["n_items_in_windows"]  # per analysed item
    s["time_Ttrial_window"] = s["window_ms"] / s["n_items_in_windows"]
    for c in ["coarse_dict_roi_fix_ms", "coarse_text_roi_fix_ms", "coarse_all_fix_ms"]:
        s[c.replace("_ms", "").replace("coarse_", "time_Tcoarse_")] = s[c] / s["S_panel"]
    s["parts_sum_minus_window_ms"] = s[[f"{p}_ms" for p in PART_COLS]].sum(axis=1) - s["window_ms"]
    return s


def load_panel_list(test):
    panels = pd.read_csv(os.path.join(CSV_DIR, "panel_features.csv"), dtype={"panel": str})[["participant", "panel"]]
    groups = pd.read_csv(os.path.join(CSV_DIR, "participant_features.csv"))[["participant", "group"]]
    panels = panels.merge(groups, on="participant", how="left", validate="m:1")
    if test:
        keep = pd.DataFrame(TEST_PANELS, columns=["participant", "panel"])
        panels = panels.merge(keep, on=["participant", "panel"], how="inner")
    return panels


def _process_participant(pid, group, panel_list):
    from participant_gaze_data_manager import ParticipantGazeDataManager
    folder_group = "pwMS" if group == "MS" else group
    out, errors = [], []
    try:
        mgr = ParticipantGazeDataManager(os.path.join(pipeline_config.main_data_path(), folder_group, pid),
                                         pipeline_config.main_data_path(), "SDMT", folder_group)
    except Exception as e:
        return out, [{"participant": pid, "panel": None, "error": repr(e)}]
    for panel in panel_list:
        try:
            out.append(partition_panel(mgr, pid, group, panel))
        except Exception as e:
            errors.append({"participant": pid, "panel": panel, "error": repr(e),
                           "trace": traceback.format_exc(limit=3)})
    return out, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="5 test panels only, written to gaze_partition_test/")
    args = ap.parse_args()
    # separate output folders so a test run can never overwrite the full run
    out_dir = os.path.join(BASE_OUT, "gaze_partition_test" if args.test else "gaze_partition")
    os.makedirs(out_dir, exist_ok=True)

    panels = load_panel_list(args.test)
    scores = pd.read_csv(os.path.join(ROOT, "behavior_scores", "all_scores.csv"), dtype={"Panel": str}).rename(
        columns={"Participant": "participant", "Panel": "panel", "SDMT_Score": "S_panel"})[["participant", "panel", "S_panel"]]

    jobs = {(r.participant, r.group): [] for r in panels.itertuples()}
    for r in panels.itertuples():
        jobs[(r.participant, r.group)].append(r.panel)

    results, errors = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_process_participant, pid, grp, pl): pid for (pid, grp), pl in jobs.items()}
        for n, fut in enumerate(as_completed(futs), 1):
            res, err = fut.result()
            results += res
            errors += err
            print(f"[{n}/{len(futs)}] {futs[fut]} panels={len(res)} errors={len(err)}", flush=True)

    trials = pd.concat(results, ignore_index=True)
    summary = panel_summary(trials, scores)
    trials.to_csv(os.path.join(out_dir, "trial_time_partition.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "panel_time_partition.csv"), index=False)
    pd.DataFrame(errors).to_csv(os.path.join(out_dir, "errors.csv"), index=False)

    pd.set_option("display.width", 250)
    print(f"panels: {len(summary)} / {len(panels)} requested; errors: {len(errors)}")
    print("max |parts sum - window| ms:", summary["parts_sum_minus_window_ms"].abs().max())
    show = ["participant", "panel", "panel_length_ms", "window_ms", "edge_pct_of_panel"] + \
           [f"{p}_pct_of_panel" for p in PART_COLS] + ["n_absorbed", "n_merged_windows", "n_undefined", "n_press_before_search_end"]
    print(summary[show].round(2).to_string(index=False) if args.test else summary[show[2:]].describe().round(2).to_string())
    print("out:", out_dir)


if __name__ == "__main__":
    main()

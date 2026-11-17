"""
Where is the gaze in panels whose targets are rarely hit by the gaze partition?
(Noam, 2026-11-17)

For every panel with > EDGE_PCT_LIMIT % of its length outside the analysed gaze windows
(gaze_partition/panel_time_partition.csv), plus two normal reference panels: rebuild the
TrialManager exactly as the feature pipeline does (drift-corrected annotated CSV) and split
the panel's fixation time into
  search        - fixations inside SearchFinder searches (dictionary)
  text_in_roi   - non-search fixations inside any text-row ROI (idx >= 18)
  text_off_roi  - non-search fixations below text_thresh but in no ROI
  buffer        - between dict_thresh and text_thresh
  dict_no_search- above dict_thresh but not part of a search (should be ~0)
  offscreen     - outside the screen
For text_off_roi fixations: signed vertical distance to the nearest ROI centre (px,
positive = below the symbol row centre), to tell a systematic offset from scatter.
Also saves one fixation-over-panel image per panel (ROIs drawn).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

_DIRECT_MOUNT = "/Volumes/Noam_M"
if os.path.isdir(os.path.join(_DIRECT_MOUNT, "Results", "Behavior", "processing_results")):
    pipeline_config.noam_m_root = lambda: _DIRECT_MOUNT

from pipeline_config import DICTIONARY_BOUNDARY_RATIO, TEXT_BOUNDARY_RATIO  # noqa: E402
from constants import SCREEN_SIZE  # noqa: E402

EDGE_PCT_LIMIT = 20.0
REFERENCE = [("AG562", "MS", "a3"), ("GR946", "HC", "0")]
BASE = os.path.join(pipeline_config.noam_m_root(), "12_11_26", "analysis_results", "sdmt_decomposition_16_11_26")
OUT_DIR = os.path.join(BASE, "gaze_partition", "bad_panel_diagnosis")


def diagnose(pid, group, panels):
    from participant_gaze_data_manager import ParticipantGazeDataManager
    from trial_manager import TrialManager
    fg = "pwMS" if group == "MS" else group
    mgr = ParticipantGazeDataManager(os.path.join(pipeline_config.main_data_path(), fg, pid),
                                     pipeline_config.main_data_path(), "SDMT", fg)
    H, W = SCREEN_SIZE
    rows = []
    for panel in panels:
        ann = pipeline_config.load_annotated_csv("model_based", fg, pid, panel, date_str="11_11_26", corrected=True)
        tm = TrialManager(mgr, panel, annotated_data=ann, annotation_method="model_based")
        in_search = {id(f) for s in tm.searches for f in s.fixations}
        rois = [r for r in tm.rois if r.idx >= 18]
        centers = np.array([r.center for r in rois])
        cat_ms = dict.fromkeys(["search", "text_in_roi", "text_off_roi", "buffer", "dict_no_search", "offscreen"], 0.0)
        dy_off, pts = [], []
        for f in tm.fixations:
            x, y = f.position
            d = (f.end_time - f.start_time) / 1000
            if id(f) in in_search:
                c = "search"
            elif not (0 <= y <= H and 0 <= x <= tm.img_resized.shape[1]):
                c = "offscreen"
            elif y < H * DICTIONARY_BOUNDARY_RATIO:
                c = "dict_no_search"
            elif y < H * TEXT_BOUNDARY_RATIO:
                c = "buffer"
            elif any(r.contains((x, y)) for r in rois):
                c = "text_in_roi"
            else:
                c = "text_off_roi"
                j = np.argmin(np.hypot(centers[:, 0] - x, centers[:, 1] - y))
                dy_off.append(y - centers[j, 1])
            cat_ms[c] += d
            pts.append((x, y, d, c))
        total = sum(cat_ms.values())
        rows.append({"participant": pid, "group": group, "panel": panel, "fixation_time_s": total / 1000,
                     **{f"{k}_pct": 100 * v / total for k, v in cat_ms.items()},
                     "off_roi_median_dy_px": float(np.median(dy_off)) if dy_off else np.nan,
                     "roi_half_height_px": float(np.median([r.y_half_up + r.y_half_down for r in rois]) / 2)})
        # fixation map
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(tm.img_resized)
        for r in rois:
            cx, cy = r.center
            ax.add_patch(Rectangle((cx - r.x_half_left, cy - r.y_half_up), r.x_half_left + r.x_half_right,
                                   r.y_half_up + r.y_half_down, fill=False, lw=0.4, ec="tab:blue"))
        colors = {"search": "tab:orange", "text_in_roi": "tab:green", "text_off_roi": "tab:red",
                  "buffer": "tab:purple", "dict_no_search": "tab:brown", "offscreen": "black"}
        for c, col in colors.items():
            p = [q for q in pts if q[3] == c]
            if p:
                ax.scatter([q[0] for q in p], [q[1] for q in p], s=[max(q[2] / 20, 1) for q in p], c=col,
                           alpha=0.5, label=c, lw=0)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"{pid}/{panel} ({group}) - fixations (size = duration)")
        ax.axis("off")
        fig.savefig(os.path.join(OUT_DIR, f"fixation_map_{pid}_{panel}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s = pd.read_csv(os.path.join(BASE, "gaze_partition", "panel_time_partition.csv"), dtype={"panel": str})
    bad = s[s["edge_pct_of_panel"] > EDGE_PCT_LIMIT][["participant", "group", "panel", "edge_pct_of_panel"]]
    jobs = {}
    for r in list(bad.itertuples(index=False)) + [(p, g, pn, np.nan) for p, g, pn in REFERENCE]:
        jobs.setdefault((r[0], r[1]), []).append(r[2])
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(diagnose, pid, grp, pl) for (pid, grp), pl in jobs.items()]
        for f in futs:
            rows += f.result()
    df = pd.DataFrame(rows).merge(bad, on=["participant", "group", "panel"], how="left")
    df["reference"] = df["edge_pct_of_panel"].isna()
    df = df.sort_values(["reference", "edge_pct_of_panel"], ascending=[True, False])
    df.to_csv(os.path.join(OUT_DIR, "bad_panel_gaze_location.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

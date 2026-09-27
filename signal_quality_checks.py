"""
Low-level raw-signal integrity checks (decision 2026-09-27, prompted by the ER635/l3
anomaly found in duration_analysis.py) - pure functions over one panel's annotated
DataFrame (t, eye_horizontal, eye_vertical, status, evt), no other dependencies, so both
run_preprocessing.py (a real exclusion criterion) and check_frozen_gaze.py (population
diagnostic scans) can import them without a circular import.
"""
import numpy as np
import pandas as pd

# A frozen/stuck tracker signal: (x, y) staying EXACTLY constant for this long is not a
# real fixation (genuine fixations still have microsaccadic jitter).
FREEZE_THRESHOLD_MS = 5000

# Normal sampling is ~1667us (~600Hz) throughout this dataset - a 1000ms gap is ~600x
# that, comfortably past any real blink/tracker-loss span, so no genuine physiological
# event could look like this (ER635/l3: a single ~35s row-to-row jump in t with no
# missing rows and no change in position).
TIMESTAMP_GAP_THRESHOLD_MS = 1000


def find_frozen_blocks(annotated_df, threshold_ms=FREEZE_THRESHOLD_MS):
    """Runs of consecutive samples with EXACTLY equal (eye_horizontal, eye_vertical),
    regardless of the status/evt columns (a frozen tracker can still be flagged valid
    and classified as a very long "fixation" - that's exactly the failure mode this is
    meant to catch, not something to filter out before looking). Returns a DataFrame,
    one row per block at/above threshold_ms: t_min, t_max, duration_ms, x, y, n_samples."""
    x = annotated_df["eye_horizontal"].to_numpy()
    y = annotated_df["eye_vertical"].to_numpy()
    t = annotated_df["t"].to_numpy()
    if len(x) == 0:
        return pd.DataFrame(columns=["t_min", "t_max", "duration_ms", "x", "y", "n_samples"])

    same_as_prev = np.empty(len(x), dtype=bool)
    same_as_prev[0] = False
    same_as_prev[1:] = (x[1:] == x[:-1]) & (y[1:] == y[:-1])
    run_id = np.cumsum(~same_as_prev)

    runs = pd.DataFrame({"run_id": run_id, "x": x, "y": y, "t": t}).groupby("run_id").agg(
        x=("x", "first"), y=("y", "first"), t_min=("t", "min"), t_max=("t", "max"), n_samples=("t", "size"))
    runs["duration_ms"] = (runs["t_max"] - runs["t_min"]) / 1000.0
    return runs[runs["duration_ms"] > threshold_ms].reset_index(drop=True)


def find_timestamp_gaps(annotated_df, threshold_ms=TIMESTAMP_GAP_THRESHOLD_MS):
    """Row-to-row jumps in t (raw microsecond timestamps) bigger than threshold_ms -
    catches the ER635/l3 pattern (single huge gap, no missing rows, position barely
    changes across it) that find_frozen_blocks does NOT catch, since nothing there is
    actually frozen. Returns one row per gap: t_before/t_after/gap_ms."""
    t = annotated_df["t"].to_numpy()
    if len(t) < 2:
        return pd.DataFrame(columns=["t_before", "t_after", "gap_ms"])
    delta_ms = np.diff(t) / 1000.0
    gap_positions = np.where(delta_ms > threshold_ms)[0]
    return pd.DataFrame({
        "t_before": t[gap_positions], "t_after": t[gap_positions + 1], "gap_ms": delta_ms[gap_positions],
    })

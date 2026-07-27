"""
Quality-control sweep over annotated gaze data CSVs.

For every panel_*_annotated_gaze_data.csv under DATA_ROOT/{group}/{participant}/,
checks:
  1. Dictionary engagement depth, graded from the median eye_vertical of dictionary-
     region fixations (eye_vertical < UPPER_Y_THRESHOLD):
       median > DICT_LOWER_Y_THRESHOLD (0.15)  -> error 1 "no dictionary engagement"
       median > DICT_MID_Y_THRESHOLD   (0.13)  -> error 2 "shallow dictionary engagement"
       median > SYMBOLS_Y_THRESHOLD    (0.075) -> error 3 "numbers only, not symbols"
       else, if < SYMBOLS_FIXATION_PCT_MIN (15%) of dictionary fixations reach the
       symbols row (eye_vertical < SYMBOLS_Y_THRESHOLD) -> error 4 "rarely on symbols"
       otherwise -> no error (good engagement with the dictionary)
  2. % of rows with evt == 1 (fixation events).
     At or below FIXATION_PCT_MIN -> error 5 "no fixations"
  3. Horizontal range used (max(eye_horizontal) - min(eye_horizontal)), compared to
     the cohort's own median range for that panel.
     Below SPACE_USED_PCT_MIN (70%) of the cohort median -> error 6 "narrow horizontal range"

Outputs a table with one row per file: group, participant, panel,
pct_upper, pct_fixation, x_span_pct, errors, error_labels.
"""

import glob
import os

import pandas as pd
# input_date = pd.Timestamp.now().strftime("%d_%m_%Y")  # looks like: "26_07_2023"
input_date = "19_07_2026"  # hard-coded for now, to match the data folder name
output_date = "22_07_2026"  # date stamp for the output results table
DATA_ROOT = f"/Volumes/ramot/Noam_M/SDMT_annotated_gaze_data_{input_date}"
OUTPUT_DIR = "/Volumes/ramot/Noam_M/visualized_data"
GROUPS = ["HC", "pwMS"]

UPPER_Y_THRESHOLD = 0.175  # bottom edge of the dictionary/key region
DICT_LOWER_Y_THRESHOLD = 0.15  # median in the dictionary below here -> not really engaging with it
DICT_MID_Y_THRESHOLD = 0.13  # median below here -> only reaching the lower part of the dictionary
SYMBOLS_Y_THRESHOLD = 0.075  # symbols row is above this line, numbers row is below it
SYMBOLS_FIXATION_PCT_MIN = 15.0  # min % of dictionary fixations that must reach the symbols row to count as good

FIXATION_PCT_MIN = 15.0

SPACE_USED_PCT_MIN = 70.0  # flag panels using less than this % of the cohort's median horizontal range

ERROR_LABELS = {
    1: "no dictionary engagement",
    2: "shallow dictionary engagement",
    3: "numbers only, not symbols",
    4: "rarely on symbols",
    5: "no fixations",
    6: "narrow horizontal range",
}


def classify_dictionary_engagement(df):
    """Grade dictionary engagement depth from the median vertical position of dictionary-region fixations.

    Returns an error code (1-4, worst to mildest problem) or None if engagement is good.
    """
    dict_fixations = df[(df["evt"] == 1) & (df["eye_vertical"] < UPPER_Y_THRESHOLD)]

    if dict_fixations.empty:
        return 1

    median_y = dict_fixations["eye_vertical"].median()
    if median_y > DICT_LOWER_Y_THRESHOLD:
        return 1
    if median_y > DICT_MID_Y_THRESHOLD:
        return 2
    if median_y > SYMBOLS_Y_THRESHOLD:
        return 3

    pct_on_symbols = (dict_fixations["eye_vertical"] < SYMBOLS_Y_THRESHOLD).mean() * 100
    if pct_on_symbols < SYMBOLS_FIXATION_PCT_MIN:
        return 4
    return None


def check_file(path):
    df = pd.read_csv(path)

    pct_upper = (df["eye_vertical"] < UPPER_Y_THRESHOLD).mean() * 100
    pct_fixation = (df["evt"] == 1).mean() * 100
    x_span_pct = (df["eye_horizontal"].max() - df["eye_horizontal"].min()) * 100

    errors = []
    dict_engagement_error = classify_dictionary_engagement(df)
    if dict_engagement_error is not None:
        errors.append(dict_engagement_error)
    if pct_fixation <= FIXATION_PCT_MIN:
        errors.append(5)

    return pct_upper, pct_fixation, x_span_pct, errors


def run_checks(data_root=DATA_ROOT, groups=GROUPS):
    rows = []
    for group in groups:
        group_dir = os.path.join(data_root, group)
        if not os.path.isdir(group_dir):
            continue
        for participant_dir in sorted(glob.glob(os.path.join(group_dir, "*"))):
            if not os.path.isdir(participant_dir):
                continue
            participant = os.path.basename(participant_dir)
            for csv_path in sorted(glob.glob(os.path.join(participant_dir, "panel_*_annotated_gaze_data.csv"))):
                fname = os.path.basename(csv_path)
                panel = fname.replace("panel_", "").replace("_annotated_gaze_data.csv", "")

                pct_upper, pct_fixation, x_span_pct, errors = check_file(csv_path)

                rows.append({
                    "group": group,
                    "participant": participant,
                    "panel": panel,
                    "pct_upper": round(pct_upper, 2),
                    "pct_fixation": round(pct_fixation, 2),
                    "x_span_pct": round(x_span_pct, 2),
                    "errors": errors,
                })

    results = pd.DataFrame(rows)

    # narrow horizontal range: flag panels using less than SPACE_USED_PCT_MIN% of the
    # cohort's own median horizontal range for that panel (raw span varies a lot between
    # panels, so a fixed absolute cutoff doesn't work - the cohort median is the reference)
    reference_span = results.groupby("panel")["x_span_pct"].transform("median")
    narrow_range = results["x_span_pct"] < (SPACE_USED_PCT_MIN / 100) * reference_span
    results.loc[narrow_range, "errors"] = results.loc[narrow_range, "errors"].apply(lambda e: e + [6])

    results["error_labels"] = results["errors"].apply(lambda e: ",".join(ERROR_LABELS[c] for c in e))
    results["errors"] = results["errors"].apply(lambda e: ",".join(str(c) for c in e))

    return results


def test_gaze_quality_checks():
    """Run the sweep and assert every file was readable and produced a result."""
    results = run_checks()
    assert not results.empty, "No annotated gaze data CSVs were found."
    assert results["pct_upper"].between(0, 100).all()
    assert results["pct_fixation"].between(0, 100).all()


if __name__ == "__main__":
    results = run_checks()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    print(results.to_string(index=False))

    flagged = results[results["errors"] != ""]
    print(f"\n{len(flagged)} / {len(results)} files flagged with an error.\n")
    if not flagged.empty:
        print(flagged.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, f"gaze_quality_check_results_{output_date}.csv")
    results.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")

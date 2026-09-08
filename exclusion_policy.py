"""
Participant/event exclusion policy (final decisions, 2026-08-26; eye-selection logic
revised 2026-09-07).

Kept separate from participant_gaze_data_manager.py on purpose (decision 5 of that
list): loading + cleaning raw samples is a data concern and lives in the manager; deciding
whether a participant/event should be USED AT ALL is a policy concern that can change
independently, so it lives here instead. One exception: as of 2026-09-07, the accuracy<2deg
criterion is no longer purely a downstream policy filter - it's now evaluated jointly with
NaN ratio, per panel, INSIDE prepare_gaze_data_for_preprocessing (a panel whose data would
fail the criterion is never loaded into matched_data at all, and the eye used for a
loaded panel is no longer always the single best-calibrated eye for the whole event - see
that function's docstring). filter_by_accuracy_threshold/best_calibrated_accuracy below are
kept only as an external double-check (e.g. auditing that the loader behaved correctly) -
they are effectively a no-op on data produced by the current loader, since every panel that
reaches build_calibration_table() already has at least one eye under threshold by
construction. best_calibrated_accuracy's min(left,right) is also not exactly "the used eye's
accuracy" in the rare rescue case (both eyes qualify on accuracy, but the better-calibrated
one's NaN ratio failed for that panel, so the WORSE-calibrated-but-still-qualifying eye was
used) - it still correctly reports "passes the <2 deg threshold" either way, just not
necessarily the accuracy of the specific eye whose data was actually used.

Participant-level criterion (still a pure downstream filter, unaffected by the above):
  Tobii_Sucks == "YES" in the behavioral summary table - a manual "this participant's
  eye-tracking data is unusable" flag from outside this pipeline. Excludes the participant
  outright, in every analysis.
"""
import pandas as pd

DEFAULT_TOBII_SUCKS_XLSX = "/Volumes/ramot/Noam_M/df_filtered_behavioral_summary_20260427_131437.xlsx"
ACCURACY_EXCLUSION_THRESHOLD_DEG = 2.0


def load_tobii_sucks_excluded_participants(xlsx_path=DEFAULT_TOBII_SUCKS_XLSX):
    """Set of Patient_IDs flagged Tobii_Sucks == 'YES' in the behavioral summary table."""
    df = pd.read_excel(xlsx_path)
    return set(df.loc[df["Tobii_Sucks"].astype(str).str.upper() == "YES", "Patient_ID"])


def passes_accuracy_threshold(used_acc, threshold_deg=ACCURACY_EXCLUSION_THRESHOLD_DEG):
    """True iff the used eye's calibration accuracy is under threshold_deg. NaN (no
    calibration message / no eye ever assigned) never passes."""
    return pd.notna(used_acc) and used_acc < threshold_deg


def best_calibrated_accuracy(calibration_df):
    """calibration_df: build_calibration_table()'s output (one row per participant+panel,
    left_acc/right_acc columns). Returns a Series of the used eye's accuracy - min(left,
    right), since the used eye is always whichever is best-calibrated (decision 1)."""
    return calibration_df[["left_acc", "right_acc"]].min(axis=1)


def filter_by_accuracy_threshold(data_df, calibration_df, threshold_deg=ACCURACY_EXCLUSION_THRESHOLD_DEG):
    """
    Filters data_df (any per participant+panel table - e.g. a whole_dictionary_drift_qa
    table) down to rows whose calibration accuracy (from calibration_df, build_calibration_table()'s
    output) passes passes_accuracy_threshold. Rows with no matching calibration_df entry, or
    with calibration_missing=True there, are dropped (can't verify the criterion, so they
    don't pass it).
    """
    calib = calibration_df[calibration_df["calibration_missing"] == False].copy()
    calib["used_acc"] = best_calibrated_accuracy(calib)
    merged = data_df.merge(calib[["participant", "panel", "used_acc"]], on=["participant", "panel"], how="left")
    passing = merged[merged["used_acc"].apply(passes_accuracy_threshold, threshold_deg=threshold_deg)]
    return passing.drop(columns=["used_acc"]).reset_index(drop=True)

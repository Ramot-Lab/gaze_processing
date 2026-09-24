"""
Targeted, one-off recovery for participants identified by the 2026-09-17 quality-check
audit (/Volumes/ramot/Noam_M/after_analysis_quality_checks_17_09_26/silent_panels_categorized.csv)
as missing panels in Stage 1's saved output with NO reason recorded anywhere in Stage 1's
own exclusion_log.csv.

The audit found three distinct causes, all upstream of any per-panel accuracy/NaN-ratio
policy:
  - BO921, BY165, EE050, PR454: a SECOND, valid, loadable calibration-event mat file
    exists in the participant's SDMT folder, but ParticipantGazeDataManager.load_data's
    "run" in filename.lower() filter (participant_gaze_data_manager.py:241) silently
    excludes it because that file's name doesn't contain "run". This script does NOT
    broaden that filter generally (other participants' non-"run"-named files are mostly
    lower-quality duplicates, not genuine second sessions - see
    _cleaned_SDM_gaze_data.mat/PROBABLY_ANOTHER_VERSION_OF_GAZE_DATA_ORIGINAL.mat in the
    audit's mat_gaze_content_check.csv) - it manually loads exactly the one specific file
    already verified per participant, reusing the same prepare_gaze_data_for_preprocessing
    / group_task_info methods the normal loader uses (so the same per-panel accuracy/
    NaN-ratio eye-selection policy still applies), then merges the recovered panels into
    the normally-loaded ParticipantGazeDataManager instance.
  - KT360, ZM425: the file that would cover the missing panels is present but corrupted
    (fails to load) or empty (loads but has zero gaze samples) - not recoverable.
  - The remaining 24 participants: only one of the two session mat files was ever
    recorded - a genuine data gap, not a bug, not recoverable.

For the 4 recoverable participants, runs annotate_gaze_events for BOTH annotation methods
on whichever panels the recovered file actually restores, and saves those CSVs into a new
"recovered" subfolder (mirroring the normal <group>/<participant>/panel_<panel>_annotated_gaze_data.csv
layout) under each method's existing SDMT_annotated_gaze_16_09_26_<method> folder - kept
separate from the main run's output rather than merged in, so this targeted recovery pass
stays clearly distinguishable from the original full run.

For every one of the 30 audited participants, appends a short, plainly-phrased reason row
to each method's EXISTING exclusion_log.csv for any panel that's still missing after this
recovery attempt, so nothing from this audit stays silently unexplained. Guarded to run
only once per method (checks for its own scope tag before appending).
"""
import os
from glob import glob

import pandas as pd
import scipy.io as scio

from participant_gaze_data_manager import ParticipantGazeDataManager, EventExcludedError
from pipeline_config import main_data_path, annotated_gaze_dir, ANNOTATION_METHODS

STAGE1_DATE = "16_09_26"
ALL_PANELS = ["0", "i1", "l4", "a3", "a5", "l3"]
POST_HOC_SCOPE = "post_hoc_quality_check_17_09_26"

# From the 2026-09-17 audit (after_analysis_quality_checks_17_09_26/silent_panels_categorized.csv):
# participant -> the specific filtered-out mat filename (relative to <participant>/SDMT/)
# verified to contain real, loadable gaze data for panels this participant is missing.
KNOWN_RECOVERABLE_FILES = {
    "BO921": "BO921_SDMT_DAT.mat",
    "BY165": "BY165_SDMT_DAT.mat",
    "EE050": "SDMT_DAT.mat",
    "PR454": "SDMT_DAT.mat",
}

# Category A: only one of the two session mat files was ever recorded for this participant.
GENUINE_GAP_PARTICIPANTS = [
    "AE504", "BD783", "BR119", "EA656", "ED186", "EO948", "HA186", "KH607", "KM269",
    "NM680", "NS960", "PY847", "SA848", "SD938", "SH625", "SO754", "SR081", "ST888",
    "TS243", "TS512", "VA784", "YN187", "YR187", "YR573",
]
GENUINE_GAP_REASON = "only one recording session found for this participant - the other session was never recorded"

# Category C: the session file is present but corrupted or empty.
CORRUPTED_FILE_PARTICIPANTS = {
    "KT360": "recording session file found but contains zero gaze samples (corrupted/empty)",
    "ZM425": "recording session file found but failed to load (corrupted .mat file)",
}


def find_group(participant, data_path):
    for g in ("HC", "pwMS"):
        if os.path.isdir(os.path.join(data_path, g, participant)):
            return g
    return None


def recover_known_filtered_panels(sd, subject_dir, data_path, participant):
    """Merge in the one known filtered-out mat file's panels for `participant`, if any is
    registered. Returns a list of {"panel", "reason"} dicts for anything even the
    recovered file could not save (e.g. a panel that still fails eye-selection within
    that file)."""
    fname = KNOWN_RECOVERABLE_FILES.get(participant)
    if not fname:
        return []
    path = os.path.join(subject_dir, "SDMT", fname)
    if not os.path.exists(path):
        return []

    extra_rows = []
    try:
        mat = scio.loadmat(path, struct_as_record=False, squeeze_me=True)
        task_png = glob(os.path.join(data_path, "panels_images", "SDMT", "*.jpg"))
        audio_recordings = glob(os.path.join(subject_dir, "SDMT", "**", "*.wav"), recursive=True)
        (sd.task_data, sd.messages, gaze_data, sd.presentation_info, sd.dom_Eye,
         sd.panel_eye_used, sd.panel_eye_reason, panel_failure_reasons) = \
            sd.prepare_gaze_data_for_preprocessing(mat, None)
        new_matched = sd.group_task_info(gaze_data, task_png, audio_recordings, sd.task_data, mat, True)
        sd.matched_data.update(new_matched)
        for reason in panel_failure_reasons:
            extra_rows.append({"panel": reason.get("panel"),
                                "reason": f"recovered file '{fname}' loaded, but {reason.get('error')}"})
    except EventExcludedError as e:
        for pd_ in (e.panel_details or []):
            extra_rows.append({"panel": pd_.get("panel"),
                                "reason": f"recovered file '{fname}' loaded, but {pd_.get('error')}"})
    except Exception as e:
        extra_rows.append({"panel": None,
                            "reason": f"recovery attempt on '{fname}' failed: {type(e).__name__}: {e}"})
    return extra_rows


def run():
    data_path = main_data_path()
    all_participants = sorted(set(KNOWN_RECOVERABLE_FILES) | set(GENUINE_GAP_PARTICIPANTS)
                               | set(CORRUPTED_FILE_PARTICIPANTS))
    print(f"Processing {len(all_participants)} previously-silent participants: {all_participants}")

    for method in ANNOTATION_METHODS:
        stage1_dir = annotated_gaze_dir(method, STAGE1_DATE)
        excl_path = os.path.join(stage1_dir, "exclusion_log.csv")
        existing_excl = pd.read_csv(excl_path)

        if (existing_excl.get("scope") == POST_HOC_SCOPE).any():
            print(f"{method}: {POST_HOC_SCOPE} rows already present in {excl_path} - skipping "
                  f"(already applied once, not re-appending).")
            continue

        existing_logged_pairs = set(zip(existing_excl["participant"], existing_excl["panel"]))
        new_rows = []
        recovered_saved = 0

        for participant in all_participants:
            group = find_group(participant, data_path)
            if group is None:
                print(f"  {participant}: could not find a raw data folder, skipping")
                continue
            subject_dir = os.path.join(data_path, group, participant)

            method_pdir = os.path.join(stage1_dir, group, participant)
            saved_before = set()
            if os.path.isdir(method_pdir):
                for f in os.listdir(method_pdir):
                    if f.startswith("panel_") and f.endswith("_annotated_gaze_data.csv"):
                        saved_before.add(f[len("panel_"):-len("_annotated_gaze_data.csv")])
            missing_before = set(ALL_PANELS) - saved_before
            if not missing_before:
                continue

            sd = ParticipantGazeDataManager(subject_dir, data_path, "SDMT", group)
            recovery_fail_rows = recover_known_filtered_panels(sd, subject_dir, data_path, participant)

            recovered_out_dir = os.path.join(stage1_dir, "recovered", group, participant)
            for panel in sorted(missing_before):
                if panel in sd.matched_data:
                    try:
                        annotated = sd.annotate_gaze_events(panel, method)
                    except Exception as e:
                        new_rows.append({
                            "participant": participant, "group": group, "panel": panel,
                            "reason": f"recovered file loaded, but annotate_gaze_events({method}) raised: "
                                      f"{type(e).__name__}: {e}",
                            "l_acc": None, "r_acc": None, "l_nan_pct": None, "r_nan_pct": None,
                            "other_errors": None, "scope": POST_HOC_SCOPE,
                        })
                        continue
                    os.makedirs(recovered_out_dir, exist_ok=True)
                    fname_out = f"panel_{panel}_annotated_gaze_data.csv"
                    annotated.to_csv(os.path.join(recovered_out_dir, fname_out), index=False)
                    recovered_saved += 1
                    print(f"  RECOVERED {participant}/{panel} ({method}) -> "
                          f"{os.path.join(recovered_out_dir, fname_out)}")
                    continue

                if (participant, panel) in existing_logged_pairs:
                    continue  # already has a real reason logged from the normal Stage 1 run

                fail_row = next((r for r in recovery_fail_rows if r["panel"] == panel), None)
                if fail_row:
                    reason = fail_row["reason"]
                elif participant in CORRUPTED_FILE_PARTICIPANTS:
                    reason = CORRUPTED_FILE_PARTICIPANTS[participant]
                else:
                    reason = GENUINE_GAP_REASON

                new_rows.append({
                    "participant": participant, "group": group, "panel": panel, "reason": reason,
                    "l_acc": None, "r_acc": None, "l_nan_pct": None, "r_nan_pct": None,
                    "other_errors": None, "scope": POST_HOC_SCOPE,
                })

        if new_rows:
            pd.DataFrame(new_rows).to_csv(excl_path, mode="a", header=False, index=False)
            print(f"{method}: appended {len(new_rows)} new exclusion rows to {excl_path}")

        recovered_readme = os.path.join(stage1_dir, "recovered", "README.txt")
        if recovered_saved and not os.path.exists(recovered_readme):
            os.makedirs(os.path.dirname(recovered_readme), exist_ok=True)
            with open(recovered_readme, "w") as f:
                f.write(
                    "Generated by recover_missing_panels.py, 2026-09-17.\n\n"
                    "Panel CSVs recovered from a second calibration-event mat file that the main "
                    "Stage 1 run's filename filter ('run' in filename.lower()) silently excluded, "
                    "for participants identified in "
                    "/Volumes/ramot/Noam_M/after_analysis_quality_checks_17_09_26/silent_panels_categorized.csv. "
                    "Same layout and columns as the main run's output "
                    "(<group>/<participant>/panel_<panel>_annotated_gaze_data.csv), kept separate "
                    "from it rather than merged in.\n"
                )
        print(f"{method}: saved {recovered_saved} recovered panel CSVs\n")


if __name__ == "__main__":
    run()

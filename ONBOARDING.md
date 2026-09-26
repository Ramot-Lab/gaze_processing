# Handoff: running the full SDMT gaze pipeline on a new machine

You're picking this up on a freshly-cloned copy of this repo. This file is the "what do I
actually need to do" note for that. Read it before touching anything.

## What this project is

A 3-stage gaze-tracking analysis pipeline (Ramot lab, MS/SDMT study):
1. **Preprocessing** (`run_preprocessing.py`) - annotates raw eye-tracking data per
   participant/panel with fixation/saccade events, applies exclusion criteria, saves one
   CSV per panel.
2. **Markov chain analysis** (`main.py`) - builds symbol/ROI transition matrices from
   Stage 1's saved CSVs, runs PCA + consistency/permutation tests.
3. **Feature analysis** (`feature_pipeline.py`) - computes trial/panel/participant-level
   features from Stage 1's saved CSVs, runs reliability + correlation + PCA analyses.

All paths, folder-naming, and pipeline-wide options live in one place: `pipeline_config.py`.
Don't hardcode paths/thresholds elsewhere - read from there.

## Standing decision: which method + correction to actually run

**Use `model_based` annotation, not `threshold_based`.** And analyses run on the
whole-dictionary-drift-**corrected** gaze data, not raw - this is already baked into
`markov_loader.py` and `feature_pipeline.py` (they always request
`pipeline_config.load_annotated_csv(..., corrected=True)`), you don't pass a flag for it.

Exclusion criteria are as currently defined in the code (do not change without asking) -
see `pipeline_config.py` and `exclusion_policy.py` for the full list:
- Dominant eye is used by default (not accuracy-based eye selection) - but the dominant
  eye's own calibration accuracy still gates exclusion (< 2.5deg required).
- NaN-ratio per panel (< 10%).
- More than 80% saccades in a panel's annotated stream -> excluded
  (`pipeline_config.MAX_SACCADE_FRACTION`).
- Gaze that never registers inside the dictionary/legend area, or never reaches the
  search-text grid -> excluded (`pipeline_config.DICTIONARY_BOUNDARY_RATIO` /
  `TEXT_BOUNDARY_RATIO`).
- 5 manually-excluded participants with specific reasons
  (`exclusion_policy.MANUALLY_EXCLUDED_PARTICIPANTS`).
- Tobii_Sucks-flagged participants are currently **included** by default
  (`pipeline_config.EXCLUDE_TOBII_SUCKS = False`) - don't flip this without asking.

## Exact steps to run the full pipeline

Run these in order, from the repo root:

```bash
# 1. Preprocessing - MUST be run fresh on this machine (there is no existing Stage 1
#    output here yet). Resumable - safe to re-run if interrupted, it skips participants
#    already in processed_participants.csv.
python run_preprocessing.py model_based

# 2. Precompute the whole-dictionary-drift correction for everything Stage 1 just saved.
#    Not strictly required (Stage 2/3 will compute it live on first read otherwise), but
#    saves paying that cost repeatedly - do this once.
python build_corrected_annotated_csvs.py model_based

# 3. Markov chain analysis (Stage 2). Already wired to read the corrected data.
python main.py model_based

# 4. Feature analysis (Stage 3). Already wired to read the corrected data.
python feature_pipeline.py model_based
```

## Things that will bite you if you don't know about them

- **Network mount naming**: the network share can remount under a different name mid-session
  (`/Volumes/ramot`, `/Volumes/ramot-1`, `/Volumes/ramot-2`, ...). `pipeline_config.find_ramot_mount()`
  already probes for whichever name is currently live - every path-building function
  re-resolves it on each call rather than caching, so this should just work. If you get a
  `FileNotFoundError` about no `/Volumes/ramot*` mount, the share probably isn't mounted at
  all yet on this machine - mount it first.
- **`run_preprocessing.py` is slow, especially for `model_based`** (it runs live gazeNET
  model inference per panel). It resumes correctly if interrupted - check
  `processed_participants.csv` in the output folder to see progress. If the connection to
  the network share is unstable/slow, it's fine to stop and restart later rather than
  push through a bad connection.
- **Date rollover**: Stage 1's output folder is named `SDMT_annotated_gaze_<date>_model`
  (today's date at the time it starts). If a run spans midnight, don't manually pass a
  `date_str` on resume - `pipeline_config.latest_annotated_gaze_date()` auto-detects the
  most recent existing folder for the method, so a plain re-run without arguments picks up
  the right one.
- **GPU/model dependency**: `model_based` annotation requires the gazeNET model
  (`GazeModel/gazeNET_0004_00003750.pth.tar`, already in the repo) and will use CUDA if
  available, else CPU (slower). Make sure `torch` is installed in whatever environment
  you're running this in.
- **Output goes to the network share, not the repo.** Everything Stage 1/2/3 produces
  lands under `<mount>/Noam_M/...` (see `pipeline_config.py`'s path functions) - nothing
  gets written into the git repo itself.

## If something looks wrong

This session did extensive debugging of the annotation/ROI/eye-selection pipeline before
arriving at the current state (dominant eye + accuracy gate + whole-dictionary correction).
If you hit something that looks like a data-quality issue (wrong-looking sequences, a
participant that seems mis-tracked, etc.), don't assume it's a new bug before checking
whether it's already a known, characterized issue - a lot of ground was already covered
here.

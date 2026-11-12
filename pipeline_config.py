"""
Single source of truth for paths, folder-naming, and pipeline-wide options shared across
the preprocessing / Markov-chain / feature-analysis stages.

Per-sample constants (FIXATION_IDX, SCREEN_SIZE, OUT_OF_RANGE_*_BOUNDS, etc.) stay in
constants.py; calibration accuracy/NaN-ratio thresholds stay in exclusion_policy.py /
constants.py - this module re-uses those, it does not duplicate them.

The network share has repeatedly remounted under a different name mid-session
(/Volumes/ramot, /Volumes/ramot-1, /Volumes/ramot-2, ...) - every path-building function
here re-resolves the live mount on each call rather than caching it at import time, so
importing this module never touches the filesystem and never fails just because the
mount happened to be down at import time.
"""
import os

import numpy as np
from datetime import datetime

ANNOTATION_METHODS = ("threshold_based", "model_based")

# Folder-name suffix per method, per the user's example ("SDMT_annotated_gaze_16_09_26_threshold")
_METHOD_FOLDER_SUFFIX = {"threshold_based": "threshold", "model_based": "model"}

# New exclusion/processing options - both default OFF/raw per the approved plan; do not
# flip these on without an explicit decision, they are intentionally unused for now.
EXCLUDE_TOBII_SUCKS = False
CALIBRATION_CORRECTION = "raw"  # "raw" | "corrected" - "corrected" method not yet decided

# Decision 2026-09-22: exclude a panel outright if more than this fraction of its
# annotated samples are saccades - a sign of tracking noise/misclassification swamping
# the panel rather than a real eye-movement pattern.
MAX_SACCADE_FRACTION = 0.8

# Normalized-y boundaries splitting the panel into dictionary/legend (y < DICTIONARY_
# BOUNDARY_RATIO), a small buffer, and the search-text grid (y >= TEXT_BOUNDARY_RATIO) -
# the single source of truth for SearchFinder.find()'s zone thresholds and for the
# preprocessing exclusion checks below (a participant whose gaze never crosses one of
# these boundaries never genuinely visited that zone).
#
# DICTIONARY_BOUNDARY_RATIO (decision 2026-09-22): the dictionary digit-row ROI boxes'
# own (already-elongated) bottom edge, measured on AA562/panel a3 via RoiFinder - see
# after_analysis_quality_checks_17_09_26/dictionary_boundary_comparison.png. Deliberately
# kept distinct from calibration_drift_qa.py's own, separately-measured
# DICTIONARY_BOUNDARY_RATIO (870/3861, a direct pixel measurement of the dict/text divider
# line, with no gap to TEXT_BOUNDARY_RATIO) - that file is intentionally left as-is.
DICTIONARY_BOUNDARY_RATIO = 0.1843 #bottom of ROI boxes for dictionary digit row
TEXT_BOUNDARY_RATIO = 0.22

# Behavioral/demographic summary table (decision 2026-11-12) - the single source for
# demographics, cognitive tests and the Tobii_Sucks flag (exclusion_policy,
# feature_pipeline.merge_ido_table). Copied into each run's features/non_SDMT/demographics/.
BEHAVIORAL_SUMMARY_XLSX = "/Volumes/ramot/Noam_M/df_clean_behavioral_summary_20261110_183415.xlsx"


def find_ramot_mount():
    """Whichever /Volumes/ramot* actually has the expected Noam_M path mounted right now."""
    candidates = ["/Volumes/ramot"] + [f"/Volumes/ramot-{i}" for i in range(1, 10)]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "Noam_M", "Results", "Behavior", "processing_results")):
            return c
    raise FileNotFoundError("no /Volumes/ramot* mount with the expected Noam_M path found")


def noam_m_root():
    return os.path.join(find_ramot_mount(), "Noam_M")


def main_data_path():
    return os.path.join(noam_m_root(), "Results", "Behavior")


def _today(date_str):
    return date_str or datetime.now().strftime("%d_%m_%y")


def latest_annotated_gaze_date(method):
    """Most recent existing SDMT_annotated_gaze_<date>_<method> folder's date, or None if
    none exist yet. Runs (especially model_based, which takes hours) routinely span a
    midnight rollover, so "today" is the wrong default when reading back a previous run's
    output - this picks whatever's actually on disk instead."""
    import glob
    import re
    suffix = _METHOD_FOLDER_SUFFIX[method]
    # Both layouts: <Noam_M>/<date>/SDMT_annotated_gaze_<date>_<suffix> (decision
    # 2026-11-12) and the older flat <Noam_M>/SDMT_annotated_gaze_<date>_<suffix>.
    paths = (glob.glob(os.path.join(noam_m_root(), "*", f"SDMT_annotated_gaze_*_{suffix}"))
             + glob.glob(os.path.join(noam_m_root(), f"SDMT_annotated_gaze_*_{suffix}")))
    candidates = []
    for path in paths:
        m = re.match(rf"^SDMT_annotated_gaze_(\d{{2}}_\d{{2}}_\d{{2}})_{suffix}$", os.path.basename(path))
        if m:
            candidates.append(m.group(1))
    if not candidates:
        return None
    return max(candidates, key=lambda d: datetime.strptime(d, "%d_%m_%y"))


def date_root(date_str):
    """One folder per Stage 1 run date (decision 2026-11-12): <Noam_M>/<DD_MM_YY>/ holds
    that run's SDMT_annotated_gaze_<date>_<method>/, analysis_results/, behavior_scores/,
    live_board/ and analysis_participant_exclusion_summary.csv."""
    return os.path.join(noam_m_root(), date_str)


def analysis_results_dir(date_str):
    return os.path.join(date_root(date_str), "analysis_results")


def annotated_gaze_dir(method, date_str=None):
    """Stage 1 output root for one method: SDMT_annotated_gaze_<date>_<method>, <date> being
    the Stage 1 run's date. date_str=None resolves to the most recent existing folder for
    this method if one exists (correct for reading back a previous run), else today
    (correct for starting a genuinely new preprocessing run).
    Where it lives: wherever it already exists - inside any <Noam_M>/<DD_MM_YY>/ date
    folder (the date folder may be named after the analysis day, not the Stage 1 day -
    e.g. 12_11_26/SDMT_annotated_gaze_11_11_26_model), or flat in Noam_M (saved before
    2026-11-12); a new one is created in <Noam_M>/<its own date>/."""
    import glob
    resolved = date_str or latest_annotated_gaze_date(method) or datetime.now().strftime("%d_%m_%y")
    name = f"SDMT_annotated_gaze_{resolved}_{_METHOD_FOLDER_SUFFIX[method]}"
    existing = sorted(glob.glob(os.path.join(noam_m_root(), "*", name)))
    if existing:
        return existing[0]
    flat = os.path.join(noam_m_root(), name)
    return flat if os.path.isdir(flat) else os.path.join(date_root(resolved), name)


def stage1_date_root(method, date_str=None):
    """The <Noam_M>/<DD_MM_YY>/ folder holding this Stage 1 run - where its analysis is
    written (decision 2026-11-12: results sit next to the Stage 1 data they came from).
    A flat pre-2026-11-12 Stage 1 folder falls back to <Noam_M>/<its date>/."""
    gaze_dir = annotated_gaze_dir(method, date_str)
    parent = os.path.dirname(gaze_dir)
    if os.path.normpath(parent) != os.path.normpath(noam_m_root()):
        return parent
    return date_root(date_str or latest_annotated_gaze_date(method))


def preliminary_results_dir(method):
    """Stage 2/3 output root for one method: .../Noam_M/preliminary_results_<method>"""
    return os.path.join(noam_m_root(), f"preliminary_results_{method}")


def markov_output_dir(method, date_str=None):
    return os.path.join(preliminary_results_dir(method), "markov_analysis", _today(date_str))


def feature_analysis_base_dir(method):
    """Base dir (no date subfolder) - what resolve_run_dir-style callers need, since they
    append/resolve the date subfolder themselves."""
    return os.path.join(preliminary_results_dir(method), "feature_analysis")


def feature_output_dir(method, date_str=None):
    return os.path.join(feature_analysis_base_dir(method), _today(date_str))


def fxsp_annotation_cache_dir():
    """Shared, run-independent cache of gazeNET per-sample labels for the fxsp task
    (saccade_latency_analysis.py) - the fxsp counterpart of Stage 1's
    SDMT_annotated_gaze_<date>_model folders. Model inference is the only slow step
    (~40 s/participant); keeping it out of the per-run folders lets every feature run
    recompute latencies in minutes. Files are keyed by participant + source .mat name,
    so a new recording gets its own entry instead of reusing a stale one."""
    return os.path.join(noam_m_root(), "fxsp_annotated_gaze_model")


def consolidated_results_dir(date_str=None):
    """One shared output root for a joint run of every analysis stage on the same day
    (decision 2026-09-27) - so a run's Markov + feature-analysis + symbol-search +
    duration-analysis output all land side by side under one dated folder instead of
    scattered across each stage's own default location. Opt-in: pass this as
    AnalysisConfig.output_base_path / feature_pipeline.main's run_dir / the other
    scripts' run_dir - nothing defaults to it on its own."""
    resolved = date_str or datetime.now().strftime("%d_%m_%y")
    return os.path.join(noam_m_root(), f"markov_features_results_{resolved}")


def _dd_mm_yy_suffix_date(path):
    """Date of a folder named <anything>_DD_MM_YY; None if it has no such suffix."""
    import re
    m = re.search(r"(\d{2})_(\d{2})_(\d{2})$", os.path.basename(path.rstrip("/")))
    if not m:
        return None
    try:
        return datetime.strptime("_".join(m.groups()), "%d_%m_%y")
    except ValueError:
        return None


def latest_dated_dir(pattern, required_relpath):
    """Newest folder under Noam_M matching `pattern` (glob, e.g. "markov_features_results_*")
    whose DD_MM_YY suffix is the latest AND that contains required_relpath. TEST folders
    are ignored. Used wherever a step reads an existing table instead of computing it, so
    it always gets the most current one rather than a hard-coded date (decision
    2026-11-11). Raises FileNotFoundError if none qualifies."""
    import glob
    candidates = [d for d in glob.glob(os.path.join(noam_m_root(), pattern))
                  if "TEST" not in os.path.basename(d) and _dd_mm_yy_suffix_date(d)
                  and os.path.exists(os.path.join(d, required_relpath))]
    if not candidates:
        raise FileNotFoundError(f"No {pattern} folder under {noam_m_root()} contains {required_relpath}")
    return max(candidates, key=_dd_mm_yy_suffix_date)


def latest_saccade_latency_csv():
    """The most current standalone saccade-latency table
    (.../Noam_M/saccade_latency_<tag>_DD_MM_YY/compiled_latencies.csv, decision
    2026-11-11). Saccade latency is the ONE feature the pipeline does not recompute from
    raw on each run - feature_pipeline.main copies this table into its run folder."""
    return os.path.join(latest_dated_dir("saccade_latency_*", "compiled_latencies.csv"), "compiled_latencies.csv")


FEATURE_TABLES_RELPATH = os.path.join("analysis_results", "features", "csvs")


def latest_date_root():
    """Newest <Noam_M>/<DD_MM_YY> folder whose analysis_results has the feature tables."""
    return latest_dated_dir("[0-9][0-9]_[0-9][0-9]_[0-9][0-9]",
                            os.path.join(FEATURE_TABLES_RELPATH, "participant_features.csv"))


def latest_feature_tables_dir():
    """Folder holding trial/panel/participant_features.csv of the newest run - the default
    input for standalone scripts that read the pipeline's tables."""
    return os.path.join(latest_date_root(), FEATURE_TABLES_RELPATH)


def annotated_csv_path(method, group, participant, panel, date_str=None, corrected=False):
    """corrected=True points at the whole-dictionary-drift-corrected CSV saved NEXT TO the
    raw one (decision 2026-09-22) - same directory, same base name, "_corrected" suffix -
    rather than a separate folder, since it's the same participant/panel/method, just with
    the dictionary-region y-drift correction applied (see
    calibration_drift_qa.correct_annotated_csv_whole_dictionary)."""
    suffix = "_corrected" if corrected else ""
    return os.path.join(annotated_gaze_dir(method, date_str), group, participant,
                         f"panel_{panel}_annotated_gaze_data{suffix}.csv")


def load_annotated_csv(method, group, participant, panel, date_str=None, corrected=False):
    """Read one of Stage 1's saved per-panel CSVs (t,eye_horizontal,eye_vertical,status,evt).
    Raises FileNotFoundError if Stage 1 hasn't produced this participant/panel yet (either
    it was excluded, or preprocessing hasn't been run for this method/date).

    corrected=True (decision 2026-09-22): prefers the saved "_corrected" CSV (see
    annotated_csv_path); if that specific file doesn't exist yet (e.g.
    build_corrected_annotated_csvs.py hasn't been run for this method/date), falls back to
    computing the correction live from the raw CSV via
    calibration_drift_qa.correct_annotated_csv_whole_dictionary - callers always get
    corrected data back, they just don't pay the correction's (cheap, but non-zero) cost
    again once it's been saved once. Only applies to analyses whose ROI logic is y-value-
    dependent (Markov chain analysis; feature analysis, since its TrialManager construction
    goes through the same y-dependent Search/Sequence building) - raw (uncorrected) stays
    the right choice elsewhere (e.g. this file's own SearchFinder-boundary exclusion checks,
    which are about where gaze genuinely was, not where it "should" have been).

    Checks exclusion_policy.MANUALLY_EXCLUDED_PANELS first (decision 2026-09-27) and
    raises FileNotFoundError for those too, same as a genuine missing file - this is the
    single chokepoint every consumer already reads through, so one entry here excludes
    the panel everywhere (Markov, feature analysis, and any one-off script) without
    needing Stage 1 to be re-run."""
    import pandas as pd
    from exclusion_policy import load_manually_excluded_panels
    manual_panels = load_manually_excluded_panels()
    if (participant, panel) in manual_panels:
        raise FileNotFoundError(f"manually_excluded_panel: {manual_panels[(participant, panel)]}")

    if corrected:
        corrected_path = annotated_csv_path(method, group, participant, panel, date_str, corrected=True)
        if os.path.exists(corrected_path):
            return pd.read_csv(corrected_path)
        from calibration_drift_qa import correct_annotated_csv_whole_dictionary
        raw = pd.read_csv(annotated_csv_path(method, group, participant, panel, date_str, corrected=False))
        fixed, _, _ = correct_annotated_csv_whole_dictionary(raw)
        return fixed

    path = annotated_csv_path(method, group, participant, panel, date_str, corrected=False)
    return pd.read_csv(path)


def interpolated_samples_path(method, group, participant, panel, date_str=None):
    """Stage 1's per-panel record of which samples were filled in by interpolation
    (decision 2026-11-10) - a separate file next to the annotated CSV, NOT a new column in
    it, because FixationHandler unpacks the annotated CSV's 5 columns positionally and the
    drift-corrected CSV keeps a fixed column list. Columns: t (same values as the
    annotated CSV's t), interpolated (0/1)."""
    return os.path.join(annotated_gaze_dir(method, date_str), group, participant,
                         f"panel_{panel}_interpolated_samples.csv")


def load_interpolated_t(method, group, participant, panel, date_str=None):
    """Set of t values (rounded to int, the annotated CSV's us timestamps) whose sample
    was filled in by interpolation, or None if Stage 1 didn't save the file (Stage 1
    folders from before 2026-11-10) - callers then can't apply the BCEA interpolation
    rules and should say so."""
    import pandas as pd
    path = interpolated_samples_path(method, group, participant, panel, date_str)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return set(np.round(df.loc[df["interpolated"] == 1, "t"].to_numpy()).astype(np.int64).tolist())


_stage1_exclusion_cache = {}


def stage1_exclusion_reason(method, participant, panel, date_str=None):
    """Look up why Stage 1 excluded (participant, panel), if it did - reads
    Stage 1's own exclusion_log.csv once per (method, date) and caches it, so Stage 2/3
    callers hitting a missing annotated CSV can report the REAL upstream reason (e.g.
    "no eye both has accuracy < 2.5 deg...") instead of re-discovering it as a fresh,
    confusingly-labeled failure (a bare KeyError, or a live-annotation fallback that hits
    the exact same missing data). Returns None if Stage 1 has no row for this exact
    (participant, panel) - either it genuinely wasn't excluded (unexpected - the caller
    should treat that as a real, new problem) or Stage 1 hasn't been run for this
    method/date at all.

    Checked BEFORE the exclusion_log.csv cache (decision 2026-09-27): a manually
    excluded panel (exclusion_policy.MANUALLY_EXCLUDED_PANELS) was never actually
    dropped by the Stage 1 run that produced this date's exclusion_log.csv - Stage 1
    saved it just fine, the exclusion happens later at load_annotated_csv - so it would
    otherwise be invisible here and reported as an unexplained new problem."""
    from exclusion_policy import load_manually_excluded_panels
    manual_panels = load_manually_excluded_panels()
    if (participant, panel) in manual_panels:
        return f"manually_excluded_panel: {manual_panels[(participant, panel)]}"

    key = (method, date_str)
    if key not in _stage1_exclusion_cache:
        import pandas as pd
        csv_path = os.path.join(annotated_gaze_dir(method, date_str), "exclusion_log.csv")
        try:
            df = pd.read_csv(csv_path)
            cache = {}
            for _, r in df.iterrows():
                # A whole-participant exclusion (e.g. run_preprocessing.py's manual-
                # exclusion check) logs ONE row with panel=NaN, not one row per panel -
                # normalize to None so it's found as a fallback below (decision
                # 2026-09-27, found via OY974 falling through to live re-annotation on
                # a full-population run: every per-panel lookup missed that NaN-keyed
                # row and reported "no reason found").
                panel_key = None if pd.isna(r["panel"]) else r["panel"]
                cache[(r["participant"], panel_key)] = r["reason"]
            _stage1_exclusion_cache[key] = cache
        except Exception:
            _stage1_exclusion_cache[key] = {}

    cache = _stage1_exclusion_cache[key]
    return cache.get((participant, panel), cache.get((participant, None)))


EXCLUSION_SUMMARY_CSV = "analysis_participant_exclusion_summary.csv"


def append_exclusion_summary_row(analysis_name, method, stats, out_path=None):
    """One shared table across every analysis stage (Markov, feature analysis, ...) and
    method, so participant counts/exclusion-category breakdowns sit side by side instead
    of scattered per-stage (decision 2026-09-27). `stats` is a dict of
    {n_total_eligible, n_included, n_excluded_technical, n_excluded_panel_count,
    n_excluded_hardcoded} - see markov_loader.DataManager.compute_exclusion_summary for
    how those categories are defined. Upserts by (analysis, method): re-running the same
    analysis/method replaces its row instead of duplicating it.

    out_path: where the shared CSV lives (None uses the old global default under
    noam_m_root()) - pass the top level of a consolidated run folder (sibling to
    markov_analysis/feature_analysis, not nested inside either) so "all steps together"
    genuinely means one file outside every per-stage folder."""
    import pandas as pd
    path = out_path if out_path is not None else os.path.join(noam_m_root(), EXCLUSION_SUMMARY_CSV)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"analysis": analysis_name, "method": method, **stats}
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[~((df["analysis"] == analysis_name) & (df["method"] == method))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
    print(f"Updated exclusion summary row '{analysis_name}'/{method} -> {path}")

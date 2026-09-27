"""
Single entry point for the whole Stage 2/3 pipeline (Markov + feature analysis +
symbol-search analysis + duration analysis), over the ENTIRE population, writing
everything into one consolidated, same-day output folder
(pipeline_config.consolidated_results_dir). Meant to be run unattended - each of the 4
stages is wrapped independently, so a failure in one (logged, with a full traceback)
does not prevent the others from running; a summary of what succeeded/failed prints at
the end.

Prerequisite: Stage 1 preprocessing (run_preprocessing.py) must already have been run
for this annotation method - this script only reads Stage 1's saved annotated CSVs, it
does not run model inference itself.

Usage:
    python run_full_pipeline.py [threshold_based|model_based] [date_str]

date_str is optional - omit it (recommended) to auto-resolve to whichever Stage 1
date-folder is most recently on disk for this method (pipeline_config.latest_annotated_gaze_date).
Only pin a specific date_str if you need to force reading from an older Stage 1 run.

To run this unattended and keep a log:
    python run_full_pipeline.py model_based > full_pipeline_run.log 2>&1 &
"""
import os
import sys
import time
import traceback

import pipeline_config as pc
import main as markov_main
import feature_pipeline as fp
import symbol_search_analysis as ssa
import duration_analysis as da


def run(annotation_method="model_based", date_str=None):
    consolidated_root = pc.consolidated_results_dir(date_str)
    markov_base = os.path.join(consolidated_root, "markov_analysis")
    feature_run_dir = os.path.join(consolidated_root, "feature_analysis")
    scores_csv_path = os.path.join(consolidated_root, "behavior_scores", "all_scores.csv")
    # Sits at the TOP LEVEL of consolidated_root - sibling to markov_analysis/ and
    # feature_analysis/, not nested inside either - so "all steps together" genuinely
    # means one shared file outside every per-stage folder (decision 2026-09-27).
    exclusion_summary_path = os.path.join(consolidated_root, "analysis_participant_exclusion_summary.csv")

    print(f"=== FULL PIPELINE RUN: method={annotation_method}, date_str={date_str or '(auto-resolve)'} ===")
    print(f"Output root: {consolidated_root}\n", flush=True)

    stages = [
        ("Markov", lambda: markov_main.run_pipeline(
            annotation_method=annotation_method, date_str=date_str, output_base_path=markov_base,
            exclusion_summary_path=exclusion_summary_path)),
        ("Feature analysis", lambda: fp.main(
            annotation_method=annotation_method, date_str=date_str, run_dir=feature_run_dir,
            markov_output_base_path=markov_base, scores_csv_path=scores_csv_path,
            exclusion_summary_path=exclusion_summary_path)),
        ("Symbol search analysis", lambda: ssa.main(
            annotation_method=annotation_method, run_dir=feature_run_dir, date_str=date_str,
            scores_csv_path=scores_csv_path)),
        ("Duration analysis", lambda: da.main(
            annotation_method=annotation_method, run_dir=feature_run_dir, date_str=date_str)),
    ]

    results = {}
    for name, fn in stages:
        print(f"\n{'=' * 60}\nSTARTING: {name}\n{'=' * 60}", flush=True)
        t0 = time.time()
        try:
            fn()
            results[name] = f"OK ({time.time() - t0:.0f}s)"
        except Exception as e:
            results[name] = f"FAILED: {type(e).__name__}: {e}"
            print(f"!!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        print(f"--- {name} finished in {time.time() - t0:.0f}s ---", flush=True)

    print(f"\n{'=' * 60}\nFULL PIPELINE RUN SUMMARY\n{'=' * 60}")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"\nOutput root: {consolidated_root}")


if __name__ == "__main__":
    method_arg = sys.argv[1] if len(sys.argv) > 1 else "model_based"
    if method_arg not in pc.ANNOTATION_METHODS:
        print(f"usage: python run_full_pipeline.py [{'|'.join(pc.ANNOTATION_METHODS)}] [date_str]")
        sys.exit(1)
    date_str_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run(method_arg, date_str_arg)

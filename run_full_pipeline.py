"""
Single entry point for the whole Stage 2/3 pipeline (scores + Markov + feature analysis +
the live board's cache), over the ENTIRE population, writing everything into the dated
folder of the Stage 1 run it analyzes (<Noam_M>/<date>/, see run() for the layout).
Meant to be run unattended - each stage is wrapped independently, so a failure in one
(logged, with a full traceback) does not prevent the others from running; a summary of
what succeeded/failed prints at the end.

Prerequisite: Stage 1 preprocessing (run_preprocessing.py) must already have been run
for this annotation method - this script only reads Stage 1's saved annotated CSVs, it
does not run model inference itself.

Usage:
    python run_full_pipeline.py [threshold_based|model_based] [date_str]

date_str is optional - omit it (recommended) to auto-resolve to whichever Stage 1
date-folder is most recently on disk for this method (pipeline_config.latest_annotated_gaze_date).
Only pin a specific date_str if you need to force reading from an older Stage 1 run.

To run this unattended (the log is also written to <date>/pipeline_run_<method>.log):
    nohup caffeinate -i python3 -u run_full_pipeline.py model_based > /dev/null 2>&1 &
"""
import os
import sys
import time
import traceback

import pipeline_config as pc
import main as markov_main
import feature_pipeline as fp


class _Tee:
    """Echo stdout/stderr into <date>/pipeline_run_<method>.log as well (decision 2026-11-12)."""
    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def _build_live_board_cache(date_root):
    # live_board_cache resolves its run folder at import time - pin it to this run first.
    os.environ["LIVE_BOARD_RUN_ROOT"] = date_root
    import live_board_cache
    live_board_cache.main(["--force", "--with-pso"])


def _save_scores(scores_csv_path):
    """SDMT scores from the WAV filenames, saved once up front so Markov (which reads this
    CSV when it exists) and feature analysis both use this run's fresh table."""
    fp.save_scores_table(fp.discover_participants(), out_path=scores_csv_path)


def run(annotation_method="model_based", date_str=None):
    """Layout (decision 2026-11-12), all under the <Noam_M>/<date>/ folder that holds the
    Stage 1 run being analyzed (pipeline_config.stage1_date_root):
        SDMT_annotated_gaze_<date>_<method>/       Stage 1 (run_preprocessing.py)
        behavior_scores/all_scores.csv
        analysis_participant_exclusion_summary.csv
        pipeline_run_<method>.log
        live_board/                                board cache + exports
        analysis_results/markov/, features/, key_entry_strategy/, symbol_count_over_panels/"""
    date_str = date_str or pc.latest_annotated_gaze_date(annotation_method)
    if date_str is None:
        raise FileNotFoundError(f"No Stage 1 folder for {annotation_method} - run run_preprocessing.py first.")
    date_root = pc.stage1_date_root(annotation_method, date_str)
    results_root = os.path.join(date_root, "analysis_results")
    markov_base = os.path.join(results_root, "markov")
    feature_run_dir = os.path.join(results_root, "features")
    scores_csv_path = os.path.join(date_root, "behavior_scores", "all_scores.csv")
    exclusion_summary_path = os.path.join(date_root, "analysis_participant_exclusion_summary.csv")

    os.makedirs(date_root, exist_ok=True)
    log_file = open(os.path.join(date_root, f"pipeline_run_{annotation_method}.log"), "a")
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, log_file), _Tee(sys.__stderr__, log_file)

    print(f"=== FULL PIPELINE RUN: method={annotation_method}, Stage 1 date={date_str} ===")
    print(f"Stage 1 input: {pc.annotated_gaze_dir(annotation_method, date_str)}")
    print(f"Output root: {date_root}\n", flush=True)

    # Symbol-search analysis and the standalone duration analysis are no longer run
    # (decision 2026-11-12); PSO duration stays a feature inside feature analysis.
    stages = [
        ("Scores", lambda: _save_scores(scores_csv_path)),
        ("Markov", lambda: markov_main.run_pipeline(
            annotation_method=annotation_method, date_str=date_str, output_base_path=markov_base,
            exclusion_summary_path=exclusion_summary_path, scores_dir=os.path.dirname(scores_csv_path))),
        ("Feature analysis", lambda: fp.main(
            annotation_method=annotation_method, date_str=date_str, run_dir=feature_run_dir,
            markov_output_base_path=markov_base, scores_csv_path=scores_csv_path,
            exclusion_summary_path=exclusion_summary_path)),
        # Last (decision 2026-11-11): rebuilds live_board's reliability cache from THIS
        # run's tables (--force, so a rerun never keeps a stale cache).
        ("Live board cache", lambda: _build_live_board_cache(date_root)),
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
    print(f"\nOutput root: {date_root}")
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    log_file.close()


if __name__ == "__main__":
    method_arg = sys.argv[1] if len(sys.argv) > 1 else "model_based"
    if method_arg not in pc.ANNOTATION_METHODS:
        print(f"usage: python run_full_pipeline.py [{'|'.join(pc.ANNOTATION_METHODS)}] [date_str]")
        sys.exit(1)
    date_str_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run(method_arg, date_str_arg)

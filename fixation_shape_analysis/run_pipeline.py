"""
Orchestrates the full fixation-shape pipeline: Step 0 (load/filter) -> Step 1 (moments) ->
Step 2 (modality) -> Step 3 (GMM) -> Step 4 (PCA/clustering/MDS, x and y independently) -> summary.md.

Usage:
    python -m fixation_shape_analysis.run_pipeline                       # full run
    python -m fixation_shape_analysis.run_pipeline --limit-units 30      # smoke test
    python -m fixation_shape_analysis.run_pipeline --no-gmm-plots        # skip the ~N*2 PNGs

Each step's module can also be run/imported standalone (see the `if __name__ == "__main__"`
block at the bottom of each step*.py) as long as the upstream CSVs it depends on already
exist under outputs/tables/.
"""

import argparse
import time

from . import config
from .step0_load_data import run_step0
from .step1_moments import run_step1
from .step2_modality import run_step2
from .step3_gmm import run_step3
from .step4_comparison import run_step4
from .summary import run_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-data-path", default=config.DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--min-fixations", type=int, default=config.MIN_FIXATIONS_PER_UNIT)
    parser.add_argument("--limit-units", type=int, default=None,
                         help="Cap the number of participant-panel units scanned in Step 0 (smoke testing).")
    parser.add_argument("--no-gmm-plots", action="store_true",
                         help="Skip the per-unit Step 3 diagnostic PNGs (still fits/reports the GMMs).")
    parser.add_argument("--include-below-dictionary-boundary", action="store_true",
                         help="Also include fixations on the digit-sequence search text, below the "
                              "dictionary boundary (by default only fixations above it are analyzed).")
    args = parser.parse_args()

    t0 = time.time()
    included_df, excluded_df = run_step0(main_data_path=args.main_data_path,
                                          min_fixations=args.min_fixations,
                                          limit_units=args.limit_units,
                                          filter_above_dictionary_boundary=not args.include_below_dictionary_boundary)

    moments_df = run_step1(included_df)
    modality_df = run_step2(included_df, moments_df)
    gmm_df = run_step3(included_df, moments_df, make_plots=not args.no_gmm_plots)
    run_step4(included_df, moments_df, modality_df, gmm_df)
    run_summary(min_fixations=args.min_fixations)

    print(f"Pipeline finished in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()

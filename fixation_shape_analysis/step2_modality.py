"""
Step 2 - modality tests per participant-panel/axis: Hartigan's dip test and the
sample-size-corrected bimodality coefficient (BC).
"""

import diptest
import numpy as np

from . import config


def bimodality_coefficient(skewness, kurtosis, n):
    """
    BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))
    kurtosis here is EXCESS kurtosis (Fisher), matching step1's `kurtosis` column - the "+3"
    correction term restores the finite-sample-corrected excess-kurtosis denominator from the
    standard SAS/Pfister et al. formulation.
    """
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (skewness ** 2 + 1) / (kurtosis + correction)


def compute_modality_tests(fixation_df, moments_df):
    rows = []
    for _, moment_row in moments_df.iterrows():
        participant_id, trial_id, axis = moment_row["participant_id"], moment_row["trial_id"], moment_row["axis"]
        values = fixation_df.loc[
            (fixation_df["participant_id"] == participant_id) & (fixation_df["trial_id"] == trial_id), axis
        ].to_numpy()

        dip_stat, dip_pval = diptest.diptest(values)
        bc = bimodality_coefficient(moment_row["skewness"], moment_row["kurtosis"], moment_row["n_fixations"])
        likely_multimodal = (dip_pval < config.DIP_TEST_ALPHA) or (bc > config.BIMODALITY_COEFFICIENT_THRESHOLD)

        rows.append({
            "participant_id": participant_id,
            "trial_id": trial_id,
            "unit_id": moment_row["unit_id"],
            "axis": axis,
            "n_fixations": moment_row["n_fixations"],
            "dip_statistic": dip_stat,
            "dip_pvalue": dip_pval,
            "bimodality_coefficient": bc,
            "modality_label": "likely multimodal" if likely_multimodal else "likely unimodal",
        })

    import pandas as pd
    return pd.DataFrame(rows)


def run_step2(fixation_df, moments_df):
    config.ensure_output_dirs()

    modality_df = compute_modality_tests(fixation_df, moments_df)
    modality_df.to_csv(config.table_path("step2_modality_tests.csv"), index=False)

    n_multimodal = (modality_df["modality_label"] == "likely multimodal").sum()
    print(f"Step 2: {n_multimodal}/{len(modality_df)} participant-panel/axis rows flagged likely multimodal.")
    return modality_df


if __name__ == "__main__":
    from .step0_load_data import run_step0
    from .step1_moments import run_step1
    included_df, _ = run_step0()
    moments_df = run_step1(included_df)
    run_step2(included_df, moments_df)

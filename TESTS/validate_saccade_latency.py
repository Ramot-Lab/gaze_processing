"""
Validates saccade_latency_analysis.py against the existing external table
(.../saccade_latency/Maysan_test/compiled_latencies.csv, test_noam.py: I-DT at 100 Hz)
on a random sample of participants.

  A. Logic check - compute_trial_latencies + filter_latencies, fed Maysan's own saved
     100 Hz classified samples, must reproduce her compiled table EXACTLY, and must match
     a literal port of her utils.saccade_latency_pm loop.
  B. New (gazeNET @ 600 Hz) vs reference (I-DT @ 100 Hz), per trial and per participant.
  C. Decomposition - the same gazeNET labels decimated [::6] to 100 Hz:
       C - reference  = annotation-method effect (I-DT -> gazeNET) at equal sampling
       B - C          = pure sampling-rate effect (100 -> 600 Hz) at equal labels
     Expected sampling effect: the count includes the saccade sample itself, so the
     estimate is biased by about +1 sample interval (Andersson, Nystrom & Holmqvist,
     2010) -> ~+10 ms at 100 Hz vs ~+1.7 ms at 600 Hz, i.e. 600 Hz latencies ~8 ms
     lower on average, with per-trial |difference| mostly <= ~10-20 ms.

Usage: python TESTS/validate_saccade_latency.py [n_participants] [seed]
Writes to <today's feature_analysis run>/saccade_latency/validation/ (no
compiled_latencies.csv is written - the full table only comes from main()).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_config  # noqa: E402
import saccade_latency_analysis as sla  # noqa: E402
from feature_pipeline import resolve_run_dir  # noqa: E402

REFERENCE_DIR = os.path.join(pipeline_config.noam_m_root(), "preliminary_results", "saccade_latency", "Maysan_test")
REFERENCE_CSV = os.path.join(REFERENCE_DIR, "compiled_latencies.csv")
REFERENCE_SAMPLES = os.path.join(REFERENCE_DIR, "classified_samples", "{p}Updated_samples_labeled_71122025.csv")
EXPECTED_SAMPLING_SHIFT_S = -(1 / 100 - 1 / 600)


def saccade_latency_pm_literal(panel_messages, samples_df, sampling_rate_hz=100):
    """Literal port of Maysan's utils.saccade_latency_pm loop (numpy arrays instead of
    df.iloc for speed - same control flow)."""
    ts = samples_df.iloc[:, 0].to_numpy(dtype=float)
    lab = samples_df.iloc[:, -1].to_numpy()
    base_ts = float(panel_messages[0][0] / 1000)
    latency_time, index_l = {}, 0
    for trial in panel_messages:
        index1, count, no_saccade = 0, 0, True
        while no_saccade:
            if index1 >= len(ts):
                break
            if float(trial[0] / 1000) - base_ts <= ts[index1]:
                count += 1
                if lab[index1] == "saccade":
                    no_saccade = False
                    latency_time[index_l] = count / sampling_rate_hz
                    index_l += 1
            index1 += 1
    return latency_time


def _subject_paths():
    return {os.path.basename(s): (g, s) for g, s in sla.iter_fxsp_subject_dirs()}


def main(n_participants=12, seed=0):
    out_dir = os.path.join(resolve_run_dir("compute", base_dir=pipeline_config.feature_analysis_base_dir(
        sla.ANNOTATION_METHOD)), sla.OUTPUT_SUBDIR)
    val_dir = os.path.join(out_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    ref = pd.read_csv(REFERENCE_CSV)
    subjects = _subject_paths()
    pool = sorted(p for p in ref["participant"].unique() if p in subjects)
    sample = list(np.random.default_rng(seed).choice(pool, size=min(n_participants, len(pool)), replace=False))
    print(f"Validating on {len(sample)} random participants (seed={seed}): {sample}")

    trial_rows, logic_rows = [], []
    for p in sample:
        group, subject_path = subjects[p]
        ref_p = ref[ref["participant"] == p].set_index("trial_idx")["saccade_latency_ms"]
        panel = sla.load_fxsp_fixation_panel(subject_path)

        # --- A: logic check on Maysan's own 100 Hz labels ---
        old_samples = pd.read_csv(REFERENCE_SAMPLES.format(p=p))
        vec = sla.filter_latencies(sla.compute_trial_latencies(panel["messages"], old_samples, 100))
        lit = sla.filter_latencies(saccade_latency_pm_literal(panel["messages"], old_samples, 100))
        vec_s = pd.Series(vec)
        logic_rows.append({
            "participant": p, "n_ref": len(ref_p), "n_recomputed": len(vec),
            "vectorized_equals_literal": vec == lit,
            "equals_reference": (len(vec_s) == len(ref_p) and vec_s.index.equals(ref_p.index)
                                 and np.allclose(vec_s.values, ref_p.values)),
        })

        # --- B: new 600 Hz model; C: same labels decimated to 100 Hz ---
        new, info, _ = sla.process_participant(group, subject_path)
        samples_600 = pd.read_parquet(sla.annotated_cache_path(p, panel["mat_path"], panel["dom_eye"]))
        samples_100 = sla.build_samples_df(panel["gaze"], samples_600["evt"].to_numpy(), decimate=6)
        dec = sla.filter_latencies(sla.compute_trial_latencies(panel["messages"], samples_100, 100))
        for k in sorted(set(ref_p.index) | set(new) | set(dec)):
            trial_rows.append({"participant": p, "group": group, "trial_idx": k,
                               "ref_idt_100hz": ref_p.get(k, np.nan), "new_model_600hz": new.get(k, np.nan),
                               "model_decimated_100hz": dec.get(k, np.nan)})
        print(f"  {p}: ref {len(ref_p)} trials, new {len(new)}, decimated {len(dec)}, "
              f"raw trials w/ saccade {info['n_latencies_raw']}/{info['n_trial_messages']}")

    logic = pd.DataFrame(logic_rows)
    trials = pd.DataFrame(trial_rows)
    logic.to_csv(os.path.join(val_dir, "A_logic_check.csv"), index=False)
    trials.to_csv(os.path.join(val_dir, "BC_trial_comparison.csv"), index=False)

    def _diff_stats(a, b, label):
        both = trials[[a, b]].dropna()
        d = (both[a] - both[b]) * 1000
        return {"comparison": label, "n_trials_both": len(both), "mean_diff_ms": d.mean(), "median_diff_ms": d.median(),
                "sd_diff_ms": d.std(), "pct_within_10ms": (d.abs() <= 10).mean() * 100,
                "pct_within_20ms": (d.abs() <= 20).mean() * 100, "trial_r": both[a].corr(both[b])}

    per_p = trials.groupby("participant")[["ref_idt_100hz", "new_model_600hz", "model_decimated_100hz"]].mean()
    summary = pd.DataFrame([
        _diff_stats("new_model_600hz", "ref_idt_100hz", "B: new 600Hz model - reference 100Hz I-DT"),
        _diff_stats("model_decimated_100hz", "ref_idt_100hz", "C: model@100Hz - reference (annotation effect)"),
        _diff_stats("new_model_600hz", "model_decimated_100hz", "B-C: model@600Hz - model@100Hz (sampling effect)"),
    ])
    summary["participant_mean_r"] = [per_p["new_model_600hz"].corr(per_p["ref_idt_100hz"]),
                                     per_p["model_decimated_100hz"].corr(per_p["ref_idt_100hz"]),
                                     per_p["new_model_600hz"].corr(per_p["model_decimated_100hz"])]
    summary.to_csv(os.path.join(val_dir, "BC_summary.csv"), index=False)
    per_p.to_csv(os.path.join(val_dir, "BC_participant_means.csv"))

    pd.set_option("display.width", 200)
    print("\nA. logic check:\n", logic.to_string(index=False))
    print("\nB/C. differences:\n", summary.round(3).to_string(index=False))
    print(f"\nExpected pure sampling shift ~{EXPECTED_SAMPLING_SHIFT_S * 1000:.1f} ms")
    print("\nParticipant means (s):\n", per_p.round(4).to_string())
    print(f"\nWritten to {val_dir}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))

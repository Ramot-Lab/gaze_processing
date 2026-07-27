import os
import re
from datetime import date as date_cls
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr, linregress, ttest_ind
from scipy.spatial import ConvexHull
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from participant_gaze_data_manager import ParticipantGazeDataManager
from trial_manager import TrialManager


# ============================================================
# CONFIG
# ============================================================
NOAM_DATA_PATH = "/Volumes/ramot/Noam_M/Results/Behavior"
SCORES_CSV = "/Volumes/ramot/Noam_M/preliminary_results/behavior_scores/all_scores.csv"
SACCADE_LATENCY_CSV = "/Volumes/ramot/Noam_M/preliminary_results/saccade_latency/Maysan_test/compiled_latencies.csv"
IDO_TABLE_XLSX = "/Volumes/ramot/Noam_M/df_filtered_behavioral_summary_20260427_131437.xlsx"
OUTPUT_BASE_DIR = "/Volumes/ramot/Noam_M/preliminary_results/all_features/"

EXCLUDED_PARTICIPANTS = ['PT914', 'LY082', 'KH736']

TRIAL_FEATURE_ATTRS = [
    "dispersions_global_bcea",
    "dispersions_global_rms",
    "fixation_counts_per_trial",
    "symbol_counts_per_trial",
    "trial_in_search_durations",
    "time_in_relevant_fixations",
    "num_of_searches_per_trial",
    "unique_counts_per_trial",
    "revisits_per_trial",
]

MIN_TRIALS_RELIABILITY = 150   # internal to trial-level reliability split-half
MIN_PANELS_RELIABILITY = 4     # internal to panel-level reliability split-half
RELIABILITY_REPETITIONS = 1000
TYPICALITY_COVERAGE_PCTL = 20  # keep participants above this percentile of total trials (~top 80%)

ID_COLS_TRIAL = {"participant", "panel", "trial_idx"}
ID_COLS_PANEL = {"participant", "panel"}
ID_COLS_PARTICIPANT = {"participant", "group", "Gender"}

PARTIAL_CORR_COVARIATES = {
    "dispersions_global_bcea_mean": "Global Dispersion (BCEA)",
    "fixation_counts_per_trial_mean": "Fixation Count",
    "symbol_counts_per_trial_mean": "Symbol Count",
    "trial_in_search_durations_mean": "Search Duration",
    "time_in_relevant_fixations_mean": "Time in Relevant Fixations",
    "num_of_searches_per_trial_mean": "Number of Searches",
}

CORRELATION_SUBSETS = {
    "all": lambda df: pd.Series(True, index=df.index),
    "MS": lambda df: df["group"].isin(["MS", "pwMS"]),
    "HC": lambda df: df["group"] == "HC",
    "female": lambda df: df["Gender"].isin(["F", "Female"]),
    "male": lambda df: df["Gender"].isin(["M", "Male"]),
}

# "compute" rebuilds everything from raw Tobii data (slow); "load" reads the
# tables from the most recent (or an explicitly named) previous run.
MODE = "load"
LOAD_DATE = None  # e.g. "2026_07_12"; None -> most recent available run

RUN_RELIABILITY = True
RUN_CORRELATIONS = True
RUN_GROUP_COMPARISON = True
RUN_PCA = True
RUN_PARTIAL_CORR = True


# ============================================================
# 1. PARTICIPANT DISCOVERY
# ============================================================
class Participant:
    def __init__(self, name, group):
        self.name = name
        self.group = group
        self.scores = {}
        self.trial_managers = {}

    def add_score(self, panel_name, score):
        self.scores[panel_name] = score


def discover_participants(data_path=NOAM_DATA_PATH):
    participants_dict = {"HC": [], "pwMS": []}
    for group in participants_dict:
        group_path = os.path.join(data_path, group)
        if not os.path.isdir(group_path):
            continue
        for p_name in os.listdir(group_path):
            if p_name == "NN111":
                continue
            if os.path.isdir(os.path.join(group_path, p_name, "SDMT")):
                participants_dict[group].append(p_name)
    return participants_dict


def load_scores(participants_dict, scores_csv=SCORES_CSV, data_path=NOAM_DATA_PATH):
    try:
        scores_df = pd.read_csv(scores_csv)
    except Exception as e:
        print(f"Warning: could not load scores CSV: {e}")
        scores_df = pd.DataFrame()

    participants = []
    for group, names in participants_dict.items():
        for p_name in names:
            p_obj = Participant(p_name, group)
            scores_found = False

            if not scores_df.empty:
                p_scores = scores_df[(scores_df["Group"] == group) & (scores_df["Participant"] == p_name)]
                if not p_scores.empty:
                    for _, row in p_scores.iterrows():
                        panel_name = str(row["Panel"]).strip().lower()
                        p_obj.add_score(panel_name, row["SDMT_Score"])
                    scores_found = True

            if not scores_found:
                sdmt_path = os.path.join(data_path, group, p_name, "SDMT")
                if os.path.exists(sdmt_path):
                    for f in os.listdir(sdmt_path):
                        if f.endswith(".wav"):
                            pm = re.search(r"img_test_(.+?)_strikes", f)
                            sm = re.search(r"_strikes_(\d+)", f)
                            if pm and sm:
                                p_obj.add_score(pm.group(1).strip().lower(), int(sm.group(1)))

            participants.append(p_obj)

    print(f"Loaded {len(participants)} participants.")
    return participants


# ============================================================
# 2. BUILD TRIAL MANAGERS (compute mode only)
# ============================================================
def build_trial_managers(participants, data_path=NOAM_DATA_PATH):
    valid_participants = []
    for participant in participants:
        participant.trial_managers = {}
        has_any_data = False
        try:
            participant_data = ParticipantGazeDataManager(participant.name, data_path, "SDMT", participant.group)
            for panel in participant.scores.keys():
                try:
                    participant.trial_managers[panel] = TrialManager(participant_data, panel)
                    has_any_data = True
                except Exception as e:
                    print(f"  Error loading {participant.name} - {panel}: {e}")
        except Exception as e:
            print(f"Could not load data manager for {participant.name}: {e}")

        if has_any_data:
            valid_participants.append(participant)
        else:
            print(f"Removing {participant.name} (no valid data found for any panel)")

    print(f"Final number of participants with valid data: {len(valid_participants)}")
    return valid_participants


# ============================================================
# 3. CORE TRIAL / PANEL TABLES
# ============================================================
def extract_trial_table(participants):
    rows = []
    for participant in participants:
        for panel, tm in participant.trial_managers.items():
            feature_dicts = {attr: getattr(tm.features, attr, {}) for attr in TRIAL_FEATURE_ATTRS}
            trial_indices = set()
            for d in feature_dicts.values():
                trial_indices.update(d.keys())

            for t_idx in sorted(trial_indices):
                row = {"participant": participant.name, "panel": panel, "trial_idx": t_idx}
                for attr in TRIAL_FEATURE_ATTRS:
                    row[attr] = feature_dicts[attr].get(t_idx, np.nan)
                rows.append(row)

    return pd.DataFrame(rows)


def build_panel_table(df_trials, participants):
    scores_lookup = {(p.name, panel): score for p in participants for panel, score in p.scores.items()}

    stats = df_trials.groupby(["participant", "panel"])[TRIAL_FEATURE_ATTRS].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.reset_index()
    stats["score"] = stats.apply(lambda r: scores_lookup.get((r["participant"], r["panel"]), np.nan), axis=1)

    # percent_{n}_unique_symbols_in_trial bins, derived straight from unique_counts_per_trial
    percent_rows = []
    for (participant, panel), group in df_trials.groupby(["participant", "panel"]):
        counts = group["unique_counts_per_trial"].dropna()
        total = len(counts)
        row = {"participant": participant, "panel": panel}
        for n in range(10):
            row[f"percent_{n}_unique_symbols_in_trial"] = (counts == n).sum() / total * 100 if total else np.nan
        percent_rows.append(row)

    df_panels = stats.merge(pd.DataFrame(percent_rows), on=["participant", "panel"], how="left")
    return df_panels


def compute_panel_slopes(df_trials, df_panels):
    def _slope(group):
        group = group.sort_values("trial_idx").dropna(subset=["symbol_counts_per_trial"])
        if len(group) < 10:
            return np.nan
        subset = pd.concat([group.head(5), group.tail(5)])
        slope, *_ = linregress(subset["trial_idx"], subset["symbol_counts_per_trial"])
        return slope

    slopes = (
        df_trials.groupby(["participant", "panel"])
        .apply(_slope, include_groups=False)
        .reset_index(name="first5_last5_slope")
    )
    return df_panels.merge(slopes, on=["participant", "panel"], how="left")


def compute_typicality(df_trials, coverage_pctl=TYPICALITY_COVERAGE_PCTL):
    total_trials_per_participant = df_trials.groupby("participant").size()
    cutoff = np.percentile(total_trials_per_participant.values, coverage_pctl)
    eligible_participants = set(total_trials_per_participant[total_trials_per_participant > cutoff].index)

    eligible_trials = df_trials[df_trials["participant"].isin(eligible_participants)]
    trial_means = (
        eligible_trials.groupby(["panel", "trial_idx"])["symbol_counts_per_trial"]
        .mean()
        .rename("trial_mean")
        .reset_index()
    )

    df_trials = df_trials.merge(trial_means, on=["panel", "trial_idx"], how="left")
    df_trials["typicality"] = np.where(
        df_trials["participant"].isin(eligible_participants),
        df_trials["symbol_counts_per_trial"] - df_trials["trial_mean"],
        np.nan,
    )
    df_trials = df_trials.drop(columns=["trial_mean"])

    print(f"Typicality: {len(eligible_participants)}/{df_trials['participant'].nunique()} participants "
          f"above the {coverage_pctl}th percentile cutoff ({cutoff:.1f} total trials).")
    return df_trials


# ============================================================
# 4. WHOLE-PANEL FIXATION DISPERSION (compute mode only)
# ============================================================
def _bcea(positions, p=0.68):
    if len(positions) < 2:
        return 0.0
    x, y = zip(*positions)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
    if std_x == 0 or std_y == 0:
        return 0.0
    rho, _ = pearsonr(x, y)
    k = -np.log(1 - p)
    return 2 * np.pi * k * std_x * std_y * np.sqrt(1 - rho ** 2)


def compute_whole_panel_dispersion(participants):
    rows = []
    for participant in participants:
        values = []
        for tm in participant.trial_managers.values():
            for f in tm.fixations:
                pts = list(zip(f.microsaccades["x"], f.microsaccades["y"]))
                if len(pts) > 2:
                    val = _bcea(pts)
                    if not np.isnan(val) and not np.isinf(val):
                        values.append(val)

        if values:
            rows.append({
                "participant": participant.name,
                "fixation_bcea_mean": np.mean(values),
                "fixation_bcea_std": np.std(values, ddof=1),
            })
        else:
            rows.append({"participant": participant.name, "fixation_bcea_mean": np.nan, "fixation_bcea_std": np.nan})

    return pd.DataFrame(rows)


# ============================================================
# 5. PARTICIPANT TABLE + EXTERNAL MERGES
# ============================================================
def build_participant_table(df_trials, df_panels, participants):
    stats = df_trials.groupby("participant")[TRIAL_FEATURE_ATTRS + ["typicality"]].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.rename(columns={"typicality_mean": "mean_typicality"})
    stats = stats.reset_index()

    total_trials = df_trials.groupby("participant").size().rename("total_trial_count").reset_index()
    mean_score = df_panels.groupby("participant")["score"].mean().rename("score").reset_index()
    group_lookup = pd.DataFrame([{"participant": p.name, "group": p.group} for p in participants])
    slopes = df_panels.groupby("participant")["first5_last5_slope"].mean().rename("first5_last5_slope_mean").reset_index()

    df_participants = stats.merge(total_trials, on="participant", how="left")
    df_participants = df_participants.merge(mean_score, on="participant", how="left")
    df_participants = df_participants.merge(group_lookup, on="participant", how="left")
    df_participants = df_participants.merge(slopes, on="participant", how="left")
    return df_participants


def merge_saccade_latency(df_participants, saccade_csv=SACCADE_LATENCY_CSV):
    try:
        df_lat = pd.read_csv(saccade_csv)
    except Exception as e:
        print(f"Warning: could not load saccade latency CSV: {e}")
        return df_participants

    lat_stats = df_lat.groupby("participant")["saccade_latency_ms"].agg(
        saccade_latency_ms_mean="mean", saccade_latency_ms_std="std"
    ).reset_index()
    return df_participants.merge(lat_stats, on="participant", how="left")


def merge_ido_table(df_participants, ido_xlsx=IDO_TABLE_XLSX):
    try:
        df_ido = pd.read_excel(ido_xlsx)
    except Exception as e:
        print(f"Warning: could not load Ido's table: {e}")
        return df_participants

    if "Tobii_Sucks" in df_ido.columns:
        df_ido = df_ido[df_ido["Tobii_Sucks"] != True]  # noqa: E712

    rename_map = {"Patient_ID": "participant", "Group": "group"}
    cols_to_keep = ["Patient_ID", "Group", "Gender", "Age_Years", "Months_Sick",
                    "KD_Total", "CCMT_Acc_Combined", "CFMT_Acc_Combined"]
    cols_to_keep = [c for c in cols_to_keep if c in df_ido.columns]
    df_ido_clean = df_ido[cols_to_keep].rename(columns=rename_map)

    folder_group = None
    if "group" in df_participants.columns:
        folder_group = df_participants.set_index("participant")["group"].to_dict()

    new_cols = [rename_map.get(c, c) for c in cols_to_keep if c != "Patient_ID"]
    df_participants = df_participants.drop(columns=[c for c in new_cols if c in df_participants.columns])

    merged = df_participants.merge(df_ido_clean, on="participant", how="left")
    if folder_group is not None and "group" in merged.columns:
        merged["group"] = merged["group"].fillna(merged["participant"].map(folder_group))
    return merged


def add_encoded_demographics(df_participants):
    df = df_participants.copy()
    if "group" in df.columns:
        df["group_MS"] = df["group"].map({"HC": 0, "MS": 1, "pwMS": 1})
    if "Gender" in df.columns:
        df["Gender_F"] = df["Gender"].map({"M": 0, "Male": 0, "F": 1, "Female": 1})
    return df


# ============================================================
# 6. RUN FOLDER RESOLUTION + SAVE/LOAD
# ============================================================
def _today_str():
    return date_cls.today().strftime("%Y_%m_%d")


def resolve_run_dir(mode, base_dir=OUTPUT_BASE_DIR, load_date=None):
    if mode == "compute":
        run_dir = os.path.join(base_dir, _today_str())
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    if mode == "load":
        if load_date:
            run_dir = os.path.join(base_dir, load_date)
            if not os.path.isdir(run_dir):
                raise FileNotFoundError(f"No run folder found at {run_dir}")
            return run_dir

        candidates = []
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path) and re.match(r"^\d{4}_\d{2}_\d{2}$", name):
                if os.path.exists(os.path.join(path, "participant_features.csv")):
                    candidates.append(name)
        if not candidates:
            raise FileNotFoundError(f"No previous runs with saved tables found under {base_dir}")
        return os.path.join(base_dir, sorted(candidates)[-1])

    raise ValueError(f"Unknown mode: {mode}")


def save_tables(df_trials, df_panels, df_participants, run_dir):
    df_trials.to_csv(os.path.join(run_dir, "trial_features.csv"), index=False)
    df_panels.to_csv(os.path.join(run_dir, "panel_features.csv"), index=False)
    df_participants.to_csv(os.path.join(run_dir, "participant_features.csv"), index=False)
    print(f"Saved tables to {run_dir}")


def load_tables(run_dir):
    df_trials = pd.read_csv(os.path.join(run_dir, "trial_features.csv"))
    df_panels = pd.read_csv(os.path.join(run_dir, "panel_features.csv"))
    df_participants = pd.read_csv(os.path.join(run_dir, "participant_features.csv"))
    print(f"Loaded tables from {run_dir}")
    return df_trials, df_panels, df_participants


# ============================================================
# 7. RELIABILITY (trial level + panel level)
# ============================================================
def plot_correlation_hist(array, l_size, subject_amount, feature_name, save_path):
    valid = np.asarray(array)
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        print(f"    Skipping histogram for {feature_name} (L={l_size}): no valid correlations (constant/NaN data).")
        return
    # Small L splits can be mathematically deterministic (same value every repetition);
    # round away float noise (~1e-16) so numpy's histogram binning doesn't choke on a
    # near-zero-but-not-exactly-zero range.
    valid = np.round(valid, 10)

    plt.figure(figsize=(8, 5))
    plt.hist(valid, bins=50, color="skyblue", edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(valid), color="red", linestyle="dashed", linewidth=2, label=f"Mean r: {np.mean(valid):.3f}")
    plt.title(f"{feature_name} - Reliability Distribution (L={l_size})\nOver {len(valid)} repetitions, N={subject_amount}")
    plt.xlabel("Pearson Correlation (r)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_random_scatter(half_a, half_b, r_value, feature_name, l_size, iteration, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(half_a, half_b, color="teal", alpha=0.7)
    if len(half_a) > 1 and np.std(half_a) > 0:
        z = np.polyfit(half_a, half_b, 1)
        p = np.poly1d(z)
        order = np.argsort(half_a)
        plt.plot(np.array(half_a)[order], p(np.array(half_a)[order]), "r--", alpha=0.8)
    plt.title(f"{feature_name} - Split-Half Scatter\n(L={l_size}, Iteration={iteration}) r={r_value:.3f}")
    plt.xlabel("Mean (Split Half A)")
    plt.ylabel("Mean (Split Half B)")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def calculate_reliability(data, l_size, repetitions, min_value, feature_name, save_dir=None, plot=True, extract_scatters=False):
    array_corr = np.zeros(repetitions)
    subjects_amount = len(data)
    random_iters = np.random.choice(repetitions, min(2, repetitions), replace=False) if extract_scatters else []

    for i in range(repetitions):
        shuffle = np.random.permutation(np.arange(min_value))
        half_a, half_b = [], []
        for participant_idx in range(subjects_amount):
            half_a.append(np.mean(data[participant_idx][shuffle[:l_size]]))
            half_b.append(np.mean(data[participant_idx][shuffle[l_size:2 * l_size]]))
        r_val = np.corrcoef(half_a, half_b)[0, 1]
        array_corr[i] = r_val

        if i in random_iters and save_dir:
            scatter_path = os.path.join(save_dir, f"{feature_name}_scatter_iter_{i}_L{l_size}.png")
            plot_random_scatter(half_a, half_b, r_val, feature_name, l_size, i, scatter_path)

    if plot and save_dir:
        hist_path = os.path.join(save_dir, f"{feature_name}_hist_L{l_size}.png")
        plot_correlation_hist(array_corr, l_size, subjects_amount, feature_name, hist_path)

    return np.mean(array_corr), l_size


def calculate_reliability_distribution(data, max_l, repetitions, min_value, feature_name):
    results = []
    for l_value in range(1, max_l + 1):
        mean_value, _ = calculate_reliability(data, l_value, repetitions, min_value, feature_name, plot=False)
        results.append(mean_value)
    return results


def evaluate_feature_reliability(df, feature_name, min_count, out_dir, repetitions=RELIABILITY_REPETITIONS):
    counts = df.groupby("participant")[feature_name].count()
    valid_participants = counts[counts >= min_count].index.tolist()
    if len(valid_participants) < 3:
        print(f"  Skipping {feature_name}: fewer than 3 participants have >= {min_count} values.")
        return

    max_l = min_count // 2
    if max_l < 1:
        print(f"  Skipping {feature_name}: min_count={min_count} too small for a split-half.")
        return

    feature_dir = os.path.join(out_dir, feature_name)
    os.makedirs(feature_dir, exist_ok=True)

    data_matrix = np.array([
        df[df["participant"] == p][feature_name].dropna().values[:min_count]
        for p in valid_participants
    ])
    print(f"  {feature_name} (N={len(valid_participants)}, min_count={min_count})")

    mean_correlations = calculate_reliability_distribution(data_matrix, max_l, repetitions, min_count, feature_name)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_l + 1), mean_correlations, marker="o", linestyle="-", color="purple")
    plt.title(f"{feature_name} - Reliability Growth Curve")
    plt.xlabel("L size (values per split half)")
    plt.ylabel("Mean Pearson correlation (r)")
    plt.ylim(-0.1, 1.0)
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(feature_dir, f"{feature_name}_growth_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    calculate_reliability(data_matrix, max_l, repetitions, min_count, feature_name,
                          save_dir=feature_dir, plot=True, extract_scatters=True)


def run_trial_reliability(df_trials, run_dir, min_trials=MIN_TRIALS_RELIABILITY):
    out_dir = os.path.join(run_dir, "reliability", "trial_level")
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = [c for c in df_trials.columns if c not in ID_COLS_TRIAL]
    print(f"Running trial-level reliability (min_trials={min_trials})...")
    for feature in feature_cols:
        evaluate_feature_reliability(df_trials, feature, min_trials, out_dir)


def run_panel_reliability(df_panels, run_dir, min_panels=MIN_PANELS_RELIABILITY):
    out_dir = os.path.join(run_dir, "reliability", "panel_level")
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = [c for c in df_panels.select_dtypes(include=[np.number]).columns if c not in ID_COLS_PANEL]
    print(f"Running panel-level reliability (min_panels={min_panels})...")
    for feature in feature_cols:
        evaluate_feature_reliability(df_panels, feature, min_panels, out_dir)


# ============================================================
# 8. CORRELATION HEATMAPS + SIGNIFICANT SCATTERPLOTS
# ============================================================
def _select_feature_columns(df, extra_exclude=()):
    exclude = ID_COLS_PARTICIPANT | set(extra_exclude)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols = [c for c in numeric_cols if c not in exclude]
    return [c for c in cols if df[c].dropna().nunique() > 1]  # drop zero-variance columns


def compute_correlation_matrix(df, features):
    corr = pd.DataFrame(np.nan, index=features, columns=features)
    pval = pd.DataFrame(np.nan, index=features, columns=features)
    annot = pd.DataFrame("", index=features, columns=features)

    for col1 in features:
        for col2 in features:
            if col1 == col2:
                corr.loc[col1, col2] = 1.0
                pval.loc[col1, col2] = 0.0
                annot.loc[col1, col2] = "1.00"
                continue

            valid = df[[col1, col2]].dropna()
            if len(valid) > 2:
                r, p = pearsonr(valid[col1], valid[col2])
                corr.loc[col1, col2] = r
                pval.loc[col1, col2] = p
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                annot.loc[col1, col2] = f"{r:.2f}{stars}"
            else:
                annot.loc[col1, col2] = "NaN"
    return corr, pval, annot


def plot_heatmap(corr, annot, title, out_path):
    n = len(corr)
    cmap = LinearSegmentedColormap.from_list("custom_blue_red", ["blue", "white", "red"], N=256)
    mask = np.eye(n, dtype=bool)

    plt.figure(figsize=(max(10, n * 0.6), max(8, n * 0.5)))
    plt.gca().set_facecolor("black")
    sns.heatmap(corr, annot=annot, fmt="", cmap=cmap, vmin=-1, vmax=1, mask=mask,
                linewidths=0.5, linecolor="gray", cbar_kws={"label": "Pearson Correlation (r)"})
    plt.title(f"{title}\n* p<0.05, ** p<0.01, *** p<0.001", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_significant_scatterplots(df, features, corr, pval, out_dir, score_col="score"):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            col1, col2 = features[i], features[j]
            p_val = pval.loc[col1, col2]
            r_val = corr.loc[col1, col2]
            if pd.isna(p_val) or p_val >= 0.05:
                continue
            count += 1

            plt.figure(figsize=(8, 6))
            has_score = score_col in df.columns and df[score_col].notna().any()
            if has_score:
                scatter = plt.scatter(df[col1], df[col2], c=df[score_col], cmap="viridis",
                                       s=60, edgecolors="black", alpha=0.8)
                cbar = plt.colorbar(scatter)
                cbar.set_label(f"Mean {score_col}", rotation=270, labelpad=15)
            else:
                plt.scatter(df[col1], df[col2], color="teal", s=60, edgecolors="black", alpha=0.8)

            valid = df[[col1, col2]].dropna()
            if len(valid) > 2 and valid[col1].std() > 0:
                z = np.polyfit(valid[col1], valid[col2], 1)
                p_line = np.poly1d(z)
                x_vals = np.linspace(df[col1].min(), df[col1].max(), 100)
                plt.plot(x_vals, p_line(x_vals), "k--", alpha=0.5)

            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            plt.title(f"{col1}\nvs\n{col2}\nr = {r_val:.3f} {stars}", pad=15)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.grid(True, alpha=0.3)

            fname = f"scatter_{col1}_vs_{col2}.png".replace("/", "_")
            plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()
    return count


def run_correlation_analysis(df_participants, run_dir, excluded=EXCLUDED_PARTICIPANTS):
    base = df_participants[~df_participants["participant"].isin(excluded)].copy()

    for name, mask_fn in CORRELATION_SUBSETS.items():
        subset = base[mask_fn(base)].copy()
        if len(subset) < 5:
            print(f"Skipping correlation subset '{name}': only {len(subset)} participants.")
            continue

        out_dir = os.path.join(run_dir, "correlations", name)
        os.makedirs(out_dir, exist_ok=True)

        features = _select_feature_columns(subset, extra_exclude={"total_trial_count"})
        corr, pval, annot = compute_correlation_matrix(subset, features)
        corr.to_csv(os.path.join(out_dir, "correlation_values.csv"))
        plot_heatmap(corr, annot, f"Feature Correlations - {name} (N={len(subset)})",
                     os.path.join(out_dir, "heatmap.png"))

        n_sig = plot_significant_scatterplots(subset, features, corr, pval, os.path.join(out_dir, "scatterplots"))
        print(f"Correlation subset '{name}': N={len(subset)}, {len(features)} features, {n_sig} significant scatterplots.")


# ============================================================
# 9. GROUP COMPARISON (MS vs HC)
# ============================================================
def run_group_comparison(df_participants, run_dir, excluded=EXCLUDED_PARTICIPANTS):
    out_dir = os.path.join(run_dir, "group_comparison")
    os.makedirs(out_dir, exist_ok=True)

    df = df_participants[~df_participants["participant"].isin(excluded)].copy()
    df = df[df["group"].isin(["HC", "MS", "pwMS"])].copy()
    df["group"] = df["group"].replace({"pwMS": "MS"})

    features = _select_feature_columns(df, extra_exclude={"total_trial_count", "group_MS"})
    print(f"Group comparison: {df['group'].value_counts().to_dict()}")

    results = []
    for feat in features:
        ms_vals = df[df["group"] == "MS"][feat].dropna()
        hc_vals = df[df["group"] == "HC"][feat].dropna()
        if len(ms_vals) < 5 or len(hc_vals) < 5:
            continue
        t_stat, p_val = ttest_ind(ms_vals, hc_vals, equal_var=False)
        results.append({"feature": feat, "MS_mean": ms_vals.mean(), "HC_mean": hc_vals.mean(),
                         "t_stat": t_stat, "p_raw": p_val})

    if not results:
        print("No features had enough data for group comparison.")
        return

    results_df = pd.DataFrame(results)
    _, p_adj, _, _ = multipletests(results_df["p_raw"], alpha=0.05, method="fdr_bh")
    results_df["p_FDR"] = p_adj
    results_df = results_df.sort_values("p_raw").reset_index(drop=True)
    results_df.to_csv(os.path.join(out_dir, "stats.csv"), index=False)

    sorted_features = results_df["feature"].tolist()
    plots_per_image = 4
    for img_idx in range(0, len(sorted_features), plots_per_image):
        chunk = sorted_features[img_idx:img_idx + plots_per_image]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, feat in enumerate(chunk):
            sns.boxplot(data=df, x="group", y=feat, ax=axes[i], hue="group",
                        palette={"MS": "lightcoral", "HC": "skyblue"}, legend=False)
            sns.stripplot(data=df, x="group", y=feat, ax=axes[i], color="black", alpha=0.3, jitter=True)
            p_val = results_df.loc[results_df["feature"] == feat, "p_raw"].values[0]
            p_fdr = results_df.loc[results_df["feature"] == feat, "p_FDR"].values[0]
            rank = img_idx + i + 1
            axes[i].set_title(f"Rank {rank}: {feat}\np_raw={p_val:.3f} | p_FDR={p_fdr:.3f}", fontweight="bold", fontsize=9)
            axes[i].set_xlabel("")
        for j in range(len(chunk), 4):
            fig.delaxes(axes[j])
        plt.suptitle(f"Features ranked {img_idx + 1}-{img_idx + len(chunk)}: MS vs HC", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_ranks_{img_idx + 1}_to_{img_idx + len(chunk)}.png"), dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Group comparison complete: {len(results_df)} features tested, saved to {out_dir}")


# ============================================================
# 10. PCA / PARETO
# ============================================================
def run_pca_pareto(df_participants, run_dir, excluded=EXCLUDED_PARTICIPANTS):
    out_dir = os.path.join(run_dir, "pca_pareto")
    os.makedirs(out_dir, exist_ok=True)

    df = df_participants[~df_participants["participant"].isin(excluded)].copy()
    features = _select_feature_columns(df, extra_exclude={"total_trial_count", "score"})
    if len(features) < 3:
        print("Not enough features for PCA/Pareto analysis.")
        return

    df = df.dropna(subset=features, how="all").reset_index(drop=True)
    df[features] = df[features].fillna(df[features].median())
    if len(df) < 4:
        print("Not enough participants for PCA/Pareto analysis.")
        return

    X_scaled = StandardScaler().fit_transform(df[features].values)

    pca_full = PCA().fit(X_scaled)
    var = pca_full.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var)
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(var) + 1), var, alpha=0.6, color="blue", label="Individual Variance")
    plt.plot(range(1, len(var) + 1), cum_var, marker="o", color="red", label="Cumulative Variance")
    plt.title("Scree / Elbow Plot of Explained Variance", weight="bold")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbow.png"), dpi=150, bbox_inches="tight")
    plt.close()

    n_components = min(3, len(features), len(df) - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    for i in range(n_components):
        plt.figure(figsize=(10, 6))
        weights = pca.components_[i]
        colors = ["red" if w > 0 else "blue" for w in weights]
        plt.bar(features, weights, color=colors)
        plt.title(f"PC{i + 1} Weights ({pca.explained_variance_ratio_[i] * 100:.1f}% Var)", fontsize=13, weight="bold")
        plt.axhline(0, color="black", linewidth=1)
        plt.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"weights_PC{i + 1}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    if n_components >= 2 and "score" in df.columns:
        pairs = [(0, 1)] if n_components == 2 else [(0, 1), (0, 2), (1, 2)]
        _, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
        axes = np.atleast_1d(axes)
        var_pct = pca.explained_variance_ratio_ * 100
        scores = df["score"]
        scatter = None

        for idx, (x_idx, y_idx) in enumerate(pairs):
            ax = axes[idx]
            scatter = ax.scatter(X_pca[:, x_idx], X_pca[:, y_idx], c=scores, cmap="viridis",
                                  alpha=0.8, edgecolors="k", s=50)
            ax.set_xlabel(f"PC{x_idx + 1} ({var_pct[x_idx]:.1f}%)")
            ax.set_ylabel(f"PC{y_idx + 1} ({var_pct[y_idx]:.1f}%)")
            ax.set_title(f"PC{x_idx + 1} vs PC{y_idx + 1}")
            ax.grid(True, alpha=0.3)

            pts = X_pca[:, [x_idx, y_idx]]
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1], "r--", linewidth=2)
                for v_idx in hull.vertices:
                    p_name = df.iloc[v_idx]["participant"]
                    ax.scatter(pts[v_idx, 0], pts[v_idx, 1], c="red", s=100, marker="*", zorder=10)
                    ax.annotate(p_name, (pts[v_idx, 0], pts[v_idx, 1]), xytext=(5, 5),
                                textcoords="offset points", fontsize=8, color="darkred")

        cbar = plt.colorbar(scatter, ax=list(axes), orientation="horizontal", fraction=0.05, pad=0.15)
        cbar.set_label("Mean Participant Score")
        plt.suptitle("PCA Scatters & Convex Hulls", fontsize=16, weight="bold", y=1.05)
        plt.savefig(os.path.join(out_dir, "hulls.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"PCA/Pareto analysis complete, saved to {out_dir}")


# ============================================================
# 11. PARTIAL CORRELATIONS
# ============================================================
def run_partial_correlations(df_participants, run_dir, excluded=EXCLUDED_PARTICIPANTS,
                              var_x="mean_typicality", var_y="score", alpha=0.05):
    out_dir = os.path.join(run_dir, "partial_correlation")
    os.makedirs(out_dir, exist_ok=True)

    df = df_participants[~df_participants["participant"].isin(excluded)].copy()
    features = _select_feature_columns(df, extra_exclude={"total_trial_count"})
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[var_y])
    df_clean[features] = df_clean[features].fillna(df_clean[features].median())

    # (a) orthogonal predictor couples
    sig_with_score = {}
    for feat in features:
        if feat == var_y or df_clean[feat].std() == 0:
            continue
        r, p = pearsonr(df_clean[feat], df_clean[var_y])
        if p < alpha:
            sig_with_score[feat] = {"r": r, "p": p}

    couples = []
    for feat1, feat2 in combinations(sig_with_score.keys(), 2):
        r_inter, p_inter = pearsonr(df_clean[feat1], df_clean[feat2])
        if p_inter >= alpha:
            couples.append({
                "feat1": feat1, "feat2": feat2,
                "feat1_r": sig_with_score[feat1]["r"], "feat2_r": sig_with_score[feat2]["r"],
                "inter_r": r_inter, "inter_p": p_inter,
                "combined_power": abs(sig_with_score[feat1]["r"]) + abs(sig_with_score[feat2]["r"]),
            })
    couples_df = pd.DataFrame(couples)
    if not couples_df.empty:
        couples_df = couples_df.sort_values("combined_power", ascending=False)
    couples_df.to_csv(os.path.join(out_dir, "orthogonal_couples.csv"), index=False)
    print(f"Partial correlation: {len(sig_with_score)} features correlated with {var_y}, {len(couples)} orthogonal couples.")

    # (b) iterative single-covariate partial correlation of var_x vs var_y
    if var_x not in df_clean.columns:
        print(f"'{var_x}' not found, skipping iterative partial correlation plot.")
        return

    base_df = df_clean.dropna(subset=[var_x, var_y])
    if len(base_df) < 4:
        print("Not enough data for iterative partial correlation.")
        return
    base_r, _ = pearsonr(base_df[var_x], base_df[var_y])

    results = []
    for cov_col, cov_label in PARTIAL_CORR_COVARIATES.items():
        if cov_col not in df_clean.columns:
            continue
        temp_df = df_clean.dropna(subset=[var_x, var_y, cov_col])
        if len(temp_df) <= 2:
            continue
        const = sm.add_constant(temp_df[cov_col])
        res_x = sm.OLS(temp_df[var_x], const).fit().resid
        res_y = sm.OLS(temp_df[var_y], const).fit().resid
        partial_r, partial_p = pearsonr(res_x, res_y)
        results.append({"Controlled Feature": cov_label, "Partial_r": partial_r, "p_value": partial_p})

    if not results:
        print("No covariates available for iterative partial correlation.")
        return

    df_results = pd.DataFrame(results).sort_values("Partial_r", ascending=True)
    colors = ["#313695" if abs(r) >= abs(base_r) else "#74add1" for r in df_results["Partial_r"]]

    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_results, x="Partial_r", y="Controlled Feature", hue="Controlled Feature",
                palette=colors, legend=False, edgecolor="black", linewidth=1.5)
    plt.axvline(base_r, color="#a50026", linestyle="--", linewidth=3, zorder=0)
    plt.text(base_r + (0.01 if base_r > 0 else -0.01), -0.5,
             f"Baseline (uncontrolled)\nr = {base_r:.3f}", color="#a50026", fontsize=12, fontweight="bold", va="center")

    for i, row in enumerate(df_results.itertuples()):
        stars = "***" if row.p_value < 0.001 else "**" if row.p_value < 0.01 else "*" if row.p_value < 0.05 else "ns"
        align = "left" if row.Partial_r > 0 else "right"
        offset = 0.01 if row.Partial_r > 0 else -0.01
        plt.text(row.Partial_r + offset, i, f" r = {row.Partial_r:.2f} ({stars})",
                  color="black", fontsize=11, fontweight="bold", va="center", ha=align)

    plt.title(f"Effect of individual features on the\n{var_x} vs {var_y} correlation", fontsize=16, fontweight="bold")
    plt.xlabel("Partial Pearson Correlation (r)", fontsize=13, fontweight="bold")
    plt.ylabel("Feature Controlled For", fontsize=13, fontweight="bold")
    xlim = plt.xlim()
    plt.xlim(xlim[0] - 0.1, xlim[1] + 0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iterative_partial_corr.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Iterative partial correlation complete, saved to {out_dir}")


# ============================================================
# 12. MAIN DRIVER
# ============================================================
def main():
    run_dir = resolve_run_dir(MODE, load_date=LOAD_DATE)
    print(f"Run directory: {run_dir} (mode={MODE})")

    if MODE == "compute":
        participants_dict = discover_participants()
        participants = load_scores(participants_dict)
        participants = build_trial_managers(participants)

        df_trials = extract_trial_table(participants)
        df_panels = build_panel_table(df_trials, participants)
        df_panels = compute_panel_slopes(df_trials, df_panels)
        df_trials = compute_typicality(df_trials)

        df_participants = build_participant_table(df_trials, df_panels, participants)
        dispersion_df = compute_whole_panel_dispersion(participants)
        df_participants = df_participants.merge(dispersion_df, on="participant", how="left")

        df_participants = merge_saccade_latency(df_participants)
        df_participants = merge_ido_table(df_participants)
        df_participants = add_encoded_demographics(df_participants)

        save_tables(df_trials, df_panels, df_participants, run_dir)
    else:
        df_trials, df_panels, df_participants = load_tables(run_dir)

    if RUN_RELIABILITY:
        run_trial_reliability(df_trials, run_dir)
        run_panel_reliability(df_panels, run_dir)
    if RUN_CORRELATIONS:
        run_correlation_analysis(df_participants, run_dir)
    if RUN_GROUP_COMPARISON:
        run_group_comparison(df_participants, run_dir)
    if RUN_PCA:
        run_pca_pareto(df_participants, run_dir)
    if RUN_PARTIAL_CORR:
        run_partial_correlations(df_participants, run_dir)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()

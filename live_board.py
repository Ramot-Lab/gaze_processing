"""
Live board (decision 2026-11-09): local Dash app for exploring the participant features
table - correlation heatmap, pairwise regression with dynamic outliers, per-feature info
and MS-vs-HC effect size, split-half stability, and a mean-vs-SD grid.

Everything is read from existing outputs (live_board_cache.py lists the paths) and goes
through feature_pipeline's own logic rather than re-implementing it:
  - population / group filter : EXCLUDED_PARTICIPANTS + CORRELATION_SUBSETS
  - "Reduced" feature set      : _reduced_feature_columns(..., DOMAIN_HEATMAP_EXTRA_EXCLUDE),
                                 exactly the by_domain heatmap's set (the default view)
  - Pearson matrix             : compute_correlation_matrix (Spearman is the same loop with
                                 spearmanr, available as a toggle)
  - orderings                  : _order_features_by_domain (default), _order_features_by_measure_type,
                                 _order_features_by_clustering (1 - r default, 1 - |r| sub-toggle)
  - colors                     : DOMAIN_COLORS / MEASURE_TYPE_COLORS, heatmap blue-white-red
  - partial correlation        : OLS residuals + Pearson, as run_partial_correlations
  - stability                  : cached feature_pipeline.calculate_reliability distributions,
                                 plus symbol_count_stability's saved trajectory results
  - mean vs SD pairs           : plot_mean_vs_sd.find_mean_sd_pairs

Usage: python live_board.py   ->   http://127.0.0.1:8050
(run python live_board_cache.py first so the stability panel has data)
"""
import glob
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from dash import Dash, dcc, html, dash_table, Input, Output, State, ALL, ctx, no_update
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests

import feature_pipeline as fp
from plot_mean_vs_sd import find_mean_sd_pairs
import live_board_markov
from live_board_cache import (PARTICIPANT_FEATURES_CSV, CACHE_DIR, SYMBOL_COUNT_STABILITY_DIR, FEATURE_TABLES_DIR,
                              EXPORT_DIR, reliability_cache_path)

GROUP_COLORS = {"HC": "#1f77b4", "MS": "#ff7f0e", "pwMS": "#ff7f0e"}  # repo convention (tab:blue/tab:orange)
OUTLIER_COLOR = "#d62728"
HEATMAP_SCALE = [[0, "blue"], [0.5, "white"], [1, "red"]]  # feature_pipeline's custom_blue_red
SCORE_KEY = "score"


# ============================================================
# 1. DATA
# ============================================================
MARKOV_PCS = ("PC1", "PC2", "PC3")
MARKOV_VARIANTS = live_board_markov.MARKOV_VARIANTS
AVAILABLE_VARIANTS = live_board_markov.AVAILABLE_VARIANTS
# PC1-3 of every Markov variant the run has, as their own heatmap columns (decision
# 2026-11-12) - col -> (variant, pc).
MARKOV_COLS = {live_board_markov.pc_column(v, pc): (v, pc) for v in AVAILABLE_VARIANTS for pc in MARKOV_PCS}
DEFAULT_PC_VARIANT = "n_transitions" if "n_transitions" in AVAILABLE_VARIANTS else AVAILABLE_VARIANTS[0]


def markov_pc_df(variant):
    """Participant x PC1-3 of one Markov variant - read through the Markov tab's own
    loader (live_board_markov.load_variant), so both tabs use the exact same PCA fit."""
    means = live_board_markov.load_variant(variant)["means"]
    pcs = [pc for pc in MARKOV_PCS if pc in means.columns]
    return means[["Participant"] + pcs].rename(
        columns={"Participant": "participant", **{pc: live_board_markov.pc_column(variant, pc) for pc in pcs}})


def load_participants():
    df = pd.read_csv(PARTICIPANT_FEATURES_CSV)
    df = df[~df["participant"].isin(fp.EXCLUDED_PARTICIPANTS)].reset_index(drop=True)
    # The pipeline's own Markov_PC1-3 are one variant's PCs; the per-variant columns below
    # replace them so no PC appears twice under two names.
    df = df.drop(columns=[c for c in ("Markov_PC1", "Markov_PC2", "Markov_PC3") if c in df.columns])
    for v in AVAILABLE_VARIANTS:
        df = df.merge(markov_pc_df(v), on="participant", how="left")
    return df


DF = load_participants()
ALL_FEATURES = fp._select_feature_columns(DF)


def subset_df(group):
    return DF[fp.CORRELATION_SUBSETS[group](DF)].copy()


def reduced_features(df):
    return fp._reduced_feature_columns(df, extra_exclude=fp.DOMAIN_HEATMAP_EXTRA_EXCLUDE)


def markov_key(variant):
    return f"markov:{variant}"


def domain_of(col):
    if col in MARKOV_COLS:
        return markov_key(MARKOV_COLS[col][0])
    return SCORE_KEY if col == "score" else fp.FEATURE_DOMAIN.get(fp._domain_base_name(col), "grey")


def source_of(col):
    if col in MARKOV_COLS:
        return markov_key(MARKOV_COLS[col][0])
    return SCORE_KEY if col == "score" else fp.MEASURE_TYPE.get(fp._domain_base_name(col), "other")


MARKOV_SHADES = ["#2f4b7c", "#5b7db1", "#7a5195", "#a57bc4"]  # one per variant, distinct from the domain colours
MARKOV_KEYS = [markov_key(v) for v in AVAILABLE_VARIANTS]
MARKOV_LABEL = {markov_key(v): f"Markov PCs · {MARKOV_VARIANTS[v][1]}" for v in AVAILABLE_VARIANTS}
MARKOV_COLOR = {markov_key(v): MARKOV_SHADES[list(MARKOV_VARIANTS).index(v)] for v in AVAILABLE_VARIANTS}
DOMAIN_KEYS = fp.DOMAIN_ORDER + MARKOV_KEYS + [SCORE_KEY]
DOMAIN_LABEL = {**fp.DOMAIN_LABELS, **MARKOV_LABEL, SCORE_KEY: "SDMT score"}
DOMAIN_COLOR = {**fp.DOMAIN_COLORS, **MARKOV_COLOR, SCORE_KEY: fp.SCORE_COLOR}
SOURCE_KEYS = fp.MEASURE_TYPE_ORDER + MARKOV_KEYS + [SCORE_KEY]
SOURCE_LABEL = {**fp.MEASURE_TYPE_LABELS, **MARKOV_LABEL, SCORE_KEY: "SDMT score"}
SOURCE_COLOR = {**fp.MEASURE_TYPE_COLORS, **MARKOV_COLOR, SCORE_KEY: fp.SCORE_COLOR}

FEATURES_BY_DOMAIN = {d: [c for c in ALL_FEATURES if domain_of(c) == d] for d in DOMAIN_KEYS}
# Default view = the pipeline's reduced set, with the default variant's PCs standing in for
# the pipeline's Markov_PC1-3 (same as before this change).
DEFAULT_SELECTION = {c for c in reduced_features(DF) if c not in MARKOV_COLS} | {
    live_board_markov.pc_column(DEFAULT_PC_VARIANT, pc) for pc in MARKOV_PCS}


def display_name(col, names=None):
    """Heatmap/scatter label: a Markov PC shows the name typed in the Markov section."""
    if col in MARKOV_COLS:
        return live_board_markov.pc_display_name(*MARKOV_COLS[col], names=names)
    return col


# ---------- exclusions (decision 2026-11-12) ----------
def family_of(col):
    """The measurement a column belongs to: excluding a participant from one column
    excludes them from its whole family - mean + std of the same feature, or all three
    PCs of a Markov variant (one mean matrix)."""
    if col in MARKOV_COLS:
        return f"markov_{MARKOV_COLS[col][0]}"
    return fp._domain_base_name(col)


FAMILY_COLUMNS = {}
for _c in ALL_FEATURES:
    FAMILY_COLUMNS.setdefault(family_of(_c), []).append(_c)


def apply_exclusions(df, excl_key):
    """excl_key: tuple of (participant, family). Sets those participants' values of every
    column in the family to NaN, so pairwise deletion drops them from every correlation
    involving that measurement and from nothing else."""
    if not excl_key:
        return df
    df = df.copy()
    for participant, family in excl_key:
        cols = [c for c in FAMILY_COLUMNS.get(family, []) if c in df.columns]
        df.loc[df["participant"] == participant, cols] = np.nan
    return df


def exclusion_key(rows):
    return tuple(sorted({(r["participant"], r["family"]) for r in (rows or [])}))


def load_reliability_cache():
    records = []
    for path in glob.glob(os.path.join(CACHE_DIR, "reliability_*.json")):
        with open(path) as f:
            records += json.load(f)["records"]
    by_col = {}
    for rec in records:
        by_col.setdefault(rec["participant_column"], []).append(rec)
    return by_col


def latest_symbol_count_stability():
    """The board's run folder's own symbol_count_stability/ output (decision 2026-11-11)."""
    d = SYMBOL_COUNT_STABILITY_DIR
    if not os.path.exists(os.path.join(d, "stability_summary.csv")):
        print(f"No symbol-count stability output at {d} - that panel stays empty.")
        return None
    return {"dir": d, "summary": pd.read_csv(os.path.join(d, "stability_summary.csv")),
            "iterations": pd.read_csv(os.path.join(d, "per_iteration_metrics.csv")),
            "splits": pd.read_csv(os.path.join(d, "per_iteration_splits.csv")),
            "trajectories": pd.read_csv(os.path.join(d, "trajectories_all_MS_HC.csv"))}


@lru_cache(maxsize=1)
def symbol_count_matrices():
    """Per-panel trial matrices for re-drawing a saved random split's two half
    trajectories - built by symbol_count_stability's own loader, so the curves are the
    same as that run's 02_example_split figures."""
    import symbol_count_stability as scs
    trials, panels, people = scs.load_data(FEATURE_TABLES_DIR)
    return scs, people, scs.build_panel_matrices(trials, panels, people["participant"].tolist())


RELIABILITY = load_reliability_cache()
SYMBOL_STABILITY = latest_symbol_count_stability()


# ============================================================
# 2. STATISTICS
# ============================================================
def spearman_matrix(df, features):
    """compute_correlation_matrix's loop (pairwise deletion, >2 points) with spearmanr."""
    corr = pd.DataFrame(np.nan, index=features, columns=features)
    pval = pd.DataFrame(np.nan, index=features, columns=features)
    annot = pd.DataFrame("", index=features, columns=features)
    for c1 in features:
        for c2 in features:
            if c1 == c2:
                corr.loc[c1, c2], pval.loc[c1, c2], annot.loc[c1, c2] = 1.0, 0.0, "1.00"
                continue
            valid = df[[c1, c2]].dropna()
            if len(valid) > 2:
                r, p = spearmanr(valid[c1], valid[c2])
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                corr.loc[c1, c2], pval.loc[c1, c2], annot.loc[c1, c2] = r, p, f"{r:.2f}{stars}"
            else:
                annot.loc[c1, c2] = "NaN"
    return corr, pval, annot


@lru_cache(maxsize=64)
def correlation_tables(group, method, features, excl_key=()):
    df = apply_exclusions(subset_df(group), excl_key)
    features = list(features)
    if method == "pearson":
        corr, pval, annot = fp.compute_correlation_matrix(df, features)
    else:
        corr, pval, annot = spearman_matrix(df, features)
    notna = df[features].notna().astype(int)
    n = notna.T.dot(notna)

    # BH-FDR over the distinct (upper-triangle) pairs of the displayed matrix, same
    # multipletests(method="fdr_bh") the pipeline uses for its group comparisons.
    p_fdr = pd.DataFrame(np.nan, index=features, columns=features)
    iu = np.triu_indices(len(features), k=1)
    raw = pval.values[iu]
    ok = ~np.isnan(raw)
    if ok.any():
        adj = np.full(raw.shape, np.nan)
        adj[ok] = multipletests(raw[ok], method="fdr_bh")[1]
        m = np.full((len(features),) * 2, np.nan)
        m[iu] = adj
        m.T[iu] = adj
        p_fdr = pd.DataFrame(m, index=features, columns=features)
    return corr, pval, p_fdr, annot, n


def _with_markov_last(ordered_fn, features):
    """The pipeline's own domain/source ordering for its features, then the Markov PC
    groups (variant by variant), then score last - same rule as the pipeline's."""
    markov = [c for c in MARKOV_COLS if c in features]
    rest = ordered_fn([f for f in features if f not in MARKOV_COLS])
    score = [f for f in rest if f == "score"]
    return [f for f in rest if f != "score"] + markov + score


def order_features(features, corr, scheme, cluster_abs, target):
    if scheme == "domain":
        return _with_markov_last(fp._order_features_by_domain, features)
    if scheme == "source":
        return _with_markov_last(fp._order_features_by_measure_type, features)
    if scheme == "cluster":
        return fp._order_features_by_clustering(corr.loc[features, features], absolute=cluster_abs)
    if scheme == "target" and target in features:
        rest = corr.loc[target].drop(target).abs().sort_values(ascending=False, na_position="last")
        return [target] + list(rest.index)
    return features


def corr_fn(method):
    return pearsonr if method == "pearson" else spearmanr


def ols_band(x, y):
    """OLS fit + 95% CI of the mean prediction (as plot_mean_vs_sd)."""
    model = sm.OLS(y, sm.add_constant(x)).fit()
    xs = np.linspace(np.min(x), np.max(x), 100)
    pred = model.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
    return xs, pred["mean"].values, pred["mean_ci_lower"].values, pred["mean_ci_upper"].values


def outlier_mask(values, rule, k, threshold):
    v = np.asarray(values, dtype=float)
    if rule == "sd":
        return np.abs(v - v.mean()) > k * v.std(ddof=1)
    if rule == "mad":
        # Scaled MAD (x1.4826, consistent with SD under normality) - Leys et al., 2013,
        # J Exp Soc Psychol 49:764, "Detecting outliers: Do not use standard deviation
        # around the mean, use absolute deviation around the median".
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        return np.abs(v - med) > k * mad if mad > 0 else np.zeros(len(v), bool)
    if rule == "abs":
        return v > threshold
    return np.zeros(len(v), bool)


def hedges_g(a, b):
    """MS - HC standardized mean difference, small-sample corrected, with the large-sample
    normal CI (Hedges & Olkin, 1985, Statistical Methods for Meta-Analysis, ch. 5)."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    g = (1 - 3 / (4 * (n1 + n2) - 9)) * (np.mean(a) - np.mean(b)) / sp
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return g, g - 1.96 * se, g + 1.96 * se


def demo_sub(df):
    """Demographics (standing rule: N, HC/pwMS, F/M on every plot) as a title subtitle,
    for figures where a corner box would collide with the legend or axis labels."""
    return "<br><sup>" + fp.demographics_text(df).replace("\n", " | ") + "</sup>"


def demo_annotation(fig, df, y=-0.12):
    fig.add_annotation(text=fp.demographics_text(df).replace("\n", " | "), xref="paper", yref="paper",
                       x=0, y=y, showarrow=False, xanchor="left", font=dict(size=10, color="#555"),
                       bgcolor="rgba(255,255,255,0.8)", bordercolor="#aaa", borderwidth=1)


# ============================================================
# 3. FIGURES
# ============================================================
def heatmap_figure(group, method, features, scheme, cluster_abs, target, mask_ns, cell_text,
                    excl_key=(), names=None):
    df = subset_df(group)
    corr, pval, p_fdr, annot, n = correlation_tables(group, method, tuple(features), excl_key)
    labels_of = {f: display_name(f, names) for f in features}
    order = order_features(features, corr, scheme, cluster_abs, target)
    corr, pval, p_fdr, annot, n = (t.loc[order, order] for t in (corr, pval, p_fdr, annot, n))
    k = len(order)

    z = corr.values.copy()
    np.fill_diagonal(z, np.nan)
    text = annot.values.copy() if cell_text == "r" else n.values.astype(str)
    if mask_ns:
        hide = ~(p_fdr.values < 0.05)
        np.fill_diagonal(hide, True)
        z[hide] = np.nan
        text = np.where(hide, "", text)

    if scheme == "source":
        key_fn, keys, labels, colors, title = source_of, SOURCE_KEYS, SOURCE_LABEL, SOURCE_COLOR, "Source"
    else:
        key_fn, keys, labels, colors, title = domain_of, DOMAIN_KEYS, DOMAIN_LABEL, DOMAIN_COLOR, "Cognitive domain"
    feat_keys = [key_fn(f) for f in order]
    used = [g for g in keys if g in feat_keys]
    idx = {g: i for i, g in enumerate(used)}
    bar = np.array([idx[g] for g in feat_keys], dtype=float)
    m = max(len(used), 1)
    bar_scale = []
    for i, g in enumerate(used):
        bar_scale += [[i / m, colors[g]], [(i + 1) / m, colors[g]]]

    fig = make_subplots(rows=2, cols=2, column_widths=[0.025, 0.975], row_heights=[0.025, 0.975],
                        horizontal_spacing=0.004, vertical_spacing=0.004, shared_xaxes=True, shared_yaxes=True)
    custom = np.dstack([corr.values, pval.values, p_fdr.values, n.values,
                        np.array([[labels_of[a] for a in order] for _ in order]),
                        np.array([[labels_of[b] for _ in order] for b in order])])
    fig.add_trace(go.Heatmap(
        z=z, x=list(range(k)), y=list(range(k)), zmin=-1, zmax=1, colorscale=HEATMAP_SCALE,
        text=text, texttemplate="%{text}", textfont=dict(size=max(6, 11 - k // 8)), customdata=custom,
        hovertemplate=("<b>%{customdata[5]}</b> vs <b>%{customdata[4]}</b><br>r = %{customdata[0]:.3f}"
                       "<br>p = %{customdata[1]:.4f}<br>p_FDR = %{customdata[2]:.4f}<br>n = %{customdata[3]}"
                       "<extra></extra>"),
        colorbar=dict(title=f"{'Pearson' if method == 'pearson' else 'Spearman'} r", x=1.01, len=0.8, y=0.45),
        xgap=1, ygap=1, name="corr"), row=2, col=2)
    bar_kw = dict(colorscale=bar_scale, zmin=0, zmax=m, showscale=False, hoverinfo="skip")
    fig.add_trace(go.Heatmap(z=bar[:, None], x=[0], y=list(range(k)), **bar_kw), row=2, col=1)
    fig.add_trace(go.Heatmap(z=bar[None, :], x=list(range(k)), y=[0], **bar_kw), row=1, col=2)

    if scheme in ("domain", "source"):
        for i in range(1, k):
            if feat_keys[i] != feat_keys[i - 1]:
                for axis in ("x", "y"):
                    fig.add_shape(type="line", xref="x4", yref="y4",
                                  x0=i - 0.5 if axis == "x" else -0.5, x1=i - 0.5 if axis == "x" else k - 0.5,
                                  y0=-0.5 if axis == "x" else i - 0.5, y1=k - 0.5 if axis == "x" else i - 0.5,
                                  line=dict(color="black", width=2))
    for g in used:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=labels[g],
                                 marker=dict(size=12, color=colors[g], symbol="square"), legendgroup="groups"))

    ticks = [f"<span style='color:{colors[key_fn(f)]}'>{'<b>' + f + '</b>' if f == 'score' else labels_of[f]}</span>"
             for f in order]
    fig.update_xaxes(tickvals=list(range(k)), ticktext=ticks, tickangle=-50, row=2, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    for r, c in ((1, 1), (1, 2), (2, 1)):
        fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, row=r, col=c)
    fig.update_yaxes(tickvals=list(range(k)), ticktext=ticks, showticklabels=True, autorange="reversed",
                     row=2, col=1)
    size = max(700, 38 * k + 300)
    fig.update_layout(
        height=size, width=size + 330, plot_bgcolor="#d9d9d9",
        title=f"Feature correlations - {group} (N={len(df)}), ordered by {scheme}"
              f"{' (1 - |r|)' if scheme == 'cluster' and cluster_abs else ''}"
              f"{' - non-significant (p_FDR >= .05) masked' if mask_ns else ''}"
              f"{f' - {len(excl_key)} participant-measure exclusion(s) applied' if excl_key else ''}<br>"
              "<sup>cell text: " + ("r, * p<.05 ** p<.01 *** p<.001 (uncorrected)" if cell_text == "r" else
                                    "n (pairwise deletion)") + " - click a cell for its scatter</sup>" + demo_sub(df),
        legend=dict(title=title, x=1.09, y=1, xanchor="left"), margin=dict(l=10, r=10, t=110, b=10))
    return fig, order


def flagged_by_axis(group, x, y, rule, k, threshold, axes_sel):
    """{"x": [...], "y": [...]} participants the scatter flags on each axis - same rows
    (both values present) and same outlier_mask as scatter_figure."""
    d = subset_df(group)[["participant", x, y]].dropna(subset=[x, y]).reset_index(drop=True)
    out = {}
    if rule == "none" or len(d) < 3:
        return out
    for axis in axes_sel:
        col = x if axis == "x" else y
        out[axis] = d.loc[outlier_mask(d[col], rule, k, threshold), "participant"].tolist()
    return out


def scatter_figure(group, method, x, y, by_group, rule, k, threshold, axes_sel, exclude, covariate):
    df = subset_df(group)
    cols = list(dict.fromkeys(["participant", "group", "Gender", x, y] + ([covariate] if covariate else [])))
    d = df[cols].dropna(subset=[x, y]).reset_index(drop=True)
    if len(d) < 3:
        return go.Figure().update_layout(title=f"Not enough data for {x} vs {y} (n={len(d)})"), ""

    mask = np.zeros(len(d), bool)
    if rule != "none":
        for axis in axes_sel:
            mask |= outlier_mask(d[x if axis == "x" else y], rule, k, threshold)

    fn = corr_fn(method)
    fig = go.Figure()
    for grp, g in d.groupby("group"):
        gm = mask[g.index]
        fig.add_trace(go.Scatter(x=g[x][~gm], y=g[y][~gm], mode="markers", name=grp,
                                 marker=dict(color=GROUP_COLORS.get(grp, "grey"), size=9, opacity=0.75,
                                             line=dict(color="black", width=0.5)),
                                 text=g["participant"][~gm], hovertemplate="%{text}<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>"))
    if mask.any():
        o = d[mask]
        fig.add_trace(go.Scatter(x=o[x], y=o[y], mode="markers+text", name="Outliers", text=o["participant"],
                                 textposition="top center", textfont=dict(size=10, color=OUTLIER_COLOR),
                                 marker=dict(color=[GROUP_COLORS.get(g, "grey") for g in o["group"]], size=12,
                                             line=dict(color=OUTLIER_COLOR, width=2.5)),
                                 hovertemplate="%{text} (outlier)<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>"))

    def add_fit(data, color, name, dash="solid", band=True):
        if len(data) < 3 or data[x].std() == 0:
            return
        xs, ym, lo, hi = ols_band(data[x].values, data[y].values)
        if band:
            fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", line=dict(width=0),
                                     fillcolor=color, opacity=0.15, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=ym, mode="lines", name=name, line=dict(color=color, width=2, dash=dash)))

    def stat(data):
        if len(data) < 3 or data[x].std() == 0 or data[y].std() == 0:
            return np.nan, np.nan, len(data)
        r, p = fn(data[x], data[y])
        return r, p, len(data)

    lines = []
    r, p, n = stat(d)
    add_fit(d, "black", "OLS fit (all points) ± 95% CI")
    lines.append(f"All points: r = {r:.3f}, p = {p:.4g}, n = {n}")
    if exclude and mask.any():
        kept = d[~mask]
        r2, p2, n2 = stat(kept)
        add_fit(kept, "#2ca02c", "OLS fit (outliers excluded)", dash="dash")
        lines.append(f"Outliers excluded: r = {r2:.3f}, p = {p2:.4g}, n = {n2} ({int(mask.sum())} removed)")
    elif mask.any():
        lines.append(f"{int(mask.sum())} outlier(s) flagged: {', '.join(d.loc[mask, 'participant'])}")
    if by_group:
        base = d[~mask] if exclude else d
        for grp, g in base.groupby("group"):
            add_fit(g, GROUP_COLORS.get(grp, "grey"), f"{grp} fit", band=False)
            rg, pg, ng = stat(g)
            lines.append(f"{grp}: r = {rg:.3f}, p = {pg:.4g}, n = {ng}")
    if covariate:
        base = (d[~mask] if exclude else d).dropna(subset=[covariate])
        if len(base) > 3 and base[covariate].std() > 0:
            const = sm.add_constant(base[covariate])
            rx = sm.OLS(base[x], const).fit().resid
            ry = sm.OLS(base[y], const).fit().resid
            pr, pp = pearsonr(rx, ry)
            lines.append(f"Partial r controlling for {covariate} (Pearson on OLS residuals, as "
                         f"run_partial_correlations): r = {pr:.3f}, p = {pp:.4g}, n = {len(base)}")

    xl, yl = display_name(x), display_name(y)
    fig.update_layout(title=f"{yl} vs {xl} - {group} ({'Pearson' if method == 'pearson' else 'Spearman'})",
                      xaxis_title=xl, yaxis_title=yl, height=560, template="plotly_white",
                      legend=dict(orientation="h", y=-0.22), margin=dict(b=130))
    demo_annotation(fig, d, y=-0.36)
    return fig, lines


def info_block(feature):
    d = DF[["participant", "group", "Gender", feature]]
    n_valid = int(d[feature].notna().sum())
    fig = go.Figure()
    for grp in ("HC", "MS"):
        vals = d.loc[fp.CORRELATION_SUBSETS[grp](d), feature].dropna()
        fig.add_trace(go.Histogram(x=vals, name=f"{grp} (n={len(vals)})", opacity=0.6,
                                   marker_color=GROUP_COLORS[grp]))
    fig.update_layout(barmode="overlay", height=280, template="plotly_white",
                      title=f"{feature} by group" + demo_sub(d.dropna(subset=[feature])),
                      margin=dict(t=60, b=40), legend=dict(orientation="h", y=-0.15))

    ms = d.loc[fp.CORRELATION_SUBSETS["MS"](d), feature].dropna().values
    hc = d.loc[fp.CORRELATION_SUBSETS["HC"](d), feature].dropna().values
    if len(ms) > 2 and len(hc) > 2:
        g, lo, hi = hedges_g(ms, hc)
        t, p = ttest_ind(ms, hc, equal_var=False)
        effect = f"MS − HC: Hedges' g = {g:.2f} [95% CI {lo:.2f}, {hi:.2f}]; Welch t = {t:.2f}, p = {p:.4g}"
    else:
        effect = "MS − HC: not enough data"
    return html.Div([
        html.H4(feature, style={"margin": "6px 0"}),
        html.Div(f"Domain: {DOMAIN_LABEL[domain_of(feature)]} | Source: {SOURCE_LABEL[source_of(feature)]}"),
        html.Div(f"n valid = {n_valid} / {len(d)} (missing {100 * (1 - n_valid / len(d)):.1f}%) - all groups"),
        html.Div(effect, style={"fontWeight": "bold"}),
        dcc.Graph(figure=fig, config={"displaylogo": False}),
    ])


def reliability_figures(rec):
    r = np.array([v for v in rec["r"] if v is not None])
    lo, hi = np.percentile(r, [2.5, 97.5])
    hist = go.Figure(go.Histogram(x=r, nbinsx=40, marker_color="#9aa5b1"))
    hist.add_vline(x=r.mean(), line_dash="dash", annotation_text=f"mean r = {r.mean():.3f}")
    hist.add_vrect(x0=lo, x1=hi, fillcolor="#9aa5b1", opacity=0.15, line_width=0)
    who = DF[DF["participant"].isin(rec["participants"])]
    hist.update_layout(height=290, template="plotly_white", margin=dict(t=70, b=40),
                       title=f"{rec['feature']} - {rec['level']}-level split-half r (L={rec['l_size']}, "
                             f"{len(r)} reps); 95% interval [{lo:.3f}, {hi:.3f}]" + demo_sub(who),
                       xaxis_title="Pearson r (half A vs half B means)")
    examples = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Near {lbl}: rep {ex['iteration']}, r = {ex['r']:.3f}" for lbl, ex in
        (("median", rec["examples"]["median"]), ("5th pctl", rec["examples"]["p05"]))])
    for i, key in enumerate(("median", "p05"), start=1):
        ex = rec["examples"][key]
        examples.add_trace(go.Scatter(x=ex["half_a"], y=ex["half_b"], mode="markers", text=rec["participants"],
                                      marker=dict(color="teal", opacity=0.7), showlegend=False,
                                      hovertemplate="%{text}<extra></extra>"), row=1, col=i)
        examples.update_xaxes(title_text="Mean (half A)", row=1, col=i)
        examples.update_yaxes(title_text="Mean (half B)", row=1, col=i)
    examples.update_layout(height=340, template="plotly_white", margin=dict(t=80, b=40),
                           title="Example split-half repetitions" + demo_sub(who))
    return [dcc.Graph(figure=hist), dcc.Graph(figure=examples)]


def symbol_trajectory_block():
    s = SYMBOL_STABILITY
    it = s["iterations"]
    panels = list(s["summary"]["panel"].astype(str))
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"Panel {p}" for p in panels])
    for i, (_, row) in enumerate(s["summary"].iterrows()):
        rr, cc = i // 3 + 1, i % 3 + 1
        vals = it.loc[it["panel"].astype(str) == str(row["panel"]), "r"].dropna()
        fig.add_trace(go.Histogram(x=vals, marker_color="#9aa5b1", showlegend=False), row=rr, col=cc)
        fig.add_vline(x=row["ms_vs_hc_r"], line_color=OUTLIER_COLOR, line_width=2.5, row=rr, col=cc)
    fig.update_layout(height=450, template="plotly_white", margin=dict(t=90),
                      title="Symbol-count trajectory stability: random split-half r (grey) vs MS–HC r (red)"
                            + demo_sub(DF))
    table = s["summary"][["panel", "null_mean_r", "null_ci95_low", "null_ci95_high", "ms_vs_hc_r",
                          "ms_vs_hc_percentile", "p_one_sided_lower"]].round(3)
    return html.Div([
        html.H5(f"Trajectory stability (cohort level, from {s['dir']})"),
        html.Table([html.Tr([html.Th(c) for c in table.columns])] +
                   [html.Tr([html.Td(v) for v in row]) for row in table.values], className="small-table"),
        dcc.Graph(figure=fig),
        dcc.Graph(figure=random_split_trajectory_figure()),
        dcc.Graph(figure=ms_hc_trajectory_figure()),
    ])


def _trajectory_grid(panels):
    return make_subplots(rows=2, cols=3, subplot_titles=panels, vertical_spacing=0.16, horizontal_spacing=0.06)


PARTICIPATION_CUTOFF = 0.8  # same 80% rule as behavioral_analyzer / symbol_search_analysis


def _participation_cutoff_trial(traj, frac=PARTICIPATION_CUTOFF):
    """First trial_idx at which fewer than frac of this curve's own participants still
    contribute (n < frac * n_participants), or None if it never drops that low."""
    below = traj.loc[traj["n"] < frac * traj["n_participants"].iloc[0], "trial_idx"]
    return None if below.empty else below.iloc[0]


def _add_trajectory(fig, traj, color, name, rr, cc, show_legend):
    # SD is undefined where a single participant contributes (late trials) - the band
    # is drawn only over trials with a real SD, so the polygon never jumps across gaps.
    band = traj.dropna(subset=["sd"])
    x, lo, hi = band["trial_idx"].to_numpy(), (band["mean"] - band["sd"]).to_numpy(), (band["mean"] + band["sd"]).to_numpy()
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]],
                             fill="toself", fillcolor=color, opacity=0.18, line=dict(width=0), hoverinfo="skip",
                             showlegend=False, legendgroup=name), row=rr, col=cc)
    fig.add_trace(go.Scatter(x=traj["trial_idx"], y=traj["mean"], mode="lines", line=dict(color=color, width=2),
                             name=name, legendgroup=name, showlegend=show_legend,
                             customdata=np.c_[traj["sd"], traj["n"]],
                             hovertemplate=f"{name}<br>trial %{{x}}<br>mean %{{y:.2f}} ± %{{customdata[0]:.2f}} SD"
                                           "<br>n = %{customdata[1]}<extra></extra>"), row=rr, col=cc)
    cut = _participation_cutoff_trial(traj)
    if cut is not None:
        fig.add_vline(x=cut, line=dict(color=color, width=1.5, dash="dot"), row=rr, col=cc)



def random_split_trajectory_figure():
    """Per panel, the saved split whose r is closest to that panel's median r (the same
    pick as symbol_count_stability's 02_example_split_median.png): both halves' mean ± SD."""
    s = SYMBOL_STABILITY
    scs, people, mats = symbol_count_matrices()
    titles, picks = [], []
    for panel in s["summary"]["panel"].astype(str):
        g = s["iterations"][s["iterations"]["panel"].astype(str) == panel].dropna(subset=["r"])
        row = g.iloc[(g["r"] - g["r"].median()).abs().argmin()]
        picks.append((panel, int(row["iteration"])))
        titles.append(f"Panel {panel} · split {int(row['iteration'])} · r = {row['r']:.3f}")
    fig = _trajectory_grid(titles)
    for i, (panel, it) in enumerate(picks):
        rr, cc = i // 3 + 1, i % 3 + 1
        half_a = set(s["splits"].loc[s["splits"]["iteration"] == it, "half_A_participants"].iloc[0].split(";"))
        mask_a = people["participant"].isin(half_a).to_numpy()
        for name, m, color in (("Half A", mask_a, "#1b9e77"), ("Half B", ~mask_a, "#7570b3")):
            traj = scs.trajectory(mats[panel], m)
            if traj is not None:
                _add_trajectory(fig, traj, color, name, rr, cc, show_legend=(i == 0))
    fig.update_xaxes(title_text="Trial index")
    fig.update_yaxes(title_text="Symbols (mean ± SD)", col=1)
    fig.update_annotations(font_size=11)
    fig.update_layout(height=600, template="plotly_white", margin=dict(t=135),
                      title="Symbol-count trajectory stability: example random split (median-r split per panel)"
                            "<br><sup>dotted line (curve's color) = first trial where < 80% of that half's "
                            "participants still contribute</sup>" + demo_sub(DF), legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"))
    return fig


def ms_hc_trajectory_figure():
    """MS vs HC mean ± SD trajectories, read from the run's saved trajectories_all_MS_HC.csv."""
    s = SYMBOL_STABILITY
    tr = s["trajectories"]
    summ = s["summary"].assign(panel=s["summary"]["panel"].astype(str)).set_index("panel")
    panels = list(summ.index)
    fig = _trajectory_grid([f"Panel {p} · r = {summ.loc[p, 'ms_vs_hc_r']:.3f} · pctl "
                            f"{summ.loc[p, 'ms_vs_hc_percentile']:.1f}" for p in panels])
    for i, panel in enumerate(panels):
        rr, cc = i // 3 + 1, i % 3 + 1
        for grp in ("HC", "MS"):
            traj = tr[(tr["panel"].astype(str) == panel) & (tr["subset"] == grp)].sort_values("trial_idx")
            if len(traj):
                _add_trajectory(fig, traj.reset_index(drop=True), GROUP_COLORS[grp], grp, rr, cc,
                                show_legend=(i == 0))
    fig.update_xaxes(title_text="Trial index")
    fig.update_yaxes(title_text="Symbols (mean ± SD)", col=1)
    fig.update_annotations(font_size=11)
    fig.update_layout(height=600, template="plotly_white", margin=dict(t=135),
                      title="Symbol-count trajectory stability: MS vs HC (r and its random-split percentile)"
                            "<br><sup>dotted line (group's color) = first trial where < 80% of that group's "
                            "participants still contribute</sup>" + demo_sub(DF), legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"))
    return fig


def stability_block(feature):
    parts = [html.H4(f"Stability: {feature}", style={"margin": "6px 0"})]
    recs = RELIABILITY.get(feature, [])
    if not recs:
        parts.append(html.Div(
            "No cached split-half reliability for this feature. Either it is a single value per participant "
            "(score, cognitive tests, Age, Markov PCs - no split-half definition exists in the pipeline), or its "
            "source wasn't cached (pso_duration / fixation_bcea need live_board_cache.py --with-pso / "
            "--with-fixation-bcea).", style={"color": "#666"}))
    for rec in recs:
        parts.append(html.Div(f"Source: {rec['source']} | min values per participant = {rec['min_count']} | "
                              f"N = {len(rec['participants'])} | seed {rec['seed']} | whole cohort "
                              "(the group filter does not apply here)", style={"fontSize": "12px"}))
        parts += reliability_figures(rec)
    if feature == "symbol_counts_per_trial_mean" and SYMBOL_STABILITY is not None:
        parts.append(symbol_trajectory_block())
    return html.Div(parts)


def mean_sd_grid(group):
    df = subset_df(group)
    pairs = find_mean_sd_pairs(df.columns)
    names = sorted(pairs)
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    titles = []
    stats = {}
    for f in names:
        m, s = pairs[f]
        d = df[[m, s]].dropna()
        r, p = pearsonr(d[m], d[s]) if len(d) > 2 and d[m].std() > 0 and d[s].std() > 0 else (np.nan, np.nan)
        stats[f] = (d, r)
        titles.append(f"{f}<br>r = {r:.2f}, p = {p:.3g}, n = {len(d)}")
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, vertical_spacing=0.09, horizontal_spacing=0.06)
    for i, f in enumerate(names):
        rr, cc = i // ncols + 1, i % ncols + 1
        m, s = pairs[f]
        d = df[[m, s, "group", "participant"]].dropna(subset=[m, s])
        for grp, g in d.groupby("group"):
            fig.add_trace(go.Scatter(x=g[m], y=g[s], mode="markers", text=g["participant"], name=grp,
                                     legendgroup=grp, showlegend=(i == 0),
                                     marker=dict(color=GROUP_COLORS.get(grp, "grey"), size=5, opacity=0.7),
                                     hovertemplate="%{text}<extra></extra>"), row=rr, col=cc)
        if len(d) > 2 and d[m].std() > 0:
            xs, ym, lo, hi = ols_band(d[m].values, d[s].values)
            fig.add_trace(go.Scatter(x=xs, y=ym, mode="lines", line=dict(color="black"), showlegend=False),
                          row=rr, col=cc)
        fig.update_xaxes(title_text="mean", title_standoff=2, row=rr, col=cc)
        fig.update_yaxes(title_text="SD", title_standoff=2, row=rr, col=cc)
    fig.update_annotations(font_size=10)
    fig.update_layout(height=290 * nrows + 80, template="plotly_white",
                      title=f"Participant mean vs SD per feature - {group} (pooled OLS fit, Pearson r)"
                            + demo_sub(df))
    return fig


# ============================================================
# 4. LAYOUT
# ============================================================
def radio(id_, options, value, inline=True):
    return dcc.RadioItems(id=id_, options=options, value=value, inline=inline,
                          inputStyle={"marginRight": "4px", "marginLeft": "10px"})


def feature_selector():
    blocks = [html.Div([
        html.Button("Reduced (pipeline)", id="preset-reduced", n_clicks=0),
        html.Button("All", id="preset-all", n_clicks=0),
        html.Button("None", id="preset-none", n_clicks=0),
    ], className="btn-row")]
    for d in DOMAIN_KEYS:
        feats = FEATURES_BY_DOMAIN[d]
        if not feats:
            continue
        blocks.append(html.Div([
            dcc.Checklist(id={"type": "dom-all", "domain": d},
                          options=[{"label": html.B(DOMAIN_LABEL[d], style={"color": DOMAIN_COLOR[d]}), "value": "all"}],
                          value=["all"] if set(feats) <= DEFAULT_SELECTION else []),
            dcc.Checklist(id={"type": "feat-list", "domain": d},
                          options=[{"label": display_name(f), "value": f} for f in feats],
                          value=[f for f in feats if f in DEFAULT_SELECTION], style={"marginLeft": "14px"},
                          inputStyle={"marginRight": "4px"}),
        ], style={"marginBottom": "8px"}))
    return html.Div(blocks, className="sidebar")


app = Dash(__name__, title="Feature live board")
app.index_string = app.index_string.replace("</head>", """<style>
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
.top{display:flex;flex-wrap:wrap;gap:18px;padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;align-items:center}
.top label{font-size:13px}.ctl{display:flex;align-items:center;gap:4px;font-size:13px}
.main{display:flex;gap:12px;padding:12px}.sidebar{width:290px;flex:none;font-size:12px;background:#fff;
border:1px solid #ddd;border-radius:6px;padding:8px;max-height:90vh;overflow:auto}
.panel{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;margin:0 12px 12px}
.btn-row button{margin:0 4px 8px 0;font-size:12px}.small-table{font-size:12px;border-collapse:collapse}
.small-table td,.small-table th{border:1px solid #ddd;padding:2px 6px}
.two{display:flex;gap:12px;flex-wrap:wrap}.two>div{flex:1;min-width:480px}
</style></head>""")

feature_options = [{"label": f, "value": f} for f in ALL_FEATURES]
covariate_options = [{"label": "None", "value": ""}] + feature_options

app.layout = html.Div([
    html.Div([
        html.B("Feature live board"),
        html.Div(["Group:", radio("group", ["all", "MS", "HC"], "all")], className="ctl"),
        html.Div(["Correlation:", radio("method", [{"label": "Pearson", "value": "pearson"},
                                                   {"label": "Spearman", "value": "spearman"}], "pearson")],
                 className="ctl"),
        html.Div(["Order:", radio("scheme", [{"label": "Cognitive domain", "value": "domain"},
                                             {"label": "Source", "value": "source"},
                                             {"label": "Clustered", "value": "cluster"},
                                             {"label": "|r| with target", "value": "target"}], "domain")],
                 className="ctl"),
        html.Div(["Cluster on:", radio("cluster-abs", [{"label": "1 − r", "value": "r"},
                                                       {"label": "1 − |r|", "value": "abs"}], "r")], className="ctl"),
        html.Div(["Target:", dcc.Dropdown(id="target", options=feature_options, value="score", clearable=False,
                                          style={"width": "230px"})], className="ctl"),
        dcc.Checklist(id="mask-ns", options=[{"label": "Mask p_FDR ≥ .05", "value": "on"}], value=[]),
        html.Div(["Cell text:", radio("cell-text", [{"label": "r", "value": "r"}, {"label": "n", "value": "n"}], "r")],
                 className="ctl"),
        html.Button("Mean vs SD grid", id="meansd-btn", n_clicks=0),
    ], className="top"),
    html.Div([feature_selector(),
              html.Div(dcc.Loading(dcc.Graph(id="heatmap", config={"displaylogo": False})),
                       style={"flex": 1, "overflow": "auto"})], className="main"),
    dcc.Store(id="order-store"),
    dcc.Store(id="pc-names-store", data=live_board_markov.load_pc_names()),
    # Exclusion table lives in the browser's local storage so a refresh doesn't lose it.
    dcc.Store(id="excl-store", storage_type="local", data=[]),
    html.Div([
        html.H3("Pairwise relationship", style={"marginTop": 0}),
        html.Div([
            html.Div(["X:", dcc.Dropdown(id="x-feat", options=feature_options, value="symbol_counts_per_trial_mean",
                                          clearable=False, style={"width": "280px"})], className="ctl"),
            html.Div(["Y:", dcc.Dropdown(id="y-feat", options=feature_options, value="score", clearable=False,
                                          style={"width": "280px"})], className="ctl"),
            dcc.Checklist(id="by-group", options=[{"label": "Separate MS/HC lines", "value": "on"}], value=[]),
            html.Div(["Partial r, control for:", dcc.Dropdown(id="covariate", options=covariate_options, value="",
                                                               style={"width": "230px"})], className="ctl"),
        ], className="top", style={"border": "none", "padding": "4px 0"}),
        html.Div([
            html.Div(["Outliers:", dcc.Dropdown(id="out-rule", clearable=False, style={"width": "190px"}, value="none",
                                                options=[{"label": "None", "value": "none"},
                                                         {"label": "mean ± k·SD", "value": "sd"},
                                                         {"label": "median ± k·MAD", "value": "mad"},
                                                         {"label": "value > threshold", "value": "abs"}])],
                     className="ctl"),
            html.Div(["k:", dcc.Input(id="out-k", type="number", value=2, step=0.5, style={"width": "60px"})],
                     className="ctl"),
            html.Div(["threshold:", dcc.Input(id="out-thr", type="number", value=10, style={"width": "80px"})],
                     className="ctl"),
            dcc.Checklist(id="out-axes", options=[{"label": "on X", "value": "x"}, {"label": "on Y", "value": "y"}],
                          value=["x", "y"], inline=True),
            dcc.Checklist(id="out-exclude", options=[{"label": "Exclude outliers (recompute)", "value": "on"}],
                          value=[]),
        ], className="top", style={"border": "none", "padding": "4px 0"}),
        html.Div([
            html.Div([dcc.Loading(dcc.Graph(id="scatter")), html.Ul(id="scatter-stats")]),
            html.Div(id="info"),
        ], className="two"),
        html.Div([
            html.H4("Exclusions (applied to the correlation heatmap)", style={"margin": "10px 0 4px"}),
            html.Div("Update adds the outliers flagged on the selected axes: an X outlier is excluded from X's whole "
                     "measurement (mean + std, or all 3 PCs of a Markov variant) and from every correlation with it; "
                     "same for Y. Delete a row (×) to restore it.", style={"fontSize": "12px", "color": "#666"}),
            html.Div([
                html.Button("Update heatmap with flagged outliers", id="excl-add", n_clicks=0),
                html.Button("Save CSV", id="excl-save", n_clicks=0),
                html.Button("Clear all", id="excl-clear", n_clicks=0),
                html.Span(id="excl-status", style={"fontSize": "12px", "color": "#444"}),
            ], className="btn-row", style={"display": "flex", "gap": "6px", "alignItems": "center",
                                           "margin": "6px 0"}),
            dash_table.DataTable(
                id="excl-table", row_deletable=True, page_size=15,
                columns=[{"name": n, "id": i} for n, i in (
                    ("participant", "participant"), ("group", "group"), ("excluded from", "family"),
                    ("columns", "features"), ("flagged on", "flagged_on"), ("rule", "rule"), ("added", "added"))],
                style_cell={"fontSize": "12px", "textAlign": "left", "padding": "3px 6px",
                            "whiteSpace": "normal", "height": "auto"},
                style_header={"fontWeight": "bold"}),
        ]),
    ], className="panel"),
    html.Div([html.H3("Stability", style={"marginTop": 0}), html.Div(id="stability", className="two")],
             className="panel"),
    html.Div(id="meansd", className="panel", style={"display": "none"}),
    live_board_markov.layout(feature_options),
])


# ============================================================
# 5. CALLBACKS
# ============================================================
@app.callback(Output({"type": "feat-list", "domain": ALL}, "value"),
              Output({"type": "dom-all", "domain": ALL}, "value"),
              Input("preset-reduced", "n_clicks"), Input("preset-all", "n_clicks"), Input("preset-none", "n_clicks"),
              Input({"type": "dom-all", "domain": ALL}, "value"),
              State({"type": "feat-list", "domain": ALL}, "value"),
              State({"type": "dom-all", "domain": ALL}, "id"),
              prevent_initial_call=True)
def update_selection(_r, _a, _n, dom_all, current, ids):
    trig = ctx.triggered_id
    domains = [i["domain"] for i in ids]
    if trig in ("preset-reduced", "preset-all", "preset-none"):
        chosen = {"preset-reduced": DEFAULT_SELECTION, "preset-all": set(ALL_FEATURES), "preset-none": set()}[trig]
        values = [[f for f in FEATURES_BY_DOMAIN[d] if f in chosen] for d in domains]
        return values, [["all"] if set(FEATURES_BY_DOMAIN[d]) <= chosen and FEATURES_BY_DOMAIN[d] else []
                        for d in domains]
    i = domains.index(trig["domain"])
    current[i] = list(FEATURES_BY_DOMAIN[domains[i]]) if dom_all[i] else []
    return current, dom_all


@app.callback(Output("heatmap", "figure"), Output("order-store", "data"),
              Input("group", "value"), Input("method", "value"), Input("scheme", "value"),
              Input("cluster-abs", "value"), Input("target", "value"), Input("mask-ns", "value"),
              Input("cell-text", "value"), Input({"type": "feat-list", "domain": ALL}, "value"),
              Input("excl-store", "data"), Input("pc-names-store", "data"))
def update_heatmap(group, method, scheme, cluster_abs, target, mask_ns, cell_text, lists, excl_rows, names):
    selected = [f for f in ALL_FEATURES if any(f in l for l in lists)]
    df = subset_df(group)
    selected = [f for f in selected if df[f].dropna().nunique() > 1]
    if len(selected) < 2:
        return go.Figure().update_layout(title="Select at least two features"), []
    fig, order = heatmap_figure(group, method, selected, scheme, cluster_abs == "abs", target, bool(mask_ns),
                                cell_text, exclusion_key(excl_rows), names)
    return fig, order


@app.callback(Output("x-feat", "options"), Output("y-feat", "options"), Output("target", "options"),
              Output("covariate", "options"), Output({"type": "feat-list", "domain": ALL}, "options"),
              Input("pc-names-store", "data"), State({"type": "feat-list", "domain": ALL}, "id"))
def name_options(names, ids):
    """Markov PCs carry the names typed in the Markov section everywhere they're listed."""
    opts = [{"label": display_name(f, names), "value": f} for f in ALL_FEATURES]
    side = [[{"label": display_name(f, names), "value": f} for f in FEATURES_BY_DOMAIN[i["domain"]]] for i in ids]
    return opts, opts, opts, [{"label": "None", "value": ""}] + opts, side


@app.callback(Output("excl-store", "data"), Output("excl-table", "data"), Output("excl-status", "children"),
              Input("excl-add", "n_clicks"), Input("excl-clear", "n_clicks"), Input("excl-table", "data"),
              State("excl-store", "data"), State("group", "value"), State("x-feat", "value"), State("y-feat", "value"),
              State("out-rule", "value"), State("out-k", "value"), State("out-thr", "value"),
              State("out-axes", "value"))
def update_exclusions(_add, _clear, table_rows, store, group, x, y, rule, k, thr, axes_sel):
    trig = ctx.triggered_id
    store = store or []
    if trig is None:  # page load: show what local storage kept
        return store, store, ""
    if trig == "excl-clear":
        return [], [], "Cleared all exclusions."
    if trig == "excl-table":  # a row was deleted in the table
        rows = table_rows or []
        return rows, no_update, f"{len(store) - len(rows)} row(s) removed." if len(rows) < len(store) else ""
    flagged = flagged_by_axis(group, x, y, rule, k or 2, thr if thr is not None else 0, axes_sel or [])
    if not any(flagged.values()):
        return no_update, no_update, "No flagged outliers to add - pick an outlier rule first."
    rule_txt = {"sd": f"mean ± {k}·SD", "mad": f"median ± {k}·MAD", "abs": f"value > {thr}"}[rule]
    have = {(r["participant"], r["family"]) for r in store}
    added = []
    for axis, col in (("x", x), ("y", y)):
        fam = family_of(col)
        for p in flagged.get(axis, []):
            if (p, fam) in have:
                continue
            have.add((p, fam))
            grp = DF.loc[DF["participant"] == p, "group"]
            added.append({"participant": p, "group": grp.iloc[0] if len(grp) else "", "family": fam,
                          "features": ", ".join(display_name(c) for c in FAMILY_COLUMNS.get(fam, [col])),
                          "flagged_on": f"{display_name(col)} ({'X' if axis == 'x' else 'Y'} of {display_name(x)} "
                                        f"vs {display_name(y)}, group={group})",
                          "rule": rule_txt, "added": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")})
    rows = store + added
    return rows, rows, f"Added {len(added)} exclusion(s); heatmap updated." if added else "Already in the table."


@app.callback(Output("excl-status", "children", allow_duplicate=True), Input("excl-save", "n_clicks"),
              State("excl-store", "data"), prevent_initial_call=True)
def save_exclusions(_n, rows):
    if not rows:
        return "Nothing to save - the table is empty."
    # Long format (one row per participant x excluded column) so it can be fed straight back
    # in: set those cells to NaN before computing the heatmap outside the board.
    long = [{**{k: r[k] for k in ("participant", "group", "family", "flagged_on", "rule", "added")}, "column": c}
            for r in rows for c in FAMILY_COLUMNS.get(r["family"], [])]
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"heatmap_exclusions_{pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')}.csv")
    pd.DataFrame(long, columns=["participant", "group", "family", "column", "flagged_on", "rule", "added"]) \
        .to_csv(path, index=False)
    return f"Saved {len(rows)} exclusion(s) ({len(long)} participant × column rows) -> {path}"


@app.callback(Output("x-feat", "value"), Output("y-feat", "value"),
              Input("heatmap", "clickData"), State("order-store", "data"), prevent_initial_call=True)
def click_cell(click, order):
    if not click or not order:
        return no_update, no_update
    pt = click["points"][0]
    if pt.get("curveNumber") != 0:
        return no_update, no_update
    return order[int(pt["x"])], order[int(pt["y"])]


@app.callback(Output("scatter", "figure"), Output("scatter-stats", "children"), Output("info", "children"),
              Input("group", "value"), Input("method", "value"), Input("x-feat", "value"), Input("y-feat", "value"),
              Input("by-group", "value"), Input("out-rule", "value"), Input("out-k", "value"),
              Input("out-thr", "value"), Input("out-axes", "value"), Input("out-exclude", "value"),
              Input("covariate", "value"))
def update_scatter(group, method, x, y, by_group, rule, k, thr, axes_sel, exclude, covariate):
    fig, lines = scatter_figure(group, method, x, y, bool(by_group), rule, k or 2, thr if thr is not None else 0,
                                axes_sel or [], bool(exclude), covariate or None)
    return fig, [html.Li(l) for l in lines], html.Div([info_block(x), info_block(y)])


@app.callback(Output("stability", "children"), Input("x-feat", "value"), Input("y-feat", "value"))
def update_stability(x, y):
    return [stability_block(x)] + ([stability_block(y)] if y != x else [])


@app.callback(Output("meansd", "children"), Output("meansd", "style"),
              Input("meansd-btn", "n_clicks"), Input("group", "value"))
def toggle_meansd(n, group):
    if not n or n % 2 == 0:
        return [], {"display": "none"}
    return [html.H3("Mean vs SD", style={"marginTop": 0}), dcc.Graph(figure=mean_sd_grid(group))], {}


live_board_markov.register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=False, port=8050)

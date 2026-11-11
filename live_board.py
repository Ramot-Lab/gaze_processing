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
from dash import Dash, dcc, html, Input, Output, State, ALL, ctx, no_update
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests

import feature_pipeline as fp
from plot_mean_vs_sd import find_mean_sd_pairs
from live_board_cache import (PARTICIPANT_FEATURES_CSV, CACHE_DIR, SYMBOL_COUNT_STABILITY_ROOT,
                              reliability_cache_path)

GROUP_COLORS = {"HC": "#1f77b4", "MS": "#ff7f0e", "pwMS": "#ff7f0e"}  # repo convention (tab:blue/tab:orange)
OUTLIER_COLOR = "#d62728"
HEATMAP_SCALE = [[0, "blue"], [0.5, "white"], [1, "red"]]  # feature_pipeline's custom_blue_red
SCORE_KEY = "score"


# ============================================================
# 1. DATA
# ============================================================
def load_participants():
    df = pd.read_csv(PARTICIPANT_FEATURES_CSV)
    return df[~df["participant"].isin(fp.EXCLUDED_PARTICIPANTS)].reset_index(drop=True)


DF = load_participants()
ALL_FEATURES = fp._select_feature_columns(DF)


def subset_df(group):
    return DF[fp.CORRELATION_SUBSETS[group](DF)].copy()


def reduced_features(df):
    return fp._reduced_feature_columns(df, extra_exclude=fp.DOMAIN_HEATMAP_EXTRA_EXCLUDE)


def domain_of(col):
    return SCORE_KEY if col == "score" else fp.FEATURE_DOMAIN.get(fp._domain_base_name(col), "grey")


def source_of(col):
    return SCORE_KEY if col == "score" else fp.MEASURE_TYPE.get(fp._domain_base_name(col), "other")


DOMAIN_KEYS = fp.DOMAIN_ORDER + [SCORE_KEY]
DOMAIN_LABEL = {**fp.DOMAIN_LABELS, SCORE_KEY: "SDMT score"}
DOMAIN_COLOR = {**fp.DOMAIN_COLORS, SCORE_KEY: fp.SCORE_COLOR}
SOURCE_KEYS = fp.MEASURE_TYPE_ORDER + [SCORE_KEY]
SOURCE_LABEL = {**fp.MEASURE_TYPE_LABELS, SCORE_KEY: "SDMT score"}
SOURCE_COLOR = {**fp.MEASURE_TYPE_COLORS, SCORE_KEY: fp.SCORE_COLOR}

FEATURES_BY_DOMAIN = {d: [c for c in ALL_FEATURES if domain_of(c) == d] for d in DOMAIN_KEYS}
DEFAULT_SELECTION = set(reduced_features(DF))


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
    runs = sorted(d for d in glob.glob(os.path.join(SYMBOL_COUNT_STABILITY_ROOT, "*"))
                  if os.path.exists(os.path.join(d, "stability_summary.csv")))
    if not runs:
        return None
    return {"dir": runs[-1], "summary": pd.read_csv(os.path.join(runs[-1], "stability_summary.csv")),
            "iterations": pd.read_csv(os.path.join(runs[-1], "per_iteration_metrics.csv")),
            "splits": pd.read_csv(os.path.join(runs[-1], "per_iteration_splits.csv")),
            "trajectories": pd.read_csv(os.path.join(runs[-1], "trajectories_all_MS_HC.csv"))}


@lru_cache(maxsize=1)
def symbol_count_matrices():
    """Per-panel trial matrices for re-drawing a saved random split's two half
    trajectories - built by symbol_count_stability's own loader, so the curves are the
    same as that run's 02_example_split figures."""
    import symbol_count_stability as scs
    trials, panels, people = scs.load_data(scs.DEFAULT_TABLES_DIR)
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
def correlation_tables(group, method, features):
    df = subset_df(group)
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


def order_features(features, corr, scheme, cluster_abs, target):
    if scheme == "domain":
        return fp._order_features_by_domain(features)
    if scheme == "source":
        return fp._order_features_by_measure_type(features)
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
def heatmap_figure(group, method, features, scheme, cluster_abs, target, mask_ns, cell_text):
    df = subset_df(group)
    corr, pval, p_fdr, annot, n = correlation_tables(group, method, tuple(features))
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
                        np.array([[a for a in order] for _ in order]),
                        np.array([[b for _ in order] for b in order])])
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

    ticks = [f"<span style='color:{colors[key_fn(f)]}'>{'<b>' + f + '</b>' if f == 'score' else f}</span>"
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
              f"{' - non-significant (p_FDR >= .05) masked' if mask_ns else ''}<br>"
              "<sup>cell text: " + ("r, * p<.05 ** p<.01 *** p<.001 (uncorrected)" if cell_text == "r" else
                                    "n (pairwise deletion)") + " - click a cell for its scatter</sup>" + demo_sub(df),
        legend=dict(title=title, x=1.09, y=1, xanchor="left"), margin=dict(l=10, r=10, t=110, b=10))
    return fig, order


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

    fig.update_layout(title=f"{y} vs {x} - {group} ({'Pearson' if method == 'pearson' else 'Spearman'})",
                      xaxis_title=x, yaxis_title=y, height=560, template="plotly_white",
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
    fig.update_layout(height=580, template="plotly_white", margin=dict(t=110),
                      title="Symbol-count trajectory stability: example random split (median-r split per panel)"
                            + demo_sub(DF), legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"))
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
    fig.update_layout(height=580, template="plotly_white", margin=dict(t=110),
                      title="Symbol-count trajectory stability: MS vs HC (r and its random-split percentile)"
                            + demo_sub(DF), legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"))
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
            dcc.Checklist(id={"type": "feat-list", "domain": d}, options=[{"label": f, "value": f} for f in feats],
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
    ], className="panel"),
    html.Div([html.H3("Stability", style={"marginTop": 0}), html.Div(id="stability", className="two")],
             className="panel"),
    html.Div(id="meansd", className="panel", style={"display": "none"}),
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
              Input("cell-text", "value"), Input({"type": "feat-list", "domain": ALL}, "value"))
def update_heatmap(group, method, scheme, cluster_abs, target, mask_ns, cell_text, lists):
    selected = [f for f in ALL_FEATURES if any(f in l for l in lists)]
    df = subset_df(group)
    selected = [f for f in selected if df[f].dropna().nunique() > 1]
    if len(selected) < 2:
        return go.Figure().update_layout(title="Select at least two features"), []
    fig, order = heatmap_figure(group, method, selected, scheme, cluster_abs == "abs", target, bool(mask_ns),
                                cell_text)
    return fig, order


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


if __name__ == "__main__":
    app.run(debug=False, port=8050)

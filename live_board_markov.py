"""
Markov section of live_board.py (decision 2026-11-11): the mean-matrix PCA from
markov_analyzer.py, for two matrix types - plain transition probabilities and
normalized by number of transitions (score-normalized deliberately not shown).

Everything is read from markov_analyzer's saved outputs (live_board_cache.markov_dir):
  - participant mean-matrix PC scores      : panel_projection/mean_matrix_scores.csv
  - every participant x panel matrix projected onto the SAME mean-matrix PCs
    (run_panel_projection - nothing refit on the panels) : panel_projection/panel_projection_scores.csv
  - squared cosines (run_squared_cosines)  : squared_cosines/squared_cosines_per_participant.csv
  - PCA weights + explained variance       : weights/pca_weights.csv (saved by run_pca since
    2026-11-11). Runs older than that saved only PNG heatmaps, so for them the fit is
    reproduced from the saved separation/feature_table_Mean_Matrix.csv with _fit_pca's exact
    steps - verified 2026-11-11 to match the saved PC1-3 scores to < 1e-12.
  - within-participant reliability of the panel PC scores: the pipeline's panel-level
    split-half, cached by live_board_cache.build_markov_pc.
"""
import glob
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, ctx
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import feature_pipeline as fp
from live_board_cache import (MARKOV_VARIANTS, MARKOV_RUN_ROOT, markov_dir, markov_reliability_level,
                              reliability_cache_path, PARTICIPANT_FEATURES_CSV, available_markov_variants,
                              MARKOV_PC_NAMES_JSON)

AVAILABLE_VARIANTS = available_markov_variants()

META_COLS = ["Participant", "Group", "Panel", "SDMT_Score"]
SPECIAL_NAMES = {"Non-Symbol Fixations": ("ENTER", "ENTER"), "Skipped Search": ("ENTER", "EXIT"),
                 "Correct Search": ("EXIT", "EXIT"), "Uncertainty": ("EXIT", "ENTER")}  # markov_analyzer.special_names
STATES = ["ENTER"] + [str(i) for i in range(1, 10)] + ["EXIT"]  # _plot_pca_weights' grid
PCS = ["PC1", "PC2", "PC3"]
GROUP_COLORS = {"HC": "#1f77b4", "pwMS": "#ff7f0e", "MS": "#ff7f0e"}
PANEL_COLORS = dict(zip(["0", "i1", "l4", "a3", "a5", "l3"],
                        ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]))
DOMINANT_COLORS = {"PC1": "#4C72B0", "PC2": "#DD8452", "PC3": "#55A868"}
PARTICIPANT_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                       "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd",
                       "#e6550d", "#31a354", "#756bb1", "#636363"]


# ============================================================
# 1. DATA
# ============================================================
def _reproduce_fit(feature_table_csv):
    """markov_analyzer._build_and_save_feature_table + _fit_pca, step for step, on the
    saved mean-matrix feature table: drop all-NaN columns, impute NaNs with the column
    mean, drop (near-)constant columns, StandardScaler, PCA(min(n, d, 20))."""
    ft = pd.read_csv(feature_table_csv)
    feats = np.array([c for c in ft.columns if c not in META_COLS])
    X = ft[feats].to_numpy(float)
    keep = ~np.isnan(X).all(axis=0)
    X, feats = X[:, keep], feats[keep]
    col_means = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_means, idx[1])
    valid = X.var(axis=0) > 1e-9
    X, feats = X[:, valid], feats[valid]
    pca = PCA(n_components=min(len(X), X.shape[1], 20))
    pca.fit(StandardScaler().fit_transform(X))
    names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    weights = pd.DataFrame(pca.components_.T, index=feats, columns=names)
    return weights, pd.Series(pca.explained_variance_ratio_, index=names)


def _demographics_source():
    d = pd.read_csv(PARTICIPANT_FEATURES_CSV)
    return d.rename(columns={"participant": "Participant"})


@lru_cache(maxsize=4)
def load_variant(variant):
    base = markov_dir(variant)
    proj = os.path.join(base, "panel_projection")
    panels = pd.read_csv(os.path.join(proj, "panel_projection_scores.csv"))
    means = pd.read_csv(os.path.join(proj, "mean_matrix_scores.csv"))
    cos2 = pd.read_csv(os.path.join(base, "squared_cos", "squared_cosines_per_participant.csv"))
    panels["Panel"] = panels["Panel"].astype(str)

    weights_csv = os.path.join(base, "weights", "pca_weights.csv")
    if os.path.exists(weights_csv):
        weights = pd.read_csv(weights_csv, index_col="feature")
        ev = pd.read_csv(os.path.join(base, "weights", "pca_explained_variance.csv")).set_index("PC")[
            "explained_variance_ratio"]
        weights_source = weights_csv
    else:
        weights, ev = _reproduce_fit(os.path.join(base, "separation", "feature_table_Mean_Matrix.csv"))
        weights_source = "reproduced from separation/feature_table_Mean_Matrix.csv (verified vs saved scores)"

    feats = _demographics_source()
    extra = cos2[["Participant", "cos2_PC1_3_sum", "dominant_PC", "z_norm"]]
    means = means.merge(extra, on="Participant", how="left").merge(feats, on="Participant", how="left")
    panels = panels.merge(extra, on="Participant", how="left").merge(feats, on="Participant", how="left")
    return {"panels": panels, "means": means, "cos2": cos2, "weights": weights, "ev": ev,
            "weights_source": weights_source, "dir": base}


def load_pc_names():
    """{variant: {"PC1": name, ...}} - the names typed next to the weights heatmaps."""
    try:
        with open(MARKOV_PC_NAMES_JSON) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_pc_names(names):
    os.makedirs(os.path.dirname(MARKOV_PC_NAMES_JSON), exist_ok=True)
    with open(MARKOV_PC_NAMES_JSON, "w") as f:
        json.dump(names, f, indent=2)


def pc_column(variant, pc):
    """Column name of a variant's PC in the board's participant table / heatmap."""
    return f"markov_{variant}_{pc}"


def pc_display_name(variant, pc, names=None):
    names = load_pc_names() if names is None else names
    given = (names.get(variant) or {}).get(pc, "").strip()
    tag = f"{MARKOV_VARIANTS[variant][1]} {pc}"
    return f"{given} [{tag}]" if given else f"Markov {tag}"


def load_reliability(variant):
    path = reliability_cache_path(markov_reliability_level(variant))
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {rec["feature"]: rec for rec in json.load(f)["records"]}


def demo_text(df):
    cols = ["Participant", "Group"] + (["Gender"] if "Gender" in df else [])
    d = df[cols].drop_duplicates("Participant").rename(columns={"Group": "group"})
    return fp.demographics_text(d).replace("\n", " | ")


def subset(df, group):
    if group == "MS":
        return df[df["Group"].isin(["MS", "pwMS"])]
    if group == "HC":
        return df[df["Group"] == "HC"]
    return df


# ============================================================
# 2. FIGURES
# ============================================================
COLOR_OPTIONS = [{"label": "Group (MS / HC)", "value": "group"},
                 {"label": "SDMT score", "value": "score"},
                 {"label": "Panel (panel points)", "value": "panel"},
                 {"label": "cos² PC1–3 (how well PC1–3 describe the participant)", "value": "cos2_PC1_3_sum"},
                 {"label": "Dominant PC (largest cos²)", "value": "dominant_PC"},
                 {"label": "Atypicality ||z|| (distance from average matrix)", "value": "z_norm"}]


def _color_spec(df, color_by, feature):
    """('cat', {label: (mask, color)}) or ('num', values, title) for the chosen coloring."""
    if color_by == "group":
        return "cat", {("MS" if g == "pwMS" else g): (df["Group"] == g, GROUP_COLORS.get(g, "grey"))
                       for g in sorted(df["Group"].dropna().unique())}
    if color_by == "panel" and "Panel" in df:
        return "cat", {f"Panel {p}": (df["Panel"] == p, PANEL_COLORS.get(p, "grey"))
                       for p in PANEL_COLORS if (df["Panel"] == p).any()}
    if color_by == "dominant_PC":
        return "cat", {pc: (df["dominant_PC"] == pc, c) for pc, c in DOMINANT_COLORS.items()}
    col = {"score": "SDMT_Score"}.get(color_by, color_by)
    if color_by == "feature":
        col = feature
    if col not in df:
        return _color_spec(df, "group", None)
    return "num", df[col].astype(float), col


MUTED = "#d3d3d3"


def _participant_colors(participants, highlight=None):
    """One color per participant; with a highlight list, only those keep a color (and a
    hull) and everyone else is muted grey - 101 hulls at once are unreadable."""
    ordered = sorted(highlight) if highlight else sorted(participants)
    colors = {p: PARTICIPANT_PALETTE[i % len(PARTICIPANT_PALETTE)] for i, p in enumerate(ordered)}
    return {p: colors.get(p, MUTED) for p in participants}


def _hover(df):
    panel = df["Panel"] if "Panel" in df else pd.Series("mean matrix", index=df.index)
    return [f"<b>{p}</b> ({g}) · {pn}<br>SDMT {s:.0f}" if pd.notna(s) else f"<b>{p}</b> ({g}) · {pn}"
            for p, g, pn, s in zip(df["Participant"], df["Group"], panel, df["SDMT_Score"])]


def _add_points(fig, df, dims, color_by, feature, symbol, size, name_suffix, per_participant, pcolors,
                show_legend, row=None, col=None, three_d=False, outline="white", coloraxis="coloraxis"):
    if df.empty:
        return
    kw = dict(row=row, col=col) if not three_d else {}
    Trace = go.Scatter3d if three_d else go.Scatter
    coords = {"x": df[dims[0]], "y": df[dims[1]]}
    if three_d:
        coords["z"] = df[dims[2]]
    marker = dict(size=size if not three_d else size * 0.6, symbol=symbol, line=dict(color=outline, width=1))
    if per_participant:
        df = df.assign(_muted=[pcolors[p] == MUTED for p in df["Participant"]]).sort_values("_muted", ascending=False)
        c = {"x": df[dims[0]], "y": df[dims[1]]}
        if three_d:
            c["z"] = df[dims[2]]
        fig.add_trace(Trace(**c, mode="markers", name=f"participants{name_suffix}", showlegend=False,
                            marker={**marker, "color": [pcolors[p] for p in df["Participant"]]},
                            text=_hover(df), hovertemplate="%{text}<extra></extra>"), **kw)
        return
    kind, *spec = _color_spec(df, color_by, feature)
    if kind == "cat":
        for label, (mask, color) in spec[0].items():
            sub = df[mask]
            if sub.empty:
                continue
            c = {"x": sub[dims[0]], "y": sub[dims[1]]}
            if three_d:
                c["z"] = sub[dims[2]]
            fig.add_trace(Trace(**c, mode="markers", name=f"{label}{name_suffix}", legendgroup=f"{label}{name_suffix}",
                                showlegend=show_legend, marker={**marker, "color": color},
                                text=_hover(sub), hovertemplate="%{text}<extra></extra>"), **kw)
    else:
        values, title = spec
        fig.add_trace(Trace(**coords, mode="markers", name=f"{title}{name_suffix}", showlegend=False,
                            marker={**marker, "color": values, "coloraxis": coloraxis},
                            text=[f"{h}<br>{title} = {v:.3g}" for h, v in zip(_hover(df), values)],
                            hovertemplate="%{text}<extra></extra>"), **kw)
        fig.update_layout(**{coloraxis: dict(colorscale="Viridis", colorbar=dict(title=title, thickness=12))})


def _axis_label(pc, ev):
    return f"{pc} ({ev[pc] * 100:.1f}%)"


def pca_2d_figure(variant, group, color_by, feature, show_panels, show_means, per_participant, highlight=None):
    data = load_variant(variant)
    panels, means = subset(data["panels"], group), subset(data["means"], group)
    pairs = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07)
    pcolors = _participant_colors(means["Participant"], highlight) if per_participant else {}
    for i, dims in enumerate(pairs, start=1):
        if per_participant:
            for p, g in panels.groupby("Participant"):
                pts = g[list(dims)].to_numpy()
                if len(pts) < 3 or (highlight and p not in highlight):
                    continue
                try:
                    hull = ConvexHull(pts)
                except QhullError:
                    continue
                v = np.r_[hull.vertices, hull.vertices[:1]]
                fig.add_trace(go.Scatter(x=pts[v, 0], y=pts[v, 1], fill="toself", mode="lines",
                                         line=dict(color=pcolors.get(p, "grey"), width=0.6),
                                         fillcolor=pcolors.get(p, "grey"), opacity=0.15, hoverinfo="text",
                                         text=p, showlegend=False), row=1, col=i)
        if show_panels:
            _add_points(fig, panels, dims, color_by, feature, "circle", 7, " · panels", per_participant, pcolors,
                        show_legend=(i == 1), row=1, col=i)
        if show_means or not show_panels:
            _add_points(fig, means, dims, color_by if color_by != "panel" else "group", feature, "diamond",
                        11 if show_panels else 9, " · mean matrix" if show_panels else "", per_participant,
                        pcolors, show_legend=(i == 1), row=1, col=i, outline="black" if show_panels else "white")
        fig.update_xaxes(title_text=_axis_label(dims[0], data["ev"]), row=1, col=i)
        fig.update_yaxes(title_text=_axis_label(dims[1], data["ev"]), row=1, col=i)
    what = ("participant × panel matrices projected onto the mean-matrix PCs" if show_panels
            else "participant mean matrices")
    fig.update_layout(height=470, template="plotly_white", margin=dict(t=90),
                      title=f"{MARKOV_VARIANTS[variant][0]} · {what}"
                            f"{' + participant means (◆)' if show_panels and show_means else ''}"
                            f"<br><sup>{demo_text(means)}</sup>",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def pca_3d_figure(variant, group, color_by, feature, show_panels, show_means, per_participant, highlight=None):
    data = load_variant(variant)
    panels, means = subset(data["panels"], group), subset(data["means"], group)
    fig = go.Figure()
    pcolors = _participant_colors(means["Participant"], highlight) if per_participant else {}
    if per_participant:
        for p, g in panels.groupby("Participant"):
            if len(g) >= 4 and (not highlight or p in highlight):
                fig.add_trace(go.Mesh3d(x=g["PC1"], y=g["PC2"], z=g["PC3"], alphahull=0, opacity=0.08,
                                        color=pcolors.get(p, "grey"), hoverinfo="skip", showlegend=False))
    if show_panels:
        _add_points(fig, panels, PCS, color_by, feature, "circle", 7, " · panels", per_participant, pcolors,
                    show_legend=True, three_d=True)
    if show_means or not show_panels:
        _add_points(fig, means, PCS, color_by if color_by != "panel" else "group", feature, "diamond", 10,
                    " · mean matrix" if show_panels else "", per_participant, pcolors, show_legend=True,
                    three_d=True, outline="black")
    ev = data["ev"]
    fig.update_layout(height=620, template="plotly_white", margin=dict(t=70, l=0, r=0, b=0),
                      title=f"PC1 × PC2 × PC3 (drag to rotate) · {MARKOV_VARIANTS[variant][0]}"
                            f"<br><sup>{demo_text(means)}</sup>",
                      scene=dict(xaxis_title=_axis_label("PC1", ev), yaxis_title=_axis_label("PC2", ev),
                                 zaxis_title=_axis_label("PC3", ev)),
                      legend=dict(orientation="h", y=0))
    return fig


def weights_figure(variant, names=None):
    data = load_variant(variant)
    w = data["weights"]
    given = (names or {}).get(variant) or {}
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, subplot_titles=[
        (f"<b>{given[pc]}</b> · " if given.get(pc, "").strip() else "")
        + f"{pc} weights ({data['ev'][pc] * 100:.1f}% var.)" for pc in PCS])
    def cell(feat):
        if feat in SPECIAL_NAMES:
            return SPECIAL_NAMES[feat]
        return tuple(x.strip() for x in feat.strip("()").replace("'", "").split(","))

    cells = {feat: cell(feat) for feat in w.index}
    used = {s for uv in cells.values() for s in uv}
    states = [s for s in STATES if s in used]
    for i, pc in enumerate(PCS, start=1):
        grid = pd.DataFrame(np.nan, index=states, columns=states)
        for feat, val in w[pc].items():
            u, v = cells[feat]
            if u in states and v in states:
                grid.loc[u, v] = val
        m = np.nanmax(np.abs(grid.values))
        fig.add_trace(go.Heatmap(z=grid.values, x=states, y=states, zmin=-m, zmax=m, colorscale="RdBu_r",
                                 text=np.where(np.isnan(grid.values), "", np.round(grid.values, 2).astype(str)),
                                 texttemplate="%{text}", textfont=dict(size=8), xgap=1, ygap=1,
                                 showscale=(i == 3), colorbar=dict(title="weight", thickness=10),
                                 hovertemplate="from %{y} → to %{x}<br>weight %{z:.3f}<extra></extra>"),
                      row=1, col=i)
        fig.update_yaxes(autorange="reversed", title_text="from" if i == 1 else None, row=1, col=i)
        fig.update_xaxes(title_text="to", row=1, col=i)
    fig.update_layout(height=430, template="plotly_white", plot_bgcolor="black", margin=dict(t=90),
                      title=f"Mean-matrix PCA weights · {MARKOV_VARIANTS[variant][0]} (empty = constant/never seen)"
                            f"<br><sup>{demo_text(data['means'])} · source: {data['weights_source']}</sup>")
    return fig


def cos2_figure(variant, group):
    data = load_variant(variant)
    d = subset(data["cos2"], group).merge(data["means"][["Participant", "Gender"]], on="Participant", how="left")
    d = d.sort_values("SDMT_Score").reset_index(drop=True)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.92, 0.08], shared_xaxes=True, vertical_spacing=0.02)
    for pc in PCS:
        fig.add_trace(go.Bar(x=d["Participant"], y=d[f"cos2_{pc}"], name=f"{pc} ({data['ev'][pc] * 100:.1f}% var.)",
                             marker_color=DOMINANT_COLORS[pc],
                             hovertemplate="%{x}<br>" + pc + " cos² %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=d["Participant"], y=d["cos2_rest"], name="Rest (outside PC1–3)", marker_color="#D0D0D0",
                         hovertemplate="%{x}<br>rest %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=d["Participant"], y=np.ones(len(d)), showlegend=False, hoverinfo="text",
                         text=d["Group"], textposition="none",
                         marker_color=[GROUP_COLORS.get(g, "grey") for g in d["Group"]]), row=2, col=1)
    fig.update_yaxes(range=[0, 1], title_text="share of deviation (cos²)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text="group", row=2, col=1)
    fig.update_xaxes(tickvals=d["Participant"], ticktext=[f"{s:.0f}" for s in d["SDMT_Score"]], tickfont=dict(size=8),
                     title_text="participant (label = mean SDMT score, low → high)", row=2, col=1)
    fig.update_layout(barmode="stack", bargap=0.1, height=460, template="plotly_white", margin=dict(t=90),
                      title="Squared cosines: how well PC1–3 explain each participant's mean matrix "
                            "(Abdi & Williams, 2010) · group strip: blue HC, orange MS"
                            f"<br><sup>{demo_text(d)}</sup>", legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    return fig


def reliability_figure(variant):
    recs = load_reliability(variant)
    data = load_variant(variant)
    if not recs:
        return go.Figure().update_layout(title="No cached panel split-half reliability for this variant - "
                                               "run python live_board_cache.py", height=200)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, subplot_titles=[
        f"{pc}: mean r = {np.mean([v for v in recs[pc]['r'] if v is not None]):.3f}" for pc in PCS if pc in recs])
    for i, pc in enumerate([p for p in PCS if p in recs], start=1):
        r = np.array([v for v in recs[pc]["r"] if v is not None])
        lo, hi = np.percentile(r, [2.5, 97.5])
        fig.add_trace(go.Histogram(x=r, nbinsx=30, marker_color="#9aa5b1", showlegend=False), row=1, col=i)
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#9aa5b1", opacity=0.15, line_width=0, row=1, col=i)
        fig.add_vline(x=r.mean(), line_dash="dash", row=1, col=i)
        fig.update_xaxes(title_text="split-half r", row=1, col=i)
    rec = next(iter(recs.values()))
    who = data["means"][data["means"]["Participant"].isin(rec["participants"])]
    fig.update_layout(height=280, template="plotly_white", margin=dict(t=90),
                      title=f"Reliability of participants' panel PC scores (panel split-half, L={rec['l_size']}, "
                            f"{rec['repetitions']} reps; shaded = 95% interval)<br><sup>{demo_text(who)}</sup>")
    return fig


# ============================================================
# 3. LAYOUT + CALLBACKS
# ============================================================
def layout(feature_options):
    return html.Div([
        html.H3("Markov analysis · mean-matrix PCA", style={"marginTop": 0}),
        html.Div(f"Source: {MARKOV_RUN_ROOT}/analysis_results/markov · PCs fit on participants' mean matrices; "
                 "panels are projected onto those same PCs (nothing refit).", style={"fontSize": "12px", "color": "#666"}),
        html.Div([
            html.Div(["Matrix:", dcc.RadioItems(id="mk-variant", inline=True, value=AVAILABLE_VARIANTS[0],
                                                options=[{"label": MARKOV_VARIANTS[v][0], "value": v}
                                                         for v in AVAILABLE_VARIANTS],
                                                inputStyle={"marginRight": "4px", "marginLeft": "10px"})],
                     className="ctl"),
            html.Div(["Color by:", dcc.Dropdown(id="mk-color", options=COLOR_OPTIONS + [
                {"label": "A participant feature…", "value": "feature"}], value="group", clearable=False,
                                                style={"width": "330px"})], className="ctl"),
            html.Div(["Feature:", dcc.Dropdown(id="mk-feature", options=feature_options, value="Age_Years",
                                               clearable=False, style={"width": "240px"})],
                     className="ctl", id="mk-feature-wrap"),
        ], className="top", style={"border": "none", "padding": "6px 0"}),
        html.Div([
            dcc.Checklist(id="mk-toggles", inline=True, value=[], inputStyle={"marginRight": "4px", "marginLeft": "12px"},
                          options=[{"label": "Show all participant × panel points", "value": "panels"},
                                   {"label": "Overlay participant means (◆) on the panels", "value": "means"},
                                   {"label": "Color by participant + hull around their panels", "value": "per_participant"}]),
            html.Div(["Highlight participants (others greyed; empty = all colored):",
                      dcc.Dropdown(id="mk-highlight", multi=True, value=[], placeholder="pick participants…",
                                   options=sorted(load_variant(AVAILABLE_VARIANTS[0])["means"]["Participant"]),
                                   style={"width": "520px"})], className="ctl", id="mk-highlight-wrap"),
        ], style={"fontSize": "13px", "marginBottom": "4px"}),
        dcc.Loading(dcc.Graph(id="mk-2d")),
        dcc.Loading(dcc.Graph(id="mk-3d")),
        html.Div([html.B("Name this variant's PCs (saved per variant; shown in the correlation heatmap):")]
                 + [html.Div([f"{pc}:", dcc.Input(id=f"mk-name-{pc}", type="text", debounce=True, value="",
                                                   placeholder=f"name {pc}…", style={"width": "220px"})],
                             className="ctl") for pc in PCS]
                 + [html.Span(id="mk-name-status", style={"fontSize": "11px", "color": "#666"})],
                 className="top", style={"border": "none", "padding": "6px 0"}),
        dcc.Loading(dcc.Graph(id="mk-weights")),
        dcc.Loading(dcc.Graph(id="mk-cos2")),
        dcc.Loading(dcc.Graph(id="mk-reliability")),
    ], className="panel")


def register_callbacks(app):
    @app.callback(Output("mk-feature-wrap", "style"), Input("mk-color", "value"))
    def _toggle_feature(color_by):
        return None if color_by == "feature" else {"display": "none"}

    @app.callback(Output("mk-highlight-wrap", "style"), Input("mk-toggles", "value"))
    def _toggle_highlight(toggles):
        return None if "per_participant" in toggles else {"display": "none"}

    @app.callback(Output("mk-2d", "figure"), Output("mk-3d", "figure"),
                  Input("mk-variant", "value"), Input("group", "value"), Input("mk-color", "value"),
                  Input("mk-feature", "value"), Input("mk-toggles", "value"), Input("mk-highlight", "value"))
    def _scatter(variant, group, color_by, feature, toggles, highlight):
        args = (variant, group, color_by, feature, "panels" in toggles or "per_participant" in toggles,
                "means" in toggles, "per_participant" in toggles, highlight or None)
        return pca_2d_figure(*args), pca_3d_figure(*args)

    @app.callback(Output("mk-weights", "figure"), Output("mk-cos2", "figure"), Output("mk-reliability", "figure"),
                  Input("mk-variant", "value"), Input("group", "value"), Input("pc-names-store", "data"))
    def _static(variant, group, names):
        return weights_figure(variant, names), cos2_figure(variant, group), reliability_figure(variant)

    @app.callback(Output("mk-name-PC1", "value"), Output("mk-name-PC2", "value"), Output("mk-name-PC3", "value"),
                  Input("mk-variant", "value"))
    def _load_names(variant):
        given = load_pc_names().get(variant) or {}
        return [given.get(pc, "") for pc in PCS]

    @app.callback(Output("pc-names-store", "data"), Output("mk-name-status", "children"),
                  Input("mk-name-PC1", "value"), Input("mk-name-PC2", "value"), Input("mk-name-PC3", "value"),
                  State("mk-variant", "value"), prevent_initial_call=True)
    def _save_names(n1, n2, n3, variant):
        names = load_pc_names()
        new = {pc: (v or "").strip() for pc, v in zip(PCS, (n1, n2, n3))}
        if names.get(variant, {}) != new:
            names[variant] = new
            save_pc_names(names)
            return names, f"Saved names for {MARKOV_VARIANTS[variant][0]} -> {MARKOV_PC_NAMES_JSON}"
        return names, ""

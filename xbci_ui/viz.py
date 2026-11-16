"""Plotly figure builders (topomaps, signals, bars, heatmaps ...). No MNE needed."""
import math
import textwrap
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GRAY, RED, GREEN = "#1a56ff", "#f08a24", "#8a94a6", "#e5484d", "#2da44e"
CLASS_COLORS = {"feet": "#2da44e", "left_hand": "#1a56ff", "right_hand": "#e5484d", "tongue": "#a855f7"}
BAND_ORDER = ["theta", "mu", "beta_low", "beta_high", "gamma_low"]
BASE_LAYOUT = dict(font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1b2433"),
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", margin=dict(l=40, r=20, t=40, b=40))


def nice(label):
    return label.replace("_", " ")


def _legend_names(fig):
    names = []
    for t in fig.data:
        if getattr(t, "showlegend", None) is False or not getattr(t, "name", None) or t.type in ("heatmap", "indicator"):
            continue
        names.append(str(t.name))
    return list(dict.fromkeys(names))


def _layout(fig, title=None, height=None, **kw):
    """Single place that guarantees: title / legend / plot area never overlap, long titles wrap,
    tick labels never get clipped."""
    fig.update_layout(**{**BASE_LAYOUT, **kw})
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    m = {**BASE_LAYOUT["margin"], **(kw.get("margin") or {})}
    need = 12
    if title:
        wrapped = "<br>".join(textwrap.wrap(str(title), 58)) or str(title)
        lines = wrapped.count("<br>") + 1
        fig.update_layout(title=dict(text=wrapped, font=dict(size=13), x=0.01, xanchor="left", y=1.0, yanchor="top", yref="container", pad=dict(t=14, l=0)))
        need = 22 + 18 * lines
    names = [] if fig.layout.showlegend is False else _legend_names(fig)
    if len(names) >= 2:                            # legend rows sit between the title and the plot
        rows = max(1, math.ceil(sum(len(n) + 7 for n in names) / 75.0))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                                      font=dict(size=11), title=dict(text="")))
        need += 24 * rows + 4
    if title and any(getattr(a, "yref", None) == "paper" and getattr(a, "yanchor", None) == "bottom" for a in fig.layout.annotations):
        need += 26                                 # subplot titles live just above each panel
    fig.update_layout(margin={**m, "t": max(m["t"], need)})
    if height:
        fig.update_layout(height=height)
    return fig


BAND_COLORS = {"theta": "#8a94a6", "mu": "#1a56ff", "beta_low": "#2da44e", "beta_high": "#f08a24", "gamma_low": "#a855f7"}


def add_band_shading(fig, bands, fmax=None):
    """Frequency-band shading. Bands are named in the legend (no text printed on top of the data)."""
    for nm, (lo, hi) in bands.items():
        if fmax is not None and lo >= fmax:
            continue
        col = BAND_COLORS.get(nm, "#8a94a6")
        fig.add_vrect(x0=lo, x1=hi, fillcolor=col, opacity=0.07, line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{nice(nm)} {lo}-{hi} Hz",
                                 marker=dict(symbol="square", size=10, color=col, opacity=0.45)))


# ----------------------------------------------------------------------------- topomap
def _fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    r2 = sol[3] + c @ c
    R = np.sqrt(r2) if r2 > 0 else 0
    if not (0.06 < R < 0.13):
        return np.zeros(3)
    return c


@lru_cache(maxsize=8)
def _project(key):
    ch_names, pos = key
    P = np.array([pos[i] for i in range(len(ch_names))], dtype=float)
    c = _fit_sphere(P)
    V = P - c
    r = np.linalg.norm(V, axis=1)
    theta = np.arccos(np.clip(V[:, 2] / r, -1, 1))
    phi = np.arctan2(V[:, 1], V[:, 0])
    scale = 0.74 / theta.max()
    return np.c_[theta * np.cos(phi), theta * np.sin(phi)] * scale


def project_2d(ch_names, ch_pos):
    key = (tuple(ch_names), tuple(tuple(ch_pos[c]) for c in ch_names))
    return _project(key)


@lru_cache(maxsize=1)
def _grid(n=61):
    xs = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, xs)
    mask = XX ** 2 + YY ** 2 <= 1.0
    return xs, XX, YY, mask


def interpolate_map(xy, v, n=61):
    from scipy.interpolate import RBFInterpolator
    v = np.asarray(v, float)
    xs, XX, YY, mask = _grid(n)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.c_[np.cos(ang), np.sin(ang)] * 1.0
    d = np.linalg.norm(ring[:, None, :] - xy[None], axis=2)
    w = 1.0 / (d ** 2 + 1e-6)
    vr = (w * v[None]).sum(1) / w.sum(1)
    pts = np.vstack([xy, ring])
    vals = np.concatenate([v, vr])
    sd = vals.std() or 1.0
    rbf = RBFInterpolator(pts, vals / sd, kernel="thin_plate_spline", smoothing=1e-2)
    Z = rbf(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape) * sd
    Z = np.clip(Z, v.min(), v.max())
    Z[~mask] = np.nan
    return xs, Z


def _head_shapes(fig, row=None, col=None):
    kw = {} if row is None else dict(row=row, col=col)
    t = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#3b4558", width=2),
                             hoverinfo="skip", showlegend=False), **kw)
    fig.add_trace(go.Scatter(x=[-0.12, 0, 0.12], y=[0.99, 1.13, 0.99], mode="lines",
                             line=dict(color="#3b4558", width=2), hoverinfo="skip", showlegend=False), **kw)
    for s in (-1, 1):
        ey = np.array([-0.12, -0.05, 0.05, 0.12])
        ex = s * (1.0 + 0.06 * (1 - (ey / 0.12) ** 2))
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#3b4558", width=2),
                                 hoverinfo="skip", showlegend=False), **kw)


def _topo_traces(values, ch_names, ch_pos, signed, show_names, zlim, colorscale, colorbar_title="", showscale=True):
    xy = project_2d(ch_names, ch_pos)
    v = np.asarray(values, float)
    xs, Z = interpolate_map(xy, v)
    if zlim is None:
        zlim = (-np.abs(v).max() or -1, np.abs(v).max() or 1) if signed else (v.min(), v.max())
        if zlim[0] == zlim[1]:
            zlim = (zlim[0] - 1, zlim[1] + 1)
    heat = go.Heatmap(x=xs, y=xs, z=Z, zmin=zlim[0], zmax=zlim[1], colorscale=colorscale, showscale=showscale,
                      hoverinfo="skip", colorbar=dict(title=colorbar_title, thickness=10, len=0.8))
    sc = go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text" if show_names else "markers",
                    text=list(ch_names) if show_names else None, textposition="top center",
                    textfont=dict(size=9, color="#1b2433"),
                    marker=dict(size=5, color="#1b2433", line=dict(color="white", width=1)),
                    customdata=np.c_[list(ch_names), np.round(v, 4)],
                    hovertemplate="%{customdata[0]}: %{customdata[1]}<extra></extra>", showlegend=False)
    return heat, sc


def topomap(values, ch_names, ch_pos, title=None, signed=True, show_names=True, zlim=None, height=360,
            colorbar_title="", colorscale=None):
    cs = colorscale or ("RdBu_r" if signed else "YlGnBu")
    fig = go.Figure()
    heat, sc = _topo_traces(values, ch_names, ch_pos, signed, show_names, zlim, cs, colorbar_title)
    fig.add_trace(heat)
    _head_shapes(fig)
    fig.add_trace(sc)
    fig.update_xaxes(visible=False, range=[-1.25, 1.25], constrain="domain")
    fig.update_yaxes(visible=False, range=[-1.12, 1.25], scaleanchor="x", scaleratio=1)
    return _layout(fig, title, height, margin=dict(l=5, r=5, t=40, b=5), plot_bgcolor="rgba(0,0,0,0)")


def topomap_grid(values_list, titles, ch_names, ch_pos, ncols=4, signed=True, height_per_row=230, show_names=False,
                 colorscale=None, share_scale=False):
    n = len(values_list)
    nrows = int(np.ceil(n / ncols))
    wrap_w = {1: 60, 2: 34, 3: 26}.get(ncols, 18)
    titles = ["<br>".join(textwrap.wrap(str(t), wrap_w)) or str(t) for t in titles]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, horizontal_spacing=0.02, vertical_spacing=0.16)
    cs = colorscale or ("RdBu_r" if signed else "YlGnBu")
    glim = None
    if share_scale:
        m = max(np.abs(np.asarray(v)).max() for v in values_list) or 1.0
        glim = (-m, m) if signed else None
    for i, v in enumerate(values_list):
        r, c = i // ncols + 1, i % ncols + 1
        heat, sc = _topo_traces(v, ch_names, ch_pos, signed, show_names, glim, cs, showscale=False)
        fig.add_trace(heat, row=r, col=c)
        _head_shapes(fig, r, c)
        fig.add_trace(sc, row=r, col=c)
        fig.update_xaxes(visible=False, range=[-1.25, 1.25], row=r, col=c)
        fig.update_yaxes(visible=False, range=[-1.12, 1.25], scaleanchor=f"x{'' if i == 0 else i + 1}", scaleratio=1, row=r, col=c)
    fig.update_annotations(font_size=11)
    return _layout(fig, None, height_per_row * nrows + 20, margin=dict(l=5, r=5, t=46, b=5), plot_bgcolor="rgba(0,0,0,0)")


# ----------------------------------------------------------------------------- signals
def signal_figure(x, ch_names, fs, title="EEG signal (4-38 Hz, µV)", channels=None, height=430, highlight=None):
    idx = list(range(len(ch_names))) if channels is None else [ch_names.index(c) for c in channels]
    t = np.arange(x.shape[1]) / fs
    scale = np.median(np.std(x, axis=1)) * 3.0 or 1.0
    fig = go.Figure()
    for k, i in enumerate(idx):
        col = RED if (highlight and ch_names[i] in highlight) else "#2a3a56"
        fig.add_trace(go.Scatter(x=t, y=x[i] / scale - k, mode="lines", line=dict(width=0.9, color=col),
                                 name=ch_names[i], hovertemplate=f"{ch_names[i]}  %{{x:.2f}}s<extra></extra>",
                                 showlegend=False))
    fig.update_yaxes(tickmode="array", tickvals=[-k for k in range(len(idx))], ticktext=[ch_names[i] for i in idx],
                     showgrid=False, zeroline=False)
    fig.update_xaxes(title="time from cue onset (s)", showgrid=True, gridcolor="#eef1f6")
    return _layout(fig, title, height, margin=dict(l=55, r=15, t=40, b=40))


def spectrum_figure(signals, fs, fmax=45, title="Average power spectrum", bands=None):
    from scipy.signal import welch
    fig = go.Figure()
    cols = [BLUE, ORANGE, GREEN, RED]
    allp = []
    for k, (lab, x) in enumerate(signals):
        f, p = welch(x, fs=fs, nperseg=min(x.shape[1], 256), axis=-1)
        pm = p.mean(0)
        sel = f <= fmax
        allp.append(pm[sel][f[sel] >= 2])
        fig.add_trace(go.Scatter(x=f[sel], y=pm[sel], mode="lines", name=lab, line=dict(width=2, color=cols[k % 4])))
    if bands:
        add_band_shading(fig, bands, fmax)
    lo = math.floor(math.log10(max(min(a.min() for a in allp), 1e-9)))
    hi = math.ceil(math.log10(max(a.max() for a in allp)))
    fig.update_yaxes(type="log", title="power (µV²/Hz)", gridcolor="#eef1f6", range=[lo, max(hi, lo + 1)], dtick=1,
                     exponentformat="power", showexponent="all")
    fig.update_xaxes(title="frequency (Hz)", gridcolor="#eef1f6")
    return _layout(fig, title, 320)


def heat_signal(M, row_labels, fs, title, x_title="time (s)", colorscale="RdBu_r", signed=True, height=300):
    t = np.arange(M.shape[1]) / fs if fs else np.arange(M.shape[1])
    lim = np.abs(M).max() or 1.0
    fig = go.Figure(go.Heatmap(z=M, x=t, y=row_labels, colorscale=colorscale if signed else "YlGnBu",
                               zmin=-lim if signed else None, zmax=lim if signed else None,
                               colorbar=dict(thickness=10)))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height, margin=dict(l=70, r=10, t=36, b=40))


# ----------------------------------------------------------------------------- probabilities & importances
def prob_bar(probs, class_names, pred=None, true=None, title="Class probabilities", color=BLUE, height=250,
             compare=None, compare_name="before edit"):
    labels = [nice(c) + (" ✔ true" if true == c else "") for c in class_names]
    fig = go.Figure()
    if compare is not None:
        fig.add_trace(go.Bar(y=labels, x=compare, orientation="h", marker_color="#c3cada", name=compare_name,
                             text=[f"{p:.0%}" for p in compare], textposition="outside"))
    cols = [color if (pred is None or c == pred) else "#b8c2d6" for c in class_names]
    fig.add_trace(go.Bar(y=labels, x=probs, orientation="h", marker_color=cols, name="now" if compare is not None else "probability",
                         text=[f"{p:.0%}" for p in probs], textposition="outside"))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", gridcolor="#eef1f6")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group", showlegend=compare is not None, legend=dict(orientation="h", y=1.18))
    return _layout(fig, title, height, margin=dict(l=120, r=40, t=40, b=30))


def bars(names, values, title, color=BLUE, signed=False, height=280, ytitle=""):
    cols = [(GREEN if v >= 0 else RED) for v in values] if signed else color
    fig = go.Figure(go.Bar(x=[nice(n) for n in names], y=values, marker_color=cols))
    fig.update_yaxes(title=ytitle, gridcolor="#eef1f6")
    return _layout(fig, title, height)


def grouped_bars(names, series: dict, title, height=300, ytitle="", colors=None, yrange=None):
    fig = go.Figure()
    for k, (lab, vals) in enumerate(series.items()):
        c = (colors or {}).get(lab, [BLUE, ORANGE, GREEN, RED][k % 4])
        fig.add_trace(go.Bar(x=[nice(str(n)) for n in names], y=vals, name=lab, marker_color=c))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15))
    fig.update_yaxes(title=ytitle, gridcolor="#eef1f6", range=yrange)
    return _layout(fig, title, height)


def time_curves(curves: dict, fs, T, title, ytitle="relative importance", height=280, smooth=0, colors=None, vspan=None):
    t = np.arange(T) / fs
    fig = go.Figure()
    for k, (lab, y) in enumerate(curves.items()):
        y = np.asarray(y, float)
        if smooth and smooth > 1:
            y = np.convolve(y, np.ones(smooth) / smooth, mode="same")
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=lab, line=dict(width=2, color=(colors or {}).get(lab, [BLUE, ORANGE][k % 2]))))
    if vspan:
        fig.add_vrect(x0=vspan[0], x1=vspan[1], fillcolor="#2da44e", opacity=0.08, line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="expected MI window",
                                 marker=dict(symbol="square", size=11, color="rgba(45,164,78,0.35)")))
    fig.update_xaxes(title="time from cue onset (s)", gridcolor="#eef1f6")
    fig.update_yaxes(title=ytitle, gridcolor="#eef1f6")
    fig.update_layout(legend=dict(orientation="h", y=1.15))
    return _layout(fig, title, height)


def tf_heat(tf, band_names, fs, T, title, height=300, signed=True):
    nb, nw = tf.shape
    edges = np.linspace(0, T, nw + 1) / fs
    centers = (edges[:-1] + edges[1:]) / 2
    lim = np.abs(tf).max() or 1.0
    fig = go.Figure(go.Heatmap(z=tf, x=centers, y=[nice(b) for b in band_names], colorscale="RdBu_r" if signed else "Magma",
                               zmin=-lim if signed else None, zmax=lim if signed else None, colorbar=dict(thickness=10),
                               hovertemplate="t=%{x:.2f}s  %{y}: %{z:.3f}<extra></extra>"))
    fig.update_xaxes(title="time from cue onset (s)")
    return _layout(fig, title, height)


def confusion_fig(cm, labels, normalize=True, title="Confusion matrix", height=340):
    cm = np.asarray(cm, float)
    z = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    txt = [[f"{z[i, j]:.0%}<br>({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}" for j in range(cm.shape[1])]
           for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(z=z, x=[nice(l) for l in labels], y=[nice(l) for l in labels], colorscale="Blues",
                               text=txt, texttemplate="%{text}", showscale=False, zmin=0, zmax=1 if normalize else None))
    fig.update_xaxes(title="predicted", side="bottom")
    fig.update_yaxes(title="true", autorange="reversed")
    return _layout(fig, title, height, margin=dict(l=80, r=10, t=40, b=60))


def contrib_bar(names, values, title, height=320, topn=12):
    values = np.asarray(values, float)
    order = np.argsort(-np.abs(values))[:topn][::-1]
    fig = go.Figure(go.Bar(y=[names[i] for i in order], x=values[order], orientation="h",
                           marker_color=[GREEN if values[i] >= 0 else RED for i in order],
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="contribution to the decision score", gridcolor="#eef1f6", zeroline=True, zerolinecolor="#3b4558")
    return _layout(fig, title, height, margin=dict(l=130, r=20, t=40, b=40))


def box_by_group(df, x, y, title, color=None, height=320, ytitle=""):
    fig = go.Figure()
    for g, d in df.groupby(x):
        fig.add_trace(go.Box(y=d[y], name=nice(str(g)), boxpoints="outliers", marker_color=CLASS_COLORS.get(str(g), BLUE)))
    fig.update_yaxes(title=ytitle, gridcolor="#eef1f6", zeroline=True, zerolinecolor="#3b4558")
    fig.update_layout(showlegend=False)
    return _layout(fig, title, height)


def line_multi(x, series: dict, title, xtitle="", ytitle="", height=300, colors=None, yrange=None):
    fig = go.Figure()
    for k, (lab, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=lab,
                                 line=dict(width=2, color=(colors or {}).get(lab, [BLUE, ORANGE][k % 2]))))
    fig.update_xaxes(title=xtitle, gridcolor="#eef1f6")
    fig.update_yaxes(title=ytitle, gridcolor="#eef1f6", range=yrange)
    fig.update_layout(legend=dict(orientation="h", y=1.15))
    return _layout(fig, title, height)


def gauge(value, title, lo=-1, hi=1, height=215):
    v = 0 if value is None or (isinstance(value, float) and np.isnan(value)) else value
    title = str(title).replace("\n", "<br>")
    fig = go.Figure(go.Indicator(mode="gauge+number", value=v, domain=dict(x=[0.05, 0.95], y=[0.0, 0.72]), number=dict(valueformat=".2f", font=dict(size=30)),
                                 gauge=dict(axis=dict(range=[lo, hi], tickfont=dict(size=10)), bar=dict(color=BLUE),
                                            steps=[dict(range=[lo, lo + (hi - lo) * 0.5], color="#f3f5f9"),
                                                   dict(range=[lo + (hi - lo) * 0.5, hi], color="#e6eeff")])))
    fig.add_annotation(text=title, x=0.5, y=1.0, xref="paper", yref="paper", yanchor="top", showarrow=False,
                       font=dict(size=12, color="#1b2433"))
    return _layout(fig, None, height, margin=dict(l=25, r=25, t=8, b=8))


def electrode_map(ch_names, ch_pos, highlight=(), title=None, height=380):
    xy = project_2d(ch_names, ch_pos)
    fig = go.Figure()
    _head_shapes(fig)
    hl = [c in highlight for c in ch_names]
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=list(ch_names), textposition="top center", textfont=dict(size=10),
                             marker=dict(size=[16 if h else 11 for h in hl], color=[RED if h else "#2a3a56" for h in hl], line=dict(color="white", width=1.5)),
                             hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.update_xaxes(visible=False, range=[-1.25, 1.25], constrain="domain")
    fig.update_yaxes(visible=False, range=[-1.12, 1.25], scaleanchor="x", scaleratio=1)
    return _layout(fig, title, height, margin=dict(l=5, r=5, t=40, b=5), plot_bgcolor="rgba(0,0,0,0)")

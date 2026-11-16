"""Renders the step-by-step 'why did the model predict this?' story for ONE model."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import analysis, signals, store, viz
from .theme import arch_flow, callout, chip, kpi_row, plot, step_header, table

CLASS_HINT = {
    "left_hand": "imagining the **left hand** — expected to change brain rhythms mostly over the **right** motor cortex (around C4), because each hemisphere controls the opposite side of the body.",
    "right_hand": "imagining the **right hand** — expected to change brain rhythms mostly over the **left** motor cortex (around C3).",
    "feet": "imagining the **feet** — the foot area lies deep in the mid-line of the motor cortex, so changes are expected near **Cz**.",
    "tongue": "imagining the **tongue** — a small, less clearly lateralised representation, usually the hardest class to separate.",
}
LEVELS = ["Beginner", "Intermediate", "Expert"]


def lv(level, b, i=None, e=None):
    """Return text by audience level (falls back to the previous level)."""
    i = b if i is None else i
    e = i if e is None else e
    return {"Beginner": b, "Intermediate": i, "Expert": e}[level]


def confidence_label(p):
    s = np.sort(p)[::-1]
    margin = s[0] - s[1]
    if s[0] >= 0.8:
        return "high", margin
    if s[0] >= 0.5:
        return "moderate", margin
    return "low", margin


def prediction_card(a, class_names, true_label, key):
    pred = class_names[a["pred"]]
    conf, margin = confidence_label(a["probs"])
    col = store.MODEL_COLORS[a["model"]]
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        verdict = ""
        if true_label:
            verdict = chip("matches the true label ✔", "ok") if true_label == pred else chip(f"true label: {true_label.replace('_', ' ')} ✘", "")
        st.markdown(
            f'<div class="xb-card"><div class="small-muted">{store.MODEL_LABELS[a["model"]]}</div>'
            f'<div style="font-size:2rem;font-weight:800;color:{col};line-height:1.1">{pred.replace("_", " ")}</div>'
            f'<div style="margin:.25rem 0 .6rem 0">probability <b>{a["probs"][a["pred"]]:.0%}</b> · confidence <b>{conf}</b> '
            f'(margin over runner-up {margin:.0%})</div>{verdict} {chip(a["checkpoint_id"], "gray")}</div>', unsafe_allow_html=True)
    with c2:
        plot(viz.prob_bar(a["probs"], class_names, pred, true_label, color=col, height=230), key=f"{key}_prob")
    return conf


# ----------------------------------------------------------------------------- CSP + LDA internals
def _csp_internals(ctx):
    a, bd, ref, level, key = ctx["a"], ctx["a"]["breakdown"], ctx["ref"], ctx["level"], ctx["key"]
    cn = ctx["class_names"]
    ch, pos = ref["channel_names_in_order"], ref["channel_positions_3d_m"]
    fs = ref["sampling_frequency_hz"]
    p, q = a["pred"], int(np.argsort(-a["probs"])[1])
    K = bd["features"].size
    stats = analysis.csp_class_feature_stats(ctx["root"], ctx["sid"])
    diff = bd["contrib"][p] - bd["contrib"][q]
    top = np.argsort(-np.abs(diff))[:3]

    step_header("3a", "Spatial filters → 'virtual channels'",
                "CSP learned 8 weighted combinations of the 22 electrodes. Each one listens to a particular scalp pattern.")
    st.markdown(lv(level,
        "Think of each virtual channel as a **tuned microphone array**: instead of listening to one electrode, it mixes all 22 with learned weights so that one brain pattern is loud and the others are quiet.",
        "Each row of the CSP matrix **W** (8×22) is a spatial filter. Multiplying W by the EEG gives 8 *virtual channels* whose variance differs most between classes.",
        "$\\mathbf{s}(t)=\\mathbf{W}\\mathbf{x}(t),\\;\\mathbf{W}\\in\\mathbb{R}^{8\\times22}$. Filters maximise $\\frac{\\mathbf w^\\top\\Sigma_{c}\\mathbf w}{\\mathbf w^\\top(\\Sigma_{c}+\\Sigma_{\\bar c})\\mathbf w}$ "
        "(multi-class: approximate joint diagonalisation). Topographies below are **patterns** $\\mathbf A=\\mathrm{pinv}(\\mathbf W)$ (interpretable; Haufe et al. 2014), not raw filter weights."))
    names = [f"CSP {k + 1}" + (" ★" if k in top else "") for k in range(K)]
    plot(viz.topomap_grid(list(bd["patterns"]), names, ch, pos, ncols=4, show_names=False, height_per_row=210), key=f"{key}_csp_pat")
    st.caption("★ = the three components that matter most for this particular decision. Red/blue = sign of the electrode's weight in the pattern.")
    s = bd["virtual_channels"]
    figv = viz.heat_signal(s, [f"CSP {k + 1}" for k in range(K)], fs, "Virtual-channel signals for this trial (µV)", height=260)
    plot(figv, key=f"{key}_csp_vc")

    step_header("3b", "Band-power features",
                "For every virtual channel the model keeps ONE number: the log of its average power during the epoch.")
    st.latex(r"f_k=\log\Big(\tfrac{1}{T}\sum_{t=1}^{T} s_k(t)^2\Big)")
    fig = go.Figure()
    for c in range(len(cn)):
        fig.add_trace(go.Bar(x=[f"CSP {k + 1}" for k in range(K)], y=stats["mean"][c], name=cn[c].replace("_", " "),
                             error_y=dict(type="data", array=stats["std"][c], visible=True, thickness=1),
                             marker_color=viz.CLASS_COLORS[cn[c]], opacity=0.55 if c != p else 0.95))
    fig.add_trace(go.Scatter(x=[f"CSP {k + 1}" for k in range(K)], y=bd["features"], mode="markers", name="THIS trial",
                             marker=dict(symbol="diamond", size=13, color="black", line=dict(color="white", width=1.5))))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15))
    fig.update_yaxes(title="log band power f_k", gridcolor="#eef1f6")
    plot(viz._layout(fig, "This trial (◆) versus the typical value of each class (training trials, mean ± sd)", 340), key=f"{key}_csp_feat")
    st.markdown(lv(level,
        "Each class has a typical 'power fingerprint'. The black diamonds show where **this** trial sits. The model prefers the class whose bars the diamonds resemble most, giving extra weight to the components that separate classes best.",
        "If a diamond sits near the predicted class's bar for a component with a large LDA weight, that component pushes the decision toward that class.",
        "LDA assumes Gaussian classes with a shared covariance, so the log-posterior is linear in $\\mathbf f$; components with large $|a_{c,k}-a_{c',k}|$ and a diamond far from the competing class mean dominate."))

    step_header("3c", "LDA scoring → the vote",
                f"The score of every class is a weighted sum of the 8 features. **{cn[p].replace('_', ' ')}** won against runner-up **{cn[q].replace('_', ' ')}**.")
    st.latex(r"z_c=\sum_{k=1}^{8}a_{c,k}\,f_k+b_c,\qquad P(c)=\frac{e^{z_c}}{\sum_j e^{z_j}}")
    names_f = [f"CSP {k + 1}" for k in range(K)] + ["intercept"]
    vals = np.append(diff, bd["intercept"][p] - bd["intercept"][q])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        plot(viz.contrib_bar(names_f, vals, f"Why '{cn[p].replace('_', ' ')}' beats '{cn[q].replace('_', ' ')}': score difference per feature", topn=9), key=f"{key}_csp_contrib")
    with c2:
        rows = [{"class": cn[c].replace("_", " "), "score z": round(float(bd["scores"][c]), 2), "probability": f"{a['probs'][c]:.1%}"} for c in range(len(cn))]
        table(pd.DataFrame(rows))
        terms = " + ".join(f"({bd['coef'][p, k]:+.2f})({bd['features'][k]:+.2f})" for k in top)
        st.markdown(f"**Worked example** (3 biggest terms for *{cn[p].replace('_', ' ')}*):")
        st.code(f"z = {terms} + … + ({bd['intercept'][p]:+.2f})\n  = {bd['scores'][p]:+.2f}", language="text")
        st.caption(f"Recomputed by hand vs the saved model: max error {bd['max_err_vs_model']:.1e} → the explanation is exact for this model.")
    with st.expander("Show the full weight table (LDA coefficients a[c,k] and intercepts)"):
        df = pd.DataFrame(bd["coef"], index=[c.replace("_", " ") for c in cn], columns=[f"CSP {k + 1}" for k in range(K)])
        df["intercept"] = bd["intercept"]
        table(df.round(3).reset_index().rename(columns={"index": "class"}))


# ----------------------------------------------------------------------------- EEGNet internals
def _layer_journey(acts, fs, cn, logits, key):
    A = lambda n: np.asarray(acts[n])
    panels = [
        ("1 · temporal filters", np.abs(A("conv_temporal")).mean(1), "filter", fs),
        ("2 · spatial filters", A("act1")[:, 0, :], "filter", fs),
        ("3 · pooling ÷4", A("pool1")[:, 0, :], "filter", fs / 4),
        ("4 · separable conv", A("act2")[:, 0, :], "map", fs / 4),
        ("5 · pooling ÷8 (features)", A("pool2")[:, 0, :], "map", fs / 32),
    ]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[p[0] for p in panels] + ["6 · class scores"],
                        horizontal_spacing=0.07, vertical_spacing=0.16)
    for i, (t, M, lab, f) in enumerate(panels):
        r, c = i // 3 + 1, i % 3 + 1
        fig.add_trace(go.Heatmap(z=M, x=np.arange(M.shape[1]) / f, colorscale="RdBu_r", zmid=0, showscale=False,
                                 hovertemplate=f"{lab} %{{y}}, t=%{{x:.2f}}s: %{{z:.2f}}<extra></extra>"), row=r, col=c)
        fig.update_yaxes(autorange="reversed", title=lab, row=r, col=c)
    fig.add_trace(go.Bar(x=[c.replace("_", " ") for c in cn], y=logits, marker_color=[viz.GREEN if v == logits.max() else "#b8c2d6" for v in logits]),
                  row=2, col=3)
    fig.update_annotations(font_size=11)
    fig.update_layout(showlegend=False)
    plot(viz._layout(fig, None, 520, margin=dict(l=40, r=10, t=40, b=30)), key=f"{key}_journey")


def _eegnet_internals(ctx):
    a, bd, ref, level, key = ctx["a"], ctx["a"]["breakdown"], ctx["ref"], ctx["level"], ctx["key"]
    cn, root, sid = ctx["class_names"], ctx["root"], ctx["sid"]
    ch, pos, fs = ref["channel_names_in_order"], ref["channel_positions_3d_m"], ref["sampling_frequency_hz"]
    p, q = a["pred"], int(np.argsort(-a["probs"])[1])
    F2, nb, D = bd["F2"], bd["n_bins"], bd["D"]
    lw = store.get_layer_weights(root, sid)
    peaks = analysis.temporal_peak_freqs(root, sid)

    step_header("3a", "Temporal filters → 'which rhythm?'",
                f"The first layer slides {lw['derived__temporal_kernels'].shape[0]} small learned filters along time. Each one acts like a band-pass filter.")
    st.markdown(lv(level,
        "Brain activity is a mixture of rhythms (slow and fast waves). The first layer learns **8 detectors**, each tuned to a different rhythm, like the tone knobs on an equaliser.",
        "Layer `conv_temporal` has 8 kernels of 64 samples (0.5 s at 128 Hz). Their frequency responses (below) show which rhythms each filter passes — typically around the mu (8–13 Hz) and beta (13–30 Hz) bands for motor imagery.",
        "$h_f(t)=\\sum_{\\tau}k_f(\\tau)\\,x(t-\\tau)$ applied per electrode; BN follows. The magnitude response $|\\mathcal F\\{k_f\\}|$ identifies the passband; peak frequencies are annotated."))
    fr, resp = lw["derived__freqs_hz"], lw["derived__temporal_freq_response"]
    sel = fr <= 45
    fig = go.Figure()
    for f in range(resp.shape[0]):
        fig.add_trace(go.Scatter(x=fr[sel], y=resp[f][sel], mode="lines", name=f"filter {f + 1}", showlegend=False))
    viz.add_band_shading(fig, signals.FREQ_BANDS, 45)
    fig.update_xaxes(title="frequency (Hz)", gridcolor="#eef1f6")
    fig.update_yaxes(title="gain", gridcolor="#eef1f6")
    plot(viz._layout(fig, "Frequency response of the 8 temporal filters (one line each)", 320), key=f"{key}_tfr")

    step_header("3b", "Spatial filters → 'where on the scalp?'",
                "The second layer mixes the 22 electrodes with 16 learned spatial filters (2 per temporal filter).")
    pats = analysis.eegnet_spatial_patterns(root, sid)
    titles = [f"#{i + 1} · temporal filter {i // D + 1} (~{peaks[i // D]:.0f} Hz)" for i in range(pats.shape[0])]
    mode = st.radio("Show", ["Patterns (recommended: interpretable scalp topography)", "Raw filter weights"], horizontal=True, key=f"{key}_pat_mode") if level != "Beginner" else "Patterns"
    vals = list(pats) if mode.startswith("Pat") else list(lw["derived__spatial_filters"])
    plot(viz.topomap_grid(vals, titles, ch, pos, ncols=4, height_per_row=205), key=f"{key}_dw")
    st.markdown(lv(level,
        "Each map shows which electrodes a spatial filter 'listens to' (red = positive weight, blue = negative). A bipolar red/blue pair across the head means the filter compares left vs right activity.",
        "Raw weights are *filters* (how the network combines electrodes); *patterns* (Haufe et al. 2014: $\\mathbf a=\\Sigma\\mathbf w/(\\mathbf w^\\top\\Sigma\\mathbf w)$, with Σ estimated from training trials filtered by the matching temporal kernel) show where the extracted source lives on the scalp.",
        "Depthwise conv: $y_{i}(t)=\\sum_c w_{i,c}\\,h_{\\lfloor i/D\\rfloor,c}(t)$ with max-norm ≤ 1. Patterns use $\\Sigma_f=\\mathrm{cov}(k_f * x)$ from the saved baseline pool (32 training trials)."))

    step_header("3c", "Layer-by-layer: how the signal is transformed",
                "Follow this particular trial through the network — every panel is a real intermediate result.")
    _layer_journey(a["acts"], fs, cn, a["logits"], key)

    step_header("3d", "Final decision → which learned feature voted for what",
                f"The last layer is a plain weighted sum over {F2 * nb} features ({F2} feature maps × {nb} time bins of {4 / nb:.2f} s).")
    st.latex(r"z_c=\sum_{j=1}^{256}W_{c,j}\,h_j+b_c,\qquad P(c)=\mathrm{softmax}(z)_c")
    diff = bd["contrib"][p] - bd["contrib"][q]                         # (F2, nb)
    c1, c2 = st.columns([1.25, 1])
    with c1:
        fig = go.Figure(go.Heatmap(z=diff, x=[f"{i * 4 / nb:.2f}-{(i + 1) * 4 / nb:.2f}s" for i in range(nb)],
                                   y=[f"map {m + 1}" for m in range(F2)], colorscale="RdBu_r", zmid=0,
                                   hovertemplate="%{y} %{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=10, title="Δz")))
        fig.update_yaxes(autorange="reversed")
        plot(viz._layout(fig, f"Evidence for '{cn[p].replace('_', ' ')}' over '{cn[q].replace('_', ' ')}' (green/red = for/against)".replace("green/red", "red/blue"), 420), key=f"{key}_eeg_heat")
    with c2:
        per_map = diff.sum(1)
        plot(viz.contrib_bar([f"map {m + 1}" for m in range(F2)], per_map, "Net vote of each feature map", topn=10, height=420), key=f"{key}_eeg_maps")
    st.caption(f"Sum of all 256 terms + bias = class score. Recomputed by hand vs the saved model: max error {bd['max_err_vs_model']:.1e}.")
    m_star = int(np.argmax(np.abs(diff.sum(1))))
    w_in = bd["pointwise"][m_star]                                     # (16,) weights of depthwise channels
    top_in = np.argsort(-np.abs(w_in))[:3]
    st.markdown(f"##### Chain of evidence for the most decisive feature map (**map {m_star + 1}**)")
    st.markdown(lv(level,
        "This feature map is built mostly from the spatial filters shown below. Following the arrows backwards: *decision ← feature map ← spatial filter (where) ← temporal filter (which rhythm).*",
        "Map m is a linear mix (pointwise conv) of the 16 depthwise channels after a temporal convolution; the three channels with the largest |weight| are shown with their dominant rhythm.",
        "$\\text{map}_m(t)=\\sum_i P_{m,i}\\,\\text{ELU}(\\text{BN}(\\text{dw}_i))$ after the separable depthwise-temporal conv (kernel 16); BN scales are ignored in this heuristic ranking."))
    cols = st.columns(3)
    for col, i in zip(cols, top_in):
        with col:
            plot(viz.topomap(pats[i], ch, pos, f"spatial pattern #{i + 1} · weight {w_in[i]:+.2f}", show_names=False, height=250),
                 key=f"{key}_chain_{i}")
            st.caption(f"temporal filter {i // D + 1} passes ≈ **{peaks[i // D]:.0f} Hz**")


# ----------------------------------------------------------------------------- attribution
def attribution_section(ctx):
    a, ref, level, key = ctx["a"], ctx["ref"], ctx["level"], ctx["key"]
    cn = ctx["class_names"]
    e = a["expl"]
    ch, pos, fs = ref["channel_names_in_order"], ref["channel_positions_3d_m"], ref["sampling_frequency_hz"]
    T = e["time_importance"].size
    tgt = cn[a["target"]].replace("_", " ")
    step_header(4, "Where was the evidence? (channels · time · frequency)",
                f"We asked: *which parts of the input pushed the score of '{tgt}' up or down?* Method: Integrated Gradients + frequency occlusion.")
    st.markdown(lv(level,
        "We take the model's decision and trace it back to the input — like highlighting the words in a sentence that mattered. **Red** areas pushed *towards* the decision, **blue** areas pushed *against* it.",
        "**Integrated Gradients** (IG) distributes the score between the input and a reference signal over every electrode and time point. **Band importance** removes one frequency band at a time and measures how much the class score drops.",
        "$\\mathrm{IG}_i(x)=(x_i-x'_i)\\int_0^1\\partial_i F(x'+\\alpha(x-x'))d\\alpha$ with 3 baselines $x'$ sampled from training trials (expected-gradients style), 32 Riemann steps; completeness error "
        f"{abs(float(e['ig_completeness_delta'])):.2e}. Band/TF maps: Δ target-logit when the band (in a time window) is subtracted via zero-phase Butterworth band-pass."))
    c1, c2 = st.columns([1, 1.25])
    with c1:
        plot(viz.topomap(e["channel_signed"], ch, pos, "Channel evidence (signed IG, summed over time)", height=380, colorbar_title="IG"),
             key=f"{key}_attr_topo")
    with c2:
        plot(viz.time_curves({"|IG| over time": e["time_importance"]}, fs, T, "When did the evidence arrive?", smooth=16, vspan=(1.0, 4.0), colors={"|IG| over time": store.MODEL_COLORS[a["model"]]}), key=f"{key}_attr_time")
        top = signals.top_channels(e["channel_importance"], ch, 4)
        st.markdown(f"**Most influential electrodes:** {', '.join(top)}  ·  **Most influential band:** {signals_band(e)}")
    c3, c4 = st.columns(2)
    with c3:
        plot(viz.bars(signals.BAND_NAMES, e["freq_importance"], "Frequency band importance (Δ class score when removed)", signed=True, color=store.MODEL_COLORS[a["model"]], ytitle="Δ logit"), key=f"{key}_attr_band")
    with c4:
        plot(viz.tf_heat(e["tf_importance"], signals.BAND_NAMES, fs, T, "Time-frequency importance (Δ logit)"), key=f"{key}_attr_tf")
    with st.expander("Channel × time attribution map (IG)"):
        plot(viz.heat_signal(e["ig"], ch, fs, "Signed IG per electrode and time", height=420), key=f"{key}_attr_map")


def signals_band(e):
    i = int(np.argmax(np.abs(e["freq_importance"])))
    lo, hi = list(signals.FREQ_BANDS.values())[i]
    return f"{signals.BAND_NAMES[i].replace('_', ' ')} ({lo}–{hi} Hz)"


# ----------------------------------------------------------------------------- sanity & caveats
def sanity_section(ctx):
    a, ref, level, key = ctx["a"], ctx["ref"], ctx["level"], ctx["key"]
    cn = ctx["class_names"]
    ch = ref["channel_names_in_order"]
    tgt = cn[a["target"]]
    imp = a["expl"]["channel_importance"]
    li = signals.lateralization_index(imp, ch)
    mid = float(sum(imp[ch.index(c)] for c in signals.MIDLINE_CH if c in ch))
    step_header(5, "Does the reasoning make neuroscientific sense?",
                "A sanity check against what is known about motor imagery (event-related desynchronisation, ERD).")
    st.markdown(f"For **{tgt.replace('_', ' ')}** the expectation is: {CLASS_HINT[tgt]}")
    c1, c2 = st.columns([1, 2])
    with c1:
        plot(viz.gauge(li, "Lateralisation of evidence\n(−1 left … +1 right hemisphere)"), key=f"{key}_lat")
    with c2:
        if tgt == "left_hand":
            ok = li > 0.1
            txt = f"Evidence leans **{'right' if li > 0 else 'left'}** (index {li:+.2f}). " + ("✔ Consistent with contralateral control." if ok else "✘ Not clearly contralateral — treat this prediction with caution.")
        elif tgt == "right_hand":
            ok = li < -0.1
            txt = f"Evidence leans **{'left' if li < 0 else 'right'}** (index {li:+.2f}). " + ("✔ Consistent with contralateral control." if ok else "✘ Not clearly contralateral — treat this prediction with caution.")
        else:
            ok = abs(li) < 0.35 or mid > 0.2
            txt = f"Mid-line electrodes carry **{mid:.0%}** of the evidence; lateralisation index {li:+.2f}. " + ("✔ Plausible for a mid-line/bilateral class." if ok else "✘ Strongly one-sided evidence is unusual for this class.")
        callout("ok" if ok else "warn", "Plausibility check", txt)
        st.caption("Attribution magnitude shows what the *model* used, not the sign of the underlying brain change (ERD vs ERS). A model can be right for the wrong reason (e.g. eye/muscle artifacts) — this check only flags obvious mismatches.")


def caveats_section(ctx, conf, edited):
    a, root, sid, key = ctx["a"], ctx["root"], ctx["sid"], ctx["key"]
    e = a["expl"]
    step_header(6, "How much should you trust it?", "Confidence, numerical checks and known limitations.")
    m = store.get_metrics(root, sid)[f"{a['model']}_test"]
    ld = float(np.abs(e["ig_completeness_delta"]))
    scale = float(np.abs(a["logits"][a["target"]])) or 1.0
    kpi_row([("Confidence", conf, f"top prob {a['probs'].max():.0%}"),
             ("Model accuracy (this subject, test session)", f"{m['accuracy']:.0%}", f"chance = 25 % · κ = {m['kappa']:.2f}"),
             ("Attribution completeness error", f"{ld:.1e}", "≈ 0 means the explanation adds up to the model score"),
             ("Edited input?", "yes" if edited else "no", "edited signals are out-of-distribution")])
    bs = store.get_baseline_stats(root, sid)
    msgs = []
    if m["accuracy"] < 0.5:
        msgs.append("This subject's model is weak overall (<50 % accuracy), so individual predictions are unreliable.")
    if conf == "low":
        msgs.append("The model is unsure: the top two classes are close.")
    if edited:
        msgs.append("You modified or synthesised this signal; the model was never trained on such inputs.")
    if bs and "channel_std_uV" in bs:
        r = float(np.std(ctx["x"], axis=1).mean() / (bs["channel_std_uV"].mean() or 1))
        if not (0.4 < r < 2.5):
            msgs.append(f"Signal amplitude is {r:.1f}× the training average — check units/preprocessing.")
    if msgs:
        callout("warn", "Things to keep in mind", "<ul>" + "".join(f"<li>{t}</li>" for t in msgs) + "</ul>")
    else:
        callout("ok", "No red flags", "Confident prediction, within the typical amplitude range, numerically consistent explanation.")


# ----------------------------------------------------------------------------- the whole story
def render_story(ctx):
    """ctx: root, sid, model, x, a (analysis), ref, class_names, true_label, level, key, edited"""
    a, ref, level, key = ctx["a"], ctx["ref"], ctx["level"], ctx["key"]
    cn = ctx["class_names"]
    fs = ref["sampling_frequency_hz"]
    ch = ref["channel_names_in_order"]
    conf = prediction_card(a, cn, ctx.get("true_label"), key)

    with st.container(border=True):
        step_header(1, "The brain signal the model received",
                    "22 scalp electrodes × 4 seconds (from the cue), already filtered to 4–38 Hz and resampled to 128 Hz.")
        st.markdown(lv(level,
            "EEG records tiny voltage changes on the scalp. When you *imagine* moving, the rhythms over the motor cortex change — that is the clue the model looks for.",
            "Epoch: 513 samples (4 s) per electrode in µV. Motor imagery changes power in the **mu (8–13 Hz)** and **beta (13–30 Hz)** bands (ERD/ERS).",
            f"Input tensor X ∈ ℝ^(22×{ctx['x'].shape[1]}), µV, band-pass 4–38 Hz (zero-phase IIR), fs = {fs:g} Hz, referenced to the left mastoid; no artifact rejection."))
        c1, c2 = st.columns([1.7, 1])
        with c1:
            plot(viz.signal_figure(ctx["x"], ch, fs, height=400), key=f"{key}_sig")
        with c2:
            plot(viz.spectrum_figure([("this trial", ctx["x"])], fs, bands=signals.FREQ_BANDS), key=f"{key}_spec")
            st.caption("Average spectrum over the 22 electrodes. Shaded zones = frequency bands used in the explanations.")

    with st.container(border=True):
        step_header(2, f"The model's recipe: {store.MODEL_LABELS[a['model']]}", "")
        if a["model"] == "eegnet":
            arch_flow([("Input", "22 × 513"), ("Temporal conv", "8 filters · rhythms"), ("Spatial conv", "16 filters · scalp areas"),
                       ("ELU + pool", "÷4"), ("Separable conv", "16 maps · temporal summary"), ("ELU + pool", "÷8"), ("Linear", "256 → 4 classes"), ("Softmax", "probabilities")])
            st.markdown(lv(level,
                "**EEGNet** is a small neural network (≈2.5 thousand adjustable numbers). It learns *by itself* which rhythms and which scalp areas are useful, directly from the examples.",
                "EEGNet-v4 (Lawhern et al., 2018): temporal convolution → depth-wise spatial convolution → separable convolution → dense layer. It is end-to-end: filters are learned jointly with the classifier.",
                "F1=8, D=2, F2=16, kernel 64 (temporal), 16 (separable); BN + ELU + average pooling (4, 8) + dropout 0.5 (train only); max-norm 1 (spatial) and 0.25 (dense). Trained with Adam, early stopping on validation loss."))
        else:
            arch_flow([("Input", "22 × 513"), ("CSP filters", "8 virtual channels"), ("Log band power", "8 numbers"), ("LDA", "weighted vote"), ("Softmax", "probabilities")])
            st.markdown(lv(level,
                "**CSP + LDA** is the classic hand-crafted pipeline of BCI research: first find scalp patterns whose *loudness* differs between the movements (CSP), then take one simple weighted vote (LDA).",
                "Common Spatial Patterns learn spatial filters maximising class differences in variance; the log-variance of each filtered signal is the feature; Linear Discriminant Analysis classifies the 8-D feature vector.",
                "Features $f_k=\\log\\mathrm{var}(\\mathbf w_k^\\top \\mathbf x)$; LDA with shrinkage covariance (Ledoit–Wolf, `lsqr`): $z_c=\\mathbf a_c^\\top\\mathbf f+b_c$. Fully linear after the log — which makes this explanation exact."))

    with st.container(border=True):
        step_header(3, "What happens inside the model", "Layer by layer, with the model's real weights.")
        (_csp_internals if a["model"] == "csp_lda" else _eegnet_internals)(ctx)

    with st.container(border=True):
        attribution_section(ctx)
    with st.container(border=True):
        sanity_section(ctx)
    with st.container(border=True):
        caveats_section(ctx, conf, ctx.get("edited", False))

"""Compare: ONE EEG signal -> BOTH models -> predictions, reasons, and how similar the reasons are."""
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import analysis, explain_ui, inputs_ui, signals, store, viz
from ..theme import callout, chip, hero, kpi_row, plot, step_header, table


def _sig(*parts):
    h = hashlib.md5()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else str(p).encode())
    return h.hexdigest()


def decisive_table(a, ref, root, sid, cn, k=4):
    p, q = a["pred"], int(np.argsort(-a["probs"])[1])
    bd = a["breakdown"]
    rows = []
    if a["model"] == "csp_lda":
        diff = bd["contrib"][p] - bd["contrib"][q]
        for i in np.argsort(-np.abs(diff))[:k]:
            rows.append({"decisive part": f"CSP component {i + 1} (log band power)", "value": f"f = {bd['features'][i]:+.2f}",
                         "weight (pred − runner-up)": f"{bd['coef'][p, i] - bd['coef'][q, i]:+.2f}", "effect on score gap": f"{diff[i]:+.2f}"})
    else:
        diff = bd["contrib"][p] - bd["contrib"][q]
        nb = bd["n_bins"]
        flat = diff.reshape(-1)
        peaks = analysis.temporal_peak_freqs(root, sid)
        for j in np.argsort(-np.abs(flat))[:k]:
            m, t = divmod(int(j), nb)
            top_in = int(np.argmax(np.abs(bd["pointwise"][m])))
            rows.append({"decisive part": f"feature map {m + 1}, {t * 4 / nb:.2f}-{(t + 1) * 4 / nb:.2f} s  (fed mainly by spatial filter #{top_in + 1}, ≈{peaks[top_in // bd['D']]:.0f} Hz)",
                         "value": f"h = {bd['h'][j]:+.2f}", "weight (pred − runner-up)": f"{bd['Wc'][p, j] - bd['Wc'][q, j]:+.3f}", "effect on score gap": f"{diff.reshape(-1)[j]:+.2f}"})
    return pd.DataFrame(rows)


def _mini_card(a, cn, true, col_hex):
    pred = cn[a["pred"]]
    conf, margin = explain_ui.confidence_label(a["probs"])
    ok = ""
    if true:
        ok = chip("✔ correct", "ok") if true == pred else chip("✘ wrong", "")
    st.markdown(f'<div class="xb-card" style="border-top:4px solid {col_hex}"><div class="small-muted">{store.MODEL_LABELS[a["model"]]}</div>'
                f'<div style="font-size:1.8rem;font-weight:800;color:{col_hex}">{pred.replace("_", " ")}</div>'
                f'<div>probability <b>{a["probs"][a["pred"]]:.0%}</b> · confidence <b>{conf}</b> {ok}</div></div>', unsafe_allow_html=True)


def _aligned(e, key):
    return np.asarray(e[key], float)


def render(S):
    root, sid, ref, cn, level, xc = S["root"], S["sid"], S["ref"], S["cn"], S["level"], S["xc"]
    ch, pos, fs = ref["channel_names_in_order"], ref["channel_positions_3d_m"], ref["sampling_frequency_hz"]
    hero("Compare the two models", "Feed <b>one</b> EEG signal to <b>EEGNet</b> and <b>CSP+LDA</b>. Do they agree? Do they reason the same way? "
         "Which weights, layers and features pushed each one to its answer?",
         [chip("same input", "blue"), chip("two architectures", "gray"), chip("explanation similarity", "gray")])
    step_header(1, "Choose the EEG signal", "It will be given, unchanged, to both models.")
    inp = inputs_ui.render_signal_builder(root, sid, ref, "cmp", cn)
    if inp is None:
        return
    with st.expander("⚙️ How should the two explanations be aligned?"):
        tmode = st.radio("Explain …", ["each model's own prediction (default)", "the true class (needs a labelled trial)", "one class I choose, for both models"], key="cmp_tmode")
        tcls = st.selectbox("Class", [c.replace("_", " ") for c in cn], key="cmp_tcls", disabled=not tmode.startswith("one"))
        st.caption("If the models disagree, explaining *different* classes makes the maps hard to compare. Choosing one class for both removes that confound.")
    if tmode.startswith("each"):
        tmode_k, tcls_i = "predicted", None
    elif tmode.startswith("the true"):
        if inp["true"] is None:
            st.warning("This signal has no label — falling back to each model's own prediction.")
            tmode_k, tcls_i = "predicted", None
        else:
            tmode_k, tcls_i = "class", cn.index(inp["true"])
    else:
        tmode_k, tcls_i = "class", cn.index(tcls.replace(" ", "_"))
    sig = _sig(inp["x"], sid, tmode_k, tcls_i)
    if st.button("▶ Run both models and compare", type="primary", key="cmp_run"):
        st.session_state["cmp_sig"] = sig
    if st.session_state.get("cmp_sig") != sig:
        callout("tip", "Ready when you are", "Press **Run both models and compare**.")
        return
    with st.spinner("Running both models and computing explanations …"):
        A = {m: analysis.analyze(root, m, sid, inp["x"], tmode_k, tcls_i, 32, 3) for m in store.MODEL_KEYS}
    ea, eb = A["eegnet"], A["csp_lda"]
    true = inp["true"]
    agree = ea["pred"] == eb["pred"]

    # ---------------------------------------------------------------- 2 verdict
    step_header(2, "Do they agree?", inp["desc"])
    if agree:
        st.markdown(f'<div class="xb-verdict agree">✔ Both models predict <b>{cn[ea["pred"]].replace("_", " ")}</b>'
                    + (f" — and that is {'correct' if true == cn[ea['pred']] else 'wrong'} (true: {true.replace('_', ' ')})" if true else "") + "</div>", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="xb-verdict disagree">⚠ They disagree: EEGNet → <b>{cn[ea["pred"]].replace("_", " ")}</b> · CSP+LDA → <b>{cn[eb["pred"]].replace("_", " ")}</b>'
                    + (f" (true: {true.replace('_', ' ')})" if true else "") + "</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        _mini_card(ea, cn, true, store.MODEL_COLORS["eegnet"])
        plot(viz.prob_bar(ea["probs"], cn, cn[ea["pred"]], true, color=store.MODEL_COLORS["eegnet"], height=230, title="EEGNet probabilities"), key="cmp_p1")
    with c2:
        _mini_card(eb, cn, true, store.MODEL_COLORS["csp_lda"])
        plot(viz.prob_bar(eb["probs"], cn, cn[eb["pred"]], true, color=store.MODEL_COLORS["csp_lda"], height=230, title="CSP+LDA probabilities"), key="cmp_p2")
    if inp["notes"]:
        st.caption("Edits applied to the signal: " + ", ".join(inp["notes"]))

    # ---------------------------------------------------------------- 3 how each reasons
    step_header(3, "How did each one reason?", "Two very different recipes reach a decision in two very different ways.")
    nparam = sum(r["n_params"] for r in store.get_layer_summary(root, sid))
    df = pd.DataFrame({
        "": ["Features", "Where spatial info comes from", "Where rhythm info comes from", "Decision rule", "Adjustable numbers", "Explanation exactness"],
        "EEGNet": ["Learned end-to-end (256 deep features)", "16 learned spatial filters", "8 learned temporal (band-pass-like) filters", "Linear layer on deep features → softmax", f"{nparam:,}", "Approximate (gradient-based attribution)"],
        "CSP + LDA": ["8 hand-designed log band-power numbers", "8 CSP spatial filters (variance-based)", "Fixed 4–38 Hz pre-filter; power only (no timing)", "Linear discriminant (weighted vote) → softmax", f"{8 * 22 + 4 * 8 + 4:,}", "Exact for the final stage (additive weights)"]})
    table(df)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**What decided EEGNet**")
        table(decisive_table(ea, ref, root, sid, cn))
    with t2:
        st.markdown("**What decided CSP+LDA**")
        table(decisive_table(eb, ref, root, sid, cn))
    with st.expander("See each model's full step-by-step story"):
        tabs = st.tabs([store.MODEL_SHORT[m] for m in store.MODEL_KEYS])
        for tb, m in zip(tabs, store.MODEL_KEYS):
            with tb:
                explain_ui.render_story(dict(root=root, sid=sid, model=m, x=inp["x"], a=A[m], ref=ref, class_names=cn, true_label=true, level=level, key=f"cmp_{m}", edited=inp["edited"]))

    # ---------------------------------------------------------------- 4 evidence side by side
    step_header(4, "Where did each model find its evidence?", "Same electrodes? Same moments? Same rhythms?")
    Ea, Eb = ea["expl"], eb["expl"]
    c1, c2, c3 = st.columns(3)
    with c1:
        plot(viz.topomap(Ea["channel_signed"], ch, pos, f"EEGNet · evidence for '{cn[ea['target']].replace('_', ' ')}'", height=340, colorbar_title="IG"), key="cmp_t1")
    with c2:
        plot(viz.topomap(Eb["channel_signed"], ch, pos, f"CSP+LDA · evidence for '{cn[eb['target']].replace('_', ' ')}'", height=340, colorbar_title="IG"), key="cmp_t2")
    with c3:
        d = Ea["channel_importance"] - Eb["channel_importance"]
        plot(viz.topomap(d, ch, pos, "Importance difference (EEGNet − CSP+LDA)", height=340, colorbar_title="Δ share"), key="cmp_t3")
    T = Ea["time_importance"].size
    plot(viz.time_curves({"EEGNet": Ea["time_importance"] / Ea["time_importance"].max(), "CSP+LDA": Eb["time_importance"] / Eb["time_importance"].max()}, fs, T,
                         "When was the evidence? (|IG|, each curve scaled to its own maximum)", smooth=16, vspan=(1.0, 4.0), colors={"EEGNet": store.MODEL_COLORS["eegnet"], "CSP+LDA": store.MODEL_COLORS["csp_lda"]}), key="cmp_time")
    c4, c5 = st.columns(2)
    with c4:
        plot(viz.grouped_bars(signals.BAND_NAMES, {"EEGNet": Ea["freq_importance"], "CSP+LDA": Eb["freq_importance"]}, "Which rhythm? (Δ target logit when the band is removed)", colors={"EEGNet": store.MODEL_COLORS["eegnet"], "CSP+LDA": store.MODEL_COLORS["csp_lda"]}), key="cmp_band")
    with c5:
        st.markdown("&nbsp;")
        li_a, li_b = signals.lateralization_index(Ea["channel_importance"], ch), signals.lateralization_index(Eb["channel_importance"], ch)
        kpi_row([("EEGNet lateralisation", f"{li_a:+.2f}", "−1 left … +1 right hemisphere"), ("CSP+LDA lateralisation", f"{li_b:+.2f}", "−1 left … +1 right hemisphere")])
        st.caption(f"Top electrodes — EEGNet: {', '.join(signals.top_channels(Ea['channel_importance'], ch, 4))} · CSP+LDA: {', '.join(signals.top_channels(Eb['channel_importance'], ch, 4))}")
    c6, c7 = st.columns(2)
    with c6:
        plot(viz.tf_heat(Ea["tf_importance"], signals.BAND_NAMES, fs, T, "EEGNet · time-frequency importance"), key="cmp_tf1")
    with c7:
        plot(viz.tf_heat(Eb["tf_importance"], signals.BAND_NAMES, fs, T, "CSP+LDA · time-frequency importance"), key="cmp_tf2")

    # ---------------------------------------------------------------- 5 similarity
    step_header(5, "How similar are the two explanations?", "1 = identical reasoning · 0 = unrelated · negative = opposite.")
    pack = lambda E: {k: np.asarray(E[k])[None] for k in ["channel_importance", "window_importance", "freq_importance", "tf_importance", "ig"]}
    sim = xc.explanation_similarity(pack(Ea), pack(Eb), 8).iloc[0]
    items = [("channel_pearson", "Electrodes\n(correlation)"), ("channel_top5_overlap", "Top-5 electrodes\n(overlap)"), ("time_window_pearson", "Timing\n(correlation)"),
             ("freq_band_pearson", "Rhythms\n(correlation)"), ("attr_map_pearson", "Channel×time map\n(correlation)")]
    cols = st.columns(5)
    for col, (k, lab) in zip(cols, items):
        with col:
            plot(viz.gauge(None if pd.isna(sim[k]) else float(sim[k]), lab, -1, 1), key=f"cmp_g_{k}")
    mean_sim = float(np.nanmean([sim[k] for k, _ in items]))
    if agree:
        callout("ok" if mean_sim > 0.5 else "warn", "Same answer — same reasons?",
                f"Both models predicted the same class. Average similarity of their explanations is **{mean_sim:.2f}**. "
                + ("They largely rely on the same evidence." if mean_sim > 0.5 else "Yet their evidence only partly overlaps: **the same prediction does not guarantee the same reasoning.**"))
    else:
        callout("warn", "Different answers", f"The models disagree; explanation similarity is **{mean_sim:.2f}**. Compare the maps above to see which one relied on the more plausible evidence (see the neuro-plausibility check in each story).")

    # ---------------------------------------------------------------- 6 dataset-level evidence
    step_header(6, "Is that typical? Evidence from all test trials", "Computed from the explanations saved during training.")
    comp = store.get_comparison_all(root)
    if comp.empty:
        st.info("No saved comparison records found.")
    else:
        scope = st.radio("Scope", [f"Subject {sid:02d} only", "All subjects"], horizontal=True, key="cmp_scope")
        c = comp if scope.startswith("All") else comp[comp["subject"] == sid]
        c = c.copy()
        c["category"] = np.where(c["agree"], np.where(c["eegnet_correct"], "agree · both correct", "agree · both wrong"), "disagree")
        cols = ["channel_pearson", "channel_top5_overlap", "time_window_pearson", "freq_band_pearson", "attr_map_pearson"]
        g = c.groupby("category")[cols].mean()
        nice = {"channel_pearson": "electrodes", "channel_top5_overlap": "top-5 overlap", "time_window_pearson": "timing", "freq_band_pearson": "rhythms", "attr_map_pearson": "channel×time"}
        fig = go.Figure()
        pal = {"agree · both correct": viz.GREEN, "agree · both wrong": viz.RED, "disagree": viz.GRAY}
        for cat in g.index:
            fig.add_trace(go.Bar(x=[nice[k] for k in cols], y=g.loc[cat].values, name=f"{cat} (n={int((c['category'] == cat).sum())})", marker_color=pal[cat]))
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.2))
        fig.update_yaxes(title="mean similarity", gridcolor="#eef1f6")
        plot(viz._layout(fig, f"Explanation similarity by outcome · agreement rate {c['agree'].mean():.0%} of {len(c)} trials", 340), key="cmp_ds")
        ag = c[c["agree"]]
        dis = c[~c["agree"]]
        gap = ag["channel_pearson"].mean() - dis["channel_pearson"].mean()
        verdict = ("Agreeing on the *answer* therefore says little about agreeing on the *reasons*: two very different feature extractors can reach the same conclusion through partly different evidence."
                   if (np.isnan(gap) or gap < 0.1) else
                   "Models that agree on the answer also tend to share somewhat more of their evidence — but the overlap is still far from perfect.")
        st.markdown(f"**Reading:** when both models give the same answer, the mean electrode-pattern similarity is **{ag['channel_pearson'].mean():.2f}**; when they disagree it is **{dis['channel_pearson'].mean():.2f}**. " + verdict)

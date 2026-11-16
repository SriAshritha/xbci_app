"""Playground: give ONE model any input (signal in any domain, or CSP features) and see why it predicted that."""
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from .. import analysis, explain_ui, inputs_ui, signals, store, viz
from ..theme import callout, chip, hero, plot, step_header, table


def _sig(*parts):
    h = hashlib.md5()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else str(p).encode())
    return h.hexdigest()


def _signal_mode(S):
    root, sid, ref, cn = S["root"], S["sid"], S["ref"], S["cn"]
    step_header("A", "Choose a model", "Both were trained on this subject's training session.")
    model = st.radio("Model", store.MODEL_KEYS, format_func=lambda m: store.MODEL_LABELS[m], horizontal=True, key="pg_model", label_visibility="collapsed")
    step_header("B", "Give it an EEG signal", "From the recordings, your own file, or built from parameters — then optionally modify it in any domain.")
    inp = inputs_ui.render_signal_builder(root, sid, ref, "pg", cn)
    if inp is None:
        return
    with st.expander("⚙️ Advanced explanation settings"):
        c1, c2, c3 = st.columns(3)
        tmode = c1.selectbox("Explain which class?", ["the model's own prediction", "a class I choose"], key="pg_tmode")
        tcls = c2.selectbox("Class", [c.replace("_", " ") for c in cn], key="pg_tcls", disabled=tmode.startswith("the"))
        steps = c3.select_slider("Integrated-Gradients steps (accuracy vs speed)", [16, 32, 64], 32, key="pg_steps")
    target_mode = "predicted" if tmode.startswith("the") else "class"
    target_class = None if target_mode == "predicted" else cn.index(tcls.replace(" ", "_"))
    sig = _sig(inp["x"], model, sid, target_mode, target_class, steps)
    step_header("C", "Run", "")
    run = st.button("▶ Run the model and explain", type="primary", key="pg_run")
    if run:
        st.session_state["pg_sig"] = sig
    if st.session_state.get("pg_sig") != sig:
        callout("tip", "Ready when you are", "Press **Run the model and explain**. (Results below refresh only when you press it, because explanations take a few seconds.)")
        return
    with st.spinner("Running the model and computing explanations …"):
        a = analysis.analyze(root, model, sid, inp["x"], target_mode, target_class, steps, 3)
    st.markdown(f"##### Result · {chip(inp['desc'], 'gray')}", unsafe_allow_html=True)
    if inp["notes"]:
        _, p0 = analysis.quick_predict(root, model, sid, inp["x_base"])
        st.markdown("###### Effect of your edits on the prediction")
        plot(viz.prob_bar(a["probs"], cn, cn[a["pred"]], inp["true"], color=store.MODEL_COLORS[model], compare=p0, compare_name="original signal", height=290), key="pg_effect")
        if int(p0.argmax()) != a["pred"]:
            callout("warn", "The edit flipped the decision", f"{cn[int(p0.argmax())].replace('_', ' ')} → **{cn[a['pred']].replace('_', ' ')}** after: {', '.join(inp['notes'])}.")
    ctx = dict(root=root, sid=sid, model=model, x=inp["x"], a=a, ref=ref, class_names=cn, true_label=inp["true"], level=S["level"], key="pg", edited=inp["edited"])
    explain_ui.render_story(ctx)


def _feature_mode(S):
    root, sid, ref, cn = S["root"], S["sid"], S["ref"], S["cn"]
    st.markdown("Skip the EEG entirely and **type in the 8 CSP band-power features** that the LDA classifier receives. "
                "This is the quickest way to see how the weights turn numbers into a decision.")
    x0, true, desc, gi = inputs_ui._dataset_trial(root, sid, "pgf", cn)
    p = store.get_csp_params(root, sid)
    W, coef, b = p["filters"].astype(np.float64), p["lda_coef"].astype(np.float64), p["lda_intercept"].astype(np.float64)
    f0 = np.log(((W @ x0.astype(np.float64)) ** 2).mean(1) + 1e-12)
    stats = analysis.csp_class_feature_stats(root, sid)
    lo = (stats["mean"] - 3 * stats["std"]).min(0)
    hi = (stats["mean"] + 3 * stats["std"]).max(0)
    st.markdown("##### Feature values  *(start = this trial's real values)*")
    cols = st.columns(4)
    f = np.zeros(8)
    for k in range(8):
        with cols[k % 4]:
            f[k] = st.slider(f"CSP {k + 1}", float(min(lo[k], f0[k]) - 0.5), float(max(hi[k], f0[k]) + 0.5), float(f0[k]), 0.01, key=f"pgf_{gi}_{k}")
    z = coef @ f + b
    pr = np.exp(z - z.max()); pr /= pr.sum()
    pred = int(pr.argmax()); q = int(np.argsort(-pr)[1])
    c1, c2 = st.columns([1, 1.2])
    with c1:
        plot(viz.prob_bar(pr, cn, cn[pred], true, color=store.MODEL_COLORS["csp_lda"], title="LDA output for your feature values"), key="pgf_prob")
        st.latex(r"z_c=\sum_k a_{c,k} f_k + b_c")
        st.code("\n".join(f"z[{cn[c]:<10}] = {z[c]:+8.2f}   P = {pr[c]:.1%}" for c in range(4)), language="text")
    with c2:
        diff = coef[pred] * f - coef[q] * f
        names = [f"CSP {k + 1}" for k in range(8)] + ["intercept"]
        plot(viz.contrib_bar(names, np.append(diff, b[pred] - b[q]), f"Why '{cn[pred].replace('_', ' ')}' beats '{cn[q].replace('_', ' ')}'"), key="pgf_contrib")
    fig = viz.go.Figure()
    for c in range(4):
        fig.add_trace(viz.go.Bar(x=[f"CSP {k + 1}" for k in range(8)], y=stats["mean"][c], name=cn[c].replace("_", " "), marker_color=viz.CLASS_COLORS[cn[c]], opacity=0.55))
    fig.add_trace(viz.go.Scatter(x=[f"CSP {k + 1}" for k in range(8)], y=f, mode="markers", name="your values", marker=dict(symbol="diamond", size=13, color="black")))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15))
    plot(viz._layout(fig, "Your features (◆) vs the typical value of each class", 330), key="pgf_cmp")
    ch, pos = ref["channel_names_in_order"], ref["channel_positions_3d_m"]
    top = np.argsort(-np.abs(diff))[:4]
    st.markdown("##### Where on the scalp do the decisive components live?")
    plot(viz.topomap_grid([p["patterns"][k] for k in top], [f"CSP {k + 1}  (Δscore {diff[k]:+.2f})" for k in top], ch, pos, ncols=4, height_per_row=220), key="pgf_topo")
    callout("tip", "What this shows", "CSP+LDA is *linear* in the features, so every number you move changes the class scores by exactly weight × change. EEGNet has no such hand-made features, which is why it is only available in signal mode.")


def render(S):
    hero("Playground", "Give a model any input, then follow — step by step — <b>why</b> it predicted what it did: the weights, the layers, the electrodes, the rhythms and the moments in time that mattered.",
         [chip("single model", "blue"), chip("time · frequency · time-frequency inputs", "gray"), chip("exact weights", "gray")])
    mode = st.radio("What do you want to give the model?", ["EEG signal  (time · frequency · time–frequency domain)", "CSP feature values  (CSP+LDA only)"],
                    horizontal=True, key="pg_mode")
    if mode.startswith("EEG"):
        _signal_mode(S)
    else:
        _feature_mode(S)

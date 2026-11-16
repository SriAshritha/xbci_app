"""Start here: what is a BCI, interactive tour, real ERD in the data, what you can do in the app."""
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .. import analysis, signals, store, viz
from ..theme import callout, chip, hero, kpi_row, plot, step_header

TOUR = [
    ("1 · Record", "22 electrodes on the scalp record 4 seconds of brain voltage, starting at the cue."),
    ("2 · Spatial filtering", "Learned weighted mixes of the electrodes (CSP filters) create 8 'virtual channels' that each listen to one scalp pattern."),
    ("3 · Band power", "Each virtual channel is squeezed into ONE number: how strong its rhythm was (log power). Diamonds = this trial, bars = typical class values."),
    ("4 · Weighted vote", "A linear classifier (LDA) multiplies each number by a learned weight. The bars show which numbers made the winning class beat the runner-up."),
    ("5 · Decision", "Scores are converted to probabilities. EEGNet does steps 2–4 too — but learns its own filters and features from scratch."),
]


def _tour(S):
    root, sid, ref, cn = S["root"], S["sid"], S["ref"], S["cn"]
    ch, pos, fs = ref["channel_names_in_order"], ref["channel_positions_3d_m"], ref["sampling_frequency_hz"]
    d = store.get_subject_data(root, sid)
    te = store.get_split(root, sid)["test_idx"]
    c1, c2 = st.columns([1, 2])
    with c1:
        cls = st.selectbox("Pick a movement the subject imagined", [c.replace("_", " ") for c in cn], index=1, key="tour_cls")
    idx = te[d["y"][te] == cn.index(cls.replace(" ", "_"))]
    x = d["X"][int(idx[0])].astype(np.float32)
    lg, pr = analysis.quick_predict(root, "csp_lda", sid, x)
    bd = analysis.csp_breakdown(root, sid, x, lg)
    stats = analysis.csp_class_feature_stats(root, sid)
    p = int(pr.argmax())
    q = int(np.argsort(-pr)[1])
    with c2:
        step = st.select_slider("Tour step", options=[t[0] for t in TOUR], key="tour_step")
    play = st.button("▶ Play the whole tour", key="tour_play")
    box = st.empty()

    def draw(i, tag):
        with box.container(border=True):
            st.markdown(f"**{TOUR[i][0]}** — {TOUR[i][1]}")
            if i == 0:
                plot(viz.signal_figure(x, ch, fs, f"True class: {cls}", height=380), key=f"tour_{tag}_0")
            elif i == 1:
                a, b = st.columns([1.2, 1])
                with a:
                    plot(viz.heat_signal(bd["virtual_channels"], [f"CSP {k + 1}" for k in range(8)], fs, "8 virtual channels", height=300), key=f"tour_{tag}_1a")
                with b:
                    top = np.argsort(-np.abs(bd["contrib"][p] - bd["contrib"][q]))[:2]
                    plot(viz.topomap_grid([bd["patterns"][k] for k in top], [f"CSP {k + 1} pattern" for k in top], ch, pos, ncols=2, height_per_row=230), key=f"tour_{tag}_1b")
            elif i == 2:
                fig = go.Figure()
                for c in range(4):
                    fig.add_trace(go.Bar(x=[f"CSP {k + 1}" for k in range(8)], y=stats["mean"][c], name=cn[c].replace("_", " "), marker_color=viz.CLASS_COLORS[cn[c]], opacity=0.5))
                fig.add_trace(go.Scatter(x=[f"CSP {k + 1}" for k in range(8)], y=bd["features"], mode="markers", name="this trial", marker=dict(symbol="diamond", size=13, color="black")))
                fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15))
                plot(viz._layout(fig, "log band power per virtual channel", 340), key=f"tour_{tag}_2")
            elif i == 3:
                diff = bd["contrib"][p] - bd["contrib"][q]
                plot(viz.contrib_bar([f"CSP {k + 1}" for k in range(8)], diff, f"What made '{cn[p].replace('_', ' ')}' beat '{cn[q].replace('_', ' ')}'", topn=8), key=f"tour_{tag}_3")
            else:
                plot(viz.prob_bar(pr, cn, cn[p], cn[d["y"][int(idx[0])]], color=viz.BLUE, title="CSP+LDA decision"), key=f"tour_{tag}_4")

    if play:
        for i in range(len(TOUR)):
            draw(i, f"p{i}")
            time.sleep(1.6)
    else:
        draw([t[0] for t in TOUR].index(step), "s")


def _erd(S):
    root, sid, ref, cn = S["root"], S["sid"], S["ref"], S["cn"]
    ch, pos, fs = ref["channel_names_in_order"], ref["channel_positions_3d_m"], ref["sampling_frequency_hz"]
    band = st.radio("Rhythm", ["mu (8–13 Hz)", "beta (13–30 Hz)"], horizontal=True, key="erd_band")
    b = (8.0, 13.0) if band.startswith("mu") else (13.0, 30.0)
    r = analysis.band_power_contrast(root, sid, b, (1.5, 3.5))
    T = r["curves"].shape[2]
    c1, c2 = st.columns([1.2, 1])
    with c1:
        curves = {}
        for cname in ["left_hand", "right_hand"]:
            for e in ["C3", "C4"]:
                curves[f"{cname.replace('_', ' ')} @ {e}"] = r["curves"][cn.index(cname), ch.index(e)]
        fig = viz.time_curves(curves, fs, T, f"{band.split()[0]} power relative to the average of all classes (dB) — training trials",
                              ytitle="dB (negative = weaker than average)", smooth=1, vspan=(1.0, 4.0), height=340)
        fig.update_traces(selector=dict(name="left hand @ C3"), line=dict(color=viz.BLUE, dash="dot"))
        fig.update_traces(selector=dict(name="left hand @ C4"), line=dict(color=viz.BLUE))
        fig.update_traces(selector=dict(name="right hand @ C3"), line=dict(color=viz.RED))
        fig.update_traces(selector=dict(name="right hand @ C4"), line=dict(color=viz.RED, dash="dot"))
        plot(fig, key="erd_curves")
    with c2:
        plot(viz.topomap_grid(list(r["maps"]), [c.replace("_", " ") for c in cn], ch, pos, ncols=2, share_scale=True, height_per_row=215), key="erd_maps")
    st.markdown("**What to look for:** imagining the **left** hand should weaken the rhythm over the **right** motor cortex (C4) relative to the average, and the **right** hand over the **left** (C3). "
                "Maps show the 1.5–3.5 s window; **blue = weaker than average (ERD-like)**.")
    callout("tip", "Real data are messy", "Not every subject shows a clean contralateral pattern (some show almost none — see *Science Q&A › why do some subjects perform near chance?*). "
            "The maps compare each class with the *average of the four classes*, not with a pre-cue baseline, because epochs start at the cue.")


def render(S):
    root, sid = S["root"], S["sid"]
    hero("See <span style='color:#1a56ff'>why</span> a brain–computer interface decides what it decides",
         "Two models read the same brain signals: a modern neural network (<b>EEGNet</b>) and the classic <b>CSP + LDA</b>. "
         "This app opens both up — their weights, layers, electrodes and rhythms — and explains each decision step by step, even if you have never seen an EEG before.",
         [chip("BCI Competition IV-2a", "blue"), chip("motor imagery · 4 classes", "gray"), chip("exact weights + attributions", "gray")])
    acc = store.accuracy_table(root)
    n_sub = len(store.list_subjects(root))
    kpi_row([("Subjects", str(n_sub), "each has its own two models"),
             ("EEGNet test accuracy", f"{acc['EEGNet test'].mean():.0%}" if len(acc) else "–", "mean over subjects · chance 25 %"),
             ("CSP+LDA test accuracy", f"{acc['CSP+LDA test'].mean():.0%}" if len(acc) else "–", "mean over subjects · chance 25 %"),
             ("Classes", "4", "left hand · right hand · feet · tongue")])

    st.markdown("### The 5-step story of a BCI")
    cols = st.columns(5)
    items = [("🧠", "Imagine", "The user imagines moving a hand, the feet or the tongue."), ("📡", "Record", "22 scalp electrodes measure microvolt brain signals."),
             ("🎛️", "Clean & cut", "Band-pass 4–38 Hz and cut a 4 s window after the cue."), ("🤖", "Decode", "A model turns the window into one of 4 commands."),
             ("🔍", "Explain", "<b>This app:</b> which electrodes, moments and rhythms drove the decision?")]
    for c, (ic, t, tx) in zip(cols, items):
        c.markdown(f'<div class="xb-card" style="min-height:150px"><div style="font-size:1.6rem">{ic}</div><b>{t}</b><div class="small-muted" style="margin-top:.3rem">{tx}</div></div>', unsafe_allow_html=True)

    st.markdown("### 🎬 Interactive tour — from brain signal to decision")
    st.caption("Uses the classical CSP+LDA pipeline because every step can be shown exactly. Press play, or drag the slider.")
    _tour(S)

    st.markdown("### 🧪 See the neuroscience in the real data")
    _erd(S)

    st.markdown("### What can you do here?")
    cards = [("🧪 Playground", "Give ONE model any input — a recorded trial, your own file, a synthetic signal built from parameters, or a signal you edit in the time, frequency or time-frequency domain — and follow the explanation step by step."),
             ("⚖️ Compare models", "Feed the SAME signal to both models. Do they agree? Do they reason alike? Which weights, layers and features decided?"),
             ("🔬 Under the hood", "Browse the real learned weights: filters, spatial patterns, LDA coefficients, layer shapes, training curves."),
             ("📊 Performance", "Honest accuracy, confusion matrices, calibration and agreement for every subject."),
             ("❓ Science Q&A", "Ten scientific questions answered from the data: which electrodes matter, are the models neuro-plausible, how robust are they …"),
             ("📚 Notes & Blog", "Plain-language articles, math, glossary, references, videos.")]
    for i in range(0, 6, 3):
        cs = st.columns(3)
        for c, (t, tx) in zip(cs, cards[i:i + 3]):
            c.markdown(f'<div class="xb-card" style="min-height:140px"><h4>{t}</h4><div class="small-muted">{tx}</div></div>', unsafe_allow_html=True)
    callout("tip", "New to BCI?", "Set <b>Explanation level → Beginner</b> in the sidebar. Every explanation switches to plain language with analogies; <b>Expert</b> adds the equations.")

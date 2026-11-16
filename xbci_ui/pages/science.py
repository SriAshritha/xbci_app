"""Science Q&A: scientific questions answered from the saved explanations and live experiments."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import analysis, signals, store, viz
from ..theme import callout, chip, hero, kpi_row, plot, table

QUESTIONS = {
    "q1": "Which electrodes does each model rely on — for each class?",
    "q2": "Do the models look where neuroscience says they should?",
    "q3": "Which rhythms (frequency bands) carry the evidence?",
    "q4": "When during the trial is the evidence strongest?",
    "q5": "Does a confident model tend to be a correct model?",
    "q6": "Which electrodes matter most when they are removed?",
    "q7": "Which frequency bands matter most when they are removed?",
    "q8": "How robust are the models to noise?",
    "q9": "If I mirror the head (swap left ↔ right), do left-hand and right-hand predictions swap?",
    "q10": "Do the two models look at the same electrodes? (per class)",
    "q11": "Why do some subjects perform near chance?",
}


def _collect(root, subjects, m, cn):
    parts, arrs = [], {k: [] for k in ["channel_importance", "window_importance", "freq_importance", "tf_importance"]}
    for s in subjects:
        try:
            z = store.get_expl_small(root, s, m)
        except Exception:
            continue
        pr = store.get_predictions(root, s, m)
        pr = pr[pr["split"] == "test"].set_index("trial_id")
        tid = z["trial_ids"]
        df = pr.loc[tid, ["y_true", "y_pred", "correct"]].reset_index(drop=True)
        df["subject"] = s
        df["conf"] = pr.loc[tid, [f"prob_{c}" for c in cn]].max(1).values
        parts.append(df)
        for k in arrs:
            arrs[k].append(z[k])
    if not parts:
        return None, None
    return pd.concat(parts, ignore_index=True), {k: np.concatenate(v) for k, v in arrs.items()}


def _scope(S, key):
    subs = store.list_subjects(S["root"])
    c = st.radio("Data", [f"Subject {S['sid']:02d}", "All subjects"], horizontal=True, key=f"sc_scope_{key}")
    return [S["sid"]] if c.startswith("Subject") else subs


def q1(S):
    cn, ref = S["cn"], S["ref"]
    ch, pos = ref["channel_names_in_order"], ref["channel_positions_3d_m"]
    subs = _scope(S, "q1")
    only_ok = st.checkbox("Only correctly classified trials", True, key="sc_q1_ok")
    for m in store.MODEL_KEYS:
        df, A = _collect(S["root"], subs, m, cn)
        if df is None:
            continue
        maps, titles = [], []
        for c in range(4):
            mk = (df["y_pred"] == c) & (df["correct"] if only_ok else True)
            mk = mk.values
            maps.append(A["channel_importance"][mk].mean(0) if mk.sum() else np.zeros(len(ch)))
            titles.append(f"{cn[c].replace('_', ' ')} (n={int(mk.sum())})")
        st.markdown(f"**{store.MODEL_LABELS[m]}**")
        plot(viz.topomap_grid(maps, titles, ch, pos, ncols=4, signed=False, height_per_row=230, share_scale=False), key=f"sc_q1_{m}")
    callout("tip", "How to read", "Each map is the average share of attribution (|Integrated Gradients|, summed over time) per electrode for trials the model assigned to that class. Dark = relied on heavily. "
            "For motor imagery we hope to see hot-spots over sensorimotor electrodes (C3, Cz, C4 and neighbours) and **hemisphere-specific** patterns for left vs right hand.")


def q2(S):
    root, sid, cn, ref = S["root"], S["sid"], S["cn"], S["ref"]
    ch, pos = ref["channel_names_in_order"], ref["channel_positions_3d_m"]
    subs = _scope(S, "q2")
    st.markdown("##### a) Lateralisation of the evidence (right − left, −1…+1)")
    fig = go.Figure()
    for m in store.MODEL_KEYS:
        df, A = _collect(root, subs, m, cn)
        if df is None:
            continue
        li = np.array([signals.lateralization_index(v, ch) for v in A["channel_importance"]])
        d = pd.DataFrame({"cls": [cn[i].replace("_", " ") for i in df["y_pred"]], "li": li, "ok": df["correct"].values})
        d = d[d["ok"]]
        for cname in [c.replace("_", " ") for c in cn]:
            fig.add_trace(go.Box(y=d[d["cls"] == cname]["li"], x=[cname] * int((d["cls"] == cname).sum()), name=store.MODEL_SHORT[m], legendgroup=m,
                                 marker_color=store.MODEL_COLORS[m], showlegend=(cname == cn[0].replace("_", " "))))
    fig.update_layout(boxmode="group", legend=dict(orientation="h", y=1.15))
    fig.add_hline(y=0, line_color="#3b4558")
    fig.update_yaxes(title="lateralisation index (+ = right hemisphere)", gridcolor="#eef1f6")
    plot(viz._layout(fig, "Evidence lateralisation by predicted class (correct trials)", 360), key="sc_q2a")
    st.caption("Neuroscience expects **left hand → positive** (right motor cortex) and **right hand → negative**. Feet/tongue should be near 0 (mid-line / bilateral).")
    st.markdown("##### b) Does the model's evidence map match the real band-power change in the data?")
    rows = []
    for s in subs:
        r_mu = analysis.band_power_contrast(root, s, (8.0, 13.0), (1.5, 3.5))["maps"]
        r_be = analysis.band_power_contrast(root, s, (13.0, 30.0), (1.5, 3.5))["maps"]
        real = (np.abs(r_mu) + np.abs(r_be)) / 2
        for m in store.MODEL_KEYS:
            df, A = _collect(root, [s], m, cn)
            if df is None:
                continue
            for c in range(4):
                mk = ((df["y_pred"] == c) & df["correct"]).values
                if mk.sum() < 3:
                    continue
                imp = A["channel_importance"][mk].mean(0)
                rows.append({"model": store.MODEL_SHORT[m], "class": cn[c].replace("_", " "), "r": np.corrcoef(imp, real[c])[0, 1], "subject": s})
    if rows:
        d = pd.DataFrame(rows).groupby(["model", "class"], as_index=False)["r"].mean()
        fig = go.Figure()
        for m in store.MODEL_KEYS:
            dd = d[d["model"] == store.MODEL_SHORT[m]]
            fig.add_trace(go.Bar(x=dd["class"], y=dd["r"], name=store.MODEL_SHORT[m], marker_color=store.MODEL_COLORS[m]))
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15))
        fig.update_yaxes(title="spatial correlation (Pearson r)", range=[-1, 1], gridcolor="#eef1f6")
        plot(viz._layout(fig, "Model evidence vs |class-specific mu/beta power change| across electrodes", 330), key="sc_q2b")
        c1, c2 = st.columns(2)
        maps_real = analysis.band_power_contrast(root, sid, (8.0, 13.0), (1.5, 3.5))["maps"]
        with c1:
            plot(viz.topomap_grid(list(maps_real), [c.replace("_", " ") for c in cn], ch, pos, ncols=4, share_scale=True, height_per_row=200), key="sc_q2c")
        st.caption("Above: real mu-band contrast for the selected subject (blue = weaker than the all-class average).")
    callout("warn", "Interpretation", "A positive correlation means the model's evidence sits where the data's rhythms really differ between classes. A model can also reach high accuracy using other cues (e.g. muscle artifacts) — "
            "so low correlations are a warning sign, not proof of failure, and high ones are encouraging, not proof of correctness.")


def q3(S):
    cn = S["cn"]
    subs = _scope(S, "q3")
    for m in store.MODEL_KEYS:
        df, A = _collect(S["root"], subs, m, cn)
        if df is None:
            continue
        M = np.stack([A["freq_importance"][((df["y_pred"] == c) & df["correct"]).values].mean(0) if ((df["y_pred"] == c) & df["correct"]).sum() else np.zeros(5) for c in range(4)])
        fig = go.Figure(go.Heatmap(z=M, x=[b.replace("_", " ") for b in signals.BAND_NAMES], y=[c.replace("_", " ") for c in cn], colorscale="RdBu_r", zmid=0,
                                   text=np.round(M, 2), texttemplate="%{text}", colorbar=dict(thickness=10, title="Δ logit")))
        fig.update_yaxes(autorange="reversed")
        plot(viz._layout(fig, f"{store.MODEL_SHORT[m]} · drop of the class score when the band is removed (correct trials)", 300), key=f"sc_q3_{m}")
    st.caption("Expect the **mu (8–13 Hz)** and **beta** bands to dominate for a motor-imagery model. Large importance in theta (4–8) or gamma (30–38) would hint at artifacts (eye movement, muscle).")


def q4(S):
    cn, ref = S["cn"], S["ref"]
    subs = _scope(S, "q4")
    fs = ref["sampling_frequency_hz"]
    T = store.get_subject_data(S["root"], S["sid"])["X"].shape[2]
    centers = (np.linspace(0, T, 9)[:-1] + np.linspace(0, T, 9)[1:]) / 2 / fs
    cols = st.columns(2)
    for col, m in zip(cols, store.MODEL_KEYS):
        df, A = _collect(S["root"], subs, m, cn)
        if df is None:
            continue
        ser = {cn[c].replace("_", " "): (A["window_importance"][((df["y_pred"] == c) & df["correct"]).values].mean(0) if ((df["y_pred"] == c) & df["correct"]).sum() else np.zeros(8)) for c in range(4)}
        fig = viz.line_multi(centers, ser, f"{store.MODEL_SHORT[m]} · share of |IG| per 0.5 s window", "time from cue (s)", "share", colors={cn[c].replace("_", " "): viz.CLASS_COLORS[cn[c]] for c in range(4)})
        fig.add_vrect(x0=1.0, x1=4.0, fillcolor="#2da44e", opacity=0.07, line_width=0)
        with col:
            plot(fig, key=f"sc_q4_{m}")
    st.caption("Motor imagery is expected at about 1–4 s of the epoch (green). Evidence concentrated in the first second (0–1 s) would be the cue/visual response rather than imagery.")


def q5(S):
    cn = S["cn"]
    subs = _scope(S, "q5")
    fig = go.Figure()
    rows = []
    for m in store.MODEL_KEYS:
        df, _ = _collect(S["root"], subs, m, cn)
        if df is None:
            continue
        for ok, name in [(True, "correct"), (False, "wrong")]:
            d = df[df["correct"] == ok]
            fig.add_trace(go.Box(y=d["conf"], x=[store.MODEL_SHORT[m]] * len(d), name=name, legendgroup=name, showlegend=(m == "eegnet"), marker_color=viz.GREEN if ok else viz.RED))
            rows.append({"model": store.MODEL_SHORT[m], "outcome": name, "median confidence": round(float(d["conf"].median()), 3) if len(d) else None, "n": len(d)})
    fig.update_layout(boxmode="group", legend=dict(orientation="h", y=1.15))
    fig.update_yaxes(title="top predicted probability", gridcolor="#eef1f6")
    plot(viz._layout(fig, "Confidence when right vs when wrong (test trials)", 340), key="sc_q5")
    table(pd.DataFrame(rows))
    st.caption("If the green boxes sit clearly above the red ones, confidence is informative (a low-confidence flag can be used to *reject* uncertain trials). If they overlap, confidence says little.")


def q6(S):
    root, ref = S["root"], S["ref"]
    ch, pos = ref["channel_names_in_order"], ref["channel_positions_3d_m"]
    st.caption(f"Live experiment on Subject {S['sid']:02d}'s test trials: each electrode is set to 0 in turn and accuracy is re-measured.")
    cols = st.columns(2)
    for col, m in zip(cols, store.MODEL_KEYS):
        with st.spinner(f"{store.MODEL_SHORT[m]} …"):
            base, drop = analysis.channel_ablation(root, m, S["sid"])
        with col:
            plot(viz.topomap(drop, ch, pos, f"{store.MODEL_SHORT[m]} · accuracy drop when removed (baseline {base:.0%})", height=360, colorbar_title="Δ acc"), key=f"sc_q6_{m}")
            top = np.argsort(-drop)[:5]
            st.caption("Most damaging to remove: " + ", ".join(f"{ch[i]} ({drop[i]:+.1%})" for i in top))
    st.caption("Positive = accuracy falls without that electrode. Redundant, neighbouring electrodes can hide each other's importance, so small drops do not mean 'unimportant'.")


def q7(S):
    root = S["root"]
    ser = {}
    for m in store.MODEL_KEYS:
        with st.spinner(f"{store.MODEL_SHORT[m]} …"):
            base, d = analysis.band_ablation(root, m, S["sid"])
        ser[store.MODEL_SHORT[m]] = list(d.values())
    plot(viz.grouped_bars(signals.BAND_NAMES, ser, f"Accuracy drop when a band is filtered out (Subject {S['sid']:02d}, test trials)", colors={"EEGNet": viz.BLUE, "CSP+LDA": viz.ORANGE}, ytitle="Δ accuracy"), key="sc_q7")
    st.caption("Global version of the per-trial band importance. Bands outside the pre-filter (<4 Hz, >38 Hz) were already removed before training.")


def q8(S):
    root = S["root"]
    ser = {}
    for m in store.MODEL_KEYS:
        with st.spinner(f"{store.MODEL_SHORT[m]} …"):
            snrs, acc = analysis.noise_robustness(root, m, S["sid"])
        ser[store.MODEL_SHORT[m]] = acc
    fig = viz.line_multi(snrs, ser, f"Accuracy when band-limited noise is added (Subject {S['sid']:02d})", "signal-to-noise ratio (dB)  ← noisier", "test accuracy",
                         colors={"EEGNet": viz.BLUE, "CSP+LDA": viz.ORANGE}, yrange=[0, 1])
    fig.add_hline(y=0.25, line_dash="dash", line_color="#8a94a6")
    fig.update_xaxes(autorange="reversed")
    plot(fig, key="sc_q8")
    st.caption("Noise is 4–38 Hz band-limited white noise at the stated SNR relative to each trial's power (2 repetitions averaged). A model whose curve falls slowly is more robust.")


def q9(S):
    root, cn = S["root"], S["cn"]
    rows = []
    for m in store.MODEL_KEYS:
        r = analysis.mirror_test(root, m, S["sid"])
        rows.append({"model": store.MODEL_SHORT[m], "left/right trials": r["n"], "accuracy (original)": f"{r['acc_orig']:.0%}",
                     "predictions that swap hand": f"{r['pred_flip_rate']:.0%}", "predictions unchanged": f"{r['pred_unchanged_rate']:.0%}",
                     "accuracy on mirrored trial vs swapped label": f"{r['acc_mirror_vs_swapped_label']:.0%}"})
    table(pd.DataFrame(rows))
    st.caption("Idea: motor control is contralateral, so swapping homologous electrodes (C3↔C4, FC3↔FC4, …) should turn a left-hand trial into something resembling a right-hand trial. "
               "A model that really uses lateralised ERD should therefore swap its answer. Subject-specific electrode impedance, anatomy and strategy make this only approximately true — compare the *two models* rather than expecting 100 %.")


def q10(S):
    root, cn = S["root"], S["cn"]
    comp = store.get_comparison_all(root)
    if comp.empty:
        st.info("No comparison records.")
        return
    scope = _scope(S, "q10")
    c = comp[comp["subject"].isin(scope)].copy()
    c["class"] = [cn[i].replace("_", " ") for i in c["y_true"]]
    c["agree"] = c["agree"].map({True: "models agree", False: "models disagree"})
    fig = go.Figure()
    for a, col in [("models agree", viz.GREEN), ("models disagree", viz.GRAY)]:
        d = c[c["agree"] == a]
        fig.add_trace(go.Box(y=d["channel_pearson"], x=d["class"], name=a, marker_color=col))
    fig.update_layout(boxmode="group", legend=dict(orientation="h", y=1.15))
    fig.update_yaxes(title="correlation of channel importance (EEGNet vs CSP+LDA)", range=[-1, 1], gridcolor="#eef1f6")
    plot(viz._layout(fig, "Do the two models use the same electrodes?", 360), key="sc_q10")
    st.caption(f"Median similarity: agree {c[c['agree'] == 'models agree']['channel_pearson'].median():.2f} · disagree {c[c['agree'] == 'models disagree']['channel_pearson'].median():.2f}.  1 = identical electrode pattern, 0 = unrelated.")


def q11(S):
    root, cn = S["root"], S["cn"]
    acc = store.accuracy_table(root)
    rows = []
    for s in store.list_subjects(root):
        mu = analysis.band_power_contrast(root, s, (8.0, 13.0), (1.5, 3.5))["maps"]
        be = analysis.band_power_contrast(root, s, (13.0, 30.0), (1.5, 3.5))["maps"]
        rows.append({"subject": s, "class_contrast_dB": float((np.abs(mu).mean() + np.abs(be).mean()) / 2)})
    d = acc.merge(pd.DataFrame(rows), on="subject")
    fig = go.Figure()
    for m, col in [("EEGNet test", viz.BLUE), ("CSP+LDA test", viz.ORANGE)]:
        fig.add_trace(go.Scatter(x=d["class_contrast_dB"], y=d[m], mode="markers+text", text=[f"S{int(s):02d}" for s in d["subject"]], textposition="top center", name=m.replace(" test", ""), marker=dict(size=11, color=col)))
    fig.add_hline(y=0.25, line_dash="dash", line_color="#8a94a6")
    fig.update_xaxes(title="average class-specific mu/beta power change (|dB| vs all-class mean, 1.5–3.5 s)", gridcolor="#eef1f6")
    fig.update_yaxes(title="test accuracy", range=[0, 1], gridcolor="#eef1f6")
    plot(viz._layout(fig, "Subjects whose classes differ more in mu/beta power are easier to decode", 400), key="sc_q11")
    if len(d) >= 4:
        r1 = np.corrcoef(d["class_contrast_dB"], d["EEGNet test"])[0, 1]
        r2 = np.corrcoef(d["class_contrast_dB"], d["CSP+LDA test"])[0, 1]
        st.markdown(f"Correlation across {len(d)} subjects: EEGNet **r = {r1:+.2f}**, CSP+LDA **r = {r2:+.2f}**.")
    callout("warn", "Caveat", "With ≤ 9 subjects any correlation is fragile, and this contrast is computed on the training session only. It illustrates the known phenomenon of **BCI inefficiency** — a sizeable minority of people produce weak or unusual motor-imagery rhythms — "
            "not a proven cause.")


FUNCS = {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6, "q7": q7, "q8": q8, "q9": q9, "q10": q10, "q11": q11}


def render(S):
    hero("Science Q&A", "Questions a neuroscientist or BCI engineer would actually ask — answered with the saved explanations and live experiments on the test data.",
         [chip(f"{len(QUESTIONS)} questions", "blue"), chip("explanations from disk", "gray"), chip("live ablation tests", "gray")])
    q = st.selectbox("Choose a question", list(QUESTIONS), format_func=lambda k: f"{list(QUESTIONS).index(k) + 1}. {QUESTIONS[k]}", key="sc_q")
    st.markdown(f"### {QUESTIONS[q]}")
    FUNCS[q](S)

"""Performance: accuracy, confusion, calibration, agreement."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import store, viz
from ..theme import callout, chip, hero, kpi_row, plot, table


def _ci(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / max(n, 1))


def render(S):
    root, sid, cn, xc = S["root"], S["sid"], S["cn"], S["xc"]
    hero("Performance", "Honest numbers: how often each model is right, which classes it confuses, whether its confidence can be trusted — and how often the two models agree.",
         [chip("test = unseen recording day", "blue"), chip("chance = 25 %", "gray")])
    acc = store.accuracy_table(root)
    if acc.empty:
        st.info("No metrics found.")
        return
    mean = acc.mean(numeric_only=True)
    n_test = 288
    kpi_row([("EEGNet · mean test accuracy", f"{mean['EEGNet test']:.1%}", f"κ = {mean['EEGNet kappa']:.2f}"), ("CSP+LDA · mean test accuracy", f"{mean['CSP+LDA test']:.1%}", f"κ = {mean['CSP+LDA kappa']:.2f}"),
             ("Best subject (EEGNet)", f"S{int(acc.loc[acc['EEGNet test'].idxmax(), 'subject']):02d}", f"{acc['EEGNet test'].max():.0%}"), ("Weakest subject (EEGNet)", f"S{int(acc.loc[acc['EEGNet test'].idxmin(), 'subject']):02d}", f"{acc['EEGNet test'].min():.0%}")])
    st.markdown("### Accuracy by subject")
    series = {"EEGNet": acc["EEGNet test"].values, "CSP+LDA": acc["CSP+LDA test"].values}
    fig = viz.grouped_bars([f"S{int(s):02d}" for s in acc["subject"]], series, "Test accuracy per subject (dashed = chance)", colors={"EEGNet": viz.BLUE, "CSP+LDA": viz.ORANGE}, height=330, yrange=[0, 1])
    fig.add_hline(y=0.25, line_dash="dash", line_color="#8a94a6")
    plot(fig, key="perf_acc")
    st.caption(f"With ≈{n_test} test trials the 95 % confidence interval of an accuracy near 60 % is about ±{_ci(0.6, n_test) * 100:.0f} points — small gaps between models or subjects are not significant.")
    show = acc.copy()
    show["subject"] = show["subject"].astype(int)
    table(show.round(3))
    st.download_button("⬇ download table (CSV)", show.to_csv(index=False).encode(), "accuracy_summary.csv", key="perf_dl")

    st.markdown("### Confusion matrices")
    c0, c1 = st.columns([1, 3])
    scope = c0.radio("Scope", [f"Subject {sid:02d}", "All subjects"], key="perf_scope")
    split = c0.radio("Split", ["test", "val"], key="perf_split")
    cols = st.columns(2)
    for col, m in zip(cols, store.MODEL_KEYS):
        df = store.get_all_predictions(root, m) if scope.startswith("All") else store.get_predictions(root, sid, m)
        df = df[df["split"] == split]
        cm = pd.crosstab(df["y_true"], df["y_pred"]).reindex(index=range(4), columns=range(4), fill_value=0).values
        with col:
            plot(viz.confusion_fig(cm, cn, True, f"{store.MODEL_SHORT[m]} · {scope} · {split} · accuracy {np.trace(cm) / max(cm.sum(), 1):.1%}"), key=f"perf_cm_{m}")
            rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
            st.caption("Recall per class: " + " · ".join(f"{cn[i].replace('_', ' ')} {rec[i]:.0%}" for i in range(4)))

    st.markdown("### Can you trust the confidence?  (calibration)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#8a94a6"), name="perfect"))
    for m in store.MODEL_KEYS:
        df = store.get_all_predictions(root, m)
        df = df[df["split"] == "test"]
        conf = df[[f"prob_{c}" for c in cn]].max(1).values
        corr = df["correct"].values.astype(float)
        bins = np.linspace(0.25, 1.0, 7)
        ix = np.digitize(conf, bins) - 1
        xs, ys, ns = [], [], []
        for b in range(len(bins) - 1):
            mk = ix == b
            if mk.sum() >= 8:
                xs.append(conf[mk].mean()); ys.append(corr[mk].mean()); ns.append(int(mk.sum()))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=store.MODEL_SHORT[m], line=dict(color=store.MODEL_COLORS[m]), text=[f"n={n}" for n in ns]))
    fig.update_xaxes(title="stated confidence (top probability)", range=[0.2, 1], gridcolor="#eef1f6")
    fig.update_yaxes(title="actual accuracy", range=[0, 1], gridcolor="#eef1f6")
    plot(viz._layout(fig, "Reliability diagram — all subjects, test trials", 340), key="perf_cal")
    st.caption("Points below the diagonal = over-confident (says 80 %, right less often); above = under-confident. Read any prediction with care when its probability is not decisively high.")

    st.markdown("### How often do the two models agree?")
    pe, pc = store.get_all_predictions(root, "eegnet"), store.get_all_predictions(root, "csp_lda")
    pe, pc = pe[pe["split"] == "test"], pc[pc["split"] == "test"]
    mg = pe.merge(pc, on=["subject", "trial_id"], suffixes=("_e", "_c"))
    agree = (mg["y_pred_e"] == mg["y_pred_c"])
    both_ok = (mg["correct_e"] & mg["correct_c"])
    kpi_row([("Agreement", f"{agree.mean():.0%}", "same predicted class"), ("Both correct", f"{both_ok.mean():.0%}", ""), ("Only EEGNet correct", f"{(mg['correct_e'] & ~mg['correct_c']).mean():.0%}", ""),
             ("Only CSP+LDA correct", f"{(~mg['correct_e'] & mg['correct_c']).mean():.0%}", "")])
    ct = pd.crosstab(mg["y_pred_e"], mg["y_pred_c"]).reindex(index=range(4), columns=range(4), fill_value=0).values
    plot(viz.confusion_fig(ct, cn, False, "Predictions: EEGNet (rows) vs CSP+LDA (columns), counts", 340), key="perf_agree")
    callout("tip", "Why it matters", "When two different methods disagree, the trial is probably hard or noisy. Use *Compare models* on such a trial to see which evidence each relied on.")

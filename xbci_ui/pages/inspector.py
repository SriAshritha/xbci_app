"""Under the hood: browse the real learned weights, layers and training history."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import analysis, signals, store, viz
from ..theme import arch_flow, callout, chip, hero, kpi_row, plot, table


def _eegnet(S):
    root, sid, ref, cn, level = S["root"], S["sid"], S["ref"], S["cn"], S["level"]
    ch, pos = ref["channel_names_in_order"], ref["channel_positions_3d_m"]
    ls = store.get_layer_summary(root, sid)
    lw = store.get_layer_weights(root, sid)
    cfg = store.get_model_config(root, "eegnet", sid)
    stt = store.get_status(root, "eegnet", sid)
    n_par = sum(r["n_params"] for r in ls)
    kpi_row([("Learnable parameters", f"{n_par:,}", "weights the training adjusted"), ("Epochs trained", str(stt.get("epochs_run")), f"best epoch {stt.get('best_epoch')}"),
             ("Best validation accuracy", f"{stt.get('best_val_acc', 0):.0%}", "model-selection split"), ("Checkpoint", stt.get("checkpoint_id", ""), "")])
    arch_flow([("Input", "22 × 513"), ("Temporal conv", "8 × (1×64)"), ("Depthwise spatial", "16 × (22×1)"), ("ELU · pool 4", ""), ("Separable conv", "16 maps"), ("ELU · pool 8", ""), ("Dense", "256 → 4")])
    st.markdown("##### Layer table")
    df = pd.DataFrame(ls)
    df["output_shape"] = df["output_shape"].apply(lambda s: " × ".join(map(str, s)))
    df["parameter shapes"] = df["param_shapes"].apply(lambda d: ", ".join(f"{k}: {tuple(v)}" for k, v in d.items()))
    table(df[["name", "type", "output_shape", "n_params", "parameter shapes"]].rename(columns={"n_params": "parameters"}))

    t1, t2, t3, t4, t5 = st.tabs(["⏱ Temporal filters", "🗺 Spatial filters / patterns", "🔗 Mixing & decision weights", "🧮 Weight browser", "📈 Training"])
    with t1:
        tk, fr, resp = lw["derived__temporal_kernels"], lw["derived__freqs_hz"], lw["derived__temporal_freq_response"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            t = np.arange(tk.shape[1]) / ref["sampling_frequency_hz"] * 1000
            for f in range(tk.shape[0]):
                fig.add_trace(go.Scatter(x=t, y=tk[f], mode="lines", name=f"filter {f + 1}"))
            fig.update_xaxes(title="lag (ms)")
            plot(viz._layout(fig, "Kernels in time (the actual 64 learned numbers per filter)", 320), key="ins_tk")
        with c2:
            fig = go.Figure()
            sel = fr <= 45
            for f in range(resp.shape[0]):
                fig.add_trace(go.Scatter(x=fr[sel], y=resp[f][sel], mode="lines", name=f"filter {f + 1}", showlegend=False))
            viz.add_band_shading(fig, signals.FREQ_BANDS, 45)
            fig.update_xaxes(title="frequency (Hz)")
            plot(viz._layout(fig, "Which rhythms each filter passes (one line per filter)", 320), key="ins_tf")
        peaks = analysis.temporal_peak_freqs(root, sid)
        st.caption("Peak frequency of each filter's response (4–38 Hz): " + ", ".join(f"#{f + 1}: {peaks[f]:.1f} Hz" for f in range(tk.shape[0])))
    with t2:
        mode = st.radio("Show", ["Patterns (scalp topography of the extracted source)", "Filters (raw weights)"], horizontal=True, key="ins_pm")
        vals = list(analysis.eegnet_spatial_patterns(root, sid)) if mode.startswith("Pat") else list(lw["derived__spatial_filters"])
        peaks = analysis.temporal_peak_freqs(root, sid)
        plot(viz.topomap_grid(vals, [f"#{i + 1} · temporal filter {i // 2 + 1} (~{peaks[i // 2]:.0f} Hz)" for i in range(len(vals))], ch, pos, ncols=4, height_per_row=215), key="ins_sp")
        st.caption("Patterns follow Haufe et al. (2014). A red/blue dipole across the central electrodes is typical of a source over one motor cortex.")
    with t3:
        pw = lw["conv_sep_point__weight"][:, :, 0, 0]
        Wc = lw["classifier__weight"]
        F2 = pw.shape[0]
        nb = Wc.shape[1] // F2
        c1, c2 = st.columns([1, 1.6])
        with c1:
            fig = go.Figure(go.Heatmap(z=pw, colorscale="RdBu_r", zmid=0, x=[f"sp #{i + 1}" for i in range(pw.shape[1])], y=[f"map {i + 1}" for i in range(F2)]))
            fig.update_yaxes(autorange="reversed")
            plot(viz._layout(fig, "Pointwise mixing (feature map ← spatial filter)", 400), key="ins_pw")
        with c2:
            fig = make_subplots(rows=1, cols=4, subplot_titles=[c.replace("_", " ") for c in cn], horizontal_spacing=0.03)
            m = np.abs(Wc).max()
            for c in range(4):
                fig.add_trace(go.Heatmap(z=Wc[c].reshape(F2, nb), colorscale="RdBu_r", zmin=-m, zmax=m, showscale=c == 3, colorbar=dict(thickness=8)), row=1, col=c + 1)
                fig.update_yaxes(autorange="reversed", showticklabels=c == 0, row=1, col=c + 1)
                fig.update_xaxes(title="time bin", row=1, col=c + 1)
            plot(viz._layout(fig, "Dense-layer weights W[class, map, time bin]", 400, margin=dict(l=40, r=10, t=60, b=40)), key="ins_wc")
        st.caption("Red = pushes toward the class, blue = pushes away.")
        st.latex(r"z_c=\sum_{m=1}^{16}\sum_{t=1}^{16}W_{c,m,t}\,h_{m,t}+b_c")
        st.caption("Bias b = " + ", ".join(f"{cn[c]}: {lw['classifier__bias'][c]:+.3f}" for c in range(4)))
    with t4:
        keys = [k for k in lw if not k.startswith("derived__") and lw[k].ndim >= 1]
        k = st.selectbox("Choose a stored tensor", keys, key="ins_key")
        a = lw[k]
        st.markdown(f"**shape** {tuple(a.shape)} · mean {a.mean():.4f} · std {a.std():.4f} · min {a.min():.4f} · max {a.max():.4f}")
        flat = a.reshape(a.shape[0], -1) if a.ndim > 1 else a[None]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Heatmap(z=flat, colorscale="RdBu_r", zmid=0))
            plot(viz._layout(fig, "values (rows × flattened columns)", 300), key="ins_wh")
        with c2:
            fig = go.Figure(go.Histogram(x=a.ravel(), nbinsx=40, marker_color=viz.BLUE))
            plot(viz._layout(fig, "distribution", 300), key="ins_wd")
        st.download_button("⬇ download as CSV", pd.DataFrame(flat).to_csv(index=False).encode(), f"{k}.csv", key="ins_dl")
    with t5:
        h = store.get_history(root, sid)
        if h.empty:
            st.info("No history found.")
        else:
            c1, c2 = st.columns(2)
            be = stt.get("best_epoch")
            with c1:
                f = viz.line_multi(h["epoch"] + 1, {"train": h["train_loss"], "validation": h["val_loss"]}, "Loss", "epoch", "cross-entropy", colors={"train": viz.BLUE, "validation": viz.ORANGE})
                if be:
                    f.add_vline(x=be, line_dash="dot", annotation_text="best")
                plot(f, key="ins_h1")
            with c2:
                f = viz.line_multi(h["epoch"] + 1, {"train": h["train_acc"], "validation": h["val_acc"]}, "Accuracy", "epoch", "", colors={"train": viz.BLUE, "validation": viz.ORANGE}, yrange=[0, 1])
                if be:
                    f.add_vline(x=be, line_dash="dot", annotation_text="best")
                plot(f, key="ins_h2")
            st.caption("Training accuracy is measured while dropout is active, so it can be lower than validation early on. A growing gap train ≫ validation means over-fitting.")
        with st.expander("Hyper-parameters & configuration"):
            st.json(cfg, expanded=False)


def _csp(S):
    root, sid, ref, cn = S["root"], S["sid"], S["ref"], S["cn"]
    ch, pos = ref["channel_names_in_order"], ref["channel_positions_3d_m"]
    p = store.get_csp_params(root, sid)
    stt = store.get_status(root, "csp_lda", sid)
    cfg = store.get_model_config(root, "csp_lda", sid)
    kpi_row([("CSP components", str(p["filters"].shape[0]), "spatial filters kept"), ("Learned numbers", f"{p['filters'].size + p['lda_coef'].size + p['lda_intercept'].size:,}", "filters + LDA weights + intercepts"),
             ("Train / val accuracy", f"{stt.get('train_acc', 0):.0%} / {stt.get('val_acc', 0):.0%}", ""), ("Checkpoint", stt.get("checkpoint_id", ""), "")])
    arch_flow([("Input", "22 × 513"), ("CSP  W (8×22)", "virtual channels"), ("log-variance", "8 features"), ("LDA  a (4×8) + b", "class scores"), ("Softmax", "4 probabilities")])
    t1, t2, t3 = st.tabs(["🗺 CSP filters & patterns", "⚖️ LDA weights", "⚙️ Configuration"])
    with t1:
        mode = st.radio("Show", ["Patterns (interpretable topography)", "Filters (raw weights)"], horizontal=True, key="ins_cm")
        arr = p["patterns"] if mode.startswith("Pat") else p["filters"]
        stats = analysis.csp_class_feature_stats(root, sid)
        best = [cn[int(np.argmax(stats["mean"][:, k]))].replace("_", " ") for k in range(arr.shape[0])]
        plot(viz.topomap_grid(list(arr), [f"CSP {k + 1} · loudest for {best[k]}" for k in range(arr.shape[0])], ch, pos, ncols=4, height_per_row=220), key="ins_cspm")
        st.caption("Title = the class whose training trials have the highest mean log-power in that component.")
    with t2:
        coef, b = p["lda_coef"], p["lda_intercept"]
        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig = go.Figure(go.Heatmap(z=coef, x=[f"CSP {k + 1}" for k in range(coef.shape[1])], y=[c.replace("_", " ") for c in cn], colorscale="RdBu_r", zmid=0,
                                       text=np.round(coef, 2), texttemplate="%{text}", colorbar=dict(thickness=10)))
            fig.update_yaxes(autorange="reversed")
            plot(viz._layout(fig, "LDA coefficients a[c, k]  (red = this feature raises the class score)", 340), key="ins_lda")
        with c2:
            plot(viz.bars([c for c in cn], b, "Intercepts b[c]", signed=True, height=340), key="ins_b")
        st.latex(r"z_c=\sum_{k=1}^{8}a_{c,k}f_k+b_c")
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure(go.Heatmap(z=p["lda_covariance"], colorscale="Blues", colorbar=dict(thickness=10)))
            fig.update_yaxes(autorange="reversed")
            plot(viz._layout(fig, "Shared (shrunk) feature covariance used by LDA", 300), key="ins_cov")
        with c4:
            st.markdown("**Class priors**")
            table(pd.DataFrame({"class": [c.replace("_", " ") for c in cn], "prior": np.round(p["lda_priors"], 3)}))
    with t3:
        st.json(cfg, expanded=False)


def render(S):
    hero("Under the hood", "Open the models and look at the real learned numbers: filters, spatial patterns, mixing weights, classifier weights and training curves.",
         [chip(f"Subject {S['sid']:02d}", "blue"), chip("every number is read from disk", "gray")])
    t1, t2 = st.tabs(["🧠 EEGNet", "📐 CSP + LDA"])
    with t1:
        _eegnet(S)
    with t2:
        _csp(S)

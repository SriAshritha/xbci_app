"""Heavy computations (cached): run a model on a signal, explain it, and break the decision down
into the mathematical parts of each model (CSP+LDA features / EEGNet layers)."""
import numpy as np
import streamlit as st
import torch

from . import store

DEFAULT_ECFG = {"ig_steps": 32, "ig_baselines": 3, "batch_size": 8, "n_windows": 8, "occlusion_batch": 128}


# ----------------------------------------------------------------------------- breakdowns
def csp_breakdown(root_str, sid, x, logits):
    p = store.get_csp_params(root_str, sid)
    W, coef, b = p["filters"], p["lda_coef"], p["lda_intercept"]
    s = W @ x.astype(np.float64)                          # (K,T) virtual channels
    f = np.log((s ** 2).mean(1) + 1e-12)                  # (K,) log-variance (band power) features
    scores = coef @ f + b
    return {"kind": "csp_lda", "virtual_channels": s, "features": f, "coef": coef, "intercept": b,
            "contrib": coef * f[None, :], "scores": scores, "patterns": p["patterns"], "filters": W,
            "max_err_vs_model": float(np.abs(scores - logits).max())}


@st.cache_data(show_spinner=False)
def csp_class_feature_stats(root_str, sid):
    """Mean/std of each CSP feature per class on the TRAINING trials (what each class 'looks like')."""
    d = store.get_subject_data(root_str, sid)
    tr = store.get_split(root_str, sid)["train_idx"]
    p = store.get_csp_params(root_str, sid)
    s = np.einsum("kc,nct->nkt", p["filters"].astype(np.float64), d["X"][tr].astype(np.float64))
    f = np.log((s ** 2).mean(-1) + 1e-12)
    y = d["y"][tr]
    n_cls = 4
    mean = np.stack([f[y == c].mean(0) for c in range(n_cls)])
    std = np.stack([f[y == c].std(0) for c in range(n_cls)])
    return {"mean": mean, "std": std}


def eegnet_breakdown(bundle, acts, logits):
    net = bundle["model"].net
    arch = bundle["config"]["architecture"]
    Wc = net.classifier.weight.detach().cpu().numpy()
    bc = net.classifier.bias.detach().cpu().numpy()
    h = np.asarray(acts["flatten"]).reshape(-1)
    F2 = arch["F2"]
    nb = h.size // F2
    contrib = (Wc * h[None, :])
    pw = net.conv_sep_point.weight.detach().cpu().numpy()[:, :, 0, 0]
    dw = net.conv_depthwise.weight.detach().cpu().numpy()[:, 0, :, 0]
    return {"kind": "eegnet", "h": h, "Wc": Wc, "bc": bc, "contrib": contrib.reshape(Wc.shape[0], F2, nb),
            "F2": F2, "n_bins": nb, "pointwise": pw, "depthwise": dw, "D": arch["D"], "F1": arch["F1"],
            "scores": contrib.sum(1) + bc, "max_err_vs_model": float(np.abs(contrib.sum(1) + bc - logits).max())}


def temporal_peak_freqs(root_str, sid, fmin=4, fmax=38):
    lw = store.get_layer_weights(root_str, sid)
    fr, resp = lw["derived__freqs_hz"], lw["derived__temporal_freq_response"]
    sel = (fr >= fmin) & (fr <= fmax)
    return np.array([fr[sel][np.argmax(r[sel])] for r in resp])


# ----------------------------------------------------------------------------- main entry
@st.cache_data(show_spinner=False, max_entries=48)
def analyze(root_str, model_name, sid, x, target_mode="predicted", target_class=None,
            ig_steps=32, ig_baselines=3):
    """Run `model_name` on x (C,T) and explain it. Returns plain numpy/dicts (cache-safe)."""
    xc = store.get_core(root_str)
    ref = store.get_reference(root_str)
    bundle = store.get_bundle(root_str, model_name, sid)
    dev = bundle["device"]
    pool = store.get_pool(root_str, sid)
    X = np.asarray(x, dtype=np.float32)[None]
    logits, probs = xc.predict_bundle(bundle, X)
    pred = int(probs.argmax(1)[0])
    tgt = pred if (target_mode == "predicted" or target_class is None) else int(target_class)
    ecfg = {**DEFAULT_ECFG, "ig_steps": int(ig_steps), "ig_baselines": int(ig_baselines)}
    e = xc.explain_trials(bundle["model"], X, np.array([tgt]), pool, ref["sampling_frequency_hz"], dev, ecfg, seed=0)
    e1 = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == 1 else v) for k, v in e.items()}
    acts, _ = bundle["model"].named_activations(torch.as_tensor(X, device=dev))
    acts = {k: v.detach().cpu().numpy()[0] for k, v in acts.items()}
    bd = csp_breakdown(root_str, sid, X[0], logits[0]) if model_name == "csp_lda" else \
        eegnet_breakdown(bundle, acts, logits[0])
    return {"model": model_name, "subject": sid, "logits": logits[0], "probs": probs[0], "pred": pred, "target": tgt,
            "expl": e1, "acts": acts, "breakdown": bd, "checkpoint_id": bundle["checkpoint_id"]}


@st.cache_data(show_spinner=False, max_entries=64)
def quick_predict(root_str, model_name, sid, x):
    xc = store.get_core(root_str)
    bundle = store.get_bundle(root_str, model_name, sid)
    lg, pr = xc.predict_bundle(bundle, np.asarray(x, dtype=np.float32)[None])
    return lg[0], pr[0]


@st.cache_data(show_spinner=False, max_entries=16)
def predict_many(root_str, model_name, sid, X):
    xc = store.get_core(root_str)
    bundle = store.get_bundle(root_str, model_name, sid)
    lg, pr = xc.predict_bundle(bundle, np.asarray(X, dtype=np.float32), batch_size=64)
    return lg, pr


@st.cache_data(show_spinner=False)
def eegnet_spatial_patterns(root_str, sid):
    """Haufe-transformed spatial PATTERNS for the 16 depthwise filters (interpretable topographies).
    pattern_i = Sigma_f w_i / (w_i' Sigma_f w_i), Sigma_f = channel covariance of pool data filtered by temporal kernel f."""
    from scipy.signal import fftconvolve
    lw = store.get_layer_weights(root_str, sid)
    tk = lw["derived__temporal_kernels"]                    # (F1, L)
    dw = lw["derived__spatial_filters"]                     # (F1*D, C)
    pool = store.get_pool(root_str, sid).astype(np.float64)  # (n, C, T)
    D = dw.shape[0] // tk.shape[0]
    pats = np.zeros_like(dw, dtype=np.float64)
    for f in range(tk.shape[0]):
        xf = fftconvolve(pool, tk[f][None, None, :].astype(np.float64), mode="same", axes=-1)
        Xc = xf.transpose(1, 0, 2).reshape(pool.shape[1], -1)
        Sig = np.cov(Xc)
        for d in range(D):
            i = f * D + d
            w = dw[i].astype(np.float64)
            den = float(w @ Sig @ w) or 1.0
            pats[i] = Sig @ w / den
    return pats


# ----------------------------------------------------------------------------- neuroscience evidence from the data
@st.cache_data(show_spinner=False)
def band_power_contrast(root_str, sid, band=(8.0, 13.0), win=(1.5, 3.5)):
    """Class-specific band power relative to the all-class average (dB), from TRAINING trials.
    Returns curves (4,C,T) and window-averaged maps (4,C). Negative = weaker than average (ERD-like)."""
    from scipy.signal import sosfiltfilt
    from .signals import bandpass_sos
    ref = store.get_reference(root_str)
    fs = ref["sampling_frequency_hz"]
    d = store.get_subject_data(root_str, sid)
    tr = store.get_split(root_str, sid)["train_idx"]
    X, y = d["X"][tr].astype(np.float64), d["y"][tr]
    xf = sosfiltfilt(bandpass_sos(fs, *band), X, axis=-1)
    p = xf ** 2
    k = max(3, int(0.25 * fs))
    ker = np.ones(k) / k
    p = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), -1, p)
    allm = p.mean(0) + 1e-12
    rel = np.stack([10 * np.log10((p[y == c].mean(0) + 1e-12) / allm) for c in range(4)])
    s, e = int(win[0] * fs), int(win[1] * fs)
    return {"curves": rel, "maps": rel[:, :, s:e].mean(-1)}


@st.cache_data(show_spinner=False)
def channel_ablation(root_str, model_name, sid):
    d = store.get_subject_data(root_str, sid)
    te = store.get_split(root_str, sid)["test_idx"]
    X, y = d["X"][te], d["y"][te]
    base = float((predict_many(root_str, model_name, sid, X)[1].argmax(1) == y).mean())
    out = np.zeros(X.shape[1])
    for c in range(X.shape[1]):
        Xc = X.copy()
        Xc[:, c, :] = 0
        out[c] = base - float((predict_many(root_str, model_name, sid, Xc)[1].argmax(1) == y).mean())
    return base, out


@st.cache_data(show_spinner=False)
def band_ablation(root_str, model_name, sid):
    from scipy.signal import sosfiltfilt
    from .signals import bandpass_sos, FREQ_BANDS
    fs = store.get_reference(root_str)["sampling_frequency_hz"]
    d = store.get_subject_data(root_str, sid)
    te = store.get_split(root_str, sid)["test_idx"]
    X, y = d["X"][te].astype(np.float64), d["y"][te]
    base = float((predict_many(root_str, model_name, sid, X.astype(np.float32))[1].argmax(1) == y).mean())
    out = {}
    for nm, (lo, hi) in FREQ_BANDS.items():
        Xr = (X - sosfiltfilt(bandpass_sos(fs, lo, hi), X, axis=-1)).astype(np.float32)
        out[nm] = base - float((predict_many(root_str, model_name, sid, Xr)[1].argmax(1) == y).mean())
    return base, out


@st.cache_data(show_spinner=False)
def noise_robustness(root_str, model_name, sid, snrs=(30, 20, 10, 5, 0), n_rep=2):
    from .signals import apply_time_edits
    ref = store.get_reference(root_str)
    ch, fs = ref["channel_names_in_order"], ref["sampling_frequency_hz"]
    d = store.get_subject_data(root_str, sid)
    te = store.get_split(root_str, sid)["test_idx"]
    X, y = d["X"][te], d["y"][te]
    accs = []
    for snr in snrs:
        a = []
        for r in range(n_rep):
            Xn = np.stack([apply_time_edits(x, fs, ch, snr_db=snr, seed=1000 * r + i) for i, x in enumerate(X)])
            a.append(float((predict_many(root_str, model_name, sid, Xn)[1].argmax(1) == y).mean()))
        accs.append(float(np.mean(a)))
    return list(snrs), accs


@st.cache_data(show_spinner=False)
def mirror_test(root_str, model_name, sid):
    from .signals import mirror_channels
    ch = store.get_reference(root_str)["channel_names_in_order"]
    cn = store.get_core(root_str).CLASS_NAMES
    d = store.get_subject_data(root_str, sid)
    te = store.get_split(root_str, sid)["test_idx"]
    X, y = d["X"][te], d["y"][te]
    sel = np.isin(y, [cn.index("left_hand"), cn.index("right_hand")])
    Xs, ys = X[sel], y[sel]
    Xm = np.stack([mirror_channels(x, ch) for x in Xs])
    p0 = predict_many(root_str, model_name, sid, Xs)[1].argmax(1)
    p1 = predict_many(root_str, model_name, sid, Xm)[1].argmax(1)
    L, R = cn.index("left_hand"), cn.index("right_hand")
    swap = lambda p: np.where(p == L, R, np.where(p == R, L, p))
    return {"n": int(sel.sum()), "acc_orig": float((p0 == ys).mean()), "acc_mirror_vs_swapped_label": float((p1 == swap(ys)).mean()),
            "pred_flip_rate": float((p1 == swap(p0)).mean()), "pred_unchanged_rate": float((p1 == p0).mean())}

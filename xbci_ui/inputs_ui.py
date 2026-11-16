"""Reusable 'give the model an EEG signal' widget: dataset trial / upload / synthetic + edits in
the time, frequency and time-frequency domains."""
import io

import numpy as np
import pandas as pd
import streamlit as st

from . import signals, store, viz
from .theme import callout, plot

SESSION_LABELS = {"test": "Test session (unseen by the models)", "val": "Validation split", "train": "Training split"}


def _rand_cb(k, n):
    st.session_state[k] = int(np.random.randint(1, n + 1))


def _dataset_trial(root, sid, key, cn):
    d = store.get_subject_data(root, sid)
    sp = store.get_split(root, sid)
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        part = st.selectbox("Which part of the recordings?", ["test", "val", "train"], format_func=lambda s: SESSION_LABELS[s], key=f"{key}_part")
    idx_all = sp[f"{part}_idx"]
    with c2:
        cls = st.selectbox("Class filter", ["any"] + [c.replace("_", " ") for c in cn], key=f"{key}_cls")
    if cls != "any":
        idx_all = idx_all[d["y"][idx_all] == cn.index(cls.replace(" ", "_"))]
    n = len(idx_all)
    kt = f"{key}_trial"
    if st.session_state.get(kt, 1) > n:
        st.session_state[kt] = 1
    with c3:
        st.slider("Trial #", 1, max(n, 2), 1, key=kt, disabled=n < 2)
        st.button("🎲 random trial", key=f"{key}_rnd", on_click=_rand_cb, args=(kt, n))
    pos = min(st.session_state.get(kt, 1), n) - 1
    gi = int(idx_all[pos])
    x = d["X"][gi].astype(np.float32)
    true = cn[int(d["y"][gi])]
    row = d["meta"].iloc[gi]
    desc = f"Subject {sid:02d} · {part} · trial_id {int(row['trial_id'])} · session {row['session']}"
    return x, true, desc, gi


def _upload(root, sid, key, ref):
    ch = ref["channel_names_in_order"]
    T = store.get_subject_data(root, sid)["X"].shape[2]
    st.markdown("Upload a **channels × samples** array (`.npy`, `.npz` or `.csv`) with the 22 electrodes **in this order**:")
    st.code(", ".join(ch), language="text")
    f = st.file_uploader("EEG file", type=["npy", "npz", "csv", "txt"], key=f"{key}_file")
    c1, c2, c3 = st.columns(3)
    fs_in = c1.number_input("Sampling rate of your file (Hz)", 1.0, 5000.0, 128.0, key=f"{key}_fs")
    unit = c2.selectbox("Unit", ["uV", "mV", "V"], key=f"{key}_unit")
    pre = c3.checkbox("Already band-passed 4–38 Hz and resampled to 128 Hz", value=True, key=f"{key}_pre")
    with st.expander("Need an example file?"):
        d = store.get_subject_data(root, sid)
        buf = io.BytesIO()
        np.save(buf, d["X"][0])
        st.download_button("⬇ example_trial.npy (22 × 513, µV)", buf.getvalue(), "example_trial.npy", key=f"{key}_ex")
    if f is None:
        return None, None, "no file yet", None
    try:
        arr = signals.orient_channels_first(signals.parse_upload(f.name, f.getvalue()), len(ch))
        x, notes = signals.preprocess_uploaded(arr, fs_in, unit, pre, 128.0, T)
    except Exception as e:  # noqa
        st.error(f"Could not use this file: {e}")
        return None, None, "invalid file", None
    if notes:
        st.caption("Applied: " + " · ".join(notes))
    return x, None, f"uploaded file «{f.name}»", None


PRESETS = {"Left motor (C3 area)": ["FC3", "C5", "C3", "C1", "CP3"], "Right motor (C4 area)": ["FC4", "C6", "C4", "C2", "CP4"],
           "Mid-line (Cz area)": ["FCz", "Cz", "CPz"], "Frontal mid-line": ["Fz", "FC1", "FC2", "FCz"], "Custom": []}


def _synthetic(root, sid, key, ref):
    ch = ref["channel_names_in_order"]
    fs = ref["sampling_frequency_hz"]
    T = store.get_subject_data(root, sid)["X"].shape[2]
    bs = store.get_baseline_stats(root, sid)
    ch_std = bs["channel_std_uV"] if bs and "channel_std_uV" in bs else np.full(len(ch), 6.0)
    st.markdown("Build a signal from **parameters**: background brain-like noise + up to 3 rhythmic components. "
                "Try e.g. a strong 10 Hz rhythm on the left motor area.")
    noise = st.slider("Background noise level (× typical amplitude of this subject)", 0.2, 2.0, 1.0, 0.1, key=f"{key}_noise")
    seed = st.number_input("Random seed", 0, 9999, 0, key=f"{key}_seed")
    comps = []
    cols = st.columns(3)
    defaults = [(True, 10.0, 8.0, "Left motor (C3 area)"), (False, 20.0, 4.0, "Right motor (C4 area)"), (False, 12.0, 6.0, "Mid-line (Cz area)")]
    for i, col in enumerate(cols):
        with col:
            on_d, f_d, a_d, p_d = defaults[i]
            with st.container(border=True):
                on = st.checkbox(f"Component {i + 1}", value=on_d, key=f"{key}_c{i}_on")
                f = st.slider("Frequency (Hz)", 4.0, 38.0, f_d, 0.5, key=f"{key}_c{i}_f", disabled=not on)
                a = st.slider("Amplitude (µV)", 0.0, 30.0, a_d, 0.5, key=f"{key}_c{i}_a", disabled=not on)
                t0, t1 = st.slider("Active from–to (s)", 0.0, T / fs, (0.0, T / fs), 0.1, key=f"{key}_c{i}_t", disabled=not on)
                pre = st.selectbox("Electrodes", list(PRESETS), index=list(PRESETS).index(p_d), key=f"{key}_c{i}_p", disabled=not on)
                chans = PRESETS[pre] if pre != "Custom" else st.multiselect("Choose electrodes", ch, default=["Cz"], key=f"{key}_c{i}_ch", disabled=not on)
                if on and chans:
                    comps.append({"freq": f, "amp": a, "t0": t0, "t1": t1, "channels": chans})
    x = signals.synthesize(ch, fs, T, ch_std, comps, noise, int(seed))
    desc = "synthetic: " + (" + ".join(f"{c['freq']:g} Hz×{c['amp']:g} µV on {len(c['channels'])} ch" for c in comps) or "noise only")
    return x, None, desc, None


def _edits(x, ref, key):
    """Returns (x_edited, list_of_edit_descriptions)."""
    ch = ref["channel_names_in_order"]
    fs = ref["sampling_frequency_hz"]
    T = x.shape[1]
    notes = []
    y = x.copy()
    with st.expander("✏️ Modify the signal in the time / frequency / time-frequency domain (optional)", expanded=False):
        t_time, t_freq, t_tf = st.tabs(["⏱ Time domain", "📶 Frequency domain", "🔥 Time–frequency"])
        with t_time:
            c1, c2, c3 = st.columns(3)
            scale = c1.slider("Amplitude gain (×)", 0.2, 3.0, 1.0, 0.05, key=f"{key}_gain")
            shift = c2.slider("Time shift (s, + = later)", -1.0, 1.0, 0.0, 0.05, key=f"{key}_shift")
            snr_on = c3.checkbox("Add noise", key=f"{key}_snr_on")
            snr = c3.slider("Signal-to-noise ratio (dB)", -5, 30, 10, key=f"{key}_snr", disabled=not snr_on)
            c4, c5 = st.columns(2)
            mirror = c4.checkbox("🪞 Mirror left ↔ right hemisphere (swap C3↔C4, FC3↔FC4 …)", key=f"{key}_mirror",
                                 help="A scientific control: mirroring should flip left-hand ↔ right-hand evidence.")
            drop = c5.multiselect("Silence electrodes (set to 0)", ch, key=f"{key}_drop")
            win_on = st.checkbox("Keep only a time window (zero elsewhere)", key=f"{key}_win_on")
            win = st.slider("Window (s from cue)", 0.0, T / fs, (1.0, 3.5), 0.1, key=f"{key}_win", disabled=not win_on)
            y = signals.apply_time_edits(y, fs, ch, scale, shift, snr if snr_on else None, drop, mirror, win if win_on else None, seed=1)
            for cond, txt in [(scale != 1, f"gain ×{scale:g}"), (shift != 0, f"shift {shift:+g}s"), (snr_on, f"noise SNR {snr} dB"),
                              (mirror, "hemispheres mirrored"), (bool(drop), f"silenced {len(drop)} ch"), (win_on, f"window {win[0]:g}–{win[1]:g}s")]:
                if cond:
                    notes.append(txt)
        with t_freq:
            st.caption("Boost or attenuate each band (in dB; +6 dB ≈ 2× amplitude). Phase is preserved.")
            cols = st.columns(5)
            gains = {}
            for col, (nm, (lo, hi)) in zip(cols, signals.FREQ_BANDS.items()):
                gains[nm] = col.slider(f"{nm.replace('_', ' ')}\n{lo}–{hi} Hz", -20, 20, 0, key=f"{key}_g_{nm}")
            cx = st.checkbox("Custom band", key=f"{key}_cx_on")
            extra = None
            if cx:
                lo, hi = st.slider("Custom band (Hz)", 4.0, 38.0, (9.0, 12.0), 0.5, key=f"{key}_cx_r")
                g = st.slider("Custom gain (dB)", -30, 30, -12, key=f"{key}_cx_g")
                extra = (lo, hi, g)
            if any(gains.values()) or extra:
                y, _, _ = signals.apply_freq_gains(y, fs, gains, extra)
                notes.append("spectrum edited: " + ", ".join(f"{k} {v:+d} dB" for k, v in gains.items() if v) + (f", custom {extra}" if extra else ""))
        with t_tf:
            st.caption("Edit the gain (dB) of each frequency band inside each time window. Example: set **mu** to −12 dB in windows 3–6 to simulate a stronger desynchronisation (ERD).")
            nw = 8
            edges = np.linspace(0, T / fs, nw + 1)
            cols_n = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}s" for i in range(nw)]
            df0 = pd.DataFrame(np.zeros((len(signals.BAND_NAMES), nw)), index=[b.replace("_", " ") for b in signals.BAND_NAMES], columns=cols_n)
            ed = st.data_editor(df0, key=f"{key}_tf_ed", column_config={c: st.column_config.NumberColumn(min_value=-30, max_value=30, step=1) for c in cols_n})
            G = ed.values.astype(float)
            if np.any(G):
                y = signals.apply_tf_gains(y, fs, G)
                notes.append("time–frequency gains edited")
    return y.astype(np.float32), notes


def render_signal_builder(root, sid, ref, key, cn):
    """Returns dict(x, x_base, true_label, desc, edited, notes) or None."""
    src = st.radio("Where does the EEG come from?", ["Recorded trial (dataset)", "Upload my own EEG", "Synthetic generator"],
                   horizontal=True, key=f"{key}_src")
    if src.startswith("Recorded"):
        x, true, desc, gi = _dataset_trial(root, sid, key, cn)
    elif src.startswith("Upload"):
        x, true, desc, gi = _upload(root, sid, key, ref)
    else:
        x, true, desc, gi = _synthetic(root, sid, key, ref)
    if x is None:
        callout("tip", "Waiting for input", "Provide a signal above to continue.")
        return None
    y, notes = _edits(x, ref, key)
    edited = bool(notes) or src.startswith("Synth")
    fs = ref["sampling_frequency_hz"]
    ch = ref["channel_names_in_order"]
    if notes:
        c1, c2 = st.columns(2)
        with c1:
            plot(viz.signal_figure(x, ch, fs, "Original", height=320), key=f"{key}_pv_orig")
        with c2:
            plot(viz.signal_figure(y, ch, fs, "After your edits", height=320), key=f"{key}_pv_edit")
        plot(viz.spectrum_figure([("original", x), ("edited", y)], fs, title="Spectrum: original vs edited", bands=signals.FREQ_BANDS), key=f"{key}_pv_spec")
    return {"x": y, "x_base": x, "true": true, "desc": desc, "edited": edited, "notes": notes, "source": src, "gi": gi}

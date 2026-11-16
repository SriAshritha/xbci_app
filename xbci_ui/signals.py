"""Signal construction/editing in the time, frequency and time-frequency domains (pure numpy/scipy)."""
import io
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

FREQ_BANDS = {"theta": (4, 8), "mu": (8, 13), "beta_low": (13, 20), "beta_high": (20, 30), "gamma_low": (30, 38)}
BAND_NAMES = list(FREQ_BANDS)

MIRROR_PAIRS = [("FC3", "FC4"), ("FC1", "FC2"), ("C5", "C6"), ("C3", "C4"), ("C1", "C2"),
                ("CP3", "CP4"), ("CP1", "CP2"), ("P1", "P2")]
LEFT_CH = [a for a, _ in MIRROR_PAIRS]
RIGHT_CH = [b for _, b in MIRROR_PAIRS]
MIDLINE_CH = ["Fz", "FCz", "Cz", "CPz", "Pz", "POz"]


def bandpass_sos(fs, lo, hi, order=4):
    nyq = fs / 2.0
    return butter(order, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="bandpass", output="sos")


def fit_length(x, T):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] >= T:
        return x[:, :T].copy()
    out = np.zeros((x.shape[0], T), dtype=np.float32)
    out[:, :x.shape[1]] = x
    return out


# ----------------------------------------------------------------------------- upload
def parse_upload(name: str, data: bytes) -> np.ndarray:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "npy":
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    elif ext == "npz":
        z = np.load(io.BytesIO(data), allow_pickle=False)
        key = next((k for k in ["x", "X", "data", "eeg"] if k in z.files), z.files[0])
        arr = z[key]
    elif ext in ("csv", "txt", "tsv"):
        import pandas as pd
        arr = pd.read_csv(io.BytesIO(data), header=None, sep=None, engine="python").values
        if not np.issubdtype(arr.dtype, np.number):       # header row present
            arr = pd.read_csv(io.BytesIO(data), sep=None, engine="python").select_dtypes("number").values
    else:
        raise ValueError("Unsupported file type. Use .npy, .npz or .csv")
    arr = np.squeeze(np.asarray(arr, dtype=np.float64))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array (channels x samples); got shape {arr.shape}")
    return arr


def orient_channels_first(arr, n_ch):
    if arr.shape[0] == n_ch:
        return arr
    if arr.shape[1] == n_ch:
        return arr.T
    raise ValueError(f"Neither dimension equals the expected {n_ch} channels (shape {arr.shape}).")


def preprocess_uploaded(x, fs_in, unit="uV", already_preprocessed=True, target_fs=128.0, T=513, band=(4.0, 38.0)):
    x = np.asarray(x, dtype=np.float64)
    x = x * {"V": 1e6, "mV": 1e3, "uV": 1.0}[unit]
    notes = []
    if not already_preprocessed:
        x = x - x.mean(axis=1, keepdims=True)
        x = sosfiltfilt(bandpass_sos(fs_in, *band), x, axis=-1)
        notes.append(f"band-pass {band[0]}-{band[1]} Hz (Butterworth, zero-phase)")
        if abs(fs_in - target_fs) > 1e-6:
            from fractions import Fraction
            fr = Fraction(target_fs / fs_in).limit_denominator(1000)
            x = resample_poly(x, fr.numerator, fr.denominator, axis=-1)
            notes.append(f"resampled {fs_in:g} -> {target_fs:g} Hz")
    elif abs(fs_in - target_fs) > 1e-6:
        notes.append(f"WARNING: you said the data is preprocessed but fs={fs_in:g} Hz != {target_fs:g} Hz")
    n0 = x.shape[1]
    x = fit_length(x, T)
    if n0 != T:
        notes.append(f"length {n0} -> {T} samples ({'cropped' if n0 > T else 'zero-padded'})")
    return x.astype(np.float32), notes


# ----------------------------------------------------------------------------- time-domain edits
def mirror_channels(x, ch_names):
    y = x.copy()
    for a, b in MIRROR_PAIRS:
        if a in ch_names and b in ch_names:
            ia, ib = ch_names.index(a), ch_names.index(b)
            y[ia], y[ib] = x[ib].copy(), x[ia].copy()
    return y


def apply_time_edits(x, fs, ch_names, scale=1.0, shift_s=0.0, snr_db=None, drop_channels=(), mirror=False,
                     window=None, seed=0):
    y = np.asarray(x, dtype=np.float32).copy()
    T = y.shape[1]
    if mirror:
        y = mirror_channels(y, ch_names)
    for c in drop_channels:
        y[ch_names.index(c)] = 0.0
    y = y * float(scale)
    sh = int(round(shift_s * fs))
    if sh != 0:
        z = np.zeros_like(y)
        if sh > 0:
            z[:, sh:] = y[:, :T - sh]
        else:
            z[:, :T + sh] = y[:, -sh:]
        y = z
    if window is not None:
        s, e = int(round(window[0] * fs)), int(round(window[1] * fs))
        m = np.zeros(T, dtype=np.float32)
        m[max(s, 0):min(e, T)] = 1.0
        k = max(3, int(0.05 * fs))
        m = np.convolve(m, np.hanning(k) / np.hanning(k).sum(), mode="same")
        y = y * m[None]
    if snr_db is not None:
        rng = np.random.RandomState(seed)
        p = float(np.mean(y ** 2)) or 1.0
        noise = sosfiltfilt(bandpass_sos(fs, 4, 38), rng.randn(*y.shape), axis=-1)
        noise *= np.sqrt(p / (10 ** (snr_db / 10.0)) / (np.mean(noise ** 2) or 1.0))
        y = y + noise.astype(np.float32)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- frequency-domain edits
def apply_freq_gains(x, fs, gains_db: dict, extra=None, smooth_hz=1.0):
    """Multiply the spectrum by a piece-wise gain curve (dB per band, optional custom band). Phase is kept."""
    T = x.shape[1]
    F = np.fft.rfftfreq(T, 1.0 / fs)
    logH = np.zeros_like(F)
    for name, (lo, hi) in FREQ_BANDS.items():
        g = float(gains_db.get(name, 0.0))
        if g:
            logH[(F >= lo) & (F < hi)] += g * np.log(10) / 20
    if extra is not None:
        lo, hi, g = extra
        logH[(F >= lo) & (F < hi)] += float(g) * np.log(10) / 20
    df = F[1] - F[0]
    k = max(3, int(round(smooth_hz / df)) | 1)
    ker = np.hanning(k + 2)[1:-1]
    ker /= ker.sum()
    logH = np.convolve(np.pad(logH, (k, k), mode="edge"), ker, mode="same")[k:-k]
    H = np.exp(logH)
    Y = np.fft.irfft(np.fft.rfft(x, axis=1) * H[None], n=T, axis=1)
    return Y.astype(np.float32), F, H


# ----------------------------------------------------------------------------- time-frequency edits
def apply_tf_gains(x, fs, gain_db, taper_s=0.1):
    """gain_db: (n_bands, n_windows) in dB. Adds (g-1) * band-limited signal inside each window."""
    gain_db = np.asarray(gain_db, float)
    nb, nw = gain_db.shape
    T = x.shape[1]
    edges = np.linspace(0, T, nw + 1).astype(int)
    y = x.astype(np.float64).copy()
    k = max(3, int(taper_s * fs) | 1)
    ker = np.hanning(k + 2)[1:-1]
    ker /= ker.sum()
    for bi, (lo, hi) in enumerate(FREQ_BANDS.values()):
        if not np.any(gain_db[bi]):
            continue
        xb = sosfiltfilt(bandpass_sos(fs, lo, hi), x.astype(np.float64), axis=-1)
        m = np.zeros(T)
        for wi in range(nw):
            m[edges[wi]:edges[wi + 1]] = 10 ** (gain_db[bi, wi] / 20.0) - 1.0
        m = np.convolve(np.pad(m, (k, k), mode="edge"), ker, mode="same")[k:-k]
        y += xb * m[None]
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- synthetic generator
def synthesize(ch_names, fs, T, ch_std, components, noise_scale=1.0, seed=0):
    """Band-limited noise (4-38 Hz) with per-channel scale + user-defined oscillatory components.
    components: list of dict(freq, amp, t0, t1, channels)."""
    rng = np.random.RandomState(seed)
    noise = sosfiltfilt(bandpass_sos(fs, 4, 38), rng.randn(len(ch_names), T), axis=-1)
    noise /= noise.std(axis=1, keepdims=True) + 1e-9
    x = noise * (np.asarray(ch_std, float)[:, None] * float(noise_scale))
    t = np.arange(T) / fs
    for c in components:
        s, e = int(c["t0"] * fs), int(min(c["t1"], T / fs) * fs)
        env = np.zeros(T)
        env[s:e] = 1.0
        k = max(3, int(0.1 * fs) | 1)
        env = np.convolve(env, np.hanning(k) / np.hanning(k).sum(), mode="same")
        for ch in c["channels"]:
            i = ch_names.index(ch)
            x[i] += c["amp"] * env * np.sin(2 * np.pi * c["freq"] * t + rng.rand() * 2 * np.pi)
    return x.astype(np.float32)


# ----------------------------------------------------------------------------- misc
def lateralization_index(channel_importance, ch_names):
    """(R - L) / (R + L) over homologous left/right channel pairs. +1 = all evidence on the right hemisphere."""
    imp = np.asarray(channel_importance, float)
    L = sum(imp[ch_names.index(c)] for c in LEFT_CH if c in ch_names)
    R = sum(imp[ch_names.index(c)] for c in RIGHT_CH if c in ch_names)
    return float((R - L) / (R + L)) if (R + L) > 0 else float("nan")


def top_channels(imp, ch_names, k=3):
    idx = np.argsort(-np.asarray(imp))[:k]
    return [ch_names[i] for i in idx]


def region_of(ch):
    if ch in MIDLINE_CH:
        side = "midline"
    elif ch in LEFT_CH:
        side = "left"
    elif ch in RIGHT_CH:
        side = "right"
    else:
        side = "?"
    row = "frontal-central" if ch.startswith("F") else ("parietal-central" if ch.startswith("CP") or ch.startswith("P") else "central")
    return side, row

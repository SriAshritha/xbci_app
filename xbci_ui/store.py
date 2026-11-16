"""Artifact store: finds the artifact folder, imports the *same* xbci_core that trained the
models, and exposes cached loaders (models, data, predictions, explanations)."""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
ENV_VAR = "XBCI_ARTIFACT_ROOT"
ROOT_NAME = "XBCI_BNCI2014_001"

MODEL_LABELS = {"eegnet": "EEGNet (deep learning)", "csp_lda": "CSP + LDA (classical)"}
MODEL_SHORT = {"eegnet": "EEGNet", "csp_lda": "CSP+LDA"}
MODEL_COLORS = {"eegnet": "#1a56ff", "csp_lda": "#f08a24"}
MODEL_KEYS = ["eegnet", "csp_lda"]


# ----------------------------------------------------------------------------- locating
def is_valid_root(p) -> bool:
    p = Path(p)
    return (p / "reference" / "reference.json").exists() and (p / "models").is_dir()


def _normalize_candidate(p: Path):
    """Accept the root itself, its parent (zip extracted one level up) or a doubled folder."""
    for q in [p, p / ROOT_NAME, p / ROOT_NAME / ROOT_NAME]:
        if is_valid_root(q):
            return q
    return None


def auto_find_root():
    cands = []
    if os.environ.get(ENV_VAR):
        cands.append(Path(os.environ[ENV_VAR]))
    cwd = Path.cwd()
    for base in [APP_DIR, APP_DIR.parent, cwd, cwd.parent, Path.home() / "Downloads", Path.home() / "Documents",
                 Path.home() / "Desktop", Path.home()]:
        cands += [base, base / ROOT_NAME, base / "artifacts", base / "demo_artifacts"]
    for c in cands:
        try:
            r = _normalize_candidate(c)
        except Exception:
            r = None
        if r is not None:
            return r
    return None


def resolve_root(user_path: str | None):
    if user_path:
        r = _normalize_candidate(Path(user_path.strip().strip('"')))
        if r is not None:
            return r
        return None
    return auto_find_root()


# ----------------------------------------------------------------------------- core import
@st.cache_resource(show_spinner=False)
def get_core(root_str: str):
    """Import xbci_core from the artifact folder (exact training code) or fall back to the app copy."""
    root = Path(root_str)
    for name in ["xbci_core"]:
        sys.modules.pop(name, None)
    code_dir = root / "code"
    added = []
    try:
        if (code_dir / "xbci_core.py").exists():
            sys.path.insert(0, str(code_dir))
            added.append(str(code_dir))
        else:
            sys.path.insert(0, str(APP_DIR))
            added.append(str(APP_DIR))
        mod = importlib.import_module("xbci_core")
    finally:
        for a in added:
            if a in sys.path:
                sys.path.remove(a)
    return mod


@st.cache_resource(show_spinner=False)
def get_device_str():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_paths(root_str: str):
    return get_core(root_str).ArtifactPaths(root_str)


# ----------------------------------------------------------------------------- simple json / reference
@st.cache_data(show_spinner=False)
def get_reference(root_str: str) -> dict:
    return get_core(root_str).load_json(Path(root_str) / "reference" / "reference.json")


@st.cache_data(show_spinner=False)
def get_versions(root_str: str) -> dict:
    p = Path(root_str) / "env" / "versions.json"
    return get_core(root_str).load_json(p) if p.exists() else {}


@st.cache_data(show_spinner=False)
def get_experiment_config(root_str: str) -> dict:
    p = Path(root_str) / "env" / "experiment_config.json"
    return get_core(root_str).load_json(p) if p.exists() else {}


@st.cache_data(show_spinner=False)
def get_registry(root_str: str) -> dict:
    p = Path(root_str) / "registry.json"
    return get_core(root_str).load_json(p) if p.exists() else {}


@st.cache_data(show_spinner=False)
def list_subjects(root_str: str) -> list:
    """Subjects for which BOTH models are complete."""
    xc = get_core(root_str)
    paths = xc.ArtifactPaths(root_str)
    out = []
    for sid in range(1, 30):
        ok = True
        for m in MODEL_KEYS:
            p = paths.model_dir(m, sid) / "status.json"
            if not p.exists() or xc.load_json(p).get("status") != "complete":
                ok = False
        if ok:
            out.append(sid)
    return out


# ----------------------------------------------------------------------------- models & data
@st.cache_resource(show_spinner=False)
def get_bundle(root_str: str, model_name: str, sid: int):
    xc = get_core(root_str)
    return xc.load_bundle(xc.ArtifactPaths(root_str), model_name, sid, device=xc.get_device())


@st.cache_resource(show_spinner=False)
def get_subject_data(root_str: str, sid: int) -> dict:
    root = Path(root_str)
    z = np.load(root / "data_processed" / f"subject_{sid:02d}.npz", allow_pickle=False)
    meta = pd.read_csv(root / "data_processed" / f"subject_{sid:02d}_meta.csv")
    info = get_core(root_str).load_json(root / "data_processed" / f"subject_{sid:02d}_info.json")
    return {"X": z["X"], "y": z["y"], "meta": meta, "info": info}


@st.cache_data(show_spinner=False)
def get_split(root_str: str, sid: int) -> dict:
    z = np.load(Path(root_str) / "experiments" / f"subject_{sid:02d}" / "split_indices.npz")
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def get_pool(root_str: str, sid: int) -> np.ndarray:
    return np.load(Path(root_str) / "experiments" / f"subject_{sid:02d}" / "baseline_pool.npz")["X"]


@st.cache_data(show_spinner=False)
def get_predictions(root_str: str, sid: int, model_name: str) -> pd.DataFrame:
    return pd.read_csv(Path(root_str) / "experiments" / f"subject_{sid:02d}" / f"predictions_{model_name}.csv")


@st.cache_data(show_spinner=False)
def get_all_predictions(root_str: str, model_name: str) -> pd.DataFrame:
    dfs = [get_predictions(root_str, s, model_name) for s in list_subjects(root_str)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_metrics(root_str: str, sid: int) -> dict:
    return get_core(root_str).load_json(Path(root_str) / "experiments" / f"subject_{sid:02d}" / "metrics.json")


@st.cache_data(show_spinner=False)
def get_history(root_str: str, sid: int) -> pd.DataFrame:
    p = Path(root_str) / "models" / "eegnet" / f"subject_{sid:02d}" / "history.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_status(root_str: str, model_name: str, sid: int) -> dict:
    return get_core(root_str).load_json(Path(root_str) / "models" / model_name / f"subject_{sid:02d}" / "status.json")


@st.cache_data(show_spinner=False)
def get_model_config(root_str: str, model_name: str, sid: int) -> dict:
    return get_core(root_str).load_json(Path(root_str) / "models" / model_name / f"subject_{sid:02d}" / "config.json")


@st.cache_data(show_spinner=False)
def get_layer_summary(root_str: str, sid: int) -> list:
    return get_core(root_str).load_json(Path(root_str) / "models" / "eegnet" / f"subject_{sid:02d}" / "layer_summary.json")


@st.cache_data(show_spinner=False)
def get_layer_weights(root_str: str, sid: int) -> dict:
    z = np.load(Path(root_str) / "models" / "eegnet" / f"subject_{sid:02d}" / "layer_weights.npz")
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def get_csp_params(root_str: str, sid: int) -> dict:
    z = np.load(Path(root_str) / "models" / "csp_lda" / f"subject_{sid:02d}" / "csp_lda_params.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def get_explanations(root_str: str, sid: int, model_name: str) -> dict:
    p = Path(root_str) / "experiments" / f"subject_{sid:02d}" / f"explanations_{model_name}.npz"
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def get_comparison_all(root_str: str) -> pd.DataFrame:
    dfs = []
    for s in list_subjects(root_str):
        p = Path(root_str) / "experiments" / f"subject_{s:02d}" / "comparison.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_baseline_stats(root_str: str, sid: int) -> dict:
    p = Path(root_str) / "reference" / f"baseline_stats_subject_{sid:02d}.npz"
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def accuracy_table(root_str: str) -> pd.DataFrame:
    rows = []
    for s in list_subjects(root_str):
        try:
            m = get_metrics(root_str, s)
        except Exception:
            continue
        rows.append({"subject": s,
                     "EEGNet val": m["eegnet_val"]["accuracy"], "EEGNet test": m["eegnet_test"]["accuracy"],
                     "EEGNet kappa": m["eegnet_test"]["kappa"],
                     "CSP+LDA val": m["csp_lda_val"]["accuracy"], "CSP+LDA test": m["csp_lda_test"]["accuracy"],
                     "CSP+LDA kappa": m["csp_lda_test"]["kappa"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_expl_small(root_str: str, sid: int, model_name: str) -> dict:
    """Explanation summaries WITHOUT the big per-sample attribution tensor (npz is lazy-loaded)."""
    p = Path(root_str) / "experiments" / f"subject_{sid:02d}" / f"explanations_{model_name}.npz"
    z = np.load(p, allow_pickle=False)
    keys = ["trial_ids", "target", "probs", "channel_importance", "time_importance", "window_importance",
            "channel_signed", "freq_importance", "tf_importance"]
    return {k: z[k] for k in keys if k in z.files}

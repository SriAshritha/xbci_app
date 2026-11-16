"""About / reproducibility: versions, configs, checkpoints, integrity check."""
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from .. import store
from ..theme import callout, chip, hero, table


def render(S):
    root, sid, xc = S["root"], S["sid"], S["xc"]
    hero("About & reproducibility", "Every number in this app comes from files on disk — no hidden state. Here is exactly what was used.",
         [chip("versions", "gray"), chip("hyper-parameters", "gray"), chip("checkpoint IDs", "gray"), chip("integrity check", "gray")])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Software used for training")
        v = store.get_versions(root)
        if v:
            table(pd.DataFrame([{"package": k, "version": x} for k, x in v.get("packages", {}).items() if x]))
            st.caption(f"Python {v.get('python', '?').split()[0]} · GPU: {v.get('gpu')} · CUDA {v.get('cuda_version')} · core {v.get('core_version')} · {v.get('timestamp')}")
        else:
            st.info("env/versions.json not found.")
    with c2:
        st.markdown("#### Experiment configuration (hyper-parameters, seed)")
        cfg = store.get_experiment_config(root)
        st.json(cfg if cfg else {"info": "env/experiment_config.json not found"}, expanded=False)
    st.markdown("#### Model checkpoints")
    rows = []
    for s in store.list_subjects(root):
        for m in store.MODEL_KEYS:
            stt = store.get_status(root, m, s)
            rows.append({"subject": s, "model": store.MODEL_SHORT[m], "checkpoint_id": stt.get("checkpoint_id"), "finished": stt.get("finished"),
                         "epochs run": str(stt.get("epochs_run", "–")), "best epoch": str(stt.get("best_epoch", "–")), "val acc": round(float(stt.get("best_val_acc", stt.get("val_acc", float('nan')))), 3)})
    table(pd.DataFrame(rows))
    st.markdown("#### Data & preprocessing")
    ref = S["ref"]
    st.json({k: ref[k] for k in ["dataset", "sampling_frequency_hz", "software_filter", "hardware_filter", "reference", "artifact_removal", "epoch", "frequency_bands_hz"] if k in ref}, expanded=False)
    st.markdown("#### Integrity check")
    mf = Path(root) / "manifest.json"
    if not mf.exists():
        st.info("manifest.json not found.")
    elif st.button("Verify SHA-256 of model & reference files", key="verify"):
        man = xc.load_json(mf)
        bad, n = [], 0
        bar = st.progress(0.0)
        files = [f for f in man["files"] if f["path"].replace("\\", "/").split("/")[0] in ("models", "reference", "code")]
        for i, f in enumerate(files):
            p = Path(root) / f["path"]
            n += 1
            if not p.exists() or xc.sha256_file(p) != f["sha256"]:
                bad.append(f["path"])
            bar.progress((i + 1) / max(len(files), 1))
        if bad:
            callout("bad", "Mismatch", f"{len(bad)} of {n} files differ from the manifest:<br>" + "<br>".join(bad[:15]))
        else:
            callout("ok", "All good", f"{n} files match the SHA-256 checksums recorded at training time.")
    st.markdown("#### Disclaimer")
    st.markdown("Research and teaching tool — **not a medical device**. Explanations describe what the *models* used, not necessarily what the brain did. See *Notes › Limitations*.")

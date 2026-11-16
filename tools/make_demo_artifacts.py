"""Build a SMALL SYNTHETIC artifact folder in exactly the same format as the Colab notebook output.

Purpose: smoke-test the Streamlit app without the real ~2 GB artifact folder.
The numbers it produces are meaningless scientifically (synthetic EEG).

    python tools/make_demo_artifacts.py --out demo_artifacts --subjects 2
"""
import argparse, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import xbci_core as xc  # noqa: E402

LEFT = ["FC3", "FC1", "C5", "C3", "C1", "CP3", "CP1"]
RIGHT = ["FC4", "FC2", "C6", "C4", "C2", "CP4", "CP2"]
MID = ["FCz", "Cz", "CPz"]


def synth_trial(cls, rng, fs=128, T=513):
    from scipy.signal import butter, sosfiltfilt
    t = np.arange(T) / fs
    chs = xc.STD_CHANNELS_2A
    sos = butter(4, [4 / (fs / 2), 38 / (fs / 2)], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, rng.randn(len(chs), T), axis=-1) * 4.0
    erd = {0: MID, 1: RIGHT, 2: LEFT, 3: ["Fz", "FC1", "FC2", "FCz"]}[cls]   # feet, left, right, tongue
    for i, c in enumerate(chs):
        base_amp = 5.0 if (c in LEFT + RIGHT + MID) else 1.5
        amp = np.where(t > 1.0, base_amp * (0.35 if c in erd else 1.0), base_amp)
        x[i] += amp * np.sin(2 * np.pi * (10.5 + 0.3 * rng.randn()) * t + rng.rand() * 6.28)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_artifacts")
    ap.add_argument("--subjects", type=int, default=2)
    ap.add_argument("--per_class", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    root = Path(a.out) / "XBCI_BNCI2014_001"
    paths = xc.ArtifactPaths(root)
    paths.make_all()
    shutil.copy(HERE / "xbci_core.py", paths.code / "xbci_core.py")
    cfg = xc.deep_update(xc.DEFAULT_CFG, {"subjects": list(range(1, a.subjects + 1)),
                         "eegnet": {"epochs": a.epochs, "patience": 30, "ckpt_every": 5},
                         "explain": {"ig_steps": 16, "ig_baselines": 2, "baseline_pool_size": 16}})
    xc.save_json(paths.env / "experiment_config.json", cfg)
    xc.save_json(paths.env / "versions.json", xc.collect_env_info())
    rng = np.random.RandomState(0)
    registry = {"dataset": xc.DATASET_NAME, "created": xc.now_iso(), "class_names": xc.CLASS_NAMES, "models": {}}
    info0 = None
    for sid in cfg["subjects"]:
        X, y, sess = [], [], []
        for s in ["0train", "1test"]:
            for c in range(4):
                for _ in range(a.per_class):
                    X.append(synth_trial(c, rng)); y.append(c); sess.append(s)
        X, y = np.array(X, dtype=np.float32), np.array(y)
        lab = [xc.CLASS_NAMES[i] for i in y]
        meta = pd.DataFrame({"subject": sid, "session": sess, "run": "0", "trial_id": np.arange(len(y)),
                             "label": lab, "y": y})
        info = {"subject": sid, "ch_names": xc.STD_CHANNELS_2A, "sfreq": 128.0, "n_trials": len(y),
                "n_channels": 22, "n_times": X.shape[2], "unit": "uV", "dataset_interval_s": [2, 6],
                "sessions": ["0train", "1test"], "x_std_uV": float(X.std()), "preprocessing": cfg["preprocessing"],
                "created": xc.now_iso(), "DEMO": True}
        xc.save_npz(paths.data / f"subject_{sid:02d}.npz", X=X, y=y)
        xc.save_csv(meta, paths.data / f"subject_{sid:02d}_meta.csv")
        xc.save_json(paths.data / f"subject_{sid:02d}_info.json", info)
        info0 = info0 or info
        xc.process_subject(sid, {"X": X, "y": y, "meta": meta, "info": info}, cfg, paths, xc.get_device())
        for m in ["eegnet", "csp_lda"]:
            st = xc.load_json(paths.model_dir(m, sid) / "status.json")
            registry["models"].setdefault(m, {})[f"{sid:02d}"] = {
                "dir": str(paths.model_dir(m, sid).relative_to(root)), "checkpoint_id": st["checkpoint_id"]}
    xc.build_reference_artifacts(info0, cfg, paths)
    xc.save_json(root / "registry.json", registry)
    print("\nDEMO artifacts written to:", root.resolve())


if __name__ == "__main__":
    main()

"""Shared fixtures. Tests read the SAME artifact folder the app uses (XBCI_ARTIFACT_ROOT).
No test trains or fits anything: they audit what was saved."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))


def _find_root():
    env = os.environ.get("XBCI_ARTIFACT_ROOT")
    cands = [Path(env)] if env else []
    cands += [APP_DIR / "demo_artifacts" / "XBCI_BNCI2014_001", APP_DIR.parent / "XBCI_BNCI2014_001"]
    for c in cands:
        for q in (c, c / "XBCI_BNCI2014_001"):
            if (q / "reference" / "reference.json").exists():
                return q
    return None


@pytest.fixture(scope="session")
def root():
    r = _find_root()
    if r is None:
        pytest.skip("Set XBCI_ARTIFACT_ROOT to the unzipped XBCI_BNCI2014_001 folder")
    return r


@pytest.fixture(scope="session")
def xc(root):
    sys.path.insert(0, str(root / "code"))
    sys.modules.pop("xbci_core", None)
    import xbci_core
    return xbci_core


@pytest.fixture(scope="session")
def subjects(root, xc):
    paths = xc.ArtifactPaths(root)
    out = []
    for s in range(1, 30):
        ok = all((paths.model_dir(m, s) / "status.json").exists() and
                 json.load(open(paths.model_dir(m, s) / "status.json")).get("status") == "complete"
                 for m in ("eegnet", "csp_lda"))
        if ok:
            out.append(s)
    assert out, "no subject with both models complete"
    return out


@pytest.fixture(scope="session")
def is_demo(root, subjects):
    info = json.load(open(root / "data_processed" / f"subject_{subjects[0]:02d}_info.json"))
    return bool(info.get("DEMO", False))


def load_subject(root, sid):
    import pandas as pd
    z = np.load(root / "data_processed" / f"subject_{sid:02d}.npz")
    meta = pd.read_csv(root / "data_processed" / f"subject_{sid:02d}_meta.csv")
    return z["X"], z["y"], meta


def load_split(root, sid):
    z = np.load(root / "experiments" / f"subject_{sid:02d}" / "split_indices.npz")
    return z["train_idx"], z["val_idx"], z["test_idx"]

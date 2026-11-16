"""Interactive-path tests (AppTest). Usage: XBCI_ARTIFACT_ROOT=<folder> python tests/test_interactive.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tests.test_smoke import APP, AppTest


def fresh(page):
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.radio(key="nav").set_value(page).run()
    return at


def ok(at, label):
    errs = [e.value[:200] for e in at.exception]
    print(("OK   " if not errs else "FAIL ") + label, errs or "")
    return not errs


def main():
    good = True
    # --- playground: both models, levels, sources
    for model in ["eegnet", "csp_lda"]:
        for level in ["Beginner", "Intermediate", "Expert"]:
            at = fresh("🧪 Playground")
            at.sidebar.radio(key="level").set_value(level).run()
            at.radio(key="pg_model").set_value(model).run()
            at.button(key="pg_run").click().run()
            good &= ok(at, f"playground {model} / {level}")
    # edits
    at = fresh("🧪 Playground")
    at.radio(key="pg_model").set_value("eegnet").run()
    at.checkbox(key="pg_mirror").set_value(True).run()
    at.slider(key="pg_g_mu").set_value(-12).run()
    at.checkbox(key="pg_snr_on").set_value(True).run()
    at.button(key="pg_run").click().run()
    good &= ok(at, "playground edits (mirror + mu -12 dB + noise)")
    # class target
    at.selectbox(key="pg_tmode").set_value("a class I choose").run()
    at.selectbox(key="pg_tcls").set_value("tongue").run()
    at.button(key="pg_run").click().run()
    good &= ok(at, "playground target = chosen class")
    # synthetic
    at = fresh("🧪 Playground")
    at.radio(key="pg_src").set_value("Synthetic generator").run()
    at.button(key="pg_run").click().run()
    good &= ok(at, "playground synthetic")
    # upload (no file yet)
    at = fresh("🧪 Playground")
    at.radio(key="pg_src").set_value("Upload my own EEG").run()
    good &= ok(at, "playground upload (empty)")
    # feature mode
    at = fresh("🧪 Playground")
    at.radio(key="pg_mode").set_value("CSP feature values  (CSP+LDA only)").run()
    good &= ok(at, "playground feature mode")
    # compare targets
    for t in ["each model's own prediction (default)", "the true class (needs a labelled trial)", "one class I choose, for both models"]:
        at = fresh("⚖️ Compare models")
        at.radio(key="cmp_tmode").set_value(t).run()
        at.button(key="cmp_run").click().run()
        good &= ok(at, f"compare / {t[:25]}")
    at.radio(key="cmp_scope").set_value("All subjects").run()
    good &= ok(at, "compare all-subject scope")
    # subject switch
    at = fresh("🔬 Under the hood")
    at.sidebar.selectbox(key="subject").set_value(2).run()
    good &= ok(at, "inspector subject 2")
    # pure functions
    from xbci_ui import signals
    x = np.random.randn(22, 513).astype(np.float32)
    y = signals.apply_tf_gains(x, 128.0, np.full((5, 8), -6.0))
    z, _, _ = signals.apply_freq_gains(x, 128.0, {"mu": -10}, (9, 12, -6))
    arr = signals.parse_upload("a.csv", "\n".join(",".join(f"{v:.3f}" for v in r) for r in x[:, :200]).encode())
    arr2 = signals.parse_upload("a.csv", ("c" + ",c".join(str(i) for i in range(22)) + "\n" + "\n".join(",".join(f"{v:.3f}" for v in r) for r in x[:, :60].T)).encode())
    o, notes = signals.preprocess_uploaded(signals.orient_channels_first(arr2, 22), 250.0, "V", False, 128.0, 513)
    print("OK   signal helpers", y.shape, z.shape, arr.shape, arr2.shape, o.shape, notes)
    print("ALL PASSED" if good else "SOME FAILED")


if __name__ == "__main__":
    main()

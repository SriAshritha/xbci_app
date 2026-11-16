"""Five audit tests for the XBCI artifacts.

Design rules
  * Nothing is re-trained or tuned to make a test pass; tests only audit saved artifacts.
  * Every threshold is stated and justified in the docstring.
  * Tests 1-4 are valid for any artifact folder; test 5 (statistics on accuracy) is skipped for the
    synthetic demo folder, because synthetic data carries no scientific claim.
Run:  XBCI_ARTIFACT_ROOT=/path/to/XBCI_BNCI2014_001 python -m pytest tests -v
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import load_split, load_subject


# --------------------------------------------------------------------------------------------
def test_1_no_data_leakage_in_splits_and_fitted_statistics(root, xc, subjects):
    """Train/val come ONLY from session T, test ONLY from session E, nothing overlaps, and every
    statistic fitted from data (EEGNet normalisation, LDA class means/priors, attribution
    baselines) can be re-derived from TRAINING trials alone (and not from val/test/all data)."""
    paths = xc.ArtifactPaths(root)
    for sid in subjects:
        X, y, meta = load_subject(root, sid)
        tr, va, te = load_split(root, sid)
        sess = meta["session"].astype(str).values
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), f"S{sid}: overlapping splits"
        train_sess, test_sess = xc.pick_sessions(meta)
        assert train_sess != test_sess
        assert set(sess[tr]) == {train_sess} and set(sess[va]) == {train_sess}, f"S{sid}: train/val not from the training session"
        assert set(sess[te]) == {test_sess}, f"S{sid}: test not from the evaluation session"
        assert set(tr) | set(va) == set(np.where(sess == train_sess)[0]), f"S{sid}: training session not fully accounted for"
        # stratified validation split: class counts differ from the ideal share by at most 1
        n_cls = len(xc.CLASS_NAMES)
        for c in range(n_cls):
            ideal = (y[np.r_[tr, va]] == c).sum() * len(va) / (len(tr) + len(va))
            assert abs((y[va] == c).sum() - ideal) <= 1.0 + 1e-9, f"S{sid}: validation split not stratified for class {c}"

        # (a) EEGNet normalisation == mean/std of TRAIN trials only
        nz = np.load(paths.model_dir("eegnet", sid) / "normalization.npz")
        np.testing.assert_allclose(nz["mean"], X[tr].mean(axis=(0, 2)), rtol=1e-3, atol=1e-4, err_msg=f"S{sid}: mean not from train")
        np.testing.assert_allclose(nz["std"], X[tr].std(axis=(0, 2)), rtol=1e-3, atol=1e-4, err_msg=f"S{sid}: std not from train")
        all_std = X.std(axis=(0, 2))
        assert np.max(np.abs(nz["std"] - all_std) / all_std) > 1e-4, f"S{sid}: normalisation looks computed on ALL data"

        # (b) LDA class means / priors == statistics of CSP features of TRAIN trials only
        p = np.load(paths.model_dir("csp_lda", sid) / "csp_lda_params.npz")
        W = p["filters"].astype(np.float64)

        def feats(idx):
            s = np.einsum("kc,nct->nkt", W, X[idx].astype(np.float64))
            return np.log((s ** 2).mean(-1))
        f_tr = feats(tr)
        means = np.stack([f_tr[y[tr] == c].mean(0) for c in range(n_cls)])
        np.testing.assert_allclose(p["lda_means"], means, rtol=2e-3, atol=2e-3, err_msg=f"S{sid}: LDA means not from train")
        np.testing.assert_allclose(p["lda_priors"], np.bincount(y[tr], minlength=n_cls) / len(tr), atol=1e-6)
        f_te = feats(te)
        means_te = np.stack([f_te[y[te] == c].mean(0) for c in range(n_cls)])
        assert np.abs(p["lda_means"] - means_te).max() > 1e-3, f"S{sid}: LDA means match TEST data (leak?)"

        # (c) attribution baselines are real TRAINING trials
        bp = np.load(root / "experiments" / f"subject_{sid:02d}" / "baseline_pool.npz")
        assert set(bp["trial_idx"]) <= set(tr), f"S{sid}: baseline pool contains non-training trials"
        np.testing.assert_allclose(bp["X"], X[bp["trial_idx"]], atol=1e-6)


# --------------------------------------------------------------------------------------------
def test_2_saved_predictions_reproduce_from_saved_weights(root, xc, subjects):
    """Reload each model from disk, predict the TEST trials again and compare with predictions_*.csv
    and metrics.json. Tolerances: probabilities 1e-3 (CPU vs GPU float32), class decisions must be
    identical for >=99.5 % of trials. Labels in the CSV must equal the labels in the data."""
    paths = xc.ArtifactPaths(root)
    dev = torch.device("cpu")
    for sid in subjects:
        X, y, meta = load_subject(root, sid)
        _, _, te = load_split(root, sid)
        metrics = json.load(open(root / "experiments" / f"subject_{sid:02d}" / "metrics.json"))
        for m in ("eegnet", "csp_lda"):
            df = pd.read_csv(root / "experiments" / f"subject_{sid:02d}" / f"predictions_{m}.csv")
            d = df[df["split"] == "test"].set_index("trial_id")
            tid = meta.iloc[te]["trial_id"].values
            assert set(d.index) == set(tid), f"S{sid} {m}: CSV does not cover exactly the test trials"
            d = d.loc[tid]
            assert (d["y_true"].values == y[te]).all(), f"S{sid} {m}: labels in CSV differ from data"
            bundle = xc.load_bundle(paths, m, sid, dev)
            logits, probs = xc.predict_bundle(bundle, X[te])
            saved = d[[f"prob_{c}" for c in xc.CLASS_NAMES]].values
            assert np.abs(probs - saved).max() < 1e-3, f"S{sid} {m}: probabilities not reproducible"
            assert (probs.argmax(1) == d["y_pred"].values).mean() >= 0.995, f"S{sid} {m}: decisions not reproducible"
            acc_fresh = float((probs.argmax(1) == y[te]).mean())
            acc_csv = float((d["y_pred"].values == d["y_true"].values).mean())
            assert abs(acc_fresh - acc_csv) <= 0.005
            assert abs(metrics[f"{m}_test"]["accuracy"] - acc_csv) < 1e-9, f"S{sid} {m}: metrics.json disagrees with the CSV"


# --------------------------------------------------------------------------------------------
def test_3_additive_decomposition_is_exact(root, xc, subjects):
    """The app claims exact additive explanations: CSP+LDA  z_c = sum_k a_ck f_k + b_c  and EEGNet's
    last layer  z_c = sum_j W_cj h_j + b_c. Recompute both by hand and compare with the model's own
    logits (atol 5e-3 for CSP because its logits are large; 1e-4 for EEGNet)."""
    paths = xc.ArtifactPaths(root)
    dev = torch.device("cpu")
    for sid in subjects:
        X, _, _ = load_subject(root, sid)
        _, _, te = load_split(root, sid)
        x = X[te[:12]]
        # CSP + LDA
        b = xc.load_bundle(paths, "csp_lda", sid, dev)
        lg, _ = xc.predict_bundle(b, x)
        p = np.load(paths.model_dir("csp_lda", sid) / "csp_lda_params.npz")
        s = np.einsum("kc,nct->nkt", p["filters"].astype(np.float64), x.astype(np.float64))
        f = np.log((s ** 2).mean(-1) + 1e-12)
        z = f @ p["lda_coef"].astype(np.float64).T + p["lda_intercept"].astype(np.float64)
        np.testing.assert_allclose(z, lg, atol=5e-3, err_msg=f"S{sid}: CSP+LDA decomposition not exact")
        contrib = f[:, None, :] * p["lda_coef"][None]                      # (n, class, k)
        np.testing.assert_allclose(contrib.sum(-1) + p["lda_intercept"], lg, atol=5e-3)
        # EEGNet last layer
        e = xc.load_bundle(paths, "eegnet", sid, dev)
        acts, logits = e["model"].named_activations(torch.as_tensor(x, dtype=torch.float32))
        h = acts["flatten"].numpy()
        net = e["model"].net
        Wc, bc = net.classifier.weight.detach().numpy(), net.classifier.bias.detach().numpy()
        np.testing.assert_allclose(h @ Wc.T + bc, logits.numpy(), atol=1e-4, err_msg=f"S{sid}: EEGNet last layer not exact")


# --------------------------------------------------------------------------------------------
def test_4_integrated_gradients_complete_and_deterministic(root, xc, subjects):
    """Integrated Gradients must satisfy completeness: sum(attribution) = F(x) - F(baseline).
    We accept a median relative error below 5 % at 64 steps (a standard Riemann-sum tolerance), and
    require identical output for identical inputs and seed (no hidden randomness)."""
    paths = xc.ArtifactPaths(root)
    dev = torch.device("cpu")
    sid = subjects[0]
    X, _, _ = load_subject(root, sid)
    tr, _, te = load_split(root, sid)
    pool = np.load(root / "experiments" / f"subject_{sid:02d}" / "baseline_pool.npz")["X"]
    x = X[te[:8]]
    for m in ("eegnet", "csp_lda"):
        b = xc.load_bundle(paths, m, sid, dev)
        lg, _ = xc.predict_bundle(b, x)
        tgt = lg.argmax(1)
        ig1, d1 = xc.integrated_gradients(b["model"], x, tgt, pool, dev, n_steps=64, n_baselines=3, batch_size=4, seed=0)
        ig2, d2 = xc.integrated_gradients(b["model"], x, tgt, pool, dev, n_steps=64, n_baselines=3, batch_size=4, seed=0)
        np.testing.assert_allclose(ig1, ig2, atol=1e-6, err_msg=f"{m}: IG not deterministic")
        scale = np.median(np.abs(lg[np.arange(len(x)), tgt])) + 1e-6
        assert np.median(np.abs(d1)) / scale < 0.05, f"{m}: IG completeness error too large ({np.median(np.abs(d1)):.4g} vs score {scale:.3g})"
        assert np.isfinite(ig1).all()
        # attribution must react to the input: a different trial gives a different map
        assert np.abs(ig1[0] - ig1[1]).max() > 0


# --------------------------------------------------------------------------------------------
def test_5_reported_accuracy_is_statistically_real_and_leak_free(root, xc, subjects, is_demo):
    """On the pooled TEST predictions of all subjects: (i) the test set is class-balanced, (ii) the
    model is significantly better than chance (one-sided exact binomial test, p < 1e-3), (iii) a
    shuffled-label control gives chance-level accuracy (so the evaluation code cannot be leaking
    labels), (iv) the model is not a constant predictor. Skipped for the synthetic demo folder."""
    if is_demo:
        pytest.skip("synthetic demo data: no scientific claim is tested")
    from scipy.stats import binomtest
    rng = np.random.RandomState(0)
    for m in ("eegnet", "csp_lda"):
        df = pd.concat([pd.read_csv(root / "experiments" / f"subject_{s:02d}" / f"predictions_{m}.csv") for s in subjects])
        d = df[df["split"] == "test"]
        n, k = len(d), int((d["y_true"] == d["y_pred"]).sum())
        share = d["y_true"].value_counts(normalize=True)
        assert share.max() - share.min() < 0.02, f"{m}: test set not class-balanced"
        chance = float(share.max())
        res = binomtest(k, n, p=chance, alternative="greater")
        assert res.pvalue < 1e-3, f"{m}: accuracy {k / n:.3f} is not significantly above chance (p={res.pvalue:.3g})"
        shuffled = [float((rng.permutation(d["y_true"].values) == d["y_pred"].values).mean()) for _ in range(300)]
        assert abs(np.mean(shuffled) - chance) < 0.02, f"{m}: shuffled-label accuracy is not at chance -> evaluation leak?"
        assert max(shuffled) < k / n, f"{m}: a shuffled-label run reached the real accuracy"
        assert d["y_pred"].nunique() >= 3, f"{m}: model predicts almost a single class"

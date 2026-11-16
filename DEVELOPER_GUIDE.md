# XBCI Developer Guide

## 1. Big picture

```
Colab notebook (training)  --->  artifact folder  --->  Streamlit app (read-only)
 trains models, writes files      the only interface       explains, never trains
```

The app never trains anything and never calls Colab. The artifact folder is the contract between the two halves.

## 2. Artifact contract (what the app reads)

| Path | Used for |
|---|---|
| `code/xbci_core.py` | The training library. The app imports **this** copy, so loading code always matches the code that saved the models. |
| `reference/reference.json` | Channel order and positions, sampling rate, filters, class map, band definitions |
| `data_processed/subject_XX.npz` (+ `_meta.csv`, `_info.json`) | Pre-processed epochs `X (n, 22, 513)` in microvolts, labels `y` |
| `models/eegnet/subject_XX/` | `model_weights.pth`, `normalization.npz`, `config.json`, `layer_summary.json`, `layer_weights.npz`, `history.csv`, `status.json` |
| `models/csp_lda/subject_XX/` | `csp_lda_params.npz`, `config.json`, `status.json` |
| `experiments/subject_XX/` | `split_indices.npz`, `baseline_pool.npz`, `predictions_*.csv`, `explanations_*.npz`, `comparison.csv`, `metrics.json` |
| `env/`, `registry.json`, `manifest.json` | Versions, settings, checksums (About tab) |

A subject appears in the app only if both `status.json` files say `complete`.

## 3. Code map

```
app.py                 page config, sidebar, navigation, routing
xbci_ui/store.py       finds the artifact root; imports xbci_core; cached loaders (st.cache_data / cache_resource)
xbci_ui/analysis.py    cached heavy work: predict, explain, exact breakdowns, ablations, noise tests
xbci_ui/explain_ui.py  the six-step "why" story for one model
xbci_ui/inputs_ui.py   signal builder widget (dataset / upload / synthetic + edits)
xbci_ui/signals.py     pure numpy signal edits (testable without Streamlit)
xbci_ui/viz.py         every Plotly figure; own topomap (no MNE needed)
xbci_ui/theme.py       CSS and small HTML components
xbci_ui/content.py     articles, glossary, references
xbci_ui/pages/*.py     one module per tab, each exposes render(S)
```

`S` is a small dict built in `app.py`: `root`, `sid`, `ref`, `cn` (class names), `level`, `xc` (the core module).

## 4. Rules we follow

1. **Charts:** always finish a figure with `viz._layout(fig, title, height)`. It wraps long titles, places the legend between title and plot, and prevents clipped labels. Do not set `legend.y` or `title.y` by hand.
2. **Rendering:** use `theme.plot(fig, key=...)` and give every chart a unique `key`.
3. **Heavy work** goes in `analysis.py` behind `st.cache_data`; return plain numpy/dicts only.
4. **Widgets** need unique keys (prefix with the page, e.g. `pg_`, `cmp_`, `sc_`).
5. **Honesty:** numbers shown to users must come from files or from a computation visible on screen. No hard-coded results; text that depends on data must be conditional.
6. **Never** train, fit or tune inside the app.

## 5. How to extend

**Add a page.** Create `xbci_ui/pages/mypage.py` with `render(S)`, add its label to `PAGES` and the routing block in `app.py`.

**Add a model.** In the training library save a `status.json`, `config.json` and weights, and make `load_bundle` return a torch module that maps `(B, 22, T)` microvolts to logits and implements `named_activations(x) -> (dict, logits)`. Then add its key to `store.MODEL_KEYS`, `MODEL_LABELS` and `MODEL_COLORS`, and add a breakdown function in `analysis.py` if it has an exact decomposition. Every other page works from logits and gradients.

**Add a science question.** Write `qN(S)` in `pages/science.py`, register it in `QUESTIONS` and `FUNCS`. State the caveat next to the plot.

## 6. Tests

```bash
export XBCI_ARTIFACT_ROOT=/path/to/XBCI_BNCI2014_001
python -m pytest tests -v
```

| Test | What it proves |
|---|---|
| `test_1_no_data_leakage...` | Train/val come only from session T, test only from session E, splits are disjoint and stratified; EEGNet normalisation, LDA class means/priors and attribution baselines can be re-derived from **training trials only** (and not from test or all data) |
| `test_2_saved_predictions_reproduce...` | Reloading the saved weights reproduces the saved predictions and `metrics.json`; labels in the CSV equal the labels in the data |
| `test_3_additive_decomposition...` | The explanations claimed to be exact really are: CSP+LDA and EEGNet's last layer recomputed by hand equal the model logits |
| `test_4_integrated_gradients...` | Attribution completeness (median relative error below 5 % at 64 steps) and determinism |
| `test_5_reported_accuracy...` | On pooled test predictions: balanced classes, accuracy significantly above chance (exact binomial test, p < 0.001), shuffled-label control at chance, model not a constant predictor |

Design choices: tests audit saved artifacts and never retrain, thresholds are documented in each docstring and were not
tuned after seeing results, and test 5 is skipped for the synthetic demo folder because synthetic data supports no
scientific claim. We checked that the tests can fail: planting a leaked normalisation file makes test 1 fail.

UI smoke tests (Streamlit AppTest): `tests/test_smoke.py` opens every page, `python tests/test_interactive.py` exercises
the editors, both models, all explanation targets and all science questions.

## 7. Release checklist

- [ ] `python -m pytest tests -v` passes on the real artifact folder
- [ ] `python tests/test_interactive.py` prints `ALL PASSED`
- [ ] Open every tab at Beginner and Expert level; look for overlapping text in charts
- [ ] README quick start works on a clean virtual environment
- [ ] No absolute paths or personal data in the repository

## 8. Known limitations

* Per-subject models only; no cross-subject or real-time mode.
* Explanations are for the models, not causal claims about the brain.
* The Dockerfile has not been tested.
* Pixel-level styling (navigation pills, fonts) was not checked in every browser.

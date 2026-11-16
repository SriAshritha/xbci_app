# XBCI: Explainable Brain-Computer Interface

A brain-computer interface (BCI) reads brain signals and turns them into a decision, for example "the person is
imagining their left hand". Most BCI models give you the answer but not the reason. XBCI is a Streamlit app that
shows the reason.

It works on the public **BCI Competition IV-2a** dataset (9 people, 22 EEG electrodes, 4 imagined movements: left hand,
right hand, both feet, tongue) and explains two very different models side by side:

* **EEGNet**, a small convolutional neural network that learns its own features.
* **CSP + LDA**, the classic hand-built pipeline where every step is a simple formula.

You can pick a model, give it a signal, and see which electrodes, moments in time and brain rhythms pushed it
towards its answer. You can also give the *same* signal to both models and compare their reasons.

## What you need

* Python 3.10 to 3.12
* The **artifact folder** from our Colab notebook (the unzipped `XBCI_BNCI2014_001/`). The app reads everything from
  this folder: the trained models, the data, the saved predictions and explanations. It does not need Colab or the
  internet while running.
* About 2 GB of free memory. A GPU is optional.

## Quick start

```bash
git clone <this repository>
cd xbci_app
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# tell the app where the artifact folder is
# Windows:  set XBCI_ARTIFACT_ROOT=C:\path\to\XBCI_BNCI2014_001
# macOS/Linux:  export XBCI_ARTIFACT_ROOT=/path/to/XBCI_BNCI2014_001
streamlit run app.py
```

Your browser opens at http://localhost:8501. If you skip the environment variable, paste the folder path into the
**Artifact folder** box in the sidebar.

No artifact folder yet? You can still try the interface with a small *synthetic* demo (the numbers mean nothing
scientifically):

```bash
pip install mne
python tools/make_demo_artifacts.py --out demo_artifacts
```

then point the app at `demo_artifacts/XBCI_BNCI2014_001`.

## What is inside the app

| Tab | What you can do there |
|---|---|
| Start here | Read a short introduction, watch an animated tour of one real trial, see the real brain-rhythm changes in the data |
| Playground | Choose one model and any input (a recorded trial, your own file, a synthetic signal, or an edited one) and follow a six-step explanation |
| Compare models | Send one signal to both models and compare predictions, reasons and how similar those reasons are |
| Under the hood | Browse the real learned weights, filters and training curves |
| Performance | Accuracy per person, confusion matrices, calibration, model agreement |
| Science Q&A | Eleven scientific questions answered from the saved data and quick live experiments |
| Notes & Blog | Eight short articles, four videos, a glossary and references |
| About | Software versions, settings, checkpoint IDs and a file-integrity check |

The sidebar lets you choose the person (each has their own two models) and the **explanation level**: Beginner,
Intermediate or Expert. Beginner uses plain language; Expert adds the equations.

Longer guides: [USER_GUIDE.md](USER_GUIDE.md) and [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

## How the explanations are made

* **CSP + LDA, and the last layer of EEGNet:** the score is an exact weighted sum, so we show each term. The app
  recomputes it by hand and checks it against the model.
* **Integrated Gradients:** traces a decision back to every electrode and time point. We use baselines taken from real
  training trials, and show a numerical check that the parts add up to the whole.
* **Frequency and time-frequency importance:** remove one rhythm (optionally in one time window) and see how much the
  score drops.
* **Plausibility check:** compares the evidence with what neuroscience expects (for example, imagining the left hand
  should involve the right side of the brain).

## Results and honest limits

In our run, average test accuracy was about **62 % for EEGNet** and **57 % for CSP + LDA** (chance is 25 %). Two
people (subjects 5 and 6) were close to chance. With 288 test trials per person, any accuracy has roughly a
plus-or-minus 5 to 6 point margin, so we do not claim that one model is clearly better.

* The models are trained per person on about 230 trials and do not transfer to new people.
* We did not remove eye or muscle artifacts.
* An explanation shows what the *model* used, not necessarily what the brain did.
* Edited or synthetic signals are unlike the training data, so they show how the model reacts, not how a brain would.
* This is a research and teaching tool, not a medical device.

## Tests

```bash
export XBCI_ARTIFACT_ROOT=/path/to/XBCI_BNCI2014_001     # set XBCI_ARTIFACT_ROOT on Windows
python -m pytest tests -v
```

Five audit tests check the saved results (no data leakage, reproducible predictions, exact explanations, valid
attributions, statistically real accuracy). Nothing is re-trained. Details are in the developer guide.

## Project layout

```
app.py                  entry point
xbci_core.py            fallback copy of the training library (the app prefers <artifacts>/code/xbci_core.py)
xbci_ui/                the app: data access, analysis, plots, pages, written content
tests/                  audit tests and UI smoke tests
tools/                  demo-data generator and video renderer
assets/videos/          explainer videos
```

## Credits

Data: Brunner et al., *BCI Competition IV, data set 2a* (loaded with MOABB). Methods: EEGNet (Lawhern et al., 2018),
CSP (Ramoser et al., 2000), Integrated Gradients (Sundararajan et al., 2017). The full list is in the app under
Notes & Blog, References.

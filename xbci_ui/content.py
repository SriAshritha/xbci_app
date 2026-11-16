"""Long-form learning content (articles, glossary, references)."""

ARTICLES = {}


def _add(key, title, minutes, summary, body):
    ARTICLES[key] = {"title": title, "minutes": minutes, "summary": summary, "body": body}


_add("bci101", "1 · Brain–computer interfaces in ten minutes", 6,
     "What a BCI is, how the signal becomes a command, and why we need explainability.", r"""
A **brain–computer interface (BCI)** translates brain activity into a command for a computer or device *without using the body's muscles*
(Wolpaw et al., 2002). Typical uses: communication for people with paralysis, neuro-rehabilitation after stroke, research tools, games.

### The five-stage pipeline
| Stage | What happens | In this project |
|---|---|---|
| 1 · Acquisition | Electrodes on the scalp measure microvolt-level voltage fluctuations (**EEG**) | BCI Competition IV-2a, 22 electrodes, 250 Hz |
| 2 · Pre-processing | Band-pass filtering, re-sampling, cutting the recording into *epochs* | 4–38 Hz, 128 Hz, 4 s epochs after the cue |
| 3 · Feature extraction | Turn the signal into informative numbers | CSP log-band-power **or** learned by EEGNet |
| 4 · Classification | Map features → intended command | LDA **or** EEGNet's dense layer |
| 5 · Feedback | The user sees/feels the result and adapts | not part of this offline study |

### Three popular BCI paradigms
* **Motor imagery (MI)** — the user *imagines* moving a limb; sensorimotor rhythms change. **← this project**
* **P300** — a rare, attended stimulus evokes a positive deflection ~300 ms later (spellers).
* **SSVEP** — looking at a flickering light evokes brain activity at the flicker frequency.

### Why *explainable* BCI?
1. **Trust & safety** – a wheelchair or a prosthesis must not act on a decision nobody can justify.
2. **Debugging** – a model that is 80 % accurate because it picked up *eye blinks* or *muscle tension* is useless in real life; explanations reveal this.
3. **Neuroscience validity** – a motor-imagery model *should* rely on sensorimotor electrodes and mu/beta rhythms.
4. **Model choice** – two models with the same accuracy can use very different evidence; explanations expose it.
5. **Calibration of expectations** – knowing *when* a model is unsure or out-of-distribution.
""")

_add("mi", "2 · Motor imagery and brain rhythms (ERD / ERS)", 7,
     "Why imagining a movement changes the EEG, and where on the scalp to look.", r"""
### Rhythms
| Rhythm | Frequency | Typical association |
|---|---|---|
| delta | 1–4 Hz | deep sleep, slow potentials |
| theta | 4–8 Hz | drowsiness, memory, frontal control |
| **mu / alpha** | **8–13 Hz** | **idle sensorimotor cortex** (mu), relaxed visual cortex (alpha) |
| **beta** | **13–30 Hz** | **sensorimotor cortex**, rebounds after movement |
| gamma | > 30 Hz | local processing (weak at the scalp, easily contaminated by muscle) |

### The key phenomenon: ERD and ERS
When the sensorimotor cortex is *idle*, large populations of neurons fire in synchrony and generate a strong **mu/beta rhythm**.
Preparing, executing **or merely imagining** a movement desynchronises those neurons: the rhythm's power **drops**. This is
**event-related desynchronisation (ERD)**. After the movement, beta power often overshoots: **event-related synchronisation (ERS)**
(Pfurtscheller & Lopes da Silva, 1999). Classically it is quantified as

$$\mathrm{ERD\%}=\frac{A-R}{R}\times100,$$

with $A$ = band power in the event window and $R$ = power in a reference (baseline) period.

### Where? Somatotopy and contralaterality
* The right hemisphere controls the **left** side of the body and vice versa → imagining the **left hand** produces ERD mainly over the **right** motor cortex (electrodes **C4**, C2, CP4 …); the **right hand** over the **left** (C3, C1, CP3 …).
* The motor homunculus puts the **hands laterally** and the **feet near the mid-line** (around **Cz**). The tongue representation is small and lateral/inferior and gives the least separable scalp signals.

### Why decoding is hard
* Signals are microvolts, buried in noise (other brain processes, eye blinks, muscle activity).
* Strong **inter-subject** differences; a sizeable minority of people produce weak MI rhythms ("BCI inefficiency").
* **Non-stationarity** — the same person differs between days (here: session T vs E).
* Few trials per class (72 per class per session in this dataset).

> **On the *Start here* tab you can see the real ERD in your selected subject's own training data.**
""")

_add("data", "3 · The dataset: BCI Competition IV-2a", 5,
     "Who, what, when — and exactly how we pre-processed and split it.", r"""
**Graz data set A** (Brunner et al., 2008; BCI Competition IV, Tangermann et al., 2012), accessed through **MOABB** (`BNCI2014_001`; Jayaram & Barachant, 2018).

| Item | Value |
|---|---|
| Participants | 9 healthy subjects |
| Classes | left hand · right hand · both feet · tongue (motor imagery) |
| Channels | 22 EEG (+3 EOG, dropped here) |
| Sampling rate | 250 Hz (re-sampled to 128 Hz here) |
| Hardware filter | 0.5–100 Hz band-pass, 50 Hz notch |
| Sessions | 2 on different days: **T** (training) and **E** (evaluation) |
| Trials | 6 runs × 48 trials = **288 per session** (72 per class) |

### Trial timeline (seconds from trial start)
`0` fixation cross + beep → `2` **cue** (arrow, 1.25 s) → `3 – 6` **motor imagery** → `6` break.
Our epoch starts at the **cue (t = 2 s)** and lasts **4 s** → imagery occupies roughly **1 – 4 s of the epoch** (shaded in the time plots).

### What *we* did
1. MOABB loads raw `.mat` files, keeps the 22 EEG channels, **IIR band-pass 4–38 Hz** (zero-phase), epochs 0–4 s, resamples to 128 Hz → **22 × 513** array per trial, in µV.
2. **Per-subject** models (the standard within-subject protocol): session **T** → 80 % *train* / 20 % *validation* (stratified, indices saved); session **E** → **test** (never used for training or model selection).
3. No artifact rejection or re-referencing (reference = left mastoid as recorded). EOG channels were removed but eye/muscle artifacts can still leak into EEG channels.

### Reading accuracies
Chance is **25 %**. With 288 test trials the 95 % confidence interval of an accuracy near 60 % is about **±5–6 points**; differences smaller than that are not meaningful.
""")

_add("eegnet", "4 · EEGNet explained", 8,
     "A compact convolutional network whose layers mirror classical EEG signal processing.", r"""
EEGNet (Lawhern et al., 2018) is a small CNN designed for EEG. Its three convolutional blocks correspond to three classical steps:

| Block | Layer | Classical analogue | Shape (this project) |
|---|---|---|---|
| 1 | **Temporal convolution** (8 kernels × 64 samples) | band-pass filter bank | 8 × 22 × 513 |
| 1 | **Depthwise spatial convolution** (2 per temporal filter; kernel 22×1) | spatial filtering (like CSP) | 16 × 1 × 513 |
| – | BatchNorm → ELU → AvgPool(1×4) → Dropout | smoothing / down-sampling | 16 × 1 × 128 |
| 2 | **Separable convolution** (depthwise 1×16 then pointwise 1×1) | temporal summary of each source | 16 × 1 × 128 |
| – | BatchNorm → ELU → AvgPool(1×8) → Dropout | | 16 × 1 × 16 |
| 3 | **Flatten → Linear** (256 → 4) | LDA-like linear classifier | 4 logits |

**Learnable parameters ≈ 2.5 thousand** (compare with millions in typical image networks) — which makes training on a few hundred trials feasible.

### Math of the final decision
With $h\in\mathbb R^{256}$ the flattened feature vector,
$$z_c=\sum_{j=1}^{256}W_{c,j}h_j+b_c,\qquad P(c)=\frac{e^{z_c}}{\sum_{k}e^{z_k}}.$$
Because this last stage is linear, the **exact contribution** of each feature $j$ to class $c$ is $W_{c,j}h_j$ — the app shows these as heat-maps (16 feature maps × 16 time bins).

### How it was trained here
Per-channel standardisation using training-set mean/std (stored inside the model wrapper) · Adam, lr 1e-3 with plateau decay · batch 32 · ≤ 300 epochs, early stopping (patience 60) on validation loss · max-norm constraints (1.0 spatial, 0.25 dense) · dropout 0.5 · seeds fixed and stored · best-validation checkpoint kept.

### Spatial filters vs patterns
The depthwise weights are **filters** (recipes for *extracting* a source). To see *where* the extracted source lives on the scalp you need the **pattern**
$\mathbf a=\Sigma\mathbf w/(\mathbf w^\top\Sigma\mathbf w)$ (Haufe et al., 2014). The app draws patterns by default.
""")

_add("csp", "5 · CSP + LDA explained", 8,
     "The classic hand-made pipeline: spatial filters that maximise variance differences, then a linear vote.", r"""
**Common Spatial Patterns (CSP)** (Ramoser et al., 2000; Blankertz et al., 2008) learns spatial filters $\mathbf w$ whose output *variance* differs maximally between classes.
For two classes with covariance matrices $\Sigma_1,\Sigma_2$:
$$\max_{\mathbf w}\;\frac{\mathbf w^\top\Sigma_1\mathbf w}{\mathbf w^\top(\Sigma_1+\Sigma_2)\mathbf w}\quad\Longleftrightarrow\quad \Sigma_1\mathbf w=\lambda(\Sigma_1+\Sigma_2)\mathbf w .$$
Filters with very large or very small eigenvalues are the useful ones. For **four classes** (as here) MNE-Python uses approximate joint diagonalisation of the class covariances and picks components by mutual information; we keep **8**.

### From filters to features
$$\mathbf s(t)=\mathbf W\mathbf x(t),\qquad f_k=\log\Big(\tfrac1T\sum_t s_k(t)^2\Big).$$
Variance of a band-limited signal **is** its band power, so $f_k$ is a *log band-power feature* — exactly what ERD changes. Throwing away timing is both CSP's strength (robust) and weakness (blind to *when*).

### LDA
Linear Discriminant Analysis models each class as a Gaussian with a **shared** covariance (here Ledoit–Wolf-shrunk). The log-posterior is then linear:
$$z_c=\mathbf a_c^\top\mathbf f+b_c,\qquad P(c)=\mathrm{softmax}(z)_c .$$
Each term $a_{c,k}f_k$ is an **exact additive contribution** — that is why the CSP+LDA explanation in this app is mathematically exact for the last stage.

### Filters vs patterns
Rows of $\mathbf W$ are filters; the columns of $\mathbf W^{-1}$ (**patterns**) are the scalp topographies of the sources (Haufe et al., 2014). The app draws **patterns**.

### Strengths / weaknesses
✔ tiny, fast, interpretable · ✔ works with few trials · ✘ hand-designed features, discards timing and phase · ✘ sensitive to outliers/artifacts because variance is not robust.
""")

_add("xai", "6 · How the app explains a decision", 9,
     "Integrated Gradients, frequency occlusion, exact breakdowns — and what each can and cannot tell you.", r"""
We use **three complementary layers** of explanation.

### A · Exact, model-specific breakdown (the "math")
* **CSP+LDA**: $z_c=\sum_k a_{c,k}f_k+b_c$ — every feature's share is exact.
* **EEGNet**: the last layer is linear too → $W_{c,j}h_j$ is exact; earlier layers are non-linear, so we *show* activations and weights instead of claiming exact shares.
The app recomputes these by hand and reports the maximum difference against the saved model (should be ≈ 1e-6).

### B · Gradient attribution — Integrated Gradients (IG)
For input $x$, reference (baseline) $x'$ and model score $F$:
$$\mathrm{IG}_i(x)=(x_i-x'_i)\int_0^1\frac{\partial F\big(x'+\alpha(x-x')\big)}{\partial x_i}\,d\alpha .$$
* **Completeness**: $\sum_i\mathrm{IG}_i=F(x)-F(x')$ — attributions add up to the score difference (Sundararajan et al., 2017). The app reports the numerical error.
* **Baselines**: a zero signal would be unnatural for EEG (and makes CSP's $\log$-variance undefined), so we average over **3 baselines sampled from real training trials** (the idea of *expected gradients*, Erion et al., 2021). The pool is saved for reproducibility.
* Same method, same baselines for both models → their maps are **directly comparable**.
* Summaries: **channel** (sum over time) · **time** (sum over electrodes) · signed maps (red = pushes toward the explained class, blue = against).

### C · Frequency / time-frequency occlusion (model-agnostic)
Remove one frequency band (Butterworth band-pass subtracted from the signal), optionally only inside one time window, and measure the **drop of the explained class score (logit)**. Big drop ⇒ the model relied on that rhythm at that moment (inspired by occlusion analyses, Zeiler & Fergus, 2014).

### Choosing the explained class
By default each model is explained **for its own predicted class** ("why did you say that?"). In *Compare* you can explain the **true** class or **one class for both** models to remove the confound of different targets.

### Honest limitations
* Attribution ≠ causation: it says what the *model* used, not what the brain did.
* IG depends on the baseline; occlusion creates unnatural signals (out-of-distribution) and ignores interactions between bands.
* Highly correlated electrodes share credit arbitrarily.
* A model can be right for the wrong reason (artifacts). The app's *neuro-plausibility check* is a coarse filter, not proof.
""")

_add("plots", "7 · How to read the plots", 4,
     "A visual dictionary for every chart in the app.", r"""
| Plot | How to read it |
|---|---|
| **Topographic map (topomap)** | Top view of the head, nose up, left ear on the left. Colour is interpolated between the 22 electrodes (dots). **Red** = positive, **blue** = negative (signed maps); darker = stronger. |
| **Stacked EEG traces** | One line per electrode over 4 s from the cue. Vertical offsets are only for display. |
| **Spectrum** | Power vs frequency, averaged over electrodes. Shaded zones = the 5 bands used in explanations. |
| **Probability bars** | Softmax output; highlighted = predicted class. |
| **Contribution bars** | Each bar = how much one feature/weight pushed the score gap (green → toward the prediction, red → against). |
| **Heat-map: feature map × time bin** | Rows = 16 learned feature maps; columns = 0.25 s windows; colour = contribution to *predicted − runner-up* score. |
| **Time-frequency importance** | Rows = bands; columns = time windows; colour = drop of class score when that band is removed in that window. |
| **Gauge: lateralisation** | −1 = all evidence on left-hemisphere electrodes, +1 = right hemisphere. |
| **Similarity gauges** | Correlation between the two models' explanations (−1…+1). |
| **Confusion matrix** | Rows = true class, columns = predicted; diagonal = correct; numbers in brackets = trials. |
| **Calibration curve** | If the model says "70 %", it should be right ~70 % of the time (curve near the diagonal). |
| **Cohen's κ** | Accuracy corrected for chance (0 = chance, 1 = perfect). |
""")

_add("limits", "8 · Limitations, ethics and honest caveats", 5,
     "What these numbers do — and do not — mean.", r"""
### Scientific limits
* **Small data**: 230 training trials per subject. Models are **subject-specific**; they do not transfer to new people or days without calibration.
* **One dataset, one protocol**: 9 healthy adults, lab conditions. Real-world BCIs face movement, fatigue, different hardware.
* **Statistical uncertainty**: ±5–6 points on any test accuracy (288 trials). Do not over-interpret small gaps between models.
* **Non-stationarity**: test (session E) is recorded on a different day than training (session T) — the drop from validation to test accuracy reflects this.
* **Artifacts are not removed**: a decision may partly rest on eye/muscle activity; the plausibility check can only flag obvious cases.
* **What-if edits are out-of-distribution**: editing a spectrum or synthesising a signal tells you how the model *reacts*, not how a brain would.

### Ethics
* Neural data are sensitive personal data; the public dataset used here is anonymised and published with consent.
* This software is a **research and teaching tool, not a medical device**. Never use it for clinical decisions.
* Explanations can create *false confidence*: a pretty topomap does not prove the model is right.
""")

GLOSSARY = {
    "EEG": "Electroencephalography — voltage fluctuations measured on the scalp.",
    "Electrode / channel": "One recording site (e.g. C3). This dataset has 22.",
    "Epoch": "A fixed-length snippet of EEG around an event (here 4 s from the cue).",
    "Cue": "The on-screen arrow telling the subject which movement to imagine.",
    "Motor imagery (MI)": "Mentally rehearsing a movement without executing it.",
    "Mu / beta rhythm": "Sensorimotor oscillations at 8–13 Hz and 13–30 Hz; their power drops during movement and imagery.",
    "ERD / ERS": "Event-related desynchronisation (power ↓) / synchronisation (power ↑).",
    "Contralateral": "Opposite side — the left hand is controlled mainly by the right hemisphere.",
    "Topomap": "A colour map of a value across the scalp, interpolated between electrodes.",
    "Spatial filter": "A weighted sum of electrodes that extracts one source of activity.",
    "Spatial pattern": "The scalp topography of a source (the interpretable counterpart of a filter).",
    "CSP": "Common Spatial Patterns — spatial filters that maximise class differences in variance.",
    "LDA": "Linear Discriminant Analysis — a linear classifier assuming Gaussian classes with shared covariance.",
    "EEGNet": "A compact convolutional neural network designed for EEG.",
    "Logit": "The raw class score before softmax.",
    "Softmax": "Turns scores into probabilities that sum to 1.",
    "Integrated Gradients": "An attribution method that distributes a prediction over the input features, with the completeness property.",
    "Baseline (attribution)": "The reference input against which attributions are measured.",
    "Occlusion": "Removing part of the input and observing the change in the output.",
    "Attribution": "A score of how much each input element contributed to the output.",
    "Calibration": "Agreement between predicted confidence and actual accuracy.",
    "Cohen's κ": "Chance-corrected agreement between predictions and truth.",
    "Artifact": "Non-brain signal in the EEG (blinks, eye movement, muscle, line noise).",
    "Session T / E": "Training / evaluation recording days of BCI Competition IV-2a.",
    "Out-of-distribution (OOD)": "Input unlike anything seen in training; predictions are unreliable.",
}

REFERENCES = [
    "Brunner, C., Leeb, R., Müller-Putz, G. R., Schlögl, A., & Pfurtscheller, G. (2008). *BCI Competition 2008 – Graz data set A.* Graz University of Technology.",
    "Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience, 6*, 55.",
    "Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering, 15*(5), 056013.",
    "Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping, 38*(11), 5391–5420.",
    "Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology, 110*(11), 1842–1857.",
    "Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering, 8*(4), 441–446.",
    "Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K.-R. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine, 25*(1), 41–56.",
    "Haufe, S., et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage, 87*, 96–110.",
    "Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of ICML.*",
    "Erion, G., Janizek, J. D., Sturmfels, P., Lundberg, S. M., & Lee, S.-I. (2021). Improving performance of deep learning models with axiomatic attribution priors and expected gradients. *Nature Machine Intelligence, 3*, 620–631.",
    "Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV.*",
    "Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis, 88*(2), 365–411.",
    "Jayaram, V., & Barachant, A. (2018). MOABB: trustworthy algorithm benchmarking for BCIs. *Journal of Neural Engineering, 15*(6), 066011.",
    "Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience, 7*, 267.",
    "Wolpaw, J. R., et al. (2002). Brain–computer interfaces for communication and control. *Clinical Neurophysiology, 113*(6), 767–791.",
]

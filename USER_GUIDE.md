# XBCI User Guide

This guide is for anyone using the app, with no programming needed. It assumes the app is already running
(see the README).

## 1. First minute

1. In the **sidebar**, check that the artifact folder is found (the header shows "artifacts loaded").
2. Choose a **subject**. Every person has their own two models, so the subject decides which models you are using.
3. Choose an **explanation level**. If you are new to EEG, pick **Beginner**.
4. Open **Start here** and press **Play the whole tour** for a two-minute picture of how a BCI decides.

## 2. Words you will meet

* **EEG:** voltages measured by electrodes on the scalp.
* **Trial:** one 4-second recording after a cue telling the person which movement to imagine.
* **Class:** the imagined movement (left hand, right hand, feet, tongue).
* **Rhythm / band:** brain activity at a range of speeds; mu (8-13 Hz) and beta (13-30 Hz) change when you imagine movement.
* **Topographic map:** a top view of the head (nose up); colour shows a value at each electrode.
* **Probability:** how strongly the model favours each class; the four add up to 100 %.

A longer glossary is in **Notes & Blog**.

## 3. The tabs

### Playground: "Why did the model say that?"
1. **Step A:** pick EEGNet or CSP + LDA.
2. **Step B:** pick an input.
   * *Recorded trial:* choose train, validation or test data, optionally a class, or press the dice for a random one.
   * *Upload my own EEG:* a `.npy`, `.npz` or `.csv` file shaped channels x samples with the 22 electrodes in the order shown. You can ask the app to filter and resample it for you.
   * *Synthetic generator:* build a signal from numbers (frequency, strength, time window, electrodes).
   * *Modify it (optional):* change the signal in time (gain, shift, noise, mirror left/right), frequency (raise or lower a band) or time-frequency (change one band in one time window).
3. **Step C:** press **Run the model and explain**. Nothing is recomputed until you press it.
4. Read the six steps from top to bottom: the signal, the model's recipe, what happens inside, where the evidence was, whether it makes neuroscientific sense, and how much to trust it.

There is also a mode where you type the eight CSP feature values yourself (CSP + LDA only).

### Compare models: "Do they agree, and do they think alike?"
Choose a signal and press **Run both models and compare**. You get a verdict (agree or not), each model's
probabilities, the weights and features that decided each answer, evidence maps side by side, five similarity
gauges, and statistics over all saved test trials. Open "How should the two explanations be aligned?" to explain the
true class, or one class for both models, which makes the comparison fairer when they disagree.

### Under the hood
The real numbers the models learned: temporal filters, spatial patterns, mixing and classifier weights for EEGNet; spatial patterns and weight table for CSP + LDA; training curves; and a browser where you can inspect any stored tensor and download it as CSV.

### Performance
Accuracy per person (dashed line = chance), confusion matrices (which classes are mixed up), a calibration curve (does "80 % sure" mean right 80 % of the time?) and how often the two models agree.

### Science Q&A
Pick a question from the list, for example "Which electrodes matter most when removed?". Questions 6 to 9 run a live experiment on the test data; the first run takes a little while, then it is cached.

### Notes & Blog, About
Articles, videos, glossary and references; and the technical record (versions, settings, checkpoints, file-integrity check).

## 4. Five things to try

1. **Left versus right hand.** In Playground, choose a left-hand test trial. Does the evidence map light up on the right side of the head?
2. **The mirror test.** Tick "Mirror left and right" in the time-domain editor and run again. A model that uses left/right patterns should change its answer.
3. **Remove a rhythm.** In the frequency editor set **mu** to -12 dB and run. Watch the "effect of your edits" bars.
4. **Same signal, two models.** In Compare, pick a trial where they disagree and compare the evidence maps.
5. **A hard subject.** Switch the sidebar to subject 5, then look at Performance and Science Q&A (question 11).

## 5. Reading the results carefully

* Accuracy has a margin of about plus or minus 5 to 6 points; do not over-read small differences.
* An explanation shows what the model used, not proof of what the brain did.
* If you edited the signal, the model has never seen anything like it; read the answer as "how it reacts".
* This is a teaching and research tool, not a medical device.

## 6. If something goes wrong

| Problem | What to do |
|---|---|
| Welcome screen instead of the app | The folder path is wrong. It must contain `models/` and `reference/`. Paste it into the sidebar box. |
| A subject is missing | Both of that person's models must be finished in the artifact folder. |
| First run is slow | Explanations take a few seconds on a normal laptop; later runs are instant. |
| Upload rejected | Check the shape (22 channels), the file type and the unit setting. The message names the problem. |

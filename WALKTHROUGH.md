# Walkthrough

A tour of the code in the order the pipelines run. Sections marked
*(plain English)* are for readers who are not machine-learning specialists.

- **Scania Component X**, the main case: `mfp/scania/`, run with
  `python -m mfp.scania.pipeline`.
- **NASA C-MAPSS**, the benchmark: `mfp/pipeline.py`, run with
  `python -m mfp.pipeline`.

Shared modules: `mfp/config.py` (every setting and assumption),
`mfp/models.py` (model builders, tuning, NASA score), `mfp/explain.py` (SHAP),
`mfp/report.py` (generated README sections) and `mfp/dashboard.py` (the logic
behind the Streamlit app).

Two rules hold everywhere, and tests check both:

1. **Test data never chooses anything.** Models, calibration, thresholds and
   decision rules are chosen on training or validation data; each test set is
   scored once. Scrambling the test data does not change any choice.
2. **Features only look backwards.** A feature for a readout or cycle uses only
   data at or before it. Deleting later data does not change earlier features.

---

## Scania Component X

### Loading: `mfp/scania/data.py`

Nine CSVs: readouts, specifications and outcomes for the training, validation
and test sets. Values are read as 32-bit floats and cached as parquet in
`data/scania/cache/`. Loading stops with a clear message if a file or column is
missing, a (vehicle, time step) pair repeats, or a split's files disagree on
vehicle IDs.

### Labels and censoring *(plain English)*: `mfp/scania/labels.py`, `mfp/scania/horizons.py`

For each training vehicle we know how long it was watched and whether the
component was repaired at the end.

- **Repaired vehicles**: the repair date is known, so every readout has an
  exact time left and therefore a class.
- **Vehicles not repaired**: the study simply ended, so a failure, if any,
  comes later. The true time left is at least what we can compute. More than
  48 steps before the end, the readout is certainly class 0; closer to the end
  its class is unknown.

The five-class labels drop the unknown readouts. The horizon labels ask four
yes/no questions instead ("fail within 6, 12, 24, 48 steps?"). A readout 30
steps before the end of an unrepaired vehicle's study is unknown for the
48-step question but a known "no" for the 6, 12 and 24-step ones, so the
horizon models keep it for those.

`time_left_to_class` holds the class boundaries; every window includes its
upper edge (48 is class 1, 6 is class 4).

### Features *(plain English)*: `mfp/scania/features.py`

Each readout carries 14 anonymised variables, all running totals:

- **Counters** work like an odometer. The value shows how much a vehicle has
  been used; the rate of change over the last readouts shows how hard it is
  used at the moment.
- **Histograms** count how often a measurement fell into each range (bin). Raw
  counts grow with age, so each bin becomes a share of the total, the total is
  kept separately, and the change in each share shows a drift in how the
  vehicle is used.
- **Specifications** become one-hot columns. Categories come from training
  vehicles; an unseen category becomes all zeros.
- Optional groups: usage intensity (totals per time step), rates over the last
  10 readouts, and each counter's intensity relative to vehicles with the same
  Spec_0 (medians from training rows only).

Training uses one row per readout. Validation and test vehicles get one row
each, built from all their readouts, because their label belongs to the last
one.

### Model selection: `mfp/scania/selection.py`

The official validation set has too few at-risk vehicles to separate methods
reliably, so selection also uses a **simulated validation set**: each training
vehicle is cut at one seeded random readout, the same way the official sets
were made, and its class follows the training labelling rules (vehicles whose
class is unknown at that point are skipped). Predictions for it come from 5
vehicle-grouped folds, so no vehicle is predicted by a model that saw it.

Candidates: the five-class gradient boosting model, the horizon gradient
boosting models (with and without equal weight per vehicle, and with each
optional feature group), and logistic regression versions as baselines.
Weighting and feature groups are kept only if they lower the simulated cost.
The headline is the gradient boosting candidate with the lowest simulated cost;
the official validation set is reported next to it. Candidates are compared on
a seeded share of training rows to keep the run time reasonable; the chosen
model is then trained on all rows.

Vehicle weighting is done by giving each vehicle the same number of rows
(`balanced_rows`). In expectation this equals per-vehicle sample weights, and
scikit-learn's gradient boosting trains on it much faster than on weighted
binning.

### Decisions and calibration *(plain English)*: `mfp/scania/evaluate.py`

Scania prices every mistake: an unneeded check costs 7 to 10, a late or missed
failure 200 to 500. The **lowest-expected-cost** rule picks, for each vehicle,
the answer with the lowest average cost under the model's probabilities. If a
vehicle is 90% class 0 and 10% class 1, answering "0" costs 0.9 × 0 + 0.1 ×
200 = 20 on average and answering "1" costs 0.9 × 7 = 6.3, so the rule says
"1". This is why any vehicle with a small risk gets flagged.

**Calibration** makes the probabilities match reality before that rule is
applied:

- *Prior shift*: training rows over-represent late readouts of repaired
  vehicles, so the class shares differ from the official sets. Each class
  probability is multiplied by (class share in the simulated validation set ÷
  class share in the training rows) and the row is renormalised. There is no
  search, so nothing is tuned to the official validation set.
- *Isotonic calibration*: a monotone map from predicted to observed at-risk
  rate, learned from out-of-fold training predictions.

**Workshop capacity.** The cost matrix assumes every flagged vehicle can be
inspected. With a capacity of k%, only the riskiest k% are flagged, and each
gets the class 1-4 of lowest expected cost. The recommended capacity is the
smallest one whose simulated-validation cost is within 5% of the unconstrained
cost. It trades a little cost for far fewer false alarms.

The test set is scored once, for every model and capacity. A bootstrap
resamples the test vehicles 1,000 times and reports the middle 95% of each
cost, of each saving against always class 0, and of the difference between
models. An interval that includes zero means the test set cannot separate the
two.

### SHAP *(plain English)*: `mfp/explain.py`

SHAP answers "why did this vehicle get this risk?" It splits the prediction into
one contribution per feature that add up to the prediction.

The headline model has five classes, and shap's `TreeExplainer` explains each
class score exactly. The at-risk log-odds is a smooth function of those
scores, so each feature's per-class contributions are combined with the
gradient of that function averaged along the path from the base scores to the
vehicle's scores. The contributions then add up to the vehicle's at-risk
log-odds (the pipeline records the largest gap), and the prior shift enters
exactly as a constant offset per class. Every test vehicle is explained, with a
separate ranking for the vehicles flagged at the recommended capacity.

SHAP shows what the model relies on, not what causes failures, and the
variables are anonymised.

---

## NASA C-MAPSS

### Data, splits and regimes: `mfp/data.py`, `mfp/split.py`, `mfp/regimes.py`

Training engines run until they fail; NASA test engines stop early, and
`RUL_FD00X.txt` gives their true remaining life. The label is 1 when an engine
fails within 30 cycles. Training engines are split by whole engine into fit,
validation and lifecycle test sets. Sensors are z-scored per operating regime
(KMeans on the operating settings, learned from fit engines), so a reading
means "how unusual for this flight condition".

### Features and models: `mfp/features.py`, `mfp/models.py`

Rolling mean, standard deviation and trend over the last 10 cycles, looking
backwards only. Gradient boosting is the headline model and logistic regression
the baseline; each gets its own alarm policy.

### The lifecycle alarm *(plain English)*: `mfp/lifecycle.py`

Replay each engine's life cycle by cycle. The alarm rings the first time the
failure probability has stayed at or above a threshold for 1, 2 or 3 cycles in
a row. Each engine gets one verdict: good catch (10 to 50 cycles of warning),
early alarm (more warning than needed, a wasted check) or missed (no warning,
or less than 10 cycles). Cost = 10 × missed + early alarms; precision = good
catches / alarms that rang.

The policy (threshold and confirm cycles) is chosen on validation engines and
replayed on lifecycle test engines next to two fixed rules: a 0.5 threshold
that rings at once, and a 0.5 threshold with the same confirm cycles. The
second separates the effect of the threshold from the effect of confirming.
The sweep is repeated for minimum warnings of 20 and 30 cycles; a fixed 0.5
threshold often rings too late once more warning is required.

### NASA test benchmark and the NASA score *(plain English)*

The NASA test engines stop before failure, so they cannot be replayed. They
give the standard benchmark: PR-AUC, precision, recall and false alarms at the
last recorded cycle, and remaining-life predictions from a gradient boosting
regressor and a ridge baseline. RMSE treats "20 cycles too many" and "20 too
few" alike; the NASA score charges more for predicting too much life left,
e^(d/10) − 1 against e^(−d/13) − 1 with d = predicted − true, summed over
engines.

### SHAP for NASA

The NASA model is binary, so `TreeExplainer` explains its log-odds of failure
directly, at the last cycle of every NASA test engine.

---

## Reports and dashboard

`reports/` holds every number and chart. `mfp/report.py` writes the README
results from these files, so the README cannot disagree with them. The
dashboard reads only `reports/` and therefore deploys without raw data or
model files.

## Questions to be ready for

Scania:

- Why drop censored readouts for the five-class labels, and how do the horizon
  labels keep some of them?
- Why does the lowest-expected-cost rule flag so many vehicles? Walk through the
  90% / 10% example.
- What does the workshop capacity change, and what does it cost?
- Why build a simulated validation set, and why can it disagree with the
  official one?
- What does the prior shift correct, and why are searched class weights only a
  comparison?
- How do you know a cost saving is not luck? What does an interval that
  includes zero mean?
- Why does vehicle age dominate the explanations, and what does that say about
  the labels?

NASA:

- Why split by engine rather than by row?
- Why judge the alarm once per engine?
- What does fixed 0.5 with the same confirm cycles reveal?
- Why does the policy's threshold drop when more warning is required?
- How much would you trust a difference of one engine on a lifecycle test set
  of 20?

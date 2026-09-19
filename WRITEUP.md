# RAILPULSE — Predictive Fault Detection for Rolling Stock

**Problem Statement 3 · Nebula X Hackathon 2026**

---

## The solution

RAILPULSE detects wear and anomaly signatures in rolling stock and flags them
before they become unplanned service disruptions. It is two things: a
detection pipeline, and — the part we spent most of our effort on — an
interface that turns what the pipeline produces into something an operator or
a new engineer can actually act on.

### Detection

Three subsystems are modelled per train, each with its own model and its own
calibrated thresholds:

| Subsystem | Signals modelled |
|-----------|------------------|
| Door | cycle time, motor current, motor temperature, motor speed |
| Bogie | axle bearing temperature, vibration RMS, traction motor temperature and speed, friction brake capacity |
| Car | HVAC supply temperature, HVAC current, lighting load, battery state of charge, battery voltage, radio signal strength |

Each subsystem gets a **GRU sequence autoencoder** (hidden 32, latent 12,
48-reading window) trained **only on healthy units**. The model learns what
normal looks like; degradation shows up as reconstruction error, because the
model has never been taught to rebuild it. Thresholds are the 97.5th and
higher percentiles of reconstruction error on *held-out healthy* units, so
"Caution" means "rarer than 97.5% of healthy behaviour", not a hand-picked
number.

That autoencoder score is fused with a conventional rolling z-score baseline
into four risk levels — Normal, Caution, Warning, Critical — each with a
recommended action.

Crucially, the fault types in our generator only move the channels they
mechanically implicate: a comms antenna fault does not touch battery charge,
a brake fault does not heat the bearings. The model therefore has to learn
which *combination* of channels moves together, rather than watching any
single one cross a line.

### Validated results

Measured on 11 ground-truth faults injected into a 36-unit, 6-train fleet:

- **11 of 11 detected before the fault was logged**
- **4.7 days average lead time**
- An ablation against the rolling z-score baseline shows the autoencoder
  produces a **sustained** alert in 11/11 cases. The baseline produces a
  sustained alert in **0/11** — its apparent lead time comes entirely from
  isolated chance spikes at a fixed 3σ cutoff, compounded over thousands of
  readings.

We publish that ablation in the dashboard rather than hiding it. A detection
only counts as genuine if the alert held for ≥70% of the remaining pre-fault
readings; a one-off spike is noise, not early warning.

### Real-data models

Alongside the synthetic pipeline, two models trained on the **actual PS3
datasets** are served from separate endpoints:

- **Door** — the stream is split into door cycles at gaps > 0.1 s, each cycle
  reduced to 24 features, and an autoencoder flags cycles exceeding the
  calibrated threshold as abnormal resistance.
- **Rail corrugation** — 129 recording columns reduced to 387 rms/std/max
  features, classified Normal / Side I / Side II by a 300-round XGBoost model.

---

## What makes it different

**1. It answers the second half of the problem.** The FAQ asks how this
information reaches an operator making a split-second call, or an engineer who
has never seen these parameters. Our answer is a **role switch** — Operator,
Engineer, Analyst — that re-levels the entire dashboard. The operator gets
triage, a digital twin, work orders and a live feed; the engineer adds the
decision trace, per-signal deviations and model behaviour; the analyst gets
validation, threshold tuning and the business case. Same data, same
endpoints, same scores — different reader.

**2. It leads with a sentence, not a number.** "Reconstruction error 62.5"
means nothing to an operator. Every unit leads with a plain reading built from
its own primary signal — *"Brake capacity on BOGIE_04_1 is reading 79.8%,
33.3σ below the healthy average for a bogie"* — with the numbers directly
underneath, demoted but never removed. Every piece of jargon carries its
definition on hover.

**3. A Health Index, because reconstruction error is not comparable.** Each
autoencoder is trained and calibrated separately, so a door's 8.0 and a car's
8.0 mean different things and a fleet table cannot rank on them. The Health
Index restates the same value 0–100, which is comparable, and it never appears
without the raw number it came from.

**4. A chatbox that cannot invent a number.** The Fleet Assistant answers
questions in plain language, pitched at whoever is asking. Its tools call the
*same functions that serve the dashboard*, so an answer can never disagree
with what is on screen. It has no write path, it cannot change a risk level,
and its prompt requires any mechanical cause to be worded as a hypothesis for
a technician to confirm — because the pipeline detects a statistical
deviation, not a diagnosis. Asked whether a Critical door could wait until
tomorrow, it declined to clear the unit for service.

**5. It tells you when it should not be trusted.** Upload your own CSV and it
is scored against the existing models — but the response carries a
distribution check comparing each signal's mean to the population the scaler
was fitted on. Past ±3σ it warns that the risk levels are not trustworthy and
the pipeline needs retraining, rather than returning a confidently red fleet.

**6. A business case that refuses to flatter itself.** The generator seeds
faults into 11 of 36 units in six weeks so validation has something to find.
Extrapolating that rate linearly implies ~398 failures per 100 units per year
and produces a fantasy number. The failure rate is therefore an *input* you
supply from your own reliability history; the seeded rate is disclosed next to
it. The model also subtracts false alarms and the cost of running the system,
because a case that only counts the upside is not a case.

**7. The design is domain-grounded.** Status colours are railway signal
aspects — green, yellow, double yellow, red — which is the exact semantics
needed and readable without a legend. Warm drafting-paper ground, condensed
transit-signage typography, monospace instrument readouts, hairlines instead
of drop shadows.

---

## Tech stack

**Models and pipeline** — Python 3.12 · PyTorch (GRU sequence autoencoders) ·
scikit-learn (scaling, baseline statistics) · XGBoost (real-data rail
classifier) · pandas / NumPy / SciPy · Matplotlib (validation plots)

**API** — FastAPI · Uvicorn · Pydantic. Serves fleet status, per-unit decision
traces, on-demand scoring, an ablation summary, a simulated live feed, and the
two real-data prediction endpoints.

**Dashboard** — vanilla HTML / CSS / JavaScript with Chart.js. No framework,
no build step: one file, served directly. The digital twin is hand-built SVG
generated from live per-signal deviations, so it lights the specific system —
the HVAC, not the whole carriage.

**AI layer** — Anthropic Claude Opus 5 for the Fleet Assistant and the
Maintenance Copilot; Exa for literature search. Both keys live server-side;
the browser never holds one. Every failure path returns a reason rather than
an error, so a missing key degrades a feature instead of breaking the page.

**Deployment** — Docker, single container serving both API and dashboard on
one origin, deployed to Google Cloud Run. The image generates the dataset and
trains the models at build time, so the container serves immediately with no
cold-start training.

**Schema portability** — `pipeline/schema_config.py` is the single file
mapping column names to a data source. Swapping a real TCMS or depot SCADA
feed in should not require touching the model or dashboard code.

---

## Honest limits

The fleet dashboard runs on a **synthetic placeholder dataset** built to be
physically plausible, not on real measurements — the page says so and so does
every figure derived from it. The real-data models are served separately and
score files you upload. Lead time and detection rate are measured on seeded
faults, which are easier than real ones. And the system detects statistical
deviation from healthy behaviour: a cause is always a hypothesis for a
technician to confirm.

Every judgment call we made is documented in `ASSUMPTIONS.md`.

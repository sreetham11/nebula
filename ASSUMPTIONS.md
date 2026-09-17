# Assumptions & judgment calls

This log covers everything I decided without checking in, so you can review
and override anything before the real dataset lands. Grouped by phase.

## Phase 1 — synthetic data generator (`data_gen/`)

- **Fleet size / topology**: 6 trains x 2 doors x 2 bogies = 12 door units,
  12 bogie units. Arbitrary but gives a believable small-fleet hackathon
  scale and a clean `train_id` grouping for the dashboard. Change
  `N_TRAINS`/`DOORS_PER_TRAIN`/`BOGIES_PER_TRAIN` in `data_gen/config.py`.
- **Time range**: 6 weeks starting 2026-08-01, readings every 5-15 min
  (jittered per unit, independent clocks). Produces ~145k total rows,
  inside the requested 50k-200k band and fast to regenerate (~2s).
- **4 degrading units per subsystem** (of 12), chosen uniformly at random.
  This gives 8 fault_log events total — enough to validate lead time
  per-fault without making "most units are healthy" untrue.
- **Degradation shape**: piecewise ramp/plateau trend (4-7 random
  segments, 30% plateau probability) plus additive noise and occasional
  "backslide" dips, normalized to a clean [0,1] wear curve, then scaled
  per-signal into real units (seconds, amps, °C, RMS). This was a
  deliberate design to make the anomaly, specifically avoid a fixed
  threshold trivially catching it. See `_build_normalized_trend` in
  `data_gen/generate_data.py`.
- **Bug I caught and fixed mid-build**: my first version of
  `_apply_noise_to_trend` added trend-independent noise (with a
  clip-at-zero bias) across a degrading unit's *entire* timeline, not just
  its injected window — so "healthy" history before the real onset looked
  subtly anomalous too, and the autoencoder picked up on unit identity
  rather than actual drift onset (lead times were coming out at 250-770h,
  often longer than the degradation window itself). Fixed by gating all
  noise/backslide terms to the degradation window (`in_window_mask`); after
  the fix, lead times fell into a plausible 48-254h range and pre-window
  readings for degrading units are statistically identical to healthy
  units. If you regenerate data with further tweaks to the noise model,
  spot-check a degrading unit's early readings the way I did here (compare
  raw values against a healthy unit's) before trusting lead-time numbers.
- **Fault trigger point**: logged when the *underlying clean trend*
  (not the noisy signal) first crosses a threshold randomized per-unit in
  [0.55, 0.95] of its final drift magnitude — mimics faults getting caught
  at varying severity rather than always at 100% wear. Severity (1-5) is
  derived from that crossing fraction plus noise.
- **Vibration degradation** is modeled as increasing *variance* (std
  multiplier up to 2-4x) plus a small mean creep, per the prompt's "vibration
  variance increasing" cue, rather than a large mean shift like temperature.
- **Door cycle_count**: cumulative counter incremented per reading based on
  a peak-hours (06:00-23:00) vs off-peak usage pattern; not tied to
  degradation.
- **No explicit train-level correlation** between a train's door and bogie
  health — each unit degrades independently. Real fleets might show
  correlated wear (e.g. one rough route wearing all subsystems on a train
  faster); skipped for scope.

## Phase 2 — detection pipeline (`pipeline/`)

- **Baseline model**: rolling window of 144 readings (~1 day at ~10 min
  cadence), min 30 periods, |z| > 3 flags. Tunable in `model_config.py`.
- **Autoencoder**: single-layer GRU, hidden size 32, latent size 12,
  window length 48 readings (~8 hours), trained up to 25 epochs w/ early
  stopping on validation loss (patience 5) on Adam, lr 1e-3. These are
  reasonable hackathon-scale defaults, not tuned — the model's val loss
  plateaus fairly high (~0.9 for doors, ~0.67-0.68 for bogies, vs. 1.0 for
  standardized noise), meaning it's a fairly weak reconstructor of fine
  noise structure. It still works for this task because degradation drift
  is large relative to noise, but if the real dataset has much subtler
  anomalies, expect to need more capacity/epochs/tuning here.
- **Train/validation split is by UNIT, not by row/window** (fixed after an
  initial pass got this wrong — see "Bug fixed after initial review"
  below). Of the 8 healthy units per subsystem, `AE_VAL_UNIT_FRACTION =
  0.25` in `model_config.py` holds out 2 units for validation and trains
  on the remaining 6 (`data_loader.split_healthy_units`, seeded). The
  scaler (`StandardScaler`) is also fit only on the 6 training units, so
  nothing about the 2 held-out units — not their scale, not their
  noise — touches training. Early stopping (patience 5, already present
  before this fix) tracks loss on the held-out units' windows.
- **Anomaly thresholds are calibrated only from the 2 held-out validation
  units' reconstruction error** (90th/97.5th/99.5th percentile for
  Caution/Warning/Critical), scored fresh after training completes —
  never from the training units' own error, which the model has already
  fit and would understate true error. This was also wrong in the initial
  pass (thresholds were computed from all 8 healthy units' dense-scored
  error, which mostly meant "in-sample" since those were the same rows
  used for training); now fixed.
- **Final numbers after the fix** (regenerate via `pipeline/run_pipeline.py`
  — these will shift slightly run-to-run only if `RANDOM_SEED` or the data
  change, both fixed by seed here):

  | Subsystem | Train units | Val units | Train loss | Val loss | Caution | Warning | Critical |
  |---|---|---|---|---|---|---|---|
  | door  | 6 | 2 | 0.9155 | 0.9089 | 1.099 | 1.219 | 1.334 |
  | bogie | 6 | 2 | 0.6705 | 0.6755 | 0.811 | 0.884 | 0.938 |

  All 8/8 ground-truth faults are still detected before their logged
  timestamp after this fix, lead times 48.2–251.4 hours — within a couple
  hours of the pre-fix (leaky) numbers in this synthetic dataset, because
  the 8 healthy units per subsystem are literally IID draws from the same
  generator, so unit identity carried little extra signal for the model to
  overfit to. That won't necessarily hold for the real dataset, where
  distinct physical units may have real systematic differences — which is
  exactly why the by-unit split matters going forward, even though it
  didn't move the numbers much here.
- **Bug fixed after initial review**: the first version split by
  window/row (`rng.permutation` over all pooled healthy windows), so
  windows from the same unit — which overlap heavily in time at stride 4
  and share that unit's specific noise realization — landed in both train
  and validation. Thresholds were then computed from all 8 healthy units'
  scored error, most of which the model had already trained on. Neither
  was numerically catastrophic here (see above), but both were
  methodologically wrong and would matter more on real data. Fixed by
  `data_loader.split_healthy_units` (unit-level) plus scoring thresholds
  only from the held-out units.
- **Fusion logic**: AE reconstruction error sets a base risk level from
  percentile bands; the baseline z-flag can escalate that by one level.
  Documented in a comment block at the top of `pipeline/fusion.py` per the
  brief, so it's easy to explain live.
- **Lead-time definition** in `validate.py`: first *sustained* crossing
  into Warning/Critical (>=70% of remaining pre-fault readings stay at or
  above Warning), not the first isolated blip — avoids crediting a single
  noisy spike as "detection." Falls back to the first isolated crossing if
  no sustained one exists.
- **Schema isolation**: all column names live in `pipeline/schema_config.py`
  only; `baseline.py`/`autoencoder.py`/`fusion.py` read `SUBSYSTEMS[...]`
  rather than hardcoding strings, per the requirement to make remapping to
  real data a one-file change.
- **Device**: auto-selects Apple Silicon MPS > CUDA > CPU
  (`autoencoder.get_device()`).

## Phase 3 — API + dashboard

- **FastAPI loads model/scaler/thresholds once at startup** from
  `models/*.pt` / `*.joblib` / `*_meta.json` (written by
  `pipeline/run_pipeline.py`) — the API does not retrain, it only scores.
  Rerun the pipeline after regenerating data before restarting the API.
- **`/predict-anomaly`** expects at least `window_length` (48) readings for
  the target subsystem. Its baseline z-score now uses each unit's
  **persisted** rolling mean/std (`BASELINE_STATS`, built at startup in
  `_build_baseline_stats()` from the same scored data — itself derived from
  `door_data.csv`/`bogie_data.csv` via `baseline.py` — that `/fleet-status`
  and `/unit/{id}` already read from), so a known unit's `baseline_flag`
  agrees with the dashboard regardless of what window a caller happens to
  submit; verified to match `/unit/{id}`'s stored `baseline_max_abs_z` to 6
  decimal places for the same unit. Response includes `baseline_source`
  (`"persisted_unit_history"` or `"request_window_fallback"`) and
  `baseline_stats_as_of` so callers can tell which was used. **Bug fixed
  after initial review**: the first version computed baseline z purely from
  the submitted window's own mean/std, which had two problems — (1) it
  couldn't agree with the dashboard for the same unit at the same moment,
  since a caller's window and the pipeline's full rolling history produce
  different statistics, and (2) it was blind to drift for a *flat* submitted
  window (zero internal variance always scored z=0 regardless of how far
  the unit's true baseline had drifted) — confirmed via a flat 48-reading
  window at cycle_time_sec=6.0 for a unit whose real baseline is ~4.2s:
  the old logic would score this z=0; the fixed version correctly returns
  baseline_max_abs_z=13.24, Critical. The persisted stats are a startup-time
  snapshot (same as the dashboard's), not updated by `/predict-anomaly`
  calls themselves — a unit with too little history for a rolling window
  (or an unrecognized `unit_id`) falls back to the old per-window
  calculation, clearly labeled via `baseline_source`.
- **`/live-feed`** is a stateless simulation: a single in-memory cursor
  walks through the already-generated historical data in timestamp order,
  wrapping at the end, advancing a few rows per poll. It is not live data
  and is not tied to per-viewer sessions — fine for a single-demo hackathon
  tool, not for concurrent multi-user use.
- **Dashboard**: single static HTML file, vanilla JS + Chart.js (via
  cdnjs), no build step, dark theme. Polls `/fleet-status` every 15s and
  `/live-feed` every 1.8s. `API_BASE` is hardcoded to
  `http://localhost:8000` at the top of the `<script>` block — change it if
  you serve the API elsewhere.
- **Tech choice**: plain HTML/JS over a framework, per the brief's "prioritize
  build speed over framework sophistication."
- **Business Impact Calculator** (top of dashboard, spans full width above
  the fleet overview): added a `GET /validation-summary` endpoint that just
  reads `validation_outputs/lead_time_summary.csv` (no pipeline changes) and
  returns detected-fault count + mean lead time. The dashboard panel is
  otherwise self-contained client-side JS — three editable inputs (defaults:
  $50,000/failure, $5,000/maintenance, 10x fleet multiplier) recompute on
  every keystroke with no network round-trip. Two numbers are shown:
  "observed" (N faults x (cost_failure - cost_maintenance), at the current
  24-unit synthetic fleet size over the actual 6-week window) and
  "annualized" (observed rate scaled from 6 weeks to 365 days, then x the
  fleet multiplier). This is a simple linear extrapolation, not a
  statistical forecast — flagged as directional in the panel's own caption,
  since 8 faults over 6 weeks is a small sample to annualize from.
- **Detection Sensitivity slider** (small panel directly above Fleet
  Overview): required exposing the raw held-out healthy reconstruction-error
  array, not just the three fixed percentile thresholds already in
  `*_meta.json` — added one line to `run_pipeline.py` saving
  `models/{subsystem}_val_recon_errors.npy` (the array was already computed
  during training, this just persists it; no modeling logic changed, and
  the pipeline was rerun once to produce the file — outputs came out
  bit-identical since it's seeded). `GET /sensitivity-scan?percentile=X` in
  the API recomputes `np.percentile()` on that array plus boolean
  comparisons against the already-scored CSVs and the same sustained-vs-
  fallback crossing logic from `ablation.py` (imported directly, not
  duplicated) — no model inference, ~70ms per call, safe to call on every
  slider tick (debounced 120ms client-side regardless). Slider range is
  90-99.5th percentile per the brief's example; defaults to whatever
  percentile `model_config.RECON_ERROR_WARNING_PERCENTILE` (97.5) already
  uses for the "Warning" threshold, fetched from the API's own default query
  param rather than hardcoded in the dashboard. In this dataset all 8/8
  faults are caught across the entire 90-99.5 range (degradation drift is
  large enough relative to noise that even the most conservative threshold
  still catches everything) — so the visible tradeoff here is alert volume
  (32,531 at 99.5th vs 42,945 at 90th) and lead time (5.8d vs 6.8d) rather
  than catch-rate; a "faults missed" case would need a much more
  conservative percentile than the brief's suggested range, or a dataset
  with subtler degradation.
- **Fleet Digital Twin diagram** (centerpiece panel, full width, between
  Business Impact and the two-column layout): purely client-side, no new
  API endpoint — built from the same `fleetCache` array `/fleet-status`
  already populates. Our schema has exactly 2 doors + 2 bogies per train
  (`data_gen/config.py`: `DOORS_PER_TRAIN=2`, `BOGIES_PER_TRAIN=2`), so
  rather than fabricate a "3-4 car" consist not backed by real units, each
  train is drawn as a 2-car consist with one door + one bogie per car —
  every shape on screen maps 1:1 to a real synthetic unit, sorted by
  unit_id into car slots so DOOR_0X_1/BOGIE_0X_1 always land on car 1. The
  layout logic (`groupUnitsByTrain`) doesn't hardcode the count of 2,
  though — it uses `Math.max(doors.length, bogies.length)` per train, so a
  future dataset with more doors/bogies per train would grow more car
  slots automatically rather than silently dropping units.
  SVG is built once (`buildTrainDiagram`, keyed off `trainDiagramBuilt`)
  and every subsequent `/fleet-status` poll (same 15s cycle as the rest of
  the dashboard) only patches each shape's `fill` in place
  (`updateTrainDiagramColors`) — verified via Playwright that the SVG DOM
  node persists (not rebuilt) across a poll. Clicking a shape calls the
  existing `selectUnit()` used by the fleet table (not a new/duplicate
  detail view) and auto-scrolls the existing Unit Detail panel into view;
  selection state is mirrored as a blue outline on the shape via the same
  `selectedUnit` variable the table highlighting already uses.
- **Maintenance Actions panel** (below Fleet Overview, above Live Feed):
  purely client-side, no new API endpoint and no backend/browser storage —
  a plain in-memory JS array (`workOrders`) that resets on page reload,
  same as every other piece of dashboard state (`fleetCache`,
  `validationCache`, etc.). On every `/fleet-status` poll,
  `generateWorkOrdersFromFleet()` opens one work order for any unit
  currently Warning/Critical that doesn't already have one
  (`workOrderByUnit` dedup map) — since `fleet_status.csv` is a static
  snapshot in this demo, all 9 currently-flagged units (8 Critical + 1
  Warning) get their work orders on the very first poll rather than
  trickling in over a session; "crosses into Warning/Critical" is
  evaluated per-poll against current state, there's no transition-history
  tracking, so this is the honest behavior for a static dataset rather
  than a limitation to fix.
  Priority is a direct mapping (Critical->Urgent, Warning->Medium).
  Description text is generated from `subsystem_type` + live
  `ae_recon_error`/`risk_level` (generic "possible motor wear / seal
  degradation / track misalignment" for doors, "possible bearing wear /
  suspension wear / wheel flat" for bogies) rather than the dataset's
  ground-truth `fault_type` — deliberately, since a real alerting system
  wouldn't know the true fault type before inspection either; using
  ground truth here would have been a bit of a demo cheat.
  State machine is intentionally minimal per the brief ("don't
  over-engineer"): Pending Assignment shows both Acknowledge and Schedule
  buttons, Acknowledged shows only Schedule, Scheduled is terminal (no
  more buttons) — verified via Playwright that clicking through both
  transitions updates the badge and that a second `/fleet-status` poll
  does not spawn duplicate work orders for the same 9 units.
- **Terminology pass**: replaced a handful of generic terms with correct
  rail-industry ones in headers/pitch-facing copy only, not tooltips or
  detail views — "Fleet Digital Twin" -> "Rolling Stock Digital Twin"
  (diagram panel header), "real network" -> "rolling stock fleet" (Business
  Impact multiplier label), "work order systems" -> "CMMS (Computerized
  Maintenance Management System)" (Maintenance Actions caption), "synthetic
  telemetry" -> "synthetic onboard telemetry" (Live Feed caption). Added a
  new **About & Architecture** panel (full width, end of `<main>`, before
  the footer) since none existed — the brief's SCADA/TCMS-integration
  mention needed a natural home, and a live JSON API response field isn't
  pitch-facing UI copy the way a panel someone reads during a demo is. Left
  `data_gen`'s fault-type taxonomy (`motor_wear`, `track_misalignment`,
  etc.), the ML-specific labels in decision traces/tooltips ("reconstruction
  error", "baseline check"), and matplotlib validation-plot text alone —
  none of those are generic terms with a better rail-specific equivalent,
  and the brief was explicit about not forcing jargon into every tooltip.

## General

- Used Python 3.12 in a dedicated venv (`venv/`) rather than the system's
  Python 3.14, since PyTorch wheels aren't yet published for 3.14.
- All synthetic-data disclaimers are deliberately loud (top-of-file
  docstrings, a persistent dashboard banner, API response fields) so
  nothing here gets mistaken for real fleet data later.

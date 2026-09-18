# NEBULA X — Predictive Fault Detection

Hackathon prototype for LTA Singapore: detects wear/anomaly signatures in
rolling stock and flags them before they become unplanned service
disruptions. Three subsystems are modelled per train, each with its own
autoencoder and its own calibrated thresholds:

| Subsystem | Unit        | Modelled signals |
|-----------|-------------|------------------|
| `door`    | `DOOR_xx_n` | cycle time, motor current, motor temperature, motor speed |
| `bogie`   | `BOGIE_xx_n`| axle bearing temperature, vibration RMS, traction motor temperature, traction motor speed, friction brake capacity |
| `car`     | `CAR_xx_n`  | HVAC supply air temperature, HVAC current, saloon lighting load, battery state of charge, battery voltage, radio signal strength |

Each fault type moves only the channels it mechanically implicates -- a
comms antenna fault does not touch battery charge, and a brake fault does
not heat the bearings -- so the model has to learn which *combination* of
channels moves together rather than watching any one of them cross a line.
See `DEGRADATION_SIGNAL_DELTAS` in `data_gen/config.py` for the full map.

**Built entirely against synthetic placeholder data** (see
`data_gen/generate_data.py` docstring) since the real dataset drops during
the event. In production, the pipeline is designed to take its input from
each train's TCMS (Train Control and Management System) and/or depot-side
SCADA feeds instead — `pipeline/schema_config.py` is the single file that
maps column names to a real data source, so swapping one in shouldn't
require touching the model or dashboard code. See `ASSUMPTIONS.md` for
every other judgment call made along the way.

## Setup

```bash
python3.12 -m venv venv        # torch wheels aren't on 3.14 yet
source venv/bin/activate
pip install -r requirements.txt
```

## Run everything, in order

```bash
# 1. Generate synthetic data -> data/door_data.csv, bogie_data.csv, car_data.csv
python data_gen/generate_data.py

# 2. Train models, score the fleet, run validation -> models/, data/*_scored.csv,
#    data/fleet_status.csv, validation_outputs/*.png
cd pipeline && python run_pipeline.py && cd ..

# 3a. Start the API (from api/, with venv active)
cd api && uvicorn main:app --reload --port 8000

# 3b. In another terminal, serve the dashboard
cd dashboard && python3 -m http.server 8080
# open http://localhost:8080
```

## Reading the numbers: Health Index vs reconstruction error

Reconstruction error is what the model produces and stays the number of
record, but on its own it is unreadable and **not comparable across
subsystems** -- each autoencoder is trained and calibrated separately, so a
door's 8.0 and a car's 8.0 mean different things.

`pipeline/health_index.py` restates it as a bounded **0-100 Health Index**,
anchored on that subsystem's own calibrated thresholds:

| Reconstruction error | Health Index |
|----------------------|--------------|
| at or below the healthy fleet median | 100 |
| at the Caution threshold  | 75 |
| at the Warning threshold  | 60 |
| at the Critical threshold | 40 |
| 10x the Critical threshold or worse | 0 |

Interpolation is linear in `log(error)`, because reconstruction error is a
squared quantity with a long right tail. Two consequences worth knowing:
the band boundaries land on the same index value for every subsystem, so 58
on a door and 58 on a car mean the same severity; and the index is monotone
in reconstruction error, so ranking by either agrees within a subsystem.

It is a presentation layer. It adds no information and **never changes a
risk level** -- risk still comes from `pipeline/fusion.py`. The dashboard
therefore never shows the index without the raw error and the threshold it
was measured against sitting next to it.

### Per-signal deviation

The API also reports each channel's deviation from the healthy fleet,
**signed so that positive always means deteriorating**: brake capacity 4σ
*below* the healthy mean and bearing temperature 4σ *above* it both report
`+4.0`. Which direction is bad for which channel lives in `SIGNAL_METADATA`
in `pipeline/schema_config.py`.

One caveat the dashboard states in-place: the reference is the *pooled*
healthy distribution, which spans every duty state a unit passes through. A
train stabled overnight genuinely shows low traction motor speed and a cool
traction motor, so ±1-2σ on a single channel is ordinary time-of-day
variation. Below 2σ the decision trace says outright that no single channel
stands out and that the detection rests on the combination.

## Score your own data (CSV / Parquet upload)

The dashboard's **Score Your Own Data** panel takes a dropped file and runs
every unit in it through the models already on disk. `POST /upload`, handled
by `api/ingest.py`.

**It does not retrain**, and nothing is persisted -- the synthetic fleet the
rest of the page reads is untouched. Column names are matched case- and
separator-insensitively against an alias table, so `Motor Current (A)` and
`motor_current_amps` both resolve; `GET /upload-schema` returns the expected
columns and their aliases. An explicit `mapping` form field (JSON, schema
column -> your column) overrides all of it.

The important part is the **sanity check** it returns underneath the
results: each signal's mean in the uploaded file against the healthy
population the scaler was fitted on. If the upload is different equipment,
or the same signals in different units of measurement, the scaler's healthy
mean/std do not apply and everything will score as anomalous. Past ±3σ that
is surfaced as a warning above the results table rather than left for
someone to discover from a suspiciously red fleet.

```bash
curl -F "file=@your_data.csv" http://localhost:8000/upload
```

## Maintenance Copilot (optional, needs an Anthropic API key)

The Unit Detail panel has a **Maintenance Copilot** card that turns a unit's
existing decision trace into plain English: why it was flagged, a hypothesised
mechanical cause, and a suggested inspection. It is a single structured Claude
API call per unit -- no agent loop -- and it is entirely optional: with no key
configured the card shows "AI explanation unavailable" and everything else on
the dashboard, decision trace included, works exactly as before.

**The API key is server-side only.** It is read by the API process from
`ANTHROPIC_API_KEY`; the browser never holds it and never calls
`api.anthropic.com` -- it only calls this project's own `/copilot/{unit_id}`.

```bash
cp api/.env.example api/.env    # then paste your key into api/.env
                                # (git-ignored; or just export ANTHROPIC_API_KEY)
pip install -r requirements.txt # adds the `anthropic` SDK
```

Claude is sent **only numbers the pipeline already computed** for that unit --
unit ID, subsystem, train, risk level, reconstruction error, the threshold it
crossed, and each signal's current value against the healthy-fleet baseline the
autoencoder was trained on. The system prompt forbids inventing readings,
dates, or fault history, and requires everything be phrased as an AI-assisted
suggestion rather than a confirmed diagnosis.

## Investigate Fault (optional, needs an Exa API key)

The Unit Detail panel also has an **Investigate Fault** button that searches
published engineering literature for failure behaviour resembling what the unit
is showing, via the Exa search API. Also server-side only (`EXA_API_KEY`), also
optional -- without a key the panel shows "Research unavailable".

The query is built from the unit's real pipeline output, never hardcoded per
unit: the subsystem, whichever signal is genuinely most deviant from the
healthy-fleet baseline, and the wear mechanism that signal implicates. Two
bogies elevated on different signals therefore produce different searches, e.g.

    temperature-driven -> "train bogie bearing overheating axle bearing
                           temperature elevated signature predictive
                           maintenance condition monitoring"
    vibration-driven   -> "train bogie bearing wear vibration elevated
                           signature predictive maintenance condition
                           monitoring"

**Framing.** The "possible failure mechanisms" bullets are *extracted* from the
sources' own titles and Exa highlight spans by `api/research.py` -- no
generative model touches them, so no new claim can be introduced, and each
bullet is attributed to the domain it came from. The caption under the results
("similar failure behaviour has been documented in... they do not confirm,
prove, or diagnose a fault on this unit") is a fixed server-side constant that
no API response can override or omit.

## Maintenance Logistics (optional, needs a Google Maps API key)

Units the pipeline has escalated to **Warning or Critical** get a depot-routing
panel: recommended depot, real road distance, estimated travel time, and a
route map. Normal/Caution units don't show the section at all -- dispatching a
healthy unit would be a recommendation the detection system never made, and the
API enforces that gate too, so the panel can't be summoned by calling the route
directly.

> ### DEMO / SIMULATED LOCATIONS
>
> **The synthetic dataset contains no geolocation data of any kind** -- no GPS
> trace, no depot assignment, no position field. `api/locations_config.py`
> invents it: four real Singapore MRT depot sites (used so routing looks
> plausible on a map) and a fixed made-up position per demo train. A train's
> position never changes and no sensor feeds it.
>
> The distances and ETAs are **genuine Google Routes API results** -- computed
> between **invented points**. The dashboard renders this notice as a bold
> banner across the top of the panel, not a footnote. If a real deployment
> supplies actual positions, `locations_config.py` is the only file to replace.

```bash
# add to api/.env, then restart the API
GOOGLE_MAPS_API_KEY=...
```

Requires the **Routes API** enabled on the key, plus the **Maps Static API**
for the route image. Without the latter the panel still renders fully, just
without the map.

**Key containment.** The map image is proxied through this service's own
`/logistics/{unit_id}/map` endpoint rather than emitting a Static Maps URL into
an `<img src>` -- that would publish the key to every viewer. The browser never
sees a Google URL.

**Cost shape.** Two Routes calls on a unit's first view (one
`computeRouteMatrix` to rank shortlisted depots by real drive time, one
`computeRoutes` for the polyline), then cached indefinitely in
`api/logistics_cache.json` -- the inputs are static constants, so a re-call
could only return the same answer.

### Before a live demo: pre-warm the cache

The first click on a unit makes a live call. Pre-generate the handful you plan
to demo so they render instantly even if the room's wifi is bad:

```bash
# with the API already running -- warms BOTH the copilot and the research
python api/prefetch_copilot.py                    # the 5 highest-risk units
python api/prefetch_copilot.py DOOR_01_1 BOGIE_06_2   # or name them
python api/prefetch_copilot.py --skip-research    # one feature only
```

Whichever keys are configured get warmed; a missing key skips that feature with
a warning instead of failing the run.

Results are cached per unit both in the browser session and on the server
(`api/copilot_cache.json`, reloaded on restart), keyed by a fingerprint of the
unit's numbers -- so re-running the pipeline expires them rather than pairing an
old explanation with new figures. Any unit a judge clicks that *wasn't*
pre-warmed still goes out live.

## What to look at first

- **`validation_outputs/*.png`** — the main deliverable. One plot per
  ground-truth fault: raw signals, reconstruction error vs. thresholds, and
  fused risk level over time, with a "DETECTED" / "FAULT LOGGED" marker
  pair. `validation_outputs/lead_time_summary.md` has the lead-time table.
- **Dashboard** (`http://localhost:8080`) — fleet overview, click any unit
  for its decision trace, live-feed panel bottom left.

## Layout

```
data_gen/       synthetic data generator + its own config
data/           generated CSVs (raw + scored + fleet_status)
pipeline/       schema_config.py (remap point), model_config.py, baseline.py,
                autoencoder.py, fusion.py, validate.py, run_pipeline.py
models/         trained model/scaler/threshold artifacts (git-ignored)
validation_outputs/  per-fault plots + lead-time summary
api/            FastAPI service + copilot.py (Claude) + research.py (Exa)
                + logistics.py (Google Routes) + locations_config.py
                (ALL fabricated demo coordinates live here)
                + prefetch_copilot.py (demo cache warm-up)
dashboard/      static HTML/JS/Chart.js dashboard
ASSUMPTIONS.md  every judgment call, for review
```

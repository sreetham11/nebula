# NEBULA X — Predictive Fault Detection

Hackathon prototype for LTA Singapore: detects wear/anomaly signatures in
door and bogie rolling stock subsystems and flags them before they become
unplanned service disruptions.

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
# 1. Generate synthetic data -> data/*.csv
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
api/            FastAPI service
dashboard/      static HTML/JS/Chart.js dashboard
ASSUMPTIONS.md  every judgment call, for review
```

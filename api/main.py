"""
NEBULA X fault detection API.

Serves fleet status, per-unit decision traces, on-demand anomaly scoring,
and a simulated live telemetry feed, all backed by the synthetic dataset +
models produced by pipeline/run_pipeline.py.

*** Backed entirely by synthetic placeholder data (see data_gen/) pending
the real LTA dataset. *** Swapping in real data should only require
rerunning the pipeline against real CSVs mapped through
pipeline/schema_config.py -- this file does not hardcode column names.

Run: uvicorn main:app --reload --port 8000   (from the api/ directory, with
the venv active)
"""

# On Windows, torch must be imported before pandas/numpy: both ship their own
# Intel OpenMP runtime (libiomp5md.dll), and if pandas' loads first, torch's
# c10.dll fails to initialise with "WinError 1114". Importing torch first is
# the documented ordering workaround; it is a no-op on Linux/macOS.
import torch  # noqa: F401,E402  -- must precede pandas/numpy imports


import json
import os
import sys
import threading

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "pipeline")))
import schema_config as sc  # noqa: E402
from model_config import BASELINE_Z_THRESHOLD, RISK_LEVELS, RECON_ERROR_WARNING_PERCENTILE  # noqa: E402
from autoencoder import SequenceAutoencoder, get_device  # noqa: E402
from fusion import _base_level_from_recon_error, _escalate  # noqa: E402
from ablation import _first_alert_crossing  # noqa: E402 -- reuse the same sustained-vs-fallback detection logic used in the ablation report

import copilot  # noqa: E402 -- Maintenance Copilot (server-side Anthropic API call; key never leaves the server)
import research  # noqa: E402 -- Investigate Fault (server-side Exa search; key never leaves the server)
import logistics  # noqa: E402 -- Maintenance Logistics (server-side Google Routes; key never leaves the server)
import locations_config  # noqa: E402 -- the single place the fabricated demo coordinates live

app = FastAPI(title="NEBULA X Fault Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

RECOMMENDED_ACTIONS = {
    "Normal": "No action needed. Continue routine monitoring.",
    "Caution": "Increase monitoring frequency. Flag for next scheduled inspection.",
    "Warning": "Schedule a maintenance inspection within 48-72 hours.",
    "Critical": "Immediate inspection required. Consider pulling unit from service.",
}

ARTIFACTS = {}
SCORED = {}
BASELINE_STATS = {}
VAL_RECON_ERRORS = {}
FAULT_LOG = None
LIVE_FEED_DF = None
LIVE_CURSOR_LOCK = threading.Lock()
LIVE_CURSOR = {"pos": 0}


def _load_artifacts():
    device = get_device()
    for subsystem_type, cfg in sc.SUBSYSTEMS.items():
        meta_path = os.path.join(sc.MODELS_DIR, f"{subsystem_type}_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        model = SequenceAutoencoder(n_features=len(cfg["signal_cols"]))
        state = torch.load(
            os.path.join(sc.MODELS_DIR, f"{subsystem_type}_autoencoder.pt"),
            map_location=device,
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        scaler = joblib.load(os.path.join(sc.MODELS_DIR, f"{subsystem_type}_scaler.joblib"))

        ARTIFACTS[subsystem_type] = {
            "model": model,
            "scaler": scaler,
            "thresholds": meta["thresholds"],
            "window_length": meta["window_length"],
            "signal_cols": meta["signal_cols"],
            "device": device,
        }

        SCORED[subsystem_type] = pd.read_csv(cfg["scored_path"], parse_dates=[cfg["timestamp_col"]])


def _build_baseline_stats():
    """
    Precomputes each unit's persistent baseline (rolling mean/std per
    signal, as of its latest reading) from the same scored data that
    /fleet-status and /unit/{id} already read from -- itself derived from
    door_data.csv/bogie_data.csv by pipeline/baseline.py's rolling
    mean/std -- so /predict-anomaly's baseline z-score agrees with what the
    dashboard shows for that unit, instead of being recomputed ad hoc from
    whatever window a caller happens to submit.
    """
    for subsystem_type, cfg in sc.SUBSYSTEMS.items():
        df = SCORED[subsystem_type]
        stats = {}
        for uid, g in df.groupby(cfg["id_col"]):
            latest = g.sort_values(cfg["timestamp_col"]).iloc[-1]
            signals = {}
            for sig in cfg["signal_cols"]:
                mean = latest.get(f"{sig}_roll_mean")
                std = latest.get(f"{sig}_roll_std")
                signals[sig] = {
                    "mean": None if pd.isna(mean) else float(mean),
                    "std": None if pd.isna(std) else float(std),
                }
            stats[uid] = {"signals": signals, "as_of": str(latest[cfg["timestamp_col"]])}
        BASELINE_STATS[subsystem_type] = stats


def _load_sensitivity_data():
    """
    Loads the raw held-out healthy reconstruction-error arrays (saved by
    run_pipeline.py alongside the model) and the fault log, so
    /sensitivity-scan can recompute threshold-cutoff outcomes at any
    percentile purely by re-slicing already-computed arrays -- no model
    inference, no retraining.
    """
    global FAULT_LOG
    for subsystem_type in sc.SUBSYSTEMS:
        path = os.path.join(sc.MODELS_DIR, f"{subsystem_type}_val_recon_errors.npy")
        VAL_RECON_ERRORS[subsystem_type] = np.load(path)
    FAULT_LOG = pd.read_csv(sc.FAULT_LOG_PATH, parse_dates=["timestamp"])


def _build_live_feed():
    global LIVE_FEED_DF
    frames = []
    for subsystem_type, cfg in sc.SUBSYSTEMS.items():
        df = SCORED[subsystem_type].copy()
        keep = [cfg["timestamp_col"], cfg["id_col"], cfg["train_col"], "risk_level", "ae_recon_error"] + cfg["signal_cols"]
        df = df[keep].rename(columns={cfg["timestamp_col"]: "timestamp", cfg["id_col"]: "unit_id", cfg["train_col"]: "train_id"})
        df["subsystem_type"] = subsystem_type
        df["signal_cols"] = [cfg["signal_cols"]] * len(df)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    LIVE_FEED_DF = combined


@app.on_event("startup")
def startup():
    _load_artifacts()
    _build_baseline_stats()
    _load_sensitivity_data()
    _build_live_feed()
    # Picks up any pre-generated demo explanations written by
    # api/prefetch_copilot.py so they survive a restart.
    copilot.load_cache()
    research.load_cache()
    logistics.load_cache()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Reading(BaseModel):
    timestamp: str
    values: Dict[str, float]


class PredictRequest(BaseModel):
    unit_id: str
    subsystem_type: str
    readings: List[Reading]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "nebula-x-fault-detection",
        "data_source": "SYNTHETIC placeholder dataset (real LTA data pending)",
        "subsystems_loaded": list(ARTIFACTS.keys()),
    }


@app.post("/predict-anomaly")
def predict_anomaly(req: PredictRequest):
    if req.subsystem_type not in sc.SUBSYSTEMS:
        raise HTTPException(400, f"subsystem_type must be one of {list(sc.SUBSYSTEMS)}")
    if req.subsystem_type not in ARTIFACTS:
        raise HTTPException(503, "model artifacts not loaded")

    cfg = sc.SUBSYSTEMS[req.subsystem_type]
    artifacts = ARTIFACTS[req.subsystem_type]
    signal_cols = cfg["signal_cols"]
    window_length = artifacts["window_length"]

    if len(req.readings) < window_length:
        raise HTTPException(
            422, f"need at least {window_length} readings for {req.subsystem_type}, got {len(req.readings)}"
        )

    readings = sorted(req.readings, key=lambda r: r.timestamp)[-window_length:]
    try:
        arr = np.array([[r.values[c] for c in signal_cols] for r in readings], dtype=np.float32)
    except KeyError as e:
        raise HTTPException(422, f"reading missing required signal column {e}; expected {signal_cols}")

    scaled = artifacts["scaler"].transform(arr).astype(np.float32)
    x = torch.from_numpy(scaled).unsqueeze(0).to(artifacts["device"])
    with torch.no_grad():
        recon = artifacts["model"](x)
        recon_error = float(torch.mean((recon - x) ** 2).item())

    last_vals = arr[-1]

    # Baseline z-score against this unit's PERSISTED rolling mean/std (the
    # same numbers /fleet-status and /unit/{id} use), not stats derived
    # from whatever window this request happens to submit -- otherwise the
    # same unit could show a different baseline_flag here than on the
    # dashboard depending on what the caller sent.
    unit_stats = BASELINE_STATS.get(req.subsystem_type, {}).get(req.unit_id)
    persisted_signals = unit_stats["signals"] if unit_stats else None
    has_persisted_stats = persisted_signals is not None and all(
        persisted_signals[sig]["mean"] is not None and (persisted_signals[sig]["std"] or 0) > 0
        for sig in signal_cols
    )

    if has_persisted_stats:
        means = np.array([persisted_signals[sig]["mean"] for sig in signal_cols])
        stds = np.array([persisted_signals[sig]["std"] for sig in signal_cols])
        z = (last_vals - means) / stds
        baseline_source = "persisted_unit_history"
        baseline_stats_as_of = unit_stats["as_of"]
    else:
        # Unknown unit_id or no persisted baseline yet (e.g. a brand-new
        # unit with too little history for a rolling window) -- fall back
        # to the submitted window's own stats so the endpoint still works,
        # but flag it clearly since this reading won't necessarily agree
        # with the dashboard.
        hist_vals = arr[:-1] if len(arr) > 1 else arr
        mean = hist_vals.mean(axis=0)
        std = hist_vals.std(axis=0)
        z = np.where(std > 0, (last_vals - mean) / std, 0.0)
        baseline_source = "request_window_fallback"
        baseline_stats_as_of = None

    baseline_max_abs_z = float(np.max(np.abs(z)))
    baseline_flag = baseline_max_abs_z > BASELINE_Z_THRESHOLD

    thresholds = artifacts["thresholds"]
    base_level = _base_level_from_recon_error(recon_error, thresholds)
    risk_level = _escalate(base_level) if baseline_flag else base_level

    return {
        "unit_id": req.unit_id,
        "subsystem_type": req.subsystem_type,
        "risk_level": risk_level,
        "ae_recon_error": recon_error,
        "baseline_flag": bool(baseline_flag),
        "baseline_max_abs_z": baseline_max_abs_z,
        "baseline_source": baseline_source,
        "baseline_stats_as_of": baseline_stats_as_of,
        "thresholds": thresholds,
        "recommended_action": RECOMMENDED_ACTIONS[risk_level],
        "evaluated_at": readings[-1].timestamp,
    }


@app.get("/sensitivity-scan")
def sensitivity_scan(percentile: float = RECON_ERROR_WARNING_PERCENTILE):
    """
    Recomputes classification outcomes (alert volume, faults caught/missed,
    average lead time) at an arbitrary reconstruction-error percentile
    threshold -- purely by re-slicing the already-computed held-out healthy
    error arrays and already-scored per-reading errors. No model inference
    and no retraining happen here; this is just numpy percentile + boolean
    comparisons, so it's safe to call on every slider tick.

    Defaults to the percentile already used for the "Warning" threshold at
    training time (model_config.RECON_ERROR_WARNING_PERCENTILE), so the
    dashboard's sensitivity slider can default to whatever the pipeline
    already chose.
    """
    if not (50.0 <= percentile <= 99.9):
        raise HTTPException(400, "percentile must be between 50 and 99.9")
    if FAULT_LOG is None or not VAL_RECON_ERRORS:
        raise HTTPException(503, "sensitivity data not loaded; run pipeline/run_pipeline.py")

    thresholds_used = {}
    total_alerts = 0
    fault_results = []

    for subsystem_type, cfg in sc.SUBSYSTEMS.items():
        threshold = float(np.percentile(VAL_RECON_ERRORS[subsystem_type], percentile))
        thresholds_used[subsystem_type] = threshold

        df = SCORED[subsystem_type]
        window_length = ARTIFACTS[subsystem_type]["window_length"]

        # ae_recon_error is scored densely (stride=1 -- one window per
        # reading, each overlapping the last by window_length-1 readings;
        # see run_pipeline.py's AE_SCORING_STRIDE). Counting every reading
        # above threshold therefore re-counts the SAME underlying anomaly
        # up to window_length (48) times over -- a unit that degrades once
        # and stays elevated for its remaining ~4,000 readings would report
        # ~4,000 "alerts" for one real event. Sample one reading per
        # non-overlapping window instead (stride = window_length, per unit,
        # in that unit's own time order) so each ~8-hour window of data
        # contributes at most one alert, matching how a real polling-based
        # alert system would fire.
        for _, g in df.groupby(cfg["id_col"]):
            sampled = g.sort_values(cfg["timestamp_col"])["ae_recon_error"].iloc[::window_length]
            total_alerts += int((sampled >= threshold).sum())

        st_faults = FAULT_LOG[FAULT_LOG["subsystem_type"] == subsystem_type]
        for _, fr in st_faults.iterrows():
            uid = fr["subsystem_id"]
            unit_df = df[df[cfg["id_col"]] == uid].sort_values(cfg["timestamp_col"]).reset_index(drop=True).copy()
            unit_df["_alert"] = unit_df["ae_recon_error"] >= threshold
            crossing_ts, method = _first_alert_crossing(unit_df, "_alert", cfg["timestamp_col"], fr["timestamp"])
            lead_hours = (fr["timestamp"] - crossing_ts).total_seconds() / 3600.0 if crossing_ts is not None else None
            fault_results.append({
                "unit_id": uid,
                "subsystem_type": subsystem_type,
                "caught": crossing_ts is not None,
                "lead_time_hours": lead_hours,
                "detection_method": method,
            })

    caught = [f for f in fault_results if f["caught"]]
    avg_lead_hours = (sum(f["lead_time_hours"] for f in caught) / len(caught)) if caught else None

    return {
        "percentile": percentile,
        "thresholds_used": thresholds_used,
        "total_alerts": total_alerts,
        "total_faults": len(fault_results),
        "faults_caught": len(caught),
        "faults_missed": len(fault_results) - len(caught),
        "avg_lead_time_hours": avg_lead_hours,
        "avg_lead_time_days": (avg_lead_hours / 24) if avg_lead_hours is not None else None,
        "fault_details": fault_results,
    }


@app.get("/fleet-status")
def fleet_status():
    if not os.path.exists(sc.FLEET_STATUS_PATH):
        raise HTTPException(503, "fleet status not yet computed; run pipeline/run_pipeline.py")
    df = pd.read_csv(sc.FLEET_STATUS_PATH, parse_dates=["latest_timestamp"])
    df["latest_timestamp"] = df["latest_timestamp"].astype(str)
    records = df.to_dict(orient="records")
    counts = df["risk_level"].value_counts().to_dict()
    return {
        "generated_from": "SYNTHETIC placeholder dataset",
        "units": records,
        "status_counts": {lvl: int(counts.get(lvl, 0)) for lvl in RISK_LEVELS},
        "total_units": len(records),
    }


@app.get("/validation-summary")
def validation_summary():
    """
    Read-only view over validate.py's already-computed output
    (validation_outputs/lead_time_summary.csv) -- does not recompute
    anything, just exposes it for the dashboard's business-impact
    calculator. Run pipeline/run_pipeline.py to (re)generate that file.
    """
    path = os.path.join(sc.VALIDATION_OUTPUT_DIR, "lead_time_summary.csv")
    if not os.path.exists(path):
        raise HTTPException(503, "validation summary not yet computed; run pipeline/run_pipeline.py")

    df = pd.read_csv(path)
    detected = df[df["detected_before_fault"] == True]  # noqa: E712

    return {
        "generated_from": "SYNTHETIC placeholder dataset validation run (validate.py)",
        "total_faults": len(df),
        "detected_count": int(len(detected)),
        "avg_lead_time_hours": float(detected["lead_time_hours"].mean()) if len(detected) else None,
        "avg_lead_time_days": float(detected["lead_time_days"].mean()) if len(detected) else None,
        "faults": df.to_dict(orient="records"),
    }


@app.get("/unit/{unit_id}")
def unit_detail(unit_id: str, history_points: int = 400):
    subsystem_type = None
    for st, cfg in sc.SUBSYSTEMS.items():
        if unit_id in SCORED[st][cfg["id_col"]].values:
            subsystem_type = st
            break
    if subsystem_type is None:
        raise HTTPException(404, f"unit_id {unit_id} not found")

    cfg = sc.SUBSYSTEMS[subsystem_type]
    df = SCORED[subsystem_type]
    unit_df = df[df[cfg["id_col"]] == unit_id].sort_values(cfg["timestamp_col"]).reset_index(drop=True)
    latest = unit_df.iloc[-1]
    risk_level = latest["risk_level"] if pd.notna(latest["risk_level"]) else "Normal"

    # Per-signal deviation from the HEALTHY-POPULATION baseline (the
    # autoencoder's own training scaler: healthy units' mean/std), not the
    # unit's own short rolling-window z-score -- a unit that has been
    # degraded for weeks has a rolling baseline that has already drifted
    # upward with it, so its rolling z-score can look small or even
    # negative despite being grossly abnormal. Comparing against the fixed
    # healthy-population mean/std instead gives a stable answer to "how far
    # from healthy is this signal right now," which is what "primary
    # signal" on the dashboard's Priority Alert cards needs.
    scaler = ARTIFACTS[subsystem_type]["scaler"]
    raw_values = np.array([latest[sig] for sig in cfg["signal_cols"]])
    healthy_zscores = (raw_values - scaler.mean_) / scaler.scale_

    if len(unit_df) > history_points:
        step = len(unit_df) // history_points
        hist = unit_df.iloc[::step]
    else:
        hist = unit_df

    history = []
    for _, row in hist.iterrows():
        entry = {"timestamp": str(row[cfg["timestamp_col"]]), "risk_level": row["risk_level"] if pd.notna(row["risk_level"]) else None,
                 "ae_recon_error": None if pd.isna(row["ae_recon_error"]) else float(row["ae_recon_error"])}
        for sig in cfg["signal_cols"]:
            entry[sig] = float(row[sig])
        history.append(entry)

    return {
        "unit_id": unit_id,
        "subsystem_type": subsystem_type,
        "train_id": str(latest[cfg["train_col"]]),
        "latest": {
            "timestamp": str(latest[cfg["timestamp_col"]]),
            "signals": {sig: float(latest[sig]) for sig in cfg["signal_cols"]},
            "signal_zscores": {
                sig: float(z) for sig, z in zip(cfg["signal_cols"], healthy_zscores)
            },
            "baseline_flag": bool(latest["baseline_flag"]),
            "baseline_max_abs_z": None if pd.isna(latest["baseline_max_abs_z"]) else float(latest["baseline_max_abs_z"]),
            "ae_recon_error": None if pd.isna(latest["ae_recon_error"]) else float(latest["ae_recon_error"]),
            "risk_level": risk_level,
        },
        "thresholds": ARTIFACTS[subsystem_type]["thresholds"],
        "recommended_action": RECOMMENDED_ACTIONS[risk_level],
        "decision_trace": [
            {"step": "Baseline check", "detail": f"max |z| across signals = "
             f"{latest['baseline_max_abs_z']:.2f}" if pd.notna(latest['baseline_max_abs_z']) else "n/a",
             "flagged": bool(latest["baseline_flag"])},
            {"step": "Autoencoder assessment", "detail": f"reconstruction error = "
             f"{latest['ae_recon_error']:.3f}" if pd.notna(latest["ae_recon_error"]) else "n/a",
             "flagged": risk_level != "Normal"},
            {"step": "Combined risk", "detail": risk_level, "flagged": risk_level in ("Warning", "Critical")},
            {"step": "Recommended action", "detail": RECOMMENDED_ACTIONS[risk_level], "flagged": False},
        ],
        "history": history,
    }


@app.get("/live-feed")
def live_feed(batch_size: int = 6):
    if LIVE_FEED_DF is None or len(LIVE_FEED_DF) == 0:
        raise HTTPException(503, "live feed not initialized")

    n = len(LIVE_FEED_DF)
    batch_size = max(1, min(batch_size, 50))
    with LIVE_CURSOR_LOCK:
        start = LIVE_CURSOR["pos"]
        idx = [(start + i) % n for i in range(batch_size)]
        LIVE_CURSOR["pos"] = (start + batch_size) % n

    rows = LIVE_FEED_DF.iloc[idx]
    events = []
    for _, row in rows.iterrows():
        events.append({
            "timestamp": str(row["timestamp"]),
            "unit_id": row["unit_id"],
            "subsystem_type": row["subsystem_type"],
            "train_id": row["train_id"],
            "risk_level": row["risk_level"] if pd.notna(row["risk_level"]) else "Normal",
            "ae_recon_error": None if pd.isna(row["ae_recon_error"]) else float(row["ae_recon_error"]),
            "signals": {sig: float(row[sig]) for sig in row["signal_cols"]},
        })
    return {"events": events, "note": "Simulated live feed replaying synthetic historical data, sped up for demo."}


# ---------------------------------------------------------------------------
# Maintenance Copilot (AI-assisted explanation layer)
# ---------------------------------------------------------------------------
#
# The browser never holds an Anthropic API key and never calls
# api.anthropic.com -- it calls this route, and this process reads
# ANTHROPIC_API_KEY from its own environment. See api/copilot.py.
#
# Everything Claude sees is assembled from unit_detail()'s own return value
# plus the fitted scaler's healthy-population statistics, so the copilot can
# only ever restate numbers the pipeline already computed. This route is
# strictly additive: /unit/{unit_id} and its decision trace are untouched and
# keep working whether or not a key is configured.

def _healthy_baseline(subsystem_type: str) -> Dict[str, Dict[str, float]]:
    """
    Per-signal mean/std of the healthy training population, read off the
    StandardScaler the autoencoder was fit with -- the same reference
    unit_detail()'s signal_zscores are measured against.
    """
    cfg = sc.SUBSYSTEMS[subsystem_type]
    scaler = ARTIFACTS[subsystem_type]["scaler"]
    return {
        sig: {"mean": float(mean), "std": float(scale)}
        for sig, mean, scale in zip(cfg["signal_cols"], scaler.mean_, scaler.scale_)
    }


@app.get("/copilot/{unit_id}")
def copilot_explain(unit_id: str, refresh: bool = False):
    """
    One structured Claude call explaining an existing detection. Returns
    {"status": "unavailable", "reason": ...} rather than an error status on
    any failure, so the dashboard can show its fallback message without the
    request looking like a dashboard bug.
    """
    unit = unit_detail(unit_id, history_points=1)  # 404s here for an unknown unit
    context = copilot.build_context(unit, _healthy_baseline(unit["subsystem_type"]))
    return copilot.explain_unit(unit_id, context, force_refresh=refresh)


@app.get("/copilot-status")
def copilot_status():
    """Which units already have a cached explanation -- used by the pre-fetch script."""
    return {
        "api_key_configured": copilot.api_key_present(),
        "model": copilot.MODEL,
        "cached_units": copilot.cache_entries(),
    }


# ---------------------------------------------------------------------------
# Investigate Fault (Exa literature search)
# ---------------------------------------------------------------------------
#
# Same containment as the copilot: EXA_API_KEY lives in this process's
# environment, the browser only ever calls /research/{unit_id}. The search
# query is derived from the unit's own pipeline output (see
# research.build_query) rather than hardcoded per unit, and the "does not
# confirm this unit's fault" caption is a fixed server-side constant that
# no API response can override.

@app.get("/research/{unit_id}")
def research_unit(unit_id: str, refresh: bool = False):
    unit = unit_detail(unit_id, history_points=1)  # 404s for an unknown unit
    context = copilot.build_context(unit, _healthy_baseline(unit["subsystem_type"]))
    return research.investigate(unit_id, context, force_refresh=refresh)


@app.get("/research-status")
def research_status():
    return {
        "api_key_configured": research.api_key_present(),
        "search_type": research.EXA_SEARCH_TYPE,
        "cached_units": research.cache_entries(),
    }


# ---------------------------------------------------------------------------
# Maintenance Logistics (Google Routes)
# ---------------------------------------------------------------------------
#
# *** Routes between FABRICATED coordinates. *** The synthetic dataset has no
# geolocation at all; api/locations_config.py invents train positions and
# depot sites, and is the only place that does. Every response carries its
# DATA_NOTICE, which the dashboard renders as a banner, not a footnote.
#
# Shown only for units the pipeline escalated to Warning/Critical --
# dispatching a healthy unit to a depot would be a recommendation the
# detection system never made.
#
# The map image is proxied through /logistics/{unit_id}/map so the Static
# Maps URL (and the API key in it) never reaches the browser.

@app.get("/logistics/{unit_id}")
def logistics_plan(unit_id: str, refresh: bool = False):
    unit = unit_detail(unit_id, history_points=1)  # 404s for an unknown unit
    return logistics.plan(
        unit_id,
        train_id=unit["train_id"],
        risk_level=unit["latest"]["risk_level"],
        force_refresh=refresh,
    )


@app.get("/logistics/{unit_id}/map")
def logistics_map(unit_id: str):
    """Static route image, fetched server-side so the key stays off the page."""
    png = logistics.map_image(unit_id)
    if png is None:
        raise HTTPException(404, "no route map available for this unit")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/logistics-status")
def logistics_status():
    return {
        "api_key_configured": logistics.api_key_present(),
        "depots": [{"id": d["id"], "name": d["name"]} for d in locations_config.DEPOTS],
        "data_notice": locations_config.DATA_NOTICE,
        "cached_units": logistics.cache_entries(),
    }

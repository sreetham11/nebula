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

# Import order below is load-bearing on two platforms at once, so it is
# pinned here rather than left to isort:
#   * xgboost before torch -- on macOS, torch first loads a second OpenMP
#     runtime and the first xgboost predict() then segfaults (exit 139).
#   * torch before pandas/numpy -- on Windows both ship their own Intel
#     OpenMP runtime (libiomp5md.dll), and if pandas' loads first torch's
#     c10.dll fails to initialise with "WinError 1114".
# xgboost is optional so the synthetic-model endpoints still start without
# it; /predict-rail-real then returns 503.
try:
    import xgboost  # noqa: F401,E402  -- must precede torch
except ImportError:
    pass

import torch  # noqa: F401,E402  -- must precede pandas/numpy


import json
import os
import sys
import threading

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "pipeline")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema_config as sc  # noqa: E402
import real_inference  # noqa: E402
from model_config import BASELINE_Z_THRESHOLD, RISK_LEVELS, RECON_ERROR_WARNING_PERCENTILE  # noqa: E402
from autoencoder import SequenceAutoencoder, get_device  # noqa: E402
from fusion import _base_level_from_recon_error, _escalate  # noqa: E402
from ablation import _first_alert_crossing  # noqa: E402 -- reuse the same sustained-vs-fallback detection logic used in the ablation report
import health_index as hi  # noqa: E402 -- bounded 0-100 restatement of reconstruction error (presentation only)

import ingest  # noqa: E402 -- CSV upload scored against the frozen trained models

import copilot  # noqa: E402 -- Maintenance Copilot (server-side Anthropic API call; key never leaves the server)
import research  # noqa: E402 -- Investigate Fault (server-side Exa search; key never leaves the server)
import chat  # noqa: E402 -- Fleet Assistant chatbox (Claude + Exa, keys stay server-side)

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

# A deviation is measured against the POOLED healthy distribution, which
# spans every duty state a unit passes through -- a train stabled overnight
# genuinely shows low traction motor rpm, a cool traction motor and a
# fully-charged battery. So one to two sigma on a single channel is normal
# time-of-day variation, not evidence, and naming it "the primary signal"
# would send a fitter to inspect a healthy motor. Below this bar the
# decision trace says so outright: the detection then rests on the
# COMBINATION of channels the autoencoder saw across the window, which is
# exactly what a per-channel threshold cannot see.
PRIMARY_SIGNAL_MIN_SIGMA = 2.0

ARTIFACTS = {}
SCORED = {}
BASELINE_STATS = {}
VAL_RECON_ERRORS = {}
FAULT_LOG = None
VALIDATION_FAULTS_BY_UNIT = {}
LIVE_FEED_DF = None
LIVE_CURSOR_LOCK = threading.Lock()
LIVE_CURSOR = {"pos": 0}
REAL_MODELS = None
REAL_MODELS_ERROR = None


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

        # Median reconstruction error across the held-out HEALTHY validation
        # units. This is the "100 on the Health Index" anchor, and the
        # denominator behind "x95 the healthy fleet median" -- it has to come
        # from the same held-out units the thresholds were calibrated on, not
        # from the whole scored fleet (which includes the faulty units and
        # would drag the reference point upward).
        val_errors = np.load(os.path.join(sc.MODELS_DIR, f"{subsystem_type}_val_recon_errors.npy"))

        ARTIFACTS[subsystem_type] = {
            "model": model,
            "scaler": scaler,
            "thresholds": meta["thresholds"],
            "window_length": meta["window_length"],
            "signal_cols": meta["signal_cols"],
            "healthy_median_recon_error": float(np.median(val_errors)),
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


def _load_validation_faults():
    """
    Loads validate.py's per-unit lead-time results (validation_outputs/
    lead_time_summary.csv -- the same file /validation-summary reads for
    the fleet-wide average lead time) into a unit_id-keyed lookup, so
    /unit/{id} can attach each of the 8 known-fault units' OWN specific
    lead time (not the fleet average) without re-reading the file on every
    request. Left empty if the file doesn't exist yet (pipeline not run) --
    /unit/{id} just omits known_fault's lead-time fields in that case.
    """
    global VALIDATION_FAULTS_BY_UNIT
    path = os.path.join(sc.VALIDATION_OUTPUT_DIR, "lead_time_summary.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    VALIDATION_FAULTS_BY_UNIT = {row["unit_id"]: row.to_dict() for _, row in df.iterrows()}


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


def _load_real_models():
    """
    Loads the real_* models (trained on the real PS3 Door / Rail_Corrugation
    data) once at startup. Fail-soft: if any file is missing or unreadable
    the server still starts and the synthetic endpoints are unaffected --
    the two *-real endpoints just return 503 with the reason.
    """
    global REAL_MODELS, REAL_MODELS_ERROR
    try:
        REAL_MODELS = real_inference.load_real_models(sc.MODELS_DIR)
    except Exception as e:  # noqa: BLE001
        REAL_MODELS = None
        REAL_MODELS_ERROR = f"{type(e).__name__}: {e}"
        print(f"WARNING: real models not loaded ({REAL_MODELS_ERROR}); "
              "/predict-door-real and /predict-rail-real will return 503")


@app.on_event("startup")
def startup():
    _load_artifacts()
    _build_baseline_stats()
    _load_sensitivity_data()
    _load_validation_faults()
    _build_live_feed()
    # Picks up any pre-generated demo explanations written by
    # api/prefetch_copilot.py so they survive a restart.
    copilot.load_cache()
    research.load_cache()
    _load_real_models()


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

def _health_fields(subsystem_type, recon_error):
    """
    The Health Index block attached to every unit the API returns.

    Always carries the raw reconstruction error, the threshold ladder it was
    measured against, and the healthy median alongside the index itself, so
    the bounded number never travels without the model output it was derived
    from. See pipeline/health_index.py for why the index exists at all.
    """
    art = ARTIFACTS[subsystem_type]
    thresholds = art["thresholds"]
    median = art["healthy_median_recon_error"]
    if recon_error is not None and pd.isna(recon_error):
        recon_error = None
    value = hi.health_index(recon_error, thresholds, median)
    return {
        "health_index": value,
        "health_band": hi.health_band(value),
        "error_vs_healthy_median": hi.error_vs_healthy(recon_error, median),
        "healthy_median_recon_error": round(median, 4),
        "thresholds": {k: round(float(v), 4) for k, v in thresholds.items()},
    }


def _signal_block(subsystem_type, row, zscores=None):
    """
    Per-signal readings with their unit, display label, and which direction
    is a deterioration -- so a falling brake capacity is never drawn as an
    improvement just because the number went down.
    """
    cfg = sc.SUBSYSTEMS[subsystem_type]
    out = []
    for i, sig in enumerate(cfg["signal_cols"]):
        meta = sc.signal_meta(sig)
        z = None if zscores is None else float(zscores[i])
        # Signed "how far from healthy, in the bad direction": a low-is-bad
        # signal 3 sigma BELOW the healthy mean is +3 deviation here.
        deviation = None if z is None else (-z if meta["worse"] == "low" else z)
        out.append({
            "signal": sig,
            "label": meta["label"],
            "unit": meta["unit"],
            "group": meta["group"],
            "worse": meta["worse"],
            "value": float(row[sig]),
            "zscore_vs_healthy": None if z is None else round(z, 2),
            "deviation_sigma": None if deviation is None else round(deviation, 2),
            "modelled": True,
        })
    for sig in cfg.get("context_cols", []):
        if sig not in row:
            continue
        meta = sc.signal_meta(sig)
        out.append({
            "signal": sig,
            "label": meta["label"],
            "unit": meta["unit"],
            "group": meta["group"],
            "worse": meta["worse"],
            "value": float(row[sig]),
            "zscore_vs_healthy": None,
            "deviation_sigma": None,
            "modelled": False,
        })
    return out


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

    # Reconstruction error is on a different scale per subsystem (each
    # autoencoder is trained and calibrated separately), so a fleet-wide
    # table cannot rank on it directly. The Health Index is the column that
    # is comparable across subsystems; the raw error rides along with it.
    for rec in records:
        st = rec["subsystem_type"]
        err = rec.get("ae_recon_error")
        if err is not None and pd.isna(err):
            err = None
            rec["ae_recon_error"] = None
        rec.update(_health_fields(st, err))

    counts = df["risk_level"].value_counts().to_dict()
    return {
        "generated_from": "SYNTHETIC placeholder dataset",
        "units": records,
        "status_counts": {lvl: int(counts.get(lvl, 0)) for lvl in RISK_LEVELS},
        "total_units": len(records),
        "subsystems": {
            st: {
                "signal_cols": art["signal_cols"],
                "thresholds": {k: round(float(v), 4) for k, v in art["thresholds"].items()},
                "healthy_median_recon_error": round(art["healthy_median_recon_error"], 4),
            }
            for st, art in ARTIFACTS.items()
        },
        "signal_metadata": {
            col: sc.signal_meta(col)
            for cfg in sc.SUBSYSTEMS.values()
            for col in cfg["signal_cols"] + cfg.get("context_cols", [])
        },
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


@app.get("/fleet-signals")
def fleet_signals():
    """
    Latest per-SIGNAL reading for every unit in the fleet, in one call.

    /fleet-status answers "how is this unit doing" with a single risk level.
    That is enough to colour a unit, but not enough to colour the individual
    systems inside it: a car unit carries HVAC, lighting, battery and radio
    on one model, so a red car says nothing about WHICH of the four is the
    problem. This endpoint exposes each channel's own deviation from the
    healthy fleet mean, which is what lets the digital twin light up the air
    conditioning rather than the whole carriage.

    Deviation is signed so that POSITIVE always means "worse": a brake
    capacity 4σ below the healthy mean and a bearing temperature 4σ above it
    both come back as +4.0.
    """
    out = {}
    for subsystem_type, cfg in sc.SUBSYSTEMS.items():
        art = ARTIFACTS[subsystem_type]
        scaler = art["scaler"]
        df = SCORED[subsystem_type]
        for uid, g in df.groupby(cfg["id_col"]):
            latest = g.sort_values(cfg["timestamp_col"]).iloc[-1]
            raw = np.array([latest[sig] for sig in cfg["signal_cols"]])
            zscores = (raw - scaler.mean_) / scaler.scale_
            err = None if pd.isna(latest["ae_recon_error"]) else float(latest["ae_recon_error"])
            out[str(uid)] = {
                "unit_id": str(uid),
                "subsystem_type": subsystem_type,
                "train_id": str(latest[cfg["train_col"]]),
                "timestamp": str(latest[cfg["timestamp_col"]]),
                "risk_level": latest["risk_level"] if pd.notna(latest["risk_level"]) else "Normal",
                "ae_recon_error": err,
                "signals": _signal_block(subsystem_type, latest, zscores),
                **_health_fields(subsystem_type, err),
            }
    return {
        "generated_from": "SYNTHETIC placeholder dataset",
        "units": out,
        "deviation_note": (
            "deviation_sigma is signed so positive is always the deteriorating "
            "direction, whichever way the underlying signal moves."
        ),
    }


@app.get("/model-validation")
def model_validation():
    """
    Read-only view over ablation.py's approach comparison
    (validation_outputs/ablation_comparison.csv) -- does not recompute
    anything, just summarizes it for the dashboard's Model Validation
    section. Run pipeline/ablation.py to (re)generate that file.

    A detection only counts as "genuine" if its method is "sustained" (the
    alert held for >=70% of the remaining pre-fault readings -- see
    ablation.py's _first_alert_crossing/SUSTAIN_FRAC). "fallback_isolated"
    means no sustained alert ever existed and a single isolated spike was
    used as a fallback lead-time estimate -- that's noise, not a genuine
    early warning, so it's excluded from the average lead time. An
    approach is only given a numeric average lead time if a majority of
    its detections are genuine; otherwise it's reported as unreliable,
    same as ablation.py's own console caveat.
    """
    path = os.path.join(sc.VALIDATION_OUTPUT_DIR, "ablation_comparison.csv")
    if not os.path.exists(path):
        raise HTTPException(503, "ablation comparison not yet computed; run pipeline/ablation.py")

    df = pd.read_csv(path)
    n = len(df)

    def summarize(lead_col, method_col):
        genuine = df[method_col] == "sustained"
        genuine_count = int(genuine.sum())
        reliable = n > 0 and genuine_count / n >= 0.5
        avg_days = float(df.loc[genuine, lead_col].mean() / 24.0) if reliable and genuine_count else None
        return {
            "genuine_detections": genuine_count,
            "total_faults": n,
            "avg_lead_time_days": avg_days,
            "reliable": reliable,
        }

    return {
        "generated_from": "SYNTHETIC placeholder dataset ablation run (pipeline/ablation.py)",
        "total_faults": n,
        "approaches": [
            {"name": "Rolling z-score baseline", **summarize("baseline_lead_time", "baseline_method")},
            {"name": "Autoencoder (healthy-trained)", **summarize("fused_lead_time", "fused_method")},
        ],
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

    # Known-fault ground truth (fault_log.csv, the generator's own record of
    # which units had a fault injected and when) -- used by the frontend's
    # "why was this flagged" explanation to only claim a pattern matches a
    # "known degradation signature" for units that actually have one,
    # rather than asserting it for every unit.
    known_fault = None
    if FAULT_LOG is not None:
        matches = FAULT_LOG[FAULT_LOG["subsystem_id"] == unit_id]
        if len(matches):
            fault_row = matches.iloc[0]
            known_fault = {
                "fault_type": fault_row["fault_type"],
                "severity": int(fault_row["severity"]),
                "fault_timestamp": str(fault_row["timestamp"]),
            }

    # If this unit is also one of validate.py's 8 known-fault cases with a
    # detected lead time, attach ITS OWN lead_time_hours/lead_time_days
    # (not the fleet-wide average /validation-summary reports) so the
    # dashboard's lead-time visual can show the specific number for
    # whichever unit is selected.
    val_fault = VALIDATION_FAULTS_BY_UNIT.get(unit_id)
    if val_fault is not None and known_fault is not None:
        detected = bool(val_fault["detected_before_fault"])
        known_fault["detected_before_fault"] = detected
        known_fault["first_warning_timestamp"] = str(val_fault["first_warning_timestamp"]) if detected else None
        known_fault["lead_time_hours"] = float(val_fault["lead_time_hours"]) if detected and pd.notna(val_fault["lead_time_hours"]) else None
        known_fault["lead_time_days"] = float(val_fault["lead_time_days"]) if detected and pd.notna(val_fault["lead_time_days"]) else None

    if len(unit_df) > history_points:
        step = len(unit_df) // history_points
        hist = unit_df.iloc[::step]
    else:
        hist = unit_df

    art = ARTIFACTS[subsystem_type]
    thresholds = art["thresholds"]
    healthy_median = art["healthy_median_recon_error"]

    history = []
    for _, row in hist.iterrows():
        err = None if pd.isna(row["ae_recon_error"]) else float(row["ae_recon_error"])
        entry = {"timestamp": str(row[cfg["timestamp_col"]]),
                 "risk_level": row["risk_level"] if pd.notna(row["risk_level"]) else None,
                 "ae_recon_error": err,
                 "health_index": hi.health_index(err, thresholds, healthy_median)}
        for sig in cfg["signal_cols"]:
            entry[sig] = float(row[sig])
        history.append(entry)

    latest_error = None if pd.isna(latest["ae_recon_error"]) else float(latest["ae_recon_error"])
    health = _health_fields(subsystem_type, latest_error)
    signals = _signal_block(subsystem_type, latest, healthy_zscores)

    # The signal actually driving the detection: largest deviation in the
    # direction that is bad for that channel, so a brake capacity 4 sigma
    # BELOW healthy outranks a bearing temperature 1 sigma above it.
    ranked = sorted(
        [sg for sg in signals if sg["deviation_sigma"] is not None],
        key=lambda sg: sg["deviation_sigma"], reverse=True,
    )
    primary = ranked[0] if ranked else None

    # Every number below carries what it is measured against. A bare
    # "reconstruction error = 62.5" is unreadable; the same number as
    # "x95 the healthy fleet median, Critical cut point 4.18" is not.
    if latest_error is None:
        ae_detail = "no score yet — needs a full window of readings"
    else:
        # A multiple below 10 needs its decimal to stay meaningful ("x1.1
        # the healthy median" is a healthy unit; rounding it to "x1" reads
        # as if the comparison had been dropped).
        ratio = health["error_vs_healthy_median"]
        ratio_text = "–" if ratio is None else (f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}")
        ae_detail = (
            f"reconstruction error {latest_error:.2f} "
            f"(healthy fleet median {healthy_median:.2f}, "
            f"x{ratio_text} that; "
            f"Critical cut point {float(thresholds['Critical']):.2f}) "
            f"-> Health Index {health['health_index']}/100"
        )

    if pd.notna(latest["baseline_max_abs_z"]):
        baseline_detail = (
            f"largest single-reading deviation {float(latest['baseline_max_abs_z']):.2f}σ "
            f"from this unit's own recent rolling mean, across "
            f"{len(cfg['signal_cols'])} modelled signals (flags above {BASELINE_Z_THRESHOLD:.1f}σ)"
        )
    else:
        baseline_detail = "n/a — not enough readings yet for a rolling baseline"

    if primary is None:
        primary_detail = "n/a"
    elif primary["deviation_sigma"] < PRIMARY_SIGNAL_MIN_SIGMA:
        primary_detail = (
            f"no single channel stands out — the largest is {primary['label']} at "
            f"{abs(primary['deviation_sigma']):.1f}σ, within normal duty-cycle variation. "
            f"This detection rests on the combination across "
            f"{len(cfg['signal_cols'])} channels, not on any one of them."
        )
    else:
        unit_suffix = f" {primary['unit']}" if primary["unit"] else ""
        direction = "below" if primary["worse"] == "low" else "above"
        primary_detail = (
            f"{primary['label']} at {primary['value']:.2f}{unit_suffix}, "
            f"{abs(primary['deviation_sigma']):.1f}σ {direction} the healthy fleet mean"
        )

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
            "signal_detail": signals,
            "primary_signal": primary,
            "signal_healthy_mean": {
                sig: float(m) for sig, m in zip(cfg["signal_cols"], scaler.mean_)
            },
            "baseline_flag": bool(latest["baseline_flag"]),
            "baseline_max_abs_z": None if pd.isna(latest["baseline_max_abs_z"]) else float(latest["baseline_max_abs_z"]),
            "baseline_z_threshold": float(BASELINE_Z_THRESHOLD),
            "ae_recon_error": latest_error,
            "risk_level": risk_level,
            **health,
        },
        "thresholds": thresholds,
        "healthy_median_recon_error": round(healthy_median, 4),
        "recommended_action": RECOMMENDED_ACTIONS[risk_level],
        "known_fault": known_fault,
        "decision_trace": [
            {"step": "Baseline check (single-reading spike)",
             "detail": baseline_detail,
             "flagged": bool(latest["baseline_flag"])},
            {"step": "Autoencoder assessment (window drift)",
             "detail": ae_detail,
             "flagged": risk_level != "Normal"},
            {"step": "Primary signal",
             "detail": primary_detail,
             "flagged": primary is not None and primary["deviation_sigma"] >= PRIMARY_SIGNAL_MIN_SIGMA},
            {"step": "Combined risk",
             "detail": f"{risk_level} — Health Index {health['health_index']}/100 ({health['health_band']})"
                       if health["health_index"] is not None else risk_level,
             "flagged": risk_level in ("Warning", "Critical")},
            {"step": "Recommended action", "detail": RECOMMENDED_ACTIONS[risk_level], "flagged": False},
        ],
        "history": history,
    }


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    subsystem_type: Optional[str] = Form(None),
    mapping: Optional[str] = Form(None),
):
    """
    Score a dropped CSV/Parquet with the models already on disk.

    No retraining and no persistence: the upload is scored in-process and
    the result returned, leaving the synthetic fleet the rest of the
    dashboard reads completely untouched. `mapping` is an optional JSON
    object of {schema_column: column_in_your_file} for a file whose headers
    the aliases in ingest.py do not already cover.
    """
    explicit_mapping = None
    if mapping:
        try:
            explicit_mapping = json.loads(mapping)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"mapping is not valid JSON: {exc}")
        if not isinstance(explicit_mapping, dict):
            raise HTTPException(400, "mapping must be a JSON object of {schema_column: file_column}")

    raw = await file.read()
    try:
        return ingest.score_upload(
            raw, file.filename, ARTIFACTS,
            subsystem_type=subsystem_type or None,
            explicit_mapping=explicit_mapping,
        )
    except ValueError as exc:
        # Every rejection in ingest.py is a ValueError carrying a message
        # written for the person who dropped the file, so it is surfaced
        # verbatim rather than flattened into a generic 400.
        raise HTTPException(400, str(exc))


@app.get("/upload-schema")
def upload_schema():
    """
    What an uploaded file needs to contain, per subsystem -- so the
    dashboard can show the expected columns before anyone drops a file,
    instead of only after a rejection.
    """
    return {
        "max_rows": ingest.MAX_ROWS,
        "max_bytes": ingest.MAX_UPLOAD_BYTES,
        "formats": ["csv", "parquet"],
        "subsystems": {
            st: {
                "id_col": cfg["id_col"],
                "timestamp_col": cfg["timestamp_col"],
                "train_col": cfg["train_col"],
                "signal_cols": cfg["signal_cols"],
                "window_length": ARTIFACTS[st]["window_length"],
                "signal_metadata": {c: sc.signal_meta(c) for c in cfg["signal_cols"]},
                "aliases": {c: ingest.ALIASES.get(c, []) for c in cfg["signal_cols"]},
            }
            for st, cfg in sc.SUBSYSTEMS.items()
        },
        "note": (
            "Files are scored against the frozen trained models — no retraining. "
            "Column names are matched case- and separator-insensitively against the "
            "aliases listed here before the file is rejected."
        ),
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
# Real-data models (trained on the actual PS3 Door / Rail_Corrugation data)
# ---------------------------------------------------------------------------

def _require_real_models():
    if REAL_MODELS is None:
        raise HTTPException(503, f"real models not loaded: {REAL_MODELS_ERROR or 'startup did not run'}")
    return REAL_MODELS


def _read_uploaded_csv(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        # A semicolon- or tab-separated export (common from Excel in some
        # locales) parses as ONE column; retry with the separator it shows.
        if df.shape[1] == 1:
            first = str(df.columns[0])
            for sep in (";", "\t"):
                if sep in first:
                    file.file.seek(0)
                    df = pd.read_csv(file.file, sep=sep)
                    break
    except UnicodeDecodeError:
        raise HTTPException(400, "the file is not UTF-8 text — re-save it as plain CSV (UTF-8) and try again")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not parse uploaded file as CSV: {e}")
    return df


@app.post("/predict-door-real")
def predict_door_real(file: UploadFile = File(...)):
    """
    Upload a real-schema Door CSV (Datetime + Motor current(mA), Motor
    Voltage(10mV), ...). The stream is split into door cycles at time gaps
    > 0.1s, each cycle is summarised as 24 features (mean/std/max/min of six
    signals), and the real-data autoencoder flags cycles whose reconstruction
    error exceeds the calibrated threshold as "Abnormal resistance".
    """
    models = _require_real_models()
    df = _read_uploaded_csv(file)
    try:
        return real_inference.predict_door(df, models)
    except real_inference.InputError as e:
        raise HTTPException(422, str(e))


@app.post("/predict-rail-real")
def predict_rail_real(file: UploadFile = File(...)):
    """
    Upload a real-schema Rail_Corrugation CSV (129 columns: Rotating speed +
    8 cars x 8 positions x vibration/shock). Extracts rms/std/max per column
    (387 features) and classifies the recording Normal / Side I / Side II
    with the real-data XGBoost classifier.
    """
    models = _require_real_models()
    df = _read_uploaded_csv(file)
    try:
        result = real_inference.predict_rail(df, models)
    except real_inference.InputError as e:
        raise HTTPException(422, str(e))
    return {"file_id": file.filename, **result}


# ---------------------------------------------------------------------------
# Fleet Assistant (chat)
# ---------------------------------------------------------------------------
#
# The conversational half of the answer to "how does a person who is not a
# data analyst use any of this". Same containment as the copilot: the browser
# posts a transcript here, this process holds the keys, and nothing the
# assistant does can write to the fleet or change a risk level.
#
# The tools it may call are wired to the very same functions that serve the
# dashboard -- unit_detail() and fleet_status() below -- so an answer can
# never disagree with what is on screen.

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    unit_id: Optional[str] = None


def _chat_get_unit(unit_id: str) -> Dict:
    """Tool backing for get_unit. Unknown ids come back as a KeyError so
    chat.py can turn them into a plain 'no such unit' answer rather than a
    500."""
    try:
        return unit_detail(unit_id, history_points=1)
    except HTTPException:
        raise KeyError(unit_id)


def _chat_rank_fleet(limit: int, risk_level: Optional[str] = None) -> List[Dict]:
    """Tool backing for rank_fleet: worst Health Index first, optionally
    filtered to one risk level."""
    rows = fleet_status()["units"]
    if risk_level:
        wanted = str(risk_level).strip().lower()
        rows = [r for r in rows if str(r.get("risk_level", "")).lower() == wanted]
    rows = sorted(rows, key=lambda r: r.get("health_index") if r.get("health_index") is not None else 101)
    return [
        {
            "unit_id": r.get("unit_id"),
            "subsystem_type": r.get("subsystem_type"),
            "train_id": r.get("train_id"),
            "risk_level": r.get("risk_level"),
            "health_index": r.get("health_index"),
            "ae_recon_error": r.get("ae_recon_error"),
        }
        for r in rows[:max(1, min(int(limit), 25))]
    ]


@app.post("/chat")
def chat_turn(req: ChatRequest):
    """
    One assistant turn. Returns status:"unavailable" with a reason rather than
    an error status on every failure path, so a missing key or a provider
    outage renders as a message in the chat window instead of looking like the
    dashboard itself is broken.
    """
    unit = None
    if req.unit_id:
        try:
            unit = unit_detail(req.unit_id, history_points=1)
        except HTTPException:
            unit = None  # a stale selection in the browser is not an error

    try:
        validation = validation_summary()
    except HTTPException:
        validation = None  # pipeline/validate.py hasn't been run

    context = chat.build_context(
        fleet=fleet_status()["units"],
        validation=validation,
        unit=unit,
    )
    return chat.answer(
        messages=[m.model_dump() for m in req.messages],
        context=context,
        get_unit=_chat_get_unit,
        rank_fleet=_chat_rank_fleet,
    )


@app.get("/chat-status")
def chat_status():
    return {
        "api_key_configured": chat.anthropic_key_present(),
        "model": chat.MODEL,
        "literature_search": chat.exa_key_present(),
    }

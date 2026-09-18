"""
Upload a CSV, score it with the already-trained models.

This is the path the real LTA dataset takes on the day: drop the file on
the dashboard, and every unit in it comes back with a risk level, a
reconstruction error, and a Health Index, computed by exactly the same
artifacts (scaler, autoencoder, calibrated thresholds, rolling-z baseline)
that produced every other number on the page.

WHAT IT DOES NOT DO
-------------------
It does not retrain. The autoencoder, its scaler and its thresholds stay
frozen at whatever pipeline/run_pipeline.py last produced, and the upload
is scored against them. That is the honest reading of "score this file with
the model I have":

  * If the uploaded data is the same equipment as the training data, this
    is the right thing -- thresholds calibrated on held-out healthy units
    still apply.
  * If the uploaded data is DIFFERENT equipment, with different healthy
    operating points, the scaler's healthy mean/std are wrong for it, and
    everything will look anomalous. That is a real limitation, not a bug,
    and score_upload() surfaces it as an explicit warning whenever the
    upload's own distribution sits far from the training scaler's -- rather
    than quietly returning a fleet of red units.

Retraining on an upload would need a trustworthy healthy window to
calibrate against, which a dropped file does not come with.

COLUMN MAPPING
--------------
Real telemetry will not use this project's column names. Mapping is
resolved in three passes, most trusted first:

  1. an explicit mapping supplied by the caller (schema column -> the
     column in their file),
  2. an exact match on the schema column name,
  3. a normalised-alias match (case, spaces, underscores and units
     stripped) against ALIASES below.

Anything still unresolved is reported back by name, with the file's actual
headers, so the mismatch is visible instead of silently mis-scored. No
unit conversion is attempted anywhere: a temperature column in Fahrenheit
mapped onto a Celsius signal will score as nonsense, so the response always
echoes the healthy-population mean of each mapped signal next to the
upload's own mean for a sanity check.
"""

import io
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "pipeline")))
import schema_config as sc  # noqa: E402
import health_index as hi  # noqa: E402
from baseline import compute_baseline_flags  # noqa: E402
from autoencoder import build_windows, score_windows  # noqa: E402
from fusion import fuse_risk  # noqa: E402
from model_config import AE_SCORING_STRIDE  # noqa: E402

# Hard caps. A dropped file is untrusted input: without these a 2 GB CSV
# would be read straight into memory by the scoring request.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_ROWS = 500_000

# How far the upload's own mean for a signal may sit from the training
# population's mean, in training standard deviations, before it is called
# out as a likely different-equipment / different-units mismatch.
DISTRIBUTION_SHIFT_SIGMA = 3.0

# Normalised aliases per schema column. Keys are the schema's own column
# names; values are additional spellings seen in the wild. Matching is done
# on _normalise() output, so "Motor Current (A)" and "motor_current_amps"
# both reduce to "motorcurrent" and need no separate entry.
ALIASES: Dict[str, List[str]] = {
    "timestamp": ["time", "datetime", "readingtime", "recordedat", "ts", "eventtime", "sampletime"],
    "door_id": ["doorid", "unitid", "assetid", "equipmentid", "id", "doorunit"],
    "bogie_id": ["bogieid", "unitid", "assetid", "equipmentid", "id", "truckid"],
    "car_id": ["carid", "unitid", "assetid", "equipmentid", "id", "coachid", "vehicleid"],
    "train_id": ["trainid", "trainsetid", "trainset", "consist", "consistid", "rakeid", "fleetid"],

    "cycle_time_sec": ["cycletime", "doorcycletime", "opencloseduration", "operationtime", "doorruntime"],
    "motor_current_amps": ["motorcurrent", "doormotorcurrent", "current", "drivecurrent"],
    "motor_temp_c": ["motortemp", "motortemperature", "doormotortemp", "windingtemp"],
    "motor_rpm": ["motorspeed", "doormotorrpm", "drivespeed", "rpm"],
    "cycle_count": ["cyclecount", "lifetimecycles", "operations", "opcount"],

    "temperature_c": ["temperature", "bearingtemp", "axletemp", "axleboxtemp", "hotboxtemp", "bogietemp"],
    "vibration_rms": ["vibration", "vibrms", "accelrms", "vibrationlevel"],
    "traction_motor_temp_c": ["tractionmotortemp", "tractiontemp", "tmtemp"],
    "traction_motor_rpm": ["tractionmotorspeed", "tractionrpm", "tmrpm", "wheelrpm"],
    "brake_capacity_pct": ["brakecapacity", "brakeefficiency", "brakeperformance", "brakeforcepct"],
    "axle_load_kg": ["axleload", "load", "wheelload"],

    "hvac_supply_temp_c": ["hvacsupplytemp", "supplyairtemp", "actemp", "airconsupplytemp", "saloonsupplytemp"],
    "hvac_current_amps": ["hvaccurrent", "accurrent", "aircurrent", "hvacload"],
    "lighting_load_pct": ["lightingload", "saloonlighting", "lightload", "lightingpct"],
    "battery_soc_pct": ["batterysoc", "soc", "stateofcharge", "batterycharge", "batterycapacity"],
    "battery_voltage_v": ["batteryvoltage", "batteryvolts", "auxvoltage", "batteryterminalvoltage"],
    "comms_rssi_dbm": ["commsrssi", "rssi", "signalstrength", "radiosignal", "radiorssi", "linkquality"],

    "hvac_setpoint_c": ["hvacsetpoint", "acsetpoint", "targettemp", "setpoint"],
}

# Unit suffixes stripped during normalisation so "temperature_c",
# "temperatureC" and "temperature (degC)" all collapse together.
_UNIT_SUFFIXES = (
    "degc", "degf", "celsius", "sec", "seconds", "secs", "amps", "amp", "a",
    "c", "f", "kg", "rpm", "pct", "percent", "v", "volts", "dbm", "rms", "g",
    # A bare trailing "s" covers both the "(s)" unit and an accidental
    # plural. It is safe because both sides of every comparison go through
    # this same function, so "Cycle Time (s)" and the schema's own
    # "cycle_time_sec" both reduce to "cycletime" and still match.
    "s",
)


def _normalise(name: str) -> str:
    """Lowercase, strip everything but letters and digits, drop a trailing
    unit suffix -- so header cosmetics never decide whether a column maps."""
    base = "".join(ch for ch in str(name).lower() if ch.isalnum())
    for suffix in sorted(_UNIT_SUFFIXES, key=len, reverse=True):
        if base.endswith(suffix) and len(base) > len(suffix) + 2:
            return base[: -len(suffix)]
    return base


def _candidates(schema_col: str) -> List[str]:
    return [_normalise(schema_col)] + [_normalise(a) for a in ALIASES.get(schema_col, [])]


def _resolve_column(schema_col, df_columns, explicit_mapping):
    """The upload column that should be read as `schema_col`, or None."""
    if explicit_mapping and schema_col in explicit_mapping:
        wanted = explicit_mapping[schema_col]
        for col in df_columns:
            if col == wanted:
                return col
        return None

    for col in df_columns:
        if col == schema_col:
            return col

    wanted = _candidates(schema_col)
    for col in df_columns:
        if _normalise(col) in wanted:
            return col
    return None


def detect_subsystem(df_columns, explicit_mapping=None):
    """
    Pick the subsystem whose modelled signals the upload covers best.

    Returns (subsystem_type, report) where report lists, per subsystem, how
    many of its signals were found -- so a rejected upload can say "this
    looks closest to bogie: 3 of 5 signals matched, missing X and Y" rather
    than just failing.
    """
    report = {}
    for st, cfg in sc.SUBSYSTEMS.items():
        matched, missing = {}, []
        for sig in cfg["signal_cols"]:
            found = _resolve_column(sig, df_columns, explicit_mapping)
            if found is None:
                missing.append(sig)
            else:
                matched[sig] = found
        report[st] = {
            "matched": matched,
            "missing": missing,
            "match_count": len(matched),
            "required_count": len(cfg["signal_cols"]),
        }

    best = max(report, key=lambda st: (report[st]["match_count"], -len(report[st]["missing"])))
    return best, report


def _resolve_id_column(cfg, df, explicit_mapping):
    """
    The unit-id column. Falls back to a structural guess: a non-numeric
    column whose distinct-value count is small relative to the row count is
    almost certainly a unit identifier, and getting this wrong is loud
    (every row becomes its own unit, so nothing has enough history to form
    a window) rather than silently wrong.
    """
    found = _resolve_column(cfg["id_col"], df.columns, explicit_mapping)
    if found is not None:
        return found, None

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_unique = df[col].nunique(dropna=True)
        if 1 <= n_unique <= max(2, len(df) // 10):
            return col, f"No unit-id column matched by name; using '{col}' ({n_unique} distinct values)."
    return None, None


def _resolve_timestamp_column(cfg, df, explicit_mapping):
    found = _resolve_column(cfg["timestamp_col"], df.columns, explicit_mapping)
    if found is not None:
        return found, None

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col].head(200), errors="coerce")
        if parsed.notna().mean() > 0.9:
            return col, f"No timestamp column matched by name; using '{col}' (parses as dates)."
    return None, None


def read_upload(raw: bytes, filename: str) -> pd.DataFrame:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File is {len(raw) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )
    name = (filename or "").lower()
    buf = io.BytesIO(raw)
    if name.endswith(".parquet"):
        df = pd.read_parquet(buf)
    else:
        df = pd.read_csv(buf)
    if len(df) == 0:
        raise ValueError("File parsed but contains no rows.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"File has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    return df


def score_upload(raw, filename, artifacts, subsystem_type=None, explicit_mapping=None):
    """
    Score an uploaded file and return a dashboard-ready result dict.

    artifacts: the API's loaded ARTIFACTS dict, keyed by subsystem type --
    the same frozen model/scaler/thresholds serving every other endpoint.
    """
    df_raw = read_upload(raw, filename)
    warnings: List[str] = []

    detected, match_report = detect_subsystem(df_raw.columns, explicit_mapping)
    if subsystem_type is None:
        subsystem_type = detected
    elif subsystem_type not in sc.SUBSYSTEMS:
        raise ValueError(f"Unknown subsystem '{subsystem_type}'. Known: {list(sc.SUBSYSTEMS)}.")

    cfg = sc.SUBSYSTEMS[subsystem_type]
    art = artifacts.get(subsystem_type)
    if art is None:
        raise ValueError(f"No trained model loaded for subsystem '{subsystem_type}'.")

    info = match_report[subsystem_type]
    if info["missing"]:
        raise ValueError(
            f"This file is closest to the '{subsystem_type}' model, but "
            f"{len(info['missing'])} of its {info['required_count']} required signals "
            f"could not be matched: {', '.join(info['missing'])}. "
            f"Columns found in the file: {', '.join(map(str, df_raw.columns))}. "
            "Rename those columns, or supply an explicit mapping."
        )

    id_col_src, id_note = _resolve_id_column(cfg, df_raw, explicit_mapping)
    if id_note:
        warnings.append(id_note)
    if id_col_src is None:
        raise ValueError(
            f"No unit-id column found. Expected something like '{cfg['id_col']}'. "
            f"Columns found: {', '.join(map(str, df_raw.columns))}."
        )

    ts_col_src, ts_note = _resolve_timestamp_column(cfg, df_raw, explicit_mapping)
    if ts_note:
        warnings.append(ts_note)
    if ts_col_src is None:
        raise ValueError(
            f"No timestamp column found. Expected something like '{cfg['timestamp_col']}'. "
            f"Columns found: {', '.join(map(str, df_raw.columns))}."
        )

    train_col_src = _resolve_column(cfg["train_col"], df_raw.columns, explicit_mapping)

    # Build a frame in the schema's own column names, so every pipeline
    # function below runs on the upload unchanged.
    rename = {id_col_src: cfg["id_col"], ts_col_src: cfg["timestamp_col"]}
    for sig, src in info["matched"].items():
        rename[src] = sig
    if train_col_src:
        rename[train_col_src] = cfg["train_col"]

    keep_src = list(dict.fromkeys(list(rename.keys())))
    df = df_raw[keep_src].rename(columns=rename).copy()

    unmapped = [c for c in df_raw.columns if c not in rename]
    if unmapped:
        warnings.append(
            f"{len(unmapped)} column(s) in the file were not used by the "
            f"{subsystem_type} model: {', '.join(map(str, unmapped[:8]))}"
            + ("…" if len(unmapped) > 8 else "")
        )

    if cfg["train_col"] not in df.columns:
        df[cfg["train_col"]] = "UPLOADED"
        warnings.append("No train/consist column found; all units grouped under 'UPLOADED'.")

    df[cfg["timestamp_col"]] = pd.to_datetime(df[cfg["timestamp_col"]], errors="coerce")
    for sig in cfg["signal_cols"]:
        df[sig] = pd.to_numeric(df[sig], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[cfg["timestamp_col"], cfg["id_col"]] + cfg["signal_cols"])
    if len(df) < before:
        warnings.append(
            f"Dropped {before - len(df):,} of {before:,} rows with a missing "
            "timestamp, unit id, or signal value."
        )
    if len(df) == 0:
        raise ValueError("Every row was dropped for missing timestamp, unit id, or signal values.")

    df = df.sort_values([cfg["id_col"], cfg["timestamp_col"]]).reset_index(drop=True)

    scaler = art["scaler"]
    thresholds = art["thresholds"]
    window_length = art["window_length"]
    healthy_median = art["healthy_median_recon_error"]

    # Different-equipment check: compare the upload's own per-signal mean
    # with the healthy population the scaler was fit on.
    distribution = []
    for i, sig in enumerate(cfg["signal_cols"]):
        upload_mean = float(df[sig].mean())
        train_mean = float(scaler.mean_[i])
        train_std = float(scaler.scale_[i]) or 1.0
        shift = (upload_mean - train_mean) / train_std
        distribution.append({
            "signal": sig,
            "label": sc.signal_meta(sig)["label"],
            "unit": sc.signal_meta(sig)["unit"],
            "upload_mean": round(upload_mean, 3),
            "training_healthy_mean": round(train_mean, 3),
            "shift_sigma": round(float(shift), 2),
        })
    shifted = [d for d in distribution if abs(d["shift_sigma"]) > DISTRIBUTION_SHIFT_SIGMA]
    if shifted:
        warnings.append(
            "Scored against the existing trained model, but "
            + ", ".join(f"{d['label']} sits {d['shift_sigma']:+.1f}σ from the healthy "
                        f"population it was calibrated on" for d in shifted)
            + ". If this is different equipment or different units, these risk levels are "
              "not trustworthy — the pipeline needs retraining on it."
        )

    df = compute_baseline_flags(df, subsystem_type)

    unit_ids = sorted(df[cfg["id_col"]].astype(str).unique())
    df[cfg["id_col"]] = df[cfg["id_col"]].astype(str)

    too_short = [
        uid for uid in unit_ids
        if (df[cfg["id_col"]] == uid).sum() < window_length
    ]
    if too_short:
        warnings.append(
            f"{len(too_short)} unit(s) have fewer than {window_length} readings and could "
            f"not be scored by the autoencoder: {', '.join(too_short[:6])}"
            + ("…" if len(too_short) > 6 else "")
        )

    X, end_index, _ = build_windows(
        df, subsystem_type, unit_ids, scaler, window_length, AE_SCORING_STRIDE
    )
    df["ae_recon_error"] = np.nan
    if len(X):
        df.loc[end_index, "ae_recon_error"] = score_windows(art["model"], art["device"], X)

    df = fuse_risk(df, "ae_recon_error", "baseline_flag", thresholds)

    units = []
    for uid, g in df.groupby(cfg["id_col"]):
        g = g.sort_values(cfg["timestamp_col"])
        scored = g[g["ae_recon_error"].notna()]
        if len(scored) == 0:
            units.append({
                "unit_id": uid,
                "subsystem_type": subsystem_type,
                "train_id": str(g.iloc[-1][cfg["train_col"]]),
                "reading_count": int(len(g)),
                "scored": False,
                "risk_level": None,
                "ae_recon_error": None,
                "health_index": None,
                "note": f"only {len(g)} readings; needs {window_length} to form one window",
            })
            continue

        latest = scored.iloc[-1]
        err = float(latest["ae_recon_error"])
        index_value = hi.health_index(err, thresholds, healthy_median)
        units.append({
            "unit_id": uid,
            "subsystem_type": subsystem_type,
            "train_id": str(latest[cfg["train_col"]]),
            "latest_timestamp": str(latest[cfg["timestamp_col"]]),
            "reading_count": int(len(g)),
            "scored": True,
            "risk_level": latest["risk_level"] if pd.notna(latest["risk_level"]) else "Normal",
            "ae_recon_error": err,
            "health_index": index_value,
            "health_band": hi.health_band(index_value),
            "error_vs_healthy_median": hi.error_vs_healthy(err, healthy_median),
            "baseline_flag": bool(latest["baseline_flag"]),
            "baseline_max_abs_z": None if pd.isna(latest["baseline_max_abs_z"]) else float(latest["baseline_max_abs_z"]),
            "signals": {sig: float(latest[sig]) for sig in cfg["signal_cols"]},
            "worst_health_index": min(
                [v for v in (hi.health_index(float(e), thresholds, healthy_median)
                             for e in scored["ae_recon_error"]) if v is not None],
                default=index_value,
            ),
        })

    units.sort(key=lambda u: (u["health_index"] is None, u["health_index"]))

    counts = {}
    for u in units:
        if u["risk_level"]:
            counts[u["risk_level"]] = counts.get(u["risk_level"], 0) + 1

    return {
        "filename": filename,
        "subsystem_type": subsystem_type,
        "subsystem_was_detected": subsystem_type == detected,
        "rows_parsed": int(before),
        "rows_scored": int(len(df)),
        "unit_count": len(units),
        "scored_unit_count": sum(1 for u in units if u["scored"]),
        "time_range": {
            "start": str(df[cfg["timestamp_col"]].min()),
            "end": str(df[cfg["timestamp_col"]].max()),
        },
        "column_mapping": {v: k for k, v in rename.items()},
        "thresholds": thresholds,
        "healthy_median_recon_error": healthy_median,
        "distribution_check": distribution,
        "status_counts": counts,
        "units": units,
        "warnings": warnings,
        "model_note": (
            "Scored with the frozen model from the last pipeline run — no retraining. "
            "Thresholds and the healthy baseline come from that run's held-out healthy units."
        ),
    }

"""
Inference for the models trained on the REAL PS3 Door and Rail_Corrugation
datasets (real_* files in models/). Kept separate from the synthetic
pipeline: none of this touches door_autoencoder.pt / door_scaler.joblib.

Feature definitions below were verified against the artifacts themselves:
the Door scaler's feature_names_in_ and the Rail classifier's
feature_names_in_ fix the exact names and order.
"""

import os
import threading

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DOOR_SIGNAL_COLS = [
    "Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force",
    "Door opening time(.1s)", "Door closing time(.1s)", "Door leaf position",
]
DOOR_STATS = ("mean", "std", "max", "min")
DOOR_FEATURE_NAMES = [f"{c}_{s}" for c in DOOR_SIGNAL_COLS for s in DOOR_STATS]

# The controller only logs while a door is moving (~20ms cadence within a
# cycle); any larger gap between consecutive rows is a cycle boundary. Same
# rule as ps3_door/config.py's GAP_THRESHOLD_SEC, which reconstructs all
# 110/110 Train_Segments_Answer.csv segments exactly.
GAP_THRESHOLD_SEC = 0.1

NORMAL_LABEL = "Normal"
ABNORMAL_LABEL = "Abnormal resistance"

RAIL_STATS = ("rms", "std", "max")


class RealDoorAutoencoder(nn.Module):
    """Mirrors the Colab `Autoencoder(input_dim=24, hidden_dim=16)` exactly:
    encoder Linear-ReLU-Linear-ReLU, decoder Linear-ReLU-Linear (no output
    activation)."""

    def __init__(self, n_features=24, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class InputError(ValueError):
    """Bad uploaded data (maps to HTTP 422)."""


def load_real_models(models_dir):
    def p(name):
        return os.path.join(models_dir, name)

    scaler = joblib.load(p("real_door_scaler.pkl"))
    if list(scaler.feature_names_in_) != DOOR_FEATURE_NAMES:
        raise RuntimeError("real_door_scaler.pkl feature names/order differ from DOOR_FEATURE_NAMES")

    model = RealDoorAutoencoder(n_features=len(DOOR_FEATURE_NAMES))
    model.load_state_dict(torch.load(p("real_door_autoencoder.pt"), map_location="cpu"))
    model.eval()

    rail_clf = joblib.load(p("real_rail_classifier.pkl"))
    rail_cols = []
    for name in rail_clf.feature_names_in_:
        base, _, stat = name.rpartition("_")
        if stat not in RAIL_STATS:
            raise RuntimeError(f"unexpected rail feature name {name!r}")
        if not rail_cols or rail_cols[-1] != base:
            rail_cols.append(base)
    if [f"{c}_{s}" for c in rail_cols for s in RAIL_STATS] != list(rail_clf.feature_names_in_):
        raise RuntimeError("real_rail_classifier.pkl feature order is not [rms, std, max] per column")

    return {
        "door_model": model,
        "door_scaler": scaler,
        "door_threshold": float(joblib.load(p("real_door_threshold.pkl"))),
        "rail_clf": rail_clf,
        "rail_encoder": joblib.load(p("real_rail_label_encoder.pkl")),
        "rail_columns": rail_cols,
        "rail_lock": threading.Lock(),
    }


# ---------------------------------------------------------------------------
# Door
# ---------------------------------------------------------------------------

def parse_door_datetime(series):
    """Y-M-D-H-Mi-S-ms, hyphen-separated, NOT zero-padded (e.g.
    '2023-7-5-0-0-3-760')."""
    parts = series.astype(str).str.split("-", expand=True)
    if parts.shape[1] != 7:
        raise InputError("Datetime must be Year-Month-Date-Hour-Minute-Second-Millisecond "
                         "(7 hyphen-separated fields, e.g. 2023-7-5-0-0-3-760)")
    try:
        parts = parts.astype(int)
    except (ValueError, TypeError):
        raise InputError("Datetime has non-numeric or missing fields; expected e.g. 2023-7-5-0-0-3-760")
    parts.columns = ["y", "mo", "d", "h", "mi", "s", "ms"]
    try:
        base = pd.to_datetime(dict(year=parts["y"], month=parts["mo"], day=parts["d"],
                                   hour=parts["h"], minute=parts["mi"], second=parts["s"]))
    except (ValueError, TypeError) as e:
        raise InputError(f"Datetime contains an invalid date/time: {e}")
    return base + pd.to_timedelta(parts["ms"], unit="ms")


def format_door_datetime(ts):
    return f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}-{ts.microsecond // 1000}"


def predict_door(df, models):
    missing = [c for c in ["Datetime"] + DOOR_SIGNAL_COLS if c not in df.columns]
    if missing:
        raise InputError(f"missing required columns: {missing}")
    if len(df) == 0:
        raise InputError("CSV has no rows")

    signals = df[DOOR_SIGNAL_COLS].apply(pd.to_numeric, errors="coerce")
    bad = int(signals.isna().any(axis=1).sum())
    if bad:
        raise InputError(f"{bad} row(s) have missing or non-numeric values in {DOOR_SIGNAL_COLS}")

    work = signals.copy()
    work["dt"] = parse_door_datetime(df["Datetime"])
    work = work.sort_values("dt").reset_index(drop=True)

    gap = work["dt"].diff().dt.total_seconds()
    seg = ((gap.isna()) | (gap > GAP_THRESHOLD_SEC)).cumsum()

    grouped = work.groupby(seg)
    feats = {}
    for c in DOOR_SIGNAL_COLS:
        g = grouped[c]
        feats[f"{c}_mean"] = g.mean()
        feats[f"{c}_std"] = g.std(ddof=0)
        feats[f"{c}_max"] = g.max()
        feats[f"{c}_min"] = g.min()
    X = pd.DataFrame(feats)[DOOR_FEATURE_NAMES]

    scaled = models["door_scaler"].transform(X).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(scaled)
        recon_error = ((models["door_model"](x) - x) ** 2).mean(dim=1).numpy()

    starts = grouped["dt"].first()
    ends = grouped["dt"].last()
    is_abnormal = recon_error > models["door_threshold"]
    predictions = [
        {"start_time": format_door_datetime(s), "end_time": format_door_datetime(e),
         "prediction": ABNORMAL_LABEL if a else NORMAL_LABEL}
        for s, e, a in zip(starts, ends, is_abnormal)
    ]
    return {
        "n_rows": int(len(work)),
        "n_segments": len(predictions),
        "n_abnormal": int(is_abnormal.sum()),
        "threshold": models["door_threshold"],
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Rail
# ---------------------------------------------------------------------------

def predict_rail(df, models):
    cols = models["rail_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise InputError(
            f"expected the {len(cols)} Rail_Corrugation columns; missing {len(missing)}, "
            f"e.g. {missing[:3]}"
        )
    if len(df) < 2:
        raise InputError("CSV needs a full recording (the dataset uses 10,000 rows per file)")

    a = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(a).all():
        raise InputError("recording contains missing, non-numeric or non-finite values")

    feats = np.empty(len(cols) * len(RAIL_STATS))
    feats[0::3] = np.sqrt((a ** 2).mean(axis=0))
    feats[1::3] = a.std(axis=0, ddof=1)
    feats[2::3] = a.max(axis=0)
    names = list(models["rail_clf"].feature_names_in_)
    X = pd.DataFrame(feats.reshape(1, -1), columns=names)

    with models["rail_lock"]:
        proba = models["rail_clf"].predict_proba(X)[0]
    classes = models["rail_encoder"].inverse_transform(models["rail_clf"].classes_)
    best = int(np.argmax(proba))
    return {
        "prediction": str(classes[best]),
        "probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
        "n_rows": int(len(df)),
    }

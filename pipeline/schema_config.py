"""
Single source of truth for column names / schema mapping.

Every other module in `pipeline/` and `api/` reads column names through this
file instead of hardcoding strings. When the real LTA dataset arrives, this
should be the *only* file that needs editing to repoint the pipeline at it
(plus possibly new entries in SIGNAL_COLUMNS if the real sensors expose
different channels) -- model code in baseline.py / autoencoder.py / fusion.py
should not need to change.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))
MODELS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "models"))
VALIDATION_OUTPUT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "validation_outputs"))

DOOR_DATA_PATH = os.path.join(DATA_DIR, "door_data.csv")
BOGIE_DATA_PATH = os.path.join(DATA_DIR, "bogie_data.csv")
CAR_DATA_PATH = os.path.join(DATA_DIR, "car_data.csv")
FAULT_LOG_PATH = os.path.join(DATA_DIR, "fault_log.csv")

SCORED_DOOR_PATH = os.path.join(DATA_DIR, "door_scored.csv")
SCORED_BOGIE_PATH = os.path.join(DATA_DIR, "bogie_scored.csv")
SCORED_CAR_PATH = os.path.join(DATA_DIR, "car_scored.csv")
FLEET_STATUS_PATH = os.path.join(DATA_DIR, "fleet_status.csv")

# ---------------------------------------------------------------------------
# Per-subsystem schema. Everything downstream (baseline, autoencoder,
# fusion, API, dashboard) keys off of this dict rather than literal column
# names, so remapping to a real dataset means changing values here only.
# ---------------------------------------------------------------------------
SUBSYSTEMS = {
    "door": {
        "data_path": DOOR_DATA_PATH,
        "scored_path": SCORED_DOOR_PATH,
        "id_col": "door_id",
        "timestamp_col": "timestamp",
        "train_col": "train_id",
        # Signals fed to both the rolling baseline and the autoencoder.
        "signal_cols": [
            "cycle_time_sec",
            "motor_current_amps",
            "motor_temp_c",
            "motor_rpm",
        ],
        # Extra columns kept around for context / dashboard display, not
        # modeled directly.
        "context_cols": ["cycle_count"],
    },
    "bogie": {
        "data_path": BOGIE_DATA_PATH,
        "scored_path": SCORED_BOGIE_PATH,
        "id_col": "bogie_id",
        "timestamp_col": "timestamp",
        "train_col": "train_id",
        "signal_cols": [
            "temperature_c",
            "vibration_rms",
            "traction_motor_temp_c",
            "traction_motor_rpm",
            "brake_capacity_pct",
        ],
        "context_cols": ["axle_load_kg"],
    },
    # Car-body auxiliary systems: HVAC, saloon lighting, the battery that
    # carries the car through a traction supply gap, and the train radio.
    # These are the systems a passenger notices failing first, and they
    # share a car body (and in practice an auxiliary converter), so they
    # are modelled as one multivariate unit rather than four.
    "car": {
        "data_path": CAR_DATA_PATH,
        "scored_path": SCORED_CAR_PATH,
        "id_col": "car_id",
        "timestamp_col": "timestamp",
        "train_col": "train_id",
        "signal_cols": [
            "hvac_supply_temp_c",
            "hvac_current_amps",
            "lighting_load_pct",
            "battery_soc_pct",
            "battery_voltage_v",
            "comms_rssi_dbm",
        ],
        "context_cols": ["hvac_setpoint_c"],
    },
}

# ---------------------------------------------------------------------------
# Display metadata: unit, short label, and which direction is "worse".
# ---------------------------------------------------------------------------
# Purely presentational -- the models never read this. It exists so the API
# and dashboard can render "62.5 degC" instead of a bare "62.5", and so a
# falling signal (brake capacity, battery voltage, rpm off rated speed) is
# drawn as a deterioration rather than an improvement. Keyed by raw column
# name, so a real-data column added to signal_cols above only needs an entry
# here to render correctly everywhere.
SIGNAL_METADATA = {
    "cycle_time_sec":        {"label": "Door cycle time",      "unit": "s",    "worse": "high", "group": "Door mechanism"},
    "motor_current_amps":    {"label": "Door motor current",   "unit": "A",    "worse": "high", "group": "Door mechanism"},
    "motor_temp_c":          {"label": "Door motor temp",      "unit": "degC", "worse": "high", "group": "Door mechanism"},
    "motor_rpm":             {"label": "Door motor speed",     "unit": "rpm",  "worse": "low",  "group": "Door mechanism"},
    "cycle_count":           {"label": "Lifetime cycles",      "unit": "",     "worse": "high", "group": "Door mechanism"},

    "temperature_c":         {"label": "Axle bearing temp",    "unit": "degC", "worse": "high", "group": "Running gear"},
    "vibration_rms":         {"label": "Vibration RMS",        "unit": "g",    "worse": "high", "group": "Running gear"},
    "traction_motor_temp_c": {"label": "Traction motor temp",  "unit": "degC", "worse": "high", "group": "Traction"},
    "traction_motor_rpm":    {"label": "Traction motor speed", "unit": "rpm",  "worse": "low",  "group": "Traction"},
    "brake_capacity_pct":    {"label": "Brake capacity",       "unit": "%",    "worse": "low",  "group": "Braking"},
    "axle_load_kg":          {"label": "Axle load",            "unit": "kg",   "worse": "high", "group": "Running gear"},

    "hvac_supply_temp_c":    {"label": "HVAC supply air",      "unit": "degC", "worse": "high", "group": "Air conditioning"},
    "hvac_current_amps":     {"label": "HVAC current",         "unit": "A",    "worse": "high", "group": "Air conditioning"},
    "lighting_load_pct":     {"label": "Saloon lighting load", "unit": "%",    "worse": "low",  "group": "Lighting"},
    "battery_soc_pct":       {"label": "Battery charge",       "unit": "%",    "worse": "low",  "group": "Battery"},
    "battery_voltage_v":     {"label": "Battery voltage",      "unit": "V",    "worse": "low",  "group": "Battery"},
    "comms_rssi_dbm":        {"label": "Radio signal",         "unit": "dBm",  "worse": "low",  "group": "Communications"},
    "hvac_setpoint_c":       {"label": "HVAC setpoint",        "unit": "degC", "worse": "high", "group": "Air conditioning"},
}


def signal_meta(col):
    """Display metadata for a signal column, with a safe default for an
    unmapped real-data column so nothing renders as undefined."""
    return SIGNAL_METADATA.get(col, {"label": col, "unit": "", "worse": "high", "group": "Other"})

FAULT_LOG_COLUMNS = {
    "timestamp": "timestamp",
    "subsystem_id": "subsystem_id",
    "subsystem_type": "subsystem_type",
    "fault_type": "fault_type",
    "severity": "severity",
}

# subsystem_type values used in fault_log.csv, matching keys of SUBSYSTEMS
SUBSYSTEM_TYPES = list(SUBSYSTEMS.keys())

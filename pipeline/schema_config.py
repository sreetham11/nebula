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
FAULT_LOG_PATH = os.path.join(DATA_DIR, "fault_log.csv")

SCORED_DOOR_PATH = os.path.join(DATA_DIR, "door_scored.csv")
SCORED_BOGIE_PATH = os.path.join(DATA_DIR, "bogie_scored.csv")
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
        "signal_cols": ["cycle_time_sec", "motor_current_amps"],
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
        "signal_cols": ["temperature_c", "vibration_rms"],
        "context_cols": ["axle_load_kg"],
    },
}

FAULT_LOG_COLUMNS = {
    "timestamp": "timestamp",
    "subsystem_id": "subsystem_id",
    "subsystem_type": "subsystem_type",
    "fault_type": "fault_type",
    "severity": "severity",
}

# subsystem_type values used in fault_log.csv, matching keys of SUBSYSTEMS
SUBSYSTEM_TYPES = list(SUBSYSTEMS.keys())

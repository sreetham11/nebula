"""
Configuration / constants block for the synthetic data generator.

Everything that controls fleet size, time range, sampling rate, noise levels,
and degradation behavior lives here so the dataset can be regenerated at a
different scale or with different fault behavior without touching the
generation logic in generate_data.py.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Fleet composition
# ---------------------------------------------------------------------------
N_TRAINS = 6
DOORS_PER_TRAIN = 2
BOGIES_PER_TRAIN = 2

N_DOORS = N_TRAINS * DOORS_PER_TRAIN      # 12
N_BOGIES = N_TRAINS * BOGIES_PER_TRAIN    # 12

# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------
START_DATE = "2026-08-01 00:00:00"
DURATION_WEEKS = 6
DURATION_DAYS = DURATION_WEEKS * 7

# Reading cadence: each unit's own timestamps are independent, spaced by a
# random interval in this range (minutes). This keeps ~50k-200k total rows
# across both subsystems while looking like real, jittered sensor polling
# rather than a perfectly synced clock.
READING_INTERVAL_MIN_MINUTES = 5
READING_INTERVAL_MAX_MINUTES = 15

# ---------------------------------------------------------------------------
# Degradation injection
# ---------------------------------------------------------------------------
N_DEGRADING_DOORS = 4
N_DEGRADING_BOGIES = 4

# Degradation must start early enough to fully play out before the dataset
# ends, and not so early there's no "healthy" history beforehand.
DEGRADATION_EARLIEST_START_DAY = 3
DEGRADATION_LATEST_START_DAY = DURATION_DAYS - 12  # leaves room for a 10-day ramp + buffer

DEGRADATION_DURATION_MIN_DAYS = 3
DEGRADATION_DURATION_MAX_DAYS = 10

# Piecewise ramp/plateau shape of the degradation trajectory (see
# _build_drift_trajectory in generate_data.py). Non-trivial on purpose:
# noisy, staged, with plateaus -- not a clean linear ramp -- so a naive
# fixed threshold detector struggles while a model that tracks trend +
# reconstruction error over a window does better.
DEGRADATION_MIN_SEGMENTS = 4
DEGRADATION_MAX_SEGMENTS = 7
DEGRADATION_PLATEAU_PROB = 0.3
DEGRADATION_SEGMENT_NOISE_STD_FRAC = 0.15  # fraction of segment's own budget
DEGRADATION_BACKSLIDE_PROB = 0.1           # chance a point dips below cumulative trend (sensor noise/maintenance blip)

# Fault is logged once cumulative drift first crosses this fraction of its
# final (target) magnitude -- i.e. "failure" happens before the ramp
# necessarily maxes out, which is realistic (units get pulled for
# maintenance once clearly faulty). Randomized per-unit within this range so
# severity-at-detection (and thus lead time) varies across the fleet instead
# of every fault landing at the same trend fraction.
FAULT_TRIGGER_FRACTION_RANGE = (0.55, 0.95)

# ---------------------------------------------------------------------------
# Door signal baselines + noise
# ---------------------------------------------------------------------------
DOOR_CYCLE_TIME_BASE_SEC = 4.2
DOOR_CYCLE_TIME_NOISE_STD = 0.15

DOOR_MOTOR_CURRENT_BASE_AMPS = 3.0
DOOR_MOTOR_CURRENT_NOISE_STD = 0.12

# Degradation targets for doors (added on top of baseline+noise)
DOOR_DEGRADED_CYCLE_TIME_DELTA_RANGE = (1.0, 3.0)     # seconds slower
DOOR_DEGRADED_MOTOR_CURRENT_DELTA_RANGE = (1.5, 4.0)  # amps higher

# Usage pattern: doors cycle more during "peak" hours (proxy for
# passenger service hours), which also drives cycle_count growth and a
# mild bump in motor current noise.
DOOR_PEAK_HOURS = (6, 23)  # trains mostly idle 23:00-06:00
DOOR_CYCLES_PER_READING_PEAK = (1, 4)
DOOR_CYCLES_PER_READING_OFFPEAK = (0, 1)

# ---------------------------------------------------------------------------
# Bogie signal baselines + noise
# ---------------------------------------------------------------------------
BOGIE_TEMP_BASE_C = 38.0
BOGIE_TEMP_NOISE_STD = 1.2
BOGIE_TEMP_DIURNAL_AMPLITUDE_C = 2.0  # ambient day/night swing

BOGIE_VIBRATION_BASE_RMS = 0.35
BOGIE_VIBRATION_NOISE_STD = 0.04

BOGIE_AXLE_LOAD_BASE_KG = 9500.0
BOGIE_AXLE_LOAD_NOISE_STD = 150.0  # ridership-driven variation, not a degradation signal

# Degradation targets for bogies
BOGIE_DEGRADED_TEMP_DELTA_RANGE = (15.0, 30.0)          # deg C above baseline at peak
BOGIE_DEGRADED_VIBRATION_STD_MULT_RANGE = (2.0, 4.0)    # vibration noise std multiplier (variance blow-up)

# ---------------------------------------------------------------------------
# Fault taxonomy
# ---------------------------------------------------------------------------
DOOR_FAULT_TYPES = ["motor_wear", "seal_degradation", "track_misalignment"]
BOGIE_FAULT_TYPES = ["bearing_failure", "suspension_wear", "wheel_flat"]

# severity is an integer 1 (minor) - 5 (severe), correlated with how far
# past the fault trigger threshold the unit had drifted when logged.
SEVERITY_MIN = 1
SEVERITY_MAX = 5

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))

DOOR_DATA_PATH = os.path.join(DATA_DIR, "door_data.csv")
BOGIE_DATA_PATH = os.path.join(DATA_DIR, "bogie_data.csv")
FAULT_LOG_PATH = os.path.join(DATA_DIR, "fault_log.csv")

RNG = np.random.default_rng(RANDOM_SEED)

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
# Car-level auxiliary systems (HVAC, saloon lighting, battery, radio). One
# "car" unit per car body, matching the 2-car consist the door/bogie counts
# above already imply, so the digital twin can line CAR_xx_n up with
# DOOR_xx_n / BOGIE_xx_n in the same car.
CARS_PER_TRAIN = 2

N_DOORS = N_TRAINS * DOORS_PER_TRAIN      # 12
N_BOGIES = N_TRAINS * BOGIES_PER_TRAIN    # 12
N_CARS = N_TRAINS * CARS_PER_TRAIN        # 12

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
N_DEGRADING_CARS = 3

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

# Door operator motor winding temperature. Sits well above ambient because
# the motor is duty-cycled all service day; rises with usage (peak hours)
# and, when the mechanism binds, with the extra work the motor is doing.
DOOR_MOTOR_TEMP_BASE_C = 41.0
DOOR_MOTOR_TEMP_NOISE_STD = 1.4
DOOR_MOTOR_TEMP_PEAK_BUMP_C = 3.5

# Door operator motor speed during a cycle. A healthy operator runs close
# to its rated speed; a binding or worn one is dragged DOWN off rated speed
# while drawing more current -- which is why rpm and current moving in
# opposite directions is the informative pattern, not either one alone.
DOOR_MOTOR_RPM_BASE = 1450.0
DOOR_MOTOR_RPM_NOISE_STD = 22.0

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

# Traction motor mounted on the bogie. Much hotter than the axle bearings
# and strongly duty-cycle driven, so it carries a larger diurnal swing.
BOGIE_TRACTION_TEMP_BASE_C = 68.0
BOGIE_TRACTION_TEMP_NOISE_STD = 2.6
BOGIE_TRACTION_TEMP_DIURNAL_AMPLITUDE_C = 6.0

# Traction motor speed, averaged over the polling interval. Near zero
# overnight when the train is stabled, at service speed during peak hours.
BOGIE_TRACTION_RPM_SERVICE = 1800.0
BOGIE_TRACTION_RPM_STABLED = 120.0
BOGIE_TRACTION_RPM_NOISE_STD = 45.0

# Friction-brake capacity as a percentage of design retardation, as the
# brake control unit self-reports it after each application. Healthy stock
# sits just under 100% and wears down slowly; this is the channel a brake
# fault shows up on first.
BOGIE_BRAKE_CAPACITY_BASE_PCT = 98.0
BOGIE_BRAKE_CAPACITY_NOISE_STD = 0.55

# Vibration is special-cased in the generator: the fault signature is a
# blow-up in VARIANCE rather than a shift in mean, so it is driven by a
# noise-std multiplier instead of an additive delta like every other signal.
BOGIE_DEGRADED_VIBRATION_STD_MULT_RANGE = (2.0, 4.0)

# ---------------------------------------------------------------------------
# Car auxiliary (HVAC / lighting / battery / radio) signal baselines + noise
# ---------------------------------------------------------------------------
# Supply-air temperature at the saloon diffuser. The HVAC pack holds this
# near setpoint; a pack losing capacity cannot pull it down far enough.
CAR_HVAC_SETPOINT_C = 24.0
CAR_HVAC_SUPPLY_TEMP_BASE_C = 12.5
CAR_HVAC_SUPPLY_TEMP_NOISE_STD = 0.55
# Ambient load: the pack works harder in the afternoon, so supply temp
# creeps up and current draw rises even on a perfectly healthy car.
CAR_HVAC_DIURNAL_AMPLITUDE_C = 1.1

CAR_HVAC_CURRENT_BASE_AMPS = 27.5
CAR_HVAC_CURRENT_NOISE_STD = 1.1
CAR_HVAC_CURRENT_DIURNAL_AMPLITUDE_A = 3.0

# Saloon lighting: percentage of the car's LED string drawing its expected
# load. Individual driver failures step this down rather than drift it.
CAR_LIGHTING_LOAD_BASE_PCT = 97.0
CAR_LIGHTING_LOAD_NOISE_STD = 0.8

# Battery state of charge and terminal voltage. SOC follows a shallow
# overnight-charge / daytime-draw cycle; voltage is the earlier indicator
# of cell degradation because SOC is itself estimated from it.
CAR_BATTERY_SOC_BASE_PCT = 92.0
CAR_BATTERY_SOC_NOISE_STD = 1.3
CAR_BATTERY_SOC_DIURNAL_AMPLITUDE_PCT = 4.0

CAR_BATTERY_VOLTAGE_BASE_V = 110.0
CAR_BATTERY_VOLTAGE_NOISE_STD = 0.7

# Train-to-ground radio received signal strength, in dBm (negative; closer
# to zero is stronger). Varies with where the train is on the line, which
# is why its healthy band is wide and a fault has to be a sustained shift.
CAR_COMMS_RSSI_BASE_DBM = -68.0
CAR_COMMS_RSSI_NOISE_STD = 2.6

# ---------------------------------------------------------------------------
# Fault taxonomy
# ---------------------------------------------------------------------------
DOOR_FAULT_TYPES = ["motor_wear", "seal_degradation", "track_misalignment"]
BOGIE_FAULT_TYPES = ["bearing_failure", "suspension_wear", "wheel_flat", "brake_pad_wear"]
CAR_FAULT_TYPES = [
    "hvac_compressor_wear",
    "battery_cell_degradation",
    "lighting_driver_failure",
    "comms_antenna_fault",
]

FAULT_TYPES_BY_SUBSYSTEM = {
    "door": DOOR_FAULT_TYPES,
    "bogie": BOGIE_FAULT_TYPES,
    "car": CAR_FAULT_TYPES,
}

# ---------------------------------------------------------------------------
# Which signals each fault type actually moves, and by how much
# ---------------------------------------------------------------------------
# Keyed subsystem -> fault_type -> {signal: (min_delta, max_delta)}. The
# delta is the signal's total excursion at the END of the degradation ramp,
# sampled per unit from that range and applied on top of baseline+noise.
# A NEGATIVE range means the signal falls (rpm dragged off rated speed,
# brake capacity worn away, battery voltage sagging, RSSI weakening).
#
# Deliberately sparse: a comms antenna fault must NOT move battery SOC, and
# a brake fault must NOT move bearing temperature much. Every subsystem's
# model therefore has to learn which COMBINATION of its channels moves
# together, which is the whole point of a multivariate autoencoder rather
# than a per-channel threshold. A signal absent from a fault's dict stays
# at its healthy baseline for that unit.
DEGRADATION_SIGNAL_DELTAS = {
    "door": {
        # Worn motor: labours, heats up, and is dragged off rated speed.
        "motor_wear": {
            "cycle_time_sec": (1.0, 2.6),
            "motor_current_amps": (1.8, 4.0),
            "motor_temp_c": (9.0, 22.0),
            "motor_rpm": (-260.0, -90.0),
        },
        # Perished seal: extra friction on close, so slower and more
        # current, but the motor itself is healthy -- barely any extra heat.
        "seal_degradation": {
            "cycle_time_sec": (1.2, 3.0),
            "motor_current_amps": (1.5, 3.2),
            "motor_temp_c": (2.0, 6.0),
        },
        # Misaligned track: mechanism binds intermittently. Current rises
        # and speed suffers; cycle time moves least of the three.
        "track_misalignment": {
            "cycle_time_sec": (0.8, 2.0),
            "motor_current_amps": (2.0, 4.0),
            "motor_temp_c": (5.0, 13.0),
            "motor_rpm": (-300.0, -120.0),
        },
    },
    "bogie": {
        # Failing axle bearing: the classic hot-box signature, with the
        # traction motor beside it picking up some of the heat.
        "bearing_failure": {
            "temperature_c": (16.0, 30.0),
            "traction_motor_temp_c": (7.0, 16.0),
        },
        # Worn suspension: variance blow-up only (see the vibration
        # std-multiplier above) -- no mean shift on any channel.
        "suspension_wear": {},
        # Wheel flat: hammering impact each revolution -- vibration variance
        # plus a modest bearing-temperature rise from the shock loading.
        "wheel_flat": {
            "temperature_c": (4.0, 11.0),
        },
        # Worn friction brake: capacity falls away, and the bogie runs
        # slightly hotter as the remaining pad area does more work.
        "brake_pad_wear": {
            "brake_capacity_pct": (-26.0, -9.0),
            "temperature_c": (3.0, 9.0),
        },
    },
    "car": {
        # Compressor losing capacity: cannot pull supply air down to
        # setpoint, and draws more current failing to.
        "hvac_compressor_wear": {
            "hvac_supply_temp_c": (3.0, 8.0),
            "hvac_current_amps": (4.0, 12.0),
        },
        # Ageing cells: terminal voltage sags first, SOC estimate follows.
        "battery_cell_degradation": {
            "battery_voltage_v": (-9.0, -3.0),
            "battery_soc_pct": (-26.0, -9.0),
        },
        # LED drivers dropping out one string at a time.
        "lighting_driver_failure": {
            "lighting_load_pct": (-22.0, -6.0),
        },
        # Antenna / feeder degradation: sustained loss of received signal.
        "comms_antenna_fault": {
            "comms_rssi_dbm": (-17.0, -6.0),
        },
    },
}

# Vibration's variance blow-up is not an additive delta, so it is listed
# separately: these bogie fault types multiply the vibration noise std.
BOGIE_VIBRATION_FAULT_TYPES = ("suspension_wear", "wheel_flat", "bearing_failure")

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
CAR_DATA_PATH = os.path.join(DATA_DIR, "car_data.csv")
FAULT_LOG_PATH = os.path.join(DATA_DIR, "fault_log.csv")

RNG = np.random.default_rng(RANDOM_SEED)

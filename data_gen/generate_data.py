"""
NEBULA X synthetic sensor data generator.

*** THIS IS SYNTHETIC DATA. ***
The real LTA telemetry dataset for this hackathon has not been released yet
(it drops during the event). Everything produced by this script is
fabricated to *mirror the expected schema* of the real dataset -- train door
sensor readings, bogie sensor readings, and a manually-verified fault log --
so that the detection pipeline and dashboard can be built, tested, and demo'd
end-to-end ahead of time. When the real dataset arrives, only the column
mapping in pipeline/schema_config.py should need to change, not the model
or generator logic.

Do not present numbers derived from this dataset as real fleet statistics.

Outputs (written to ../data/):
  - door_data.csv   : one row per reading per door unit
  - bogie_data.csv  : one row per reading per bogie unit
  - car_data.csv    : one row per reading per car unit (HVAC, saloon
                       lighting, auxiliary battery, train radio)
  - fault_log.csv   : sparse ground-truth fault events (manually-verified
                       equivalent -- in this synthetic set, the point at
                       which each injected degradation crossed a "failure"
                       threshold)

Run: python data_gen/generate_data.py
"""

import os

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------

def _generate_timestamps(rng, start_dt, end_dt, desync_max_minutes):
    """Jittered reading timestamps for one unit across [start_dt, end_dt)."""
    offset = rng.uniform(0, desync_max_minutes)
    t = start_dt + pd.Timedelta(minutes=offset)
    timestamps = []
    while t < end_dt:
        timestamps.append(t)
        step = rng.uniform(
            config.READING_INTERVAL_MIN_MINUTES,
            config.READING_INTERVAL_MAX_MINUTES,
        )
        t = t + pd.Timedelta(minutes=step)
    return pd.DatetimeIndex(timestamps).round("s")


# ---------------------------------------------------------------------------
# Degradation trajectory
# ---------------------------------------------------------------------------

def _build_normalized_trend(elapsed_frac, rng):
    """
    Piecewise ramp/plateau wear trajectory, normalized to end at 1.0.

    Deliberately NOT a clean linear ramp: random segment boundaries, each
    segment is either a "ramp" (wear accumulates) or a "plateau" (wear
    holds steady -- e.g. the unit gets a partial service, or degradation
    naturally stalls for a while), so a fixed-threshold detector sees long
    flat stretches punctuated by jumps rather than a smooth trend.
    """
    n_segments = rng.integers(config.DEGRADATION_MIN_SEGMENTS, config.DEGRADATION_MAX_SEGMENTS + 1)
    interior = np.sort(rng.uniform(0, 1, n_segments - 1))
    bounds = np.concatenate(([0.0], interior, [1.0]))

    seg_types = ["plateau" if rng.random() < config.DEGRADATION_PLATEAU_PROB else "ramp"
                 for _ in range(n_segments)]
    if "ramp" not in seg_types:
        seg_types[int(rng.integers(0, n_segments))] = "ramp"

    ramp_idx = [i for i, t in enumerate(seg_types) if t == "ramp"]
    weights = rng.uniform(0.5, 1.5, len(ramp_idx))
    weights = weights / weights.sum()
    seg_budget = dict(zip(ramp_idx, weights))

    cumulative = [0.0]
    for i in range(n_segments):
        prev = cumulative[-1]
        cumulative.append(prev + seg_budget[i] if i in seg_budget else prev)
    cumulative = np.array(cumulative)
    cumulative = cumulative / cumulative[-1]  # normalize exactly to 1.0 at the end

    trend = np.interp(elapsed_frac, bounds, cumulative)
    return trend


def _apply_noise_to_trend(trend, magnitude, rng, noise_std_frac, backslide_prob, in_window_mask):
    """
    Turn a clean [0,1] trend into a noisy real-world signal delta.

    Noise (and the occasional "backslide" dip) is only added where
    in_window_mask is True -- i.e. strictly within the unit's injected
    degradation window. Outside that window trend is already 0/pinned, and
    delta must come out to exactly 0 there too: without masking, a
    zero-mean noise term added everywhere plus a clip at 0 would leave a
    systematic bias on every pre-degradation (and, differently,
    post-window) reading, quietly making "healthy" history for degrading
    units look anomalous from day one.

    `magnitude` may be NEGATIVE, for signals a fault drags DOWN rather than
    up (motor rpm off rated speed, brake capacity, battery voltage, radio
    RSSI). Everything below is written against |magnitude| and re-signed at
    the end, so a backslide is always a move back TOWARD healthy and the
    clip always bounds the excursion on the correct side of zero.
    """
    sign = -1.0 if magnitude < 0 else 1.0
    mag = abs(magnitude)

    delta = trend * mag
    noise = rng.normal(0, noise_std_frac * mag, size=len(delta))
    backslide_mask = (rng.random(len(delta)) < backslide_prob) & in_window_mask
    noise[backslide_mask] -= np.abs(rng.normal(0, 0.3 * mag, size=int(backslide_mask.sum())))
    noise = np.where(in_window_mask, noise, 0.0)
    delta = np.clip(delta + noise, 0, mag * 1.3)
    return sign * delta


class DegradationProfile:
    """Holds the injected-fault trajectory for one degrading unit."""

    def __init__(self, unit_id, subsystem_type, start_dt, end_dt, rng):
        self.unit_id = unit_id
        self.subsystem_type = subsystem_type
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.rng = rng
        self.fault_timestamp = None
        self.fault_type = None
        self.severity = None

    def in_window(self, timestamps):
        return (timestamps >= self.start_dt) & (timestamps <= self.end_dt)

    def trend_at(self, timestamps):
        """Normalized [0,1] wear trend, 0 before window, held at final value after."""
        elapsed_frac = np.clip(
            (timestamps - self.start_dt) / (self.end_dt - self.start_dt), 0, 1
        ).astype(float)
        # Reuse full-window trend shape for any timestamp, clipped to [0,1]
        # so post-window readings stay pinned at the final wear level
        # (component remains worn until it would be serviced).
        return self._trend_fn(elapsed_frac)

    def build(self):
        # Build a fixed set of anchor points across [0,1] used to define the
        # trend shape once; trend_at() then interpolates/clips against it.
        anchor_fracs = np.linspace(0, 1, 400)
        anchor_trend = _build_normalized_trend(anchor_fracs, self.rng)
        self._trend_fn = lambda fracs: np.interp(fracs, anchor_fracs, anchor_trend)

        # Determine fault trigger point (first crossing of trigger fraction)
        # against the clean trend evaluated on a fine time grid.
        fine_times = pd.date_range(self.start_dt, self.end_dt, periods=2000)
        fine_frac = np.linspace(0, 1, 2000)
        fine_trend = self._trend_fn(fine_frac)
        trigger_fraction = self.rng.uniform(*config.FAULT_TRIGGER_FRACTION_RANGE)
        crossing_idx = np.argmax(fine_trend >= trigger_fraction)
        self.fault_timestamp = fine_times[crossing_idx]
        frac_at_fault = fine_trend[crossing_idx]
        severity = 1 + 4 * frac_at_fault + self.rng.normal(0, 0.3)
        self.severity = int(np.clip(round(severity), config.SEVERITY_MIN, config.SEVERITY_MAX))

        self.fault_type = str(self.rng.choice(
            config.FAULT_TYPES_BY_SUBSYSTEM[self.subsystem_type]
        ))

        # Sample this unit's own end-of-ramp excursion for each signal the
        # chosen fault type actually implicates. Signals absent from the
        # map are simply never perturbed for this unit, so each fault type
        # leaves a distinct multi-channel fingerprint rather than pushing
        # every channel of the subsystem at once.
        deltas = config.DEGRADATION_SIGNAL_DELTAS[self.subsystem_type][self.fault_type]
        self.signal_magnitudes = {
            sig: float(self.rng.uniform(lo, hi)) for sig, (lo, hi) in deltas.items()
        }
        return self

    def delta_for(self, signal, timestamps):
        """
        Noisy additive excursion this unit's fault applies to `signal` at
        each timestamp, or a scalar 0.0 if this fault does not touch that
        signal (broadcasts cleanly when added to a signal array).
        """
        magnitude = self.signal_magnitudes.get(signal)
        if magnitude is None:
            return 0.0
        return _apply_noise_to_trend(
            self.trend_at(timestamps), magnitude, self.rng,
            config.DEGRADATION_SEGMENT_NOISE_STD_FRAC,
            config.DEGRADATION_BACKSLIDE_PROB,
            self.in_window(timestamps),
        )


def _delta(profile, signal, timestamps):
    """profile.delta_for for a possibly-healthy unit (profile may be None)."""
    if profile is None:
        return 0.0
    return profile.delta_for(signal, timestamps)


# ---------------------------------------------------------------------------
# Door data
# ---------------------------------------------------------------------------

def _generate_door_unit(unit_id, train_id, start_dt, end_dt, rng, profile=None):
    timestamps = _generate_timestamps(rng, start_dt, end_dt, config.READING_INTERVAL_MAX_MINUTES)
    n = len(timestamps)
    hours = timestamps.hour.values

    peak_mask = (hours >= config.DOOR_PEAK_HOURS[0]) & (hours < config.DOOR_PEAK_HOURS[1])
    cycles_per_reading = np.where(
        peak_mask,
        rng.integers(*config.DOOR_CYCLES_PER_READING_PEAK, size=n, endpoint=True),
        rng.integers(*config.DOOR_CYCLES_PER_READING_OFFPEAK, size=n, endpoint=True),
    )
    start_cycle_count = int(rng.integers(2000, 60000))
    cycle_count = start_cycle_count + np.cumsum(cycles_per_reading)

    cycle_time = (
        config.DOOR_CYCLE_TIME_BASE_SEC
        + rng.normal(0, config.DOOR_CYCLE_TIME_NOISE_STD, size=n)
    )
    motor_current = (
        config.DOOR_MOTOR_CURRENT_BASE_AMPS
        + rng.normal(0, config.DOOR_MOTOR_CURRENT_NOISE_STD, size=n)
        + 0.15 * (cycles_per_reading > 2)  # mild extra draw on heavy-usage reads
    )

    # Winding temperature tracks duty cycle: warmer through service hours
    # and on readings where the door happened to cycle repeatedly.
    motor_temp = (
        config.DOOR_MOTOR_TEMP_BASE_C
        + config.DOOR_MOTOR_TEMP_PEAK_BUMP_C * peak_mask
        + 0.4 * cycles_per_reading
        + rng.normal(0, config.DOOR_MOTOR_TEMP_NOISE_STD, size=n)
    )

    # Operator speed sits just under rated speed on a healthy door, dipping
    # very slightly on heavy-usage reads as the motor warms.
    motor_rpm = (
        config.DOOR_MOTOR_RPM_BASE
        - 4.0 * cycles_per_reading
        + rng.normal(0, config.DOOR_MOTOR_RPM_NOISE_STD, size=n)
    )

    cycle_time = cycle_time + _delta(profile, "cycle_time_sec", timestamps)
    motor_current = motor_current + _delta(profile, "motor_current_amps", timestamps)
    motor_temp = motor_temp + _delta(profile, "motor_temp_c", timestamps)
    motor_rpm = motor_rpm + _delta(profile, "motor_rpm", timestamps)

    return pd.DataFrame({
        "timestamp": timestamps,
        "door_id": unit_id,
        "cycle_time_sec": np.round(cycle_time, 3),
        "motor_current_amps": np.round(motor_current, 3),
        "motor_temp_c": np.round(motor_temp, 2),
        "motor_rpm": np.round(np.clip(motor_rpm, 0, None), 1),
        "cycle_count": cycle_count,
        "train_id": train_id,
    })


# ---------------------------------------------------------------------------
# Bogie data
# ---------------------------------------------------------------------------

def _generate_bogie_unit(unit_id, train_id, start_dt, end_dt, rng, profile=None):
    timestamps = _generate_timestamps(rng, start_dt, end_dt, config.READING_INTERVAL_MAX_MINUTES)
    n = len(timestamps)
    hour_frac = timestamps.hour.values + timestamps.minute.values / 60.0

    diurnal = config.BOGIE_TEMP_DIURNAL_AMPLITUDE_C * np.sin((hour_frac - 6) / 24 * 2 * np.pi)
    temperature = (
        config.BOGIE_TEMP_BASE_C
        + diurnal
        + rng.normal(0, config.BOGIE_TEMP_NOISE_STD, size=n)
    )

    vib_std_mult = np.ones(n)
    vibration = config.BOGIE_VIBRATION_BASE_RMS + rng.normal(0, config.BOGIE_VIBRATION_NOISE_STD, size=n)

    axle_load = config.BOGIE_AXLE_LOAD_BASE_KG + rng.normal(0, config.BOGIE_AXLE_LOAD_NOISE_STD, size=n)

    # Duty cycle: the traction motor is spinning at service speed during
    # traffic hours and essentially stabled overnight. Both its temperature
    # and its speed key off the same in-service mask, so an unremarkable
    # overnight rpm drop is normal and must not read as a fault.
    hours = timestamps.hour.values
    in_service = (hours >= config.DOOR_PEAK_HOURS[0]) & (hours < config.DOOR_PEAK_HOURS[1])

    traction_diurnal = config.BOGIE_TRACTION_TEMP_DIURNAL_AMPLITUDE_C * np.sin((hour_frac - 6) / 24 * 2 * np.pi)
    traction_temp = (
        config.BOGIE_TRACTION_TEMP_BASE_C
        + traction_diurnal
        - 6.0 * (~in_service)
        + rng.normal(0, config.BOGIE_TRACTION_TEMP_NOISE_STD, size=n)
    )

    traction_rpm = np.where(
        in_service, config.BOGIE_TRACTION_RPM_SERVICE, config.BOGIE_TRACTION_RPM_STABLED
    ) + rng.normal(0, config.BOGIE_TRACTION_RPM_NOISE_STD, size=n)

    # Friction brake capacity: slow, near-flat wear on a healthy bogie.
    brake_capacity = (
        config.BOGIE_BRAKE_CAPACITY_BASE_PCT
        + rng.normal(0, config.BOGIE_BRAKE_CAPACITY_NOISE_STD, size=n)
    )

    if profile is not None:
        trend = profile.trend_at(timestamps)

        # Vibration is the one channel whose fault signature is a variance
        # blow-up rather than a mean shift, so it is driven here by a noise
        # multiplier instead of through the additive delta map -- and only
        # for the fault types that mechanically cause it.
        if profile.fault_type in config.BOGIE_VIBRATION_FAULT_TYPES:
            std_mult_target = rng.uniform(*config.BOGIE_DEGRADED_VIBRATION_STD_MULT_RANGE)
            vib_std_mult = 1 + trend * (std_mult_target - 1)
            vib_mean_creep = trend * rng.uniform(0.1, 0.3) * config.BOGIE_VIBRATION_BASE_RMS
            vibration = (
                config.BOGIE_VIBRATION_BASE_RMS
                + vib_mean_creep
                + rng.normal(0, config.BOGIE_VIBRATION_NOISE_STD, size=n) * vib_std_mult
            )

    temperature = temperature + _delta(profile, "temperature_c", timestamps)
    traction_temp = traction_temp + _delta(profile, "traction_motor_temp_c", timestamps)
    traction_rpm = traction_rpm + _delta(profile, "traction_motor_rpm", timestamps)
    brake_capacity = brake_capacity + _delta(profile, "brake_capacity_pct", timestamps)

    return pd.DataFrame({
        "timestamp": timestamps,
        "bogie_id": unit_id,
        "temperature_c": np.round(temperature, 3),
        "vibration_rms": np.round(np.clip(vibration, 0, None), 4),
        "traction_motor_temp_c": np.round(traction_temp, 2),
        "traction_motor_rpm": np.round(np.clip(traction_rpm, 0, None), 1),
        "brake_capacity_pct": np.round(np.clip(brake_capacity, 0, 100), 2),
        "axle_load_kg": np.round(axle_load, 1),
        "train_id": train_id,
    })


# ---------------------------------------------------------------------------
# Car auxiliary data (HVAC / saloon lighting / battery / train radio)
# ---------------------------------------------------------------------------

def _generate_car_unit(unit_id, train_id, start_dt, end_dt, rng, profile=None):
    """
    One row per polling interval for a car body's auxiliary systems.

    These are the systems a passenger notices before any bogie ever gets
    hot: air conditioning, lighting, the battery that holds the car up when
    traction power drops, and the radio link to the control centre. They
    share a single unit because they share a car body and, in practice, a
    single auxiliary converter -- but each fault type below moves a
    different subset of the six channels (see DEGRADATION_SIGNAL_DELTAS),
    so the model has to learn the combination, not a threshold.
    """
    timestamps = _generate_timestamps(rng, start_dt, end_dt, config.READING_INTERVAL_MAX_MINUTES)
    n = len(timestamps)
    hour_frac = timestamps.hour.values + timestamps.minute.values / 60.0
    hours = timestamps.hour.values
    in_service = (hours >= config.DOOR_PEAK_HOURS[0]) & (hours < config.DOOR_PEAK_HOURS[1])

    # Afternoon ambient peak: the pack works hardest mid-afternoon, so
    # supply air creeps up and current draw rises on a healthy car too.
    ambient = np.sin((hour_frac - 9) / 24 * 2 * np.pi)

    hvac_supply_temp = (
        config.CAR_HVAC_SUPPLY_TEMP_BASE_C
        + config.CAR_HVAC_DIURNAL_AMPLITUDE_C * ambient
        + rng.normal(0, config.CAR_HVAC_SUPPLY_TEMP_NOISE_STD, size=n)
    )
    hvac_current = (
        config.CAR_HVAC_CURRENT_BASE_AMPS
        + config.CAR_HVAC_CURRENT_DIURNAL_AMPLITUDE_A * ambient
        - 4.0 * (~in_service)  # packs idle down overnight
        + rng.normal(0, config.CAR_HVAC_CURRENT_NOISE_STD, size=n)
    )

    lighting_load = (
        config.CAR_LIGHTING_LOAD_BASE_PCT
        + rng.normal(0, config.CAR_LIGHTING_LOAD_NOISE_STD, size=n)
    )

    # Battery charges overnight in the depot and is drawn on in service.
    battery_soc = (
        config.CAR_BATTERY_SOC_BASE_PCT
        - config.CAR_BATTERY_SOC_DIURNAL_AMPLITUDE_PCT * in_service
        + rng.normal(0, config.CAR_BATTERY_SOC_NOISE_STD, size=n)
    )
    battery_voltage = (
        config.CAR_BATTERY_VOLTAGE_BASE_V
        - 1.5 * in_service
        + rng.normal(0, config.CAR_BATTERY_VOLTAGE_NOISE_STD, size=n)
    )

    comms_rssi = (
        config.CAR_COMMS_RSSI_BASE_DBM
        + rng.normal(0, config.CAR_COMMS_RSSI_NOISE_STD, size=n)
    )

    hvac_supply_temp = hvac_supply_temp + _delta(profile, "hvac_supply_temp_c", timestamps)
    hvac_current = hvac_current + _delta(profile, "hvac_current_amps", timestamps)
    lighting_load = lighting_load + _delta(profile, "lighting_load_pct", timestamps)
    battery_soc = battery_soc + _delta(profile, "battery_soc_pct", timestamps)
    battery_voltage = battery_voltage + _delta(profile, "battery_voltage_v", timestamps)
    comms_rssi = comms_rssi + _delta(profile, "comms_rssi_dbm", timestamps)

    return pd.DataFrame({
        "timestamp": timestamps,
        "car_id": unit_id,
        "hvac_supply_temp_c": np.round(hvac_supply_temp, 2),
        "hvac_current_amps": np.round(np.clip(hvac_current, 0, None), 2),
        "lighting_load_pct": np.round(np.clip(lighting_load, 0, 100), 2),
        "battery_soc_pct": np.round(np.clip(battery_soc, 0, 100), 2),
        "battery_voltage_v": np.round(np.clip(battery_voltage, 0, None), 2),
        "comms_rssi_dbm": np.round(comms_rssi, 2),
        "hvac_setpoint_c": config.CAR_HVAC_SETPOINT_C,
        "train_id": train_id,
    })


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    rng = config.RNG
    start_dt = pd.Timestamp(config.START_DATE)
    end_dt = start_dt + pd.Timedelta(days=config.DURATION_DAYS)

    train_ids = [f"TRAIN_{t:02d}" for t in range(1, config.N_TRAINS + 1)]
    door_ids, door_train_map = [], {}
    bogie_ids, bogie_train_map = [], {}
    car_ids, car_train_map = [], {}
    for t_idx, train_id in enumerate(train_ids, start=1):
        for n in range(1, config.DOORS_PER_TRAIN + 1):
            uid = f"DOOR_{t_idx:02d}_{n}"
            door_ids.append(uid)
            door_train_map[uid] = train_id
        for n in range(1, config.BOGIES_PER_TRAIN + 1):
            uid = f"BOGIE_{t_idx:02d}_{n}"
            bogie_ids.append(uid)
            bogie_train_map[uid] = train_id
        for n in range(1, config.CARS_PER_TRAIN + 1):
            uid = f"CAR_{t_idx:02d}_{n}"
            car_ids.append(uid)
            car_train_map[uid] = train_id

    degrading_doors = set(rng.choice(door_ids, config.N_DEGRADING_DOORS, replace=False))
    degrading_bogies = set(rng.choice(bogie_ids, config.N_DEGRADING_BOGIES, replace=False))
    degrading_cars = set(rng.choice(car_ids, config.N_DEGRADING_CARS, replace=False))

    def make_profile(unit_id, subsystem_type):
        start_day = rng.uniform(config.DEGRADATION_EARLIEST_START_DAY, config.DEGRADATION_LATEST_START_DAY)
        duration_days = rng.uniform(config.DEGRADATION_DURATION_MIN_DAYS, config.DEGRADATION_DURATION_MAX_DAYS)
        deg_start = start_dt + pd.Timedelta(days=start_day)
        deg_end = min(deg_start + pd.Timedelta(days=duration_days), end_dt - pd.Timedelta(hours=6))
        profile = DegradationProfile(unit_id, subsystem_type, deg_start, deg_end, rng)
        return profile.build()

    door_frames, fault_rows = [], []
    for uid in door_ids:
        profile = make_profile(uid, "door") if uid in degrading_doors else None
        door_frames.append(_generate_door_unit(uid, door_train_map[uid], start_dt, end_dt, rng, profile))
        if profile is not None:
            fault_rows.append({
                "timestamp": profile.fault_timestamp,
                "subsystem_id": uid,
                "subsystem_type": "door",
                "fault_type": profile.fault_type,
                "severity": profile.severity,
            })

    bogie_frames = []
    for uid in bogie_ids:
        profile = make_profile(uid, "bogie") if uid in degrading_bogies else None
        bogie_frames.append(_generate_bogie_unit(uid, bogie_train_map[uid], start_dt, end_dt, rng, profile))
        if profile is not None:
            fault_rows.append({
                "timestamp": profile.fault_timestamp,
                "subsystem_id": uid,
                "subsystem_type": "bogie",
                "fault_type": profile.fault_type,
                "severity": profile.severity,
            })

    car_frames = []
    for uid in car_ids:
        profile = make_profile(uid, "car") if uid in degrading_cars else None
        car_frames.append(_generate_car_unit(uid, car_train_map[uid], start_dt, end_dt, rng, profile))
        if profile is not None:
            fault_rows.append({
                "timestamp": profile.fault_timestamp,
                "subsystem_id": uid,
                "subsystem_type": "car",
                "fault_type": profile.fault_type,
                "severity": profile.severity,
            })

    door_df = pd.concat(door_frames, ignore_index=True).sort_values(["door_id", "timestamp"]).reset_index(drop=True)
    bogie_df = pd.concat(bogie_frames, ignore_index=True).sort_values(["bogie_id", "timestamp"]).reset_index(drop=True)
    car_df = pd.concat(car_frames, ignore_index=True).sort_values(["car_id", "timestamp"]).reset_index(drop=True)
    fault_df = pd.DataFrame(fault_rows).sort_values("timestamp").reset_index(drop=True)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    door_df.to_csv(config.DOOR_DATA_PATH, index=False)
    bogie_df.to_csv(config.BOGIE_DATA_PATH, index=False)
    car_df.to_csv(config.CAR_DATA_PATH, index=False)
    fault_df.to_csv(config.FAULT_LOG_PATH, index=False)

    total_rows = len(door_df) + len(bogie_df) + len(car_df)
    print(f"door_data.csv:  {len(door_df):,} rows across {len(door_ids)} units "
          f"({len(degrading_doors)} degrading: {sorted(degrading_doors)})")
    print(f"bogie_data.csv: {len(bogie_df):,} rows across {len(bogie_ids)} units "
          f"({len(degrading_bogies)} degrading: {sorted(degrading_bogies)})")
    print(f"car_data.csv:   {len(car_df):,} rows across {len(car_ids)} units "
          f"({len(degrading_cars)} degrading: {sorted(degrading_cars)})")
    print(f"fault_log.csv:  {len(fault_df)} fault events")
    print(f"total sensor rows: {total_rows:,}")
    # Sanity band, widened from the original two-subsystem 50k-200k when the
    # car auxiliary subsystem was added -- three subsystems at this cadence
    # land around 215k rows, which is still comfortably demo-sized.
    if not (50_000 <= total_rows <= 300_000):
        print(f"WARNING: total rows {total_rows:,} outside target 50k-300k range")


if __name__ == "__main__":
    main()

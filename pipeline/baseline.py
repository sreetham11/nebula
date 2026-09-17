"""
Baseline anomaly model: per-unit, per-signal rolling mean/std with a
configurable z-score threshold. This is the fast, explainable "tripwire"
half of the fusion logic in fusion.py -- cheap to compute, easy to justify
in a demo ("this reading is N std devs from this unit's own recent
normal"), but blind to slow multi-signal drift that stays within any single
reading's rolling window statistics until quite late.
"""

import numpy as np
import pandas as pd

from schema_config import SUBSYSTEMS
from model_config import BASELINE_SHORT_WINDOW, BASELINE_MIN_PERIODS, BASELINE_Z_THRESHOLD


def compute_baseline_flags(df, subsystem_type, window=BASELINE_SHORT_WINDOW,
                            min_periods=BASELINE_MIN_PERIODS, z_threshold=BASELINE_Z_THRESHOLD):
    """
    Returns df with added columns per signal:
      <signal>_roll_mean, <signal>_roll_std, <signal>_zscore
    plus a combined `baseline_flag` (bool) and `baseline_max_abs_z` (float)
    taken as the max |z| across that unit's signals at each reading.

    Rolling stats are computed per unit (groupby) over the unit's own
    reading sequence (index order), not wall-clock time, since sampling
    interval is jittered per unit.
    """
    cfg = SUBSYSTEMS[subsystem_type]
    id_col = cfg["id_col"]
    signal_cols = cfg["signal_cols"]

    df = df.copy()
    z_cols = []
    for sig in signal_cols:
        grouped = df.groupby(id_col)[sig]
        roll_mean = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).mean()
        )
        roll_std = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).std()
        )
        z = (df[sig] - roll_mean) / roll_std.replace(0, np.nan)

        df[f"{sig}_roll_mean"] = roll_mean
        df[f"{sig}_roll_std"] = roll_std
        df[f"{sig}_zscore"] = z
        z_cols.append(f"{sig}_zscore")

    abs_z = df[z_cols].abs()
    df["baseline_max_abs_z"] = abs_z.max(axis=1)
    df["baseline_flag"] = (df["baseline_max_abs_z"] > z_threshold).fillna(False)
    return df

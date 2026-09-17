"""
Loads raw sensor CSVs + fault log, and determines which units are
"healthy-only" (never appear in fault_log) vs "faulty" (had >=1 injected
fault). All column access goes through schema_config so this module works
unchanged once schema_config is repointed at real data.
"""

import pandas as pd

from schema_config import SUBSYSTEMS, FAULT_LOG_PATH, FAULT_LOG_COLUMNS


def load_subsystem_data(subsystem_type):
    cfg = SUBSYSTEMS[subsystem_type]
    df = pd.read_csv(cfg["data_path"], parse_dates=[cfg["timestamp_col"]])
    df = df.sort_values([cfg["id_col"], cfg["timestamp_col"]]).reset_index(drop=True)
    return df


def load_fault_log():
    df = pd.read_csv(FAULT_LOG_PATH, parse_dates=[FAULT_LOG_COLUMNS["timestamp"]])
    return df.sort_values(FAULT_LOG_COLUMNS["timestamp"]).reset_index(drop=True)


def faulty_unit_ids(fault_log, subsystem_type):
    col = FAULT_LOG_COLUMNS["subsystem_type"]
    id_col = FAULT_LOG_COLUMNS["subsystem_id"]
    return set(fault_log.loc[fault_log[col] == subsystem_type, id_col].unique())


def healthy_unit_ids(df, subsystem_type, fault_log):
    cfg = SUBSYSTEMS[subsystem_type]
    all_units = set(df[cfg["id_col"]].unique())
    return sorted(all_units - faulty_unit_ids(fault_log, subsystem_type))


def all_unit_ids(df, subsystem_type):
    cfg = SUBSYSTEMS[subsystem_type]
    return sorted(df[cfg["id_col"]].unique())


def split_healthy_units(healthy_ids, val_fraction, rng):
    """
    Splits healthy units (not windows) into disjoint train/val sets, so no
    window from the same unit can appear in both -- windows from one unit
    overlap heavily in time (same stride-sliced sequence) and share the
    same sensor-noise characteristics, so a row-level split leaks that
    unit's specific noise profile into validation and understates true
    generalization error.

    At least 1 unit is held out for validation and at least 1 remains for
    training, as long as len(healthy_ids) >= 2.
    """
    ids = list(healthy_ids)
    if len(ids) < 2:
        raise ValueError("need at least 2 healthy units to split by unit into train/val")

    n_val = max(1, round(len(ids) * val_fraction))
    n_val = min(n_val, len(ids) - 1)

    shuffled = list(rng.permutation(ids))
    val_units = sorted(shuffled[:n_val])
    train_units = sorted(shuffled[n_val:])
    return train_units, val_units
